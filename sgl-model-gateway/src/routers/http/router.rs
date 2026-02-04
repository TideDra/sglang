use std::{sync::Arc, time::Instant};

use axum::{
    body::{to_bytes, Body},
    extract::Request,
    http::{
        header::{CONTENT_LENGTH, CONTENT_TYPE},
        HeaderMap, HeaderValue, Method, StatusCode,
    },
    response::{IntoResponse, Response},
    Json,
};
use futures_util::StreamExt;
use reqwest::Client;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{debug, error};
use dashmap::{DashMap, DashSet};
use serde_json::{Value, json};
use uuid::Uuid;
use crate::{
    app_context::AppContext,
    config::types::RetryConfig,
    core::{
        ConnectionMode, RetryExecutor, UNKNOWN_MODEL_ID, Worker, WorkerLoadGuard, WorkerRegistry, WorkerType, is_retryable_status
    },
    observability::{
        events::{self, Event},
        metrics::{Metrics, bool_to_static_str, metrics_labels},
        otel_trace::inject_trace_context_http,
    },
    policies::{PolicyRegistry, SelectWorkerInfo},
    protocols::{
        chat::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageContent},
        classify::ClassifyRequest,
        common::GenerationRequest,
        completion::CompletionRequest,
        embedding::EmbeddingRequest,
        generate::GenerateRequest,
        rerank::{RerankRequest, RerankResponse, RerankResult},
        responses::{ResponsesGetParams, ResponsesRequest},
    },
    routers::{
        RouterTrait, error::{self, extract_error_code_from_response}, grpc::{regular::responses::conversions::{chat_to_responses, responses_to_chat}, utils::{error_type_from_status, route_to_endpoint}}, header_utils
    },
};

/// RAII guard to ensure traj_id is removed from processing set when dropped
struct TrajLockGuard<'a> {
    traj_id: String,
    processing_set: &'a DashSet<String>,
}

impl<'a> Drop for TrajLockGuard<'a> {
    fn drop(&mut self) {
        self.processing_set.remove(&self.traj_id);
        debug!("Released lock for traj_id={}", self.traj_id);
    }
}

/// Regular router that uses injected load balancing policies
pub struct Router {
    worker_registry: Arc<WorkerRegistry>,
    policy_registry: Arc<PolicyRegistry>,
    client: Client,
    dp_aware: bool,
    enable_igw: bool,
    retry_config: RetryConfig,
    traj_map: DashMap<String, Value>,
    processing_traj_ids: DashSet<String>,
}

impl std::fmt::Debug for Router {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Router")
            .field("worker_registry", &self.worker_registry)
            .field("policy_registry", &self.policy_registry)
            .field("client", &self.client)
            .field("dp_aware", &self.dp_aware)
            .field("enable_igw", &self.enable_igw)
            .field("retry_config", &self.retry_config)
            .finish()
    }
}

impl Router {
    /// Create a new router with injected policy and client
    pub async fn new(ctx: &Arc<AppContext>) -> Result<Self, String> {
        Ok(Router {
            worker_registry: ctx.worker_registry.clone(),
            policy_registry: ctx.policy_registry.clone(),
            client: ctx.client.clone(),
            dp_aware: ctx.router_config.dp_aware,
            enable_igw: ctx.router_config.enable_igw,
            retry_config: ctx.router_config.effective_retry_config(),
            traj_map: DashMap::new(),
            processing_traj_ids: DashSet::new(),
        })
    }

    fn select_first_worker(&self) -> Result<String, String> {
        let workers = self.worker_registry.get_all();
        let healthy_workers: Vec<_> = workers.iter().filter(|w| w.is_healthy()).collect();
        if healthy_workers.is_empty() {
            Err("No workers are available".to_string())
        } else {
            Ok(healthy_workers[0].url().to_string())
        }
    }

    // Helper method to proxy GET requests to the first available worker
    async fn proxy_get_request(&self, req: Request<Body>, endpoint: &str) -> Response {
        let headers = header_utils::copy_request_headers(&req);

        match self.select_first_worker() {
            Ok(worker_url) => {
                let mut request_builder = self.client.get(format!("{}/{}", worker_url, endpoint));
                for (name, value) in headers {
                    // Use eq_ignore_ascii_case to avoid string allocation
                    if !name.eq_ignore_ascii_case("content-type")
                        && !name.eq_ignore_ascii_case("content-length")
                    {
                        request_builder = request_builder.header(name, value);
                    }
                }

                match request_builder.send().await {
                    Ok(res) => {
                        let status = StatusCode::from_u16(res.status().as_u16())
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

                        // Preserve headers from backend
                        let response_headers =
                            header_utils::preserve_response_headers(res.headers());

                        match res.bytes().await {
                            Ok(body) => {
                                let mut response = Response::new(Body::from(body));
                                *response.status_mut() = status;
                                *response.headers_mut() = response_headers;
                                response
                            }
                            Err(e) => error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ),
                        }
                    }
                    Err(e) => convert_reqwest_error(e),
                }
            }
            Err(e) => error::service_unavailable("no_workers", e),
        }
    }

    /// Select worker for a specific model considering circuit breaker state
    fn select_worker_for_model(
        &self,
        model_id: Option<&str>,
        text: Option<&str>,
        headers: Option<&HeaderMap>,
    ) -> Option<Arc<dyn Worker>> {
        let effective_model_id = if !self.enable_igw { None } else { model_id };

        // Get workers for the specified model O(1), filtered by connection mode
        let workers = self.worker_registry.get_workers_filtered(
            effective_model_id,
            Some(WorkerType::Regular),
            Some(ConnectionMode::Http),
            None,  // any runtime type
            false, // get all workers, we'll filter by is_available() next
        );

        let available: Vec<Arc<dyn Worker>> = workers
            .iter()
            .filter(|w| w.is_available())
            .cloned()
            .collect();
        if available.is_empty() {
            return None;
        }

        // Get the appropriate policy for this model
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        // Get cached hash ring for consistent hashing (O(log n) lookup)
        let hash_ring = self
            .worker_registry
            .get_hash_ring(effective_model_id.unwrap_or(UNKNOWN_MODEL_ID));

        let idx = policy.select_worker(
            &available,
            &SelectWorkerInfo {
                request_text: text,
                tokens: None, // HTTP doesn't have tokens, use gRPC for PrefixHash
                headers,
                hash_ring,
            },
        )?;

        // Record worker selection metric (Layer 3)
        Metrics::record_worker_selection(
            metrics_labels::WORKER_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model_id.unwrap_or("default"),
            policy.name(),
        );

        Some(available[idx].clone())
    }

    pub async fn route_typed_request<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
    ) -> Response {
        let start = Instant::now();
        let is_stream = typed_req.is_stream();
        let text = typed_req.extract_text_for_routing();
        let model = model_id.unwrap_or("default");
        let endpoint = route_to_endpoint(route);

        // Record request start (Layer 2)
        Metrics::record_router_request(
            metrics_labels::ROUTER_HTTP,
            metrics_labels::BACKEND_REGULAR,
            metrics_labels::CONNECTION_HTTP,
            model,
            endpoint,
            bool_to_static_str(is_stream),
        );

        let response = RetryExecutor::execute_response_with_retry(
            &self.retry_config,
            // operation per attempt
            |_: u32| async {
                let res = self
                    .route_typed_request_once(headers, typed_req, route, model_id, is_stream, &text)
                    .await;

                // Need to be outside `route_typed_request_once` because that function has multiple return paths
                Metrics::record_router_upstream_response(
                    metrics_labels::ROUTER_HTTP,
                    res.status().as_u16(),
                    extract_error_code_from_response(&res),
                );

                res
            },
            // should_retry predicate
            |res, _attempt| is_retryable_status(res.status()),
            // on_backoff hook
            |delay, attempt| {
                // Layer 3 worker metrics
                Metrics::record_worker_retry(metrics_labels::WORKER_REGULAR, endpoint);
                Metrics::record_worker_retry_backoff(attempt, delay);
            },
            // on_exhausted hook
            || {
                Metrics::record_worker_retries_exhausted(metrics_labels::WORKER_REGULAR, endpoint);
            },
        )
        .await;

        if response.status().is_success() {
            let duration = start.elapsed();
            Metrics::record_router_duration(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                duration,
            );
        } else if !is_retryable_status(response.status()) {
            Metrics::record_router_error(
                metrics_labels::ROUTER_HTTP,
                metrics_labels::BACKEND_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                model,
                endpoint,
                error_type_from_status(response.status()),
            );
        }

        response
    }

    async fn route_typed_request_once<T: GenerationRequest + serde::Serialize + Clone>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        model_id: Option<&str>,
        is_stream: bool,
        text: &str,
    ) -> Response {
        let worker = match self.select_worker_for_model(model_id, Some(text), headers) {
            Some(w) => w,
            None => {
                return error::service_unavailable(
                    "no_available_workers",
                    "No available workers (all circuits open or unhealthy)",
                );
            }
        };

        // Optional load tracking for cache-aware policy
        // Get the policy for this model to check if it's cache-aware
        let policy = match model_id {
            Some(model) => self.policy_registry.get_policy_or_default(model),
            None => self.policy_registry.get_default_policy(),
        };

        let load_guard =
            (policy.name() == "cache_aware").then(|| WorkerLoadGuard::new(worker.clone()));

        // Note: Using borrowed reference avoids heap allocation
        events::RequestSentEvent { url: worker.url() }.emit();
        let mut headers_with_trace = headers.cloned().unwrap_or_default();
        inject_trace_context_http(&mut headers_with_trace);
        let headers = Some(&headers_with_trace);

        let response = self
            .send_typed_request(
                headers,
                typed_req,
                route,
                worker.url(),
                is_stream,
                load_guard,
            )
            .await;

        events::RequestReceivedEvent {}.emit();

        let status = response.status();
        worker.record_outcome(status.is_success());

        // Record worker errors for server errors (5xx)
        if status.is_server_error() {
            Metrics::record_worker_error(
                metrics_labels::WORKER_REGULAR,
                metrics_labels::CONNECTION_HTTP,
                error_type_from_status(status),
            );
        }

        response
    }

    // Helper: return base worker URL (strips DP suffix when enabled)
    fn worker_base_url(&self, worker_url: &str) -> String {
        if self.dp_aware {
            if let Ok((prefix, _)) = Self::extract_dp_rank(worker_url) {
                return prefix.to_string();
            }
        }
        worker_url.to_string()
    }

    // Generic simple routing for GET/POST without JSON body
    async fn route_simple_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
        method: Method,
    ) -> Response {
        // TODO: currently the sglang worker is using in-memory state management, so this implementation has to fan out to all workers.
        // Eventually, we need to have router to manage the chat history with a proper database, will update this implementation accordingly.
        let workers = self.worker_registry.get_all();
        if workers.is_empty() {
            return error::service_unavailable("no_workers", "No available workers");
        }

        // Pre-filter headers once before the loop to avoid repeated lowercasing
        let filtered_headers: Vec<_> = headers
            .map(|hdrs| {
                hdrs.iter()
                    .filter(|(name, _)| {
                        !name.as_str().eq_ignore_ascii_case("content-type")
                            && !name.as_str().eq_ignore_ascii_case("content-length")
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut last_response: Option<Response> = None;
        for worker in workers {
            let worker_url = worker.url();
            let base = self.worker_base_url(worker_url);

            let url = format!("{}/{}", base, endpoint);
            let mut request_builder = match method {
                Method::GET => self.client.get(url),
                Method::POST => self.client.post(url),
                _ => {
                    return error::method_not_allowed(
                        "unsupported_method",
                        "Unsupported method for simple routing",
                    )
                }
            };

            if let Some(api_key) = worker.api_key() {
                // Pre-allocate string with capacity to avoid reallocation
                let mut auth_header = String::with_capacity(7 + api_key.len());
                auth_header.push_str("Bearer ");
                auth_header.push_str(api_key);
                request_builder = request_builder.header("Authorization", auth_header);
            }

            // Apply pre-filtered headers
            for (name, value) in &filtered_headers {
                request_builder = request_builder.header(*name, *value);
            }

            match request_builder.send().await {
                Ok(res) => {
                    let status = StatusCode::from_u16(res.status().as_u16())
                        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
                    let response_headers = header_utils::preserve_response_headers(res.headers());
                    match res.bytes().await {
                        Ok(body) => {
                            let mut response = Response::new(Body::from(body));
                            *response.status_mut() = status;
                            *response.headers_mut() = response_headers;
                            if status.is_success() {
                                return response;
                            }
                            last_response = Some(response);
                        }
                        Err(e) => {
                            last_response = Some(error::internal_error(
                                "read_response_failed",
                                format!("Failed to read response: {}", e),
                            ));
                        }
                    }
                }
                Err(e) => {
                    last_response = Some(convert_reqwest_error(e));
                }
            }
        }

        last_response
            .unwrap_or_else(|| error::bad_gateway("no_worker_response", "No worker response"))
    }

    // Route a GET request with provided headers to a specific endpoint
    async fn route_get_request(&self, headers: Option<&HeaderMap>, endpoint: &str) -> Response {
        self.route_simple_request(headers, endpoint, Method::GET)
            .await
    }

    // Route a POST request with empty body to a specific endpoint
    async fn route_post_empty_request(
        &self,
        headers: Option<&HeaderMap>,
        endpoint: &str,
    ) -> Response {
        self.route_simple_request(headers, endpoint, Method::POST)
            .await
    }

    // TODO (rui): Better accommodate to the Worker abstraction
    fn extract_dp_rank(worker_url: &str) -> Result<(&str, usize), String> {
        let parts: Vec<&str> = worker_url.split('@').collect();
        if parts.len() != 2 {
            return Err(format!("invalid worker_url format: {}", worker_url));
        }

        // Parse the second part (dp_rank) into an integer
        match parts[1].parse::<usize>() {
            Ok(dp_rank) => Ok((parts[0], dp_rank)),
            Err(_) => Err(format!(
                "failed to parse dp_rank from worker_url: {}",
                worker_url
            )),
        }
    }

    // Send typed request directly without conversion
    async fn send_typed_request<T: serde::Serialize>(
        &self,
        headers: Option<&HeaderMap>,
        typed_req: &T,
        route: &'static str,
        worker_url: &str,
        is_stream: bool,
        load_guard: Option<WorkerLoadGuard>,
    ) -> Response {
        // Get the worker once and reuse for API key and load tracking
        let worker = self.worker_registry.get_by_url(worker_url);
        let api_key = worker.as_ref().and_then(|w| w.api_key().clone());

        // Static key string to avoid per-request allocations
        const DP_RANK_KEY: &str = "data_parallel_rank";

        let mut request_builder = if self.dp_aware {
            let (worker_url_prefix, dp_rank) = match Self::extract_dp_rank(worker_url) {
                Ok(tup) => tup,
                Err(e) => {
                    error!("Failed to extract dp_rank: {}", e);
                    return error::internal_error(
                        "dp_rank_extraction_failed",
                        format!("Failed to extract dp_rank: {}", e),
                    );
                }
            };

            let mut json_val = match serde_json::to_value(typed_req) {
                Ok(j) => j,
                Err(e) => {
                    return error::bad_request(
                        "serialization_failed",
                        format!("Convert into serde_json::Value failed: {}", e),
                    );
                }
            };

            if let Some(map) = json_val.as_object_mut() {
                // Use static key string to avoid allocation
                map.insert(DP_RANK_KEY.to_string(), serde_json::json!(dp_rank));
                // Only serialize if debug logging is enabled to avoid CPU overhead
                if tracing::enabled!(tracing::Level::DEBUG) {
                    debug!(
                        "Modified request body: {}",
                        serde_json::to_string(&json_val).unwrap_or_else(|_| String::from("ERR"))
                    );
                }
            } else {
                return error::bad_request(
                    "dp_rank_insertion_failed",
                    "Failed to insert the data_parallel_rank field into the request body",
                );
            }

            self.client
                .post(format!("{}{}", worker_url_prefix, route))
                .json(&json_val)
        } else {
            self.client
                .post(format!("{}{}", worker_url, route))
                .json(typed_req) // Use json() directly with typed request
        };

        if let Some(key) = api_key {
            // Pre-allocate string with capacity to avoid reallocation
            let mut auth_header = String::with_capacity(7 + key.len());
            auth_header.push_str("Bearer ");
            auth_header.push_str(&key);
            request_builder = request_builder.header("Authorization", auth_header);
        }

        // Copy all headers from original request if provided
        if let Some(headers) = headers {
            for (name, value) in headers {
                // Skip Content-Type and Content-Length as .json() sets them
                if *name != CONTENT_TYPE && *name != CONTENT_LENGTH {
                    request_builder = request_builder.header(name, value);
                }
            }
        }

        let res = match request_builder.send().await {
            Ok(res) => res,
            Err(e) => {
                error!(
                    "Failed to send typed request worker_url={} route={} error={}",
                    worker_url, route, e
                );

                return convert_reqwest_error(e);
            }
        };

        let status = StatusCode::from_u16(res.status().as_u16())
            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);

        if !is_stream {
            // For non-streaming requests, preserve headers
            let response_headers = header_utils::preserve_response_headers(res.headers());

            let response = match res.bytes().await {
                Ok(body) => {
                    let mut response = Response::new(Body::from(body));
                    *response.status_mut() = status;
                    *response.headers_mut() = response_headers;
                    response
                }
                Err(e) => {
                    let error_msg = format!("Failed to get response body: {}", e);
                    error::internal_error("read_response_body_failed", error_msg)
                }
            };

            // load_guard dropped here automatically after response body is read
            response
        } else {
            // Preserve headers for streaming response
            let mut response_headers = header_utils::preserve_response_headers(res.headers());
            // Ensure we set the correct content-type for SSE
            response_headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));

            let stream = res.bytes_stream();
            let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

            // Spawn task to forward stream
            tokio::spawn(async move {
                let mut stream = stream;
                while let Some(chunk) = stream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            if tx.send(Ok(bytes)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(format!("Stream error: {}", e)));
                            break;
                        }
                    }
                }
            });

            let stream = UnboundedReceiverStream::new(rx);
            let body = Body::from_stream(stream);

            let mut response = Response::new(body);
            *response.status_mut() = status;
            *response.headers_mut() = response_headers;

            // Attach load guard to response body for proper RAII lifecycle
            // Guard is dropped when response body is consumed or client disconnects
            if let Some(guard) = load_guard {
                response = guard.attach_to_response(response);
            }
            response
        }
    }

    async fn build_rerank_response(
        req: &RerankRequest,
        response: Response,
    ) -> anyhow::Result<Response> {
        let (_, response_body) = response.into_parts();
        let body_bytes = to_bytes(response_body, usize::MAX).await?;
        let rerank_results = serde_json::from_slice::<Vec<RerankResult>>(&body_bytes)?;
        let mut rerank_response =
            RerankResponse::new(rerank_results, req.model.clone(), req.rid.clone());
        // Sorting is handled by Python worker (serving_rerank.py)
        if let Some(top_k) = req.top_k {
            rerank_response.apply_top_k(top_k);
        }
        if !req.return_documents {
            rerank_response.drop_documents();
        }
        Ok(Json(rerank_response).into_response())
    }
}

fn convert_reqwest_error(e: reqwest::Error) -> Response {
    let url = e
        .url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let message = format!("{}. URL: {}", e, url);

    // TODO improve error status code
    let (status, code) = if let Some(upstream_status) = e.status() {
        (upstream_status, "call_upstream_status_error")
    } else if e.is_builder() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_builder_error",
        )
    } else if e.is_request() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_error",
        )
    } else if e.is_redirect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_redirect_error",
        )
    } else if e.is_body() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_body_error",
        )
    } else if e.is_decode() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_decode_error",
        )
    } else if e.is_timeout() {
        (StatusCode::GATEWAY_TIMEOUT, "call_upstream_timeout")
    } else if e.is_connect() {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_connection_failed",
        )
    } else {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "call_upstream_request_failed",
        )
    };

    error::create_error(status, code, message)
}

use async_trait::async_trait;

#[async_trait]
impl RouterTrait for Router {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    async fn health_generate(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "health_generate").await
    }

    async fn get_server_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_server_info").await
    }

    async fn get_models(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "v1/models").await
    }

    async fn get_model_info(&self, req: Request<Body>) -> Response {
        self.proxy_get_request(req, "get_model_info").await
    }

    async fn route_generate(
        &self,
        headers: Option<&HeaderMap>,
        body: &GenerateRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/generate", model_id)
            .await
    }

    async fn route_chat(
        &self,
        headers: Option<&HeaderMap>,
        body: &ChatCompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        if let Some(traj_id) = &body.traj_id {
            // trajectory management only supported for non-streaming requests
            if body.stream {
                return error::bad_request(
                    "trajectory_not_supported_for_streaming",
                    "Trajectory management (traj_id) is not supported for streaming requests",
                );
            }

            // Try to acquire exclusive lock for this traj_id
            if !self.processing_traj_ids.insert(traj_id.clone()) {
                // traj_id is already being processed by another request
                return error::bad_request(
                    "trajectory_locked",
                    format!("Trajectory with id '{}' is currently being processed by another request", traj_id),
                );
            }

            // Create RAII guard to ensure lock is released when function returns
            let _lock_guard = TrajLockGuard {
                traj_id: traj_id.clone(),
                processing_set: &self.processing_traj_ids,
            };

            if !self.traj_map.contains_key(traj_id) {
                // Initialize a new trajectory.
                self.traj_map.insert(traj_id.clone(), json!({"cached_token_ids": null, "output_token_mask": null, "cached_request": null, "cached_tools_text": null, "eos_token_id": null, "cached_token_logprobs": null}));
            }
            // Clone the trajectory value and immediately drop the lock
            let trajectory_value = self.traj_map.get(traj_id)
                .expect("Unexpected error: traj not found")
                .value()
                .clone();
            // Now the lock is released
            drop(_lock_guard);
            let mut new_body = body.clone();
            new_body.trajectory = Some(trajectory_value);

            let result = self.route_typed_request(headers, &new_body, "/v1/chat/completions", model_id).await;

            // Extract and process response body
            if result.status().is_success() {
                let status = result.status();
                let headers_clone = result.headers().clone();

                // Extract response body (limit to 100MB for safety)
                const MAX_BODY_SIZE: usize = 100 * 1024 * 1024; // 100MB
                let body_bytes = match to_bytes(result.into_body(), MAX_BODY_SIZE).await {
                    Ok(bytes) => {
                        bytes
                    }
                    Err(e) => {
                        error!("Failed to read response body: {}", e);
                        return error::internal_error("read_response_body_failed", e.to_string());
                    }
                };

                // Deserialize as JSON - must succeed
                let mut json_data = match serde_json::from_slice::<Value>(&body_bytes) {
                    Ok(data) => {
                        data
                    }
                    Err(e) => {
                        error!("Failed to parse response as JSON: {}", e);
                        return error::internal_error("parse_response_json_failed", format!("Failed to parse response as JSON: {}", e));
                    }
                };

                // Extract metadata - must exist
                let metadata = match json_data.get_mut("metadata") {
                    Some(meta) => {
                        meta
                    }
                    None => {
                        error!("Response missing metadata field");
                        return error::internal_error("missing_metadata", "Response missing metadata field");
                    }
                };

                // Extract trajectory - must exist
                let trajectory = match metadata.get("trajectory") {
                    Some(traj) => {
                        traj
                    }
                    None => {
                        error!("Response metadata missing trajectory field");
                        return error::internal_error("missing_trajectory", "Response metadata missing trajectory field");
                    }
                };

                // Update traj_map with the new trajectory
                self.traj_map.insert(traj_id.clone(), trajectory.clone());

                // Remove trajectory from response metadata
                if let Some(obj) = metadata.as_object_mut() {
                    obj.remove("trajectory");
                }

                // Serialize back to bytes - must succeed
                let modified_body = match serde_json::to_vec(&json_data) {
                    Ok(body) => {
                        body
                    }
                    Err(e) => {
                        error!("Failed to serialize modified response: {}", e);
                        return error::internal_error("serialize_response_failed", format!("Failed to serialize modified response: {}", e));
                    }
                };

                let body_len = modified_body.len();
                let mut response = Response::new(Body::from(modified_body));
                *response.status_mut() = status;
                *response.headers_mut() = headers_clone;

                // Update Content-Length header to match the modified body size
                response.headers_mut().insert(
                    CONTENT_TYPE,
                    HeaderValue::from_static("application/json"),
                );
                response.headers_mut().insert(
                    CONTENT_LENGTH,
                    HeaderValue::from_str(&body_len.to_string()).unwrap(),
                );

                debug!("Returning response for traj_id={} with body_len={}", traj_id, body_len);
                response
            } else {
                // For error responses, just return as-is
                result
            }
        } else {
            self.route_typed_request(headers, body, "/v1/chat/completions", model_id)
                .await
        }
    }

    async fn get_trajectory(&self, _headers: Option<&HeaderMap>, traj_id: &str) -> Response {
        // Check if trajectory is currently being processed
        if self.processing_traj_ids.contains(traj_id) {
            return error::bad_request(
                "trajectory_locked",
                format!("Trajectory with id '{}' is currently being processed", traj_id),
            );
        }

        if self.traj_map.contains_key(traj_id) {
            let trajectory = self.traj_map.get(traj_id).unwrap().value().clone();
            return Json(json!({ "trajectory": trajectory["cached_request"], "token_ids": trajectory["cached_token_ids"], "output_token_mask": trajectory["output_token_mask"], "token_logprobs": trajectory["cached_token_logprobs"] })).into_response();
        } else {
            return error::not_found("trajectory_not_found", format!("Trajectory with id '{}' not found", traj_id));
        }
    }

    async fn delete_trajectory(&self, _headers: Option<&HeaderMap>, traj_id: &str) -> Response {
        // Check if trajectory is currently being processed
        if self.processing_traj_ids.contains(traj_id) {
            return error::bad_request(
                "trajectory_locked",
                format!("Trajectory with id '{}' is currently being processed, cannot delete", traj_id),
            );
        }

        if self.traj_map.contains_key(traj_id) {
            self.traj_map.remove(traj_id);
            return Json(serde_json::json!({ "message": "Trajectory deleted" })).into_response();
        } else {
            return error::not_found("trajectory_not_found", format!("Trajectory with id '{}' not found", traj_id));
        }
    }

    async fn route_completion(
        &self,
        headers: Option<&HeaderMap>,
        body: &CompletionRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/completions", model_id)
            .await
    }

    async fn route_responses(
        &self,
        headers: Option<&HeaderMap>,
        body: &ResponsesRequest,
        model_id: Option<&str>,
    ) -> Response {
        let traj_id = match &body.previous_response_id {
            Some(id) => {
                if !self.traj_map.contains_key(id) {
                    return error::not_found("trajectory_not_found", format!("Trajectory with id '{}' not found", id));
                }
                id
            },
            None => &format!("resp_{}", Uuid::new_v4()), // Root node. Will not be used by user.
        };

        if body.stream.unwrap_or(false) {
            return error::bad_request(
                "trajectory_not_supported_for_streaming",
                "Trajectory management (traj_id) is not supported for streaming requests",
            );
        }

        // Try to acquire exclusive lock for this traj_id
        if !self.processing_traj_ids.insert(traj_id.clone()) {
            // traj_id is already being processed by another request
            return error::bad_request(
                "trajectory_locked",
                format!("Trajectory with id '{}' is currently being processed by another request", traj_id),
            );
        }

        // Create RAII guard to ensure lock is released when function returns
        let _lock_guard = TrajLockGuard {
            traj_id: traj_id.clone(),
            processing_set: &self.processing_traj_ids,
        };

        let mut chat_body = match self.traj_map.get(traj_id) {
            Some(traj) => {
                let cached_request = traj.value().get("cached_request").unwrap();
                let chat_body = serde_json::from_value::<ChatCompletionRequest>(cached_request.clone()).unwrap();
                let raw_messages = chat_body.messages;
                let raw_instruction = match raw_messages.first().unwrap() {
                    ChatMessage::System { content, name:_ } => Some(content.to_simple_string().clone()),
                    _ => None,
                };
                let mut body_without_instructions = body.clone();
                body_without_instructions.instructions = None;
                let mut new_chat_body = responses_to_chat(&body_without_instructions).unwrap();
                let incoming_messages = new_chat_body.messages;
                let mut merged_messages = Vec::new();
                merged_messages.extend(raw_messages);
                merged_messages.extend(incoming_messages);
                new_chat_body.messages = merged_messages;
                new_chat_body.traj_id = Some(traj_id.clone());
                if raw_instruction != body.instructions {
                    let first_msg = new_chat_body.messages.first_mut().unwrap();
                    if raw_instruction.is_none() {
                        // We have new instructions
                        if let ChatMessage::System { content:_, name:_ } = first_msg {
                            return error::internal_error("instructions_mismatch", "Unexpected error: System message should not exist.");
                        }
                        new_chat_body.messages.insert(0, ChatMessage::System { content: MessageContent::Text(body.instructions.clone().unwrap()), name: None });
                    } else if body.instructions.is_none() {
                        // We have old instructions
                        match first_msg {
                            ChatMessage::System { content:_, name:_ } => {},
                            _ => {
                                return error::internal_error("instructions_mismatch", "Unexpected error: System message should exist.");
                            }
                        }
                        new_chat_body.messages.remove(0);
                    } else {
                        // We have different instructions
                        match first_msg {
                            ChatMessage::System { content, name:_ } => {
                                *content = MessageContent::Text(body.instructions.clone().unwrap());
                            }
                            _ => {
                                return error::internal_error("instructions_mismatch", "Unexpected error: System message should exist.");
                            }
                        }
                    }
                    // Create new traj tree
                    new_chat_body.traj_id = Some(format!("resp_{}", Uuid::new_v4()));
                }
                new_chat_body

            }
            None => {
                let mut new_chat_body = responses_to_chat(body).unwrap();
                new_chat_body.traj_id = Some(traj_id.clone());
                new_chat_body
            }
        };
        chat_body.logprobs = true;

        let traj_id = chat_body.traj_id.as_ref().unwrap();

        if !self.traj_map.contains_key(traj_id) {
            // Initialize a new trajectory.
            self.traj_map.insert(traj_id.clone(), json!({"cached_token_ids": null, "output_token_mask": null, "cached_request": null, "cached_tools_text": null, "eos_token_id": null, "cached_token_logprobs": null}));
        }
        // Clone the trajectory value and immediately drop the lock
        let trajectory_value = self.traj_map.get(traj_id)
            .expect("Unexpected error: traj not found")
            .value()
            .clone();
        // Now the lock is released

        chat_body.trajectory = Some(trajectory_value);
        drop(_lock_guard);

        let result = self.route_typed_request(headers, &chat_body, "/v1/chat/completions", model_id).await;

        // Extract and process response body
        if result.status().is_success() {
            let status = result.status();
            let headers_clone = result.headers().clone();

            // Extract response body (limit to 100MB for safety)
            const MAX_BODY_SIZE: usize = 100 * 1024 * 1024; // 100MB
            let body_bytes = match to_bytes(result.into_body(), MAX_BODY_SIZE).await {
                Ok(bytes) => {
                    bytes
                }
                Err(e) => {
                    error!("Failed to read response body: {}", e);
                    return error::internal_error("read_response_body_failed", e.to_string());
                }
            };

            // Deserialize as JSON - must succeed
            let chat_response = match serde_json::from_slice::<ChatCompletionResponse>(&body_bytes) {
                Ok(data) => {
                    data
                }
                Err(e) => {
                    error!("Failed to parse response as JSON: {}", e);
                    return error::internal_error("parse_response_json_failed", format!("Failed to parse response as JSON: {}", e));
                }
            };

            // Extract metadata - must exist
            let metadata = match chat_response.metadata.as_ref() {
                Some(meta) => {
                    meta
                }
                None => {
                    error!("Response missing metadata field");
                    return error::internal_error("missing_metadata", "Response missing metadata field");
                }
            };

            // Extract trajectory - must exist
            let trajectory = match metadata.get("trajectory") {
                Some(traj) => {
                    traj
                }
                None => {
                    error!("Response metadata missing trajectory field");
                    return error::internal_error("missing_trajectory", "Response metadata missing trajectory field");
                }
            };
            let new_response_id = format!("resp_{}", Uuid::new_v4());
            // Update traj_map with the new trajectory
            self.traj_map.insert(new_response_id.clone(), trajectory.clone());

            let responses_response = chat_to_responses(&chat_response, body, Some(new_response_id)).expect("Failed to convert chat response to responses response");

            // Serialize back to bytes - must succeed
            let modified_body = match serde_json::to_vec(&responses_response) {
                Ok(body) => {
                    body
                }
                Err(e) => {
                    error!("Failed to serialize modified response: {}", e);
                    return error::internal_error("serialize_response_failed", format!("Failed to serialize modified response: {}", e));
                }
            };

            let body_len = modified_body.len();
            let mut response = Response::new(Body::from(modified_body));
            *response.status_mut() = status;
            *response.headers_mut() = headers_clone;

            // Update Content-Length header to match the modified body size
            response.headers_mut().insert(
                CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            );
            response.headers_mut().insert(
                CONTENT_LENGTH,
                HeaderValue::from_str(&body_len.to_string()).unwrap(),
            );

            debug!("Returning response for traj_id={} with body_len={}", traj_id, body_len);
            response
        } else {
            // For error responses, just return as-is
            result
        }
    }

    async fn get_response(
        &self,
        headers: Option<&HeaderMap>,
        response_id: &str,
        _params: &ResponsesGetParams,
    ) -> Response {
        let endpoint = format!("v1/responses/{}", response_id);
        self.route_get_request(headers, &endpoint).await
    }

    async fn cancel_response(&self, headers: Option<&HeaderMap>, response_id: &str) -> Response {
        let endpoint = format!("v1/responses/{}/cancel", response_id);
        self.route_post_empty_request(headers, &endpoint).await
    }

    async fn route_embeddings(
        &self,
        headers: Option<&HeaderMap>,
        body: &EmbeddingRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/embeddings", model_id)
            .await
    }

    async fn route_classify(
        &self,
        headers: Option<&HeaderMap>,
        body: &ClassifyRequest,
        model_id: Option<&str>,
    ) -> Response {
        self.route_typed_request(headers, body, "/v1/classify", model_id)
            .await
    }

    async fn route_rerank(
        &self,
        headers: Option<&HeaderMap>,
        body: &RerankRequest,
        model_id: Option<&str>,
    ) -> Response {
        let response = self
            .route_typed_request(headers, body, "/v1/rerank", model_id)
            .await;
        if response.status().is_success() {
            match Self::build_rerank_response(body, response).await {
                Ok(rerank_response) => rerank_response,
                Err(e) => {
                    error!("Failed to build rerank response: {}", e);
                    return error::internal_error(
                        "rerank_response_build_failed",
                        "Failed to build rerank response",
                    );
                }
            }
        } else {
            response
        }
    }

    fn router_type(&self) -> &'static str {
        "regular"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::BasicWorkerBuilder;

    fn create_test_regular_router() -> Router {
        // Create registries
        let worker_registry = Arc::new(WorkerRegistry::new());
        let policy_registry = Arc::new(PolicyRegistry::new(
            crate::config::types::PolicyConfig::RoundRobin,
        ));

        // Register test workers
        let worker1 = BasicWorkerBuilder::new("http://worker1:8080")
            .worker_type(WorkerType::Regular)
            .build();
        let worker2 = BasicWorkerBuilder::new("http://worker2:8080")
            .worker_type(WorkerType::Regular)
            .build();
        worker_registry.register(Arc::new(worker1));
        worker_registry.register(Arc::new(worker2));

        Router {
            worker_registry,
            policy_registry,
            dp_aware: false,
            client: Client::new(),
            retry_config: RetryConfig::default(),
            enable_igw: false,
            traj_map: DashMap::new(),
            processing_traj_ids: DashSet::new(),
        }
    }

    fn create_test_unhealthy_router() -> Router {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        workers[0].set_healthy(false);
        router
    }

    #[test]
    fn test_router_get_worker_urls_regular() {
        let router = create_test_regular_router();
        let workers = router.worker_registry.get_all();
        let urls: Vec<String> = workers.iter().map(|w| w.url().to_string()).collect();

        assert_eq!(urls.len(), 2);
        assert!(urls.contains(&"http://worker1:8080".to_string()));
        assert!(urls.contains(&"http://worker2:8080".to_string()));
    }

    #[test]
    fn test_select_first_worker_regular() {
        let router = create_test_regular_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();
        // DashMap doesn't guarantee order, so just check we get one of the workers
        assert!(url == "http://worker1:8080" || url == "http://worker2:8080");
    }

    #[test]
    fn test_select_first_worker_with_unhealthy_worker() {
        let router = create_test_unhealthy_router();
        let result = router.select_first_worker();

        assert!(result.is_ok());
        let url = result.unwrap();

        let worker = router.worker_registry.get_by_url(&url).unwrap();
        assert!(worker.is_healthy());
    }
}
