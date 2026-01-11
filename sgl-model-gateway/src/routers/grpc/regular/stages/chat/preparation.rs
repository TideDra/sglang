//! Chat preparation stage: Filter tools, process messages, tokenize, build constraints

use std::borrow::Cow;

use async_trait::async_trait;
use axum::response::Response;
use tracing::error;

use crate::{
    protocols::chat::{ChatCompletionRequest, Trajectory, extract_text_from_messages},
    routers::{
        error,
        grpc::{
            common::stages::PipelineStage,
            context::{PreparationOutput, RequestContext},
            utils,
        },
    },
};

/// Chat preparation stage
///
/// Extracts chat-specific preparation logic from the old unified PreparationStage.
/// This is a direct extraction without architectural changes.
pub(crate) struct ChatPreparationStage;

#[async_trait]
impl PipelineStage for ChatPreparationStage {
    async fn execute(&self, ctx: &mut RequestContext) -> Result<Option<Response>, Response> {
        let request = ctx.chat_request_arc();
        self.prepare_chat(ctx, &request).await?;
        Ok(None)
    }

    fn name(&self) -> &'static str {
        "ChatPreparation"
    }
}

impl ChatPreparationStage {
    async fn prepare_chat(
        &self,
        ctx: &mut RequestContext,
        request: &ChatCompletionRequest,
    ) -> Result<(), Response> {
        // Step 0: Resolve tokenizer from registry (cached for reuse in response processing)
        let tokenizer =
            utils::resolve_tokenizer(ctx, "ChatPreparationStage::prepare_chat").map_err(|e| *e)?;

        // Step 1: Filter tools if needed
        let body_ref = utils::filter_chat_request_by_tool_choice(request);

        // Step 2: Get cached trajectory if it exists
        let maybe_cached_traj_id = request.traj_id.as_ref();
        let cached_traj = if let Some(traj_id) = maybe_cached_traj_id {
            let traj = ctx.components.trajectory_map.get_mut(traj_id).or_else(|| {
                // Create a new trajectory
                ctx.components.trajectory_map.insert(traj_id.to_string(), Trajectory::default());
                ctx.components.trajectory_map.get_mut(traj_id)
            });
            match traj {
                Some(traj) => Some(traj),
                None => {
                    error!(function = "ChatPreparationStage::execute", "Failed to get cached trajectory");
                    return Err(error::internal_error("failed_to_get_cached_trajectory", "Failed to get cached trajectory"));
                }
            }
        } else {
            None
        };

        // Step 3: Process messages and apply chat template
        let processed_messages = match utils::process_chat_messages(&body_ref, &*tokenizer, true) {
            Ok(msgs) => msgs,
            Err(e) => {
                error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process chat messages");
                return Err(error::bad_request("process_messages_failed", e));
            }
        };

        // Step 4: Tokenize the processed text (no special tokens - chat template already handles them)
        let new_messages_token_ids = {
            let encoding = match tokenizer.encode(&processed_messages.text, false) {
                Ok(encoding) => encoding,
                Err(e) => {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                    return Err(error::internal_error(
                        "tokenization_failed",
                        format!("Tokenization failed: {}", e),
                    ));
                }
            };
            encoding.token_ids().to_vec()
        };

        // Step 5: Compute delta token ids if cached trajectory exists
        let token_ids = match cached_traj {
            Some(mut traj) => {
                let new_messages = &body_ref.messages;
                let cached_messages = &traj.cached_request.messages;
                let cached_length = cached_messages.len();
                if cached_length == 0 {
                    traj.cached_token_ids = new_messages_token_ids.clone();
                    traj.cached_request = body_ref.clone().into_owned();
                    traj.output_token_mask = vec![0; new_messages_token_ids.len()];
                    new_messages_token_ids
                } else {
                    let new_length = new_messages.len();
                    if new_length < cached_length {
                        return Err(error::bad_request("new_messages_too_short", "New messages are shorter than cached messages"));
                    }
                    let new_messages_prefix = &new_messages[..cached_length];
                    // Check whether the new messages prefix is equal to the cached messages
                    if extract_text_from_messages(new_messages_prefix) != extract_text_from_messages(cached_messages) {
                        return Err(error::bad_request("new_messages_prefix_not_equal_to_cached_messages", "New messages prefix is not equal to cached messages"));
                    }
                    // Check whether the tools are the same
                    let cached_tool_text = match serde_json::to_string(&traj.cached_request.tools) {
                        Ok(text) => text,
                        Err(e) => {
                            error!(function = "ChatPreparationStage::execute", error = %e, "Failed to serialize cached tools");
                            return Err(error::internal_error("serialize_cached_tools_failed", format!("Failed to serialize cached tools: {}", e)));
                        }
                    };
                    let new_tool_text = match serde_json::to_string(&body_ref.tools) {
                        Ok(text) => text,
                        Err(e) => {
                            error!(function = "ChatPreparationStage::execute", error = %e, "Failed to serialize new tools");
                            return Err(error::internal_error("serialize_new_tools_failed", format!("Failed to serialize new tools: {}", e)));
                        }
                    };
                    if cached_tool_text != new_tool_text {
                        return Err(error::bad_request("tools_not_equal", "Tools are not equal"));
                    }
                    // Perform delta tokenization
                    let cached_messages_token_ids = {
                        // Last message of cached_messages should be assistant message
                        let processed_cached_messages = match utils::process_chat_messages(&traj.cached_request, &*tokenizer, false) {
                            Ok(msgs) => msgs,
                            Err(e) => {
                                error!(function = "ChatPreparationStage::execute", error = %e, "Failed to process cached chat messages");
                                return Err(error::bad_request("process_messages_failed", e));
                            }
                        };
                        match tokenizer.encode(&processed_cached_messages.text, false) {
                            Ok(encoding) => encoding.token_ids().to_vec(),
                            Err(e) => {
                                error!(function = "ChatPreparationStage::execute", error = %e, "Tokenization failed");
                                return Err(error::internal_error("tokenization_failed", format!("Tokenization failed: {}", e)));
                            }
                        }
                    };
                    let cached_messages_token_ids_len = {
                        // Since the cached token ids are end with eos token, we need to truncate from the eos token
                        // to avoid missing any other tokens.
                        let eos_token = &tokenizer.get_special_tokens().eos_token;
                        match eos_token {
                            Some(eos_token) => {
                                let eos_token_id = tokenizer.token_to_id(eos_token);
                                if let Some(eos_token_id) = eos_token_id {
                                    //reverse search the eos_token_id
                                    let eos_token_id_index = cached_messages_token_ids.iter().rposition(|&id| id == eos_token_id);
                                    if let Some(eos_token_id_index) = eos_token_id_index {
                                        eos_token_id_index + 1
                                    } else {
                                        return Err(error::internal_error("eos_token_id_not_found", "EOS token ID not found"));
                                    }
                                } else {
                                    return Err(error::internal_error("eos_token_id_unknown", "EOS token ID unknown"));
                                }
                            }
                            None => cached_messages_token_ids.len(),
                        }
                    };
                
                    let delta_token_ids = new_messages_token_ids[cached_messages_token_ids_len..].to_vec();
                    let delta_token_mask = vec![0; delta_token_ids.len()];
                    traj.cached_token_ids.extend(delta_token_ids);
                    traj.cached_request = body_ref.clone().into_owned();
                    traj.output_token_mask.extend(delta_token_mask);
                    traj.cached_token_ids.clone()
                }
            },
            None => new_messages_token_ids,
        };

        // Step 6: Build tool constraints if needed
        let tool_call_constraint = if let Some(tools) = body_ref.tools.as_ref() {
            utils::generate_tool_constraints(tools, &request.tool_choice, &request.model)
                .map_err(|e| {
                    error!(function = "ChatPreparationStage::execute", error = %e, "Invalid tool configuration");
                    error::bad_request("invalid_tool_configuration", format!("Invalid tool configuration: {}", e))
                })?
        } else {
            None
        };

        // Step 7: Create stop sequence decoder (build once, reuse in non-stream)
        let stop_decoder = utils::create_stop_decoder(
            &tokenizer,
            request.stop.as_ref(),
            request.stop_token_ids.as_ref(),
            request.skip_special_tokens,
            request.no_stop_trim,
        );

        // Store results in context
        ctx.state.preparation = Some(PreparationOutput {
            original_text: Some(processed_messages.text.clone()),
            token_ids,
            processed_messages: Some(processed_messages),
            tool_constraints: tool_call_constraint,
            filtered_request: if matches!(body_ref, Cow::Owned(_)) {
                Some(body_ref.into_owned())
            } else {
                None
            },
            // Harmony fields (not used for regular preparation)
            harmony_mode: false,
            selection_text: None,
            harmony_messages: None,
            harmony_stop_ids: None,
        });

        // Store stop decoder for reuse in response processing
        ctx.state.response.stop_decoder = Some(stop_decoder);

        Ok(())
    }
}
