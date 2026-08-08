"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import asyncio
import json
import unittest
import uuid
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

import torch
from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    MessageProcessingResult,
    Trajectory,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import BatchStrOutput, GenerateReqInput
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.managers.tokenizer_manager import ReqState, TokenizerManager
from sglang.srt.utils import find_nth_token_index, get_or_create_event_loop
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=10, suite="stage-b-test-small-1-gpu-amd")


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.model_config.hf_eos_token_id = {2}
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
        )
        # Mock hf_config for _use_dpsk_v32_encoding check
        mock_hf_config = Mock()
        mock_hf_config.architectures = ["LlamaForCausalLM"]
        self.model_config.hf_config = mock_hf_config

        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1
        self.tokenizer.eos_token_id = 2

        # async generator stub for generate_request
        async def _mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = "llama-3"
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = None


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.chat = OpenAIServingChat(self.tm, self.template_manager)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    def test_find_nth_token_index(self):
        token_ids = [2, 10, 2, 20, 2]

        self.assertEqual(find_nth_token_index(token_ids, 2, 1), 0)
        self.assertEqual(find_nth_token_index(token_ids, 2, 3), 4)
        self.assertIsNone(find_nth_token_index(token_ids, 2, 4))
        self.assertIsNone(find_nth_token_index(token_ids, 2, 0))

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = MessageProcessingResult(
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    def test_multimodal_trajectory_requests_postprocessor_input_ids(self):
        self.tm.model_config.is_multimodal = True
        processed_messages = MessageProcessingResult(
            prompt="<image>Test prompt",
            prompt_ids=[10, 11, 12],
            image_data=["image-data"],
            video_data=None,
            audio_data=None,
            modalities=["image"],
            stop=[],
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=[1, 151655, 13, 20, 2],
            output_token_mask=[0, 0, 0, 1, 1],
            cached_token_logprobs=[0] * 5,
            cached_request=request,
            cached_tools_text="None",
            eos_token_id=2,
        )

        with patch.object(
            self.chat, "_process_messages", return_value=processed_messages
        ):
            adapted, _ = self.chat._convert_to_internal_request(request)

        self.assertEqual(adapted.text, processed_messages.prompt)
        self.assertIsNone(adapted.input_ids)
        self.assertTrue(adapted.return_input_ids)
        self.assertEqual(adapted.trajectory_input_ids, [1, 151655, 13, 20, 2])
        self.assertEqual(adapted.trajectory_eos_token_id, 2)

    def test_tokenizer_manager_restores_multimodal_trajectory_before_scheduler(self):
        manager = object.__new__(TokenizerManager)
        manager.mm_processor = object()
        manager.server_args = Mock(
            language_only=False,
            encoder_transfer_backend=None,
        )
        manager.tokenizer = Mock()
        manager.max_req_input_len = 100
        manager._tokenize_texts = AsyncMock(return_value=([999], None))
        manager._validate_mm_limits = Mock()
        manager._validate_one_request = Mock()

        processed_input_ids = [
            1,
            90,
            90,
            13,
            30,
            31,
            32,
            2,
            40,
            90,
            90,
            41,
            42,
            2,
            50,
            90,
            90,
            2,
            60,
        ]
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(1, 2), (9, 10), (15, 16)],
        )
        mrope_positions = torch.tensor(
            [
                [0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13, 14, 15, 15, 17, 18],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
                [0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 9, 11, 12, 13, 14, 15, 15, 17, 18],
            ]
        )
        mm_inputs = {
            "input_ids": processed_input_ids,
            "mm_items": [mm_item],
            "mrope_positions": mrope_positions,
            "mrope_position_delta": torch.tensor([[0]]),
        }
        manager.mm_data_processor = Mock(process=AsyncMock(return_value=mm_inputs))
        manager._create_tokenized_object = Mock(
            side_effect=lambda _obj, _text, input_ids, _embeds, inputs, _types: Mock(
                input_ids=input_ids, mm_inputs=inputs
            )
        )

        request = GenerateReqInput(
            text="<image> first turn <image> second turn <image> third turn",
            image_data=["image-1", "image-2", "image-3"],
            sampling_params={},
            trajectory_input_ids=[1, 90, 90, 13, 20, 2, 40, 90, 90, 21, 2],
            trajectory_eos_token_id=2,
            rid="request-id",
        )

        tokenized = asyncio.run(manager._tokenize_one_request(request))

        self.assertEqual(
            tokenized.input_ids,
            [1, 90, 90, 13, 20, 2, 40, 90, 90, 21, 2, 50, 90, 90, 2, 60],
        )
        self.assertEqual(tokenized.mm_inputs["input_ids"], tokenized.input_ids)
        self.assertEqual(mm_item.offsets, [(1, 2), (7, 8), (12, 13)])
        self.assertEqual(tokenized.mm_inputs["mrope_positions"].shape, (3, 16))
        self.assertTrue(
            torch.equal(
                tokenized.mm_inputs["mrope_positions"][:, 12:14],
                torch.tensor([[12, 12], [12, 13], [12, 12]]),
            )
        )

    def test_tokenizer_manager_returns_requested_input_ids(self):
        manager = object.__new__(TokenizerManager)
        request = GenerateReqInput(
            text="prompt",
            sampling_params={},
            return_input_ids=True,
            log_metrics=False,
            rid="request-id",
        )
        state = ReqState(
            out_list=[],
            finished=False,
            event=asyncio.Event(),
            obj=request,
            created_time=0,
            input_ids=[1, 151655, 151655, 13],
        )
        manager.rid_to_state = {request.rid: state}
        manager.server_args = Mock(
            weight_version="test",
            speculative_algorithm=None,
            enable_lora=False,
            dp_size=1,
        )
        manager.enable_metrics = False
        manager.enable_trace = False
        manager.dump_requests_folder = None
        manager.crash_dump_folder = None

        output = Mock(spec=BatchStrOutput)
        output.rids = [request.rid]
        output.finished_reasons = [{"type": "stop", "matched": 2}]
        output.prompt_tokens = [4]
        output.completion_tokens = [1]
        output.cached_tokens = [0]
        output.retraction_counts = [0]
        output.output_strs = ["answer"]
        output.output_ids = [[2]]
        output.output_hidden_states = None
        output.routed_experts = None
        output.customized_info = None
        output.cached_tokens_details = None
        output.load = None

        with patch("sglang.srt.managers.tokenizer_manager.trace_req_finish"):
            manager._handle_batch_output(output)

        self.assertEqual(state.out_list[0]["meta_info"]["input_ids"], state.input_ids)

    def test_multimodal_trajectory_defers_prompt_cache_until_postprocessing(self):
        self.template_manager.chat_template_name = None
        self.tm.tokenizer.eos_token_id = 2
        processed_messages = MessageProcessingResult(
            prompt="<image>Test prompt",
            prompt_ids=[10, 11, 12],
            image_data=["image-data"],
            video_data=None,
            audio_data=None,
            modalities=["image"],
            stop=[],
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )

        with patch.object(
            self.chat, "_apply_jinja_template", return_value=processed_messages
        ):
            result = self.chat._process_messages(request, is_multimodal=True)

        self.assertIs(result, processed_messages)
        trajectory = self.chat.traj_map[request.traj_id]
        self.assertEqual(trajectory.cached_token_ids, [])
        self.assertEqual(trajectory.output_token_mask, [])
        self.assertEqual(trajectory.cached_token_logprobs, [])

    def test_trajectory_uses_chat_template_eos_token(self):
        self.template_manager.chat_template_name = None
        self.tm.model_config.hf_eos_token_id = {1, 106}
        self.tm.tokenizer.eos_token_id = 1
        processed_messages = MessageProcessingResult(
            prompt="<image>Test prompt",
            prompt_ids=[2, 105, 10, 106, 105],
            image_data=["image-data"],
            video_data=None,
            audio_data=None,
            modalities=["image"],
            stop=[],
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )

        with patch.object(
            self.chat, "_apply_jinja_template", return_value=processed_messages
        ):
            self.chat._process_messages(request, is_multimodal=True)

        self.assertEqual(self.chat.traj_map[request.traj_id].eos_token_id, 106)

    def test_multimodal_trajectory_accepts_cached_message_prefix(self):
        self.template_manager.chat_template_name = None
        self.tm.model_config.is_multimodal = True
        cached_request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Describe the image"},
                {
                    "role": "assistant",
                    "reasoning_content": "The image contains a cat.",
                    "content": "A cat.",
                },
            ],
            traj_id="mm-trajectory",
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[
                *(message.model_dump() for message in cached_request.messages),
                {"role": "user", "content": "What color is it?"},
            ],
            traj_id="mm-trajectory",
        )
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=[1, 10, 2, 20, 2],
            output_token_mask=[0, 0, 0, 1, 1],
            cached_token_logprobs=[0] * 5,
            cached_request=cached_request,
            cached_tools_text="None",
            eos_token_id=2,
        )
        processed_messages = MessageProcessingResult(
            # A thinking model may strip historical reasoning here, so these
            # re-tokenized IDs need not start with the prior rendered prompt.
            prompt="rendered prompt without historical reasoning",
            prompt_ids=[1, 10, 2, 30, 2, 40],
            image_data=["image-data"],
            video_data=None,
            audio_data=None,
            modalities=["image"],
            stop=[],
        )

        with patch.object(
            self.chat,
            "_apply_jinja_template",
            return_value=processed_messages,
        ) as apply_template:
            result = self.chat._process_messages(request, is_multimodal=True)

        self.assertIs(result, processed_messages)
        apply_template.assert_called_once()
        self.assertIs(
            self.chat.traj_map[request.traj_id].cached_request,
            request,
        )

    def test_multimodal_trajectory_rejects_changed_message_prefix(self):
        self.template_manager.chat_template_name = None
        self.tm.model_config.is_multimodal = True
        cached_request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )
        request = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "Describe a different image"},
                {"role": "user", "content": "What color is it?"},
            ],
            traj_id="mm-trajectory",
        )
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=[1, 10, 2],
            output_token_mask=[0, 0, 0],
            cached_token_logprobs=[0] * 3,
            cached_request=cached_request,
            cached_tools_text="None",
            eos_token_id=2,
        )
        processed_messages = MessageProcessingResult(
            prompt="changed rendered prompt",
            prompt_ids=[1, 11, 2, 40],
            image_data=["image-data"],
            video_data=None,
            audio_data=None,
            modalities=["image"],
            stop=[],
        )

        with patch.object(
            self.chat,
            "_apply_jinja_template",
            return_value=processed_messages,
        ), self.assertRaisesRegex(
            ValueError,
            "The new prompt does not start with the cached prompt",
        ):
            self.chat._process_messages(request, is_multimodal=True)

    def test_multimodal_trajectory_caches_expanded_prompt_and_raw_output_ids(self):
        self.tm.model_config.is_multimodal = True
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=[],
            output_token_mask=[],
            cached_token_logprobs=[],
            cached_request=request,
            cached_tools_text="None",
            eos_token_id=2,
        )
        ret = [
            {
                "text": "A cat.",
                "output_ids": [20, 21, 2],
                "meta_info": {
                    "id": "chatcmpl-mm",
                    "input_ids": [1, 151655, 151655, 13],
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 2},
                    "weight_version": "test",
                },
            }
        ]

        self.chat._build_chat_response(request, ret, created=0)

        trajectory = self.chat.traj_map[request.traj_id]
        self.assertEqual(
            trajectory.cached_token_ids,
            [1, 151655, 151655, 13, 20, 21, 2],
        )
        self.assertEqual(trajectory.output_token_mask, [0, 0, 0, 0, 1, 1, 1])
        self.assertNotIn("input_ids", ret[0]["meta_info"])

    def test_multimodal_trajectory_accepts_additional_eos_token(self):
        self.tm.model_config.is_multimodal = True
        self.tm.model_config.hf_eos_token_id = {1, 106}
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=[],
            output_token_mask=[],
            cached_token_logprobs=[],
            cached_request=request,
            cached_tools_text="None",
            eos_token_id=1,
        )
        ret = [
            {
                "text": "A cat.",
                "output_ids": [20, 21, 106],
                "meta_info": {
                    "id": "chatcmpl-gemma-mm",
                    "input_ids": [2, 10, 106, 105],
                    "prompt_tokens": 4,
                    "completion_tokens": 3,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 106},
                    "weight_version": "test",
                },
            }
        ]

        self.chat._build_chat_response(request, ret, created=0)

        trajectory = self.chat.traj_map[request.traj_id]
        self.assertEqual(trajectory.eos_token_id, 106)
        self.assertEqual(
            trajectory.cached_token_ids,
            [2, 10, 106, 105, 20, 21, 106],
        )
        self.assertEqual(trajectory.output_token_mask, [0, 0, 0, 0, 1, 1, 1])

    def test_multimodal_trajectory_length_finish_appends_template_eos(self):
        self.tm.model_config.is_multimodal = True
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Describe the image"}],
            traj_id="mm-trajectory",
        )
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=[],
            output_token_mask=[],
            cached_token_logprobs=[],
            cached_request=request,
            cached_tools_text="None",
            eos_token_id=106,
        )
        ret = [
            {
                "text": "A",
                "output_ids": [20],
                "meta_info": {
                    "id": "chatcmpl-gemma-mm-length",
                    "input_ids": [2, 10, 106, 105],
                    "prompt_tokens": 4,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "length", "matched": None},
                    "weight_version": "test",
                },
            }
        ]

        self.chat._build_chat_response(request, ret, created=0)

        trajectory = self.chat.traj_map[request.traj_id]
        self.assertEqual(
            trajectory.cached_token_ids,
            [2, 10, 106, 105, 20, 106],
        )
        self.assertEqual(trajectory.output_token_mask, [0, 0, 0, 0, 1, 0])

    def test_multimodal_trajectory_continuation_preserves_raw_output_ids(self):
        self.tm.model_config.is_multimodal = True
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What color is it?"}],
            traj_id="mm-trajectory",
        )
        original_ids = [1, 151655, 151655, 13, 20, 21, 2]
        self.chat.traj_map[request.traj_id] = Trajectory(
            cached_token_ids=original_ids.copy(),
            output_token_mask=[0, 0, 0, 0, 1, 1, 1],
            cached_token_logprobs=[0] * len(original_ids),
            cached_request=request,
            cached_tools_text="None",
            eos_token_id=2,
        )
        ret = [
            {
                "text": "Orange.",
                "output_ids": [40, 2],
                "meta_info": {
                    "id": "chatcmpl-mm-2",
                    # The tokenizer manager has already restored the raw prior
                    # answer. The response path must append only the new suffix.
                    "input_ids": original_ids + [32, 33, 2, 34],
                    "prompt_tokens": 11,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": 2},
                    "weight_version": "test",
                },
            }
        ]

        self.chat._build_chat_response(request, ret, created=0)

        trajectory = self.chat.traj_map[request.traj_id]
        self.assertEqual(
            trajectory.cached_token_ids,
            original_ids + [32, 33, 2, 34, 40, 2],
        )
        self.assertEqual(
            trajectory.output_token_mask,
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        )

    def test_jinja_uses_openai_tool_schema_first(self):
        """Ensure Jinja chat templates receive OpenAI-shaped tools by default."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.chat._process_messages(req, is_multimodal=False)

        expected_tools = [tool.model_dump() for tool in req.tools]
        kwargs = self.tm.tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual(kwargs["tools"], expected_tools)

    def test_jinja_tool_schema_fallback_to_flat_function(self):
        """Fallback to function-only schema when template rejects OpenAI wrapper."""
        self.template_manager.chat_template_name = None
        self.template_manager.jinja_template_content_format = "string"

        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "add",
                        "description": "Add two numbers.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "integer"},
                                "b": {"type": "integer"},
                            },
                            "required": ["a", "b"],
                        },
                    },
                }
            ],
        )

        self.tm.tokenizer.apply_chat_template.side_effect = [
            RuntimeError("template expects flat tools format"),
            [1, 2, 3],
        ]

        self.chat._process_messages(req, is_multimodal=False)

        first_tools = self.tm.tokenizer.apply_chat_template.call_args_list[0].kwargs[
            "tools"
        ]
        second_tools = self.tm.tokenizer.apply_chat_template.call_args_list[1].kwargs[
            "tools"
        ]
        self.assertEqual(first_tools, [tool.model_dump() for tool in req.tools])
        self.assertEqual(
            second_tools, [tool.function.model_dump() for tool in req.tools]
        )

    def test_stop_str_isolation_between_requests(self):
        """Test that stop strings from one request don't affect subsequent requests.

        This tests the fix for the bug where conv.stop_str was being mutated globally,
        causing stop strings from one request to persist in subsequent requests.
        """
        # Mock conversation template with initial stop_str
        initial_stop_str = ["\n"]

        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock:
            # Create a mock conversation object that will be returned by generate_chat_conv
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = None
            conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = (
                initial_stop_str.copy()
            )  # Template's default stop strings
            conv_mock.return_value = conv_ins

            # First request with additional stop string
            req1 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "First request"}],
                stop=["CUSTOM_STOP"],
            )

            # Call the actual _apply_conversation_template method (not mocked)
            result1 = self.chat._apply_conversation_template(req1, is_multimodal=False)

            # Verify first request has both stop strings
            expected_stop1 = initial_stop_str + ["CUSTOM_STOP"]
            self.assertEqual(result1.stop, expected_stop1)

            # Verify the original template's stop_str wasn't mutated after first request
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

            # Second request without additional stop string
            req2 = ChatCompletionRequest(
                model="x",
                messages=[{"role": "user", "content": "Second request"}],
                # No custom stop strings
            )
            result2 = self.chat._apply_conversation_template(req2, is_multimodal=False)

            # Verify second request only has original stop strings (no CUSTOM_STOP from req1)
            self.assertEqual(result2.stop, initial_stop_str)
            self.assertNotIn("CUSTOM_STOP", result2.stop)
            self.assertEqual(conv_ins.stop_str, initial_stop_str)

    def test_unstreamed_tool_args_completion(self):
        """Test that remaining tool call arguments are sent when generation finishes."""

        # Mock FunctionCallParser with detector that has partial tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was partially streamed
        mock_detector.prev_tool_call_arr = [
            {
                "name": "get_weather",
                "arguments": {"location": "San Francisco", "unit": "celsius"},
            }
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"'  # Partial arguments streamed so far
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return a chunk with remaining arguments
        self.assertIsNotNone(result, "Should return chunk with remaining arguments")

        # Parse the result to verify content
        self.assertTrue(result.startswith("data: "))
        chunk = json.loads(result[6:])
        tool_calls = chunk["choices"][0]["delta"]["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        arguments = tool_calls[0]["function"]["arguments"]
        self.assertIn(', "unit": "celsius"}', arguments)

        self.assertIn(
            '"finish_reason":null',
            result,
            "Should not include finish_reason in completion chunk",
        )

    def test_unstreamed_tool_args_no_completion_needed(self):
        """Test that no completion chunk is sent when all arguments were already streamed."""

        # Mock FunctionCallParser with detector that has complete tool call data
        mock_parser = Mock()
        mock_detector = Mock()

        # Simulate a tool call that was completely streamed
        mock_detector.prev_tool_call_arr = [
            {"name": "get_weather", "arguments": {"location": "San Francisco"}}
        ]
        mock_detector.streamed_args_for_tool = [
            '{"location": "San Francisco"}'  # All arguments already streamed
        ]
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since no completion is needed
        self.assertIsNone(result, "Should return None when no completion is needed")

    def test_unstreamed_tool_args_no_parser_data(self):
        """Test that no completion chunk is sent when parser has no tool call data."""

        # Mock FunctionCallParser with empty detector
        mock_parser = Mock()
        mock_detector = Mock()
        mock_detector.prev_tool_call_arr = []
        mock_detector.streamed_args_for_tool = []
        mock_parser.detector = mock_detector

        content = {
            "meta_info": {
                "id": "chatcmpl-test123",
            }
        }

        request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
        )

        # Test the completion method
        result = self.chat._check_for_unstreamed_tool_args(
            parser=mock_parser,
            content=content,
            request=request,
            index=0,
        )

        # Should return None since there's no parser data
        self.assertIsNone(
            result, "Should return None when parser has no tool call data"
        )

    # ------------- kimi_k2 tool_call_id formatting -------------
    def test_kimi_k2_non_streaming_tool_call_id_format(self):
        """Ensure non-streaming tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Paris"}'
            call_info.tool_index = 0

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = ("", [call_info])

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            tool_calls, remaining_text, finish_reason = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
            )

            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 1)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:0")
            self.assertEqual(tool_calls[0].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_format(self):
        """Ensure streaming first chunk tool_call.id matches functions.{name}:{index} for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tools
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:0")

    def test_kimi_k2_non_streaming_tool_call_id_with_history(self):
        """Ensure non-streaming tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA and Tokyo?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=False,
        )

        # Mock FunctionCallParser.parse_non_stream to return one tool call
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # Build a mock ToolCallItem-like object
            call_info = Mock()
            call_info.name = "get_weather"
            call_info.parameters = '{"city":"Loa Angeles"}'
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            call_info.tool_index = 0

            call_info2 = Mock()
            call_info2.name = "get_weather"
            call_info2.parameters = '{"city":"Tokyo"}'
            call_info2.tool_index = 1

            parser_instance.has_tool_call.return_value = True
            parser_instance.parse_non_stream.return_value = (
                "",
                [call_info, call_info2],
            )

            finish_reason = {"type": "stop", "matched": None}
            tools = [
                {"type": "function", "function": {"name": "get_weather"}},
            ]

            history_tool_calls_cnt = self.chat._get_history_tool_calls_cnt(req)
            tool_calls, remaining_text, _ = self.chat._process_tool_calls(
                text="<|tool_calls_section_begin|>...",
                tools=tools,
                finish_reason=finish_reason,
                history_tool_calls_cnt=history_tool_calls_cnt,
            )

            self.assertEqual(history_tool_calls_cnt, 1)
            self.assertIsNotNone(tool_calls)
            self.assertEqual(len(tool_calls), 2)
            self.assertEqual(tool_calls[0].id, "functions.get_weather:1")
            self.assertEqual(tool_calls[0].function.name, "get_weather")
            self.assertEqual(tool_calls[1].id, "functions.get_weather:2")
            self.assertEqual(tool_calls[1].function.name, "get_weather")

    def test_kimi_k2_streaming_tool_call_id_with_history(self):
        """Ensure streaming first chunk tool_call.id increase with tool calls history for kimi_k2 parser."""

        # Force kimi_k2 parser
        self.chat.tool_call_parser = "kimi_k2"

        # Prepare request with tool calls history
        req = ChatCompletionRequest(
            model="x",
            messages=[
                {"role": "user", "content": "What's the weather today in paris?"},
                {
                    "role": "assistant",
                    "content": "Let me do some search first.",
                    "tool_calls": [
                        {
                            "id": "functions.get_weather:0",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "It's rainy in paris now.",
                    "tool_call_id": "functions.get_weather:0",
                },
                {
                    "role": "assistant",
                    "content": "It's rainy now.",
                },
                {
                    "role": "user",
                    "content": "What about LA?",
                },
            ],
            tools=[{"type": "function", "function": {"name": "get_weather"}}],
            stream=True,
        )

        # Patch FunctionCallParser used inside _process_tool_call_stream
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as ParserMock:
            parser_instance = ParserMock.return_value

            # First call returns one ToolCallItem-like chunk (with name)
            first_chunk_call = Mock()
            # Kimi-K2 series models might generate fixed number tool_indx,
            # ignoring the tool calls history and mess up all the following tool calls
            first_chunk_call.tool_index = 0
            first_chunk_call.name = "get_weather"
            first_chunk_call.parameters = ""
            parser_instance.parse_stream_chunk.side_effect = [
                ("", [first_chunk_call]),
                ("", []),
            ]

            async def collect_first_tool_chunk():
                gen = self.chat._process_tool_call_stream(
                    index=0,
                    delta="irrelevant",
                    parser_dict={},
                    content={"meta_info": {"id": "chatcmpl-test"}},
                    request=req,
                    has_tool_calls={},
                )
                # Get first yielded SSE line
                line = None
                async for emitted in gen:
                    line = emitted
                    break
                return line

            loop = get_or_create_event_loop()
            line = loop.run_until_complete(collect_first_tool_chunk())
            self.assertIsNotNone(line)
            self.assertTrue(line.startswith("data: "))

            payload = json.loads(line[len("data: ") :])
            tool_calls = payload["choices"][0]["delta"]["tool_calls"]
            self.assertEqual(tool_calls[0]["id"], "functions.get_weather:1")

    def test_dpsk_v32_encoding_path(self):
        """Test DeepSeek V3.2 encoding path detection and application."""
        from sglang.srt.managers.template_manager import TemplateManager
        from sglang.srt.server_args import PortArgs, ServerArgs

        server_args = ServerArgs(model_path="deepseek-ai/DeepSeek-V3.2")
        port_args = PortArgs.init_new(server_args)

        # Use mocks for TokenizerManager components to avoid full initialization
        with patch(
            "sglang.srt.managers.tokenizer_manager.TokenizerManager"
        ) as MockTokenizerManager:
            tokenizer_manager = MockTokenizerManager(server_args, port_args)
            tokenizer_manager.server_args = server_args
            tokenizer_manager.model_config = Mock()
            tokenizer_manager.model_config.get_default_sampling_params.return_value = (
                None
            )

            # Mock hf_config
            mock_hf_config = Mock()
            mock_hf_config.architectures = ["DeepseekV32ForCausalLM"]

            tokenizer_manager.model_config.hf_config = mock_hf_config

            # Case 1: No chat template in tokenizer -> should use dpsk encoding
            tokenizer_manager.tokenizer = Mock()
            tokenizer_manager.tokenizer.chat_template = None

            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertTrue(serving_chat.use_dpsk_v32_encoding)

            # Case 2: Chat template exists -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = "some template"
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)

            # Case 3: Not DeepSeek V3.2 architecture -> should NOT use dpsk encoding
            tokenizer_manager.tokenizer.chat_template = None
            mock_hf_config.architectures = ["LlamaForCausalLM"]
            serving_chat = OpenAIServingChat(tokenizer_manager, TemplateManager())
            self.assertFalse(serving_chat.use_dpsk_v32_encoding)


if __name__ == "__main__":
    unittest.main(verbosity=2)
