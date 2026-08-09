import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.multimodal.processors.phi3v import (
    IMAGE_TOKEN_ID,
    Phi3VMultimodalProcessor,
    renumber_image_tokens,
)
from sglang.srt.parser.conversation import (
    chat_templates,
    get_conv_template_by_model_path,
)


class _FakePhi3VProcessor:
    def __init__(self, num_crops=4):
        self.call_kwargs = None
        self.image_processor = SimpleNamespace(num_crops=num_crops)

    def __call__(self, **kwargs):
        self.call_kwargs = kwargs
        return {
            "input_ids": torch.tensor([[1, -1, -1, 10, -2, 2]]),
            "pixel_values": torch.zeros(2, 5, 3, 336, 336),
            "image_sizes": torch.tensor([[336, 336], [336, 336]]),
        }


class TestPhi3VProcessor(unittest.TestCase):
    def test_renumber_image_tokens(self):
        prompt = "a<|image_9|>b<|image_9|>c<|image_2|>"
        self.assertEqual(
            renumber_image_tokens(prompt),
            "a<|image_1|>b<|image_2|>c<|image_3|>",
        )

    def test_process_mm_data_replaces_negative_ids(self):
        hf_processor = _FakePhi3VProcessor()
        processor = object.__new__(Phi3VMultimodalProcessor)
        processor._processor = hf_processor
        processor.server_args = SimpleNamespace(keep_mm_feature_on_device=False)
        processor.FEATURE_NAMES = ["pixel_values"]

        result = processor.process_mm_data(
            "<|image_1|><|image_1|>", images=[object(), object()]
        )

        self.assertEqual(hf_processor.call_kwargs["text"], "<|image_1|><|image_2|>")
        self.assertEqual(
            result["input_ids"].tolist(),
            [[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 10, IMAGE_TOKEN_ID, 2]],
        )

    def test_single_image_uses_16_crops_without_changing_multi_image_processor(self):
        hf_processor = _FakePhi3VProcessor()
        processor = object.__new__(Phi3VMultimodalProcessor)
        processor._processor = hf_processor
        processor._single_image_processor = _FakePhi3VProcessor(num_crops=16)
        processor.server_args = SimpleNamespace(keep_mm_feature_on_device=False)
        processor.FEATURE_NAMES = ["pixel_values"]

        processor.process_mm_data("<|image_1|>", images=[object()])

        self.assertIsNone(hf_processor.call_kwargs)
        self.assertIsNotNone(processor._single_image_processor.call_kwargs)
        self.assertEqual(hf_processor.image_processor.num_crops, 4)
        self.assertEqual(
            processor._single_image_processor.image_processor.num_crops, 16
        )

    def test_trajectory_single_image_uses_stable_multi_image_crop_policy(self):
        hf_processor = _FakePhi3VProcessor()
        processor = object.__new__(Phi3VMultimodalProcessor)
        processor._processor = hf_processor
        processor._single_image_processor = _FakePhi3VProcessor(num_crops=16)
        processor.server_args = SimpleNamespace(keep_mm_feature_on_device=False)
        processor.FEATURE_NAMES = ["pixel_values"]

        processor.process_mm_data(
            "<|image_1|>",
            images=[object()],
            use_single_image_processor=False,
        )

        self.assertIsNotNone(hf_processor.call_kwargs)
        self.assertIsNone(processor._single_image_processor.call_kwargs)


class TestPhi3VConversation(unittest.TestCase):
    def test_builtin_jinja_template_preserves_multimodal_content(self):
        tokenizer = SimpleNamespace(chat_template=None)
        tokenizer_manager = SimpleNamespace(
            tokenizer=tokenizer,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="phi3_v")
            ),
        )
        template_manager = TemplateManager()

        template_manager.load_chat_template(
            tokenizer_manager,
            chat_template_arg=None,
            model_path="microsoft/Phi-3.5-vision-instruct",
        )

        self.assertIsNone(template_manager.chat_template_name)
        self.assertEqual(template_manager.jinja_template_content_format, "openai")
        self.assertIn("<|image_1|>", tokenizer.chat_template)
        self.assertIn("'<|end|>' + eos_token", tokenizer.chat_template)

    def test_model_template_match_and_prompt(self):
        self.assertEqual(
            get_conv_template_by_model_path("microsoft/Phi-3.5-vision-instruct"),
            "phi-3-vision",
        )

        conversation = chat_templates["phi-3-vision"].copy()
        conversation.append_message(
            conversation.roles[0], "<|image_1|>Describe this image."
        )
        conversation.append_message(conversation.roles[1], None)
        self.assertEqual(
            conversation.get_prompt(),
            "<|user|>\n<|image_1|>Describe this image.<|end|>\n" "<|assistant|>\n",
        )

    def test_system_prompt(self):
        conversation = chat_templates["phi-3-vision"].copy()
        conversation.system_message = "Be concise."
        conversation.append_message(conversation.roles[0], "Hello")
        conversation.append_message(conversation.roles[1], None)
        self.assertEqual(
            conversation.get_prompt(),
            "<|system|>\nBe concise.<|end|>\n"
            "<|user|>\nHello<|end|>\n"
            "<|assistant|>\n",
        )


if __name__ == "__main__":
    unittest.main()
