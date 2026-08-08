import copy
import re
from typing import List, Union

import torch

from sglang.srt.models.phi3v import Phi3VForCausalLM
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

IMAGE_TOKEN_ID = 32044
IMAGE_TOKEN_PATTERN = re.compile(r"<\|image_\d+\|>")


def renumber_image_tokens(prompt: str) -> str:
    """Make image placeholders contiguous in their order of appearance."""
    image_idx = 0

    def replacement(_match: re.Match) -> str:
        nonlocal image_idx
        image_idx += 1
        return f"<|image_{image_idx}|>"

    return IMAGE_TOKEN_PATTERN.sub(replacement, prompt)


class Phi3VMultimodalProcessor(BaseMultimodalProcessor):
    models = [Phi3VForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        # Microsoft recommends 16 crops for single-image inputs and 4 crops for
        # multi-image inputs. Keep separate processor objects so concurrent
        # requests cannot race while changing the image processor configuration.
        self._single_image_processor = copy.copy(_processor)
        self._single_image_processor.image_processor = copy.copy(
            _processor.image_processor
        )
        self._single_image_processor.image_processor.num_crops = 16
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image_1|>",
            image_token_id=IMAGE_TOKEN_ID,
            image_token_regex=IMAGE_TOKEN_PATTERN,
        ).build(_processor)

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ) -> dict:
        if videos or audios:
            raise ValueError("Phi-3.5 Vision only supports image inputs")

        # SGLang conversation templates use one static image placeholder. The
        # upstream Phi3V processor requires placeholders numbered 1..N.
        input_text = renumber_image_tokens(input_text)
        processor = (
            getattr(self, "_single_image_processor", self._processor)
            if images is not None and len(images) == 1
            else self._processor
        )
        result = processor(
            text=input_text,
            images=images,
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        # The upstream processor expands <|image_n|> into negative token IDs
        # (-1 for image 1, -2 for image 2, ...). SGLang uses one positive token
        # ID plus per-item hashes so the prompt can participate in RadixAttention.
        input_ids = result["input_ids"]
        result["input_ids"] = input_ids.masked_fill(input_ids < 0, IMAGE_TOKEN_ID)

        if not self.server_args.keep_mm_feature_on_device:
            for feature_name in self.FEATURE_NAMES:
                feature = result.get(feature_name)
                if isinstance(feature, torch.Tensor):
                    result[feature_name] = feature.cpu()
        return result

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        return {
            "input_ids": input_ids.tolist(),
            "mm_items": mm_items,
            "im_token_id": IMAGE_TOKEN_ID,
        }
