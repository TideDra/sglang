"""Inference-only LLaVa model compatible with HuggingFace weights."""

from typing import List, Optional

import numpy as np
import torch
from sglang.srt.managers.router.infer_batch import ForwardMode
from sglang.srt.managers.router.model_runner import InputMetadata
from sglang.srt.mm_utils import (
    get_anyres_image_grid_shape,
    unpad_image,
    unpad_image_shape,
)
from sglang.srt.models.llama2 import LlamaForCausalLM
from sglang.srt.models.llava import LlavaLlamaForCausalLM
from torch import nn
from transformers import CLIPVisionModel, LlamaConfig, LlavaConfig
from transformers.models.llava.modeling_llava import LlavaMultiModalProjector
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.model_executor.weight_utils import (
    default_weight_loader,
    hf_model_weights_iterator,
)


class LlavaForConditionalGeneration(LlavaLlamaForCausalLM):
    pass


EntryClass = LlavaForConditionalGeneration
