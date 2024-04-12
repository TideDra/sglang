"""Inference-only LLaVa model compatible with HuggingFace weights."""
from sglang.srt.models.llava import LlavaLlamaForCausalLM


class LlavaForConditionalGeneration(LlavaLlamaForCausalLM):
    pass

EntryClass = LlavaForConditionalGeneration
