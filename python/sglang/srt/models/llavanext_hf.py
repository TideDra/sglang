"""Inference-only LLaVa model compatible with HuggingFace weights."""
from sglang.srt.models.llava import LlavaLlamaForCausalLM


class LlavaNextForConditionalGeneration(LlavaLlamaForCausalLM):
    pass

EntryClass = LlavaNextForConditionalGeneration
