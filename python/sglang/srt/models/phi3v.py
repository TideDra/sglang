# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/phi3v.py
# https://huggingface.co/microsoft/Phi-3.5-vision-instruct/blob/main/modeling_phi3_v.py

import copy
import logging
from collections.abc import Iterable
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPVisionConfig, PretrainedConfig

from sglang.srt.layers.quantization import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.clip import CLIPVisionModel
from sglang.srt.models.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)


CLIP_VIT_LARGE_PATCH14_336_CONFIG = CLIPVisionConfig(
    attention_dropout=0.0,
    dropout=0.0,
    hidden_act="quick_gelu",
    hidden_size=1024,
    image_size=336,
    initializer_factor=1.0,
    initializer_range=0.02,
    intermediate_size=4096,
    layer_norm_eps=1e-5,
    num_attention_heads=16,
    num_channels=3,
    num_hidden_layers=24,
    patch_size=14,
    projection_dim=768,
)


class Phi3VImageEncoder(nn.Module):
    """CLIP-HD image encoder used by Phi-3 and Phi-3.5 Vision."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        hidden_size = getattr(config, "n_embd", config.hidden_size)
        image_config = config.img_processor
        embedding_config = config.embd_layer

        if image_config.get("model_name") != "openai/clip-vit-large-patch14-336":
            raise NotImplementedError(
                f"Unsupported Phi-3V image processor: {image_config.get('model_name')}"
            )

        # The checkpoint selects hidden_states[-2], so only the first 23 of the
        # 24 CLIP layers are needed.
        clip_config = copy.deepcopy(CLIP_VIT_LARGE_PATCH14_336_CONFIG)
        layer_idx = image_config.get("layer_idx", -2)
        clip_config.num_hidden_layers = (
            clip_config.num_hidden_layers + layer_idx + 1
            if layer_idx < 0
            else layer_idx + 1
        )
        self.img_processor = CLIPVisionModel(
            clip_config,
            quant_config=quant_config,
            prefix=f"{prefix}.img_processor" if prefix else "img_processor",
        )

        self.image_dim_out = image_config["image_dim_out"]
        self.num_img_tokens = image_config["num_img_tokens"]
        self.type_feature = image_config.get("type_feature", "patch")
        self.num_clip_layers = clip_config.num_hidden_layers

        self.use_hd_transform = embedding_config.get("use_hd_transform", False)
        self.with_learnable_separator = embedding_config.get(
            "with_learnable_separator", False
        )
        self.hd_transform_order = embedding_config.get("hd_transform_order", "glb_sub")
        if not self.use_hd_transform or not self.with_learnable_separator:
            raise NotImplementedError("Phi-3V requires the CLIP-HD transform")
        if self.hd_transform_order != "sub_glb":
            raise NotImplementedError(
                f"Unsupported Phi-3V HD transform order: {self.hd_transform_order}"
            )

        merged_dim = self.image_dim_out * 4
        self.glb_GN = nn.Parameter(torch.empty(1, 1, merged_dim))
        self.sub_GN = nn.Parameter(torch.empty(1, 1, 1, merged_dim))

        if embedding_config.get("projection_cls") != "mlp":
            raise NotImplementedError(
                f"Unsupported Phi-3V projection: {embedding_config.get('projection_cls')}"
            )
        self.img_projection = nn.Sequential(
            nn.Linear(merged_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def get_img_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_model = self.img_processor.vision_model
        hidden_states = vision_model.embeddings(pixel_values.to(vision_model.device))
        hidden_states = vision_model.pre_layrnorm(hidden_states)
        hidden_states = vision_model.encoder(inputs_embeds=hidden_states)

        if self.type_feature == "patch":
            return hidden_states[:, 1:]
        if self.type_feature == "cls_patch":
            return hidden_states
        raise NotImplementedError(
            f"Unsupported Phi-3V vision feature type: {self.type_feature}"
        )

    @staticmethod
    def reshape_hd_patches_2x2merge(
        image_features: torch.Tensor, h_crop: int, w_crop: int
    ) -> torch.Tensor:
        num_crops, num_patches, channels = image_features.shape
        if num_patches != 576 or channels != 1024:
            raise ValueError(
                "Phi-3V expects 576 CLIP patches with 1024 channels, got "
                f"{num_patches} patches with {channels} channels"
            )
        if num_crops % (h_crop * w_crop) != 0:
            raise ValueError("The number of image crops does not match image_sizes")

        num_images = num_crops // (h_crop * w_crop)
        height = int(num_patches**0.5)
        return (
            image_features.reshape(num_crops, height, height, channels)
            .reshape(
                num_crops,
                height // 2,
                2,
                height // 2,
                2,
                channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(num_crops, -1, 4 * channels)
            .reshape(
                num_images,
                h_crop,
                w_crop,
                height // 2,
                height // 2,
                4 * channels,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(
                num_images,
                h_crop * height // 2,
                w_crop * height // 2,
                4 * channels,
            )
        )

    def add_image_newline(self, image_features: torch.Tensor) -> torch.Tensor:
        num_images, height, _width, hidden_size = image_features.shape
        newlines = self.sub_GN.expand(num_images, height, -1, -1)
        return torch.cat([image_features, newlines], dim=2).reshape(
            num_images, -1, hidden_size
        )

    def hd_feature_transform(
        self, image_features: torch.Tensor, image_sizes: torch.Tensor
    ) -> List[torch.Tensor]:
        projection_param = next(self.img_projection.parameters())
        target_device = projection_param.device
        target_dtype = projection_param.dtype

        global_features = image_features[:, 0]
        global_features = self.reshape_hd_patches_2x2merge(global_features, 1, 1)
        global_features = self.add_image_newline(global_features)

        projected_features = []
        for image_idx, image_size in enumerate(image_sizes):
            height, width = image_size
            h_crop = int(height // 336)
            w_crop = int(width // 336)
            num_crops = h_crop * w_crop

            sub_features = image_features[image_idx, 1 : 1 + num_crops]
            sub_features = self.reshape_hd_patches_2x2merge(
                sub_features, h_crop, w_crop
            )
            sub_features = self.add_image_newline(sub_features).squeeze(0)

            combined = torch.cat(
                [sub_features, self.glb_GN.squeeze(0), global_features[image_idx]]
            )
            projected_features.append(
                self.img_projection(combined.to(target_device, target_dtype))
            )

        return projected_features

    def forward(
        self, pixel_values: torch.Tensor, image_sizes: torch.Tensor
    ) -> List[torch.Tensor]:
        num_images, num_crops, channels, height, width = pixel_values.shape
        if channels != 3 or height != 336 or width != 336:
            raise ValueError(
                "Phi-3V expects image crops shaped (3, 336, 336), got "
                f"({channels}, {height}, {width})"
            )

        image_features = self.get_img_features(pixel_values.flatten(0, 1))
        image_features = image_features.reshape(
            num_images, num_crops, -1, self.image_dim_out
        )
        return self.hd_feature_transform(image_features, image_sizes)


class Phi3VForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # "su" was the original name for Phi-3 LongRoPE. SGLang and newer
        # Transformers releases use the standardized "longrope" name.
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
            if rope_type == "su":
                config.rope_scaling = dict(rope_scaling)
                key = "rope_type" if "rope_type" in rope_scaling else "type"
                config.rope_scaling[key] = "longrope"

        self.config = config
        self.language_model = LlamaForCausalLM(
            config=config, quant_config=quant_config, prefix=prefix
        )
        self.vision_encoder = Phi3VImageEncoder(
            config,
            quant_config=quant_config,
            prefix="model.vision_embed_tokens",
        )

    def get_image_feature(self, items: List[MultimodalDataItem]) -> torch.Tensor:
        encoder_param = next(self.vision_encoder.parameters())
        pixel_values = torch.cat([item.feature for item in items], dim=0).to(
            device=encoder_param.device, dtype=encoder_param.dtype
        )
        image_sizes = torch.cat([item.image_sizes for item in items], dim=0).to(
            encoder_param.device
        )
        image_features = self.vision_encoder(pixel_values, image_sizes)
        return torch.cat(image_features, dim=0).to(encoder_param.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ):
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.IMAGE: self.get_image_feature},
            positions=positions,
        )

    def pad_input_ids(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        return pattern.pad_input_tokens(input_ids, mm_inputs)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        stacked_params_mapping = [
            (".self_attn.qkv_proj", ".self_attn.q_proj", "q"),
            (".self_attn.qkv_proj", ".self_attn.k_proj", "k"),
            (".self_attn.qkv_proj", ".self_attn.v_proj", "v"),
        ]
        prefix_mapping = {
            "model.vision_embed_tokens.": "vision_encoder.",
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
        }

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            for old_prefix, new_prefix in prefix_mapping.items():
                if name.startswith(old_prefix):
                    name = new_prefix + name[len(old_prefix) :]
                    break

            if name.startswith(
                "vision_encoder.img_processor.vision_model.encoder.layers."
            ):
                layer_idx = int(name.split(".")[5])
                if layer_idx >= self.vision_encoder.num_clip_layers:
                    continue

            name = name.replace(".self_attn.out_proj", ".self_attn.proj")

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is None:
                    logger.warning("Parameter %s not found in Phi3V", name)
                    break
                param.weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict.get(name)
                if param is None:
                    logger.warning("Parameter %s not found in Phi3V", name)
                    continue
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)


EntryClass = [Phi3VForCausalLM]
