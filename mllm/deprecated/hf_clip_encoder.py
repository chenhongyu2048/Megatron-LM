# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
from transformers import CLIPVisionModel, LlavaNextVideoConfig
from transformers import CLIPVisionConfig
from transformers.models.llava_next_video.modeling_llava_next_video import (
    LlavaNextVideoPooler,
)


class HFCLIPEncoderWrapper(torch.nn.Module):
    """CLIP encoder wrapper that extracts last_hidden_state."""

    def __init__(self, feature_layer_index=-2, is_video_input: bool = False):
        """Initialize the HFCLIPEncoderWrapper.

        Args:
            feature_layer_index (int): Index of the feature layer to extract from the encoder's hidden states.
                                       Default is -2 (second to last layer).
            is_video_input (bool): If True, expects video input and applies vision resampler.
        """
        super().__init__()
        import os
        # clip_model_path = os.path.expanduser('~/run/.local/clip-vit-large-patch14-336')
        # self.encoder = CLIPVisionModel.from_pretrained(clip_model_path)
        clip_config = CLIPVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            image_size=336,
            patch_size=14,
            projection_dim=768,
        )
        self.encoder = CLIPVisionModel(clip_config)
        
        # self.encoder.eval()
        self.feature_layer_index = feature_layer_index
        self.is_video_input = is_video_input
        if self.is_video_input:
            config = LlavaNextVideoConfig()
            self.vision_resampler = LlavaNextVideoPooler(config)
            
        # Freeze parameters that are not in the computation graph path
        # feature_layer_index=-2 means the last 1 layer and post_layernorm are not used
        self._freeze_unused_layers()

    def _freeze_unused_layers(self):
        """Freeze layers that do not participate in forward computation based on feature_layer_index."""
        if self.feature_layer_index < -1 or (
            self.feature_layer_index >= 0
            and self.feature_layer_index < len(self.encoder.vision_model.encoder.layers) - 1
        ):
            # Calculate how many layers are not used
            total_layers = len(self.encoder.vision_model.encoder.layers)
            if self.feature_layer_index < 0:
                # -2 -> last 1 layer not used, -3 -> last 2 layers not used
                num_unused = -self.feature_layer_index - 1
            else:
                # Positive index: hidden_states[i] corresponds to the (i-1)-th layer output (0 is embedding)
                # Therefore, layers after feature_layer_index are not used
                num_unused = total_layers - self.feature_layer_index - 1

            # Freeze the last num_unused layers
            for i in range(total_layers - num_unused, total_layers):
                for param in self.encoder.vision_model.encoder.layers[i].parameters():
                    param.requires_grad = False

            # post_layernorm is also not in the computation path
            for param in self.encoder.vision_model.post_layernorm.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor):
        """Input: (B, F, 3, 336, 336) if video, else (B, 3, 336, 336) or (num_frames, 3, 336, 336)."""
        # Process through encoder and extract last_hidden_state
        # with torch.no_grad():
        if self.is_video_input:
            batch_size, frames, channels, height, width = pixel_values.shape
            pixel_values = pixel_values.reshape(batch_size * frames, channels, height, width)
        

        last_hidden_state = self.encoder(pixel_values, output_hidden_states=True)
        # -1 index is image features
        image_features = last_hidden_state[-1]
        # select last but second layer
        image_features = image_features[self.feature_layer_index]
        # drop cls token
        image_features = image_features[:, 1:, :]
        if self.is_video_input:
            image_features = self.vision_resampler(image_features)
        return image_features