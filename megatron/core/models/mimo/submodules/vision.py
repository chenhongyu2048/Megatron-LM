# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from megatron.core.models.mimo.submodules.base import ModalitySubmodules


logger = logging.getLogger(__name__)


class VisionModalitySubmodules(ModalitySubmodules):
    """Vision modality submodules for encoding, decoding, and projecting image data.

    Handles image processing through vision encoders and projections in a multi-modal model.
    """

    def __init__(
        self,
        encoders: Optional[Dict[str, nn.Module]] = None,
        decoders: Optional[Dict[str, nn.Module]] = None,
        input_projections: Optional[List[nn.Module]] = None,
        output_projections: Optional[List[nn.Module]] = None,
        **kwargs,
    ):
        """Initialize vision modality submodules.

        Args:
            encoders: Dictionary of encoder modules
            decoders: Dictionary of decoder modules
            input_projections: List of input projection modules
            output_projections: List of output projection modules
            **kwargs: Additional keyword arguments
        """
        super().__init__(
            encoders=encoders,
            decoders=decoders,
            input_projections=input_projections,
            output_projections=output_projections,
            **kwargs,
        )

        if self.input_projections:
            assert (
                len(self.input_projections) <= 1
            ), "VisionModalitySubmodules currently supports only one input projection"

        if self.output_projections:
            assert (
                len(self.output_projections) <= 1
            ), "VisionModalitySubmodules currently supports only one output projection"

    def decode(self, embeddings, data_batch: Dict):
        """Decode embeddings into image tensors."""
        raise NotImplementedError("No decoders support yet")

    def combine_embeddings(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Combine multiple embeddings from different encoders by concatenation.

        This method is used for combining encoder outputs before input projection.

        Args:
            embeddings: List of embeddings to combine

        Returns:
            Combined embedding tensor
        """
        if not embeddings:
            raise ValueError("Cannot combine empty list of embeddings")

        if len(embeddings) == 1:
            return embeddings[0]

        # each embedding is [total_tokens, hidden_dim]
        #  Make this configurable in the future
        combined = torch.cat(embeddings, dim=0)
        logger.debug(f"Combined embeddings shape after concatenation: {combined.shape}")
        return combined

    def project_embeddings(
        self, embeddings: List[torch.Tensor], is_input: bool = True
    ) -> torch.Tensor:
        """Project image embeddings using input or output projections.

        Args:
            embeddings: List of image embeddings to project
            is_input: If True, use input projections, otherwise use output projections

        Returns:
            Projected image embeddings or None if no embeddings
        """
        if is_input:
            embeddings = self.combine_embeddings(embeddings)

        # Get the appropriate projection (input or output)
        projections = self.input_projections if is_input else self.output_projections

        # Apply projection if available
        if projections:
            # We've asserted in __init__ that there's only one projection
            projection = projections[0]
            projected = projection(embeddings)
            logger.debug(f"Post-projection embeddings shape: {projected.shape}")
            return projected

        return embeddings

    def forward(
        self,
        encoder_inputs: Optional[Dict[str, Any]] = None,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Process image data through encoding and projection.

        Args:
            encoder_inputs: Dictionary where keys match encoder names in self.encoders
                and values are dictionaries of encoder-specific parameters.
                Used when this vision module is the first stage.
            hidden_states: Hidden states from previous pipeline stage.
                Used when this vision module is not the first stage.

        Returns:
            Flattened image embeddings with shape [total_embeddings, hidden_dim],
            or None if no valid inputs were provided.
        """
        if self.is_first_stage:
            if encoder_inputs is None:
                return None
            embeddings = self.encode(encoder_inputs)
            if not embeddings:
                return None
            combined = self.combine_embeddings(embeddings)
        else:
            if hidden_states is None:
                return None
            combined = hidden_states

        if self.is_last_stage:
            projected = self.project_embeddings([combined], is_input=True)
            logger.debug(f"Projected vision embeddings shape: {projected.shape}")
            return projected

        return combined
