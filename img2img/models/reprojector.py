"""Module for reprojector models."""

import torch
import torch.nn
import torch.nn.functional

from img2img.models.mixer import MLPMixerBlock


class SimpleReprojectorModel(torch.nn.Module):
    """Simplest reprojector model. Basically just two linear layers."""

    def __init__(
        self, image_embedder_dims: tuple[int, int], text_embedder_dims: tuple[int, int]
    ):
        super().__init__()
        self.first_stage = torch.nn.Linear(
            image_embedder_dims[1], text_embedder_dims[1]
        )
        self.second_stage = torch.nn.Linear(
            image_embedder_dims[0], text_embedder_dims[0]
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Run the forward pass of the model.

        Parameters
        ----------
        image_embeddings : torch.Tensor
            The image embeddings.

        Returns
        -------
        torch.Tensor
            The text embeddings.
        """
        # image_embeddings is (batch_size, image_embedder_dims[0], image_embedder_dims[1])
        values = self.first_stage(image_embeddings)
        # x is (batch_size, image_embedder_dims[0], text_embedder_dims[1])
        # relu
        values = torch.nn.functional.relu(values)
        # transpose x to (batch_size, text_embedder_dims[1], image_embedder_dims[0])
        values = values.transpose(1, 2)
        # x is (batch_size, text_embedder_dims[1], image_embedder_dims[0])
        values = self.second_stage(values)
        # x is (batch_size, text_embedder_dims[1], text_embedder_dims[0])
        # transpose x to (batch_size, text_embedder_dims[0], text_embedder_dims[1])
        values = values.transpose(1, 2)
        # x is (batch_size, text_embedder_dims[0], text_embedder_dims[1])
        return values


class MixerReprojectorModel(torch.nn.Module):
    """A model that takes in an image embedding and outputs a text embedding.
    Internally it uses a first stage of MLPMixer blocks and a second stage of
    a simple reprojector model."""

    def __init__(
        self,
        image_embedder_dims: tuple[int, int],
        text_embedder_dims: tuple[int, int],
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()
        self.first_stage = torch.nn.Sequential(
            *[
                MLPMixerBlock(image_embedder_dims, hidden_size, hidden_size)
                for _ in range(num_layers)
            ]
        )
        self.second_stage = SimpleReprojectorModel(
            image_embedder_dims, text_embedder_dims
        )

    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """Run the forward pass of the model.

        Parameters
        ----------
        image_embeddings : torch.Tensor
            The image embeddings.

        Returns
        -------
        torch.Tensor
            The text embeddings.
        """
        # image_embeddings is (batch_size, image_embedder_dims[0], image_embedder_dims[1])
        values = self.first_stage(image_embeddings)
        # x is (batch_size, image_embedder_dims[0], image_embedder_dims[1])
        values = self.second_stage(values)
        # x is (batch_size, text_embedder_dims[0], text_embedder_dims[1])
        return values
