"""Generic implementation of the MLPMixer model.
"""

import torch


class MLPBlock(torch.nn.Module):
    """Multi-layer perceptron block.

    Attributes
    ----------
    linear1: torch.nn.Linear
        The first linear layer.
    linear2: torch.nn.Linear
        The second linear layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        x = self.linear1(x)
        x = torch.nn.functional.gelu(x)
        x = self.linear2(x)
        return x


class MLPMixerBlock(torch.nn.Module):
    """MLP Mixer block.

    Attributes
    ----------
    mlp1: MLPBlock
        The first MLP block.
    mlp2: MLPBlock
        The second MLP block.
    layernorm1: torch.nn.LayerNorm
        The first layer norm.
    layernorm2: torch.nn.LayerNorm
        The second layer norm.
    """

    def __init__(self, minibatch_shape, mlp1_hidden_size, mlp2_hidden_size):
        super().__init__()
        self.mlp1 = MLPBlock(minibatch_shape[0], mlp1_hidden_size, minibatch_shape[0])
        self.mlp2 = MLPBlock(minibatch_shape[1], mlp2_hidden_size, minibatch_shape[1])
        self.layernorm1 = torch.nn.LayerNorm(minibatch_shape)
        self.layernorm2 = torch.nn.LayerNorm(minibatch_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """

        # x is (batch, entry, feature)
        normed1 = self.layernorm1(x)
        # normed1 is (batch, entry, feature)
        # now we tranpose it to (batch, feature, entry)
        normed1 = normed1.transpose(1, 2)
        # feed it to the first mlp
        mlped1 = self.mlp1(normed1)
        # mlped1 is (batch, feature, entry)
        # now we tranpose it to (batch, entry, feature)
        mlped1 = mlped1.transpose(1, 2)
        # apply second norm, with skip connection
        normed2 = self.layernorm2(mlped1 + x)
        # feed it to the second mlp
        mlped2 = self.mlp2(normed2)
        # add skip connection
        mlped2 = mlped2 + mlped1

        return mlped2
