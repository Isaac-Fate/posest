from torch import Tensor
from torch import nn

from .conv import ConvBNReLU


class InvertedResidual(nn.Module):
    """
    References
    ----------
        This implementation refers to https://github.com/pytorch/vision/blob/1aef87d01eec2c0989458387fa04baebcc86ea7b/torchvision/models/mobilenet.py#L45.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        expansion_ratio: float,
    ):
        super().__init__()

        # The stride can either be 1 or 2
        assert stride in (1, 2), "Stride must either be 1 or 2"

        # Whether to use the residual connection
        self._has_res_connection = stride == 1 and in_channels == out_channels

        # Calculate the hiddin dim
        # This determines how thick the internal layers are
        hidden_dim = round(in_channels * expansion_ratio)

        # Layers
        layers = []

        if in_channels != hidden_dim:
            layers.append(
                ConvBNReLU(
                    in_channels,
                    hidden_dim,
                    kernel_size=1,
                )
            )

        layers.extend(
            [
                ConvBNReLU(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                    stride=stride,
                    # One filter in each group
                    groups=hidden_dim,
                ),
                nn.Conv2d(
                    hidden_dim,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            ]
        )

        # Wrap the layers
        self.conv = nn.Sequential(*layers)

    @property
    def has_res_connection(self) -> bool:
        """Whether there is a residual connection in this block."""

        return self._has_res_connection

    def forward(self, x: Tensor) -> Tensor:

        # Apply residual connection short cut
        if self._has_res_connection:
            return self.conv(x) + x

        # No residual connection
        # since the shapes do not match
        else:
            return self.conv(x)


class InvertedResidualBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int,
        expansion_ratio: float,
        n_layers: int,
    ) -> None:

        super().__init__()

        # Layers consisting of inverted residual layers
        layers = []

        # Add the first layer
        layers.append(
            InvertedResidual(
                in_channels,
                out_channels,
                # Only the first layer has the specified stride
                stride=stride,
                expansion_ratio=expansion_ratio,
            )
        )

        # The rest layers
        for _ in range(1, n_layers):

            # Update the number of input channels
            in_channels = out_channels

            # The inverted residual layer
            layer = InvertedResidual(
                in_channels,
                out_channels,
                # The stride of each of the rest layers is 1
                stride=1,
                expansion_ratio=expansion_ratio,
            )

            # Add the layer
            layers.append(layer)

        # Wrap all the layers
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        return self.conv(x)
