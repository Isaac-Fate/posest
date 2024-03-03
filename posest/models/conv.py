from torch import Tensor
from torch import nn


class ConvBNReLU(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:

        super().__init__()

        # Ensure that the input kernel size is odd
        assert kernel_size % 2 == 1, "Kernel size must be an odd number"

        # By adding this padding,
        # we will keep the height and width of the input unchanged
        padding = (kernel_size - 1) // 2

        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
        )

        # Batch norm layer
        self.bn = nn.BatchNorm2d(out_channels)

        # Activation layer
        self.activation = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:

        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x
