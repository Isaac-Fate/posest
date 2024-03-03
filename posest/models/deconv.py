from torch import Tensor
from torch import nn


class DeConv(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        hidden_dim: int,
    ) -> None:

        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden_dim,
                kernel_size=1,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.PReLU(hidden_dim),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                out_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: Tensor) -> Tensor:
        """Forward propagation.

        Parameters
        ----------
        x : Tensor
            Shape: (N, C, H, W)

        Returns
        -------
        Tensor
            Shape: (N, C_out, 2H, 2W)
        """

        # Input x has shape (N, C, H, W)

        # (N, C_hidden, H, W)
        x = self.conv1(x)

        # (N, C_out, H, W)
        x = self.conv2(x)

        # (N, C_out, 2H, 2W)
        x = self.upsampling(x)

        return x
