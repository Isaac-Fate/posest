import torch
from torch import Tensor
from torch import nn

from . import ConvBNReLU, InvertedResidualBlock, DeConv


class SkipConcat(nn.Module):

    def __init__(
        self,
        *,
        embedding_dim: int = 2048,
    ) -> None:

        super().__init__()

        # First conv layer
        self.conv1 = ConvBNReLU(
            3,
            48,
            kernel_size=3,
            stride=2,
        )

        # Invert residual blocks

        self.inv_res_block1 = InvertedResidualBlock(
            48,
            64,
            stride=2,
            expansion_ratio=1.0,
            n_layers=1,
        )

        self.inv_res_block2 = InvertedResidualBlock(
            64,
            48,
            stride=2,
            expansion_ratio=6.0,
            n_layers=2,
        )

        self.inv_res_block3 = InvertedResidualBlock(
            48,
            48,
            stride=2,
            expansion_ratio=6.0,
            n_layers=3,
        )

        self.inv_res_block4 = InvertedResidualBlock(
            48,
            64,
            stride=2,
            expansion_ratio=6.0,
            n_layers=4,
        )

        self.inv_res_block5 = InvertedResidualBlock(
            64,
            96,
            stride=2,
            expansion_ratio=6.0,
            n_layers=3,
        )

        self.inv_res_block6 = InvertedResidualBlock(
            96,
            160,
            stride=1,
            expansion_ratio=6.0,
            n_layers=3,
        )

        self.inv_res_block7 = InvertedResidualBlock(
            160,
            320,
            stride=1,
            expansion_ratio=6.0,
            n_layers=1,
        )

        self.conv2 = ConvBNReLU(
            320,
            embedding_dim,
            kernel_size=1,
        )

        # Deconvolutional layers

        self.deconv1 = DeConv(
            embedding_dim + 96,
            256,
            # 96 is the output dim of res block5
            hidden_dim=96,
        )

        self.deconv2 = DeConv(
            256 + 64,
            256,
            # 64 is the output dim of res block4
            hidden_dim=64,
        )

        self.deconv3 = DeConv(
            256 + 48,
            256,
            # 48 is the output dim of res block3
            hidden_dim=48,
        )

    def forward(self, image: Tensor) -> Tensor:

        # Input image has shape (N, 3, 256, 256)

        x1 = self.conv1(image)

        u1 = self.inv_res_block1(x1)
        u2 = self.inv_res_block2(u1)
        u3 = self.inv_res_block3(u2)
        u4 = self.inv_res_block4(u3)
        u5 = self.inv_res_block5(u4)
        u6 = self.inv_res_block6(u5)
        u7 = self.inv_res_block7(u6)

        x2 = self.conv2(u7)

        concat1 = torch.concat((u5, x2), dim=1)
        z1 = self.deconv1(concat1)

        concat2 = torch.concat((u4, z1), dim=1)
        z2 = self.deconv2(concat2)

        concat3 = torch.concat((u3, z2), dim=1)
        out = self.deconv3(concat3)

        return out
