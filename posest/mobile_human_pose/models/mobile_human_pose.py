import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from .config import MobileHumanPoseConfig
from .backbone import SkipConcat


class MobileHumanPose(nn.Module):

    def __init__(
        self,
        config: MobileHumanPoseConfig,
    ) -> None:

        super().__init__()

        self._config = config

        # Backbone
        self.backbone = SkipConcat()

        # Final layer
        self.conv = nn.Conv2d(
            256,
            config.num_keypoints * 32,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @property
    def config(self) -> MobileHumanPoseConfig:
        """The model configuration."""

        return self._config

    def forward(self, image: Tensor) -> Tensor:
        """Predict the coordinates of the keypoints from the input image.

        Parameters
        ----------
        image : Tensor
            Shape: (N, 3, 256, 256)

        Returns
        -------
        Tensor
            Shape: (N, K, 3)

            K is the number of keypoints.
            It represents the (x, y, z) coordinate for
            each keypoint in each batch.
        """

        # Forward pass
        x = self.backbone(image)
        out = self.conv(x)

        # Cet heatmaps
        heatmaps = self.get_heatmaps(out)

        # Get the keypoint coordinates
        keypoints = self.get_keypoints(heatmaps)

        return keypoints

    def get_heatmaps(self, out: Tensor) -> Tensor:
        """Get the volumetric heatmaps from the output tensor.

        Parameters
        ----------
        out : Tensor
            Shape: (N, C, H, W)

            N is the batch size.
            The height and width are equal H = W = D.
            The number of ouput channels satisfies C = K * D
            where K is the number of keypoints.

        Returns
        -------
        Tensor
            Shape: (N, K, D, D, D)
        """

        # Check shape
        n_batches, c, h, w = out.shape
        assert h == w, "Height and width of the output must be the same"
        d = h

        # Get the number of keypoints
        num_keypoints, r = divmod(c, d)
        assert r == 0, "Number of ouput channels must be a multiple of height/width"

        # Reshape to (N, K, D^3)
        # so that we can easily apply softmax
        heatmaps = out.reshape(-1, num_keypoints, d * d * d)

        # Apply softmax
        heatmaps = F.softmax(heatmaps, dim=-1)

        # Convert to the desired shape of heatamps
        heatmaps = heatmaps.reshape(-1, num_keypoints, d, d, d)

        return heatmaps

    def get_keypoints(self, heatmaps: Tensor) -> Tensor:
        """Extract the x, y and z coordinates of the keypoints from the input heatmaps.

        Parameters
        ----------
        heatmaps : Tensor
            Shape: (N, K, D, D, D)

            where N is the batch size,
            K is the number of keypoints, and
            (D, D, D) is the dimension of each volumetric heatmap.

        Returns
        -------
        Tensor
            Shape: (N, K, 3)

            It represents the (x, y, z) coordinate for
            each keypoint in each batch.
        """

        # Get device
        device = heatmaps.device

        # Get D
        d = heatmaps.shape[-1]

        # Discrete values in one dimension
        x_values = torch.linspace(
            self._config.x_lim[0],
            self._config.x_lim[1],
            d,
        ).to(device)

        y_values = torch.linspace(
            self._config.y_lim[0],
            self._config.y_lim[1],
            d,
        ).to(device)

        z_values = torch.linspace(
            self._config.z_lim[0],
            self._config.z_lim[1],
            d,
        ).to(device)

        # Marginal probabilities
        x_probs = heatmaps.sum(dim=(-1, -2))
        y_probs = heatmaps.sum(dim=(-1, -3))
        z_probs = heatmaps.sum(dim=(-2, -3))

        # Expected coordinate values
        x = torch.einsum("ijk, k -> ij", x_probs, x_values).to(device)
        y = torch.einsum("ijk, k -> ij", y_probs, y_values).to(device)
        z = torch.einsum("ijk, k -> ij", z_probs, z_values).to(device)

        # Wrap the components
        coords = torch.stack((x, y, z), dim=-1)

        return coords
