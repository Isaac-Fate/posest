from PIL import Image
from torch import Tensor
from torchvision import transforms

from ..models import MobileHumanPoseSpec


def transform_image_to_tensor(image: Image.Image) -> Tensor:
    """Transform the image to a tensor that can be forwarded to the table net.

    Parameters
    ----------
    image : Image.Image
        Image.

    Returns
    -------
    Tensor
        Tensor to forward to the table net.
    """

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    MobileHumanPoseSpec.input_image_width,
                    MobileHumanPoseSpec.input_image_height,
                )
            ),
            transforms.ToTensor(),
        ]
    )

    return transform(image)
