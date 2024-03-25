from PIL import Image
from torch import Tensor
from torchvision import transforms

from posest.data.jta import BBox


class ImageConverter:

    def __init__(
        self,
        *,
        image_tensor_height: int,
        image_tensor_width: int,
    ) -> None:

        self._image_tensor_height = image_tensor_height
        self._image_tensor_width = image_tensor_width

    def __call__(
        self,
        image: Image.Image,
        *,
        bbox: BBox,
    ) -> Tensor:

        return self.convert_to_tensor(image, bbox=bbox)

    def crop(
        self,
        image: Image.Image,
        *,
        bbox: BBox,
    ) -> Image.Image:

        # Convert image to tensor
        tensor = transforms.functional.to_tensor(image)

        # Crop the image by slicing the tensor
        # In this way, no black padding will be introduced
        tensor = tensor[..., bbox.y_min : bbox.y_max, bbox.x_min : bbox.x_max]

        # Convert back to image
        cropped_image = transforms.functional.to_pil_image(tensor)

        return cropped_image

    def resize(self, image: Image.Image) -> Image.Image:

        resized_image = transforms.functional.resize(
            image,
            (
                self._image_tensor_height,
                self._image_tensor_width,
            ),
        )

        return resized_image

    def convert_to_tensor(
        self,
        image: Image.Image,
        *,
        bbox: BBox,
    ) -> Tensor:

        # Crop the image
        image = self.crop(image, bbox=bbox)

        # Resize the image
        image = self.resize(image)

        # Convert to tensor
        tensor = transforms.functional.to_tensor(image)

        return tensor
