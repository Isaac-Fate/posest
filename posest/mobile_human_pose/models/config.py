from pydantic import BaseModel


class MobileHumanPoseConfig(BaseModel):

    image_tensor_height: int
    image_tensor_width: int
    num_keypoints: int
    x_lim: tuple[float, float]
    y_lim: tuple[float, float]
    z_lim: tuple[float, float]
