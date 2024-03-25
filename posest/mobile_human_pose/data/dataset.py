from enum import StrEnum
from collections import namedtuple
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from posest.data import DataSubsetType
from posest.data.jta import Pose
from .image_converter import ImageConverter


ANNOTATIONS_DIR_NAME = "custom-annotations"


JTASample = namedtuple(
    "JTASample",
    (
        "image",
        "keypoint_world_coordinates",
    ),
)


class JTADataset(Dataset):

    def __init__(
        self,
        dataset_dir: Path,
        *,
        type: DataSubsetType,
        image_tensor_height: int,
        image_tensor_width: int,
        load_sample_names_from_file: bool = True,
    ) -> None:

        super().__init__()

        self._dataset_dir = dataset_dir
        self._type = type

        self._image_tensor_height = image_tensor_height
        self._image_tensor_width = image_tensor_width

        # Whether to load sample names from TXT file
        self._load_sample_names_from_file = load_sample_names_from_file

        # Create a image converter
        self._image_converter = ImageConverter(
            image_tensor_height=image_tensor_height,
            image_tensor_width=image_tensor_width,
        )

        # The custom annotations insteal of the `annotations` directory  provided by the JTA dataset
        self._annotations_dir = dataset_dir.joinpath(ANNOTATIONS_DIR_NAME)

        # The directory containing the CSV annotations for poses
        self._poses_dir = self._annotations_dir.joinpath("poses")

        # The subdirectory of the root directory of the poses
        self._poses_subdir = self._poses_dir.joinpath(self.type)

        # Sample names
        self._sample_names: list[str] = []
        self._load_sample_names()

        # This contains the frames captured from video sequences
        self._images_dir = dataset_dir.joinpath("frames")

        # Subdirectory of the root directory of the images
        self._images_subdir = self._images_dir.joinpath(self.type)

    @property
    def type(self) -> DataSubsetType:
        """Data subset type. One of "train", "val" or "test"."""

        return self._type

    @property
    def image_converter(self) -> ImageConverter:
        """Image converter."""

        return self._image_converter

    @property
    def annotations_dir(self) -> Path:
        """The custom annotations."""

        return self._annotations_dir

    def __len__(self) -> int:

        return len(self._sample_names)

    def __getitem__(self, index: int) -> JTASample:

        return self.get_sample(
            index,
            image_to_tensor=True,
        )

    def get_pose_path(self, index: int) -> Path:
        """Get the CSV file path of the pose at the given index.

        Parameters
        ----------
        index: int
            Sample index.

        Returns
        -------
        Path
            Pose CSV file path.
        """

        # Get the sample name
        sample_name = self._sample_names[index]

        return self._poses_subdir.joinpath(f"{sample_name}.csv")

    def get_image_path(self, index: int) -> Path:
        """Get the image file path of the image at the given index.

        Parameters
        ----------
        index: int
            Sample index.

        Returns
        -------
        Path
            Image file path.
        """

        # Get the sample name
        sample_name = self._sample_names[index]

        # Extract the sequence number, frame number from the sample name
        seq_number, frame_number, _ = sample_name.split("_")

        return self._images_subdir.joinpath(f"seq_{seq_number}").joinpath(
            f"{frame_number}.jpg"
        )

    def get_pose(self, index: int) -> Pose:
        """Get the `Pose` object at the given index.

        Parameters
        ----------
        index: int
            Sample index.

        Returns
        -------
        Pose
            `Pose` object.
        """

        # Annotation file path
        pose_path = self.get_pose_path(index)

        # Load pose
        pose = Pose.from_csv(pose_path)

        return pose

    def get_sample(
        self,
        index: int,
        image_to_tensor: bool = False,
    ) -> JTASample:

        # Load annotation

        # Load pose
        pose = self.get_pose(index)

        # Get sequence number and frame number
        seq_number = pose.seq_number
        frame_number = pose.frame_number

        # Get the keypoint world coordinates
        keypoint_world_coordinates = pose.keypoint_world_coordinates

        # Find the image path
        image_path = self._images_subdir.joinpath(f"seq_{seq_number}").joinpath(
            f"{frame_number}.jpg"
        )

        # Load the image
        image = Image.open(image_path)

        # Convert the image to a tensor if required
        if image_to_tensor:
            image = self._image_converter.convert_to_tensor(image, bbox=pose.bbox)

        # Just return the cropped the image
        else:
            image = self._image_converter.crop(image, bbox=pose.bbox)

        return JTASample(
            image=image,
            keypoint_world_coordinates=keypoint_world_coordinates,
        )

    def _load_sample_names(self) -> None:

        # The text file consisting of sample names
        sample_names_path = self._annotations_dir.joinpath(
            f"{self.type}-sample-names.txt"
        )

        # Get sample names if this file exsits
        if self._load_sample_names_from_file and sample_names_path.is_file():
            with open(sample_names_path, "r") as f:
                self._sample_names = f.read().splitlines()

        # Load by iterating over the pose subdirectory
        else:
            for path in self._poses_subdir.glob("*.csv"):
                sample_name = path.stem
                self._sample_names.append(sample_name)
