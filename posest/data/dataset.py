from collections import namedtuple
from pathlib import Path
from PIL import Image
import json
import h5py
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from .transforms import transform_image_to_tensor


Human36mSample = namedtuple(
    "Human36mSample",
    (
        "image",
        "keypoints",
    ),
)


class Human36mDataset(Dataset):

    # Lower bound of the original (x, y, z) coordinate
    coord_bound_min = np.array([-1922.66037218, -1856.25528127, 2047.20152873])

    # Upper bound of the original (x, y, z) coordinate
    coord_bound_max = np.array([2198.2496672, 963.7077495, 8015.18267695])

    def __init__(
        self,
        dataset_dir: Path,
        *,
        sample_names: list[str],
    ) -> None:

        super().__init__()

        self._dataset_dir = dataset_dir

        # Annotations
        self._annotations_dir = dataset_dir.joinpath("annot")

        # This file maps each training sample name to its index
        self._train_sample_names_to_index_filepath = self._annotations_dir.joinpath(
            "train-sample-name-to-index.json"
        )

        # This file contains the annotations for the training
        self._train_annotations_filepath = self._annotations_dir.joinpath("train.h5")

        # This file maps each validation sample name to its index
        self._valid_sample_names_to_index_filepath = self._annotations_dir.joinpath(
            "valid-sample-name-to-index.json"
        )

        # This file contains the annotations for the validation
        self._valid_annotations_filepath = self._annotations_dir.joinpath("valid.h5")

        # Images
        self._images_dir = dataset_dir.joinpath("images")

        # Sample names
        self._sample_names = sample_names

        # Get the sample-name-to-index dictionary
        self._sample_name_to_index = self._get_sample_name_to_index()

        # Get all keypoint coordinates
        self._keypoints = self._get_keypoints()

    def __getitem__(self, index: int) -> Human36mSample:

        # Get sample name
        sample_name = self._sample_names[index]

        # Load image
        image_filepath = self._images_dir.joinpath(f"{sample_name}.jpg")
        image = Image.open(image_filepath)
        image_tensor = transform_image_to_tensor(image)

        # Get the index of the sample in annotations
        sample_annot_index = self._sample_name_to_index[sample_name]

        # Get keypoint coordinates
        keypoints = self._keypoints[sample_annot_index]

        # Create the sample
        sample = Human36mSample(
            image=image_tensor,
            keypoints=keypoints,
        )

        return sample

    def __len__(self) -> int:

        return len(self._sample_names)

    @property
    def train(self) -> bool:
        """`True` means that the samples are for training.
        Otherwise they are for validation.
        """

        return self._train

    def _get_sample_name_to_index(self) -> dict:
        """Get the training or validation sample-name-to-index dictionary.
        It will determine which one to use based on the first sample name.

        Returns
        -------
        dict
            The sample-name-to-index dictionary.
        """

        sample_name_to_index = {}

        # Read the sample name to index file
        with open(self._train_sample_names_to_index_filepath, "r") as f:
            train_sample_name_to_index: dict = json.load(f)
        with open(self._valid_sample_names_to_index_filepath, "r") as f:
            valid_sample_name_to_index: dict = json.load(f)

        # Get the first sample
        sample_name = self._sample_names[0]
        if sample_name in train_sample_name_to_index.keys():
            sample_name_to_index = train_sample_name_to_index

            # A flag to indicate training
            self._train = True
        elif sample_name in valid_sample_name_to_index.keys():
            sample_name_to_index = valid_sample_name_to_index

            # Set the flag
            self._train = False
        else:
            raise ValueError(f"Unknown sample name: {sample_name}")

        return sample_name_to_index

    def _get_keypoints(self) -> Tensor:
        """Get the coordinates of all keypoints for the training or validation samples.
        The x, y and z components of the coordinates are normalized to [0, 1].
        It will determine which one to use based on the first sample name.

        Returns
        -------
        Tensor
            The coordinates of all keypoints.
        """

        # Get file path to the annotations
        if self.train:
            filepath = self._train_annotations_filepath
        else:
            filepath = self._valid_annotations_filepath

        # Load the coordinates of all keypoints
        with h5py.File(filepath, "r") as f:
            keypoints = f["S"][:]

        # Normalize
        keypoints = (keypoints - Human36mDataset.coord_bound_min) / (
            Human36mDataset.coord_bound_max - Human36mDataset.coord_bound_min
        )

        # Convert to tensor
        keypoints = torch.tensor(keypoints, dtype=torch.float)

        # Further clip the coordinates to [0, 1] in case of some outliers
        keypoints = torch.clip(keypoints, 0.0, 1.0)

        return keypoints
