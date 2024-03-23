from typing import Self
from collections import namedtuple
from pathlib import Path
import numpy as np
import pandas as pd

from .keypoint_type import KeypointType


BBox = namedtuple(
    "BBox",
    (
        "x_min",
        "y_min",
        "x_max",
        "y_max",
    ),
)


class Pose:

    def __init__(
        self,
        *,
        seq_number: int,
        frame_number: int,
        person_id: int,
        keypoint_frame_coordinates: np.ndarray,
        keypoint_world_coordinates: np.ndarray,
        is_keypoint_occluded_flags: np.ndarray,
        is_keypoint_self_occluded_flags: np.ndarray,
        set_hip_as_origin: bool = True,
    ) -> None:

        self._seq_number = seq_number
        self._frame_number = frame_number
        self._person_id = person_id

        # Check array shapes
        assert keypoint_frame_coordinates.shape == (
            KeypointType.num(),
            2,
        ), f"keypoint_frame_coordinates should have shape ({KeypointType.num()}, 2)"
        assert keypoint_world_coordinates.shape == (
            KeypointType.num(),
            3,
        ), f"keypoint_world_coordinates should have shape ({KeypointType.num()}, 3)"
        assert is_keypoint_occluded_flags.shape == (
            KeypointType.num(),
        ), f"is_keypoint_occluded_flags should have shape ({KeypointType.num()},)"
        assert is_keypoint_self_occluded_flags.shape == (
            KeypointType.num(),
        ), f"is_keypoint_self_occluded_flags should have shape ({KeypointType.num()},)"

        self._keypoint_frame_coordinates = keypoint_frame_coordinates
        self._keypoint_world_coordinates = keypoint_world_coordinates
        self._is_keypoint_occluded_flags = is_keypoint_occluded_flags
        self._is_keypoint_self_occluded_flags = is_keypoint_self_occluded_flags

        # Transform the camera coordinates to world coordinates
        # by setting the hip coordinate as the origin if required
        self._set_hip_as_origin = set_hip_as_origin
        if set_hip_as_origin:
            self._keypoint_world_coordinates -= self._keypoint_world_coordinates[
                KeypointType.HIP
            ]

    @property
    def seq_number(self) -> int:
        """Video sequence number."""

        return self._seq_number

    @property
    def frame_number(self) -> int:
        """Frame number of the pose in the video sequence."""

        return self._frame_number

    @property
    def person_id(self) -> int:
        """Person ID."""

        return self._person_id

    @property
    def is_hip_origin(self) -> bool:
        """Whether the hip is the origin of the pose in 3D world coordinates."""

        return self._set_hip_as_origin

    @property
    def keypoint_frame_coordinates(self) -> np.ndarray:
        """Keypoint frame coordinates in the image."""

        return self._keypoint_frame_coordinates

    @property
    def keypoint_world_coordinates(self) -> np.ndarray:
        """Keypoint world coordinates in the 3D world."""

        return self._keypoint_world_coordinates

    @property
    def bbox(self) -> BBox:
        """Bounding box of the pose in the image."""

        return BBox(
            x_min=self._keypoint_frame_coordinates[:, 0].min(),
            y_min=self._keypoint_frame_coordinates[:, 1].min(),
            x_max=self._keypoint_frame_coordinates[:, 0].max(),
            y_max=self._keypoint_frame_coordinates[:, 1].max(),
        )

    @property
    def is_visible(self) -> bool:
        """Whether the pose is visible in the image.
        A pose is visible if at least the head top, head center, and neck are visible.
        """

        keypoint_types_of_interest = (
            KeypointType.HEAD_TOP,
            KeypointType.HEAD_CENTER,
            KeypointType.NECK,
        )

        for keypoint_type in keypoint_types_of_interest:
            if self.is_keypoint_occluded(keypoint_type):
                return False

        return True

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        *,
        set_hip_as_origin: bool = True,
    ) -> Self:
        """Construct a `Pose` object from a CSV file.

        Parameters
        ----------
        path: Path | str
            The path to the CSV file.
            The file name should be in the format of `<seq_number>-<frame_number>-<person_id>.csv`.

        set_hip_as_origin: bool
            Whether to set the hip coordinate as the origin of the pose in 3D world coordinates.
            The default value is True.

        Returns
        -------
        Self
            A `Pose` object.
        """

        # File name without extension
        file_name = Path(path).stem

        # Get metadata of the pose
        seq_number, frame_number, person_id = map(int, file_name.split("-"))

        # Read the CSV file into a data frame
        data_frame = pd.read_csv(path)

        return cls.from_data_frame(
            seq_number=seq_number,
            frame_number=frame_number,
            person_id=person_id,
            data_frame=data_frame,
            set_hip_as_origin=set_hip_as_origin,
        )

    @classmethod
    def from_data_frame(
        cls,
        *,
        seq_number: int,
        frame_number: int,
        person_id: int,
        data_frame: pd.DataFrame,
        set_hip_as_origin: bool = True,
    ) -> Self:
        """Construct a `Pose` object from a `pd.DataFrame`.

        Parameters
        ----------
        seq_number: int
            Video sequence number.

        frame_number: int
            Frame number of the pose in the video sequence.

        person_id: int
            Person ID.

        data_frame: pd.DataFrame
            The columns of the `pd.DataFrame` should be the following:
            - frame_x
            - frame_y
            - world_x
            - world_y
            - world_z
            - occluded
            - self_occluded

        set_hip_as_origin: bool
            Whether to set the hip coordinate as the origin of the pose in 3D world coordinates.
            The default value is True.

        Returns
        -------
        Self
            A `Pose` object.
        """

        return cls(
            seq_number=seq_number,
            frame_number=frame_number,
            person_id=person_id,
            keypoint_frame_coordinates=data_frame.loc[
                :,
                ("frame_x", "frame_y"),
            ].to_numpy(),
            keypoint_world_coordinates=data_frame.loc[
                :, ("world_x", "world_y", "world_z")
            ].to_numpy(),
            is_keypoint_occluded_flags=data_frame.is_occluded.to_numpy(),
            is_keypoint_self_occluded_flags=data_frame.is_self_occluded.to_numpy(),
            set_hip_as_origin=set_hip_as_origin,
        )

    def get_keypoint_frame_coordinate(self, keypoint_type: KeypointType) -> np.ndarray:
        """Get the frame coordinate of the given keypoint type.

        Parameters
        ----------
        keypoint_type: KeypointType
            The type of the keypoint.

        Returns
        -------
        np.ndarray
            The 2D frame coordinate of the keypoint.
        """

        return self._keypoint_frame_coordinates[keypoint_type]

    def get_keypoint_world_coordinate(self, keypoint_type: KeypointType) -> np.ndarray:
        """Get the world coordinate of the given keypoint type.

        Parameters
        ----------
        keypoint_type: KeypointType
            The type of the keypoint.

        Returns
        -------
        np.ndarray
            The 3D world coordinate of the keypoint.
        """

        return self._keypoint_world_coordinates[keypoint_type]

    def is_keypoint_occluded(self, keypoint_type: KeypointType) -> bool:
        """Whether the given keypoint is occluded.

        Parameters
        ----------
        keypoint_type : KeypointType
            The type of the keypoint.

        Returns
        -------
        bool
            True if the keypoint is occluded, False otherwise.
        """

        return self._is_keypoint_occluded_flags[keypoint_type]

    def to_csv(self, parent_dir: Path | str) -> None:
        """Write the pose to a CSV file.

        Parameters
        ----------
        parent_dir: Path | str
            The parent directory of the CSV file to be written.
            The file name should be in the format of `<seq_number>-<frame_number>-<person_id>.csv`.
        """

        # Construct the file name
        file_name = f"{self.seq_number}-{self.frame_number}-{self.person_id}"

        # File path
        path = Path(parent_dir).joinpath(file_name).with_suffix(".csv")

        # Write the CSV file
        self.to_data_frame().to_csv(path, index=False)

    def to_data_frame(self) -> pd.DataFrame:
        """Convert the pose to a `pd.DataFrame`.

        Returns
        -------
        pd.DataFrame
            The columns of the `pd.DataFrame` are the following:
            - frame_x
            - frame_y
            - world_x
            - world_y
            - world_z
            - occluded
            - self_occluded
        """

        return pd.DataFrame(
            {
                "frame_x": self._keypoint_frame_coordinates[:, 0],
                "frame_y": self._keypoint_frame_coordinates[:, 1],
                "world_x": self._keypoint_world_coordinates[:, 0],
                "world_y": self._keypoint_world_coordinates[:, 1],
                "world_z": self._keypoint_world_coordinates[:, 2],
                "is_occluded": self._is_keypoint_occluded_flags,
                "is_self_occluded": self._is_keypoint_self_occluded_flags,
            }
        )
