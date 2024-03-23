from collections import namedtuple
import numpy as np

from .keypoint_name import KeypointName


Coordinate = namedtuple(
    "Coordinate",
    (
        "x",
        "y",
        "z",
    ),
)


class Pose:

    def __init__(
        self,
        keypoints: np.ndarray,
    ) -> None:

        self._keypoints = keypoints

    def __getitem__(self, keypoint_name: KeypointName) -> Coordinate:

        # Get the index of the keypoint
        index = keypoint_name.index

        return Coordinate(
            x=self._keypoints[index, 0],
            y=self._keypoints[index, 1],
            z=self._keypoints[index, 2],
        )
