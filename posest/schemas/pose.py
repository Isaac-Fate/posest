from collections import namedtuple
import numpy as np

from .keypoint_name import KeypointName


Coordinate = namedtuple("Coordinate", ("x", "y", "z"))


class Pose:

    def __init__(
        self,
        keypoint_coords: np.ndarray,
    ) -> None:

        self._keypoint_coords = keypoint_coords

    def __getitem__(self, keypoint_name: KeypointName) -> Coordinate:

        # Get the index of the keypoint
        index = keypoint_name.index

        return Coordinate(
            x=self._keypoint_coords[index, 0],
            y=self._keypoint_coords[index, 1],
            z=self._keypoint_coords[index, 2],
        )
