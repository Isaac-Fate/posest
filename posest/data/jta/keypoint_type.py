from typing import Self
from enum import IntEnum


class KeypointType(IntEnum):

    HEAD_TOP = 0
    HEAD_CENTER = 1
    NECK = 2

    RIGHT_CLAVICLE = 3
    RIGHT_SHOULDER = 4
    RIGHT_ELBOW = 5
    RIGHT_WRIST = 6

    LEFT_CLAVICLE = 7
    LEFT_SHOULDER = 8
    LEFT_ELBOW = 9
    LEFT_WRIST = 10

    SPINE4 = 11
    SPINE3 = 12
    SPINE2 = 13
    SPINE1 = 14
    HIP = 15

    RIGHT_HIP = 16
    RIGHT_KNEE = 17
    RIGHT_ANKLE = 18

    LEFT_HIP = 19
    LEFT_KNEE = 20
    LEFT_ANKLE = 21

    @classmethod
    def all(cls) -> tuple[Self]:
        """Get all the keypoint types."""

        return tuple(cls._member_map_.values())

    @classmethod
    def num(cls) -> int:
        """Get the number of keypoint types."""

        return len(cls.all())
