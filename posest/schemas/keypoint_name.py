from typing import Self
from enum import StrEnum


class KeypointName(StrEnum):

    HEAD = "head"
    NOSE = "nose"
    NECK = "neck"

    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_HAND = "left_hand"
    RIGHT_HAND = "right_hand"

    SPINE = "spine"
    ROOT = "root"

    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"

    @classmethod
    def from_index(cls, index: int) -> Self:
        match index:
            case 10:
                return cls.HEAD
            case 9:
                return cls.NOSE
            case 8:
                return cls.NECK

            case 11:
                return cls.LEFT_SHOULDER
            case 12:
                return cls.LEFT_ELBOW
            case 13:
                return cls.LEFT_HAND

            case 14:
                return cls.RIGHT_SHOULDER
            case 15:
                return cls.RIGHT_ELBOW
            case 16:
                return cls.RIGHT_HAND

            case 7:
                return cls.SPINE
            case 0:
                return cls.ROOT

            case 1:
                return cls.LEFT_HIP
            case 2:
                return cls.LEFT_KNEE
            case 3:
                return cls.LEFT_ANKLE

            case 4:
                return cls.RIGHT_HIP
            case 5:
                return cls.RIGHT_KNEE
            case 6:
                return cls.RIGHT_ANKLE

            case _:
                raise ValueError("Unkown index")

    @classmethod
    def n(cls) -> int:
        """Total number of keypoints."""

        return len(KeypointName._member_names_)

    @property
    def index(self) -> int:
        """Associated index of this keypoint in the array of keypoint coordinates,
        which has the shape (n_keypoints, 3).
        """

        match self:
            case KeypointName.HEAD:
                return 10
            case KeypointName.NOSE:
                return 9
            case KeypointName.NECK:
                return 8

            case KeypointName.LEFT_SHOULDER:
                return 11
            case KeypointName.LEFT_ELBOW:
                return 12
            case KeypointName.LEFT_HAND:
                return 13

            case KeypointName.RIGHT_SHOULDER:
                return 14
            case KeypointName.RIGHT_ELBOW:
                return 15
            case KeypointName.RIGHT_HAND:
                return 16

            case KeypointName.SPINE:
                return 7
            case KeypointName.ROOT:
                return 0

            case KeypointName.LEFT_HIP:
                return 1
            case KeypointName.LEFT_KNEE:
                return 2
            case KeypointName.LEFT_ANKLE:
                return 3

            case KeypointName.RIGHT_HIP:
                return 4
            case KeypointName.RIGHT_KNEE:
                return 5
            case KeypointName.RIGHT_ANKLE:
                return 6
