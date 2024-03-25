from typing import Self
from enum import StrEnum


class DataSubsetType(StrEnum):

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def all(cls) -> tuple[Self]:
        """Get all the data subset types."""

        return tuple(cls._member_map_.values())
