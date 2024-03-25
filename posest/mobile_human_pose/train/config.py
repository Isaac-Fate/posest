from typing import Optional
from pathlib import Path
from pydantic import BaseModel

from ..models import MobileHumanPoseConfig


class TrainingConfig(BaseModel):

    # Root directory of the run,
    # which stores the log file and all checkpoints
    run_dir: Path

    # Root directory of the dataset
    dataset_dir: Path

    # Model configuration
    mobile_human_pose_config: MobileHumanPoseConfig

    # Batch sizes
    train_batch_size: int
    val_batch_size: int

    # Save a checkpoint every n epochs
    # The checkpoint of the last epochs is always saved
    save_every_n_epochs: int

    # Number of epochs to train
    num_epochs: int

    # Learning rate of the optimizer
    adam_lr: float

    # Exponential learning rate scheduler
    exponential_lr_gamma: float

    # Whether to enable wandb
    wandb: bool = False

    # Random seed
    seed: Optional[int] = None

    @property
    def run_name(self) -> str:
        """The name of the run is given by
        the stem of the run directory.
        """

        return self.run_dir.stem

    @property
    def log_filepath(self) -> Path:
        """File path of the training log."""

        return self.run_dir.joinpath("train").with_suffix(".log")

    @property
    def checkpoints_dir(self) -> Path:
        """Directory storing all checkpoints."""

        path = self.run_dir.joinpath("checkpoints")

        # Create the dir if it does not exist
        if not path.is_dir():
            path.mkdir()

        return path
