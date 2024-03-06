from pathlib import Path
from pydantic import BaseModel
import json


class TrainingConfig(BaseModel):

    # Root directory of the run,
    # which stores the log file and all checkpoints
    run_dir: Path

    # Root directory of the dataset
    dataset_dir: Path

    # Batch sizes
    train_batch_size: int
    valid_batch_size: int

    # Save a checkpoint every n epochs
    # The checkpoint of the last epochs is always saved
    save_every_n_epochs: int

    # Hyperparameters
    n_epochs: int
    adam_lr: float
    exponential_lr_gamma: float

    # Whether to enable wandb
    wandb: bool = False

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
    def train_sample_names(self) -> list[str]:
        """Names of the training samples."""

        with open(self.run_dir.joinpath("train-samples.json"), "r") as f:
            sample_names = json.load(f)

        return sample_names

    @property
    def valid_sample_names(self) -> list[str]:
        """Names of the validation samples."""

        with open(self.run_dir.joinpath("valid-samples.json"), "r") as f:
            sample_names = json.load(f)

        return sample_names

    @property
    def checkpoints_dir(self) -> Path:
        """Directory storing all checkpoints."""

        path = self.run_dir.joinpath("checkpoints")

        # Create the dir if it does not exist
        if not path.is_dir():
            path.mkdir()

        return path
