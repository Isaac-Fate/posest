from typing import Self, Optional
import tomllib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

import wandb
from loguru import logger

from .config import TrainingConfig
from .logging import TrainingLogRecord, ValidationLogRecord
from ..data import Human36mDataset, Human36mSample
from ..models import MobileHumanPose
from .criterions import mpjpe_loss


class Trainer:

    def __init__(self, config: TrainingConfig) -> None:

        self._config = config

        # Set up logger
        self._logger = logger.bind(key="table-net")
        self._logger.add(self._config.log_filepath)

        # Check if CUDA (GPU) is available, else use CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self._dataset: Optional[Human36mDataset] = None
        self._train_dataloader: Optional[DataLoader] = None
        self._valid_dataloader: Optional[DataLoader] = None

        # Create table net model
        self._model = MobileHumanPose()

        # Move the model to device
        self._model.to(self._device)

    @property
    def device(self) -> str:

        return self._device

    @property
    def model(self) -> MobileHumanPose:

        return self._model

    @classmethod
    def from_config_file(cls, filepath: Path | str) -> Self:

        # Get the run dir
        run_dir = filepath.parent

        # Read the configuration file
        with open(filepath, "rb") as f:
            data = tomllib.load(f)

        # Set run dir
        data["run_dir"] = run_dir

        # Convert to TrainingConfig
        config = TrainingConfig.model_validate(data)

        return cls(config)

    def train(self):

        if self._config.wandb:

            # Log in to Wandb
            wandb.login()

            # Create a run
            run = wandb.init(
                # Project name
                project="mobile-human-pose",
                # Run name
                name=self._config.run_name,
            )

        # Prepare data
        self._prepare_data()

        # Optimizer
        optimizer = Adam(
            self._model.parameters(),
            lr=self._config.adam_lr,
        )

        # Log
        self._logger.info("Start training...")
        self._logger.info(f"Training Configuration: {self._config}")

        # Training loop
        for i in range(self._config.n_epochs):

            # Epoch number
            epoch = i + 1

            # Total number of batches
            n_batches = len(self._train_dataloader)

            samples: Human36mSample
            for i, samples in enumerate(self._train_dataloader):

                # Batch number
                batch = i + 1

                # Zero the gradients
                optimizer.zero_grad()

                # Unpack batched samples, and
                # move data to device
                image_batch = samples.image.to(self._device)
                keypoints_batch = samples.keypoints.to(self._device)

                # Forward pass
                pred_keypoints = self._model(image_batch)

                # Compute loss
                loss = mpjpe_loss(pred_keypoints, keypoints_batch)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                # Log

                # Log record
                train_log_record = TrainingLogRecord(
                    epoch=epoch,
                    batch=batch,
                    n_batches=n_batches,
                    train_loss=loss.item(),
                )

                # Loguru
                self._logger.info(train_log_record.to_message())

                # Wandb
                if self._config.wandb:
                    wandb.log(train_log_record.model_dump())

            # Validate
            self._validate(epoch)

            # Save checkpoint
            self._save_checkpoint(epoch)

    def _prepare_data(self) -> None:

        # Load datasets
        train_dataset = Human36mDataset(
            self._config.dataset_dir,
            sample_names=self._config.train_sample_names,
        )
        valid_dataset = Human36mDataset(
            self._config.dataset_dir,
            sample_names=self._config.valid_sample_names,
        )

        # Create data loaders
        train_dataloader = DataLoader(train_dataset, batch_size=self._config.batch_size)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self._config.batch_size)

        self._train_dataloader = train_dataloader
        self._valid_dataloader = valid_dataloader

    def _validate(self, epoch: int) -> None:
        with torch.no_grad():
            # List of loss of each batch
            valid_losses = []

            # Validation loop
            n_batches = len(self._valid_dataloader)
            samples: Human36mSample
            for samples in self._valid_dataloader:
                # Unpack batched samples
                image_batch = samples.image.to(self._device)
                keypoints_batch = samples.keypoints.to(self._device)

                # Predict keypoints
                pred_keypoints = self._model(image_batch)

                # Compute loss
                loss = mpjpe_loss(pred_keypoints, keypoints_batch)

                # Add to list
                valid_losses.append(loss.item())

            # Overall validation performance
            valid_loss = sum(valid_losses) / n_batches

            # Log

            # Log record
            valid_log_record = ValidationLogRecord(
                epoch=epoch,
                valid_loss=valid_loss,
            )

            # Loguru
            logger.info(valid_log_record.to_message())

            # Wandb
            if self._config.wandb:
                wandb.log(valid_log_record.model_dump())

    def _save_checkpoint(self, epoch: int) -> None:

        # No need to save
        if (
            epoch % self._config.save_every_n_epochs != 0
            and epoch != self._config.n_epochs
        ):
            return

        # State dict
        state_dict = self._model.state_dict()

        # File path
        checkpoint_filepath = self._config.checkpoints_dir.joinpath(
            f"checkpoint-{epoch}"
        ).with_suffix(".pt")

        # Save
        torch.save(state_dict, checkpoint_filepath)

        # Log
        self._logger.info(f"Checkpoint is saved at: {checkpoint_filepath}")
