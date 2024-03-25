from typing import Self, Optional
import tomllib
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

import wandb
from loguru import logger

from .config import TrainingConfig
from .logging import BatchTrainingLogRecord, EpochTrainingLogRecord, ValidationLogRecord
from posest.data import DataSubsetType
from ..data import JTADataset, JTASample
from ..models import MobileHumanPose
from .criterions import mpjpe_loss


PROJECT_NAME = "mobile-human-pose"


class Trainer:

    def __init__(self, config: TrainingConfig) -> None:

        self._config = config

        # Set up logger
        self._logger = logger.bind(key=PROJECT_NAME)
        self._logger.add(self._config.log_filepath)

        # Check if CUDA (GPU) is available, else use CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Data
        self._dataset: Optional[JTADataset] = None
        self._train_data_loader: Optional[DataLoader] = None
        self._val_data_loader: Optional[DataLoader] = None

        # Create the model, and
        # move the model to device
        self._model = MobileHumanPose(config.mobile_human_pose_config).to(self.device)

    @property
    def config(self) -> TrainingConfig:
        """Training configuration."""

        return self._config

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

        # Set the random seed manually if required
        if self._config.seed is not None:
            torch.manual_seed(self._config.seed)

        if self._config.wandb:

            # Log in to Wandb
            wandb.login()

            # Create a run
            run = wandb.init(
                # Project name
                project=PROJECT_NAME,
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

        # Scheduler
        # It is used to decrease the learning rate
        scheduler = ExponentialLR(
            optimizer,
            gamma=self._config.exponential_lr_gamma,
        )

        # Log
        self._logger.info("Start training...")
        self._logger.info(f"Training Configuration: {self._config}")

        # Training loop
        for i in range(self._config.num_epochs):

            # Set the model to training mode
            self.model.train()

            # Epoch number
            epoch = i + 1

            # Training losse of each batch
            train_losses = []

            # Total number of batches
            num_batches = len(self._train_data_loader)

            batch: JTASample
            for i, batch in enumerate(self._train_data_loader):

                # Batch number
                batch_number = i + 1

                # Zero the gradients
                optimizer.zero_grad()

                # Forward this batch and get the loss
                loss = self._forward_batch(batch)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()

                # Add loss
                train_losses.append(loss.item())

                # Batch training log

                # Log record
                train_log_record = BatchTrainingLogRecord(
                    epoch=epoch,
                    batch_number=batch_number,
                    num_batches=num_batches,
                    train_loss=loss.item(),
                )

                # Loguru
                self._logger.info(train_log_record.to_message())

                # Wandb
                if self._config.wandb:
                    wandb.log(train_log_record.model_dump())

            # Epoch training log

            # Log record
            train_log_record = EpochTrainingLogRecord(
                epoch=epoch,
                lr=scheduler.get_last_lr()[0],
                avg_train_loss=sum(train_losses) / num_batches,
            )

            # Loguru
            self._logger.info(train_log_record.to_message())

            # Wandb
            if self._config.wandb:
                wandb.log(train_log_record.model_dump())

            # Validate
            self._validate(epoch)

            # Adjust learning rate
            scheduler.step()

            # Save checkpoint
            self._save_checkpoint(epoch)

    def _prepare_data(self) -> None:

        # Load datasets
        train_dataset = JTADataset(
            self._config.dataset_dir,
            type=DataSubsetType.TRAIN,
            image_tensor_height=self._config.mobile_human_pose_config.image_tensor_height,
            image_tensor_width=self._config.mobile_human_pose_config.image_tensor_width,
        )
        val_dataset = JTADataset(
            self._config.dataset_dir,
            type=DataSubsetType.VAL,
            image_tensor_height=self._config.mobile_human_pose_config.image_tensor_height,
            image_tensor_width=self._config.mobile_human_pose_config.image_tensor_width,
        )

        # Create data loaders
        train_data_loader = DataLoader(
            train_dataset,
            batch_size=self._config.train_batch_size,
            shuffle=True,
        )
        val_data_loader = DataLoader(
            val_dataset,
            batch_size=self._config.val_batch_size,
            shuffle=True,
        )

        self._train_data_loader = train_data_loader
        self._val_data_loader = val_data_loader

    def _validate(self, epoch: int) -> None:
        # Set the model to evaluation mode
        self.model.eval()

        with torch.no_grad():
            # List of loss of each batch
            val_losses = []

            # Validation loop
            num_batches = len(self._val_data_loader)
            batch: JTASample
            for batch in self._val_data_loader:
                # Forward this batch and get the loss
                loss = self._forward_batch(batch)

                # Add to list
                val_losses.append(loss.item())

            # Overall validation performance
            val_loss = sum(val_losses) / num_batches

            # Log

            # Log record
            val_log_record = ValidationLogRecord(
                epoch=epoch,
                val_loss=val_loss,
            )

            # Loguru
            logger.info(val_log_record.to_message())

            # Wandb
            if self._config.wandb:
                wandb.log(val_log_record.model_dump())

    def _forward_batch(self, batch: JTASample) -> Tensor:
        """Feed forward a batch, and return the loss.

        Parameters
        ----------
        batch: JTASample
            Batch of samples.

        Returns
        -------
        Tensor
            Loss of this batch.
        """

        # Unpack batched samples, and
        # move data to device
        image = batch.image.to(self.device)
        keypoint_world_coordinates = batch.keypoint_world_coordinates.to(self.device)

        # Forward pass
        pred_keypoint_world_coordinates = self._model(image)

        # Compute loss
        loss = mpjpe_loss(pred_keypoint_world_coordinates, keypoint_world_coordinates)

        return loss

    def _save_checkpoint(self, epoch: int) -> None:

        # No need to save
        if (
            epoch % self._config.save_every_n_epochs != 0
            and epoch != self._config.num_epochs
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
