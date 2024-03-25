from typing import Annotated
from pathlib import Path
import typer

from .app import app


@app.command(help="Train the Mobile Human Pose model.")
def train(
    file: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="Training configuration file (in the format of TOML).",
        ),
    ]
):
    from ..train import Trainer

    # Create a trainer
    trainer = Trainer.from_config_file(file)

    # Train!
    trainer.train()
