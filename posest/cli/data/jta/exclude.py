from typing import Annotated, Optional
from pathlib import Path
import json
import typer
from rich.progress import track

from .app import jta_app


@jta_app.command(
    name="exclude",
    help="Exclude some samples from the dataset.",
)
def exclude_samples(
    dataset_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="The root directory of the JTA dataset.",
        ),
    ]
):
    pass
