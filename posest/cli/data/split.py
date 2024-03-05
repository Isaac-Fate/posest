from typing import Annotated, Optional
from pathlib import Path
import typer
import json

from .app import data_app


@data_app.command(
    help="Split the dataset into training and validation datasets. Generate JSON files of associated sample names under the run directory.",
)
def split(
    run_dir: Path = typer.Argument(
        show_default=False,
        help="The directory containing the configuration, logs, and results of this single run.",
    ),
    dataset_dir: Path = typer.Argument(
        show_default=False,
        help="The path to the Human3.6M data directory.",
    ),
    n_train_samples: Annotated[
        int,
        typer.Option(
            "--train",
            help="Number of training samples.",
        ),
    ] = 1000,
    n_valid_samples: Annotated[
        int,
        typer.Option(
            "--valid",
            help="Number of validation samples.",
        ),
    ] = 100,
    random_seed: Annotated[
        Optional[int],
        typer.Option(
            "--seed",
            help="Random seed.",
        ),
    ] = None,
):

    # Create run directory if it does not exist
    run_dir.mkdir(parents=True, exist_ok=True)

    # Annotations directory
    annotations_dir = dataset_dir.joinpath("annot")

    # Sample names for training
    train_sample_to_index_filepath = annotations_dir.joinpath(
        "train-sample-name-to-index.json"
    )
    with open(train_sample_to_index_filepath, "r") as f:
        train_sample_name_to_index: dict = json.load(f)
    train_sample_names = list(train_sample_name_to_index.keys())

    # Sample names for validation
    valid_sample_to_index_filepath = annotations_dir.joinpath(
        "valid-sample-name-to-index.json"
    )
    with open(valid_sample_to_index_filepath, "r") as f:
        valid_sample_name_to_index: dict = json.load(f)
    valid_sample_names = list(valid_sample_name_to_index.keys())

    # Import NumPy
    import numpy as np

    # Random generator
    rng = np.random.default_rng(random_seed)

    # Select training samples

    train_sample_indices = rng.choice(
        len(train_sample_names),
        size=n_train_samples,
        replace=False,
    )

    train_sample_names = [train_sample_names[i] for i in train_sample_indices]

    # Select validation samples

    valid_sample_indices = rng.choice(
        len(valid_sample_names),
        size=n_valid_samples,
        replace=False,
    )

    valid_sample_names = [valid_sample_names[i] for i in valid_sample_indices]

    # Write to files

    with open(run_dir.joinpath("train-samples.json"), "w") as f:
        json.dump(train_sample_names, f, indent=4)

    with open(run_dir.joinpath("valid-samples.json"), "w") as f:
        json.dump(valid_sample_names, f, indent=4)
