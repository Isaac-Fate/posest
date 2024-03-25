from typing import Annotated, Optional
from pathlib import Path
import json
import typer
from rich.progress import track

from .app import jta_app
from posest.data import DataSubsetType


@jta_app.command(
    name="csn",
    help="Create TXT file(s) containing the sample names of data subset(s).",
)
def exclude_samples(
    dataset_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="The root directory of the JTA dataset.",
        ),
    ],
    data_subset_type: Annotated[
        Optional[DataSubsetType],
        typer.Option(
            "-t",
            "--type",
            help="Data subset type. If it is None, then the program will exclude samples from all data subsets.",
        ),
    ] = None,
) -> None:

    from posest.data.jta import Pose
    from ...data.dataset import ANNOTATIONS_DIR_NAME

    annotations_dir = dataset_dir.joinpath(ANNOTATIONS_DIR_NAME)

    # Root directory of all the poses
    poses_dir = annotations_dir.joinpath("poses")

    # Load excluded sample names from a TXT file
    excluded_sample_names_path = annotations_dir.joinpath("excluded-sample-names.txt")
    with open(excluded_sample_names_path, "r") as f:
        excluded_sample_names = f.read().splitlines()

    # Convert to a set
    excluded_sample_names = set(excluded_sample_names)

    # Get data subset types
    if data_subset_type is None:
        data_subset_types = DataSubsetType.all()
    else:
        data_subset_types = (data_subset_type,)

    for data_subset_type in data_subset_types:

        # Sample names
        sample_names = []

        # Directory of the poses of a data subset
        poses_subdir = poses_dir.joinpath(data_subset_type)

        # Iterate over the pose subdirectory
        for pose_path in poses_subdir.glob("*.csv"):
            sample_name = pose_path.stem
            if sample_name not in excluded_sample_names:
                sample_names.append(sample_name)

        # Write sample names to a TXT file
        file_path = annotations_dir.joinpath(f"{data_subset_type}-sample-names.txt")

        with open(file_path, "w") as f:
            f.writelines(map(lambda sample_name: sample_name + "\n", sample_names))
