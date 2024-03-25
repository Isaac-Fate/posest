from typing import Annotated, Optional
from pathlib import Path
import json
import typer
from rich.progress import track

from .app import jta_app
from posest.data import DataSubsetType


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
    ],
    data_subset_type: Annotated[
        Optional[DataSubsetType],
        typer.Option(
            "-t",
            "--type",
            help="Data subset type. If it is None, then the program will exclude samples from all data subsets.",
        ),
    ] = None,
    image_tensor_height: Annotated[
        Optional[int],
        typer.Option(
            "-h",
            "--height",
            help="Height of the image tensor.",
        ),
    ] = 256,
    image_tensor_width: Annotated[
        Optional[int],
        typer.Option(
            "-w",
            "--width",
            help="Width of the image tensor.",
        ),
    ] = 256,
    min_bbox_area: Annotated[
        Optional[float],
        typer.Option(
            "-a",
            "--area",
            help="Minimum bounding box area of the pose samples to be excluded.",
        ),
    ] = 9000,
    output_file_path: Annotated[
        Optional[Path],
        typer.Option(
            "-o",
            "--out",
            help="The path of the output TXT file. If it is None, the file will be saved as `excluded-sample-names.txt` under the annotations directory used by the `JTADataset`.",
        ),
    ] = None,
) -> None:

    from posest.data.jta import Pose
    from ...data.dataset import JTADataset

    # Samples to exclude
    excluded_sample_names = []

    # Get data subset types
    if data_subset_type is None:
        data_subset_types = DataSubsetType.all()
    else:
        data_subset_types = (data_subset_type,)

    for data_subset_type in data_subset_types:

        # Create dataset
        dataset = JTADataset(
            dataset_dir,
            type=data_subset_type,
            image_tensor_height=image_tensor_height,
            image_tensor_width=image_tensor_width,
        )

        # Total number of samples
        num_samples = len(dataset)

        for index in track(
            range(num_samples),
            description=f"Excluding samples from {data_subset_type}...",
        ):
            # Get pose path
            pose_path = dataset.get_pose_path(index)

            # Get pose
            pose = Pose.from_csv(pose_path)

            # Bounding box
            bbox = pose.bbox

            # Calculate the area
            bbox_area = (bbox.x_max - bbox.x_min) * (bbox.y_max - bbox.y_min)

            # Check if the area is too small
            if bbox_area < min_bbox_area:

                # Get sample name
                sample_name = pose_path.stem

                # Add the sample name
                excluded_sample_names.append(sample_name)

    # Write excluded sample names to a TXT file

    if output_file_path is None:
        output_file_path = dataset.annotations_dir.joinpath("excluded-sample-names.txt")

    with open(output_file_path, "w") as f:
        f.writelines(map(lambda sample_name: sample_name + "\n", excluded_sample_names))
