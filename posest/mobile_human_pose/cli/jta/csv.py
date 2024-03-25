from typing import Annotated, Optional
from pathlib import Path
import json
import typer
from rich.progress import track

from .app import jta_app


@jta_app.command(
    name="csv",
    help="Create CSV annotation files.",
)
def create_csv_annotations(
    json_annotations_dir: Annotated[
        Path,
        typer.Argument(
            show_default=False,
            help="The root directory containing the JSON annotations.",
        ),
    ],
    *,
    parent_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--parent",
            "-p",
            help="The parent directory of the CSV annotations to be created. If not specified, the parent directory of the JSON annotations will be used.",
        ),
    ] = None,
    dir_name: Annotated[
        str,
        typer.Option(
            "--name",
            "-n",
            help="The name of the CSV annotations directory to be created. ",
        ),
    ] = "csv-annotations",
) -> None:

    # Import packages that take long time to load
    import pandas as pd
    from posest.data.jta import Pose

    if parent_dir is None:
        parent_dir = json_annotations_dir.parent

    # Root directory for CSV annotations
    csv_annotations_dir = parent_dir.joinpath(dir_name)
    csv_annotations_dir.mkdir(parents=True, exist_ok=True)

    for sub_dir_name in ("train", "val", "test"):

        # Enter sub directory
        sub_dir = json_annotations_dir.joinpath(sub_dir_name)

        # Sub directory of CSV annotations
        csv_sub_dir = csv_annotations_dir.joinpath(sub_dir_name)
        csv_sub_dir.mkdir(parents=True, exist_ok=True)

        # Each json annotation file contains all annotations for one entire video sequence
        json_annotation_paths = list(sub_dir.glob("*.json"))
        for json_annotation_path in track(
            json_annotation_paths,
            description=f"Writing into {csv_sub_dir}...",
        ):

            # Get file name without extension
            # It should be in the format of `seq_<seq_number>`
            file_name = json_annotation_path.stem

            # Get the sequence number
            seq_number = int(file_name.split("_")[-1])

            # Read the JSON file
            with open(json_annotation_path, "r") as f:
                seq_annotation = json.load(f)

            # Convert to a data frame
            seq_annotation = pd.DataFrame(
                seq_annotation,
                columns=(
                    "frame_number",
                    "person_id",
                    "keypoint_type",
                    "frame_x",
                    "frame_y",
                    "world_x",
                    "world_y",
                    "world_z",
                    "is_occluded",
                    "is_self_occluded",
                ),
            ).astype(
                {
                    "frame_number": int,
                    "person_id": int,
                    "keypoint_type": int,
                    "frame_x": int,
                    "frame_y": int,
                    "is_occluded": bool,
                    "is_self_occluded": bool,
                }
            )

            # Get frame-number-person-id pairs
            frame_number_person_id_pairs = seq_annotation.loc[
                :, ("frame_number", "person_id")
            ].drop_duplicates()
            frame_number_person_id_pairs = tuple(
                zip(
                    frame_number_person_id_pairs.frame_number,
                    frame_number_person_id_pairs.person_id,
                )
            )

            for frame_number, person_id in frame_number_person_id_pairs:
                # Get the data frame for the specified frame and person
                data_frame = seq_annotation.query(
                    "frame_number == @frame_number and person_id == @person_id"
                ).sort_values(by="keypoint_type")

                # Create a pose object
                pose = Pose.from_data_frame(
                    seq_number=seq_number,
                    frame_number=frame_number,
                    person_id=person_id,
                    data_frame=data_frame,
                    # Don't set the hip coordinate as the origin
                    # since we want to keep the original data
                    set_hip_as_origin=False,
                )

                # Save the pose only if it is visible
                if pose.is_visible:
                    pose.to_csv(csv_sub_dir)
