from pathlib import Path
import json
import typer

from .app import data_app


@data_app.command(help="Create JSON files that map sample names to indices.")
def smp2idx(
    dataset_dir: Path = typer.Argument(
        show_default=False,
        help="The path to the Human3.6M data directory.",
    ),
):

    # Annotations directory
    annotations_dir = dataset_dir.joinpath("annot")

    # List of image names for training
    train_image_names_filepath = annotations_dir.joinpath("train_images.txt")

    # List of image names for validation
    valid_image_names_filepath = annotations_dir.joinpath("valid_images.txt")

    # Create sample name to index file for training
    create_sample_name_to_index_file(train_image_names_filepath)

    # Create sample name to index file for validation
    create_sample_name_to_index_file(valid_image_names_filepath)


def create_sample_name_to_index_file(image_names_filepath: Path):

    # A dictionary that maps sample name to its index
    sample_name_to_index = {}

    # Read the list of image names
    with open(image_names_filepath, "r") as f:
        content = f.read().strip()
        image_names = content.split("\n")

    for index, image_name in enumerate(image_names):

        # Get the sample name
        sample_name = image_name.removesuffix(".jpg")

        # Add the sample name to the dictionary
        sample_name_to_index[sample_name] = index

    # Write to file

    if image_names_filepath.stem.startswith("train"):
        prefix = "train"
    elif image_names_filepath.stem.startswith("valid"):
        prefix = "valid"
    else:
        raise ValueError(f"Invalid file name: {image_names_filepath}")

    # Output filemame
    filename = image_names_filepath.parent.joinpath(
        f"{prefix}-sample-name-to-index.json"
    )

    with open(filename, "w") as f:
        json.dump(sample_name_to_index, f, indent=4)
