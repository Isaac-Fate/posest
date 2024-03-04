import plotly.graph_objects as go

from ..schemas import Pose, KeypointName


def get_line_segment_data(
    pose: Pose,
    keypoint_name_from: KeypointName,
    keypoint_name_to: KeypointName,
    *,
    color: str = "black",
    width: int = 10,
) -> list[go.Scatter3d]:
    """
    Generate line segment data for 3D scatter plot.

    Parameters
    ----------
        pose (Pose): The pose containing the keypoint data.
        keypoint_name_from (KeypointName): The name of the starting keypoint.
        keypoint_name_to (KeypointName): The name of the ending keypoint.
        color (str, optional): The color of the line segment. Defaults to "black".
        width (int, optional): The width of the line segment. Defaults to 10.

    Returns
    -------
        list[go.Scatter3d]: A list of 3D scatter plot line segments.
    """

    return [
        go.Scatter3d(
            x=[
                pose[keypoint_name_from].x,
                pose[keypoint_name_to].x,
            ],
            y=[
                pose[keypoint_name_from].y,
                pose[keypoint_name_to].y,
            ],
            z=[
                pose[keypoint_name_from].z,
                pose[keypoint_name_to].z,
            ],
            mode="lines",
            line=dict(color=color, width=width),
        )
    ]
