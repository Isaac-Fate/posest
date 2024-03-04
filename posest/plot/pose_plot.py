import plotly.graph_objects as go

from ..schemas import Pose, KeypointName
from .line_segment_data import get_line_segment_data


def get_plot_data(
    pose: Pose,
    *,
    line_width: float,
    line_color: str,
) -> list:

    # Plot data
    plot_data = []

    # Left arm

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_SHOULDER,
            keypoint_name_to=KeypointName.LEFT_ELBOW,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_ELBOW,
            keypoint_name_to=KeypointName.LEFT_WRIST,
            width=line_width,
            color=line_color,
        )
    )

    # Right arm

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_SHOULDER,
            keypoint_name_to=KeypointName.RIGHT_ELBOW,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_ELBOW,
            keypoint_name_to=KeypointName.RIGHT_WRIST,
            width=line_width,
            color=line_color,
        )
    )

    # Body

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.THORAX,
            keypoint_name_to=KeypointName.LEFT_SHOULDER,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.THORAX,
            keypoint_name_to=KeypointName.RIGHT_SHOULDER,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.HEAD,
            keypoint_name_to=KeypointName.NECK,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.NECK,
            keypoint_name_to=KeypointName.THORAX,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.THORAX,
            keypoint_name_to=KeypointName.SPINE,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.SPINE,
            keypoint_name_to=KeypointName.HIP,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.HIP,
            keypoint_name_to=KeypointName.LEFT_HIP,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.HIP,
            keypoint_name_to=KeypointName.RIGHT_HIP,
            width=line_width,
            color=line_color,
        )
    )

    # Left leg

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_HIP,
            keypoint_name_to=KeypointName.LEFT_KNEE,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_KNEE,
            keypoint_name_to=KeypointName.LEFT_ANKLE,
            width=line_width,
            color=line_color,
        )
    )

    # Right leg

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_HIP,
            keypoint_name_to=KeypointName.RIGHT_KNEE,
            width=line_width,
            color=line_color,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_KNEE,
            keypoint_name_to=KeypointName.RIGHT_ANKLE,
            width=line_width,
            color=line_color,
        )
    )

    return plot_data


def plot_pose(
    pose: Pose,
    *,
    line_width: float = 7.0,
    line_color: str = "black",
) -> None:

    fig = go.Figure(
        data=get_plot_data(
            pose,
            line_width=line_width,
            line_color=line_color,
        ),
        layout=go.Layout(
            showlegend=False,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=1,
                ),
                xaxis=dict(
                    nticks=10,
                    range=[0, 1],
                ),
                yaxis=dict(
                    nticks=10,
                    range=[0, 1],
                ),
                zaxis=dict(
                    nticks=10,
                    range=[0, 1],
                ),
                camera=dict(
                    up=dict(
                        x=0,
                        y=-1,
                        z=0,
                    ),
                    eye=dict(
                        x=0,
                        y=0,
                        z=-2,
                    ),
                ),
            ),
        ),
    )

    # Tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Plot!
    fig.show()
