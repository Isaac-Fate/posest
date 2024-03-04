import plotly.graph_objects as go

from ..schemas import Pose, KeypointName
from .line_segment_data import get_line_segment_data


def get_plot_data(pose: Pose) -> list:

    # Plot data
    plot_data = []

    # Left arm

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_SHOULDER,
            keypoint_name_to=KeypointName.LEFT_ELBOW,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_ELBOW,
            keypoint_name_to=KeypointName.LEFT_HAND,
        )
    )

    # Right arm

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_SHOULDER,
            keypoint_name_to=KeypointName.RIGHT_ELBOW,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_ELBOW,
            keypoint_name_to=KeypointName.RIGHT_HAND,
        )
    )

    # Body

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.NECK,
            keypoint_name_to=KeypointName.LEFT_SHOULDER,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.NECK,
            keypoint_name_to=KeypointName.RIGHT_SHOULDER,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.HEAD,
            keypoint_name_to=KeypointName.NECK,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.NECK,
            keypoint_name_to=KeypointName.SPINE,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.SPINE,
            keypoint_name_to=KeypointName.ROOT,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.ROOT,
            keypoint_name_to=KeypointName.LEFT_HIP,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.ROOT,
            keypoint_name_to=KeypointName.RIGHT_HIP,
        )
    )

    # Left leg

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_HIP,
            keypoint_name_to=KeypointName.LEFT_KNEE,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.LEFT_KNEE,
            keypoint_name_to=KeypointName.LEFT_ANKLE,
        )
    )

    # Right leg

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_HIP,
            keypoint_name_to=KeypointName.RIGHT_KNEE,
        )
    )

    plot_data.extend(
        get_line_segment_data(
            pose=pose,
            keypoint_name_from=KeypointName.RIGHT_KNEE,
            keypoint_name_to=KeypointName.RIGHT_ANKLE,
        )
    )

    return plot_data


def plot_pose(pose: Pose):

    fig = go.Figure(
        data=get_plot_data(pose),
        layout=go.Layout(
            showlegend=False,
            scene=dict(
                aspectmode="manual",
                aspectratio=dict(
                    x=1,
                    y=1,
                    z=1,
                ),
                # xaxis=dict(
                #     nticks=10,
                #     range=[-1, 1],
                # ),
                # yaxis=dict(
                #     nticks=10,
                #     range=[-1, 1],
                # ),
                # zaxis=dict(
                #     nticks=10,
                #     range=[-1, 1],
                # ),
                # camera=dict(
                #     up=dict(
                #         x=0,
                #         y=-1,
                #         z=0,
                #     ),
                #     eye=dict(
                #         x=0,
                #         y=0,
                #         z=-2,
                #     ),
                # ),
            ),
        ),
    )

    # Tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

    # Plot!
    fig.show()
