from typing import Self
from pydantic import BaseModel
import numpy as np
import plotly.graph_objects as go

from ..pose import Pose
from ..keypoint_type import KeypointType


class PlotConfig(BaseModel):

    line_color: str = "black"
    line_width: int = 10
    keypoint_marker_color: str = "blue"
    keypoint_marker_size: int = 5


class Pose3DPlotter:

    def __init__(
        self,
        keypoint_world_coordinates: np.ndarray,
        *,
        plot_config: PlotConfig = PlotConfig(),
    ) -> None:

        self._keypoint_world_coordinates = keypoint_world_coordinates
        self._plot_config = plot_config

    @classmethod
    def from_pose(
        cls,
        pose: Pose,
        *,
        plot_config: PlotConfig = PlotConfig(),
    ) -> Self:

        return cls(
            keypoint_world_coordinates=pose.keypoint_world_coordinates,
            plot_config=plot_config,
        )

    def figure(
        self,
    ) -> go.Figure:

        # Layout
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                # * Use data mode to have the correct aspect ratio
                aspectmode="data",
                camera=dict(
                    # The nagetive y-axis is up
                    up=dict(x=0, y=-1, z=0),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=0, y=0, z=-3),
                ),
            ),
            showlegend=False,
        )

        # Create pose data
        data = self.pose_data()

        # Create figure
        fig = go.Figure(
            data=data,
            layout=layout,
        )

        return fig

    def pose_data(self) -> list[go.Scatter3d]:

        # Initialize data consists of line segments of body parts
        data = []

        # Head
        data.extend(
            [
                self._line_segment_data(
                    from_keypoint_type=KeypointType.HEAD_TOP,
                    to_keypoint_type=KeypointType.HEAD_CENTER,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.HEAD_CENTER,
                    to_keypoint_type=KeypointType.NECK,
                ),
            ]
        )

        # Left arm
        data.extend(
            [
                self._line_segment_data(
                    from_keypoint_type=KeypointType.NECK,
                    to_keypoint_type=KeypointType.LEFT_CLAVICLE,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.LEFT_CLAVICLE,
                    to_keypoint_type=KeypointType.LEFT_SHOULDER,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.LEFT_SHOULDER,
                    to_keypoint_type=KeypointType.LEFT_ELBOW,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.LEFT_ELBOW,
                    to_keypoint_type=KeypointType.LEFT_WRIST,
                ),
            ]
        )

        # Right arm
        data.extend(
            [
                self._line_segment_data(
                    from_keypoint_type=KeypointType.NECK,
                    to_keypoint_type=KeypointType.RIGHT_CLAVICLE,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.RIGHT_CLAVICLE,
                    to_keypoint_type=KeypointType.RIGHT_SHOULDER,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.RIGHT_SHOULDER,
                    to_keypoint_type=KeypointType.RIGHT_ELBOW,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.RIGHT_ELBOW,
                    to_keypoint_type=KeypointType.RIGHT_WRIST,
                ),
            ]
        )

        # Spine
        data.extend(
            [
                self._line_segment_data(
                    from_keypoint_type=KeypointType.HIP,
                    to_keypoint_type=KeypointType.SPINE1,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.SPINE1,
                    to_keypoint_type=KeypointType.SPINE2,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.SPINE2,
                    to_keypoint_type=KeypointType.SPINE3,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.SPINE3,
                    to_keypoint_type=KeypointType.SPINE4,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.SPINE4,
                    to_keypoint_type=KeypointType.NECK,
                ),
            ]
        )

        # Left leg
        data.extend(
            [
                self._line_segment_data(
                    from_keypoint_type=KeypointType.HIP,
                    to_keypoint_type=KeypointType.LEFT_HIP,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.LEFT_HIP,
                    to_keypoint_type=KeypointType.LEFT_KNEE,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.LEFT_KNEE,
                    to_keypoint_type=KeypointType.LEFT_ANKLE,
                ),
            ]
        )

        # Right leg
        data.extend(
            [
                self._line_segment_data(
                    from_keypoint_type=KeypointType.HIP,
                    to_keypoint_type=KeypointType.RIGHT_HIP,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.RIGHT_HIP,
                    to_keypoint_type=KeypointType.RIGHT_KNEE,
                ),
                self._line_segment_data(
                    from_keypoint_type=KeypointType.RIGHT_KNEE,
                    to_keypoint_type=KeypointType.RIGHT_ANKLE,
                ),
            ]
        )

        return data

    def _line_segment_data(
        self,
        from_keypoint_type: KeypointType,
        to_keypoint_type: KeypointType,
    ) -> go.Scatter3d:

        from_keypoint_world_coordinate = self._keypoint_world_coordinates[
            from_keypoint_type
        ]
        to_keypoint_world_coordinate = self._keypoint_world_coordinates[
            to_keypoint_type
        ]

        return go.Scatter3d(
            x=[from_keypoint_world_coordinate[0], to_keypoint_world_coordinate[0]],
            y=[from_keypoint_world_coordinate[1], to_keypoint_world_coordinate[1]],
            z=[from_keypoint_world_coordinate[2], to_keypoint_world_coordinate[2]],
            line=go.scatter3d.Line(
                color=self._plot_config.line_color,
                width=self._plot_config.line_width,
            ),
            marker=go.scatter3d.Marker(
                color=self._plot_config.keypoint_marker_color,
                size=self._plot_config.keypoint_marker_size,
            ),
        )
