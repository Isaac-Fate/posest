import torch
from torch import Tensor


def mpjpe_loss(pred_keypoints: Tensor, true_keypoints: Tensor) -> Tensor:
    """Mean per joint position error.

    Parameters
    ----------
    pred_keypoints : Tensor
        Predicted coordinates.
    true_keypoints : Tensor
        True coordinates.

    Returns
    -------
    Tensor
        Mean per joint position error.
    """

    # Shape: (N, K)
    # The norm of the difference between the predicted and true coordinates
    # for each keypoint in each batch
    per_joint_position_error = torch.norm(pred_keypoints - true_keypoints, dim=-1)

    # Shape: (N,)
    # Mean per joint position error
    mean_per_joint_position_error = torch.mean(per_joint_position_error, dim=-1)

    # Mean sample error
    mean_sample_error = torch.mean(mean_per_joint_position_error)

    return mean_sample_error
