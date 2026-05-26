"""
Tversky loss for instance segmentation.
Generalization of Dice loss with separate weights for false positives and false negatives.
"""
import torch
import torch.nn.functional as F


def tversky_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ):
    """
    Compute Tversky loss for binary segmentation masks.

    Args:
        inputs: predictions (logits), shape [N, ...]
        targets: ground truth binary masks, shape [N, ...]
        num_masks: normalization factor (number of masks in the batch)
        alpha: weight for false positives (lower = less penalty on FP)
        beta: weight for false negatives (higher = more emphasis on recall)
        smooth: numerical stability term

    Notes:
        - alpha=0.3, beta=0.7: emphasizes recall (catches rare classes better)
        - alpha=beta=0.5: equivalent to Dice loss
        - Sum of alpha + beta does NOT need to equal 1

    Returns:
        Loss tensor (scalar)
    """
    # Apply sigmoid to convert logits to probabilities
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    # Compute true positives, false positives, false negatives
    TP = (inputs * targets).sum(-1)
    FP = (inputs * (1 - targets)).sum(-1)
    FN = ((1 - inputs) * targets).sum(-1)

    # Tversky index: generalization of Dice index
    tversky_index = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    loss = 1 - tversky_index

    return loss.sum() / num_masks


# JIT compilation for speed (matches the pattern used by dice_loss_jit in the project)
tversky_loss_jit = torch.jit.script(tversky_loss)
