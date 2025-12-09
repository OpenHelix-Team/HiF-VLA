"""Utils for training/fine-tuning scripts."""

import torch

from prismatic.vla.constants import ACTION_DIM, ACTION_TOKEN_BEGIN_IDX, IGNORE_INDEX


def get_current_action_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = (1 <= cumsum) & (cumsum <= ACTION_DIM)

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def get_next_actions_mask(token_ids):
    # Create a tensor marking positions of IGNORE_INDEX
    newline_positions = token_ids != IGNORE_INDEX

    # Calculate cumulative sum to identify regions between newlines
    cumsum = torch.cumsum(newline_positions, dim=1)

    # Create the mask
    mask = cumsum > ACTION_DIM

    # Extract the action part only
    action_tokens_only_mask = token_ids > ACTION_TOKEN_BEGIN_IDX
    mask = action_tokens_only_mask * mask

    return mask


def compute_token_accuracy(predicted_token_ids, ground_truth_token_ids, mask):
    correct_preds = (predicted_token_ids == ground_truth_token_ids) & mask
    accuracy = correct_preds.sum().float() / mask.sum().float()
    return accuracy


def compute_actions_l1_loss(action_tokenizer, predicted_token_ids, ground_truth_token_ids, mask):
    pred_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(predicted_token_ids[mask].cpu().numpy())
    )
    true_continuous_actions = torch.tensor(
        action_tokenizer.decode_token_ids_to_actions(ground_truth_token_ids[mask].cpu().numpy())
    )
    l1_loss = torch.nn.functional.l1_loss(pred_continuous_actions, true_continuous_actions)
    return l1_loss


def compute_motion_mse_loss(predicted_motion, ground_truth_motion):
    """
    Computes the Mean Squared Error (MSE) loss for motion data.
    
    Args:
        predicted_motion (torch.Tensor): Predicted motion data.
        ground_truth_motion (torch.Tensor): Ground truth motion data.
        mask (torch.Tensor): Mask to apply to the loss calculation.
        
    Returns:
        torch.Tensor: Computed MSE loss.
    """
    # masked_predicted_motion = predicted_motion[mask]
    # masked_ground_truth_motion = ground_truth_motion[mask]
    #F.mse_loss
    # mse_loss = torch.nn.functional.mse_loss(predicted_motion, ground_truth_motion)
    mse_loss = torch.nn.functional.l1_loss(predicted_motion, ground_truth_motion)
    return mse_loss
