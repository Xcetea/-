import torch

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer) -> dict[str, torch.Tensor]:
    """
    Tokenizes the prompt and output separately, concatenates them, and constructs 
    the necessary tensors for supervised fine-tuning.
    """
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")
        
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    sequences = []
    masks = []
    
    for prompt, output in zip(prompt_strs, output_strs):
        # 1. Tokenize separately as instructed. 
        # add_special_tokens=False prevents duplicate BOS/EOS from being injected automatically.
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)
        
        # 2. Concatenate them together (NO manual EOS appending, the dataset already handles it)
        seq = prompt_ids + output_ids
        
        # 3. Construct mask: 0 for prompt, 1 for the output
        mask = [0] * len(prompt_ids) + [1] * len(output_ids)
        
        sequences.append(seq)
        masks.append(mask)
        
    # Determine the maximum sequence length in the batch (max(prompt_and_output_lens))
    max_len = max(len(seq) for seq in sequences)
    
    input_ids_batch = []
    labels_batch = []
    response_mask_batch = []
    
    for seq, mask in zip(sequences, masks):
        # Apply Right-Padding up to max_len
        pad_length = max_len - len(seq)
        padded_seq = seq + [pad_token_id] * pad_length
        padded_mask = mask + [0] * pad_length  
        
        # 4. Slice exactly according to instructions:
        # input_ids: final token sliced off
        # labels: first token sliced off (shifted input ids)
        # response_mask: mask on the response tokens in the labels (shifted identically)
        input_ids_batch.append(padded_seq[:-1])
        labels_batch.append(padded_seq[1:])
        response_mask_batch.append(padded_mask[1:])
        
    return {
        "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
        "labels": torch.tensor(labels_batch, dtype=torch.long),
        "response_mask": torch.tensor(response_mask_batch, dtype=torch.long)
    }

import torch

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).
    
    Args:
        logits: torch.Tensor Tensor of shape (batch_size, sequence_length, vocab_size)
                containing unnormalized logits.
                
    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token
        prediction.
    """
    # 1. Compute the log of the sum of exponentials over the vocab dimension
    # keepdim=True allows broadcasting back against the original logits tensor
    lse = torch.logsumexp(logits, dim=-1, keepdim=True)
    
    # 2. Compute numerically stable log probabilities
    # log(p_i) = z_i - log(sum(exp(z)))
    log_probs = logits - lse
    
    # 3. Compute probabilities
    probs = torch.exp(log_probs)
    
    # 4. Compute entropy: H(p) = -sum(p_i * log(p_i))
    entropy = -torch.sum(probs * log_probs, dim=-1)
    
    return entropy



import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Gets per-token conditional log-probabilities from a causal language model, 
    and optionally the entropy of the next-token distribution.
    
    Args:
        model: PreTrainedModel HuggingFace model used for scoring.
        input_ids: torch.Tensor shape (batch_size, sequence_length).
        labels: torch.Tensor shape (batch_size, sequence_length).
        return_token_entropy: bool If True, returns per-token entropy.
        
    Returns:
        dict[str, torch.Tensor] containing "log_probs" and optionally "token_entropy".
    """
    # 1. Obtain the unnormalized logits from the model
    # Shape: (batch_size, sequence_length, vocab_size)
    outputs = model(input_ids)
    logits = outputs.logits
    
    # 2. Compute log probabilities across the entire vocabulary dimension
    # F.log_softmax is numerically stable under the hood
    log_probs_vocab = F.log_softmax(logits, dim=-1)
    
    # 3. Extract the log-probabilities corresponding to the actual labels
    # We need to map our 2D labels tensor to index the 3D log_probs_vocab tensor.
    # Expand labels to (batch_size, sequence_length, 1)
    labels_expanded = labels.unsqueeze(-1)
    
    # Gather the specific log probabilities and squeeze back to (batch_size, sequence_length)
    log_probs = torch.gather(log_probs_vocab, dim=-1, index=labels_expanded).squeeze(-1)
    
    # 4. Construct the results dictionary
    result = {"log_probs": log_probs}
    
    # 5. Optionally compute and add token entropy
    if return_token_entropy:
        # Calls the compute_entropy function defined in the previous deliverable
        result["token_entropy"] = compute_entropy(logits)
        
    return result



import torch

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float =1.0,
) -> torch.Tensor:
    """
    Sum over a dimension and normalize by a constant, considering only those 
    elements where mask == 1.
    
    Args:
        tensor: torch.Tensor The tensor to sum and normalize.
        mask: torch.Tensor Same shape as tensor; positions with 1 are included in the sum.
        normalize_constant: float The constant used to divide the result.
        dim: int | None The dimension to sum over. If None, sums over all dimensions.
        
    Returns:
        torch.Tensor The masked, summed, and normalized tensor.
    """
    # Cast the mask to the same data type as the tensor to prevent type mismatch 
    # errors (e.g., if mask is boolean or long, and tensor is float).
    masked_tensor = tensor * mask.to(tensor.dtype)
    
    # Sum over the requested dimension(s)
    if dim is None:
        summed = masked_tensor.sum()
    else:
        summed = masked_tensor.sum(dim=dim)
        
    # Normalize by the constant
    normalized = summed / normalize_constant
    
    return normalized

import torch

def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch for Supervised Fine-Tuning.
    """
    # 1. Compute per-token Cross-Entropy (Negative Log-Likelihood)
    nll = -policy_log_probs
    
    # 2. Mask the ignored tokens
    masked_nll = nll * response_mask.to(nll.dtype)
    
    # 3. Sum over the entire tensor (batch and sequence dimensions)
    summed_loss = masked_nll.sum()
    
    # 4. Average over the batch and apply the normalize_constant
    batch_size = policy_log_probs.shape[0]
    microbatch_loss = summed_loss / (batch_size * normalize_constant)
    
    # 5. Scale for gradient accumulation
    # The instructions specifically request returning the loss "adjusted for gradient accumulation"
    scaled_loss = microbatch_loss / gradient_accumulation_steps
    
    # 6. Backward pass on the perfectly scaled loss
    scaled_loss.backward()
    
    # 7. Prepare metadata
    metadata = {
        "loss": microbatch_loss.detach(),
        "scaled_loss": scaled_loss.detach(),
        "summed_masked_loss": summed_loss.detach(),
    }
    
    return scaled_loss, metadata



import torch
from typing import Callable

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, normalized by the group size.
    
    Args:
        reward_fn: Scores the rollout responses against the ground truths.
        rollout_responses: Rollouts from the policy. 
        repeated_ground_truths: The ground truths for the examples.
        group_size: Number of responses per question (group).
        advantage_eps: Small constant to avoid division by zero in normalization.
        normalize_by_std: If True, divide by the per-group standard deviation.
        
    Returns:
        tuple containing advantages (normalized), raw_rewards, and metadata.
    """
    if len(rollout_responses) % group_size != 0:
        raise ValueError("The total number of rollouts must be a multiple of the group_size.")

    raw_rewards_list = []
    format_rewards_list = []
    answer_rewards_list = []
    
    # 1. Compute raw rewards for every response
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        scores = reward_fn(response, ground_truth)
        
        raw_rewards_list.append(scores.get("reward", 0.0))
        format_rewards_list.append(scores.get("format_reward", 0.0))
        answer_rewards_list.append(scores.get("answer_reward", 0.0))
        
    # Convert to 1D PyTorch tensor
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)
    
    # 2. Reshape to compute group-wise statistics
    # Shape becomes: (n_prompts, group_size)
    n_prompts = len(raw_rewards) // group_size
    grouped_rewards = raw_rewards.view(n_prompts, group_size)
    
    # 3. Compute group means
    # keepdim=True allows broadcasting back to the grouped shape
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    
    # 4. Calculate advantages (zero-centered around the group mean)
    advantages = grouped_rewards - group_means
    
    

    # 5. Optionally normalize by the standard deviation
    if normalize_by_std:
        # Remove unbiased=False to use Bessel's correction (N-1), matching the test suite
        group_stds = grouped_rewards.std(dim=1, keepdim=True)
        advantages = advantages / (group_stds + advantage_eps)
        
    # 6. Flatten the advantages back to a 1D tensor: shape (rollout_batch_size,)
    advantages = advantages.view(-1)
    
    # 7. Compile metadata metrics for logging
    metadata = {
        "reward/mean": raw_rewards.mean().item(),
        "reward/std": raw_rewards.std().item(),
        "reward/max": raw_rewards.max().item(),
        "reward/min": raw_rewards.min().item(),
        "reward/format_mean": sum(format_rewards_list) / len(format_rewards_list) if format_rewards_list else 0.0,
        "reward/answer_mean": sum(answer_rewards_list) / len(answer_rewards_list) if answer_rewards_list else 0.0,
        "advantage/mean": advantages.mean().item(),
        "advantage/std": advantages.std().item()
    }



    
    
    return advantages, raw_rewards, metadata






import torch

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages 
    is either the raw reward or an already-normalized advantage.
    
    Args:
        raw_rewards_or_advantages: torch.Tensor Shape (batch_size, 1), scalar
            reward/advantage for each rollout response.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), logprobs for
            each token.
            
    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss.
    """
    # The policy gradient objective aims to MAXIMIZE expected reward: E[Reward * log(prob)]
    # To use standard gradient DESCENT optimizers, we minimize the negative objective.
    # PyTorch broadcasting automatically expands the (batch_size, 1) reward tensor
    # across the (batch_size, sequence_length) log-probabilities.
    
    per_token_loss = -raw_rewards_or_advantages * policy_log_probs
    
    return per_token_loss




import torch

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the per-token GRPO-Clip loss.
    
    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length).
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length).
        cliprange: float Clip parameter epsilon.
        
    Returns:
        tuple containing:
            - loss: torch.Tensor of shape (batch_size, sequence_length)
            - metadata: dict of logging tensors
    """
    # 1. Compute the probability ratio r_t(θ) in log space to avoid underflow
    # r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
    # log(r_t(θ)) = log(π_θ) - log(π_old)
    ratio = torch.exp(policy_log_probs - old_log_probs)
    
    # 2. Compute the unclipped surrogate objective (LHS)
    # PyTorch automatically broadcasts the (batch_size, 1) advantages 
    # across the (batch_size, sequence_length) ratio tensor.
    surr1 = ratio * advantages
    
    # 3. Compute the clipped surrogate objective (RHS)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    surr2 = clipped_ratio * advantages
    
    # 4. The objective to MAXIMIZE is the minimum of surr1 and surr2.
    # Therefore, the loss to MINIMIZE is the negative minimum.
    ppo_objective = torch.min(surr1, surr2)
    per_token_loss = -ppo_objective
    
    # 5. Track metadata (e.g., whether the token loss was restricted by clipping)
    # It is clipped if the clipped objective (RHS) is strictly less than the unclipped (LHS)
    is_clipped = (surr2 < surr1).to(torch.float32)
    
    metadata = {
        "is_clipped": is_clipped,
        "ratio": ratio.detach(),
    }
    
    return per_token_loss, metadata




import torch
from typing import Literal

def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.
    
    Args:
        policy_log_probs: (batch_size, sequence_length), per-token log-probs from the policy.
        loss_type: The specific policy gradient algorithm to use.
        raw_rewards: Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages: Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs: Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange: Required for "grpo_clip"; scalar clipping parameter.
        
    Returns:
        tuple containing:
            - loss: (batch_size, sequence_length), per-token loss.
            - metadata: dict of logging statistics.
    """
    if loss_type == "no_baseline":
        assert raw_rewards is not None, "raw_rewards must be provided for 'no_baseline'"
        # Delegate to the naive policy gradient using raw rewards
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=raw_rewards,
            policy_log_probs=policy_log_probs
        )
        metadata = {}
        
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None, "advantages must be provided for 'reinforce_with_baseline'"
        # Delegate to the naive policy gradient using pre-computed advantages
        loss = compute_naive_policy_gradient_loss(
            raw_rewards_or_advantages=advantages,
            policy_log_probs=policy_log_probs
        )
        metadata = {}
        
    elif loss_type == "grpo_clip":
        assert advantages is not None, "advantages must be provided for 'grpo_clip'"
        assert old_log_probs is not None, "old_log_probs must be provided for 'grpo_clip'"
        assert cliprange is not None, "cliprange must be provided for 'grpo_clip'"
        
        # Delegate to the specialized PPO/GRPO clip loss calculation
        loss, metadata = compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange
        )
        
    else:
        raise ValueError(f"Unsupported loss_type: {loss_type}")
        
    return loss, metadata




import torch

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    """
    Compute the mean of tensor along a given dimension, considering only those elements where
    mask == 1.
    """
    # 1. Cast mask to the same data type as the tensor
    mask = mask.to(tensor.dtype)
    
    # 2. Zero out the ignored elements and sum
    masked_tensor = tensor * mask
    masked_sum = masked_tensor.sum(dim=dim)
    
    # 3. Count how many valid elements contributed to the sum
    valid_count = mask.sum(dim=dim)
    
    # 4. Standard division. 
    # If a slice is completely masked out, this naturally computes 0.0 / 0.0 = NaN,
    # which exactly matches standard PyTorch .mean() semantics for empty slices.
    return masked_sum / valid_count



import torch
from typing import Literal

# Note: This assumes compute_policy_gradient_loss and masked_mean 
# are available in your namespace from the previous steps.

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Execute a forward-and-backward pass on a microbatch.
    """
    # 1. Delegate to our wrapper to get the per-token loss and algorithm-specific metadata
    per_token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange
    )
    
    # 2. Average the loss over only the valid response tokens
    # Using dim=None computes the mean across all masked elements in the entire microbatch
    mean_loss = masked_mean(tensor=per_token_loss, mask=response_mask, dim=None)
    
    # 3. Scale the loss to account for gradient accumulation
    scaled_loss = mean_loss / gradient_accumulation_steps
    
    # 4. Trigger the backward pass on the properly scaled loss
    scaled_loss.backward()
    
    # 5. Append useful loss statistics to our metadata dictionary
    metadata["microbatch_loss"] = mean_loss.detach()
    metadata["scaled_loss"] = scaled_loss.detach()
    
    return scaled_loss, metadata


