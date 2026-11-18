"""The per-token losses (TODO 1 to 3). Test them with `python -m hw3.check`.

All functions work on a batch of B responses, each padded to length T, with a vocabulary of V tokens. At every
position t they receive the logits (unnormalized log-probabilities) that predict token t of the response:

    student_logits  [B, T, V]   the student, reading x:        p_S = softmax(student_logits) = pi_theta( . | x, y_<t)
    teacher_logits  [B, T, V]   the teacher, reading x and c:  p_T = softmax(teacher_logits) = pi_0( . | x, c, y_<t)
    y               [B, T]      the token ids of the responses (only in cross_entropy and advantage)

Each function returns one number per position, a tensor of shape [B, T]. Padding is handled by the caller, which
averages over the real tokens only. Gradients flow through student_logits; teacher_logits has none.

cross_entropy is done for you and shows the pattern. Work with log-probabilities (F.log_softmax): taking the log of
F.softmax gives -inf for very unlikely tokens.
"""

import torch
import torch.nn.functional as F


def cross_entropy(student_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The SFT loss, -log p_S(y_t). Given.

    Args:
        student_logits: [B, T, V].
        y: [B, T], the tokens of the response.
    Returns:
        [B, T]."""
    logp_s = F.log_softmax(student_logits, dim=-1)  # [B, T, V]
    return -logp_s.gather(-1, y[..., None]).squeeze(-1)  # pick the log-probability of y_t at every position


def forward_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """TODO 1. The forward KL, KL(p_T || p_S) = sum_v p_T(v) [log p_T(v) - log p_S(v)], at every position.

    The sum runs over the whole vocabulary (the last dimension). This is the SDFT loss.

    Args:
        student_logits: [B, T, V].
        teacher_logits: [B, T, V].
    Returns:
        [B, T], one KL per position, each >= 0."""
    raise NotImplementedError("TODO 1")


def reverse_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """TODO 2. The reverse KL, KL(p_S || p_T) = sum_v p_S(v) [log p_S(v) - log p_T(v)], at every position.

    The same as TODO 1 with the roles swapped. It is also used to measure kl_to_base.

    Args:
        student_logits: [B, T, V].
        teacher_logits: [B, T, V].
    Returns:
        [B, T], one KL per position, each >= 0."""
    raise NotImplementedError("TODO 2")


def advantage(student_logits: torch.Tensor, teacher_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """TODO 3. The per-token advantage A_t = log p_T(y_t) - log p_S(y_t).

    It is positive where the teacher finds the sampled token more likely than the student does. It is not a loss:
    analyze.py draws it as a heatmap over the response. Look at cross_entropy for how to pick out the token y_t.

    Args:
        student_logits: [B, T, V].
        teacher_logits: [B, T, V].
        y: [B, T], the tokens of the response.
    Returns:
        [B, T]."""
    raise NotImplementedError("TODO 3")
