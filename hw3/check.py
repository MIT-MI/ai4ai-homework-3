"""Tests for TODO 1-3 in losses.py. They run on a CPU in a second.

    python -m hw3.check
"""

import torch
from torch.distributions import Categorical, kl_divergence

import losses


def main():
    torch.manual_seed(0)
    B, T, V = 2, 5, 50
    s = (3 * torch.randn(B, T, V)).requires_grad_()
    t = 3 * torch.randn(B, T, V)
    y = torch.randint(V, (B, T))
    ps, pt = Categorical(logits=s), Categorical(logits=t)

    def check(name, fn, expected):
        try:
            got = fn()
        except NotImplementedError:
            return print(f"{name}: not implemented yet")
        assert got.shape == (B, T), f"{name}: shape {tuple(got.shape)}, expected {(B, T)} (one number per token)"
        assert torch.allclose(got, expected, atol=1e-4), f"{name}: wrong values"
        assert got.requires_grad, f"{name}: no gradient reaches the student"
        print(f"{name}: ok")

    check("TODO 1 forward_kl", lambda: losses.forward_kl(s, t), kl_divergence(pt, ps))
    check("TODO 2 reverse_kl", lambda: losses.reverse_kl(s, t), kl_divergence(ps, pt))
    check("TODO 3 advantage", lambda: losses.advantage(s, t, y), pt.log_prob(y) - ps.log_prob(y))

    # Both KLs are 0 when student and teacher agree, and finite when one of them is very sure of itself.
    sure = torch.zeros(1, 1, V)
    sure[..., 0] = 200.0
    for name, fn in (("forward_kl", losses.forward_kl), ("reverse_kl", losses.reverse_kl)):
        try:
            assert fn(s, s.detach()).abs().max() < 1e-4, f"{name}: should be 0 for equal distributions"
            assert torch.isfinite(fn(sure, torch.zeros(1, 1, V))).all(), f"{name}: not finite for a one-hot student"
            assert torch.isfinite(fn(torch.zeros(1, 1, V), sure)).all(), f"{name}: not finite for a one-hot teacher"
        except NotImplementedError:
            continue
        print(f"{name}: zero when equal, finite at the extremes: ok")


if __name__ == "__main__":
    main()
