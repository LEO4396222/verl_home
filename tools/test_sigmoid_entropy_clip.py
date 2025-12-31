#!/usr/bin/env python3
import torch

from verl.trainer.ppo.core_algos import apply_advantage_clip


def main() -> int:
    advantages = torch.tensor([[2.0, -2.0]], dtype=torch.float32)
    probs = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    log_prob = torch.log(probs)
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "mode": "sigmoid",
        "sigmoid_p0_prob": 0.5,
        "sigmoid_alpha_pos": 2.0,
        "sigmoid_alpha_neg": 3.0,
    }

    clipped, stats = apply_advantage_clip(
        advantages,
        log_prob,
        response_mask,
        clip_cfg,
        return_clip_metrics=True,
        entropy_current=0.5,
        entropy_target=1.0,
    )

    alpha_pos_actual = torch.tensor(1.0)
    alpha_neg_actual = torch.tensor(-1.5)
    f_plus = 2.0 / (1.0 + torch.exp(alpha_pos_actual * (probs - 0.5)))
    f_minus = 2.0 / (1.0 + torch.exp(alpha_neg_actual * (probs - 0.5)))
    expected = advantages * torch.tensor([f_plus[0, 0], f_minus[0, 1]], dtype=torch.float32)

    if not torch.allclose(clipped, expected, atol=1e-6):
        print("ERROR: clipped mismatch")
        print("clipped:", clipped)
        print("expected:", expected)
        return 1

    clip_cfg_quantile = {
        "enable": True,
        "mode": "sigmoid",
        "sigmoid_p0_prob": None,
        "sigmoid_p0_quantile": 0.75,
        "sigmoid_alpha_pos": 2.0,
        "sigmoid_alpha_neg": 3.0,
    }
    clipped_q, stats_q = apply_advantage_clip(
        advantages,
        log_prob,
        response_mask,
        clip_cfg_quantile,
        return_clip_metrics=True,
        entropy_current=0.5,
        entropy_target=1.0,
    )

    p0_q = torch.quantile(probs[response_mask.bool()], 0.75)
    p0_q = torch.clamp(p0_q, min=1e-6, max=1.0)
    f_plus_q = 2.0 / (1.0 + torch.exp(alpha_pos_actual * (probs - p0_q)))
    f_minus_q = 2.0 / (1.0 + torch.exp(alpha_neg_actual * (probs - p0_q)))
    expected_q = advantages * torch.tensor([f_plus_q[0, 0], f_minus_q[0, 1]], dtype=torch.float32)

    if not torch.allclose(clipped_q, expected_q, atol=1e-6):
        print("ERROR: clipped quantile mismatch")
        print("clipped:", clipped_q)
        print("expected:", expected_q)
        return 1

    print(
        "OK:",
        stats["adv_sigmoid_alpha_pos"].item(),
        stats["adv_sigmoid_alpha_neg"].item(),
        stats_q["adv_sigmoid_p0"].item(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
