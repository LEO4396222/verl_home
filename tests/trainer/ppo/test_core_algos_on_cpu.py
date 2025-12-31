# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    apply_advantage_clip,
    compute_gae_advantage_return,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)


def mock_test_fn():
    pass


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def test_apply_advantage_clip_coef_mode_matches_previous_behavior():
    """mode=coef 时保持现有概率系数裁剪逻辑不变。"""
    advantages = torch.tensor([[2.0, -2.0, 0.5, -0.5]], dtype=torch.float32)
    log_prob = torch.log(torch.tensor([[0.2, 0.8, 0.5, 0.5]], dtype=torch.float32))
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "positive_coef": 1.0,
        "negative_coef": -1.0,
        "prob_epsilon": 1e-6,
        # 不显式设置 mode，默认应为 coef
    }

    clipped, stats = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )
    expected = torch.tensor([[2.0, -0.8, 0.5, -0.5]], dtype=torch.float32)
    assert torch.allclose(clipped, expected, atol=1e-6)
    assert stats is not None
    assert stats["adv_clipfrac_pos"].item() == pytest.approx(0.0, rel=1e-6)
    assert stats["adv_clipfrac_neg"].item() == pytest.approx(0.25, rel=1e-6)


def test_apply_advantage_clip_quantile_mode_clips_extremes():
    """mode=quantile 时在极端概率/优势上触发高低分支。"""
    advantages = torch.tensor([[10.0, 8.0, 6.0, -6.0, -8.0, -10.0]], dtype=torch.float32)
    probs = torch.tensor([[0.95, 0.9, 0.85, 0.1, 0.05, 0.01]], dtype=torch.float32)
    log_prob = torch.log(probs)
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "mode": "quantile",
        "prob_high_quantile": 0.8,
        "prob_low_quantile": 0.2,
        "adv_high_quantile": 0.8,
        "adv_low_quantile": 0.2,
        "high_clip_scale": 1.0,
        "low_clip_scale": 1.0,
    }

    clipped, stats = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )
    expected = torch.tensor([[9.2, 8.0, 6.0, -6.0, -8.0, -9.2]], dtype=torch.float32)
    assert torch.allclose(clipped, expected, atol=1e-6)
    assert stats is not None
    assert stats["adv_clipfrac_pos"].item() == pytest.approx(1 / 6, rel=1e-6)
    assert stats["adv_clipfrac_neg"].item() == pytest.approx(1 / 6, rel=1e-6)


def test_quantile_clip_splits_pos_neg_advantages():
    """正负优势各自分位裁剪，避免混排影响阈值。"""
    advantages = torch.tensor([[5.0, 4.0, -0.1, -10.0]], dtype=torch.float32)
    probs = torch.tensor([[0.9, 0.89, 0.88, 0.87]], dtype=torch.float32)
    log_prob = torch.log(probs)
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "mode": "quantile",
        "prob_high_quantile": 0.5,
        "prob_low_quantile": 0.5,
        "adv_high_quantile": 0.5,
        "adv_low_quantile": 0.5,
        "high_clip_scale": 1.0,
        "low_clip_scale": 1.0,
    }

    clipped, stats = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )
    expected = torch.tensor([[4.5, 4.0, -0.1, -5.05]], dtype=torch.float32)
    assert torch.allclose(clipped, expected, atol=1e-6)
    assert stats is not None
    assert stats["adv_clipfrac_pos"].item() == pytest.approx(0.25, rel=1e-6)
    assert stats["adv_clipfrac_neg"].item() == pytest.approx(0.25, rel=1e-6)


def test_quantile_clip_high_requires_positive_advantage():
    """高概率但优势为负时不应触发 high 裁剪。"""
    advantages = torch.tensor([[-2.0, -1.0]], dtype=torch.float32)
    probs = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
    log_prob = torch.log(probs)
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "mode": "quantile",
        "prob_high_quantile": 0.5,
        "adv_high_quantile": 0.5,
        "enable_clip_high": True,
        "enable_clip_low": False,
    }
    clipped, stats = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )
    assert torch.allclose(clipped, advantages, atol=1e-6)
    assert stats is not None
    assert stats["adv_clipfrac_pos"].item() == pytest.approx(0.0, abs=1e-6)


def test_quantile_clip_low_requires_negative_advantage():
    """低概率但优势为正时不应触发 low 裁剪。"""
    advantages = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
    probs = torch.tensor([[0.01, 0.02]], dtype=torch.float32)
    log_prob = torch.log(probs)
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "mode": "quantile",
        "prob_low_quantile": 0.5,
        "adv_low_quantile": 0.5,
        "enable_clip_high": False,
        "enable_clip_low": True,
    }
    clipped, stats = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )
    assert torch.allclose(clipped, advantages, atol=1e-6)
    assert stats is not None
    assert stats["adv_clipfrac_neg"].item() == pytest.approx(0.0, abs=1e-6)


def test_apply_advantage_clip_sigmoid_mode_reweights_and_logs_p0():
    """sigmoid 模式按分位数概率平滑重加权正负优势，并输出 p0 日志。"""
    advantages = torch.tensor([[2.0, -2.0, 1.0, -1.0]], dtype=torch.float32)
    probs = torch.tensor([[0.9, 0.1, 0.2, 0.8]], dtype=torch.float32)
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
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )

    # 手工计算预期权重并比对
    expected_p0 = torch.tensor(0.5)
    alpha_pos = torch.tensor(2.0)
    alpha_neg = torch.tensor(3.0)
    f_plus = 2.0 / (1.0 + torch.exp(alpha_pos * (probs - expected_p0)))
    f_minus = 2.0 / (1.0 + torch.exp(-alpha_neg * (probs - expected_p0)))
    expected = advantages * torch.tensor(
        [f_plus[0, 0], f_minus[0, 1], f_plus[0, 2], f_minus[0, 3]],
        dtype=torch.float32,
    )

    assert torch.allclose(clipped, expected, atol=1e-6)
    assert stats is not None
    assert stats["adv_sigmoid_p0"].item() == pytest.approx(expected_p0.item(), rel=1e-6)


def test_apply_advantage_clip_sigmoid_mode_side_disabled_by_default():
    """未设置 alpha_pos/neg 时，sigmoid 模式不生效，返回原始优势且不产生 metrics。"""
    advantages = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
    probs = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    log_prob = torch.log(probs)
    response_mask = torch.ones_like(advantages)
    clip_cfg = {
        "enable": True,
        "mode": "sigmoid",
        # 不提供 sigmoid_alpha_pos/neg，应视为关闭
    }

    clipped, stats = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=True
    )
    assert torch.allclose(clipped, advantages)
    assert stats is None

def test_apply_advantage_clip_sigmoid_entropy_target_scales_alpha():
    """熵目标低于当前熵时关闭放缩，低于目标时按差值缩放 alpha。"""
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

    assert torch.allclose(clipped, expected, atol=1e-6)
    assert stats is not None
    assert stats["adv_sigmoid_alpha_pos"].item() == pytest.approx(alpha_pos_actual.item(), rel=1e-6)
    assert stats["adv_sigmoid_alpha_neg"].item() == pytest.approx(alpha_neg_actual.item(), rel=1e-6)

    clipped_high, stats_high = apply_advantage_clip(
        advantages,
        log_prob,
        response_mask,
        clip_cfg,
        return_clip_metrics=True,
        entropy_current=1.5,
        entropy_target=1.0,
    )
    assert torch.allclose(clipped_high, advantages, atol=1e-6)
    assert stats_high is not None
    assert stats_high["adv_sigmoid_alpha_pos"].item() == pytest.approx(0.0, abs=1e-6)
    assert stats_high["adv_sigmoid_alpha_neg"].item() == pytest.approx(0.0, abs=1e-6)


def test_positional_weight_front_heavy_without_clip():
    """位置权重启用时，前段优势被放大，后段被缩小，mask 为 0 的位置不变。"""
    advantages = torch.tensor([[1.0, 1.0, 1.0, 1.0]], dtype=torch.float32)
    log_prob = torch.log(torch.full_like(advantages, 0.5))
    response_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.float32)  # 最后一个 token 无效
    clip_cfg = {
        "enable": True,
        "mode": "sigmoid",
        "pos_weight_enable": True,
        "pos_weight_alpha": 2.0,
        # 不设置 sigmoid_alpha_pos/neg，sigmoid 裁剪关闭，仅位置权重生效
    }

    clipped = apply_advantage_clip(
        advantages, log_prob, response_mask, clip_cfg, return_clip_metrics=False
    )

    # 有效长度 3，位置 0/1/2，对应权重应前>1，中~1，后<1，末位 mask=0 保持 1
    w = 2.0 / (1.0 + torch.exp(torch.tensor(2.0) * (torch.tensor([0.0, 0.5, 1.0]) - 0.5)))
    expected = torch.tensor([[w[0], w[1], w[2], 1.0]], dtype=torch.float32)
    assert torch.allclose(clipped, expected, atol=1e-6)
if __name__ == "__main__":
    unittest.main()
