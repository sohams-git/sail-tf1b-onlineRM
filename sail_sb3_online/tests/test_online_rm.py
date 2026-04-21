"""
Unit tests for OnlinePrefRewardModel (reward_models/online_pref_rm.py).

Run from sail-tf1b-onlineRM directory:
    python sail_sb3_online/tests/test_online_rm.py
"""

import sys
import os

_TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
_ONLINE_ROOT = os.path.dirname(_TESTS_DIR)
if _ONLINE_ROOT not in sys.path:
    sys.path.insert(0, _ONLINE_ROOT)

import numpy as np
import torch

from datasets.segment_pref_buffer import SegmentStore, SOURCE_TEACHER, SOURCE_STUDENT
from reward_models.online_pref_rm import OnlinePrefRewardModel


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

OBS_DIM = 18
ACT_DIM = 6
SEG_LEN = 50


def make_rm(obs_dim=OBS_DIM, act_dim=ACT_DIM):
    return OnlinePrefRewardModel(obs_dim, act_dim, hidden_size=64,
                                  lr=3e-4, device=torch.device("cpu"))


def make_batch_with_signal(batch_size=64, segment_len=SEG_LEN,
                            obs_dim=OBS_DIM, act_dim=ACT_DIM):
    """
    Synthetic batch where segment 2 always has higher true return.
    label = 1 for all pairs.
    The RM should be able to learn this after enough updates.
    """
    rng = np.random.RandomState(42)

    # Segment 1: obs and act are random, ret = low_val
    obs1 = rng.randn(batch_size, segment_len, obs_dim).astype(np.float32)
    act1 = rng.randn(batch_size, segment_len, act_dim).astype(np.float32) * 0.1
    mask1 = np.ones((batch_size, segment_len), dtype=np.float32)

    # Segment 2: obs and act are random but scaled higher, ret = high_val
    obs2 = rng.randn(batch_size, segment_len, obs_dim).astype(np.float32)
    act2 = rng.randn(batch_size, segment_len, act_dim).astype(np.float32) * 10.0
    mask2 = np.ones((batch_size, segment_len), dtype=np.float32)

    label = np.ones((batch_size, 1), dtype=np.float32)

    return {
        'obs1':  torch.tensor(obs1),
        'act1':  torch.tensor(act1),
        'mask1': torch.tensor(mask1),
        'obs2':  torch.tensor(obs2),
        'act2':  torch.tensor(act2),
        'mask2': torch.tensor(mask2),
        'label': torch.tensor(label),
        'tie_count': 0,
    }


# -----------------------------------------------------------------------
# Test 1: reward() interface — output shape and dtype
# -----------------------------------------------------------------------

def test_reward_interface():
    rm = make_rm()
    T = 200
    obs_np = np.random.randn(T, OBS_DIM).astype(np.float32)
    act_np = np.random.randn(T, ACT_DIM).astype(np.float32)

    r = rm.reward(obs_np, act_np)

    assert isinstance(r, np.ndarray), f"Expected ndarray, got {type(r)}"
    assert r.shape == (T,), f"Expected shape ({T},), got {r.shape}"
    assert r.dtype == np.float32, f"Expected float32, got {r.dtype}"

    print(f"  PASS  test_reward_interface: shape={r.shape}, dtype={r.dtype}")


# -----------------------------------------------------------------------
# Test 2: forward output shape
# -----------------------------------------------------------------------

def test_forward_shapes():
    rm = make_rm()
    B, L = 16, SEG_LEN
    obs = torch.randn(B, L, OBS_DIM)
    act = torch.randn(B, L, ACT_DIM)

    r = rm(obs, act)
    assert r.shape == (B, L, 1), f"Expected ({B},{L},1), got {r.shape}"
    print(f"  PASS  test_forward_shapes: output shape ({B},{L},1) correct")


# -----------------------------------------------------------------------
# Test 3: loss decreases over training iterations
# -----------------------------------------------------------------------

def test_loss_decreases():
    """RM should lower its loss on a fixed consistent batch over 100 steps."""
    rm = make_rm()
    batch = make_batch_with_signal(batch_size=128)

    # Record initial loss
    initial_loss = rm.update(batch)

    # Train for 100 steps on same batch
    for _ in range(99):
        rm.update(batch)

    final_loss = rm.update(batch)

    assert final_loss < initial_loss, \
        f"Loss should decrease: initial={initial_loss:.4f}, final={final_loss:.4f}"

    print(f"  PASS  test_loss_decreases: {initial_loss:.4f} -> {final_loss:.4f}")


# -----------------------------------------------------------------------
# Test 4: accuracy improves to > 0.6 after training
# -----------------------------------------------------------------------

def test_accuracy_improves():
    """After 200 updates on a consistent signal, accuracy should exceed 0.6."""
    rm = make_rm()
    batch = make_batch_with_signal(batch_size=128)

    acc_before = rm.evaluate_pairs(batch)

    for _ in range(200):
        rm.update(batch)

    acc_after = rm.evaluate_pairs(batch)

    assert acc_after > 0.6, \
        f"Accuracy should exceed 0.6 after training: {acc_after:.3f}"
    assert acc_after > acc_before, \
        f"Accuracy should improve: before={acc_before:.3f} after={acc_after:.3f}"

    print(f"  PASS  test_accuracy_improves: {acc_before:.3f} -> {acc_after:.3f}")


# -----------------------------------------------------------------------
# Test 5: no gradient leakage to external tensors
# -----------------------------------------------------------------------

def test_gradient_isolation():
    """RM optimizer step must not affect tensors outside the RM."""
    rm = make_rm()

    # External tensor that should NOT be touched
    external = torch.nn.Linear(10, 10)
    external_params_before = [p.clone() for p in external.parameters()]

    batch = make_batch_with_signal(batch_size=32)
    rm.update(batch)

    for p_before, p_after in zip(external_params_before, external.parameters()):
        assert torch.allclose(p_before, p_after), \
            "External parameters changed after RM update!"

    print(f"  PASS  test_gradient_isolation")


# -----------------------------------------------------------------------
# Test 6: update_count increments correctly
# -----------------------------------------------------------------------

def test_update_count():
    rm = make_rm()
    assert rm._update_count == 0

    batch = make_batch_with_signal(batch_size=16)
    for i in range(5):
        rm.update(batch)
        assert rm._update_count == i + 1, \
            f"Expected count={i+1}, got {rm._update_count}"

    print(f"  PASS  test_update_count: {rm._update_count} updates tracked")


# -----------------------------------------------------------------------
# Test 7: reward() output is deterministic (no-grad, eval mode not needed)
# -----------------------------------------------------------------------

def test_reward_determinism():
    rm = make_rm()
    obs_np = np.random.randn(100, OBS_DIM).astype(np.float32)
    act_np = np.random.randn(100, ACT_DIM).astype(np.float32)

    r1 = rm.reward(obs_np, act_np)
    r2 = rm.reward(obs_np, act_np)

    assert np.allclose(r1, r2), "reward() must be deterministic on same inputs"
    print(f"  PASS  test_reward_determinism")


# -----------------------------------------------------------------------
# Test 8: integration with SegmentStore
# -----------------------------------------------------------------------

def test_integration_with_store():
    """RM can update on batches drawn directly from SegmentStore."""
    store = SegmentStore(max_segments=500, segment_len=SEG_LEN,
                         obs_dim=OBS_DIM, act_dim=ACT_DIM)
    rm = make_rm()

    rng = np.random.RandomState(0)

    # Add teacher episodes with high returns and student episodes with low returns
    for i in range(20):
        T = SEG_LEN * 5
        obs = rng.randn(T, OBS_DIM).astype(np.float32)
        act = rng.randn(T, ACT_DIM).astype(np.float32)
        rew_teacher = np.full(T, 5.0, dtype=np.float32)   # high
        rew_student = np.full(T, 1.0, dtype=np.float32)   # low
        store.add_from_episode(obs, act, rew_teacher, SOURCE_TEACHER)
        store.add_from_episode(obs, act, rew_student, SOURCE_STUDENT)

    assert store.is_ready(min_segments=64), f"Store not ready: {len(store)} segments"

    # Run 10 RM updates on store batches — should not crash
    losses = []
    for _ in range(10):
        batch = store.sample_pair_batch(batch_size=64, tie_margin=0.1)
        loss = rm.update(batch)
        losses.append(loss)

    # Loss should be finite
    assert all(np.isfinite(l) for l in losses), f"Non-finite losses: {losses}"
    # Loss should generally decrease (or at least not explode)
    assert losses[-1] < 2.0, f"Final loss too high: {losses[-1]:.4f}"

    print(f"  PASS  test_integration_with_store: "
          f"10 updates, loss {losses[0]:.4f}->{losses[-1]:.4f}")


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def run_all():
    tests = [
        test_reward_interface,
        test_forward_shapes,
        test_loss_decreases,
        test_accuracy_improves,
        test_gradient_isolation,
        test_update_count,
        test_reward_determinism,
        test_integration_with_store,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"OnlinePrefRewardModel tests: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
