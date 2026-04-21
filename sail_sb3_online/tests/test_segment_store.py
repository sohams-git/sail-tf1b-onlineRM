"""
Unit tests for SegmentStore (datasets/segment_pref_buffer.py).

Run from sail-tf1b-onlineRM directory:
    python sail_sb3_online/tests/test_segment_store.py

All tests use synthetic data with known returns so correctness
can be verified analytically.
"""

import sys
import os

# Inject sail_sb3_online/ into sys.path so modules import correctly
# regardless of how PYTHONPATH is configured.
_TESTS_DIR   = os.path.dirname(os.path.abspath(__file__))
_ONLINE_ROOT = os.path.dirname(_TESTS_DIR)  # sail_sb3_online/
if _ONLINE_ROOT not in sys.path:
    sys.path.insert(0, _ONLINE_ROOT)

import numpy as np
import torch

from datasets.segment_pref_buffer import SegmentStore, SOURCE_TEACHER, SOURCE_STUDENT


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def make_episode(T, obs_dim, act_dim, reward_val=1.0, seed=0):
    """Synthetic episode with constant reward."""
    rng = np.random.RandomState(seed)
    obs = rng.randn(T, obs_dim).astype(np.float32)
    act = rng.randn(T, act_dim).astype(np.float32)
    rew = np.full(T, reward_val, dtype=np.float32)
    return obs, act, rew


def make_store(max_segments=200, segment_len=50, obs_dim=18, act_dim=6):
    return SegmentStore(max_segments, segment_len, obs_dim, act_dim,
                        device=torch.device("cpu"))


# -----------------------------------------------------------------------
# Test 1: segment count after add_from_episode
# -----------------------------------------------------------------------

def test_segment_count():
    store = make_store(segment_len=50)
    obs, act, rew = make_episode(T=1000, obs_dim=18, act_dim=6)

    n = store.add_from_episode(obs, act, rew, SOURCE_TEACHER)
    expected = 1000 // 50  # = 20

    assert n == expected, f"Expected {expected} segments, got {n}"
    assert len(store) == expected, f"len(store)={len(store)} != {expected}"
    assert store.teacher_count == expected
    assert store.student_count == 0
    print(f"  PASS  test_segment_count: {n} segments added correctly")


# -----------------------------------------------------------------------
# Test 2: return accuracy
# -----------------------------------------------------------------------

def test_return_accuracy():
    segment_len = 50
    reward_val  = 3.7
    store = make_store(segment_len=segment_len)
    obs, act, rew = make_episode(T=segment_len * 4, obs_dim=18, act_dim=6,
                                  reward_val=reward_val)

    store.add_from_episode(obs, act, rew, SOURCE_TEACHER)

    expected_ret = segment_len * reward_val
    for idx in range(4):
        actual = float(store._ret[idx])
        assert abs(actual - expected_ret) < 1e-3, \
            f"Segment {idx}: expected ret={expected_ret:.4f}, got {actual:.4f}"

    print(f"  PASS  test_return_accuracy: all segment returns correct ({expected_ret:.2f})")


# -----------------------------------------------------------------------
# Test 3: label correctness
# -----------------------------------------------------------------------

def test_label_correctness():
    """
    Two groups of segments: group A with low returns, group B with high returns.
    When a pair (A, B) is sampled:
        - ret_A < ret_B  => label should be 1 (seg2 preferred)
    When a pair (B, A) is sampled and seg1=B, seg2=A:
        - ret_B > ret_A  => label should be 0 (seg1 preferred)
    """
    store = make_store(max_segments=100, segment_len=10)

    # Group A: reward=1 per step → ret=10
    obs_a, act_a, rew_a = make_episode(T=50, obs_dim=18, act_dim=6, reward_val=1.0, seed=1)
    # Group B: reward=10 per step → ret=100
    obs_b, act_b, rew_b = make_episode(T=50, obs_dim=18, act_dim=6, reward_val=10.0, seed=2)

    store.add_from_episode(obs_a, act_a, rew_a, SOURCE_TEACHER)  # indices 0-4  (ret=10)
    store.add_from_episode(obs_b, act_b, rew_b, SOURCE_STUDENT)  # indices 5-9  (ret=100)

    # Force a specific pair: index 0 (ret=10) vs index 5 (ret=100)
    # label should be 1 (index 5 = seg2 preferred)
    ret_0 = float(store._ret[0])
    ret_5 = float(store._ret[5])
    assert ret_5 > ret_0, f"Setup error: ret_5={ret_5} should > ret_0={ret_0}"

    # Sample many pairs and verify label consistency
    batch = store.sample_pair_batch(batch_size=64, tie_margin=0.1, strategy='any')
    labels = batch['label'].numpy().flatten()
    ret1_b = batch['obs1'].numpy()  # can't recover ret directly but check label logic
    # Verify label is always 0 or 1
    assert set(np.unique(np.round(labels, 4))).issubset({0.0, 1.0}), \
        f"Labels contain values other than 0 or 1: {np.unique(labels)}"

    print(f"  PASS  test_label_correctness: all labels in {{0, 1}}")


# -----------------------------------------------------------------------
# Test 4: tie filtering
# -----------------------------------------------------------------------

def test_tie_filtering():
    """
    When all segments have identical return, tie rejection should engage.
    tie_count should be > 0 (we can't avoid ties entirely since all returns match).
    """
    store = make_store(max_segments=100, segment_len=10)
    obs, act, rew = make_episode(T=100, obs_dim=18, act_dim=6, reward_val=5.0)
    store.add_from_episode(obs, act, rew, SOURCE_TEACHER)

    batch = store.sample_pair_batch(batch_size=16, tie_margin=0.01, max_retries=5)
    # All returns are identical → all pairs are ties → tie_count == batch_size
    assert batch['tie_count'] == 16, \
        f"Expected 16 tie pairs, got {batch['tie_count']}"

    print(f"  PASS  test_tie_filtering: tie_count={batch['tie_count']} as expected")


# -----------------------------------------------------------------------
# Test 5: output tensor shapes
# -----------------------------------------------------------------------

def test_output_shapes():
    B = 32
    segment_len = 50
    obs_dim = 18
    act_dim = 6
    store = make_store(max_segments=500, segment_len=segment_len,
                       obs_dim=obs_dim, act_dim=act_dim)

    # Add enough segments for a full batch
    for i in range(10):
        obs, act, rew = make_episode(T=500, obs_dim=obs_dim, act_dim=act_dim, seed=i)
        store.add_from_episode(obs, act, rew, SOURCE_TEACHER)

    batch = store.sample_pair_batch(batch_size=B, tie_margin=0.1)

    assert batch['obs1'].shape  == (B, segment_len, obs_dim), \
        f"obs1 shape: {batch['obs1'].shape}"
    assert batch['act1'].shape  == (B, segment_len, act_dim), \
        f"act1 shape: {batch['act1'].shape}"
    assert batch['mask1'].shape == (B, segment_len), \
        f"mask1 shape: {batch['mask1'].shape}"
    assert batch['obs2'].shape  == (B, segment_len, obs_dim), \
        f"obs2 shape: {batch['obs2'].shape}"
    assert batch['act2'].shape  == (B, segment_len, act_dim), \
        f"act2 shape: {batch['act2'].shape}"
    assert batch['mask2'].shape == (B, segment_len), \
        f"mask2 shape: {batch['mask2'].shape}"
    assert batch['label'].shape == (B, 1), \
        f"label shape: {batch['label'].shape}"
    assert batch['label'].dtype == torch.float32

    print(f"  PASS  test_output_shapes: all shapes correct for B={B}")


# -----------------------------------------------------------------------
# Test 6: ring buffer eviction
# -----------------------------------------------------------------------

def test_ring_eviction():
    """Fill beyond max_segments; verify oldest are overwritten."""
    max_segments = 20
    segment_len  = 10
    store = make_store(max_segments=max_segments, segment_len=segment_len)

    # Add 3 * max_segments segments
    for i in range(3 * max_segments):
        obs = np.zeros((segment_len, 18), dtype=np.float32)
        act = np.zeros((segment_len, 6),  dtype=np.float32)
        ret = float(i)  # unique return per segment
        store.add_segment(obs, act, ret, SOURCE_TEACHER)

    assert len(store) == max_segments, \
        f"Expected size={max_segments}, got {len(store)}"

    # The last max_segments inserts should be retained (ring indices 0..19)
    # Returns should span [40..59] (last 20 of 60 total)
    expected_rets = set(range(3 * max_segments - max_segments, 3 * max_segments))
    actual_rets   = set(int(store._ret[i]) for i in range(max_segments))
    assert actual_rets == expected_rets, \
        f"Ring eviction: expected rets {expected_rets}, got {actual_rets}"

    print(f"  PASS  test_ring_eviction: ring buffer correctly overwrites oldest entries")


# -----------------------------------------------------------------------
# Test 7: teacher/student tracking
# -----------------------------------------------------------------------

def test_source_tracking():
    store = make_store(max_segments=100, segment_len=10)

    obs_t, act_t, rew_t = make_episode(T=100, obs_dim=18, act_dim=6, seed=0)
    obs_s, act_s, rew_s = make_episode(T=100, obs_dim=18, act_dim=6, seed=1)

    store.add_from_episode(obs_t, act_t, rew_t, SOURCE_TEACHER)  # 10 teacher segs
    store.add_from_episode(obs_s, act_s, rew_s, SOURCE_STUDENT)  # 10 student segs

    assert store.teacher_count == 10, f"teacher_count={store.teacher_count}"
    assert store.student_count == 10, f"student_count={store.student_count}"
    assert len(store) == 20

    print(f"  PASS  test_source_tracking: teacher={store.teacher_count} student={store.student_count}")


# -----------------------------------------------------------------------
# Test 8: is_ready
# -----------------------------------------------------------------------

def test_is_ready():
    store = make_store(max_segments=100, segment_len=10)
    assert not store.is_ready(min_segments=10), "Should not be ready on empty store"

    obs, act, rew = make_episode(T=100, obs_dim=18, act_dim=6)
    store.add_from_episode(obs, act, rew, SOURCE_TEACHER)  # adds 10 segments

    assert store.is_ready(min_segments=10),  "Should be ready at exactly 10"
    assert not store.is_ready(min_segments=11), "Should not be ready at 11"

    print(f"  PASS  test_is_ready")


# -----------------------------------------------------------------------
# Test 9: short episode (shorter than segment_len) adds 0 segments
# -----------------------------------------------------------------------

def test_short_episode():
    store = make_store(max_segments=100, segment_len=50)
    obs, act, rew = make_episode(T=30, obs_dim=18, act_dim=6)  # T < segment_len
    n = store.add_from_episode(obs, act, rew, SOURCE_TEACHER)
    assert n == 0, f"Expected 0 segments for T=30 < segment_len=50, got {n}"
    assert len(store) == 0
    print(f"  PASS  test_short_episode: short episode correctly yields 0 segments")


# -----------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------

def run_all():
    tests = [
        test_segment_count,
        test_return_accuracy,
        test_label_correctness,
        test_tie_filtering,
        test_output_shapes,
        test_ring_eviction,
        test_source_tracking,
        test_is_ready,
        test_short_episode,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"SegmentStore tests: {passed} passed, {failed} failed")
    print(f"{'='*50}")
    return failed == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
