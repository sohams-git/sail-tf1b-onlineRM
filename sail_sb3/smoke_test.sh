#!/bin/bash
# Smoke tests for SAIL variants
# Tests that each mode initializes and runs without crashing

REPO_ROOT=/nfs/turbo/umd-sabymath/Soham/sail-tf1b-onlineRM
PYTHON_BIN=/home/sohams/miniconda3/envs/sail_sb3_env/bin/python
EXPERT_DATA=${REPO_ROOT}/teacher_dataset/expert_data_no_img_HalfCheetah_scores_5600_episodes_4.npz
PREF_RM=/nfs/turbo/umd-sridas/Soham/BPref/runs/halfcheetah/pebble_oracle_b500_seg50_disa_s0/exports/reward_model_ts.pt

cd "${REPO_ROOT}" || exit 1

echo "========================================"
echo "SAIL Smoke Tests"
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version
echo "========================================"

# Test 1: Vanilla SAIL
echo ""
echo "TEST 1: Vanilla SAIL (5000 steps)"
echo "----------------------------------------"
PYTHONPATH="${REPO_ROOT}" \
timeout 120 "$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 5000 \
  --learning_starts 1000 \
  --seed 0
TEST1_EXIT=$?
if [ $TEST1_EXIT -eq 0 ]; then
    echo "✓ TEST 1 PASSED"
else
    echo "✗ TEST 1 FAILED (exit code: $TEST1_EXIT)"
fi

# Test 2: SAIL + Pref Ranking
echo ""
echo "TEST 2: SAIL + Pref Ranking (5000 steps)"
echo "----------------------------------------"
PYTHONPATH="${REPO_ROOT}" \
timeout 120 "$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 5000 \
  --learning_starts 1000 \
  --pref_rank_disc \
  --pref_rm "${PREF_RM}" \
  --pref_rank_weight 0.1 \
  --pref_rank_batch_size 16 \
  --seed 0
TEST2_EXIT=$?
if [ $TEST2_EXIT -eq 0 ]; then
    echo "✓ TEST 2 PASSED"
else
    echo "✗ TEST 2 FAILED (exit code: $TEST2_EXIT)"
fi

# Test 3: Adaptive SAIL
echo ""
echo "TEST 3: Adaptive SAIL (5000 steps)"
echo "----------------------------------------"
PYTHONPATH="${REPO_ROOT}" \
timeout 120 "$PYTHON_BIN" sail_sb3/scripts/train_sail.py \
  --env HalfCheetah-v2 \
  --expert_data "${EXPERT_DATA}" \
  --total_timesteps 5000 \
  --learning_starts 1000 \
  --adaptive \
  --seed 0
TEST3_EXIT=$?
if [ $TEST3_EXIT -eq 0 ]; then
    echo "✓ TEST 3 PASSED"
else
    echo "✗ TEST 3 FAILED (exit code: $TEST3_EXIT)"
fi

# Summary
echo ""
echo "========================================"
echo "SMOKE TEST SUMMARY"
echo "========================================"
echo "Vanilla SAIL:        $([ $TEST1_EXIT -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "SAIL + Pref Ranking: $([ $TEST2_EXIT -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "Adaptive SAIL:       $([ $TEST3_EXIT -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "========================================"

# Exit with failure if any test failed
if [ $TEST1_EXIT -ne 0 ] || [ $TEST2_EXIT -ne 0 ] || [ $TEST3_EXIT -ne 0 ]; then
    exit 1
fi
exit 0
