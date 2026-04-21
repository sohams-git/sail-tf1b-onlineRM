# Online Reward Model Integration Plan — Revised (v2)

All six decisions from the design review are locked in and reflected throughout.
This document supersedes `online_rm_integration_plan.md`.

---

## Locked Decisions Summary

| # | Decision | Status |
|---|---|---|
| 1 | Buffer stores individual segments; pairs sampled dynamically at RM train time | Locked |
| 2 | No offline RM fallback; pure online pipeline only | Locked |
| 3 | Teacher demos segmented once at startup; segments and returns cached | Locked |
| 4 | Labels = raw undiscounted `sum(r_t)` over segment; no discounting, no normalization | Locked |
| 5 | Conservative RM activation gating before any rescoring or teacher reweighting | Locked |
| 6 | First experiment: online RM drives teacher weighting only; all other RM paths off | Locked |

---

## 1. Final Architecture Choice

The online RM is a **single trainable MLP** (not an ensemble in v1) with the following properties:

- **Input**: `concat(obs_t, act_t)` for each timestep — identical input format to the discriminator (`Adversary`)
- **Architecture**: `Linear(obs_dim + act_dim, 256) → Tanh → Linear(256, 256) → Tanh → Linear(256, 1)` — matching the discriminator's hidden structure for consistency and ease of debugging
- **Output**: per-step scalar reward `r_t ∈ ℝ`, shape `[T, 1]`
- **Segment return**: `R = sum(r_t for t in segment)` — no discounting, matching the label definition
- **Optimizer**: AdamW, lr=3e-4, weight_decay=1e-4
- **Loss**: Bradley-Terry BCE — `L = BCE(sigmoid(R2 - R1), label.float())` where `label = 1` if segment 2 has higher true return
- **No gradient penalty** in v1 — keep training simple and stable

The RM is owned and managed exclusively by `OnlineRMManager`. No other module holds a reference to the raw `OnlinePrefRewardModel`.

### Why single MLP, not ensemble

Ensemble uncertainty is useful for active querying (APEC uses it to decide which pairs to label). In this setup, labels come from the environment automatically — there is no query budget. Ensemble disagreement would add complexity without a clear use case in v1. Promote to ensemble in v2 if needed for stability diagnosis.

---

## 2. Final Buffer Design — Option B: Segment Store with Dynamic Pairing

### Decision rationale

**Option A (prebuilt pairs)**: pairs are fixed at insert time. The pairing decision is irreversible. No control over teacher/student mix at training time. Cannot later support different pair sampling strategies without discarding existing buffer state.

**Option B (segment store, dynamic pairing)**: segments are stored individually. Pairs are sampled fresh on every RM training batch. The pairing strategy is a policy applied at sample time, not at insert time. This is strictly more flexible and matches how SB3's `ReplayBuffer` works conceptually.

**Chosen: Option B.** The buffer is a segment store. Pairing happens during RM training.

### `SegmentStore` design

```
SegmentStore
  obs_buf      [max_segments, segment_len, obs_dim]    float32
  act_buf      [max_segments, segment_len, act_dim]    float32
  ret_buf      [max_segments]                          float32   (raw undiscounted sum)
  mask_buf     [max_segments, segment_len]             float32   (1 = valid, 0 = pad)
  source_buf   [max_segments]                          str/int   (0=teacher, 1=student)
  ptr          int                                     ring pointer
  size         int                                     current fill level
  max_segments int                                     capacity
```

**Key methods:**
- `add_segment(obs, act, ret, mask, source)`: writes one fixed-length segment into the ring at `ptr`, advances `ptr`, increments `size`
- `add_from_episode(obs_ep, act_ep, rew_ep, source)`: slices episode into `floor(T / segment_len)` contiguous non-overlapping segments; computes `ret = sum(rew_ep[start:end])` for each; pads any short final tail and discards it; calls `add_segment` for each
- `sample_pair_batch(batch_size, strategy='mixed')`: samples `batch_size` pairs by drawing two indices independently (see pairing strategy below); returns batched tensors
- `is_ready(min_segments)` → bool
- `teacher_count`, `student_count` → int properties
- `__len__` → current size

### Pairing strategy at sample time

`sample_pair_batch` accepts a `strategy` argument:

| Strategy | How indices are drawn | Use case |
|---|---|---|
| `'any'` | Both indices drawn uniformly from full buffer | Default; maximizes diversity |
| `'teacher_student'` | Index 1 from teacher segments, index 2 from student segments | Enforces cross-source pairs |
| `'same_source'` | Both from same source, randomly chosen | Useful for purity checks |

In v1, use `strategy='any'`. The strategy is a single argument to `sample_pair_batch` — switching it requires no structural changes.

**Label assignment at sample time** (not at insert time):
```python
label = (ret_2 > ret_1).float()   # 1 if segment 2 is preferred
```
Tie handling: if `|ret_1 - ret_2| < tie_margin`, this pair is re-sampled. Implement as a rejection loop with a max-retry limit (e.g., 10 retries). If max retries exceeded, emit the pair anyway and log it — do not silently stall RM training.

### Why dynamic label assignment is safe

Labels are deterministic given the two segments' cached `ret` values. There is no stochasticity in label assignment — the same pair always gets the same label. Dynamic pairing merely selects which two segments to compare, and the label follows immediately. There is no risk of label inconsistency.

---

## 3. Teacher Segmentation — Static, One-Time at Startup

Teacher demos are fixed for the entire run. There is no reason to re-segment them.

### Startup procedure (inside `OnlineRMManager.__init__`)

1. Accept `expert_obs [N, obs_dim]`, `expert_act [N, act_dim]`, `expert_rew [N]`, `expert_dones [N]` — the same arrays already loaded by `TeacherBuffer` from the `.npz` file
2. Reconstruct episode boundaries from `expert_dones` (or `episode_starts`)
3. For each expert episode, call `segment_store.add_from_episode(obs_ep, act_ep, rew_ep, source='teacher')`
4. Log: number of teacher episodes processed, number of teacher segments added, total teacher return distribution (min/mean/max)
5. Cache `teacher_segment_count = segment_store.teacher_count` — used later to monitor teacher/student ratio

**This happens once, before any training step.** Teacher segments are never re-added, re-segmented, or modified during training.

### Implication for `train_sail.py`

The expert `.npz` data is already loaded into `TeacherBuffer` before `SAIL` is constructed. Pass the raw arrays (not the `TeacherBuffer` object) to `OnlineRMManager.__init__`. This avoids a circular dependency and keeps `OnlineRMManager` independent of `TeacherBuffer`'s internal structure.

---

## 4. Label Definition — Exact Specification

**One single rule, no exceptions in v1:**

```
ret(segment) = sum(r_t  for t in [start, end))
             = r_start + r_{start+1} + ... + r_{end-1}

where r_t = true environment reward at step t
           (from Monitor wrapper, same as info['episode']['r'])
```

Properties:
- **Undiscounted**: no `gamma^t` weighting
- **Unnormalized**: raw environment scale (e.g., HalfCheetah rewards are in the range ~0–10 per step)
- **No RM-derived pseudo labels**: the RM is never used to label its own training data
- **No noise injection**: no teacher-beta, no epsilon-mistake (APEC's noisy oracle model is not used)
- **Segment-level, not episode-level**: return is computed over the fixed-length segment window only, not the full episode

**Where this is computed**: inside `SegmentStore.add_from_episode()`, once per segment, stored in `ret_buf`. Never recomputed. The cached value is used directly for all future pair label assignments.

**Why raw undiscounted return is the right oracle for v1**:
- It is always available without any model
- It is unambiguous — no hyperparameter sensitivity
- It is consistent with how APEC's MuJoCo implementation assigns labels (`get_label()` uses `np.sum(r_t_1, axis=1)`)
- Discounting would deprioritize late-segment rewards, biasing the RM toward short-horizon quality — undesirable for locomotion tasks where the full segment matters

---

## 5. Online RM Activation Policy

This is the most safety-critical part of the design. A noisy early RM that reweights teacher episodes before it has learned anything can corrupt the discriminator's training distribution and destabilize the entire run.

### Three-gate activation model

The online RM is allowed to affect training only when **all three gates are open simultaneously**:

```
Gate 1 (Data gate):    segment_store.size >= min_segments_before_train
                       Recommended default: 500 segments
                       (covers ~25 teacher episodes + first student episodes)

Gate 2 (Training gate): rm_update_count >= min_rm_updates_before_active
                        Recommended default: 50 RM gradient steps
                        (ensures RM has seen enough pairs before scoring)

Gate 3 (Quality gate):  held_out_pair_accuracy >= min_rm_accuracy
                        Recommended default: 0.60
                        (above random chance; measured on held-out teacher pairs)
```

`OnlineRMManager.is_active` → bool, returns `True` only when all three gates are satisfied.

### Held-out pair accuracy — how it is computed

At startup, after teacher segments are added, reserve a fixed held-out set: sample 100 pairs from teacher segments only (both segments are teacher, labels from `ret_buf`). Store these pairs as `self._held_out_pairs` — they are never used for RM training. After every `eval_freq` RM gradient steps (e.g., every 10 updates), compute:

```python
with torch.no_grad():
    R1 = sum(rm(obs1, act1) * mask1)
    R2 = sum(rm(obs2, act2) * mask2)
    pred_label = (R2 > R1).float()
accuracy = mean(pred_label == held_out_label)
```

Log as `train/online_rm_held_out_acc`. Gate 3 opens once this exceeds `min_rm_accuracy=0.60`.

### Rescoring frequency — conservative schedule

Once `is_active=True`, rescoring `pref_episodes` J values is allowed, but must not happen frequently:

```
rescore_freq: every 20,000 env steps (recommended starting value)
```

This is 20× the RM training frequency (assuming `rm_train_freq=1000`). The rationale: the RM changes slowly once trained; frequent rescoring amplifies noise from individual RM updates into the teacher weighting signal.

**Additional rescore guard**: even if the time condition is met, do not rescore if `held_out_pair_accuracy < min_rm_accuracy`. This ensures a degraded RM (e.g., after a batch of noisy segments) cannot corrupt teacher weights.

### Safe fallback when RM is not yet active

When `is_active=False`, the system uses **uniform teacher weights**:
- `pref_reweight_teacher` effectively becomes a uniform sampling (all `w_i = 1/N`)
- No J-score-based ordering
- No RM-driven loss terms

This is equivalent to vanilla SAIL teacher sampling. There is no offline RM involved — just the standard `TeacherBuffer.sample_batch()` path, which is already uniform.

**Important**: the RM is still training in the background during this warmup period (gates 1 and 2 open before gate 3). Shadow training during warmup means the RM is further along when it activates, reducing the risk of a noisy first activation.

### Activation state machine

```
State: WARMUP
  Condition: any gate closed
  Behavior: uniform teacher weights; RM trains passively if gates 1+2 open
  Transition → ACTIVE: all three gates open

State: ACTIVE
  Condition: all gates open
  Behavior: RM scores used for Boltzmann reweighting on rescore schedule
  Transition → WARMUP: if held_out_acc drops below min_rm_accuracy - 0.05 (hysteresis)
```

The hysteresis band on deactivation (`0.05` below threshold) prevents rapid toggling if accuracy fluctuates near the boundary.

---

## 6. First Active Integration Point — Teacher Weighting Only

The first experiment activates exactly one downstream use of the online RM.

### What is ON

| Component | Status | Notes |
|---|---|---|
| `SegmentStore` (teacher + student segments) | ON | Fills from startup |
| `OnlinePrefRewardModel` (RM training) | ON | Trains every `rm_train_freq` steps once data gate opens |
| Teacher Boltzmann reweighting (`pref_reweight_teacher`) | ON (after activation) | Sole downstream use |
| `rescore_all()` on `pref_episodes` | ON (after activation, on schedule) | Drives the weights |

### What is OFF

| Component | Status |
|---|---|
| `pref_rank_disc` (discriminator preference ranking loss) | OFF |
| `soft_tac` (tanh alignment loss) | OFF |
| `qpref` (critic Q-preference ranking loss) | OFF |
| Offline RM loading (`--pref_rm`) | OFF / not used |
| RM-based adaptive promotion scoring | OFF (use GT return for promotion) |

### What "teacher Boltzmann reweighting with online RM" means end-to-end

1. Startup: teacher segments extracted, `SegmentStore` filled with teacher segments
2. Training runs: student episodes added to `SegmentStore`; RM trains every `rm_train_freq` steps
3. Warmup completes: all three gates open, `is_active=True`
4. On `rescore_freq` schedule: `rescore_all()` calls `online_rm.reward(obs_ep, act_ep)` for each episode in `pref_episodes`, updates `J` in-place
5. `teacher_buffer._recompute_pref_weights(beta)` is called immediately after rescore
6. Next discriminator update: `sample_batch_weighted()` uses updated Boltzmann weights — high-J teacher episodes contribute more to the expert BCE loss
7. The discriminator learns to distinguish expert-quality behavior with emphasis on the best teacher demos as judged by the online RM

This is a minimal, clean integration: one data flow change (teacher weights), everything else unchanged.

---

## 7. Revised Training Loop Integration Plan

### Step 1 — Initialization (`train_sail.py:main()`)

```
Load expert .npz data
  └─> Load into TeacherBuffer (unchanged)
  └─> Also pass raw arrays to OnlineRMManager.__init__()

OnlineRMManager.__init__():
  └─> Construct SegmentStore(max_segments, segment_len, obs_dim, act_dim)
  └─> Construct OnlinePrefRewardModel(obs_dim, act_dim, hidden=256, lr=3e-4)
  └─> Segment all teacher episodes → add to SegmentStore with source='teacher'
  └─> Build held-out pair set (100 teacher-teacher pairs, fixed)
  └─> Log: n_teacher_episodes, n_teacher_segments, ret_mean/std of teacher segments

SAIL.__init__():
  └─> Accept online_rm_manager argument (None if --online_rm not set)
  └─> Store as self.online_rm_manager

SAILAdaptiveCallback.__init__():
  └─> Accept online_rm_manager argument
  └─> Store as self.online_rm_manager
```

### Step 2 — Per-Episode Student Data Collection (`SAILAdaptiveCallback._on_step()`)

Existing flow (unchanged):
- Accumulate transitions in `EpisodeBuffer`
- At episode end: check promotion condition using GT return (not RM)

New addition at episode end:
```
if self.online_rm_manager is not None:
    self.online_rm_manager.add_student_episode(obs_ep, act_ep, rew_ep)
```

`add_student_episode()` calls `segment_store.add_from_episode(..., source='student')`.

Note: GT return is always used for adaptive promotion decisions in v1 — `score_source='gt'` is the only supported mode when using the online RM pipeline.

### Step 3 — RM Training Trigger (`SAIL._store_transition()`)

After the existing discriminator update trigger, add:
```
if self.online_rm_manager is not None:
    self.online_rm_manager.maybe_train(self.num_timesteps)
```

Inside `maybe_train(global_step)`:
```
if global_step % rm_train_freq != 0: return
if not segment_store.is_ready(min_segments): return

for _ in range(rm_gradient_steps):
    batch = segment_store.sample_pair_batch(rm_batch_size, strategy='any')
    R1 = sum_masked(rm(batch.obs1, batch.act1), batch.mask1)
    R2 = sum_masked(rm(batch.obs2, batch.act2), batch.mask2)
    loss = BCE(sigmoid(R2 - R1), batch.label)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    rm_update_count += 1

if rm_update_count % eval_freq == 0:
    held_out_acc = evaluate_held_out()   # updates self._held_out_acc
    update is_active status

if is_active and global_step % rescore_freq == 0:
    if held_out_acc >= min_rm_accuracy:
        teacher_buffer.rescore_all(self)
        teacher_buffer._recompute_pref_weights(beta)
        rescore_count += 1
```

### Step 4 — Discriminator Update (`SAIL._update_discriminator()`)

No structural change needed. The existing `pref_reweight_teacher` path already reads from `teacher_buffer.pref_teacher_weights` and calls `sample_batch_weighted()`. Once `rescore_all()` has updated those weights, the discriminator automatically uses the online-RM-derived weights on its next update.

The only required change: remove the condition `if pref_rm is not None` that gates this path. Replace with:
```python
use_reweight = (pref_reweight_teacher and
                online_rm_manager is not None and
                online_rm_manager.is_active and
                len(pref_episodes) >= 2)
```

When `is_active=False`, fall through to uniform `sample_batch()`. This is safe and requires no offline RM.

### Step 5 — Logging (drained in `SAIL.train()`)

Add to existing logging drain:
```
train/online_rm_loss              — last RM batch loss
train/online_rm_updates           — total gradient steps taken
train/online_rm_held_out_acc      — held-out pair accuracy
train/online_rm_is_active         — 0 or 1
train/segment_store_size          — total segments in store
train/segment_teacher_count       — teacher segments
train/segment_student_count       — student segments
pref_reweight/rescore_count       — number of rescore calls so far
pref_reweight/j_score_std         — std of J values across pref_episodes (after rescore)
pref_reweight/weight_entropy      — entropy of Boltzmann weights (higher = more uniform)
```

---

## 8. Revised File-by-File Action Map

### New Files to Create (in order of implementation)

| File | Phase | Content |
|---|---|---|
| `datasets/segment_pref_buffer.py` | Phase 2 | `SegmentStore` class: ring buffer, `add_from_episode()`, `add_segment()`, `sample_pair_batch()`, `is_ready()`, source tracking |
| `reward_models/online_pref_rm.py` | Phase 2 | `OnlinePrefRewardModel`: MLP, `forward()`, `reward()` (numpy interface), `update(batch)` |
| `reward_models/online_rm_manager.py` | Phase 3 | `OnlineRMManager`: owns store + model, `maybe_train()`, `add_student_episode()`, `is_active`, held-out eval, `rescore_pref_episodes()` |
| `tests/test_segment_store.py` | Phase 2 | Standalone unit test for `SegmentStore` |
| `tests/test_online_rm.py` | Phase 2 | Standalone unit test for `OnlinePrefRewardModel` |

### Existing Files to Edit

| File | Phase | Changes |
|---|---|---|
| `scripts/train_sail.py` | Phase 3 | New flags: `--online_rm`, `--rm_train_freq` (1000), `--rm_gradient_steps` (10), `--segment_len` (50), `--rescore_freq` (20000), `--rm_lr` (3e-4), `--rm_min_segments` (500), `--rm_min_updates` (50), `--rm_min_acc` (0.60), `--rm_tie_margin` (0.5); construct `OnlineRMManager`; pass to `SAIL` and callback |
| `algorithms/sail.py` | Phase 3 | Accept `online_rm_manager`; call `manager.maybe_train()` in `_store_transition()`; update `pref_reweight_teacher` gate condition; add RM metrics to logging drain |
| `datasets/teacher_buffer.py` | Phase 4 | Add `rescore_all(rm)` method: iterates `pref_episodes`, calls `rm.reward(ep['obs'], ep['actions'])`, updates `ep['J']` in-place; calls `_recompute_pref_weights(beta)` |
| `utils/callbacks.py` | Phase 3 | Accept `online_rm_manager`; call `manager.add_student_episode()` at episode end; remove any offline RM reference from promotion scoring |

### Files That Need No Changes

| File | Reason |
|---|---|
| `reward_models/adversary.py` | `compute_pref_loss()` and `compute_soft_tac_loss()` unchanged; those paths are OFF in v1 |
| `reward_models/pref_rm_eval.py` | Still exists as a file but is not instantiated in the new pipeline |
| `datasets/episode_buffer.py` | Unchanged; still accumulates student transitions for the callback |
| `datasets/expert_loader.py` | Unchanged; still parses `.npz` format |

---

## 9. Revised Phased Implementation Order

### Phase 1 — Freeze Map (no code changes)

**Goal**: document every location where offline RM is referenced or where `J` values from `pref_episodes` are consumed.

Grep targets across the codebase:
- `pref_rm`
- `pref_episodes`
- `pref_teacher_weights`
- `.reward(`
- `score_source`
- `rm_expert_scores`

For each hit, record: file, line number, what value is read, what it gates. This is the checklist that must be satisfied before v2 integration is complete. Doing this first prevents missed consumption points during refactoring.

### Phase 2 — Build and Test New Modules in Isolation

**Goal**: `SegmentStore` and `OnlinePrefRewardModel` are fully functional and tested before touching any training code.

**Test for `SegmentStore`**:
1. Feed 50 synthetic episodes of length 1000 with known per-step rewards
2. Verify segment count = `floor(1000 / segment_len) * 50`
3. Verify `ret_buf` values match hand-computed sums
4. Sample 256 pairs; verify all `label` values are `0` or `1`
5. Verify tie rejection works: inject two identical-return segments; confirm they are never selected as a pair (or only after max retries)
6. Verify ring buffer: after filling past `max_segments`, oldest segments are overwritten

**Test for `OnlinePrefRewardModel`**:
1. Create pairs where segment 2 has consistently higher return; verify loss decreases over 100 steps
2. Verify `reward(obs_np, act_np)` returns `ndarray` of shape `[T]`, dtype float32
3. Verify no gradient leaks to external parameters (fresh model with no shared weights)

### Phase 3 — Passive Integration (shadow mode)

**Goal**: manager is wired into the training loop; RM trains; nothing downstream is affected.

Checklist before ending this phase:
- [ ] `train/online_rm_loss` appears in tensorboard and decreases
- [ ] `train/segment_store_size` grows at expected rate
- [ ] `train/segment_teacher_count` is correct (matches expected from expert data)
- [ ] `train/online_rm_held_out_acc` logged every `eval_freq` steps
- [ ] `train/online_rm_is_active` stays `0` for the warmup period, then flips to `1`
- [ ] SAIL performance (episode return, disc loss) is identical to a baseline run without `--online_rm`

The last point is the most important: passive integration must have **zero effect** on training dynamics.

### Phase 4 — Activate Teacher Weighting

**Goal**: once shadow mode is verified, enable the one downstream path.

Enable by adding `--pref_reweight_teacher` to the experiment config (alongside `--online_rm`). All other RM paths remain off.

Checklist before ending this phase:
- [ ] `pref_reweight/online_rm_active` flips to `1` after warmup
- [ ] `pref_reweight/j_score_std` is non-zero after first rescore
- [ ] `pref_reweight/weight_entropy` decreases over time (weights concentrate on better episodes)
- [ ] Discriminator loss remains stable through first rescore event
- [ ] Episode return is not worse than baseline within first 500K steps

### Phase 5 — Validation and Ablation

**Goal**: compare online RM reweighting against baseline; tune hyperparameters.

Experiments:
1. Baseline: vanilla SAIL, no RM (`--online_rm` off, `--pref_reweight_teacher` off)
2. Online RM passive: `--online_rm` on, `--pref_reweight_teacher` off (shadow only)
3. Online RM active: `--online_rm` on, `--pref_reweight_teacher` on
4. Sensitivity: vary `segment_len` (25, 50, 100), `rescore_freq` (10K, 20K, 50K), `rm_min_acc` (0.55, 0.60, 0.65)

Key evaluation metrics per run:
- Episode return at 500K, 1M, 2M env steps
- `eval/online_rm_return_corr`: Spearman ρ between RM scores and GT returns on held-out teacher episodes
- `pref_reweight/weight_entropy`: trajectory of weight concentration over training
- Stability: `disc_loss` variance around rescore events

---

## 10. Validation and Debugging Plan (Revised)

### Tests Before Any RM Activation

| Test | Pass Condition |
|---|---|
| Segment count | `store.size == sum(floor(T_ep / segment_len) for each episode)` |
| Return accuracy | `store.ret_buf[i] == sum(rew_ep[start:end])` within 1e-5 |
| Label correctness | `label == (ret_2 > ret_1)` for all sampled pairs |
| No padding leakage | `ret` only sums over valid steps (mask=1); padding rows (mask=0) contribute 0 |
| Interface match | `online_rm.reward(obs, act).shape == (T,)`, dtype float32 |
| Gradient isolation | RM optimizer step does not change discriminator or critic weights |
| Teacher/student split | `store.teacher_count` matches expected after startup; `store.student_count` grows during training |

### Verifying RM Learning

Primary signal: `train/online_rm_held_out_acc` — should rise from ~0.5 (random) to >0.6 within first 50K env steps.

Secondary signals:
- `train/online_rm_loss` should decrease within each `rm_gradient_steps` burst
- `pref_reweight/j_score_std` after first rescore should be non-zero; compare to zero-std (constant RM) as a null check
- Visual check: print top-5 and bottom-5 J-scored episodes after first rescore; verify the ordering makes intuitive sense (e.g., episodes with high GT return have high J)

### Detecting Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `held_out_acc` stuck at 0.5 | RM not learning; pairs all near tie_margin | Reduce `tie_margin`; increase `rm_gradient_steps` |
| `j_score_std ≈ 0` after rescore | RM outputs near-constant reward | Check input normalization; verify no dead activations |
| `disc_loss` spikes at rescore events | Weights change too abruptly | Increase `rescore_freq`; add weight EMA option |
| `segment_store_size` grows very slowly | Episodes too short relative to `segment_len` | Reduce `segment_len` |
| `weight_entropy` stays high (≈ log N) | RM not differentiating episodes | Increase `rm_gradient_steps`; check `min_rm_accuracy` gate |
| Episode return collapses after RM activation | RM actively harmful; gate opened too early | Raise `min_rm_accuracy`; raise `min_rm_updates` |
| `held_out_acc` drops after rising | Non-stationarity overwhelming RM | Reduce `rm_train_freq` to train more frequently; consider larger buffer |

---

## 11. Risks and Design Cautions (Revised)

### Non-Stationary Data Distribution
Ring buffer eviction (`max_segments`) handles this naturally. Old student segments from the early policy are overwritten by recent segments. Teacher segments are written first and will eventually be partially overwritten as the ring fills — this is acceptable because the RM should generalize from teacher+early-student data to teacher+late-student data organically.

**Recommended `max_segments`**: 10,000 segments. At `segment_len=50` steps per segment, this is 500,000 steps of experience — roughly matching the SB3 replay buffer policy of holding ~1M transitions.

### Reward Model Drift
Conservative rescoring frequency (every 20K steps, gated by quality check) is the primary mitigation. An additional protection: after each rescore, log `mean(|J_new - J_old|)` across `pref_episodes`. If this exceeds a threshold (e.g., 20% of mean `|J|`), flag a warning and optionally skip the weight update for that rescore cycle.

### Teacher/Student Imbalance
Teacher segments are fixed at startup. As the student generates episodes, `student_count` grows and `teacher_count / total` shrinks. This is the desired direction — the RM should eventually learn from mostly student data. Monitor `segment_teacher_ratio`. If it drops below 0.05 after 500K steps, consider freezing student additions temporarily or oversampling teacher segments during RM training (simple: when sampling a pair, with probability `p_teacher` force index 1 to be a teacher segment).

### No Offline RM Fallback — Implication for Early Training
Without an offline RM, the very first teacher weighting done by the online RM is the first time any RM has influenced training. This is safe only because of the three-gate activation policy. The uniform weight fallback during warmup means the discriminator trains identically to vanilla SAIL until the RM is ready. There is no cold-start risk.

### Why QPREF, pref_rank_disc, and soft_tac are deferred

- **pref_rank_disc**: adds a Bradley-Terry loss directly to the discriminator. If the online RM's J scores are noisy, this loss pulls the discriminator in incorrect directions, corrupting the GAIL signal. Teacher weighting is softer — it biases which data enters the BCE loss, but the BCE loss itself is still correct.
- **soft_tac**: uses `y = sign(J_pos - J_neg)` as a hard discrete label for tanh alignment. A single wrong sign (noisy RM) produces a loss that actively pushes the discriminator's ranking in the wrong direction. Very sensitive to RM quality.
- **qpref**: as described in v1, creates a feedback loop between RM and critic that amplifies errors. Should not be attempted until the RM is demonstrably stable over a full 2M step run.

The conservative ordering (reweighting first, then ranking, then alignment, never critic in v1) reflects the sensitivity ranking of these four paths.

---

## Recommended First Coding Step

**Create `datasets/segment_pref_buffer.py`** implementing `SegmentStore` with the following minimal interface:
- `__init__(max_segments, segment_len, obs_dim, act_dim)`
- `add_segment(obs, act, ret, source)` — stores one padded segment
- `add_from_episode(obs_ep, act_ep, rew_ep, source)` — segments + stores a full episode
- `sample_pair_batch(batch_size, tie_margin, strategy='any')` — returns dict of tensors
- `is_ready(min_segments)` → bool
- `teacher_count`, `student_count`, `__len__`

Then immediately write `tests/test_segment_store.py` against it. Do not proceed to `OnlinePrefRewardModel` until all `SegmentStore` tests pass.

This order matters because every other component (`OnlinePrefRewardModel`, `OnlineRMManager`, the training loop hooks) depends on `SegmentStore` producing correctly shaped, correctly labeled batches. A bug here propagates everywhere.

---

## Recommended First Experiment After Implementation

**Configuration for HalfCheetah, 1M env steps:**

```
--env HalfCheetah-v3
--total_timesteps 1000000
--expert_data <path_to_expert_npz>

# Online RM
--online_rm
--segment_len 50
--rm_train_freq 1000
--rm_gradient_steps 10
--rm_lr 3e-4
--rm_min_segments 500
--rm_min_updates 50
--rm_min_acc 0.60
--rm_tie_margin 0.5
--rescore_freq 20000

# Active integration: teacher reweighting only
--pref_reweight_teacher
--pref_beta 1.0

# Everything else: OFF
# (no --pref_rank_disc, no --soft_tac, no --qpref, no --pref_rm)

# Adaptive (use GT return for promotion, not RM)
--adaptive
--score_source gt
```

Run this alongside an identical baseline with `--online_rm` removed (vanilla SAIL + adaptive). The comparison is clean: the only difference is whether teacher weights are driven by the online RM or are uniform.

Expected outcomes to validate:
1. `train/online_rm_is_active` flips to 1 within first 100K steps
2. `pref_reweight/j_score_std` is non-zero after first rescore
3. Episode return at 1M steps is equal or better than baseline (not worse)
4. No disc_loss spikes at rescore events
