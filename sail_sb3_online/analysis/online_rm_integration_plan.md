# Online Reward Model Integration Plan for `sail_sb3_online`

---

## 1. Understanding of the Current Setup

### 1a. How `sail_sb3_online` Currently Uses the Offline Reward Model

The offline RM (`PrefRewardModel` in `reward_models/pref_rm_eval.py`) is a **static inference-only artifact** loaded once at the start of training. It is never trained or updated during a run. It enters the pipeline through two paths:

**Path 1 — At startup (inside `TeacherBuffer.__init__`):**
- Every expert episode from the `.npz` file is scored: `J = sum(rm.reward(obs, act))`
- These scores are stored in `pref_episodes` as `{"obs": ..., "actions": ..., "J": J}`
- Boltzmann weights are pre-computed over `pref_episodes`: `w_i = softmax(J_i / beta)`
- These weights and episode list never change unless a student is promoted

**Path 2 — At runtime (inside `SAILAdaptiveCallback._on_step()`):**
- When a student episode ends, if `score_source='rm'`, the RM scores the episode: `student_score = sum(rm.reward(obs_ep, act_ep))`
- This score is compared to the promotion threshold (first expert score)

**Path 3 — Used by discriminator training (`_update_discriminator()`):**

The RM score `J` (pre-computed and stored in `pref_episodes`) drives three discriminator-side mechanisms:

| Mechanism | How RM enters | Flag |
|---|---|---|
| `pref_reweight_teacher` | Boltzmann weights `w_i ∝ exp(J_i/β)` multiply expert BCE loss | `--pref_reweight_teacher` |
| `pref_rank_disc` | Pairs sampled by J-rank; disc trained with Bradley-Terry loss | `--pref_rank_disc` |
| `soft_tac` | `y = sign(J_pos - J_neg)` used as discrete label in tanh alignment loss | `--soft_tac` |

**Path 4 — Used by critic training (`train()`):**
- `pref_rank_disc` pairs (teacher pool) are also used for Q-preference ranking loss on the critic (`--qpref`). **This is the path we will NOT touch initially.**

### 1b. What Would Need to Change Conceptually

The offline RM is a frozen oracle. Replacing it with an online RM means:

1. **The RM must be a trainable module** with its own optimizer, not just an inference wrapper.
2. **The RM must be periodically retrained** using newly collected data as the student improves.
3. **Segment data must be accumulated** from both teacher demos and student rollouts into a preference replay buffer.
4. **Preference labels** must come from a source that does not require the RM (bootstrapping problem): true environment return over the segment is the natural oracle.
5. **The `J` scores** stored in `pref_episodes` must be recomputed periodically as the RM updates.
6. **The three RM-consuming paths** (reweighting, pref ranking, soft-tac) can remain architecturally identical — only the source of `J` scores changes from frozen-RM to online-RM.

### 1c. How APEC Trains Its Reward Model (High Level)

APEC's MuJoCo variant (`apec_mjc`) contains the directly relevant implementation:

- **RM architecture**: Ensemble of 3 independent MLPs, each mapping `concat(obs, act)` → scalar reward. Uses LeakyReLU, 3 hidden layers of 256 units.
- **Segment buffer**: Stores trajectory pairs `(sa_t_1, sa_t_2, r_t_1, r_t_2, masks_1, masks_2, pref_label)`. Segment length is configurable (e.g., 50–1000 steps).
- **Label source**: True environment rewards `r_t_1, r_t_2` are summed per segment: `sum_r_1 = sum(r_t_1)`, `sum_r_2 = sum(r_t_2)`. The higher-return segment is labeled as preferred. No human in the loop.
- **Loss**: Bradley-Terry BCE: `L = BCE(sigmoid(R2 - R1), label)` where R = sum of predicted rewards over the segment.
- **Update frequency**: Scheduled iterations (`n_iters`) over the preference buffer, with gradient accumulation.
- **Training**: Offline in APEC (train reward model, then train policy). The online variant for `sail_sb3_online` must interleave these.

### 1d. Which APEC Ideas Are Reusable

| APEC Component | Reusable? | Notes |
|---|---|---|
| Bradley-Terry BCE loss on return differences | **Yes, directly** | Core RM loss; identical math to `pref_rank_disc` in discriminator |
| True-return label assignment (`sum(r_t) > sum(r_t')`) | **Yes, directly** | Clean oracle for online labels without human feedback |
| Fixed-length segment extraction with masks | **Yes, conceptually** | Segment length tunable; masks handle variable-length padding |
| Ensemble RM (3 MLP members) | **Yes, useful** | Gives uncertainty estimates; start with single MLP, promote to ensemble |
| Segment replay buffer with `(sa_1, sa_2, masks, label)` | **Yes, the interface** | Adapt to use SB3-compatible tensors |
| Segment label noise / teacher gamma decay | **No** | Adds complexity without clear benefit for the first version |
| Expert distance-based labels (Wasserstein) | **No** | APEC's DMControl variant; inapplicable here |
| Pixel encoder / CNN trunk | **No** | APEC's DMControl feature is for pixel obs; HalfCheetah is state-based |
| Standalone reward training script (offline) | **No** | We want online interleaved training |
| Teacher-beta Bradley-Terry noisy oracle | **No** | Unnecessary complication in first version |

---

## 2. Exact Design Proposal

### 2a. New Modules / Classes / Files to Create

**File 1: `reward_models/online_pref_rm.py`**

A trainable preference reward model with its own optimizer.

- **Class `OnlinePrefRewardModel(nn.Module)`**
  - Architecture: MLP mapping `concat(obs, act)` → scalar reward. Start with 2 hidden layers of 256 units, Tanh activations (matching the discriminator architecture for consistency).
  - `forward(obs, act)` → per-step rewards `[T, 1]`
  - `predict_return(obs, act, mask)` → scalar return (sum of masked rewards)
  - `reward(obs_np, act_np)` → numpy array `[T]` (drop-in replacement for offline RM interface used in `pref_rm_eval.py`)
  - `update(batch)` → runs one gradient step, returns loss value
  - Optimizer: AdamW, lr configurable (start at 3e-4)
  - No gradient penalty in first version (keep it simple)

**File 2: `datasets/segment_pref_buffer.py`**

A replay buffer storing preference pairs of trajectory segments, drawn from both teacher demos and student rollouts.

- **Class `SegmentPrefBuffer`**
  - Stores pairs: `(obs_1[T, obs_dim], act_1[T, act_dim], ret_1, obs_2[T, act_dim], act_2[T, act_dim], ret_2, label, source_1, source_2)` where `source ∈ {"teacher", "student"}`
  - Fixed `segment_len` (e.g., 50 steps); shorter episodes padded with zeros, mask tracks valid steps
  - `max_pairs` ring buffer (e.g., 5000 pairs)
  - `add_pair(seg1, seg2)`: assigns label via `int(ret_1 < ret_2)` (1 if seg2 is preferred), rejects ties within a configurable margin
  - `sample(batch_size)` → dict of batched tensors
  - `add_from_episode(obs_ep, act_ep, rew_ep, source)`: extracts `n_segments_per_ep` non-overlapping fixed-length segments from a completed episode, then forms pairs immediately
  - `__len__`, `is_ready(min_pairs)` → bool

**File 3: `reward_models/online_rm_manager.py`**

A manager class that owns the online RM, the segment buffer, the training schedule, and the rescoring logic. This is the main interface that `sail.py` and `callbacks.py` interact with.

- **Class `OnlineRMManager`**
  - Owns: `OnlinePrefRewardModel`, `SegmentPrefBuffer`
  - Configuration: `segment_len`, `rm_train_freq` (env steps between RM updates), `rm_gradient_steps`, `rm_batch_size`, `rescore_freq` (how often to rescore `pref_episodes` with updated RM), `min_pairs_before_train`, `rm_lr`, `device`
  - `add_episode(obs_ep, act_ep, rew_ep, source)`: delegates to `SegmentPrefBuffer.add_from_episode()`
  - `maybe_train(global_step)`: checks if `global_step % rm_train_freq == 0` and `buffer.is_ready()`; if so runs `rm_gradient_steps` gradient steps
  - `rescore_pref_episodes(pref_episodes)`: re-runs `reward()` over all episodes in `pref_episodes`, updates `J` values in-place; called with frequency `rescore_freq`
  - `reward(obs_np, act_np)` → delegates to `OnlinePrefRewardModel.reward()`
  - `is_trained` property → bool (True once at least one RM update has occurred)
  - `get_metrics()` → dict of last training loss, buffer size, etc.

### 2b. Existing Files That Need Changes

| File | What Changes |
|---|---|
| `scripts/train_sail.py` | Add CLI flags for online RM; construct `OnlineRMManager`; pass it to `SAIL` and `SAILAdaptiveCallback`; add `--online_rm` flag to gate the entire feature |
| `algorithms/sail.py` | Accept `online_rm_manager` arg; in `_update_discriminator()`, call `manager.maybe_train()`; use `manager.reward()` in place of `pref_rm.reward()` for pref pair scoring once trained; trigger `rescore_pref_episodes` on schedule |
| `datasets/teacher_buffer.py` | The `pref_episodes` list's `J` values must be mutable (they already are dicts, so this is structural); add a method `rescore_all(rm)` that iterates `pref_episodes` and updates each `J` |
| `utils/callbacks.py` | Accept `online_rm_manager`; in `_on_step()`, after episode end, call `manager.add_episode(obs_ep, act_ep, rew_ep, source='student')`; if `manager.is_trained`, use `manager.reward()` for promotion scoring instead of offline RM |

### 2c. Segment Replay Buffer Design

- **Segment extraction from an episode**: Given episode `obs_ep [T, D], act_ep [T, A], rew_ep [T]`, extract `floor(T / segment_len)` non-overlapping contiguous segments. Partial tails are discarded to avoid heavy padding. Each segment has a ground-truth return `ret = sum(rew_ep[start:end])`.
- **Pairing strategy (at add time)**: When a new segment arrives, randomly pair it with an existing segment from the buffer (uniform sampling). This is simpler than deferred pairing and keeps the buffer always ready to train.
- **Label assignment**: `label = 1 if ret_new > ret_existing else 0`. Ties (within `tie_margin`) are discarded — do not add the pair.
- **Buffer layout**: Store as padded tensors `[max_pairs, segment_len, dim]` using a ring-buffer index for efficiency. Avoid storing full episodes; store only fixed-length segments.
- **Teacher vs. student**: Tag each stored segment with `source`. Log teacher/student mix ratio.

### 2d. Teacher and Student Segment Storage

- **Teacher segments**: Extracted once at startup from expert `.npz` data by `OnlineRMManager` initialization. These are added with `source='teacher'` immediately.
- **Student segments**: Added incrementally at episode completion via `SAILAdaptiveCallback._on_step()`.
- **Ratio control**: No hard ratio enforcement in phase 1. Track and log the ratio; enforce balance (e.g., oversample teacher) in a later phase if the RM drifts.

### 2e. Preference Pair Generation

Pairs are generated at add time (eager pairing):

1. Extract segments from new episode
2. For each new segment, sample one existing segment uniformly from the buffer
3. Compute `ret_new` and `ret_sampled`
4. If `|ret_new - ret_sampled| < tie_margin`: discard
5. Otherwise: create pair with label based on which segment has higher return
6. Add pair to ring buffer

This is equivalent to APEC's MuJoCo approach. The resulting `(seg1, seg2, label)` triples are stored directly.

### 2f. Preference Labels Using True Segment Return

Labels come entirely from the environment:

```
ret_1 = sum(rew_ep[start_1 : end_1])   # true environment rewards
ret_2 = sum(rew_ep[start_2 : end_2])
label = int(ret_1 < ret_2)              # 1 if seg2 preferred, 0 if seg1 preferred
```

No reward model is used for labeling. This is the crucial bootstrapping property: the RM learns from a return oracle that is always available and correct.

### 2g. RM Training Schedule During SAIL Training

- **RM update trigger**: every `rm_train_freq` env steps (e.g., 1000–2000 steps), inside `SAIL._store_transition()` or `_update_discriminator()`, after the discriminator update.
- **Wait condition**: RM training only starts once `segment_buffer.is_ready(min_pairs=256)` — i.e., enough pairs to form a batch.
- **Gradient steps per trigger**: `rm_gradient_steps` (e.g., 10–20 steps per trigger, matching `disc_gradient_steps`).
- **Rescoring trigger**: every `rescore_freq` env steps (e.g., 10,000 steps), `pref_episodes` `J` values are recomputed with the updated RM.
- **Warm-start period**: Before `min_pairs` are available, the offline RM (if provided) or uniform weights are used. This prevents unstable early training.

### 2h. How the Trained RM Should Be Queried

The `OnlinePrefRewardModel.reward(obs_np, act_np)` method provides a drop-in replacement for `PrefRewardModel.reward()`. The query interface is identical:
- Input: `obs_np [T, obs_dim]`, `act_np [T, act_dim]`
- Output: `ndarray [T]` of per-step rewards

In `sail.py`, wherever the offline RM is referenced as `pref_rm.reward(...)`, replace with a guard:

```python
active_rm = online_rm_manager if (online_rm_manager is not None and online_rm_manager.is_trained) else pref_rm
J = sum(active_rm.reward(obs_ep, act_ep))
```

This graceful fallback means the system degrades to offline RM behavior when the online RM is not yet trained.

---

## 3. Training Loop Integration Plan

### Step-by-Step Changes to the Training Loop

**1. Initialization (in `train_sail.py:main()`)**
- After loading expert data into `TeacherBuffer`, construct `OnlineRMManager(segment_len, rm_train_freq, rm_gradient_steps, rm_batch_size, obs_dim, act_dim, device)`
- Extract segments from all expert episodes and populate the segment buffer with `manager.add_episode(obs_ep, act_ep, rew_ep, source='teacher')` for each expert episode
- Pass `online_rm_manager` to `SAIL(...)` constructor and to `SAILAdaptiveCallback(...)`
- Log: number of teacher segments added at init, initial buffer size

**2. Rollout Collection (in `SAILAdaptiveCallback._on_step()`)**
- Existing: accumulate transitions in `EpisodeBuffer`
- **Add**: at episode end, call `manager.add_episode(obs_ep, act_ep, rew_ep, source='student')`
- This populates the segment buffer with student segments that are paired against existing segments

**3. Segment Extraction (inside `SegmentPrefBuffer.add_from_episode()`)**
- On each call, slice episode into `floor(T / segment_len)` contiguous segments
- For each segment, compute `ret = sum(rew[start:end])`
- Immediately pair each new segment with a randomly sampled existing segment
- If tie: discard. Otherwise: add pair with label to ring buffer.

**4. Preference Pair Generation (same as above, eager)**
- No separate step — pairs are generated at segment add time
- Pairs stored as `(obs_1, act_1, obs_2, act_2, label)` tensors of shape `[segment_len, D]`

**5. Online RM Updates (inside `SAIL._store_transition()` or `_update_discriminator()`)**
- After discriminator update, call `manager.maybe_train(global_step)`
- Inside `maybe_train`: if `global_step % rm_train_freq == 0` and `buffer.is_ready()`:
  - For `rm_gradient_steps` iterations:
    - Sample a batch of pairs from `SegmentPrefBuffer`
    - Forward: `R1 = sum(rm.forward(obs1, act1) * mask1)`, `R2 = ...`
    - Loss: `BCE(sigmoid(R2 - R1), label.float())`
    - Backward + optimizer step
  - Update `rm_last_loss`, increment `rm_update_count`
- If `global_step % rescore_freq == 0` and `manager.is_trained`:
  - Call `teacher_buffer.rescore_all(manager)`
  - Recompute Boltzmann weights in `teacher_buffer._recompute_pref_weights(beta)`

**6. Using RM Outputs for Teacher Weighting and Discriminator Preference Loss**
- **Teacher weighting** (`pref_reweight_teacher`): No code change needed here — once `pref_episodes` `J` values are updated by `rescore_all()`, the existing Boltzmann weight computation and `sample_batch_weighted()` use the updated scores automatically.
- **Discriminator preference loss** (`pref_rank_disc`): Similarly no change needed — `sample_pref_pairs()` ranks by `J` values in `pref_episodes`. After rescoring, the pair rankings reflect the online RM's preference ordering.
- **Soft-TAC**: `sample_pref_pairs(return_J=True)` returns `pos_J, neg_J` which are the stored RM scores. After rescoring, these are online-RM-derived. The `y = sign(pos_J - neg_J)` label in `compute_soft_tac_loss()` updates accordingly. No code change needed.
- **Guard**: Add `if manager.is_trained` before any of the above paths that depend on RM-derived J values. Until then, either use offline RM scores (if available) or skip that loss component.

**7. Logging and Evaluation**
- Log in `SAIL.train()` drain section: `train/online_rm_loss`, `train/online_rm_updates`, `train/segment_buffer_size`, `train/segment_teacher_student_ratio`
- Log in `_update_discriminator()`: `pref_reweight/online_rm_active` (bool), `pref_reweight/rescore_count`
- Log in callback: `adaptive/online_rm_used_for_scoring` (bool per episode)
- Periodic evaluation: compute Spearman rank correlation between online RM episode scores and GT episode returns for a fixed held-out set of teacher episodes

---

## 4. Recommended Phased Implementation Order

### Phase 1: Inspect and Map Existing Offline RM Usage

**Goal**: Create a complete map of every code location that reads from `pref_rm`, `pref_episodes[i]["J"]`, or `pref_teacher_weights`.

**Why first**: Before touching anything, you need a complete and verified map of all RM consumption points. Missing one during refactoring causes silent bugs (e.g., the scoring in the callback still using the old frozen RM after you've replaced the scoring in the discriminator).

**Actions**:
- Search for `pref_rm`, `pref_episodes`, `pref_teacher_weights`, `.reward(` across all files
- Document: file, line, what value is consumed, what it gates
- Verify the exact tensor/numpy shapes at each consumption point
- This produces the "freeze map" — everything that must remain working during the transition

### Phase 2: Create `OnlinePrefRewardModel` and `SegmentPrefBuffer` in Isolation

**Goal**: Write and unit-test the new modules with no integration into the training loop.

**Why second**: These modules have zero dependencies on `sail.py` or `callbacks.py`. Building them in isolation means you can test correctness (loss decreases, buffer fill/drain logic, segment extraction) before introducing any risk to the training run.

**Actions**:
- Write `reward_models/online_pref_rm.py`: MLP, optimizer, `reward()` interface matching offline RM
- Write `datasets/segment_pref_buffer.py`: ring buffer, segment extraction, eager pairing, label assignment
- Write a standalone test script: feed synthetic episodes with known returns, verify that the RM loss decreases and that the buffer pair ratios are correct
- Verify `reward()` output shape matches `PrefRewardModel.reward()` exactly

### Phase 3: Create `OnlineRMManager` and Wire it Passively into Training

**Goal**: The manager is constructed and receives data (episodes → segments → pairs) during training, but the RM does NOT yet influence any existing loss or score.

**Why third**: This is the "shadow mode" phase. The manager runs its training loop and updates the RM, but nothing downstream uses the result. You can verify RM learning curves without any risk of destabilizing SAIL training.

**Actions**:
- Write `reward_models/online_rm_manager.py`
- In `train_sail.py`: construct manager, add `--online_rm` flag (off by default)
- In `SAILAdaptiveCallback._on_step()`: call `manager.add_episode()` after episode end (adds to buffer only)
- In `SAIL._store_transition()`: call `manager.maybe_train(global_step)` (trains RM only)
- Add all logging (`online_rm_loss`, `segment_buffer_size`, etc.)
- Run a full training job and observe: Does loss decrease? Does the buffer fill correctly? Do teacher/student segment ratios look right? Does RM loss track GT return correlation?

### Phase 4: Connect RM to One Integration Point

**Goal**: Enable one downstream use of the online RM: either teacher weighting or discriminator preference ranking (not both simultaneously).

**Why fourth**: Connecting both at once makes debugging ambiguous. Start with the lowest-risk path: `pref_reweight_teacher` (Boltzmann weights). This path is the softest — incorrect weights produce suboptimal performance, not crashes or divergence.

**Recommended first connection**: teacher weighting (`pref_reweight_teacher`).

**Actions**:
- Add `teacher_buffer.rescore_all(rm)` method that iterates `pref_episodes` and updates `J` in-place
- In `manager.maybe_train()`: after RM updates, on `rescore_freq` schedule, call `rescore_all`; then call `teacher_buffer._recompute_pref_weights(beta)`
- Add guard: only do rescoring when `manager.is_trained`
- Run training with `--pref_reweight_teacher --online_rm` (without `--pref_rm`): verify that Boltzmann weights change over training (early: uniform; later: differentiated)
- Compare performance against baseline run with offline RM
- **Second connection**: once teacher weighting is stable, enable `pref_rank_disc` (preference ranking loss on discriminator). Same mechanism: pairs are drawn from `pref_episodes` ranked by `J`, which now comes from the online RM.

### Phase 5: Test, Validate, and Iterate

**Goal**: Systematic comparison and stability testing.

**Actions**:
- Run ablation: offline RM only vs. online RM only vs. online RM with offline RM warm-start
- Vary `rm_train_freq`, `rm_gradient_steps`, `segment_len`, `rescore_freq`
- Add held-out evaluation: freeze 10 teacher episodes, compute Spearman rank correlation of online RM scores vs. GT returns at each checkpoint
- Test with `--pref_rank_disc` enabled on top of reweighting
- Document failure modes encountered

**Why this order is safest**: Each phase introduces one new dependency and can be validated independently. The fallback at each phase is simply to disable the new feature and return to the previous state. No phase requires breaking existing code — all changes are additive until Phase 4.

---

## 5. File-by-File Action Map

### Files to Inspect First (Before Writing Anything)

| File | Why to Inspect |
|---|---|
| `algorithms/sail.py` | Complete training loop, all RM consumption points in discriminator and critic |
| `datasets/teacher_buffer.py` | `pref_episodes` data structure, `J` mutability, existing `add_episode()` behavior |
| `utils/callbacks.py` | Episode accumulation, promotion logic, where to hook `manager.add_episode()` |
| `scripts/train_sail.py` | All CLI flags, construction order, where online RM manager should be built |
| `reward_models/pref_rm_eval.py` | Exact `reward()` interface signature and return shape to match |
| `datasets/episode_buffer.py` | Episode data layout — confirm `rew_ep` is ground-truth env rewards, not surrogate |

### Files to Inspect in APEC First

| APEC File | Why to Inspect |
|---|---|
| `apec_mjc/model/pref_reward.py` | Full `get_label()`, `train_reward()`, `bt_loss()` implementations to adapt |
| `apec_mjc/utils/replay_buffer.py` | `PreferenceBuffer` interface — segment storage format |
| `apec_mjc/f_div.py` | `bt_loss()` function — exact loss computation to copy/adapt |
| `apec_mjc/train_pbrl.py` | Training schedule and update loop |

### New Files to Create

| New File | Purpose |
|---|---|
| `reward_models/online_pref_rm.py` | Trainable MLP reward model with `reward()` interface |
| `datasets/segment_pref_buffer.py` | Ring buffer for fixed-length trajectory segment pairs |
| `reward_models/online_rm_manager.py` | Manager class: owns RM + buffer, handles training schedule + rescoring |
| `tests/test_online_rm.py` | Unit tests: loss convergence, buffer fill/drain, label correctness |

### Files to Edit

| File | Changes |
|---|---|
| `scripts/train_sail.py` | Add `--online_rm`, `--rm_train_freq`, `--rm_gradient_steps`, `--segment_len`, `--rescore_freq`, `--rm_lr`, `--rm_min_pairs` flags; construct `OnlineRMManager`; pass to `SAIL` and callback |
| `algorithms/sail.py` | Accept `online_rm_manager`; call `manager.maybe_train(global_step)` in `_store_transition()`; call `teacher_buffer.rescore_all(manager)` on schedule; add `manager.is_trained` guard before RM-dependent losses |
| `datasets/teacher_buffer.py` | Add `rescore_all(rm)` method that updates `J` in `pref_episodes` in-place and calls `_recompute_pref_weights()` |
| `utils/callbacks.py` | Accept `online_rm_manager`; call `manager.add_episode()` at episode end |
| `reward_models/adversary.py` | No changes needed — `compute_pref_loss()` and `compute_soft_tac_loss()` already accept generic `J` values |

---

## 6. Validation and Debugging Plan

### What to Test Before Turning the RM "On"

1. **Buffer correctness**: After a synthetic run of N episodes, verify `segment_buffer.__len__()` grows at the expected rate (`floor(T / segment_len)` segments per episode, each paired once).
2. **Label correctness**: Given two segments with known returns `r=10` and `r=2`, verify the stored label is 1 (segment 1 preferred) and not reversed.
3. **Tie filtering**: Two segments with identical returns should not produce a stored pair.
4. **Shape consistency**: `segment_buffer.sample(B)` must return tensors of shape `[B, segment_len, obs_dim]`, `[B, segment_len, act_dim]`, `[B, 1]` — matching what `OnlinePrefRewardModel` expects.
5. **`reward()` interface match**: `online_rm.reward(obs_np, act_np)` must return `ndarray [T]` with the same dtype and shape as `PrefRewardModel.reward()`.
6. **No gradient leakage into SAIL**: Verify that RM backward pass does not affect discriminator or critic parameters (separate optimizer).

### How to Verify the RM Is Actually Learning

- **Primary metric**: Spearman rank correlation between `online_rm.reward(obs_ep, act_ep).sum()` and `true_return = info['episode']['r']` on a held-out set of teacher episodes. Should increase over training. Target: ρ > 0.7 before enabling teacher weighting.
- **Loss curve**: `train/online_rm_loss` should monotonically decrease within each training burst and overall trending down.
- **Pair accuracy**: For a held-out set of pairs with known labels, compute `accuracy = mean(sigmoid(R2 - R1) > 0.5 when label=1)`. Should exceed 60% quickly.
- **Score spread**: Compute `std(J)` across `pref_episodes` after each rescore. An RM that learned nothing produces near-zero spread; a useful RM produces variance that correlates with trajectory quality.

### What Logs/Metrics to Add

| Metric | Location | Meaning |
|---|---|---|
| `train/online_rm_loss` | `SAIL.train()` drain | RM training loss per trigger |
| `train/online_rm_updates` | `SAIL.train()` | Total RM gradient steps taken |
| `train/segment_buffer_size` | `SAIL.train()` | Total pairs in buffer |
| `train/segment_teacher_ratio` | `SAIL.train()` | Fraction of segments from teacher |
| `pref_reweight/online_rm_active` | `_update_discriminator()` | Whether online RM is driving weights |
| `pref_reweight/j_score_std` | `_update_discriminator()` | Spread of J scores (RM usefulness proxy) |
| `pref_reweight/rescore_count` | `_update_discriminator()` | How many times pref_episodes rescored |
| `adaptive/rm_type` | `callbacks.py` | "offline" or "online" per episode |
| `eval/online_rm_return_corr` | Periodic eval callback | Spearman ρ on held-out episodes |

### How to Check Teacher/Student Segment Mixture

- Log `source` tags in `SegmentPrefBuffer`. At sample time, count how many stored pairs are `(teacher, teacher)`, `(student, student)`, `(teacher, student)`.
- Expected early behavior: mostly `(teacher, teacher)` and `(teacher, student)` pairs.
- Expected late behavior: more `(student, student)` pairs as student improves.
- Alert if teacher segments drop below 10% of stored segments — the RM may be learning only from student data.

### How to Detect Failure Modes Early

- **RM not learning**: `online_rm_loss` not decreasing after 50K env steps → check buffer fill rate, check learning rate, check tie margin
- **J scores not differentiating**: `j_score_std ≈ 0` after rescore → RM is outputting near-constant rewards → increase batch size or reduce segment length
- **SAIL destabilized by rescoring**: `disc_loss` spikes after `rescore_count` increments → rescoring too frequently; increase `rescore_freq`
- **Buffer starved**: `segment_buffer_size` growing too slowly → decrease `segment_len` or decrease `min_pairs_before_train`
- **Teacher/student imbalance**: `segment_teacher_ratio > 0.95` after 100K steps → student episodes not contributing → check `add_episode()` call in callback

---

## 7. Risks and Design Cautions

### Non-Stationary Data Distribution

The student policy improves over time, so the segments added to the preference buffer early in training represent a weak policy. The RM trained on those segments may assign low scores to states that the mature student visits. This creates a **distribution mismatch**: the RM learned from early data is asked to rank late-training trajectories.

**Mitigation**: Use a ring buffer with `max_pairs` much smaller than total pairs generated, so old pairs are evicted. This keeps the RM training on recent data. Also, the `rescore_freq` mechanism re-evaluates stored `J` scores with the current RM, which partially corrects for outdated scores.

### Reward Model Drift

As the RM is periodically retrained, its scoring of the same episode can change. If `rescore_freq` is too high (rescoring happens often), the `pref_episodes` J scores change rapidly, causing the Boltzmann weights to oscillate. This can destabilize teacher weighting.

**Mitigation**: Set `rescore_freq` much lower than `rm_train_freq` (e.g., rescore every 10K steps but train RM every 1K steps). Use an exponential moving average of J scores rather than hard replacement (optional, for a later phase).

### Teacher/Student Imbalance

If teacher episodes dominate the segment buffer (common early in training when few student episodes have been collected), the RM learns a teacher-centric reward function that may not generalize to student trajectories. Conversely, if student episodes dominate (common after heavy adaptive promotion), the RM forgets expert-quality representation.

**Mitigation**: Track teacher/student ratio explicitly (metric: `segment_teacher_ratio`). If the ratio falls below a configurable floor (e.g., 0.2), temporarily skip adding new student segments or oversample from teacher segments during RM training batches. Do not enforce this in phase 4; just log and monitor.

### Noisy Pair Generation

True environment return is a noisy estimate of trajectory quality, especially for short segments. A segment of length 50 in a stochastic environment may have high variance in return even for identical policies. This produces uninformative or contradictory pairs.

**Mitigation**: Increase `tie_margin` to filter out pairs with small return differences. A margin of 5–10% of the expected per-segment return range is a safe starting point. Log the pair rejection rate; if >50% of pairs are rejected, reduce the margin or increase segment length.

### Online RM Destabilizing AIL Training

The discriminator's training signal is the core of SAIL. If the online RM's scores change the teacher weighting dramatically (e.g., suddenly upweighting poor-quality teacher episodes due to a misfit RM), the discriminator loss can spike and the policy can collapse.

**Mitigation**:
1. The fallback guard (`active_rm = online_rm if trained else offline_rm`) means a poorly trained RM does not activate.
2. Gate rescoring behind a minimum quality check: only call `rescore_all()` if `online_rm_return_corr > 0.5`.
3. In phase 4, only enable one of the three RM-dependent paths at a time.
4. Keep the offline RM loaded in parallel as a monitoring baseline — compare `J_offline` vs `J_online` at each rescore to catch dramatic deviations.

### Why Critic-Side Integration (QPREF) Should Be Avoided Initially

QPREF backpropagates the online RM's preference labels directly into the critic's Q-function via a Bradley-Terry loss on trajectory-mean-Q values. This creates a **bi-directional coupling**: the RM influences the Q-function, and the Q-function is used for policy improvement, which generates new rollouts, which train the RM. This feedback loop can:

1. **Amplify RM errors**: An RM that slightly overvalues a class of trajectories will push the Q-function to also overvalue them, causing the policy to collect more of those trajectories, reinforcing the RM's bias.
2. **Interfere with TD learning**: The QPREF loss adds an auxiliary ranking term to the Bellman update. If the RM ranks poorly, it actively degrades the Q-function's TD accuracy.
3. **Make debugging intractable**: If QPREF is failing, it appears as critic divergence, which is indistinguishable from reward function issues or Bellman instability.

The teacher weighting and discriminator preference loss paths are much safer: they influence which data the discriminator trains on (teacher weighting) or add an auxiliary loss to the discriminator (pref ranking), but they do not touch the Bellman backup or the critic's value estimates. The policy is insulated from RM noise through the discriminator's own training stability.

---

## Executive Summary

`sail_sb3_online` currently uses a frozen, pre-trained preference RM to score trajectory episodes, driving teacher Boltzmann reweighting, discriminator preference ranking, and Soft-TAC alignment. The online RM project replaces this frozen oracle with a trainable RM that learns incrementally during SAIL training using true environment returns as preference labels over fixed-length trajectory segments. APEC's MuJoCo implementation provides directly reusable ideas: Bradley-Terry BCE loss on return differences, fixed-length segment extraction with masks, and true-return labeling. The implementation requires three new files (`online_pref_rm.py`, `segment_pref_buffer.py`, `online_rm_manager.py`) and targeted edits to four existing files. The RM does not interact with critic or Q-value learning in the first version.

## Recommended First Milestone

**End of Phase 3**: The online RM trains passively during a full SAIL run on HalfCheetah. `train/online_rm_loss` is logged and decreasing. `eval/online_rm_return_corr` reaches ρ > 0.5 on held-out teacher episodes by 300K env steps. The segment buffer has stable teacher/student ratio. SAIL performance is unchanged from baseline (because the online RM is not yet driving any loss).

## Recommended First Code Change After Planning

**Create `datasets/segment_pref_buffer.py`** with `SegmentPrefBuffer`: ring buffer, `add_from_episode()`, `sample()`, `is_ready()`. Write a standalone test that feeds 100 synthetic episodes with known returns and verifies that (a) buffer fills at the correct rate, (b) labels are correct, (c) ties are filtered, and (d) `sample()` returns tensors of the correct shape. This is the foundation all other components depend on, has zero coupling to the training loop, and can be fully validated before touching `sail.py` or `callbacks.py`.
