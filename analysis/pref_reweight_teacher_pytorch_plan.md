# `--pref-reweight-teacher` PyTorch Implementation Plan

**Date:** 2026-04-07  
**Based on:** TF sail.py (2016 lines), gail/adversary.py (~900 lines), run/train_sail.py (~1175 lines)  
**PyTorch files audited:** sail_sb3/algorithms/sail.py, reward_models/adversary.py, scripts/train_sail.py, datasets/teacher_buffer.py, utils/callbacks.py

---

## 1. TF Behavior Summary

### What `--pref-reweight-teacher` does

It replaces the uniform expert sampling in the discriminator expert batch with a **Boltzmann-weighted episode-level sampling**. Each expert episode gets a weight proportional to `exp(J_phi(τ) / beta)`, where `J_phi(τ) = sum_t R_phi(s_t, a_t)` is the cumulative preference RM score for that episode. Episodes with higher RM score are sampled more often and receive a higher per-sample weight in the discriminator expert loss.

This biases the discriminator toward treating RM-preferred behavior as more expert-like, steering GAIL away from treating all expert transitions uniformly.

### Data structures

Three parallel arrays on the SAIL object (all initialized in `initialize_expert_buffer`, called from `learn()`):

```python
self.pref_teacher_episodes  # list of dicts: {'obs': np.array[T, obs_dim],
                            #                   'acs': np.array[T, act_dim],
                            #                   'J': float}
self.pref_teacher_scores    # np.array[N_eps], J values in insertion order
self.pref_teacher_weights   # np.array[N_eps], softmax(J / beta) in insertion order
```

These are completely separate from `demo_replay_buffer` (the flat transition ring buffer used for vanilla GAIL sampling).

### Episode scoring

`J_phi(τ) = sum_t R_phi(s_t, a_t)` computed by `_ensure_pref_teacher_rm()` which lazy-loads a `PrefRewardModel` from `cfg['pref_rm_path']`. This is the **same RM** used by `pref_rank_disc` and `qpref` — all three share the same RM loader via `_ensure_pref_teacher_rm()`. Scores are RM-based only. GT returns are NOT used for reweighting.

### Weight computation (`_recompute_pref_teacher_weights`, line 664)

```python
scores = np.array([ep['J'] for ep in self.pref_teacher_episodes], dtype=np.float64)
beta = float(self.config.get('pref_beta', 1.0))
s = scores / beta
s = s - s.max()               # numerically stable max-subtraction
w = np.exp(s)
w = w / (w.sum() + 1e-8)      # L1 normalization to sum=1
self.pref_teacher_weights = w
```

**Key properties:**
- Softmax over `J/beta`, NOT over the raw J values
- Max-subtraction before `exp()` for numerical stability
- Denominator has `1e-8` to prevent division by zero
- Result is a probability distribution over N episodes (sums to ~1)
- `beta=1.0` is the default; larger beta → flatter distribution; smaller beta → sharper (more concentrated on high-J episodes)

### Sampling (`_sample_pref_weighted_expert`, line 780)

This is the critical function. Here is the **exact TF behavior**:

```python
for _ in range(B):
    ep_idx = np.random.randint(0, n_eps)    # <-- UNIFORM episode sampling, NOT weighted!
    ep = episodes[ep_idx]
    T = ep['acs'].shape[0]
    t = np.random.randint(0, T)             # uniform timestep within episode

    obs_list.append(ep['obs'][t])
    ac_list.append(ep['acs'][t])
    w_list.append(weights[ep_idx])          # <-- softmax weight attached per sample
```

**Critical observation:** TF samples episodes **uniformly** (`np.random.randint(0, n_eps)`) — NOT proportionally to `weights`. The `weights` array is NOT used as a sampling distribution. Instead, the per-sample weight (`weights[ep_idx]`) is returned alongside the sampled transition and passed to the discriminator loss as a **multiplicative weight on the expert BCE loss**.

This is a **weighted loss** design, not a weighted sampling design. The probability of drawing a sample is uniform over episodes × timesteps. But once drawn, high-J episodes get a higher loss weight (making those samples "count more" in the gradient).

The fallback (when `pref_reweight_teacher=False` or no pref pool): returns uniform `expert_w = ones(B, 1)` and samples from `demo_replay_buffer` uniformly.

### How weights enter the discriminator loss (gail/adversary.py, line 742–756)

```python
sample_expert_loss = F.binary_cross_entropy_with_logits(...)   # per-sample [N, 1]
sample_expert_loss = tf.reshape(sample_expert_loss, (-1, 1))
weights = self.expert_w_ph  # [N, 1] — passed in via feed_dict

if self.use_expert_weights:
    # Weighted mean: sum(w_i * loss_i) / sum(w_i)
    expert_loss = tf.reduce_sum(sample_expert_loss * weights) / (tf.reduce_sum(weights) + 1e-8)
else:
    expert_loss = tf.reduce_mean(sample_expert_loss)
```

**Critical:** The weighted expert loss is a **weighted mean**, not a weighted sum. This normalizes by the total weight so the loss scale stays comparable regardless of weight distribution. This is `sum(w_i * loss_i) / sum(w_i)`, not `mean(w_i * loss_i)`.

The flag `use_expert_weights` on the discriminator must be `True` for weights to matter. This flag is set separately via `cfg['disc_use_expert_weights']` in `setup_model()`. `pref_reweight_teacher=True` implies `disc_use_expert_weights=True` — but they are set via separate config flags.

**Important:** `expert_w_ph` is **always fed** in the TF discriminator call (see line 1219) regardless of whether `pref_reweight_teacher` is True or False. When it is False, `expert_w = ones(B, 1)` is passed. When `disc_use_expert_weights=False`, the weights are received but ignored in the loss computation. This means:

- `pref_reweight_teacher=True` + `disc_use_expert_weights=True`: weights used, non-uniform sampling weighted mean
- `pref_reweight_teacher=True` + `disc_use_expert_weights=False`: weights computed and sampled correctly, but discarded in loss — **effectively no reweighting** despite the flag name
- `pref_reweight_teacher=False`: `expert_w = ones(B,1)`, always uniform, `disc_use_expert_weights` doesn't matter

In practice, you want **both** flags True for the feature to actually work.

### Interaction with adaptive promotion (lines 1574–1608)

When a student episode is promoted (score > `expert_scores[0]`):
1. Transitions added to `demo_replay_buffer` as usual (transition-level ring buffer).
2. **Additionally**, if `pref_reweight_teacher=True` (or `pref_rank_disc=True`, or `qpref=True`), the episode is scored by RM and appended to `pref_teacher_episodes`.
3. Pruning: if `len(pref_teacher_episodes) > pref_max_teacher_trajs` (default 500), low-J episodes are dropped: keep all with `J >= quantile(scores, pref_promote_quantile)` (default 0.75). At least 1 episode (best) is always kept.
4. `_recompute_pref_teacher_weights()` is called immediately after adding/pruning.

This means the pref pool **grows monotonically** with promotions (pruning is quantile-based, not FIFO). The TF transition ring buffer and the pref pool are completely independent growth paths.

### Initial buffer build (`_build_pref_teacher_buffer`, line 683)

Called from `initialize_expert_buffer()`, which in TF is called **before** ring truncation logic. In TF, `demo_replay_buffer` is `ReplayBufferExtend(self.demo_buffer_size)` where `demo_buffer_size = 1000`. All 4000 transitions are loaded then overflow evicts the oldest, leaving only the last 1000 transitions in `demo_replay_buffer`. But `_build_pref_teacher_buffer` takes `demo_obs, demo_actions, demo_dones` from the **full NPZ data** (all 4000 transitions), not from the ring buffer, so all 4 expert episodes are captured in `pref_teacher_episodes`. This is the exact TF behavior.

### Config flags (train_sail.py lines 943–946)

```python
if config['pref_reweight_teacher']:
    config['pref_beta'] = args.pref_beta
    config['pref_promote_quantile'] = args.pref_promote_quantile
    config['pref_max_teacher_trajs'] = args.pref_max_teacher_trajs
```

`pref_beta` is only parsed/stored when `pref_reweight_teacher=True`. It is not stored for `pref_rank_disc` alone.

### Answering the 15 TF questions precisely

| # | Question | TF Answer |
|---|----------|-----------|
| 1 | What does it do? | Weights discriminator expert batch by Boltzmann(J_phi/beta) — episodes with higher RM score count more in expert loss |
| 2 | Stage of training? | Every discriminator update (every 500 env steps), inside `_train_discriminator` → `generate_discriminator_data` → `_sample_pref_weighted_expert` |
| 3 | Expert batches only, or elsewhere too? | **Expert batch only.** Policy batch (from `replay_buffer`) is always uniform. LfD mixing (critic batch) is NOT affected. Pref pair sampling for `pref_rank_disc` is also not affected (uses same `pref_teacher_episodes` pool). |
| 4 | Data structure? | `list[dict]` with `{'obs': np.array, 'acs': np.array, 'J': float}`, alongside `np.array` of scores and weights |
| 5 | How are scores computed? | `J = sum_t R_phi(s_t, a_t)` via offline BPref RM |
| 6 | GT or RM? | **RM only.** No GT leakage. |
| 7 | How are weights formed? | `softmax(J / beta)` with max-subtraction, normalized to sum=1 |
| 8 | Normalized? | Yes — explicitly divided by `w.sum() + 1e-8` |
| 9 | Role of beta? | Temperature: larger beta = flatter (more uniform), smaller beta = sharper (concentrate on top episodes) |
| 10 | How are episodes sampled after weighting? | **Uniformly** by `np.random.randint(0, n_eps)` — NOT proportionally to weights. Weights are attached as per-sample loss multipliers, not as sampling probabilities. |
| 11 | Adaptive promotion interaction? | Promoted episodes are scored by RM and appended to `pref_teacher_episodes`. Weights recomputed immediately. Quantile-based pruning if pool exceeds `pref_max_teacher_trajs`. |
| 12 | When does a promoted episode enter the pool? | Immediately on promotion (inside the `if episode_score > expert_scores[0]` block at line 1574), before `expert_scores` is updated, before `episode_buffer.reset()`. |
| 13 | Episode-level or transition-level weighting? | **Episode-level**: one weight per episode, applied uniformly to all transitions sampled from that episode. |
| 14 | Where in discriminator update? | In `generate_discriminator_data()` → returned as `expert_w` → fed as `expert_w_ph` to discriminator → used in weighted expert loss computation. |
| 15 | Demo ring buffer, separate episode pool, or both? | **Separate episode pool** (`pref_teacher_episodes`) for weighted sampling. Demo ring buffer (`demo_replay_buffer`) used as fallback when `pref_reweight_teacher=False`. When enabled, ring buffer is bypassed entirely for the expert batch. |

---

## 2. Current PyTorch Status

### What already exists

| Component | Location | State |
|-----------|----------|-------|
| `pref_episodes` pool (list of dicts: obs, acs, J) | `TeacherBuffer.pref_episodes` | ✅ Exists — built from ALL 4 expert episodes before ring truncation (SepPool change) |
| RM scorer | `TeacherBuffer.pref_rm` (PrefRewardModel) | ✅ Exists |
| Episode scoring (J computation) | `TeacherBuffer._build_pref_episodes()`, `TeacherBuffer.add_episode()` | ✅ Exists |
| Promoted episode appended to pref pool | `TeacherBuffer.add_episode()` | ✅ Exists |
| Pref pair sampling (for pref_rank_disc) | `TeacherBuffer.sample_pref_pairs()` | ✅ Exists |
| Discriminator expert batch sampling | `SAIL._update_discriminator()` → `teacher_buffer.sample_batch()` | ✅ Exists but **always uniform**, no weights |
| Expert loss computation | `Adversary.compute_loss()` | ✅ Exists but **no weighting**, always `reduce_mean` |
| Weighted expert loss support | `Adversary.compute_loss()` | ❌ Missing — no `expert_w` argument |
| Softmax weight computation over episodes | `TeacherBuffer` | ❌ Missing — `pref_teacher_weights` not computed |
| Weighted episode sampling path | `TeacherBuffer` | ❌ Missing — no `sample_batch_weighted()` |
| `pref_reweight_teacher` flag | `SAIL.__init__`, `train_sail.py` | ❌ Missing |
| `pref_beta` parameter | anywhere | ❌ Missing |
| Promoted-episode weight recomputation | `TeacherBuffer.add_episode()` | ❌ Missing — `add_episode` appends to pref pool but does not recompute weights |
| Quantile-based pref pool pruning | `TeacherBuffer.add_episode()` | ❌ Missing — pool grows unboundedly |

### How the separate pref pool changes the design

**This is the key architectural insight:** The PyTorch SepPool implementation already has the correct data structure for `pref_reweight_teacher`. `TeacherBuffer.pref_episodes` is already the `pref_teacher_episodes` equivalent from TF. It already:
- Contains all 4 original expert episodes (built before ring truncation)
- Has `J` scores for each episode
- Is appended to on promotion

What is missing is:
1. Weight computation (softmax over J/beta)
2. Weighted sampling path (returns episodes with their weights)
3. Weighted expert loss in the discriminator

The pref pool should be **the canonical episode-level object** for `pref_reweight_teacher`. The ring/transition buffer (`teacher_buffer.states/actions`) stays completely unchanged — it continues to serve disc/LfD batches when reweighting is disabled and for LfD mixing in the critic path.

---

## 3. Mismatch Table

| Component | TF behavior | PyTorch current | Needed change |
|-----------|-------------|-----------------|---------------|
| Expert pref episode pool | `pref_teacher_episodes` list of dicts, built from full NPZ before ring | `teacher_buffer.pref_episodes` — identical structure, already correct | None needed |
| Weight computation | `softmax(J/beta)`, stored as `pref_teacher_weights` np.array | Not computed | Add `compute_pref_weights(beta)` to `TeacherBuffer`, returns/stores `pref_teacher_weights` |
| Expert batch sampling | Uniform episode sampling + per-sample weight attached | `sample_batch()` — uniform transition sampling, no weights | Add `sample_batch_weighted(B, beta)` that returns `(states, actions, weights)` |
| Discriminator expert loss | Weighted mean: `sum(w_i * loss_i) / sum(w_i)` | Unweighted mean: `F.binary_cross_entropy_with_logits(...).mean()` | Add `expert_w` arg to `compute_loss()`, gated by `use_expert_weights` flag |
| Discriminator flag for weights | `use_expert_weights` on discriminator object | Not present | Add `use_expert_weights: bool` param to `Adversary.__init__` |
| SAIL sampling call | `generate_discriminator_data()` calls `_sample_pref_weighted_expert()` | `SAIL._update_discriminator()` calls `teacher_buffer.sample_batch()` | When `pref_reweight_teacher=True`, call `sample_batch_weighted()` instead |
| Weight always fed to disc | `expert_w_ph` always fed, ones when disabled | Not fed at all | Always compute+pass `expert_w`; ones when `pref_reweight_teacher=False` |
| Weights after promotion | `_recompute_pref_teacher_weights()` immediately | Not computed | Call `compute_pref_weights()` inside `TeacherBuffer.add_episode()` when RM loaded |
| Pool pruning | Quantile-based: keep J >= quantile(scores, q=0.75), always keep best | Not pruned | Add pruning logic to `add_episode()`, gated by `pref_max_teacher_trajs` |
| CLI flag | `--pref-reweight-teacher` | Not present | Add `--pref_reweight_teacher` to `train_sail.py` argparse |
| `pref_beta` | `pref_beta=1.0` default, only used when flag=True | Not present | Add `--pref_beta` to `train_sail.py` argparse |
| `pref_max_teacher_trajs` | Default 500, quantile-prune when exceeded | Not present | Add `--pref_max_teacher_trajs` to argparse; pass to TeacherBuffer |
| Logging | No W&B logging for weight stats | Existing logging | Add `pref_reweight/weight_max`, `pref_reweight/pool_size` metrics |

---

## 4. Step-by-Step Implementation Plan

### Step 1 — Add weight computation to `TeacherBuffer`

**File:** `sail_sb3/datasets/teacher_buffer.py`  
**Function:** New method `_recompute_pref_weights(beta: float = 1.0)`

Logic (exact TF parity):
```
scores = [ep['J'] for ep in self.pref_episodes]
s = np.array(scores, dtype=np.float64) / beta
s = s - s.max()
w = np.exp(s)
w = w / (w.sum() + 1e-8)
self.pref_teacher_weights = w   # np.array[N], sums to ~1
```

Also add to `__init__`:
```python
self.pref_teacher_weights = None  # np.array[N] or None
self._pref_reweight_beta = 1.0    # set later by SAIL or train_sail
```

**When to call:** 
- After `_build_pref_episodes()` (end of `__init__`, if RM is loaded)  
- At end of `add_episode()` (after appending to `pref_episodes`)

**Pruning in `add_episode()`** (new logic, gated by `pref_max_teacher_trajs`):
```
if pref_max_teacher_trajs is not None and len(self.pref_episodes) > pref_max_teacher_trajs:
    scores = np.array([ep['J'] for ep in self.pref_episodes])
    thresh = np.quantile(scores, pref_promote_quantile)  # default 0.75
    keep = [i for i, ep in enumerate(self.pref_episodes) if ep['J'] >= thresh]
    if len(keep) == 0:
        keep = [int(np.argmax(scores))]
    self.pref_episodes = [self.pref_episodes[i] for i in keep]
```

New `__init__` params: `pref_max_teacher_trajs: int = None`, `pref_promote_quantile: float = 0.75`

### Step 2 — Add weighted sampling to `TeacherBuffer`

**File:** `sail_sb3/datasets/teacher_buffer.py`  
**Function:** New method `sample_batch_weighted(batch_size: int) -> tuple[Tensor, Tensor, Tensor]`

Logic (exact TF parity — uniform episode, weighted loss):
```python
def sample_batch_weighted(self, batch_size: int):
    """
    Returns (states, actions, expert_w) where expert_w[i] is the softmax
    weight of the episode that transition i was drawn from.
    
    TF parity: episodes are sampled UNIFORMLY (not by weight).
    The weights are returned as per-sample loss multipliers.
    Requires self.pref_teacher_weights to be computed.
    """
    if self.pref_teacher_weights is None or len(self.pref_episodes) == 0:
        # Fallback: uniform from transition buffer, ones weights
        batch = self.sample_batch(batch_size)
        expert_w = torch.ones(batch_size, 1, device=self.device)
        return batch['states'], batch['actions'], expert_w

    N = len(self.pref_episodes)
    obs_list, acs_list, w_list = [], [], []
    for _ in range(batch_size):
        ep_idx = np.random.randint(0, N)        # UNIFORM — TF parity
        ep = self.pref_episodes[ep_idx]
        T = ep['obs'].shape[0]
        t = np.random.randint(0, T)
        obs_list.append(ep['obs'][t].float())
        acs_list.append(ep['acs'][t].float())
        w_list.append(float(self.pref_teacher_weights[ep_idx]))

    states  = torch.stack(obs_list).to(self.device)
    actions = torch.stack(acs_list).to(self.device)
    expert_w = torch.tensor(w_list, dtype=torch.float32, device=self.device).reshape(-1, 1)
    return states, actions, expert_w
```

### Step 3 — Add weighted expert loss to `Adversary`

**File:** `sail_sb3/reward_models/adversary.py`  
**Function:** Modify `compute_loss()` signature and loss computation

New signature:
```python
def compute_loss(self,
                 expert_state, expert_action,
                 policy_state, policy_action,
                 expert_w=None          # NEW: optional (N, 1) weight tensor
                 ) -> tuple:
```

New expert loss computation (gated by `self.use_expert_weights`):
```python
expert_logits = self.forward(expert_state, expert_action)
sample_expert_loss = F.binary_cross_entropy_with_logits(
    expert_logits, torch.ones_like(expert_logits), reduction='none')  # [N, 1]

if self.use_expert_weights and expert_w is not None:
    # Weighted mean: sum(w_i * loss_i) / sum(w_i)  — TF parity
    expert_loss = (sample_expert_loss * expert_w).sum() / (expert_w.sum() + 1e-8)
else:
    expert_loss = sample_expert_loss.mean()
```

Add `use_expert_weights: bool = False` to `Adversary.__init__`.

### Step 4 — Wire weighted sampling into `SAIL._update_discriminator()`

**File:** `sail_sb3/algorithms/sail.py`  
**Function:** `_update_discriminator()`

Change the expert batch construction block:
```python
# Before (current):
expert_batch = self.teacher_buffer.sample_batch(self.disc_batch_size)
expert_states  = expert_batch['states'].to(self.device)
expert_actions = expert_batch['actions'].to(self.device)

# After (new):
if self.pref_reweight_teacher and len(self.teacher_buffer.pref_episodes) >= 2:
    expert_states, expert_actions, expert_w = \
        self.teacher_buffer.sample_batch_weighted(self.disc_batch_size)
    expert_states  = expert_states.to(self.device)
    expert_actions = expert_actions.to(self.device)
    expert_w       = expert_w.to(self.device)
else:
    expert_batch = self.teacher_buffer.sample_batch(self.disc_batch_size)
    expert_states  = expert_batch['states'].to(self.device)
    expert_actions = expert_batch['actions'].to(self.device)
    expert_w = torch.ones(self.disc_batch_size, 1, device=self.device)

# Pass expert_w to compute_loss:
total_loss, e_bce, p_bce, entropy, gp = self.discriminator.compute_loss(
    expert_states, expert_actions,
    replay_data.observations, replay_data.actions,
    expert_w=expert_w
)
```

Add to `SAIL.__init__`:
```python
pref_reweight_teacher: bool = False,
```

Store it:
```python
self.pref_reweight_teacher = pref_reweight_teacher
```

### Step 5 — Add CLI arguments to `train_sail.py`

**File:** `sail_sb3/scripts/train_sail.py`

New argparse entries:
```python
parser.add_argument("--pref_reweight_teacher", action="store_true",
    help="Enable Boltzmann-weighted discriminator expert batch (PR-SAIL). "
         "Requires --pref_rm.")
parser.add_argument("--pref_beta", type=float, default=1.0,
    help="Temperature for Boltzmann weighting: larger=flatter, smaller=sharper. "
         "Only used when --pref_reweight_teacher is set.")
parser.add_argument("--pref_max_teacher_trajs", type=int, default=500,
    help="Max episodes in pref pool before quantile pruning (TF default: 500).")
parser.add_argument("--pref_promote_quantile", type=float, default=0.75,
    help="Quantile threshold for pruning low-J episodes from pref pool (TF default: 0.75).")
```

Also update the `need_pref_rm` check:
```python
need_pref_rm = args.pref_rank_disc or (args.adaptive_score_source == "rm") \
               or args.pref_reweight_teacher
```

Update `TeacherBuffer` construction to pass new params:
```python
teacher_buffer = TeacherBuffer(
    args.expert_data, device,
    pref_rm_path=args.pref_rm if need_pref_rm else None,
    expect_obs_dim=args.pref_expect_obs_dim,
    max_size=args.teacher_buffer_size,
    pref_max_teacher_trajs=args.pref_max_teacher_trajs if args.pref_reweight_teacher else None,
    pref_promote_quantile=args.pref_promote_quantile,
)
```

Set beta on teacher buffer after construction:
```python
if args.pref_reweight_teacher:
    teacher_buffer._pref_reweight_beta = args.pref_beta
```

Update `Adversary` construction:
```python
discriminator = Adversary(
    ...,
    use_expert_weights=args.pref_reweight_teacher,
)
```

Update `SAIL` construction:
```python
model = SAIL(
    ...,
    pref_reweight_teacher=args.pref_reweight_teacher,
)
```

### Step 6 — Beta propagation

The beta value needs to reach `sample_batch_weighted`. Two clean options:

**Option A (recommended):** Store beta on `TeacherBuffer` as `self._pref_reweight_beta` (set from `train_sail.py` after construction). `sample_batch_weighted()` uses `self._pref_reweight_beta` internally to call `_recompute_pref_weights()` if needed.

**Option B:** Pass beta to each `sample_batch_weighted(B, beta)` call. Requires SAIL to store beta.

Recommend Option A — matches TF's config-dict approach and avoids threading beta through SAIL.

However, note that `_recompute_pref_weights()` should be called whenever the pool changes (after `add_episode()`) and should use the stored beta. The weights should be pre-computed and cached, not recomputed on every sampling call.

### Step 7 — Logging

In `SAIL._update_discriminator()`, when `pref_reweight_teacher=True`, log:
```python
self.logger.record("pref_reweight/pool_size", len(self.teacher_buffer.pref_episodes))
self.logger.record("pref_reweight/weight_max", float(self.teacher_buffer.pref_teacher_weights.max()))
self.logger.record("pref_reweight/weight_min", float(self.teacher_buffer.pref_teacher_weights.min()))
```
Log these only once per train() call (not every disc gradient step) — drain alongside `_pending_disc_losses`.

---

## 5. Episode-Level vs Transition-Level Weighting

### TF behavior

**Episode-level weighting, transition-level application.** One scalar weight per episode; all transitions from that episode receive the same weight. Within an episode, the timestep is sampled uniformly. The weight vector is `[N_eps]`, broadcast to `[B_transitions]` by the sampling loop.

### Three possible PyTorch approaches

**Approach 1 (TF-faithful, recommended):** Sample `B` transitions by: first pick a random episode uniformly, then pick a random timestep uniformly. Attach `weights[ep_idx]` to each sampled transition. Return `(states, actions, expert_w [B, 1])`. Pass `expert_w` to discriminator loss as weighted mean.

**Approach 2 (Alternative — proportional episode sampling):** Sample episodes proportionally to their weights (using `np.random.choice(..., p=weights)`), then uniform timestep within. This would give high-J episodes more samples rather than more loss weight per sample. NOT TF behavior.

**Approach 3 (Transition-level weights):** Expand episode weights to all their transitions (proportional to episode length), build a flat transition-level probability array, sample transitions directly. Different behavior from TF.

### Recommendation

**Use Approach 1 (TF-faithful).** The key reason TF used uniform sampling with weighted loss rather than proportional sampling is numerical stability and simplicity. With `beta=1.0` and RM scores in the [1300–1400] range for HalfCheetah, the softmax weights are extremely concentrated — proportional sampling would degenerate to sampling only the top 1–2 episodes. Uniform sampling with weighted loss preserves diversity while still biasing the gradient toward high-quality episodes. This is what TF does, and it's the right design for the continuous RM score range.

---

## 6. Edge Cases / Failure Modes

### 1. `pref_teacher_weights` is None before first `_build_pref_episodes` completes
If `pref_rm_path` is None or RM fails to load, `pref_episodes` will be empty and `pref_teacher_weights` will be None. `sample_batch_weighted()` must fall back to `sample_batch()` with ones weights. Already handled by the `if pref_teacher_weights is None` guard.

### 2. All episodes have identical J scores
If all 4 initial expert episodes have the same RM score (spread=0), softmax weights are `[0.25, 0.25, 0.25, 0.25]` — perfectly uniform. This is correct and the fallback uniform behavior. No special handling needed.

### 3. Very small beta (e.g., 0.01) with large J spread
With HalfCheetah RM scores ~1300–1400 and beta=0.01, `scores/beta` ranges from 130000 to 140000. Max-subtraction handles this: `s - s.max()` → range is [0, -10000/0.01 = -1000]. `exp(-1000) ≈ 0`, so only the single best episode has nonzero weight. This degenerates `sample_batch_weighted` to always sampling from one episode. Should be documented in help text.

### 4. Promoted episode has J below all expert episodes
This can happen early in training when a student episode is promoted by GT score but its RM score is lower than the expert pool. After adding, `_recompute_pref_teacher_weights` gives it very low weight. Correct behavior — the discriminator still gets expert transitions, just less of them from the low-J student.

### 5. Pool pruning removes promoted episode immediately
If a high-GT-score student episode has low RM score (RM and GT disagree), the quantile pruning could remove it. This would create a mismatch between what's in `demo_replay_buffer` and what's in `pref_episodes`. This is acceptable — they are intentionally separate pools serving different purposes.

### 6. `pref_reweight_teacher=True` without `disc_use_expert_weights=True` (TF bug risk)
In TF, these are separate flags. If someone sets `pref_reweight_teacher=True` but forgets `disc_use_expert_weights=True`, the weights are computed and sampled but discarded in loss computation — effectively a no-op. In PyTorch, these should be coupled: `use_expert_weights = pref_reweight_teacher` in `train_sail.py`. Only expose one flag (`--pref_reweight_teacher`) to the user.

### 7. `sample_batch_weighted` samples from `pref_episodes` which stores `obs` as tensors (from `add_episode`)
In PyTorch `TeacherBuffer`, initial expert episodes are stored as `torch.Tensor` in `pref_episodes[i]['obs']` (see `_build_pref_episodes()`). Promoted student episodes' `obs` are also tensors (see `add_episode()`). When indexing `ep['obs'][t]`, `.float()` must be called. The implementation above already does this.

### 8. Obs normalization in discriminator
When `pref_reweight_teacher=True`, the discriminator's `update_obs_rms()` is called on expert transitions sampled from `pref_episodes`. These may not include the truncated ring-buffer episodes (the full 4 vs the ring-1 episode). This is fine — the normalization gets more diverse expert observations as input.

---

## 7. Minimal Validation Plan

After implementation, verify the following before submitting full runs:

### Unit validation (offline, no training)

1. **Weight computation:** Load 4-episode expert data, call `_recompute_pref_weights(beta=1.0)`. Verify `sum(weights) ≈ 1.0`, `len(weights) == 4`, all weights > 0.

2. **Weighted sampling shape:** Call `sample_batch_weighted(256)`. Verify `states.shape == (256, obs_dim)`, `actions.shape == (256, act_dim)`, `expert_w.shape == (256, 1)`, `expert_w` values match `pref_teacher_weights[ep_idx]` for the sampled episodes.

3. **Weighted loss computation:** Pass `expert_w` to `compute_loss()`. Verify that different weight vectors produce different loss values. Verify that `expert_w = ones` gives the same result as the unweighted path.

4. **Pool pruning:** Add 600 synthetic episodes with random J scores to a buffer, check that pool is pruned to ~150 (75th percentile kept).

### Training validation (from logs at ~10k–50k steps)

1. **Startup logs should show:**
   ```
   [TeacherBuffer] Pref pool: 4 expert episodes
   [TeacherBuffer] Pref pool RM scores: mean=1353.5 ...
   [pref_reweight] weights computed: min=X max=Y sum≈1.0
   ```

2. **pref_reweight/weight_max should decrease over time** (as more diverse student episodes are added and the pool diversifies).

3. **disc_loss should be similar to the non-reweighted runs** — reweighting is a subtle change, not a drastic one. Any jump in disc_loss at start of training is a bug.

4. **expert_w in discriminator call should not be all-ones** — verify via a debug print that at least some weights differ from 0.25.

---

## 8. Open Questions / Ambiguities from TF Code

### A. `disc_use_expert_weights` and `pref_reweight_teacher` are separate flags
In TF, `disc_use_expert_weights` is set independently (via `--disc-use-expert-weights`). The fact that both must be True for reweighting to work is not enforced. In the sbatch scripts, it is unclear whether both were set. Recommendation: **couple them in PyTorch** — `pref_reweight_teacher=True` automatically implies `use_expert_weights=True` in the adversary. No separate flag exposed.

### B. Sampling is uniform, NOT weighted by `w`
Line 804: `ep_idx = np.random.randint(0, n_eps)` — this is uniform, not `np.random.choice(n_eps, p=weights)`. The comment above it says `# instead of np.random.choice(..., p=weights)`. This appears to be a **deliberate choice or a bug** in TF. The comment reads as if proportional sampling was considered but not used. Implication: the softmax weights are used **purely as per-sample importance weights in the loss**, not as sampling probabilities. This gives each episode equal representation in the batch but different gradient magnitude. We should replicate this exactly as it is documented.

### C. `pref_beta` is only read when `pref_reweight_teacher=True`
In TF, `pref_beta` is not stored in config unless `pref_reweight_teacher=True` (line 944). But `_recompute_pref_teacher_weights` reads `cfg.get('pref_beta', 1.0)` with default 1.0. So if `pref_rank_disc=True` and `pref_reweight_teacher=False`, `beta=1.0` is used implicitly. In PyTorch, beta is only relevant for `pref_reweight_teacher`, so storing it with that flag is correct.

### D. Whether `pref_teacher_episodes` is also used by `pref_rank_disc` in TF
Yes. `_sample_filtered_pref_pairs()` and `_sample_full_episode_pref_pairs()` both read from `self.pref_teacher_episodes` (same pool as reweighting). In PyTorch, `teacher_buffer.pref_episodes` already serves both `pref_rank_disc` (via `sample_pref_pairs()`) and will serve `pref_reweight_teacher` (via `sample_batch_weighted()`). This sharing is intentional and correct.

### E. TF prunes with `pref_promote_quantile=0.75` — meaning 25% of episodes are pruned each time
This is aggressive for small pools. With 4 initial + 10 promoted = 14 episodes, pruning at 0.75 quantile keeps ~3–4 episodes. This could cause the pool to oscillate in size. Recommend a conservative default (`pref_max_teacher_trajs=500`) so pruning only activates for long runs with many promotions.

### F. The pref pool `pref_episodes` in PyTorch stores tensors; TF stores numpy arrays
TF `pref_teacher_episodes` stores numpy arrays (`obs_ep.copy()`). PyTorch `pref_episodes` stores tensors (see `_build_pref_episodes` and `add_episode`). In `sample_batch_weighted`, indexing `ep['obs'][t]` returns a scalar-indexed tensor. This needs `.float()` and no `.copy()`. This difference is fine and already handled.

---

## Summary: Which parts stay different from TF (intentionally)

| Aspect | TF | PyTorch (intentionally different) | Reason |
|--------|----|------------------------------------|--------|
| pref pool data type | numpy arrays | torch tensors | PyTorch architecture |
| beta storage | in `self.config` dict | as `teacher_buffer._pref_reweight_beta` | No global config dict in SB3 |
| `disc_use_expert_weights` flag | separate from `pref_reweight_teacher` | coupled (same flag) | Prevent footgun |
| Pruning call site | in SAIL `learn()` loop | in `TeacherBuffer.add_episode()` | Encapsulation |
| Pref pool object | `SAIL.pref_teacher_episodes` | `TeacherBuffer.pref_episodes` | Already exists in SepPool arch |
| Pool building | from full NPZ inside SAIL.initialize_expert_buffer | from full NPZ inside TeacherBuffer.__init__ (SepPool) | Already correct |
