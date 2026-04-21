# QPREF: TF → PyTorch Implementation Plan

**Date:** 2026-04-09  
**Status:** Analysis + plan only. No code written yet.

---

## Section 1 — TF QPREF Audit

### 1.1 CLI flags

File: `stable-baselines/run/train_sail.py`, lines 579–627

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--qpref` | store_true | False | Master enable switch |
| `--qpref-weight` | float | 0.0 | Lambda λ for qpref term in critic loss. If `--qpref` is set but weight=0, auto-defaults to 0.1 (line 897–899) |
| `--qpref-temp` | float | 1.0 | Temperature T for the logistic loss: softplus(−Δ/T) |
| `--qpref-batch-size` | int | 32 | Number of preference pairs sampled per critic update step |
| `--qpref-start-step` | int | 0 | Gate: QPREF is silently zero until `step >= qpref_start_step` |
| `--qpref-source` | str (choices: teacher/student) | teacher | Episode pool to sample pairs from |
| `--pref-max-student-trajs` | int | 500 | Max size of student episode pool before pruning |

Config dict population: lines 893–906.

---

### 1.2 Data structures initialized at `__init__`

File: `stable-baselines/stable_baselines/td3/sail.py`, lines 172–181

```python
# Teacher pool (shared with pref_reweight_teacher and pref_rank_disc)
self.pref_teacher_episodes = None   # list[dict]: {'obs': [T, obs_dim], 'acs': [T, act_dim], 'J': float}
self.pref_teacher_scores   = None   # np.array [N_eps]
self.pref_teacher_weights  = None   # np.array [N_eps], softmax(J/beta)
self._pref_teacher_rm      = None   # offline pref RM (lazy-loaded)

# Student pool (only used for qpref_source=student)
self.pref_student_episodes = None   # list[dict]: same structure
self.pref_student_scores   = None   # np.array [N_eps]
self.pref_student_weights  = None   # np.array [N_eps]
```

**Note:** `pref_teacher_episodes` is a SHARED pool — the same list is used by
`pref_reweight_teacher`, `pref_rank_disc`, AND `qpref_source=teacher`. This means
QPREF teacher mode does NOT require its own buffer; it reuses the SepPool built in
`_build_pref_teacher_buffer()`.

---

### 1.3 Teacher pool initialization

File: `sail.py`, `_build_pref_teacher_buffer()`, lines 683–728

Called once at startup when any of `{pref_reweight_teacher, pref_rank_disc, qpref}` is True.
Steps:
1. Load all expert transitions; split into episodes using `demo_dones` boundaries.
2. For each episode: score with pref RM → `J = sum(rm.reward(obs_ep, acs_ep))`.
3. Store as `pref_teacher_episodes` list.
4. Call `_recompute_pref_teacher_weights()` (softmax over J/beta).

---

### 1.4 TF graph construction — placeholders

File: `sail.py`, lines 274–284

```python
obs_shape = (None,) + self.observation_space.shape  # (None, obs_dim)
act_shape = (None,) + self.action_space.shape        # (None, act_dim)

self.qpref_pos_obs_ph = tf.placeholder(tf.float32, shape=obs_shape, name="qpref_pos_obs")
self.qpref_pos_acs_ph = tf.placeholder(tf.float32, shape=act_shape, name="qpref_pos_acs")
self.qpref_neg_obs_ph = tf.placeholder(tf.float32, shape=obs_shape, name="qpref_neg_obs")
self.qpref_neg_acs_ph = tf.placeholder(tf.float32, shape=act_shape, name="qpref_neg_acs")

self.qpref_weight_ph = tf.placeholder(tf.float32, shape=(), name="qpref_weight")
self.qpref_temp_ph   = tf.placeholder(tf.float32, shape=(), name="qpref_temp")
```

Each placeholder batch dimension is dynamic (`None`) — the batch size is `qpref_batch_size`
at runtime but `1` during the zero-fill default feed.

---

### 1.5 TF graph construction — Q values for pairs

File: `sail.py`, lines 298–313

```python
if (self.config or {}).get("qpref", False):
    q1_pos, q2_pos = self.policy_tf.make_critics(
        self.qpref_pos_obs_ph, self.qpref_pos_acs_ph, reuse=True)   # SAME weights as Bellman
    q1_neg, q2_neg = self.policy_tf.make_critics(
        self.qpref_neg_obs_ph, self.qpref_neg_acs_ph, reuse=True)
    self.qpref_q1_pos, self.qpref_q2_pos = q1_pos, q2_pos
    self.qpref_q1_neg, self.qpref_q2_neg = q1_neg, q2_neg
```

**Critical:** `reuse=True` means the qpref forward passes use the SAME critic weights
as the Bellman TD loss. There are no separate networks. The qpref loss backprops through
the same `model/values_fn/` params that the TD error uses.

---

### 1.6 TF loss construction

File: `sail.py`, lines 374–387

```python
if (self.config or {}).get("qpref", False):
    q1_pos, q2_pos = self.qpref_q1_pos, self.qpref_q2_pos
    q1_neg, q2_neg = self.qpref_q1_neg, self.qpref_q2_neg

    q_pos = tf.minimum(q1_pos, q2_pos)            # shape [B, 1]
    q_neg = tf.minimum(q1_neg, q2_neg)            # shape [B, 1]

    temp  = tf.maximum(self.qpref_temp_ph, 1e-6)
    delta = (q_pos - q_neg) / temp                # shape [B, 1]

    self.qpref_loss = tf.reduce_mean(tf.nn.softplus(-delta))  # scalar ≥ 0

    qvalues_losses = qvalues_losses + self.qpref_weight_ph * self.qpref_loss
```

Where `qvalues_losses = qf1_loss + qf2_loss` (line 369) — both Q-network Bellman MSE terms.

Combined loss is minimized with one optimizer step:
```python
train_values_op = qvalues_optimizer.minimize(qvalues_losses, var_list=qvalues_params)
# qvalues_params = get_vars('model/values_fn/')
```

So QPREF loss + both TD errors share a **single Adam update**.

---

### 1.7 Sampling: `_sample_qpref_start_pairs`

File: `sail.py`, lines 896–940

```python
def _sample_qpref_start_pairs(self, B, source="teacher"):
    eps = self.pref_student_episodes if source == "student" else self.pref_teacher_episodes
    if (eps is None) or (len(eps) < 2):
        return None

    N = len(eps)
    idx_a = np.random.randint(0, N, size=B)
    idx_b = np.random.randint(0, N, size=B)
    for k in range(B):
        if idx_b[k] == idx_a[k]:
            idx_b[k] = (idx_b[k] + 1) % N       # deterministic tie-break

    pos_obs, pos_acs, neg_obs, neg_acs = [], [], [], []
    for a, b in zip(idx_a, idx_b):
        Ja = float(eps[a]['J'])
        Jb = float(eps[b]['J'])

        epP, epN = (eps[a], eps[b]) if Ja >= Jb else (eps[b], eps[a])

        tP = np.random.randint(0, epP['acs'].shape[0])   # ONE random timestep per traj
        tN = np.random.randint(0, epN['acs'].shape[0])

        pos_obs.append(epP['obs'][tP])    # single (s,a) from positive trajectory
        pos_acs.append(epP['acs'][tP])
        neg_obs.append(epN['obs'][tN])    # single (s,a) from negative trajectory
        neg_acs.append(epN['acs'][tN])

    return (np.asarray(pos_obs),   # [B, obs_dim]
            np.asarray(pos_acs),   # [B, act_dim]
            np.asarray(neg_obs),   # [B, obs_dim]
            np.asarray(neg_acs))   # [B, act_dim]
```

**The fundamental sampling design:** Each preference pair contributes exactly **one
randomly sampled (s_t, a_t)** per trajectory. There is no aggregation over time. The
trajectory-level J determines which is positive vs negative, but only a single
state-action is fed to the Q network.

---

### 1.8 Feed dict gating (`_train_step`)

File: `sail.py`, lines 494–542

```python
# Always feed zeros (avoids TF "must feed" error)
feed_dict[self.qpref_pos_obs_ph] = np.zeros((1, obs_dim), dtype=np.float32)
feed_dict[self.qpref_weight_ph]  = 0.0     # weight=0 → loss is zeroed out

# Only fill real pairs when:
#   (a) qpref_weight > 0
#   (b) step >= qpref_start_step
#   (c) enough episodes exist (>= 2) in the chosen pool
if (qpref_w > 0.0) and (step >= qpref_start):
    pair = self._sample_qpref_start_pairs(qpref_B, source=qpref_source)
    if pair is not None:
        feed_dict.update({
            self.qpref_pos_obs_ph: pos_obs,
            self.qpref_pos_acs_ph: pos_acs,
            self.qpref_neg_obs_ph: neg_obs,
            self.qpref_neg_acs_ph: neg_acs,
            self.qpref_weight_ph: qpref_w,
            self.qpref_temp_ph: max(qpref_T, 1e-6),
        })
```

QPREF activates at **every TD3 critic gradient step** once gating conditions are met.
It is NOT applied only at disc update frequency or at a different schedule from the critic.
Since `_train_step` is called inside `for grad_step in range(gradient_steps)` at
`step % train_freq == 0`, QPREF fires at the same cadence as TD3 critic updates.

---

### 1.9 Student pool management

File: `sail.py`, `_add_student_episode_to_pref_buffer()`, lines 746–778;
called from main training loop at episode end, lines 1616–1631.

Gating: only active when `qpref=True AND qpref_source='student'`.

Steps:
1. At episode end, collect `obs_ep, acs_ep` from `episode_buffer`.
2. Score: `J = sum(rm.reward(obs_ep, acs_ep))`.
3. Append `{'obs': obs_ep, 'acs': acs_ep, 'J': J}` to `pref_student_episodes`.
4. Prune by **recency** (keep last `pref_max_student_trajs`): `pref_student_episodes[-max_traj:]`.
5. Recompute weights via `_recompute_pref_student_weights()`.

**Difference from teacher pruning:** Teacher pool is pruned by quality (quantile 0.75 on J).
Student pool is pruned by recency (most recent episodes kept). This makes sense:
teacher episodes are fixed-quality expert demos; student episodes improve over time so
recency ≈ quality.

---

### 1.10 RM loading for QPREF

File: `sail.py`, `_ensure_pref_teacher_rm()`, lines 575–615

QPREF shares the same pref RM as `pref_reweight_teacher` and `pref_rank_disc`.
The RM is lazy-loaded once and reused for teacher scoring (startup) and student scoring
(every episode end in student mode). This is the same `pref_rm_path` config key.

---

## Section 2 — Exact Answers to Source/Buffer/Loss Questions

### A. QPREF sources

Two sources, selected by `--qpref-source`:
1. **`teacher`** (default): `pref_teacher_episodes` — built once from expert demos at startup
2. **`student`**: `pref_student_episodes` — built incrementally from student rollouts during training

No third source. Both use identical sampling logic in `_sample_qpref_start_pairs`.

---

### B. qpref_source=teacher

**Buffer:** `self.pref_teacher_episodes` — shared with `pref_reweight_teacher` and `pref_rank_disc`.

**Sampling unit:** NOT a full trajectory, NOT a segment. **One randomly selected
transition (s_t, a_t)** per trajectory per pair. The trajectory J determines which is
positive/negative; only a single step is used for the actual Q evaluation.

**Preference label:** `J(τ_pos) >= J(τ_neg)` where J = sum of pref RM reward over the trajectory.
Label is deterministic from stored J values (no tie-breaking by probability, no sigma).
Ties go to episode a (because the condition is `>=`).

**Q target:** There is no explicit Q target. The loss operates on the CURRENT critic's
Q values at the sampled (s,a) pairs. It is a ranking loss added to the TD Bellman loss —
it pushes Q(s_pos, a_pos) > Q(s_neg, a_neg) by gradient descent on the same optimizer
step that minimizes TD error.

---

### C. qpref_source=student

**Buffer:** `self.pref_student_episodes` — built incrementally during training.

**Episode selection:** Each completed rollout episode is scored with the pref RM (J =
Σ_t R_phi(s_t, a_t)) and appended to the student pool. The pool is pruned by
recency: `pref_student_episodes[-pref_max_student_trajs:]`.

**Pair labels:** Same as teacher: compare stored J values. Higher J → positive.

**Q input:** Same as teacher: one uniformly random timestep from each of the two
selected episodes.

---

### D. J comparison vs Q aggregation

**TF uses: one random Q(s_t, a_t) per trajectory, NOT aggregate.**

The trajectory-level J(τ) determines the preference label (pos/neg). But the actual
scalar fed to the critic for preference comparison is Q(s_t, a_t) at a single uniformly
random timestep t. There is:
- No sum over the trajectory
- No mean over the trajectory
- No masking
- No weighting by step

This is the key design choice, discussed in depth in Section 3.

---

### E. Exact loss formula

From `sail.py` lines 378–385:

```
q_pos = min(q1_pos, q2_pos)             shape [B, 1]
q_neg = min(q1_neg, q2_neg)             shape [B, 1]
temp  = max(qpref_temp_ph, 1e-6)
delta = (q_pos - q_neg) / temp          shape [B, 1]

L_qpref = mean_B [ softplus(−delta) ]
         = mean_B [ log(1 + exp(−delta)) ]    scalar ≥ 0

L_total_critic = L_TD_q1 + L_TD_q2 + λ · L_qpref
```

Where:
- `softplus(−x) = log(1 + exp(−x))` is exactly the Bradley-Terry preference loss
  (same as `−log(sigmoid(x))`)
- Temperature T scales the sensitivity: large T → softer gradient; small T → sharper
- λ = `qpref_weight` (default 0.0, auto-defaulted to 0.1 if flag set without weight)
- Reduction: `reduce_mean` over B pairs
- Both critic networks q1 and q2 receive gradient through the `min(q1, q2)` operation
  (only the smaller Q gets gradient, same structure as the Bellman target's `min`)
- This is **NOT** a separate optimizer — it is added to `qvalues_losses` and minimized
  by the same `qvalues_optimizer.minimize()` step

---

### F. Training stage

QPREF activates at **every TD3 critic gradient step** when:
1. `qpref_weight > 0.0` (or auto-defaulted)
2. `num_timesteps >= qpref_start_step` (default 0, meaning from the first update)
3. The chosen pool has `>= 2` episodes

For `qpref_source=teacher` with SepPool active, condition 3 is satisfied from
timestep 0 (pref_teacher_episodes is built at startup with all 4 expert episodes).
For `qpref_source=student`, condition 3 is not satisfied until the second rollout
episode completes.

QPREF is independent of `learning_starts` — but in practice it cannot fire before
`learning_starts` because `_train_step` is never called before `learning_starts`.

---

## Section 3 — Is Random-Q vs Mean-Q Conceptually Correct?

### The fundamental tension

Trajectory J(τ) is defined as the cumulative reward:
```
J(τ) = Σ_{t=0}^{T-1}  R(s_t, a_t)     [undiscounted TF convention with pref RM]
```

The critic Q-function is defined as the discounted value from a given state:
```
Q(s_t, a_t) = Σ_{k=0}^{∞}  γ^k · R(s_{t+k}, a_{t+k})
```

These are different quantities. J(τ) is the total undiscounted RM reward for the
episode. Q(s_t, a_t) is the discounted sum from step t onward. Neither is exactly
equal to J(τ), and the question is which aggregation of Q(s_t, a_t) values best
"tracks" J(τ) as a proxy for trajectory quality.

---

### Random single Q: TF's actual behavior

TF uses one randomly sampled Q(s_t, a_t). The preference label comes from J(τ) ≥ J(τ'),
but the scalar that gradient flows through is a random Q at a random step.

**Problem:** Q(s_t, a_t) depends heavily on t. For HalfCheetah:
- Q(s_0, a_0) ≈ discounted future return from step 0 ≈ J(τ) roughly (minus discounting)
- Q(s_{999}, a_{999}) ≈ reward for only the last few steps; tiny value
- Q(s_{500}, a_{500}) ≈ halfway-through value; moderate

If τ_pos has J slightly above τ_neg but we randomly pick t=999 from τ_pos and t=1
from τ_neg, we get Q_pos << Q_neg even though τ_pos is the preferred trajectory.
The ranking signal is thus **highly noisy** — correct on average (high-J trajectories
tend to have higher Q at most timesteps) but with substantial within-batch variance.

This noise averages out over many gradient steps (the expectation of Q(s_t) over
random t is still correlated with J), but it adds training instability and requires
more samples to converge.

---

### Mean-Q aggregation

```
Q_mean(τ) = (1/T) · Σ_{t=0}^{T-1}  Q(s_t, a_t)
```

**Why it is better aligned with J:**
- By averaging over all steps, you remove the dependence on which specific t was sampled
- If J(τ+) > J(τ-), then at MOST timesteps t, Q(s_t+, a_t+) > Q(s_t-, a_t-), assuming
  the critic has converged to reflect trajectory quality
- The mean is a consistent estimator: with enough gradient steps, the softplus loss on
  mean-Q pairs produces the same training signal as on individual Q, but with much lower
  variance per step

**Limitation of mean-Q:**
- Mean-Q mixes Q values from different horizons (early Q is discounted return from t=0,
  late Q is a small near-terminal value). Mean of incoherent quantities.
- Empirically the heterogeneity is reduced because HalfCheetah trajectories are dense
  (1000 steps of roughly uniform reward), so Q(s_t) ≈ constant minus a discount-only
  correction

---

### Sum-Q aggregation

```
Q_sum(τ) = Σ_{t=0}^{T-1}  Q(s_t, a_t)
```

**Why it is WORSE:**
- Q_sum scales with trajectory length T. If T is fixed (HalfCheetah always 1000 steps),
  sum and mean differ only by a constant, making them equivalent.
- If T varies (early termination in other envs), sum would confound quality with length.
  A short mediocre trajectory could look worse than a long bad one.
- For HalfCheetah this is moot (T=1000 always), but mean is the safer choice in general.

---

### Scale and the softplus loss

The Bradley-Terry loss `softplus(−delta/T)` only cares about the **sign and magnitude
of delta = Q_pos − Q_neg**. Absolute scale does not affect which direction gradients push.
Temperature T controls sensitivity:
- Large T: small gradients even for large Q differences → slow convergence
- Small T: large gradients even for small Q differences → risk of dominating TD loss

With single-step Q, delta has high variance (can be large positive or large negative due
to timestep noise). With mean-Q, delta is smoother — pairs that are genuinely ranked
correctly produce a more consistent positive delta.

Scale matching between J and Q is irrelevant for the ranking loss direction but
relevant for temperature calibration. If you use mean-Q with T calibrated for single-Q
scale, the gradients will be ~T times smaller (because mean-Q values are smaller than
sum-Q). This suggests:
- TF-faithful: same T as TF
- Mean-Q variant: may need T adjusted downward by ~(1/T_mean_correction) or just tune it

---

### Relationship back to TF

TF uses single random Q — confirmed by lines 929–935 of `sail.py`. This is likely a
pragmatic simplicity choice: no trajectory aggregation loop, direct parallelism with the
pref_rank_disc design (which also feeds single Q per timestep per trajectory). The TF
authors may have reasoned that with gradient_steps=1000 per env step, noise averages out.

Mean-Q is a strictly more consistent estimator for trajectory quality, with tradeoff of:
1. Requiring a full-trajectory forward pass through the critic (T × batch_size evaluations)
2. Needing to store full trajectories (already done in pref_teacher/student_episodes)
3. Potentially needing temperature recalibration

**Verdict:** TF's random-Q is correct on average but noisy. Mean-Q is more principled
and consistent. For an initial port, implement TF-faithful first to establish a baseline.
Then implement mean-Q as Variant 2 to test if reduced variance helps.

---

## Section 4 — PyTorch Plan for TF-Faithful QPREF (Variant 1)

### Files to change: 4

1. `sail_sb3/scripts/train_sail.py`
2. `sail_sb3/algorithms/sail.py`
3. `sail_sb3/datasets/teacher_buffer.py`
4. No change to `sail_sb3/reward_models/adversary.py`

---

### Step 1: New CLI args in `train_sail.py`

Add 5 new argparse entries:

```python
# In the argparse section
parser.add_argument('--qpref', action='store_true',
    help='Enable Q-preference loss on TD3 critic (QPREF).')
parser.add_argument('--qpref_weight', type=float, default=0.0,
    help='Lambda for QPREF loss. Auto-defaulted to 0.1 if --qpref set without explicit weight.')
parser.add_argument('--qpref_temp', type=float, default=1.0,
    help='Temperature T: loss = softplus(-(Q_pos - Q_neg)/T).')
parser.add_argument('--qpref_batch_size', type=int, default=32,
    help='Number of preference pairs per critic gradient step.')
parser.add_argument('--qpref_start_step', type=int, default=0,
    help='Apply QPREF only after this many env steps.')
parser.add_argument('--qpref_source', type=str, default='teacher',
    choices=['teacher', 'student'],
    help='teacher = pref_teacher_episodes; student = pref_student_episodes.')
parser.add_argument('--pref_max_student_trajs', type=int, default=500,
    help='Max episodes in student pref pool before pruning by recency.')
```

Weight auto-default logic (TF parity):
```python
if args.qpref and args.qpref_weight <= 0.0:
    args.qpref_weight = 0.1
```

QPREF requires pref_rm: update `need_pref_rm` to include `or args.qpref`.

Pass `pref_max_student_trajs` to TeacherBuffer if `qpref_source=student`.

Pass all QPREF args to `SAIL(...)` constructor.

---

### Step 2: New field in `teacher_buffer.py` — student pool

`TeacherBuffer` currently has no student pool. Add:

**`__init__` additions:**
```python
self.pref_max_student_trajs: int = pref_max_student_trajs   # new param, default None
self.pref_student_episodes: list = []
```

**New method `add_student_episode(obs_ep, acs_ep, pref_rm)`:**
```python
def add_student_episode(self, obs_ep: np.ndarray, acs_ep: np.ndarray, pref_rm) -> None:
    """Score episode with pref RM, append to student pool, prune by recency."""
    # Score
    r = pref_rm.reward(obs_ep, acs_ep)
    J = float(np.asarray(r).reshape(-1).sum())
    self.pref_student_episodes.append({
        'obs': torch.tensor(obs_ep, dtype=torch.float32),
        'acs': torch.tensor(acs_ep, dtype=torch.float32),
        'J': J
    })
    # Prune by recency (TF parity: keep last pref_max_student_trajs)
    if (self.pref_max_student_trajs is not None
            and len(self.pref_student_episodes) > self.pref_max_student_trajs):
        self.pref_student_episodes = self.pref_student_episodes[-self.pref_max_student_trajs:]
```

**New method `sample_qpref_pairs(batch_size, source='teacher')`:**
```python
def sample_qpref_pairs(self, batch_size: int, source: str = 'teacher'):
    """
    TF-faithful: sample B (pos_s, pos_a, neg_s, neg_a) pairs.
    Each pair: one random timestep from pos traj, one from neg.
    Returns tensors on self.device or None if fewer than 2 episodes.
    """
    eps = self.pref_episodes if source == 'teacher' else self.pref_student_episodes
    if (eps is None) or (len(eps) < 2):
        return None
    N = len(eps)

    idx_a = np.random.randint(0, N, size=batch_size)
    idx_b = np.random.randint(0, N, size=batch_size)
    for k in range(batch_size):
        if idx_b[k] == idx_a[k]:
            idx_b[k] = (idx_b[k] + 1) % N

    pos_obs_list, pos_acs_list = [], []
    neg_obs_list, neg_acs_list = [], []
    for a, b in zip(idx_a, idx_b):
        Ja = float(eps[a]['J'])
        Jb = float(eps[b]['J'])
        epP, epN = (eps[a], eps[b]) if Ja >= Jb else (eps[b], eps[a])

        # TF parity: one random timestep per trajectory
        tP = np.random.randint(0, epP['acs'].shape[0])
        tN = np.random.randint(0, epN['acs'].shape[0])

        pos_obs_list.append(epP['obs'][tP])
        pos_acs_list.append(epP['acs'][tP])
        neg_obs_list.append(epN['obs'][tN])
        neg_acs_list.append(epN['acs'][tN])

    pos_obs = torch.stack(pos_obs_list).to(self.device)  # [B, obs_dim]
    pos_acs = torch.stack(pos_acs_list).to(self.device)  # [B, act_dim]
    neg_obs = torch.stack(neg_obs_list).to(self.device)  # [B, obs_dim]
    neg_acs = torch.stack(neg_acs_list).to(self.device)  # [B, act_dim]
    return pos_obs, pos_acs, neg_obs, neg_acs
```

---

### Step 3: New fields and logic in `sail.py`

**`__init__` new params:**
```python
qpref: bool = False,
qpref_weight: float = 0.0,
qpref_temp: float = 1.0,
qpref_batch_size: int = 32,
qpref_start_step: int = 0,
qpref_source: str = 'teacher',
pref_rm = None,               # pass the loaded pref RM (same as already used)
```

Store all on `self`.

**`__init__` new pending metric field:**
```python
self._pending_qpref_losses: list = []
```

**New method `compute_qpref_loss(pos_obs, pos_acs, neg_obs, neg_acs)`:**
```python
def compute_qpref_loss(self, pos_obs, pos_acs, neg_obs, neg_acs):
    """
    TF-faithful: min(q1,q2) at sampled pairs, Bradley-Terry softplus loss.
    Uses CURRENT critic (not target), same as TF reuse=True.
    """
    q1_pos, q2_pos = self.critic(pos_obs, pos_acs)     # [B,1], [B,1]
    q1_neg, q2_neg = self.critic(neg_obs, neg_acs)     # [B,1], [B,1]

    q_pos = torch.minimum(q1_pos, q2_pos)              # [B,1]
    q_neg = torch.minimum(q1_neg, q2_neg)              # [B,1]

    temp  = max(float(self.qpref_temp), 1e-6)
    delta = (q_pos - q_neg) / temp                     # [B,1]

    loss = F.softplus(-delta).mean()                   # scalar ≥ 0
    return loss
```

**In `train()` — add QPREF to critic update loop:**

Current structure:
```python
# --- Critic Update ---
current_q1, current_q2 = self.critic(mixed_obs, mixed_acts)
critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
```

New structure:
```python
# --- Critic Update (TD error) ---
current_q1, current_q2 = self.critic(mixed_obs, mixed_acts)
td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

# --- QPREF loss (optional) ---
qpref_loss_val = 0.0
use_qpref = (
    self.qpref
    and self.qpref_weight > 0.0
    and self.num_timesteps >= self.qpref_start_step
)
if use_qpref:
    pair = self.teacher_buffer.sample_qpref_pairs(
        self.qpref_batch_size, source=self.qpref_source)
    if pair is not None:
        pos_obs, pos_acs, neg_obs, neg_acs = pair
        qpref_loss = self.compute_qpref_loss(pos_obs, pos_acs, neg_obs, neg_acs)
        qpref_loss_val = qpref_loss.item()
        self._pending_qpref_losses.append(qpref_loss_val)
    else:
        qpref_loss = torch.tensor(0.0, device=self.device)
else:
    qpref_loss = torch.tensor(0.0, device=self.device)

critic_loss = td_loss + self.qpref_weight * qpref_loss

self.critic.optimizer.zero_grad()
critic_loss.backward()
self.critic.optimizer.step()
```

**In `train()` — drain and log qpref:**
```python
if self._pending_qpref_losses:
    self.logger.record("train/qpref_loss", np.mean(self._pending_qpref_losses))
    self._pending_qpref_losses = []
```

**Student episode accumulation:**

In `_store_transition` or a new `on_episode_end` callback, when `qpref_source='student'`
and a new episode completes: call `self.teacher_buffer.add_student_episode(obs_ep, acs_ep, self.pref_rm)`.

The cleanest integration: add a `SAILAdaptiveCallback.on_rollout_end()` hook OR add a
check in `_store_transition` using the `infos` done flag (same pattern as adaptive callback).
Recommended: add to `SAILAdaptiveCallback` in `utils/callbacks.py` since it already
handles episode-end logic (lines where promotion is decided).

---

### Step 4: `train_sail.py` wiring

After TeacherBuffer construction:
```python
if args.qpref:
    sail_kwargs['qpref'] = True
    sail_kwargs['qpref_weight'] = args.qpref_weight
    sail_kwargs['qpref_temp'] = args.qpref_temp
    sail_kwargs['qpref_batch_size'] = args.qpref_batch_size
    sail_kwargs['qpref_start_step'] = args.qpref_start_step
    sail_kwargs['qpref_source'] = args.qpref_source
    sail_kwargs['pref_rm'] = pref_rm    # already loaded if need_pref_rm=True
```

Print startup diagnostics:
```python
if args.qpref:
    pool_size = len(teacher_buffer.pref_episodes)   # teacher always ready
    print(f"[train_sail] QPREF: source={args.qpref_source} weight={args.qpref_weight} "
          f"temp={args.qpref_temp} batch={args.qpref_batch_size} "
          f"start_step={args.qpref_start_step} teacher_pool={pool_size}")
```

---

### Summary of Variant 1 changes

| File | Changes |
|------|---------|
| `train_sail.py` | 7 new args, wiring to SAIL, startup print, `need_pref_rm` update |
| `sail.py` | 7 new params, `_pending_qpref_losses`, `compute_qpref_loss()`, critic loss update, logging |
| `teacher_buffer.py` | `pref_student_episodes` list, `add_student_episode()`, `sample_qpref_pairs()` |
| `utils/callbacks.py` | If `qpref_source=student`: call `add_student_episode()` at episode end |

---

## Section 5 — PyTorch Plan for Trajectory-Aggregate QPREF (Variant 2)

### Design decision: mean-Q vs sum-Q

Use **mean-Q**. Reasons:
1. More principled than sum-Q when comparing trajectories of different lengths
2. HalfCheetah is fixed-length (T=1000), so sum and mean differ only by 1/1000 — use mean
   for generality across environments
3. Mean-Q has smaller absolute values than sum-Q, requiring lower temperature to produce
   comparable delta magnitudes — but this is a tuning matter, not a correctness issue

---

### New `sample_qpref_pairs_aggregate(batch_size, source='teacher')` in `teacher_buffer.py`

```python
def sample_qpref_pairs_aggregate(self, batch_size: int, source: str = 'teacher'):
    """
    Variant 2: returns full padded trajectories for mean-Q aggregation.
    Same pair selection as Variant 1, but returns all timesteps per traj
    with a mask, not a single random step.

    Returns:
        pos_obs:  [B, T_max, obs_dim]
        pos_acs:  [B, T_max, act_dim]
        pos_mask: [B, T_max]   -- 1.0 for valid steps
        neg_obs:  [B, T_max, obs_dim]
        neg_acs:  [B, T_max, act_dim]
        neg_mask: [B, T_max]
    or None if fewer than 2 episodes.
    """
    eps = self.pref_episodes if source == 'teacher' else self.pref_student_episodes
    if (eps is None) or (len(eps) < 2):
        return None

    N = len(eps)
    idx_a = np.random.randint(0, N, size=batch_size)
    idx_b = np.random.randint(0, N, size=batch_size)
    for k in range(batch_size):
        if idx_b[k] == idx_a[k]:
            idx_b[k] = (idx_b[k] + 1) % N

    pos_eps_sel, neg_eps_sel = [], []
    for a, b in zip(idx_a, idx_b):
        epP, epN = (eps[a], eps[b]) if float(eps[a]['J']) >= float(eps[b]['J']) else (eps[b], eps[a])
        pos_eps_sel.append(epP)
        neg_eps_sel.append(epN)

    # Pad to T_max within this batch
    def T(ep): return int(ep['acs'].shape[0])
    T_max = max(max(T(ep) for ep in pos_eps_sel), max(T(ep) for ep in neg_eps_sel))
    obs_dim = pos_eps_sel[0]['obs'].shape[-1]
    act_dim = pos_eps_sel[0]['acs'].shape[-1]
    B = batch_size

    pos_obs  = torch.zeros(B, T_max, obs_dim, dtype=torch.float32)
    pos_acs  = torch.zeros(B, T_max, act_dim, dtype=torch.float32)
    pos_mask = torch.zeros(B, T_max, dtype=torch.float32)
    neg_obs  = torch.zeros(B, T_max, obs_dim, dtype=torch.float32)
    neg_acs  = torch.zeros(B, T_max, act_dim, dtype=torch.float32)
    neg_mask = torch.zeros(B, T_max, dtype=torch.float32)

    for i in range(B):
        epP, epN = pos_eps_sel[i], neg_eps_sel[i]
        LP, LN = T(epP), T(epN)
        pos_obs[i, :LP]  = epP['obs'][:LP]
        pos_acs[i, :LP]  = epP['acs'][:LP]
        pos_mask[i, :LP] = 1.0
        neg_obs[i, :LN]  = epN['obs'][:LN]
        neg_acs[i, :LN]  = epN['acs'][:LN]
        neg_mask[i, :LN] = 1.0

    return (pos_obs.to(self.device), pos_acs.to(self.device), pos_mask.to(self.device),
            neg_obs.to(self.device), neg_acs.to(self.device), neg_mask.to(self.device))
```

Note: This format is identical to `sample_pref_pairs()` already in `teacher_buffer.py`
for `pref_rank_disc`. Can reuse that method or factor common padding logic.

---

### New `compute_qpref_loss_aggregate(...)` in `sail.py`

```python
def compute_qpref_loss_aggregate(self, pos_obs, pos_acs, pos_mask,
                                  neg_obs, neg_acs, neg_mask):
    """
    Variant 2: mean-Q over each trajectory, then softplus ranking loss.
    pos_obs: [B, T_max, obs_dim], pos_mask: [B, T_max]
    """
    B, T_max, obs_dim = pos_obs.shape
    act_dim = pos_acs.shape[-1]

    # Flatten to [B*T_max, obs_dim] for batch critic evaluation
    pos_obs_flat = pos_obs.reshape(B * T_max, obs_dim)
    pos_acs_flat = pos_acs.reshape(B * T_max, act_dim)
    neg_obs_flat = neg_obs.reshape(B * T_max, obs_dim)
    neg_acs_flat = neg_acs.reshape(B * T_max, act_dim)

    q1_pos_flat, q2_pos_flat = self.critic(pos_obs_flat, pos_acs_flat)  # [B*T_max, 1]
    q1_neg_flat, q2_neg_flat = self.critic(neg_obs_flat, neg_acs_flat)

    q_pos_flat = torch.minimum(q1_pos_flat, q2_pos_flat).reshape(B, T_max)  # [B, T_max]
    q_neg_flat = torch.minimum(q1_neg_flat, q2_neg_flat).reshape(B, T_max)

    # Masked mean over valid timesteps
    q_pos_mean = (q_pos_flat * pos_mask).sum(dim=1) / (pos_mask.sum(dim=1) + 1e-8)  # [B]
    q_neg_mean = (q_neg_flat * neg_mask).sum(dim=1) / (neg_mask.sum(dim=1) + 1e-8)  # [B]

    temp  = max(float(self.qpref_temp), 1e-6)
    delta = (q_pos_mean - q_neg_mean) / temp  # [B]

    loss = F.softplus(-delta).mean()          # scalar ≥ 0
    return loss
```

**Computational cost:** B × T_max forward passes through the critic per critic gradient step.
For B=32, T_max=1000: 32,000 critic evaluations per step. At batch_size=256 for the
main critic update, this is 125× more critic FLOPs per gradient step. This is
computationally expensive and likely infeasible at the same `qpref_batch_size=32` as
Variant 1. Recommend `qpref_batch_size=4-8` for Variant 2.

**Integration in `train()`:**
```python
if use_qpref_aggregate:
    pair = self.teacher_buffer.sample_qpref_pairs_aggregate(
        self.qpref_batch_size, source=self.qpref_source)
    if pair is not None:
        pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask = pair
        qpref_loss = self.compute_qpref_loss_aggregate(
            pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask)
```

Flag to select variant: add `--qpref_aggregate` bool flag. Default False (TF-faithful).

---

### Variant 2 summary: what changes vs Variant 1

| Component | Variant 1 | Variant 2 |
|-----------|-----------|-----------|
| Sampling | single random step | all steps (padded) |
| Q eval | 2×B forward passes | 2×B×T_max forward passes |
| Aggregation | none | masked mean |
| Q shape into loss | [B,1] | [B] (after mean) |
| Compute cost | ~negligible | heavy (~125× critic FLOPs) |
| Flag | default | `--qpref_aggregate` |

---

## Section 6 — Recommendation

### Implement Variant 1 (TF-faithful) first.

**Reasons:**

1. **Baseline parity.** You need to verify that the PyTorch port of QPREF produces
   comparable behavior to TF before varying the design. Variant 1 gives a clean
   comparison to the TF runs.

2. **Compute budget.** Variant 2 is 100× more expensive per critic step. For a 100k
   smoke test on CPU that already takes 20-50 min, Variant 2 could take 30+ hours for
   the same timesteps. Variant 1 adds ~negligible compute (32 critic forward passes).

3. **Single random Q is sufficient for ordering.** The signal is noisy per step but
   consistent over many gradient steps. TD3 already runs 1000 gradient steps per 1000
   env steps — QPREF fires 1000 times per env-step block. Even high per-step noise
   averages out quickly.

4. **TF used Variant 1 and it works.** The TF codebase has QPREF active in runs that
   were functional. Faithfully porting it first is the scientific baseline.

### Implementation order

1. Implement Variant 1: `teacher_buffer.py` → `sail.py` → `train_sail.py` → sanity checks
2. Run a 100k smoke test with `--qpref --qpref_source teacher`
3. Run a 100k smoke test with `--qpref --qpref_source student` (exercises student pool)
4. Only if Variant 1 shows signal in 1M runs: implement Variant 2 and compare

### Recommended default flags for first full run

```bash
--qpref
--qpref_weight 0.1
--qpref_temp 1.0
--qpref_batch_size 32
--qpref_source teacher
--adaptive --adaptive_score_source rm
--pref_rank_disc --pref_rank_weight 0.1 --pref_rank_batch_size 16
--pref_reweight_teacher --pref_beta 1.0
--pref_rm ${PREF_RM}
--entcoeff 0.05
--lfd_mixing
--teacher_buffer_size 1000
```

Rationale: `qpref_source=teacher` because pref_teacher_episodes (SepPool) is ready at
timestep 0 with 4 expert episodes. `qpref_source=student` requires waiting for 2+
rollout episodes and adds per-episode pref RM scoring overhead.

### Ambiguities / open questions

1. **Gradient flow through `min(q1, q2)`.** PyTorch's `torch.minimum` produces gradients
   only through the element that was smaller (like `min` in TF). This means per sample,
   only one critic head gets QPREF gradient. This is correct TF-parity behavior.

2. **student episode obs format.** `_store_transition` in PyTorch SAIL does not currently
   accumulate full episode obs. The student episode obs must be collected by the adaptive
   callback (which already has episode obs from the promotion check). Confirm the obs
   include the time feature (18-dim for HalfCheetah) before passing to pref RM.

3. **pref_rm object availability in SAIL.** Currently `pref_rm` is loaded in `train_sail.py`
   and used to score promotions in the callback. For student QPREF, `SAIL` needs access
   to the same `pref_rm` object. Two options: (a) pass `pref_rm` to `SAIL.__init__`,
   (b) keep scoring in the callback and call `teacher_buffer.add_student_episode()` from
   the callback. Option (b) is cleaner — no new SAIL dependency, callback already has
   the pref_rm reference.

4. **Whether QPREF and pref_rank_disc are additive.** Yes — they operate on different
   networks (QPREF on critic, pref_rank_disc on discriminator) with different batches.
   They share `pref_teacher_episodes` (read-only from their perspective). No conflict.

5. **qpref_start_step=0 with SepPool.** With SepPool, teacher pool is built before
   training. So `qpref_source=teacher` with start_step=0 fires from the very first
   critic update at 10k steps (learning_starts). This is the intended TF behavior.
