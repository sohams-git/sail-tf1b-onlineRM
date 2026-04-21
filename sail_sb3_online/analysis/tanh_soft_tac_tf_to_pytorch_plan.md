# Tanh Soft-TAC: TF Audit → PyTorch Implementation Plan

**Date:** 2026-04-09  
**TF reference:** `stable-baselines/stable_baselines/gail/adversary.py` and `stable_baselines/td3/sail.py`  
**Status:** Plan only — no code changed.

---

## What Soft-TAC (Tanh) Is Trying to Do Conceptually

The discriminator in GAIL/SAIL assigns every (state, action) pair a logit. Summing those over a trajectory gives a discriminator-based episode return J_disc. The standard preference ranking loss (pref_rank_disc) pushes J_disc to be higher for the preferred episode in every pair — but it treats all pairs equally regardless of how strongly RM scores separate them.

Soft-TAC replaces this with an **alignment loss**: it asks "does the discriminator's *ranking* of episode pairs agree with the RM's ranking?" Specifically:

1. Compute J_disc_pos and J_disc_neg (cumulative disc rewards for the two episodes in a pair).
2. Look at the RM-based episode scores J_rm stored at sampling time. Build a discrete label y ∈ {-1, 0, +1}: y=+1 if the RM clearly prefers the positive episode, y=0 if they are tied (within eps), y=-1 if RM reverses the ordering (impossible after pair sorting, but handled by the formula).
3. Loss: `1 - mean(y * tanh(delta_J / T))` where `delta_J = J_disc_pos - J_disc_neg`.
   - When y=+1: loss pushes `tanh(delta_J/T) → 1`, i.e., the disc needs to score the preferred episode higher by a clear margin. `tanh` gives bounded, smooth gradients — it saturates (stops pushing) once the margin is large enough.
   - When y=0 (tied pair): `y*tanh = 0` regardless, so the pair contributes zero gradient. Ties are masked out.
   - Temperature T controls how sharply the tanh saturates: lower T = steeper, more demanding alignment.

The critical conceptual distinction: the standard pref_rank_disc loss pushes the discriminator to rank pairs correctly by always treating the sampled pair as y=+1. Soft-TAC is gentler — tied pairs (|J_rm_pos - J_rm_neg| ≤ eps) are silenced, and the tanh saturation means the disc is not penalized once it already agrees with the RM ordering by a comfortable margin.

---

## Part 1: Complete TF Audit

### 1.1 Loss Definition — `adversary.py:685–720`

```python
# adversary.py lines 685–720
def _compute_soft_tac_loss_unweighted():
    def _disc_step_reward(obs_bt, acs_bt):
        # obs_bt: [B, T, obs_dim]; flatten → [B*T]
        # call discriminator with reuse=True
        logits = disc(obs_flat, acs_flat)                 # [B*T, 1]
        r = -log(1 - sigmoid(logits) + 1e-8)             # [B*T, 1] — same formula as surrogate reward
        r = reshape(r, (B, T))                           # [B, T]
        return r

    r_pos = _disc_step_reward(pref_pos_obs_ph, pref_pos_acs_ph)  # [B, T]
    r_neg = _disc_step_reward(pref_neg_obs_ph, pref_neg_acs_ph)

    J_pos = sum(r_pos * pos_mask, axis=1)                # [B] — disc episode returns
    J_neg = sum(r_neg * neg_mask, axis=1)

    delta_J = J_pos - J_neg                              # [B]
    alpha = 1.0 / max(pref_soft_temp, 1e-6)
    tac_term = pref_soft_label_ph * tanh(alpha * delta_J)  # [B] — label-gated alignment
    tac_alignment = mean(tac_term)                         # scalar ∈ [-1, 1]
    soft_tac_unweighted = 1.0 - tac_alignment
    return soft_tac_unweighted, tac_alignment

soft_tac_loss = pref_soft_weight * soft_tac_unweighted
```

**Total loss (adversary.py:818):**
```
total_loss = generator_loss + expert_loss + entropy_loss + grad_penalty
           + pref_loss          # hard BT softplus ranking (pref_rank_disc)
           + pref_rm_loss       # RM scalar BT softplus (separate, rarely used)
           + soft_tac_loss      # tanh alignment (the new term)
```

### 1.2 Label Computation — `sail.py:1281–1295`

```python
# sail.py lines 1281–1295 (inside discriminator update loop)
if (w_soft > 0.0) and (pos_J is not None) and (neg_J is not None):
    T = config['pref_soft_rank_temp']       # default 1.0
    eps = config['pref_tac_tie_eps']        # default 0.0
    diff = (pos_J - neg_J).reshape(-1)      # [B], always ≥ 0 after pair sorting
    y = zeros_like(diff)
    y[diff > eps]  = +1.0                   # clear preference → active
    y[diff < -eps] = -1.0                   # impossible post-sort, but formula handles it
    # y = 0 → tied, zero gradient
    feed y, T, w_soft into disc placeholders
```

### 1.3 Pair Sampling — `sail.py:819–894` (shared with pref_rank_disc)

`_sample_full_episode_pref_pairs(B)` returns an **8-tuple**:
- `(pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, pos_J, neg_J)`
- `pos_J`, `neg_J`: RM-scored episode returns, shape `[B, 1]`, extracted from `episode['J']`
- Episodes sorted so pos_J ≥ neg_J (higher RM score → positive)
- The same pair data is used for BOTH pref_loss and soft_tac_loss in TF

### 1.4 Data Source — `sail.py:683–728`

```python
# _build_pref_teacher_buffer: runs at startup, feeds pref_teacher_episodes
for each expert episode:
    r_pref = pref_rm.reward(obs_ep, acs_ep)   # per-step RM rewards, called ONCE
    J = float(r_pref.sum())                    # cached in episode['J'] — NOT recomputed online
    episodes.append({'obs': obs_ep, 'acs': acs_ep, 'J': J})
```

**Critical:** J is cached at startup and at each adaptive promotion. The RM is NOT called during discriminator training. Soft-TAC uses the cached J, not fresh RM inference.

### 1.5 Adaptive Promotion Interaction — `sail.py:1574–1605`

```python
# sail.py:1574 — promotion guard
if (config.get('pref_reweight_teacher') or config.get('pref_rank_disc') or config.get('qpref')):
    J = pref_rm.reward(obs_ep, acs_ep).sum()       # score new student episode
    pref_teacher_episodes.append({'obs', 'acs', 'J'})
    # prune to top pref_promote_quantile if > pref_max_teacher_trajs
```

**Note:** `pref_soft_rank_disc` is NOT in this guard. Promoted students only enter the pool when `pref_rank_disc` (or other flags) is also set. All TAC sbatch scripts include `--pref-rank-disc` for this reason.

### 1.6 Teacher Buffer Population Guard — `sail.py:691`

```python
if not (config.get('pref_reweight_teacher') or config.get('pref_rank_disc') or config.get('qpref')):
    return  # teacher buffer not built
```

Same implication: `pref_soft_rank_disc` alone does not trigger teacher buffer building. Must combine with `pref_rank_disc` (or other pool-building flags).

### 1.7 Separate `pref_rm_loss` — `adversary.py:665–683`

This is a **distinct** third preference loss (not soft_tac):
```python
pref_rm_loss = mean(softplus(-(pos_J_rm - neg_J_rm)))   # operates on scalar RM J directly
```
It feeds the RM J scalars as placeholders — no disc forward pass. Not enabled in any sbatch examined. Controlled by `pref_rm_loss_weight` (never set). **NOT planned for this port.**

---

## Part 2: Exact Answers to All 13 Questions

**1. Exact tanh Soft-TAC formula in TF?**
```
soft_tac_loss = w * (1.0 - mean(y * tanh((J_disc_pos - J_disc_neg) / T)))
```
where J_disc = sum_t(-log(1 - sigmoid(disc(s_t, a_t)) + 1e-8) * mask_t).

**2. What quantity is passed into tanh?**
`(J_disc_pos - J_disc_neg) / T` — the **discriminator-based** episode return difference, scaled by 1/T.

**3. Is it RM reward per transition, RM cumulative score, or something else?**
Neither — the **discriminator's own reward** (same formula as surrogate reward: `-log(1-sigmoid(logits)+1e-8)`) is summed per episode to form J_disc. The RM only enters via the discrete labels y.

**4. What is the final soft target range?**
y ∈ {-1, 0, +1}. Not a continuous soft probability — discrete. The "soft" in Soft-TAC refers to the tanh saturation behavior, not to continuous labels.

**5. How is that target used inside discriminator training?**
y multiplies `tanh(delta_J/T)` to gate the gradient. y=0 eliminates gradient for tied pairs. y=+1 pushes delta_J → +∞ (with tanh saturation once margin is large). The loss is added to `total_disc_loss` (along with BCE, entropy, GP).

**6. Does it act on expert samples only, or on both expert and policy samples?**
It acts on the **preference episode pool** (`pref_teacher_episodes`), which at startup contains only expert episodes. After adaptive promotion, it also contains promoted student episodes. It does NOT act on the current policy rollout batch (policy_state/action from replay buffer).

**7. Is it a replacement for BCE labels or an auxiliary loss added on top?**
**Auxiliary — added on top.** The standard BCE losses (generator_loss + expert_loss) remain unchanged. soft_tac_loss is an additional term in `total_loss`.

**8. What are the exact TF config flags controlling it?**
| Flag (CLI) | Config key | Default |
|---|---|---|
| `--pref-soft-rank-disc` | `pref_soft_rank_disc` | False (bool gate) |
| `--pref-soft-rank-weight` | `pref_soft_rank_weight` | 0.0 (PRIMARY control) |
| `--pref-soft-rank-temp` | `pref_soft_rank_temp` | 1.0 |
| `--pref-tac-tie-eps` | `pref_tac_tie_eps` | 0.0 |
Plus: `--pref-rank-disc` (required to populate teacher buffer), `--pref-rank-batch-size` (shared batch size).

**9. What data source does it use in adaptive mode?**
`pref_teacher_episodes`: contains all 4 expert episodes at startup (loaded before ring buffer truncation in PyTorch SepPool). After each successful adaptive promotion, the promoted student episode is appended (scored by pref_rm, pruned by quantile if > max_trajs).

**10. How does it interact with promoted student trajectories?**
Promoted students enter `pref_teacher_episodes` (and thus the Soft-TAC pool) automatically, as a side-effect of `pref_rank_disc` being required. No special handling needed. The label y for a promoted student pair is computed the same way: compare stored J scores.

**11. Is there gating (start step / only after promotions / only with RM loaded)?**
- No start-step gate — activates immediately when pool has ≥2 episodes.
- No "only after promotions" gate — expert episodes alone activate it at step 0.
- Implicit RM requirement: pool building requires RM (sail.py:694 calls `_ensure_pref_teacher_rm()`), so no RM = no pool = no loss.
- Explicit gate: `pref_soft_rank_weight > 0` (sail.py:1228) and pair data is not None.

**12. Does it use the separate episode pool logic, or only transition buffers?**
It uses **the episode pool** (`pref_teacher_episodes` / PyTorch `pref_episodes`). The transition ring buffer is not involved. In PyTorch, this is the SepPool built before ring buffer truncation — exactly what we want.

**13. Sigmoid Soft-TAC vs tanh Soft-TAC — is there a difference?**
There is **only tanh**. The `--pref-soft-rank-disc` help text says "sigma((Jpos-Jneg)/T)" (misleadingly suggesting sigmoid), but the actual TF implementation at `adversary.py:709` uses `tf.tanh`. There is no sigmoid variant. The "soft BT target" description in the CLI help is inaccurate — the actual formula is tanh alignment, not sigmoid cross-entropy.

---

## Part 3: Current PyTorch Codebase — What Exists and What Is Missing

### Already Exists and Can Be Reused

| Component | Location | Usable for Soft-TAC? |
|---|---|---|
| `pref_episodes` pool | `teacher_buffer.py:70` | YES — same pool, same J scores |
| `sample_pref_pairs()` | `teacher_buffer.py:349` | PARTIALLY — needs to return `pos_J, neg_J` |
| `compute_pref_loss()` | `adversary.py:184` | YES — disc J computation already there; needs tanh variant |
| `_update_discriminator()` — pref_rank_disc block | `sail.py:140–153` | YES — can extend in-place |
| pref pool promotion via `add_episode()` | `teacher_buffer.py:137` | YES — already adds to pref_episodes |
| Adaptive callback | `callbacks.py` | YES — no changes needed |

### Missing

1. **`Adversary.compute_soft_tac_loss()`** — tanh alignment loss method does not exist
2. **`sample_pref_pairs()` returning J scores** — current returns 6-tuple without pos_J/neg_J
3. **`SAIL` Soft-TAC fields** — `soft_tac`, `soft_tac_weight`, `soft_tac_temp`, `tac_tie_eps`
4. **Soft-TAC block in `_update_discriminator()`** — no tanh alignment step
5. **CLI flags** — 4 new argparse flags
6. **Logging** — `train/soft_tac_loss` and `train/tac_alignment`

### Architecture Decisions

**Reuse the pref_episodes pool (YES):**
The same pool is used by pref_rank_disc, pref_reweight_teacher, qpref, and soft_tac. No new storage needed.

**Reuse transition ring buffer (NO):**
Soft-TAC uses the episode-level pool, not transition-level samples. Teacher ring buffer remains unchanged.

**Episode-level or transition-level RM values?**
Episode-level (J = sum of per-step RM rewards). The label y is computed from stored J scores. The disc still operates per-step internally (forward pass over all timesteps), but the label assignment is at the episode level.

**TF-faithful behavior:**
1. Share pair data between pref_rank_disc and soft_tac in a single discriminator gradient step (one `sample_pref_pairs_with_J()` call per step, both losses computed from same pair)
2. Compute disc J using `-log(1-sigmoid(logits)+1e-8)` (identical to surrogate reward formula)
3. Labels y from cached J scores, discrete {-1, 0, +1}
4. No start-step gate, no promotion gate
5. Guard: requires `pref_rank_disc` (or `pref_reweight_teacher`) for pool population

**Requires pref_rank_disc for pool population:**
In PyTorch, `_build_pref_episodes()` is called when `pref_rm_path is not None` (teacher_buffer.py:80). This is already gated on RM path existence. The guard in the PyTorch callback (for promotion) also requires `pref_rank_disc` or similar. Soft-TAC should be documented as requiring `--pref_rank_disc` to be set (even with `--pref_rank_weight 0.0`). This matches all TF usage.

---

## Part 4: Step-by-Step Implementation Plan

### Step 1: Extend `sample_pref_pairs()` in `teacher_buffer.py`

**Purpose:** Return the cached RM J scores alongside episode tensors so the discriminator update can compute labels y without hitting the RM again.

**Change:** Add `return_J: bool = False` parameter.
- When `return_J=True`: return 8-tuple `(pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, pos_J, neg_J)` where `pos_J`, `neg_J` are `torch.Tensor([B], dtype=float32)` on `self.device`.
- When `return_J=False` (default): existing behavior, 6-tuple. Backward compatible — existing callers unchanged.

**No new method needed.** Extend in-place with the parameter.

```python
# Inside sample_pref_pairs() at return statement:
if return_J:
    pos_J = torch.tensor([ep['J'] for ep in pos_ep_list], dtype=torch.float32, device=self.device)
    neg_J = torch.tensor([ep['J'] for ep in neg_ep_list], dtype=torch.float32, device=self.device)
    return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask, pos_J, neg_J
return pos_obs, pos_acs, pos_mask, neg_obs, neg_acs, neg_mask
```

Requires tracking `pos_ep_list` and `neg_ep_list` during the sampling loop (currently the loop builds batch tensors but doesn't store episode refs). Minor refactor of the loop body.

---

### Step 2: Add `compute_soft_tac_loss()` to `Adversary` in `adversary.py`

**Purpose:** Compute the tanh alignment loss from episode pairs and RM-derived labels.

**Signature:**
```python
def compute_soft_tac_loss(
    self,
    pos_obs: torch.Tensor,    # [B, T_max, obs_dim]
    pos_acs: torch.Tensor,    # [B, T_max, act_dim]
    pos_mask: torch.Tensor,   # [B, T_max]
    neg_obs: torch.Tensor,    # [B, T_max, obs_dim]
    neg_acs: torch.Tensor,    # [B, T_max, act_dim]
    neg_mask: torch.Tensor,   # [B, T_max]
    y_labels: torch.Tensor,   # [B] — discrete {-1, 0, +1} from RM score diff
    temp: float = 1.0,        # T — temperature
) -> tuple[torch.Tensor, torch.Tensor]:
    # returns (soft_tac_loss_unweighted, tac_alignment) — both scalars
```

**Implementation:**
```python
def _disc_episode_J(obs_bt, acs_bt, mask_bt):
    B, T = obs_bt.shape[0], obs_bt.shape[1]
    logits = self.forward(obs_bt.reshape(B*T, -1), acs_bt.reshape(B*T, -1))  # [B*T, 1]
    prob = torch.sigmoid(logits)
    r = -torch.log(1.0 - prob + 1e-8)  # same formula as get_reward()
    r = r.reshape(B, T)
    J = (r * mask_bt).sum(dim=1)        # [B]
    return J

J_pos = _disc_episode_J(pos_obs, pos_acs, pos_mask)
J_neg = _disc_episode_J(neg_obs, neg_acs, neg_mask)
delta_J = J_pos - J_neg                           # [B]
alpha = 1.0 / max(temp, 1e-6)
tac_term = y_labels * torch.tanh(alpha * delta_J)  # [B]
tac_alignment = tac_term.mean()                    # scalar
soft_tac_unweighted = 1.0 - tac_alignment
return soft_tac_unweighted, tac_alignment
```

**Note on gradients:** `self.forward()` is called inside `_update_discriminator()` where disc params require grad. This is correct — soft_tac_loss backpropagates through the disc parameters (same as compute_loss and compute_pref_loss).

---

### Step 3: Add Soft-TAC Fields to `SAIL.__init__()` in `sail.py`

New parameters:
```python
soft_tac: bool = False,
soft_tac_weight: float = 0.0,
soft_tac_temp: float = 1.0,
tac_tie_eps: float = 0.0,
```

New instance fields:
```python
self.soft_tac = soft_tac
self.soft_tac_weight = soft_tac_weight
self.soft_tac_temp = max(float(soft_tac_temp), 1e-6)
self.tac_tie_eps = float(tac_tie_eps)
```

New logging accumulator lists:
```python
self._pending_soft_tac_losses: list = []
self._pending_tac_alignments: list = []
```

---

### Step 4: Wire Soft-TAC into `_update_discriminator()` in `sail.py`

**Replace the existing pref_rank_disc block** (lines 140–153) with a unified block that handles both pref_rank_disc and soft_tac, sharing one pair sample:

```python
# Determine if either preference loss is active
need_pairs = ((self.pref_rank_disc and self.pref_rank_weight > 0.0)
              or (self.soft_tac and self.soft_tac_weight > 0.0))

if need_pairs and len(self.teacher_buffer.pref_episodes) >= 2:
    return_J = (self.soft_tac and self.soft_tac_weight > 0.0)
    pair_result = self.teacher_buffer.sample_pref_pairs(
        self.pref_rank_batch_size, return_J=return_J)

    if return_J:
        p_obs, p_acs, p_mask, n_obs, n_acs, n_mask, pos_J, neg_J = pair_result
    else:
        p_obs, p_acs, p_mask, n_obs, n_acs, n_mask = pair_result
        pos_J = neg_J = None

    p_obs, p_acs, p_mask = [t.to(self.device) for t in (p_obs, p_acs, p_mask)]
    n_obs, n_acs, n_mask = [t.to(self.device) for t in (n_obs, n_acs, n_mask)]

    # Hard BT ranking (pref_rank_disc) — uses disc J, always y=+1 implicitly
    if self.pref_rank_disc and self.pref_rank_weight > 0.0:
        pref_loss = self.discriminator.compute_pref_loss(
            p_obs, p_acs, p_mask, n_obs, n_acs, n_mask)
        total_loss = total_loss + self.pref_rank_weight * pref_loss
        pref_loss_val = pref_loss.item()
        self._pending_pref_losses.append(pref_loss_val)

    # Soft-TAC (tanh disc alignment with RM-derived labels)
    if self.soft_tac and self.soft_tac_weight > 0.0 and pos_J is not None:
        diff = (pos_J - neg_J).cpu().numpy()           # [B]
        y = np.zeros_like(diff)
        y[diff > self.tac_tie_eps]  = 1.0
        y[diff < -self.tac_tie_eps] = -1.0
        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)

        tac_unweighted, tac_alignment = self.discriminator.compute_soft_tac_loss(
            p_obs, p_acs, p_mask, n_obs, n_acs, n_mask,
            y_labels=y_tensor, temp=self.soft_tac_temp)
        soft_tac_loss = self.soft_tac_weight * tac_unweighted
        total_loss = total_loss + soft_tac_loss
        self._pending_soft_tac_losses.append(soft_tac_loss.item())
        self._pending_tac_alignments.append(tac_alignment.item())
```

**Interaction with existing code:**
- `total_loss` is the scalar already built from `self.discriminator.compute_loss()`. Adding to it before `.backward()` is correct — same pattern as pref_rank_disc.
- `self.disc_optimizer.zero_grad() / total_loss.backward() / self.disc_optimizer.step()` remain unchanged.
- Both losses share the same episode pair batch (one `sample_pref_pairs()` call). TF-faithful.

---

### Step 5: Add CLI Flags to `train_sail.py`

Four new argparse flags (add next to existing pref_rank_disc flags):

```python
parser.add_argument("--soft_tac",             action="store_true",
    help="Enable Soft-TAC (tanh disc alignment) loss on discriminator. "
         "Requires --pref_rm and --pref_rank_disc (for teacher pool).")
parser.add_argument("--soft_tac_weight",      type=float, default=0.0,
    help="Weight for Soft-TAC loss. Auto-defaults to 0.5 if --soft_tac set without explicit value. "
         "TF reference: --pref-soft-rank-weight.")
parser.add_argument("--soft_tac_temp",        type=float, default=1.0,
    help="Temperature T for tanh(delta_J/T) in Soft-TAC. Lower T → steeper alignment. "
         "TF reference: --pref-soft-rank-temp.")
parser.add_argument("--tac_tie_eps",          type=float, default=0.0,
    help="Tie margin epsilon: pairs with |J_rm_pos - J_rm_neg| <= eps produce y=0 (no gradient). "
         "TF reference: --pref-tac-tie-eps.")
```

Auto-default for weight (like qpref):
```python
if args.soft_tac and args.soft_tac_weight <= 0.0:
    args.soft_tac_weight = 0.5
    print("[train_sail] Soft-TAC: soft_tac_weight not set, defaulting to 0.5")
```

Warning if --soft_tac without --pref_rank_disc:
```python
if args.soft_tac and not args.pref_rank_disc:
    print("[train_sail] WARNING: --soft_tac requires --pref_rank_disc to populate "
          "the teacher pool. Adding --pref_rank_disc automatically.")
    args.pref_rank_disc = True
    # Note: pref_rank_weight can stay 0.0 (only pool building is needed)
```

Update `SAIL(...)` constructor call to pass 4 new fields:
```python
soft_tac=args.soft_tac,
soft_tac_weight=args.soft_tac_weight,
soft_tac_temp=args.soft_tac_temp,
tac_tie_eps=args.tac_tie_eps,
```

Startup print block (after QPREF print):
```python
if args.soft_tac:
    print(f"[train_sail] Soft-TAC: weight={args.soft_tac_weight} temp={args.soft_tac_temp} "
          f"tie_eps={args.tac_tie_eps}  teacher_pool={n_teacher_pool} eps")
```

---

### Step 6: Add Logging in `train()` in `sail.py`

Drain accumulators at the end of `train()`, next to existing disc metric logging:

```python
if self.soft_tac and self._pending_soft_tac_losses:
    self.logger.record("train/soft_tac_loss",
                       float(np.mean(self._pending_soft_tac_losses)))
    self.logger.record("train/tac_alignment",
                       float(np.mean(self._pending_tac_alignments)))
    self._pending_soft_tac_losses.clear()
    self._pending_tac_alignments.clear()
```

**Metrics:**
| Metric | Meaning |
|---|---|
| `train/soft_tac_loss` | `w * (1 - mean(y*tanh(delta/T)))` — lower is better |
| `train/tac_alignment` | `mean(y*tanh(delta/T))` — ranges [-1, 1]; positive = disc agrees with RM |

Optional additional metrics (can add later):
- `train/tac_y_pos_frac` — fraction of pairs with y=+1 (shows how many non-tied pairs)
- `train/tac_delta_j_mean` — mean J_disc_pos - J_disc_neg

---

## Interaction Analysis with Other Codepaths

### With `pref_rank_disc`
- **Compatible.** They share the same pair sample and both add to `total_disc_loss`. Typical TF usage: `pref_rank_weight=0.0, soft_tac_weight=0.5` — hard loss disabled, only soft-TAC active. Also valid to run both simultaneously.
- `pref_rank_disc` MUST be set (even at weight 0.0) for the teacher pool to be populated.

### With `pref_reweight_teacher`
- **Compatible.** Operates on entirely different part of discriminator loss (expert BCE weighting vs auxiliary alignment term). No interaction.

### With `qpref`
- **Compatible.** QPREF modifies the **critic** loss; Soft-TAC modifies the **discriminator** loss. Zero interaction.

### With `adaptive_score_source rm`
- **Compatible.** Adaptive promotion uses RM scores for its threshold logic; soft_tac uses RM-scored J from the episode pool. Both use the same pref_rm but for different purposes.

### Explicitly blocked combinations
- `--soft_tac` without `--pref_rm`: no-op (pool never built). Startup print should warn.
- `--soft_tac` without `--pref_rank_disc`: pool may not be built (TF parity: always pair these). Auto-enable `pref_rank_disc` in train_sail.py with a warning.

---

## TF-Faithful Behavior Notes and Ambiguities

### Resolved

1. **Discriminator reward formula for J_disc:** TF uses `-log(1-sigmoid(logits)+1e-8)` (same as surrogate reward). PyTorch must match this exactly — NOT `softplus(logits)`. The two differ by a small constant at saturation but are architecturally consistent.

2. **Labels are discrete, not sigmoid-soft:** Despite the CLI help text saying "sigma((Jpos-Jneg)/T)", the actual TF code uses discrete {-1, 0, +1}. Use discrete labels.

3. **J scores are cached, not recomputed:** J is computed once at episode creation (startup or promotion) via `pref_rm.reward()`. Not recomputed per disc step. PyTorch already stores J in `pref_episodes[i]['J']`.

4. **Shared pair sample:** Both pref_rank_disc and soft_tac use the same B episode pairs per discriminator gradient step. Don't call `sample_pref_pairs()` twice.

### Ambiguity: Pair sorting direction

After `_sample_full_episode_pref_pairs()` sorts pairs so pos_J ≥ neg_J, label `y[diff < -eps] = -1.0` is unreachable (diff ≥ 0 always). TF's code handles it for generality, but in practice soft_tac with sorted pairs only ever sees y ∈ {0, +1}.

**Plan:** Implement exactly as TF — compute diff, apply the 3-way threshold. The y=-1 path adds negligible overhead and future-proofs the code (e.g., if unsorted pairs are ever used).

### Ambiguity: Interaction with `pref_pair_filter`

TF has an optional `pref_pair_filter` (e.g., `disc_sim_rm_diff`) that filters pairs by discriminator confidence similarity. This is complex and not in PyTorch. For the initial TF-faithful port: use `pref_pair_filter=none` (default behavior, which calls `_sample_full_episode_pref_pairs` directly). Document as out of scope.

---

## Smoke Tests to Write

### Test A: Soft-TAC alone (pref_rank_weight=0.0, soft_tac_weight=0.5)
```
HC_SoftTAC_Adapt_LfD_test.sbatch
--adaptive --lfd_mixing --teacher_buffer_size 1000 --adaptive_score_source gt
--pref_rm ${PREF_RM}
--pref_rank_disc --pref_rank_weight 0.0 --pref_rank_batch_size 16
--soft_tac --soft_tac_weight 0.5 --soft_tac_temp 1.0 --tac_tie_eps 0.0
--entcoeff 0.05 --total_timesteps 100000 --array 1-2
```
Check:
1. `[train_sail] Soft-TAC: ...  teacher_pool=4 eps` at startup
2. `train/soft_tac_loss` appears at first disc update (~10k steps)
3. `train/tac_alignment` is finite, increasing over time (disc learning to agree with RM)
4. `train/disc_loss` shows no regression vs baseline
5. `rollout/ep_rew_mean` follows normal learning curve

### Test B: Soft-TAC + PrefRank combined
```
HC_SoftTAC_AdaptPref_LfD_test.sbatch
Same as A plus: --pref_rank_weight 0.1
```
Check: both `train/disc_loss_pref` and `train/soft_tac_loss` appear and are non-zero.

### Test C: Unit test (no SLURM)
```python
# Verify compute_soft_tac_loss with fake data
disc = Adversary(18, 6)
B, T = 4, 500
pos_obs = torch.randn(B, T, 18); pos_mask = torch.ones(B, T)
neg_obs = torch.randn(B, T, 18); neg_mask = torch.ones(B, T)
pos_acs = torch.randn(B, T, 6); neg_acs = torch.randn(B, T, 6)
y = torch.tensor([1., 1., 0., 1.])  # one tie
loss, align = disc.compute_soft_tac_loss(pos_obs, pos_acs, pos_mask,
                                          neg_obs, neg_acs, neg_mask, y, temp=1.0)
assert 0.0 <= loss.item() <= 2.0
assert -1.0 <= align.item() <= 1.0
# Check backward pass
loss.backward()
assert disc.net[0].weight.grad is not None
```

---

## Files to Modify (Summary)

| File | Change |
|---|---|
| `sail_sb3/datasets/teacher_buffer.py` | Add `return_J: bool = False` to `sample_pref_pairs()` |
| `sail_sb3/reward_models/adversary.py` | Add `compute_soft_tac_loss()` method |
| `sail_sb3/algorithms/sail.py` | Add 4 init params, 2 accumulator lists, soft_tac block in `_update_discriminator()`, 2 metrics in `train()` |
| `sail_sb3/scripts/train_sail.py` | Add 4 argparse flags, auto-default logic, auto-enable pref_rank_disc warning, pass to SAIL() |
| `sail_sb3/HC_SoftTAC_Adapt_LfD_test.sbatch` | New — smoke test A |
| `sail_sb3/HC_SoftTAC_AdaptPref_LfD_test.sbatch` | New — smoke test B |

**No changes to:**
- `callbacks.py` — no new callback logic needed
- `teacher_buffer.py` promotion path — `add_episode()` already adds to `pref_episodes` when pref_rm is loaded

---

## Summary: TF-Faithful vs Optional Improvements

### TF-Faithful (implement now)
- Discrete labels y ∈ {-1, 0, +1}
- J_disc via `-log(1-sigmoid+1e-8)`, summed per episode
- Shared pair sample with pref_rank_disc
- No start-step gate
- Requires pref_rank_disc for pool population (auto-enable with warning)
- Weight, temp, tie_eps as explicit flags

### Optional Improvements (do not implement now)
- Continuous soft labels: `y = sigmoid((pos_J_rm - neg_J_rm) / T_rm)` — smoother, no tie-clipping needed. Not in TF; would require new `--tac_label_mode` flag.
- Joint pair filter (`pref_pair_filter`): select informationally rich pairs (high RM gap, small disc gap). Complex, not in current PyTorch scope.
- Separate temperature for RM label sharpness vs tanh saturation sharpness — TF uses a single T for both; could be split.
