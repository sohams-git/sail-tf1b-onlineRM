"""
SegmentStore — fixed-length trajectory segment buffer for online preference RM training.

Design (v2 plan, Option B):
  - Stores individual fixed-length segments (not pre-built pairs).
  - Preference pairs are sampled dynamically at RM training time.
  - Labels are assigned at sample time from cached per-segment true returns.
  - Supports teacher and student segments with source tracking.

Labels use raw undiscounted true environment return:
    ret(segment) = sum(r_t for t in [start, end))
No discounting, no normalization, no RM-derived pseudo-labels.
"""

import numpy as np
import torch


# Source identifiers
SOURCE_TEACHER = 0
SOURCE_STUDENT = 1


class SegmentStore:
    """
    Ring buffer of fixed-length trajectory segments.

    Each slot stores:
        obs   : [segment_len, obs_dim]  float32
        act   : [segment_len, act_dim]  float32
        mask  : [segment_len]           float32  (1=valid, 0=pad — for short final segments)
        ret   : scalar float            raw undiscounted sum(r_t) over the segment
        source: int                     SOURCE_TEACHER=0 or SOURCE_STUDENT=1

    Pairing is done at sample time via sample_pair_batch().
    """

    def __init__(self, max_segments: int, segment_len: int,
                 obs_dim: int, act_dim: int, device: torch.device = None):
        """
        Args:
            max_segments: ring buffer capacity (e.g. 10000)
            segment_len:  fixed number of timesteps per segment (e.g. 50)
            obs_dim:      observation dimension
            act_dim:      action dimension
            device:       torch device for returned tensors (None = cpu)
        """
        self.max_segments = max_segments
        self.segment_len  = segment_len
        self.obs_dim      = obs_dim
        self.act_dim      = act_dim
        self.device       = device if device is not None else torch.device("cpu")

        # Pre-allocate ring buffer arrays (numpy, cheap to copy from)
        self._obs    = np.zeros((max_segments, segment_len, obs_dim),  dtype=np.float32)
        self._act    = np.zeros((max_segments, segment_len, act_dim),  dtype=np.float32)
        self._mask   = np.zeros((max_segments, segment_len),           dtype=np.float32)
        self._ret    = np.zeros(max_segments,                          dtype=np.float32)
        self._source = np.zeros(max_segments,                          dtype=np.int8)

        self._ptr  = 0   # next write position
        self._size = 0   # current fill level (≤ max_segments)

        # Separate index lists for fast teacher/student subset sampling
        # These are sets of active ring indices (wrap-around safe)
        self._teacher_indices: set = set()
        self._student_indices: set = set()

    # ------------------------------------------------------------------
    # Core insertion methods
    # ------------------------------------------------------------------

    def add_segment(self, obs: np.ndarray, act: np.ndarray,
                    ret: float, source: int,
                    mask: np.ndarray = None) -> None:
        """
        Write one segment into the ring buffer.

        Args:
            obs:    [segment_len, obs_dim] or shorter (will be zero-padded)
            act:    [segment_len, act_dim] or shorter
            ret:    raw undiscounted return for the segment
            source: SOURCE_TEACHER or SOURCE_STUDENT
            mask:   [segment_len] float32 — if None, inferred from obs length
        """
        T = obs.shape[0]
        slot = self._ptr

        # Remove previous occupant from index sets before overwriting
        if self._size == self.max_segments:
            prev_src = int(self._source[slot])
            if prev_src == SOURCE_TEACHER:
                self._teacher_indices.discard(slot)
            else:
                self._student_indices.discard(slot)

        # Zero-fill slot first (handles padding automatically)
        self._obs[slot]  = 0.0
        self._act[slot]  = 0.0
        self._mask[slot] = 0.0

        write_len = min(T, self.segment_len)
        self._obs[slot,  :write_len] = obs[:write_len]
        self._act[slot,  :write_len] = act[:write_len]
        if mask is not None:
            self._mask[slot, :write_len] = mask[:write_len]
        else:
            self._mask[slot, :write_len] = 1.0

        self._ret[slot]    = float(ret)
        self._source[slot] = int(source)

        # Update index sets
        if source == SOURCE_TEACHER:
            self._teacher_indices.add(slot)
        else:
            self._student_indices.add(slot)

        # Advance ring pointer
        self._ptr = (self._ptr + 1) % self.max_segments
        if self._size < self.max_segments:
            self._size += 1

    def add_from_episode(self, obs_ep: np.ndarray, act_ep: np.ndarray,
                         rew_ep: np.ndarray, source: int) -> int:
        """
        Slice a complete episode into non-overlapping fixed-length segments
        and add each to the store.

        Segments: [0:L], [L:2L], [2L:3L], ...
        Tail shorter than segment_len is discarded (avoids heavy padding).

        Returns the number of segments added.

        Args:
            obs_ep: [T, obs_dim]
            act_ep: [T, act_dim]
            rew_ep: [T]         raw environment rewards
            source: SOURCE_TEACHER or SOURCE_STUDENT
        """
        T = len(obs_ep)
        L = self.segment_len
        n_segs = T // L
        if n_segs == 0:
            return 0

        for k in range(n_segs):
            start = k * L
            end   = start + L
            seg_obs = obs_ep[start:end]
            seg_act = act_ep[start:end]
            seg_ret = float(np.sum(rew_ep[start:end]))
            # All steps are valid (exact length L, no padding needed)
            self.add_segment(seg_obs, seg_act, seg_ret, source)

        return n_segs

    # ------------------------------------------------------------------
    # Pair sampling — called at RM training time
    # ------------------------------------------------------------------

    def sample_pair_batch(self, batch_size: int,
                          tie_margin: float = 0.5,
                          strategy: str = 'any',
                          max_retries: int = 10):
        """
        Sample batch_size preference pairs dynamically.

        Labels are assigned at sample time:
            label = 1  if ret_2 > ret_1  (segment 2 preferred)
            label = 0  if ret_1 > ret_2  (segment 1 preferred)

        Ties (|ret_1 - ret_2| < tie_margin) are re-sampled up to max_retries
        times. If still tied after retries, the pair is emitted with label=0
        and is logged (high tie rate → reduce tie_margin or increase segment_len).

        Args:
            batch_size:  number of pairs B
            tie_margin:  minimum |ret_1 - ret_2| to form a non-tie pair
            strategy:    'any'            — both indices from full buffer (default)
                         'teacher_student'— idx1 from teacher, idx2 from student
                         'same_source'    — both from same source (randomly chosen)
            max_retries: re-sample attempts per pair when tie condition is met

        Returns dict with keys:
            obs1   : Tensor [B, segment_len, obs_dim]
            act1   : Tensor [B, segment_len, act_dim]
            mask1  : Tensor [B, segment_len]
            obs2   : Tensor [B, segment_len, obs_dim]
            act2   : Tensor [B, segment_len, act_dim]
            mask2  : Tensor [B, segment_len]
            label  : Tensor [B, 1]  float32  (1 if seg2 preferred, 0 otherwise)
            tie_count: int  number of pairs emitted despite tie (diagnostic)
        """
        if self._size < 2:
            raise RuntimeError(
                f"SegmentStore.sample_pair_batch: need >= 2 segments, have {self._size}")

        active = np.arange(self._size) if self._size < self.max_segments else \
                 np.arange(self.max_segments)

        teacher_arr = np.array(sorted(self._teacher_indices), dtype=np.int64)
        student_arr = np.array(sorted(self._student_indices), dtype=np.int64)

        obs1_list,  act1_list,  mask1_list  = [], [], []
        obs2_list,  act2_list,  mask2_list  = [], [], []
        label_list = []
        tie_count  = 0

        for _ in range(batch_size):
            i, j = self._sample_pair_indices(
                active, teacher_arr, student_arr, strategy)

            # Tie-rejection loop
            for attempt in range(max_retries):
                ret_i = float(self._ret[i])
                ret_j = float(self._ret[j])
                if abs(ret_i - ret_j) >= tie_margin:
                    break
                i, j = self._sample_pair_indices(
                    active, teacher_arr, student_arr, strategy)
            else:
                # Exhausted retries — emit anyway, label by current values
                tie_count += 1

            ret_i = float(self._ret[i])
            ret_j = float(self._ret[j])
            label = 1.0 if ret_j > ret_i else 0.0

            obs1_list.append(self._obs[i])
            act1_list.append(self._act[i])
            mask1_list.append(self._mask[i])
            obs2_list.append(self._obs[j])
            act2_list.append(self._act[j])
            mask2_list.append(self._mask[j])
            label_list.append(label)

        def to_tensor(lst):
            return torch.tensor(np.stack(lst, axis=0),
                                dtype=torch.float32, device=self.device)

        return {
            'obs1':      to_tensor(obs1_list),
            'act1':      to_tensor(act1_list),
            'mask1':     to_tensor(mask1_list),
            'obs2':      to_tensor(obs2_list),
            'act2':      to_tensor(act2_list),
            'mask2':     to_tensor(mask2_list),
            'label':     to_tensor([[l] for l in label_list]),
            'tie_count': tie_count,
        }

    def _sample_pair_indices(self, active, teacher_arr, student_arr,
                             strategy: str):
        """Sample two distinct ring indices according to strategy."""
        n = len(active)

        if strategy == 'any':
            i = int(active[np.random.randint(n)])
            j = int(active[np.random.randint(n)])
            while j == i:
                j = int(active[np.random.randint(n)])

        elif strategy == 'teacher_student':
            if len(teacher_arr) == 0 or len(student_arr) == 0:
                # Fallback to any
                return self._sample_pair_indices(active, teacher_arr, student_arr, 'any')
            i = int(teacher_arr[np.random.randint(len(teacher_arr))])
            j = int(student_arr[np.random.randint(len(student_arr))])

        elif strategy == 'same_source':
            # Pick a source that has >= 2 segments
            t_ok = len(teacher_arr) >= 2
            s_ok = len(student_arr) >= 2
            if t_ok and s_ok:
                pool = teacher_arr if np.random.random() < 0.5 else student_arr
            elif t_ok:
                pool = teacher_arr
            elif s_ok:
                pool = student_arr
            else:
                return self._sample_pair_indices(active, teacher_arr, student_arr, 'any')
            idx = np.random.choice(len(pool), size=2, replace=False)
            i, j = int(pool[idx[0]]), int(pool[idx[1]])

        else:
            raise ValueError(f"Unknown strategy: {strategy!r}. "
                             f"Choose 'any', 'teacher_student', or 'same_source'.")

        return i, j

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def is_ready(self, min_segments: int = 256) -> bool:
        """True when the buffer has at least min_segments segments."""
        return self._size >= min_segments

    @property
    def teacher_count(self) -> int:
        return len(self._teacher_indices)

    @property
    def student_count(self) -> int:
        return len(self._student_indices)

    def __len__(self) -> int:
        return self._size

    def return_stats(self):
        """Return dict with min/mean/max/std of cached returns (for logging)."""
        if self._size == 0:
            return {}
        active_ret = self._ret[:self._size] if self._size < self.max_segments \
                     else self._ret
        return {
            'ret_min':  float(np.min(active_ret)),
            'ret_mean': float(np.mean(active_ret)),
            'ret_max':  float(np.max(active_ret)),
            'ret_std':  float(np.std(active_ret)),
        }
