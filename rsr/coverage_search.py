"""Coverage-aware Stage-1 boundary search for RSR.

This module is a *prototype* implementing the suggestion from Reviewer 3
(Comment 3.2, and diagnosed in 3.1) of the STRUCS-D-26-00220 response:
make the boundary search "coverage-aware" so that each newly identified
reference state classifies as much of the remaining unclassified sample
pool as possible, thereby reducing the *number* of reference states needed
to drive the unclassified probability p^u down. This is the scaling wall on
large networks (e.g. IEEE-118 hitting R_max = 100,000 with ~16 % still
unclassified), which the O(N log2 M) binary-search speed-up does not touch.

The baseline Stage-1 (rsr.minimise_upper_states_random /
minimise_lower_states_random) resolves each component once, in a *single*
random coordinate order, giving a componentwise-local boundary reference
state. Two knobs are added here on top of the *existing* minimiser:

  * ``n_orders`` : generate several candidate boundary states from the same
    unclassified seed using different random coordinate orders (Reviewer's
    "try several coordinate orders"), then keep the one covering the most
    currently-unclassified samples.

  * greedy batch : across all seeds/candidates in a round, add them in order
    of *marginal* coverage over the still-uncovered unclassified pool
    (Reviewer's "add the candidate, or a small batch of candidates, that
    gives the largest reduction in the unclassified sample pool").

Coverage is measured empirically on the round's own Monte-Carlo unclassified
samples, reusing the exact subset test used by the classifier, so the score
is consistent with how references are applied downstream. Nothing here
changes the semantics of a reference state; it only changes *which* valid
reference states get added, and in what order.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from .rsr import (
    minimise_upper_states_random,
    minimise_lower_states_random,
    from_ref_dict_to_mat,
    _check_any_subset,
    _coverage_minimize_worker,
)


def _ref_coverage_mask(
    ref_dict: Dict[str, Any],
    unknown_flat: torch.Tensor,
    row_names: List[str],
    n_state: int,
) -> torch.Tensor:
    """Boolean mask over ``unknown_flat`` of samples this reference classifies.

    A sample ``x`` (one-hot, flattened) is covered by reference ``r`` iff it is
    a subset of ``r``'s admissible-state matrix — identical to the test in
    :func:`rsr.classify_samples_with_indices`. Works for both an upper
    (survival, ``>=``) and a lower (failure, ``<=``) reference because
    :func:`from_ref_dict_to_mat` encodes the operator into the matrix.

    Args:
        ref_dict: ``{name: (op, state)}`` reference, e.g. from the minimiser.
        unknown_flat: ``(n_unknown, n_var * n_state)`` float16 flattened
            one-hot samples that are currently unclassified.
        row_names: component names in row order.
        n_state: number of component states (M).

    Returns:
        ``(n_unknown,)`` bool tensor, True where the reference covers the sample.
    """
    ref_mat = from_ref_dict_to_mat(ref_dict, row_names, n_state)  # (n_var, n_state)
    not_ref = (~ref_mat.bool()).reshape(1, -1).to(
        dtype=unknown_flat.dtype, device=unknown_flat.device)
    return _check_any_subset(unknown_flat, not_ref)


def coverage_aware_round(
    *,
    samples: torch.Tensor,          # (B, n_var, n_state) one-hot batch
    idx_unknown: torch.Tensor,      # indices into samples of unclassified rows
    sfun: Callable,
    row_names: List[str],
    n_state: int,
    sys_upper_st: int,
    n_seeds: int = 8,               # unclassified seeds examined per round
    n_orders: int = 4,              # coordinate orders tried per seed
    max_add: int = 4,               # references committed per round (greedy)
    generator: Optional[torch.Generator] = None,
    pool: Optional[Any] = None,     # multiprocessing pool for candidate generation
) -> Dict[str, Any]:
    """One coverage-aware Stage-1 round.

    Drop-in replacement for the "we have unknowns -> minimise -> add" block in
    :func:`rsr.run_ref_extraction_by_mcs`. Instead of picking one (or
    ``n_workers``) random unclassified seeds and adding whatever single-order
    boundary they yield, it:

      1. samples ``n_seeds`` unclassified seeds,
      2. builds up to ``n_orders`` candidate boundary references per seed
         (one minimisation per (seed, order); farmed out to ``pool`` when given,
         since the candidate minimisations are independent),
      3. greedily commits the candidates with the largest *marginal* coverage
         over the round's still-uncovered unclassified samples, up to
         ``max_add``.

    ``pool`` is an optional ``multiprocessing`` pool whose workers have the
    system function set as a fork-inherited global (as in
    :func:`rsr.run_ref_extraction_by_mcs`). When ``None``, minimisations run
    serially using the ``sfun`` passed here. The greedy selection is always
    serial (it is cheap tensor work).

    Returns a dict with ``new_upper`` / ``new_lower`` reference-dict lists to
    feed into ``update_refs_batch``, plus bookkeeping: ``n_sfun_upper``,
    ``n_sfun_lower``, ``covered`` (unclassified samples in this batch newly
    covered by the committed set), ``n_unknown`` (batch pool size), and the
    per-candidate coverage trace for diagnostics / plots.
    """
    device = samples.device
    n_unknown = int(idx_unknown.numel())
    out: Dict[str, Any] = {
        "new_upper": [], "new_lower": [],
        "n_sfun_upper": 0, "n_sfun_lower": 0,
        "covered": 0, "n_unknown": n_unknown, "trace": [],
    }
    if n_unknown == 0:
        return out

    # Flattened unclassified pool, matching the classifier's dtype/layout.
    unknown_samples = samples[idx_unknown]                       # (nu, n_var, n_state)
    unknown_flat = unknown_samples.reshape(n_unknown, -1).to(dtype=torch.float16)

    # --- pick seeds from the unclassified pool ---
    n_pick = min(n_seeds, n_unknown)
    perm = torch.randperm(n_unknown, generator=generator, device=device)[:n_pick]

    # --- decide each seed's side (one sfun call per seed, done here) and build
    #     one minimisation task per (seed, coordinate order). The tasks are
    #     independent, so they are farmed out to the worker pool when available;
    #     only the cheap greedy selection below stays serial. ---
    tasks: List[Tuple[Dict[str, int], str, int]] = []
    for rank, p in enumerate(perm.tolist()):
        s0 = unknown_samples[p]
        seed_state = {row_names[k]: int(v)
                      for k, v in enumerate(torch.argmax(s0, dim=1).tolist())}
        _fval, sys_st, _ = sfun(seed_state)
        side = "upper" if sys_st >= sys_upper_st else "lower"
        out[f"n_sfun_{side}"] += 1          # the seed sfun call above
        for j in range(max(1, n_orders)):
            tasks.append((seed_state, side, rank * 1_000 + j))

    if pool is not None:
        # workers read the fork-inherited _MP_* globals (sfun set at pool creation)
        results = pool.map(_coverage_minimize_worker, tasks)
    else:
        # serial fallback: use the sfun passed to this call (no _MP_* globals)
        def _run(task):
            seed_state, side, perm_seed = task
            if side == "upper":
                ref, info = minimise_upper_states_random(
                    seed_state, sfun, sys_upper_st=sys_upper_st, fval=None, seed=perm_seed)
            else:
                ref, info = minimise_lower_states_random(
                    seed_state, sfun, max_state=n_state - 1,
                    sys_lower_st=sys_upper_st - 1, fval=None, seed=perm_seed)
            return side, ref, int(info.get("attempts", 0))
        results = [_run(t) for t in tasks]

    # --- collect candidates and account for minimisation cost ---
    candidates: List[Tuple[str, Dict[str, Any]]] = []   # (side, ref_dict)
    seen: set = set()
    for side, ref, attempts in results:
        out[f"n_sfun_{side}"] += 1 + attempts   # +1 keeps parity with the serial accounting
        key = tuple(sorted((k, op, s) for k, (op, s) in ref.items()))
        if key not in seen:                     # drop duplicate boundaries across orders/seeds
            seen.add(key)
            candidates.append((side, ref))

    if not candidates:
        return out

    # --- precompute each candidate's coverage mask over the pool ---
    masks = [_ref_coverage_mask(c, unknown_flat, row_names, n_state)
             for _, c in candidates]

    # --- greedy max-marginal-coverage selection ---
    remaining = torch.ones(n_unknown, dtype=torch.bool, device=device)
    used = [False] * len(candidates)
    for _ in range(min(max_add, len(candidates))):
        best, best_gain = -1, 0
        for ci, m in enumerate(masks):
            if used[ci]:
                continue
            gain = int((m & remaining).sum().item())
            if gain > best_gain:
                best_gain, best = gain, ci
        if best < 0 or best_gain == 0:
            break                       # nothing left to cover -> stop early
        used[best] = True
        remaining &= ~masks[best]
        side, ref = candidates[best]
        (out["new_upper"] if side == "upper" else out["new_lower"]).append(ref)
        out["trace"].append({"side": side, "marginal_coverage": best_gain})

    out["covered"] = n_unknown - int(remaining.sum().item())
    return out
