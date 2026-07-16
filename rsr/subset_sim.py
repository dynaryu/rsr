"""
Subset Simulation (Au & Beck 2001) for RSR reference-state extraction.

Implements a discrete-state variant of the component-wise Metropolis-Hastings
Subset Simulation originally developed for continuous reliability problems.

Key adaptations for RSR discrete multi-state components:

  1. Proposal: sticky independence proposal using the component marginal prior.
     For each component i independently, with probability beta keep the current
     state, else resample from prior_i. Because RSR assumes independent
     components, the joint M-H acceptance ratio reduces to 1 and the effective
     acceptance becomes I(candidate in F_j). One sfun call per chain step.

  2. Severity: sfun returns (fval, sys_st, _). We define a transformed severity
     s = severity_sign * fval so that higher s always means "more failed". The
     level threshold b_j is the p0-quantile of seed severities.

  3. Termination: SuS levels halt as soon as every seed at the current level is
     an actual failure (sys_st < sys_surv_st). All collected failed samples are
     returned to the caller for rule minimisation.

This module is self-contained and is intended to be wired into
`run_ref_extraction_by_mcs` as an alternative search-phase sampler. The
unbiased probability estimation phase remains prior-based and is handled by the
caller.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Parallel-worker shared state (set by the caller BEFORE the fork-based Pool
# is created so that child processes inherit it via fork()).
# ---------------------------------------------------------------------------
_SUS_SFUN: Optional[Callable] = None
_SUS_ROW_NAMES: Optional[List[str]] = None
_SUS_SYS_SURV_ST: Optional[int] = None


def set_worker_state(sfun: Callable, row_names: List[str], sys_surv_st: int = -1) -> None:
    """Populate module globals inherited by forked workers.

    sys_surv_st is currently unused inside the worker but is stored for future
    extensions (e.g. workers that pre-classify samples).
    """
    global _SUS_SFUN, _SUS_ROW_NAMES, _SUS_SYS_SURV_ST
    _SUS_SFUN = sfun
    _SUS_ROW_NAMES = row_names
    _SUS_SYS_SURV_ST = sys_surv_st


def _sus_eval_worker(sts_tuple: Tuple[int, ...]) -> Tuple[float, int]:
    """Worker: evaluate sfun on a single state tuple; return (fval, sys_st)."""
    sfun = _SUS_SFUN
    names = _SUS_ROW_NAMES
    cst = {names[k]: int(sts_tuple[k]) for k in range(len(names))}
    fval, sys_st, _ = sfun(cst)
    return float(fval), int(sys_st)


# ---------------------------------------------------------------------------
# Batch sfun evaluation
# ---------------------------------------------------------------------------
def eval_batch(
    states: torch.Tensor,
    sfun: Callable,
    row_names: List[str],
    pool=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate sfun on a batch of states.

    Args:
        states: (N, n_var) int tensor of component states.
        sfun: system function with signature sfun(comps_st) -> (fval, sys_st, _).
        row_names: ordered list of component names.
        pool: optional multiprocessing.Pool; if None, runs serially.

    Returns:
        fvals: (N,) float64 numpy array.
        sys_sts: (N,) int64 numpy array.
    """
    states_list = states.detach().cpu().tolist()
    tasks = [tuple(row) for row in states_list]

    if pool is not None:
        results = pool.map(_sus_eval_worker, tasks)
    else:
        # Serial path: make sure globals are set (idempotent if caller already did).
        # We do NOT clobber an existing setting since the caller may have configured
        # sys_surv_st correctly already.
        if _SUS_SFUN is None:
            set_worker_state(sfun, row_names, sys_surv_st=-1)
        results = [_sus_eval_worker(t) for t in tasks]

    fvals = np.fromiter((r[0] for r in results), dtype=np.float64, count=len(results))
    sys_sts = np.fromiter((r[1] for r in results), dtype=np.int64, count=len(results))
    return fvals, sys_sts


# ---------------------------------------------------------------------------
# Prior sampler for discrete multi-state components
# ---------------------------------------------------------------------------
def sample_prior_states(probs: torch.Tensor, n: int) -> torch.Tensor:
    """
    Draw n samples from the per-component categorical prior.

    Args:
        probs: (n_var, n_state) float tensor.
        n: number of samples.

    Returns:
        states: (n, n_var) int64 tensor of component state indices.
    """
    device = probs.device
    n_var, n_state = probs.shape
    cum = torch.cumsum(probs, dim=1)                # (n_var, n_state)
    r = torch.rand(n, n_var, device=device)         # (n, n_var)
    # Vectorised inverse-CDF: count how many cumulative bins r exceeds.
    # r:   (n, n_var, 1); cum: (1, n_var, n_state) -> compare and sum along last dim.
    states = (r.unsqueeze(-1) >= cum.unsqueeze(0)).sum(dim=-1).clamp_(max=n_state - 1)
    return states.to(torch.int64)


# ---------------------------------------------------------------------------
# CWM-H chain advance with sticky-prior independence proposal
# ---------------------------------------------------------------------------
def cwmh_chain(
    seed_states: torch.Tensor,
    seed_fvals: np.ndarray,
    seed_sys_sts: np.ndarray,
    probs: torch.Tensor,
    threshold_s: float,
    severity_sign: int,
    chain_length: int,
    sfun: Callable,
    row_names: List[str],
    n_flip_mean: float = 5.0,
    pool=None,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    """
    Advance n_seeds parallel chains of total length `chain_length` (seed
    included) using the sticky-prior independence proposal.

    At each step, for every chain we:
      1. Propose a candidate: per component, with probability beta keep the
         current state, else resample from the prior.
      2. Batch-evaluate sfun on all candidates.
      3. Accept if s(candidate) = severity_sign * fval >= threshold_s; else stay.

    beta is derived from n_flip_mean: beta = 1 - n_flip_mean / n_var, so on
    average `n_flip_mean` components are perturbed per step.

    Returns:
        states:  (n_seeds * chain_length, n_var) int tensor
        fvals:   (n_seeds * chain_length,) float64 array
        sys_sts: (n_seeds * chain_length,) int64 array
        n_sfun_calls: total number of sfun evaluations performed
    """
    device = probs.device
    n_seeds, n_var = seed_states.shape

    beta = 1.0 - min(n_flip_mean, float(n_var)) / float(n_var)

    cur_states = seed_states.clone()
    cur_fvals = seed_fvals.copy()
    cur_sys = seed_sys_sts.copy()

    all_states = [cur_states.clone()]
    all_fvals = [cur_fvals.copy()]
    all_sys = [cur_sys.copy()]

    n_sfun_calls = 0

    for _step in range(1, chain_length):
        # Proposal
        flip_mask = torch.rand(n_seeds, n_var, device=device) >= beta  # True -> resample
        cand_fresh = sample_prior_states(probs, n_seeds)                # (n_seeds, n_var)
        cand_states = torch.where(flip_mask, cand_fresh, cur_states)

        # Batch-evaluate sfun on candidates
        cand_fvals, cand_sys = eval_batch(cand_states, sfun, row_names, pool=pool)
        n_sfun_calls += n_seeds

        # Accept / reject (vectorised over chains)
        cand_s = severity_sign * cand_fvals
        accept = cand_s >= threshold_s                        # numpy bool (n_seeds,)
        accept_t = torch.from_numpy(accept).to(device=device)

        cur_states = torch.where(accept_t.unsqueeze(1), cand_states, cur_states)
        cur_fvals = np.where(accept, cand_fvals, cur_fvals)
        cur_sys = np.where(accept, cand_sys, cur_sys)

        all_states.append(cur_states.clone())
        all_fvals.append(cur_fvals.copy())
        all_sys.append(cur_sys.copy())

    states_out = torch.cat(all_states, dim=0)
    fvals_out = np.concatenate(all_fvals)
    sys_out = np.concatenate(all_sys)
    return states_out, fvals_out, sys_out, n_sfun_calls


# ---------------------------------------------------------------------------
# Outer Subset Simulation loop
# ---------------------------------------------------------------------------
def subset_sim_search(
    probs: torch.Tensor,
    sfun: Callable,
    row_names: List[str],
    sys_surv_st: int,
    n_per_level: int = 1000,
    p0: float = 0.1,
    max_levels: int = 10,
    severity_sign: int = +1,
    n_flip_mean: float = 5.0,
    pool=None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Au & Beck (2001) Subset Simulation search for failed states.

    Nested events F_1 supset F_2 supset ... supset F_M are defined by increasing
    thresholds on the transformed severity s = severity_sign * fval. At each
    level the top p0 fraction of samples by severity become the seeds for the
    next level, and `chain_length = round(1/p0)` CWM-H steps are taken per seed.

    The search terminates at the first level whose seeds are all actual
    failures (sys_st < sys_surv_st), when the level threshold becomes
    degenerate (no further progress possible), or when `max_levels` is
    reached.

    Note: this routine intentionally does NOT compute the SuS probability
    estimate p_F = p0^(M-1) * (n_fail_final / N). Unbiased probability
    estimation is the caller's responsibility (the run_rule_extraction_by_mcs
    main loop maintains a separate prior-based estimator).

    Args:
        probs:         (n_var, n_state) prior probability tensor.
        sfun:          system function.
        row_names:     ordered component names, matching probs rows.
        sys_surv_st:   minimum sys_st considered survival. sys_st < this is failure.
        n_per_level:   N, samples per level (Au & Beck use 1000).
        p0:            intermediate conditional probability (0.1 is standard).
        max_levels:    hard cap on number of SuS levels.
        severity_sign: +1 if higher fval = more failed (e.g. DC-OPF blackout %),
                       -1 if lower fval = more failed.
        n_flip_mean:   average components perturbed per CWM-H step.
        pool:          optional multiprocessing.Pool (caller must call
                       set_worker_state(...) before forking).
        verbose:       print per-level diagnostics.

    Returns:
        dict with keys:
          'failed_states':    (M, n_var) int tensor of samples with sys_st < sys_surv_st
          'failed_fvals':     (M,) float array
          'level_thresholds': list of floats (in original fval space)
          'n_levels':         int, number of levels executed
          'n_sfun_calls':     int
          'final_states':     (n_per_level, n_var) states at the last level
          'final_fvals':      (n_per_level,) fvals at the last level
          'final_sys_sts':    (n_per_level,) sys_sts at the last level
          'terminated_by':    'failure_boundary' | 'max_levels' | 'degenerate_threshold'
    """
    assert severity_sign in (+1, -1), "severity_sign must be +1 or -1"
    assert 0.0 < p0 < 1.0, "p0 must lie in (0, 1)"

    n_seed = max(1, int(round(p0 * n_per_level)))
    chain_length = max(2, int(round(1.0 / p0)))
    # If chain_length * n_seed != n_per_level we still proceed; level size just drifts.
    level_thresholds: List[float] = []
    n_sfun_total = 0

    # ---- Level 0: prior Monte Carlo ----
    states = sample_prior_states(probs, n_per_level)
    fvals, sys_sts = eval_batch(states, sfun, row_names, pool=pool)
    n_sfun_total += n_per_level
    severities = severity_sign * fvals

    if verbose:
        n_fail0 = int((sys_sts < sys_surv_st).sum())
        print(f"[SuS] Level 0 (prior MC): N={n_per_level}, "
              f"fail_samples={n_fail0}, "
              f"fval range=[{fvals.min():.4g}, {fvals.max():.4g}]")

    terminated_by = "max_levels"

    for level in range(1, max_levels + 1):
        # Pick top p0 fraction by transformed severity (largest = worst)
        order = np.argsort(-severities)            # descending
        seed_idx = order[:n_seed]
        seed_sev = severities[seed_idx]
        b_j_s = float(seed_sev[-1])                # smallest severity among seeds
        b_j_fval = b_j_s * severity_sign           # back to original fval space (sign^2 = 1)
        level_thresholds.append(b_j_fval)

        seed_states = states[seed_idx]
        seed_fvals_arr = fvals[seed_idx]
        seed_sys_arr = sys_sts[seed_idx]

        n_fail_in_seeds = int((seed_sys_arr < sys_surv_st).sum())

        if verbose:
            print(f"[SuS] Level {level}: threshold fval={b_j_fval:.4g}, "
                  f"n_seeds={n_seed}, failures_in_seeds={n_fail_in_seeds}/{n_seed}")

        if n_fail_in_seeds >= n_seed:
            terminated_by = "failure_boundary"
            break

        # Degenerate-threshold guard: if the seed threshold equals (or exceeds)
        # the maximum severity in the current level, every subsequent CWM-H
        # candidate would be rejected. This typically signals that the prior
        # has limited support beyond this point or that severity has saturated.
        if b_j_s >= float(severities.max()):
            if verbose:
                print(f"[SuS] Level {level}: degenerate threshold "
                      f"(b_j={b_j_s:.4g} >= max severity); halting.")
            terminated_by = "degenerate_threshold"
            break

        # Shuffle seed order (standard Au & Beck practice)
        perm = np.random.permutation(n_seed)
        seed_states = seed_states[torch.from_numpy(perm).to(seed_states.device)]
        seed_fvals_arr = seed_fvals_arr[perm]
        seed_sys_arr = seed_sys_arr[perm]

        # Run chains
        states, fvals, sys_sts, n_calls = cwmh_chain(
            seed_states=seed_states,
            seed_fvals=seed_fvals_arr,
            seed_sys_sts=seed_sys_arr,
            probs=probs,
            threshold_s=b_j_s,
            severity_sign=severity_sign,
            chain_length=chain_length,
            sfun=sfun,
            row_names=row_names,
            n_flip_mean=n_flip_mean,
            pool=pool,
        )
        n_sfun_total += n_calls
        severities = severity_sign * fvals

    all_failed_mask = sys_sts < sys_surv_st
    failed_states = states[torch.from_numpy(all_failed_mask).to(states.device)]
    failed_fvals = fvals[all_failed_mask]

    if verbose:
        print(f"[SuS] Done: levels={len(level_thresholds)}, "
              f"terminated_by={terminated_by}, "
              f"failed_samples={int(all_failed_mask.sum())}, "
              f"sfun_calls={n_sfun_total}")

    return {
        "failed_states": failed_states,
        "failed_fvals": failed_fvals,
        "level_thresholds": level_thresholds,
        "n_levels": len(level_thresholds),
        "n_sfun_calls": n_sfun_total,
        "final_states": states,
        "final_fvals": fvals,
        "final_sys_sts": sys_sts,
        "terminated_by": terminated_by,
    }
