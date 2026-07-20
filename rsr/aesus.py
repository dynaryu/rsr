"""aE-SuS-EC: adaptive-effort Subset Simulation with chain extension.

Python port of ``aE_SuS_EC.m`` from Chan et al.'s benchmark repository
(github.com/chanovo/adaptMCS-benchmarks), the method of:

  J. Chan, I. Papaioannou, D. Straub, "An adaptive subset simulation
  algorithm for system reliability analysis with discontinuous limit
  states", Structural Safety 97:102222, 2022; and
  J. Chan, R. Paredes, I. Papaioannou, L. Duenas-Osorio, D. Straub,
  "Adaptive Monte Carlo methods for estimating rare events in power
  grids", 2024 (doi:10.36227/techrxiv.170664651.36297221).

Why this exists next to ``rsr.subset_sim`` (basic SuS): the basic
component-wise MH sampler does not mix on high-dimensional grids
(measured on ACTIVSg2000, 5206 components: per-run estimates spread 5-10
decades, in both n_flip_mean directions). aE-SuS fixes this with three
ingredients, all ported here:

  1. **Latent-Gaussian space.** The chain lives in R^n standard normals
     u; the discrete component state of variable i is the bin of
     Phi(u_i) in that component's cumulative categorical distribution
     (exactly the mapping used in the reference repo's PIM code).
     Proposals move smoothly — no discrete rejection cliff.
  2. **aCS sampler** (Papaioannou et al.): componentwise conditional
     Gaussian proposal u' = rho*u + sqrt(1-rho^2)*eps with
     sigma = min(1, lambda), rho = sqrt(1-sigma^2); lambda adapted by
     Robbins-Monro toward the optimal 0.44 acceptance rate.
  3. **Adaptive effort with chain extension ("EC").** The level
     threshold is the (p0*N0)-th order statistic; when discrete ties
     make it collide with the sample maximum (a stuck plateau — the
     failure mode of fixed-p0 SuS on discontinuous CDFs), the level's
     chains are *extended* from their tails (per-chain lambda frozen)
     and the sample size grown until at least tol*N0*p0 seeds fall
     strictly below. Level probabilities are the *measured* conditional
     fractions, so the discrete-CDF bias of the naive p0^m estimator
     never enters.

LSF convention (as in the reference): failure iff LSF <= 0. For the
blackout demos LSF(u) = threshold_fval - severity(states(u)), so
"failure" is severity >= threshold_fval — the same event
P(blackout% >= gamma) the benchmarks define.

Deviations from the MATLAB (documented, intentional):
  * chains advance in lockstep so each sweep evaluates ~n_seeds
    candidates in one parallel batch (the reference walks chains
    sequentially); lambda therefore adapts once per sweep from that
    sweep's acceptance rate rather than once per ~100-sample chain
    group;
  * a latent tie-guard: the MATLAB compares the threshold to the sample
    maximum via an undefined variable (``f_sort``) on one branch; we
    implement the intended comparison.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from rsr.subset_sim import eval_batch, pool_run_stats, set_worker_state


# ---------------------------------------------------------------------------
# Latent-Gaussian <-> discrete-state mapping
# ---------------------------------------------------------------------------
def states_from_latent(u: torch.Tensor, cum: torch.Tensor) -> torch.Tensor:
    """Map standard normals (B, n_var) to state indices via inverse CDF.

    ``cum`` is the per-variable cumulative categorical distribution
    (n_var, n_state). Low u -> low Phi(u) -> state 0 (the failed end).
    """
    phi = torch.special.ndtr(u.to(torch.float64))            # (B, n_var)
    st = torch.searchsorted(cum, phi.T.contiguous()).T       # (B, n_var)
    return st.clamp_(max=cum.shape[1] - 1).to(torch.int64)


# ---------------------------------------------------------------------------
# Single aE-SuS-EC run
# ---------------------------------------------------------------------------
def _aesus_single_run(
    lsf_batch: Callable[[torch.Tensor], np.ndarray],
    dim: int,
    n0: int,
    p0: float,
    tol: float,
    max_levels: int,
    lambda0: float,
    scheme: str,
) -> Dict[str, Any]:
    """One aE-SuS-EC run. ``lsf_batch(U) -> (B,) float64``, failure <= 0."""
    n_seed0 = int(round(p0 * n0))
    assert abs(p0 * n0 - n_seed0) < 1e-9, "n0*p0 must be an integer"

    n_lsf = 0

    def lsf(u_batch):
        nonlocal n_lsf
        n_lsf += u_batch.shape[0]
        return lsf_batch(u_batch)

    # ---- aCS / CS chain sweep (lockstep across chains) ----
    def run_chains(seeds, f_seeds, n_new, b_val, b_sign, lam, adapt):
        """Generate ``n_new`` samples conditional on the current level.

        Returns (samples (n_new, dim), f (n_new,), lam_out, tails, f_tails).
        ``adapt`` True -> aCS (shared lambda, Robbins-Monro update per
        sweep); False -> CS (per-chain frozen lambda vector).
        """
        ns = seeds.shape[0]
        nt = n_new + ns
        if adapt:
            perm = torch.randperm(ns)
            seeds, f_seeds = seeds[perm], f_seeds[perm.numpy()]
            lam_ch = np.full(ns, float(lam))
            lam_scalar = float(lam)
        else:
            lam_ch = np.asarray(lam, dtype=np.float64)
            assert lam_ch.shape == (ns,), "CS mode needs one lambda per chain"
            lam_scalar = None

        len_chain = np.full(ns, nt // ns, dtype=np.int64)
        extra = np.random.permutation(ns)[: nt % ns]
        len_chain[extra] += 1
        n_sweeps = int(len_chain.max()) - 1

        cur = seeds.clone()
        f_cur = f_seeds.copy()
        out_u: List[torch.Tensor] = []
        out_f: List[np.ndarray] = []
        it = 1
        for sweep in range(1, n_sweeps + 1):
            active = np.where(len_chain - 1 >= sweep)[0]
            if len(active) == 0:
                break
            lam_act = torch.from_numpy(lam_ch[active]).unsqueeze(1)   # (A,1)
            sigma = torch.clamp(lam_act, max=1.0)
            rho = torch.sqrt(1.0 - sigma ** 2)
            eps = torch.randn(len(active), dim, dtype=torch.float64)
            cand = rho * cur[active] + torch.sqrt(1.0 - rho ** 2) * eps
            f_cand = lsf(cand)
            ok = (f_cand <= b_val) if b_sign == 1 else (f_cand < b_val)
            acc_idx = active[np.where(ok)[0]]
            cur[acc_idx] = cand[torch.from_numpy(np.where(ok)[0])]
            f_cur[acc_idx] = f_cand[np.where(ok)[0]]
            out_u.append(cur[active].clone())
            out_f.append(f_cur[active].copy())
            if adapt:
                acc_rate = float(np.mean(ok))
                lam_scalar = lam_scalar * math.exp(
                    it ** (-0.5) * (acc_rate - 0.44))
                lam_ch[:] = lam_scalar
                it += 1
        samples = torch.cat(out_u) if out_u else torch.empty(0, dim,
                                                             dtype=torch.float64)
        f = np.concatenate(out_f) if out_f else np.empty(0)
        lam_out = lam_ch if not adapt else np.full(ns, lam_scalar)
        return samples, f, lam_out, cur, f_cur

    # ---- level loop (mirrors aE_SuS_EC.m) ----
    lam = lambda0
    cond_probs: List[float] = []
    thresholds: List[float] = []
    level = 0
    b_val, b_sign = math.inf, 1
    samples = torch.empty(0, dim, dtype=torch.float64)
    f_samples = np.empty(0)
    seeds = f_seeds = lam_ch = None
    terminated_by = "max_levels"

    while level < max_levels:
        n_gen = n0 if level == 0 else n0 - seeds.shape[0]
        it = 1
        while True:
            if level == 0:
                new_u = torch.randn(n_gen, dim, dtype=torch.float64)
                new_f = lsf(new_u)
                lam_run = np.full(max(n_seed0, 1), lam)
            else:
                adapt = (it == 1)
                lam_arg = lam if adapt else lam_run
                new_u, new_f, lam_run, tails, f_tails = run_chains(
                    seeds, f_seeds, n_gen, b_val, b_sign, lam_arg, adapt)
            samples = torch.cat([samples, new_u])
            f_samples = np.concatenate([f_samples, new_f])

            order = np.argsort(f_samples, kind="stable")
            samples = samples[torch.from_numpy(order)]
            f_samples = f_samples[order]
            n_cur = len(f_samples)

            if it == 1:
                b_temp = float(f_samples[n_seed0 - 1])
                if b_temp <= 0.0:
                    b_temp = 0.0
                    ns = int(np.sum(f_samples <= 0.0)); sign = 1
                    break
                elif b_temp != float(f_samples[-1]):
                    ns = int(np.sum(f_samples <= b_temp)); sign = 1
                    break
            ns = int(np.sum(f_samples < b_temp)); sign = 0
            if ns >= tol * n0 * p0:
                break
            # ---- adaptive effort: extend the chains ("EC") ----
            if scheme == "geometric":
                n_next = 2 * n_cur
            elif scheme == "arithmetic":
                n_next = n_cur + n0
            else:                                       # 'adaptive'
                n_next = math.ceil(tol * n0 * p0 / max(ns, 1) * n_cur)
            n_gen = n_next - n_cur
            if level > 0:
                seeds, f_seeds = tails, f_tails         # extend from tails
            it += 1

        p_l = ns / n_cur
        cond_probs.append(p_l)
        thresholds.append(b_temp)
        b_val, b_sign = b_temp, sign
        lam = float(np.asarray(lam_run).reshape(-1)[-1])
        seeds = samples[:ns].clone()
        f_seeds = f_samples[:ns].copy()
        samples, f_samples = seeds.clone(), f_seeds.copy()
        level += 1
        if b_val <= 0.0:
            terminated_by = "failure_boundary"
            break

    pf = float(np.prod(cond_probs)) if cond_probs else 0.0
    reached = terminated_by == "failure_boundary" and pf > 0.0
    return {"p_fail": pf if reached else 0.0,
            "cond_probs": cond_probs,
            "thresholds": thresholds,
            "n_levels": level,
            "n_sfun": n_lsf,
            "terminated_by": terminated_by,
            "reached": reached}


# ---------------------------------------------------------------------------
# Public API: multi-run estimate (interface mirrors subset_sim_estimate)
# ---------------------------------------------------------------------------
def aesus_estimate(
    probs: torch.Tensor,
    sfun: Callable,
    row_names: List[str],
    sys_surv_st: int,
    *,
    threshold_fval: float,
    severity_sign: int = +1,
    n_runs: int = 5,
    n0: int = 2000,
    p0: float = 0.1,
    tol: float = 0.8,
    max_levels: int = 50,
    lambda0: float = 0.6,
    scheme: str = "adaptive",
    n_workers: int = 1,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Estimate P(severity >= threshold_fval) by aE-SuS-EC.

    Same multi-run empirical-CI methodology as
    :func:`rsr.subset_sim.subset_sim_estimate` — ``n_runs`` independent
    repetitions, pooled by :func:`rsr.subset_sim.pool_run_stats` — but with
    the adaptive latent-Gaussian engine, which converges on grids where the
    basic sampler cannot mix (paper hyperparameters: N0=2000, p0=0.1,
    tol=0.8).

    Args:
        probs:          (n_var, n_state) categorical prior (moved to CPU).
        sfun:           ``comps_st -> (fval, sys_st, _)``; the graded
                        severity ``fval`` defines the limit state.
        threshold_fval: failure iff severity_sign*fval >= this (e.g. the
                        blackout %% threshold of the demo).
        n0/p0/tol:      aE-SuS effort parameters (N0*p0 must be integer).
        lambda0:        initial aCS spread (reference uses 0.6).
        scheme:         sample-growth scheme: adaptive|geometric|arithmetic.
    """
    import multiprocessing as mp

    probs = probs.detach().cpu().to(torch.float64)
    cum = torch.cumsum(probs, dim=1)
    dim = probs.shape[0]

    set_worker_state(sfun, row_names, sys_surv_st)      # before fork
    pool = mp.get_context("fork").Pool(n_workers) if n_workers > 1 else None

    def lsf_batch(u: torch.Tensor) -> np.ndarray:
        states = states_from_latent(u, cum)
        fvals, _sys = eval_batch(states, sfun, row_names, pool=pool)
        return threshold_fval - severity_sign * np.asarray(fvals,
                                                           dtype=np.float64)

    per_run: List[Dict[str, Any]] = []
    total = 0
    try:
        for r in range(n_runs):
            if seed is not None:
                torch.manual_seed(seed + r)
                np.random.seed(seed + r)
            res = _aesus_single_run(lsf_batch, dim, n0, p0, tol,
                                    max_levels, lambda0, scheme)
            per_run.append(res)
            total += res["n_sfun"]
            if verbose:
                tag = "" if res["reached"] else "  (did not reach failure!)"
                print(f"[aesus] run {r}: p_f={res['p_fail']:.3e}  "
                      f"levels={res['n_levels']} "
                      f"cond={['%.3f' % c for c in res['cond_probs']]} "
                      f"sfun={res['n_sfun']}{tag}", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    out = pool_run_stats(per_run, n_runs, total)
    if verbose and out["ci95"] is not None:
        print(f"[aesus] geometric mean = {out['geom_mean']:.3e}  "
              f"95% CI [{out['ci95'][0]:.3e}, {out['ci95'][1]:.3e}]  "
              f"arithmetic mean = {out['p_fail']:.3e}  "
              f"(c.o.v. single-run {out['cov_single_run']:.2f}, "
              f"spread {out['log10_spread']:.1f} decades)  "
              f"{total:,} sfun over {n_runs} runs")
        if out["converged"] is False:
            print("[aesus] WARNING: NOT CONVERGED — runs disagree beyond a "
                  "decade; raise n0 or inspect the LSF.")
    return out
