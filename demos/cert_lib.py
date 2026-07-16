"""LP-certificate reference generators for the DC-OPF blackout demos.

Provides the two certificate extractors prototyped in ``proto_cert.py`` as a
library, so they can be plugged into ``rsr.run_ref_extraction_by_mcs`` via its
``ref_generator`` / ``cut_generator`` hooks (see ``blackout_lib.py --cert``):

  * ``CertModel.extract`` — survival box read off ONE utilisation-minimising
    LP solve: the optimal dispatch at a surviving state stays feasible for any
    state whose capacities exceed the utilisations it uses, so rounding those
    utilisations up to the next discrete state gives a ``comp >= state``
    reference. The caller verifies the box corner with one sfun call, giving
    the same soundness contract as the componentwise minimiser (~2 solves per
    reference instead of O(n log m)).

  * ``CertModel.extract_failure_cut`` — failure half-space from ONE
    transportation-relaxation solve at a failed state: weak LP duality bounds
    the served load of EVERY state by W(x) = sum w_i * cap_i(x_i), so
    {x : W(x) <= req} is certified failure with no coherence assumption.

The extractors are DC-OPF specific (they mirror ``func_dcopt_py``'s network
reduction); the rsr hooks they plug into are model-agnostic. The dataset's
``scripts/`` directory must be on ``sys.path`` before constructing CertModel
(``blackout_lib.load_dataset`` already does this).
"""

from copy import deepcopy

import numpy as np
import torch
from scipy.optimize import linprog

# RSR state -> generator capacity fraction (= 1 - gen_state_map of sfun_dcopt)
GEN_FRAC = {0: 0.0, 1: 0.4, 2: 0.8, 3: 1.0}
FLOW_TOL = 1e-6    # p.u.; utilisations below this are treated as unused


# ---------------------------------------------------------------------------
# Dual failure cut (Benders-style half-space certificate)
# ---------------------------------------------------------------------------
class FailureCut:
    """Certified failure region {x : W(x) <= req0}.

    W(x) = sum_d w_d * alive(bus_d) + sum_g w_g * frac(gen_g)
         + sum_l w_l * alive(br_l) * alive(bus_f) * alive(bus_t)

    is an upper bound on the served load at ANY state x, obtained from an
    optimal dual of the transportation relaxation at one failed state (weak
    LP duality; the relaxation itself over-serves the DC-OPF, so the bound
    chain served_dcopf <= served_transport <= W holds with no coherence or
    monotonicity assumption). All weights are >= 0, so W is monotone and the
    certified region is a valid lower (failure) set — but a half-space
    through the lattice rather than a box.
    """

    def __init__(self, bus_terms, gen_terms, line_terms, req0):
        self.bus_terms = bus_terms      # [(name, w)]
        self.gen_terms = gen_terms      # [(name, w, frac_by_state list)]
        self.line_terms = line_terms    # [(w, br_name, fbus_name, tbus_name)]
        self.req0 = req0
        self._compiled = None

    @property
    def n_terms(self):
        return len(self.bus_terms) + len(self.gen_terms) + len(self.line_terms)

    def support(self):
        """Component names with nonzero weight — the only components the cut
        conditions on; everything else is certified regardless of its state."""
        names = {n for n, _ in self.bus_terms}
        names |= {n for n, _, _ in self.gen_terms}
        for _w, b, f, t in self.line_terms:
            names |= {b, f, t}
        return names

    def __getstate__(self):             # keep pickles small (worker -> parent)
        d = dict(self.__dict__)
        d['_compiled'] = None
        return d

    def _compile(self, name_pos, n_state):
        bus = [(name_pos[n], w) for n, w in self.bus_terms]
        gen = [(name_pos[n], torch.tensor(
            [w * f for f in fr] + [w * fr[-1]] * (n_state - len(fr)),
            dtype=torch.float64)) for n, w, fr in self.gen_terms]
        line = [(w, name_pos[b], name_pos[f], name_pos[t])
                for w, b, f, t in self.line_terms]
        self._compiled = (bus, gen, line)

    def W(self, states, name_pos, n_state):
        """Served-load upper bound per sample of a (N, n_var) state-index pool."""
        if self._compiled is None:
            self._compile(name_pos, n_state)
        bus, gen, line = self._compiled
        w = torch.zeros(states.shape[0], dtype=torch.float64)
        for pos, wt in bus:
            w += wt * (states[:, pos] >= 1)
        for pos, table in gen:
            w += table[states[:, pos].long()]
        for wt, pb, pf, pt in line:
            w += wt * ((states[:, pb] >= 1) & (states[:, pf] >= 1)
                       & (states[:, pt] >= 1))
        return w

    def certify(self, states, name_pos, n_state):
        return self.W(states, name_pos, n_state) <= self.req0 - 1e-7

    def batch_tables(self, name_pos, n_state):
        """Vectorisation protocol for rsr's batched cut evaluation
        (:func:`rsr.rsr._compile_cuts`).

        Returns ``(table, product_terms, threshold)`` where ``table`` is a
        (n_var, n_state) float64 tensor with ``table[i, s]`` = this cut's
        linear contribution of component i at state s (bus-alive and
        generator-fraction terms), ``product_terms`` is a list of
        ``(w, (pos_br, pos_f, pos_t))`` for the 3-way line-availability
        terms, and a sample is certified iff its total W <= ``threshold``.
        """
        table = torch.zeros(len(name_pos), n_state, dtype=torch.float64)
        for n, w in self.bus_terms:
            table[name_pos[n], 1:] += w
        for n, w, fr in self.gen_terms:
            row = [w * f for f in fr] + [w * fr[-1]] * (n_state - len(fr))
            table[name_pos[n]] += torch.tensor(row, dtype=torch.float64)
        prod = [(w, (name_pos[b], name_pos[f], name_pos[t]))
                for w, b, f, t in self.line_terms]
        return table, prod, self.req0 - 1e-7


# ---------------------------------------------------------------------------
# Certificate extractor
# ---------------------------------------------------------------------------
class CertModel:
    """Re-implements func_dcopt's network reduction, exposing the LP solution
    so a survival box can be read off the primal utilisations."""

    def __init__(self, case_path, probs_dict, threshold, alpha):
        from func_dcopt_py import load_case, add_branch_capacity, load2disp
        from pypower.idx_bus import BUS_I, PD, GS
        from pypower.idx_gen import GEN_BUS, PG, PMAX, PMIN, GEN_STATUS
        from pypower.idx_brch import F_BUS, T_BUS, RATE_A

        ppc0 = add_branch_capacity(load_case(case_path), alpha)
        self.threshold = threshold
        self.bus_dic = ppc0['bus'][:, BUS_I].astype(int)
        self.gen_dic = ppc0['gen'][:, GEN_BUS].astype(int)
        assert len(set(self.gen_dic)) == len(self.gen_dic), "one gen per bus assumed"
        self.nb, self.ng = len(self.bus_dic), len(self.gen_dic)
        self.nl = ppc0['branch'].shape[0]
        self.baseMVA = ppc0.get('baseMVA', 100.0)

        # mirror func_dcopt's per-call preparation once
        mpc = load2disp(ppc0)
        mpc['bus'][:, PD] = 0
        mpc['bus'][:, GS] = 0
        mpc['gen'][:self.ng, PMIN] = 0
        mpc['gen'][:self.ng, GEN_STATUS] = 1
        self.mpc_base = mpc
        self.totpf = -np.sum(mpc['gen'][self.ng:, PG])          # MW, all loads
        self.pm_orig_pu = ppc0['gen'][:, PMAX] / self.baseMVA   # original caps

        self.probs_dict = probs_dict
        self._gen_state_map = {0: 1.0, 1: 0.6, 2: 0.2, 3: 0.0}
        self.n_solves = 0

        # ---- full-network structures for the transportation relaxation ----
        # (failure cuts; no reduction — component states enter as capacities)
        self.bus_pos = {int(b): i for i, b in enumerate(self.bus_dic)}
        br = self.mpc_base['branch']
        self.br_f_pos = np.array([self.bus_pos[int(b)] for b in br[:, F_BUS]])
        self.br_t_pos = np.array([self.bus_pos[int(b)] for b in br[:, T_BUS]])
        self.rate_pu = br[:, RATE_A].copy() / self.baseMVA
        self.rate_pu[self.rate_pu == 0] = 1e10
        load_gen = self.mpc_base['gen'][self.ng:]
        self.load_bus = load_gen[:, GEN_BUS].astype(int)
        self.load_pd_pu = -load_gen[:, PG] / self.baseMVA
        self.gen_bus_pos = np.array([self.bus_pos[int(b)] for b in self.gen_dic])
        # failure iff served <= req0 (blackout >= threshold)
        self.req0 = (1.0 - self.threshold / 100.0) * self.totpf / self.baseMVA

    # -- state mapping (identical to sfun_dcopt) --------------------------
    def _system_state(self, comps_st):
        ss = np.zeros(self.nb + self.nl)
        gen_bus_set = set(self.gen_dic.tolist())
        for i, bus_id in enumerate(self.bus_dic):
            st = comps_st.get(f"vbus{bus_id}")
            if st is None:
                continue
            if bus_id in gen_bus_set:
                ss[i] = self._gen_state_map.get(st, 0.0)
            else:
                ss[i] = {0: 1.0, 1: 0.0}.get(st, 0.0)
        for j in range(self.nl):
            st = comps_st.get(f"br{j + 1}")
            if st is not None:
                ss[self.nb + j] = {0: 1.0, 1: 0.0}.get(st, 0.0)
        return ss

    def _sf(self, name, s):
        """P(comp >= s) from the dataset's categorical probabilities."""
        p = self.probs_dict[name]
        return sum(v["p"] for k, v in p.items() if int(k) >= s)

    # -- reduction (mirrors func_dcopt) ------------------------------------
    def _reduce(self, comps_st):
        from pypower.idx_gen import GEN_BUS, PMAX
        from pypower.idx_brch import F_BUS, T_BUS

        ss = self._system_state(comps_st)
        bus_state, branch_state = ss[:self.nb], ss[self.nb:]
        gen_pos = np.array([np.where(self.bus_dic == g)[0][0] for g in self.gen_dic])
        gen_state = bus_state[gen_pos]

        mpc = deepcopy(self.mpc_base)
        removed_bus_idx = np.where(bus_state == 1)[0]
        removed_bus_no = self.bus_dic[removed_bus_idx]
        mpc['bus'] = np.delete(mpc['bus'], removed_bus_idx, axis=0)

        mpc['gen'][:self.ng, PMAX] *= (1 - gen_state)
        gen_at_removed = np.isin(mpc['gen'][:, GEN_BUS], removed_bus_no)
        kept_gen_orig_idx = np.where(~gen_at_removed)[0]   # rows into mpc_base gen
        mpc['gen'] = mpc['gen'][~gen_at_removed]

        br_fail = branch_state == 1
        br_from = np.isin(self.mpc_base['branch'][:, F_BUS], removed_bus_no)
        br_to = np.isin(self.mpc_base['branch'][:, T_BUS], removed_bus_no)
        kept_br_idx = np.where(~(br_fail | br_from | br_to))[0]
        mpc['branch'] = self.mpc_base['branch'][kept_br_idx]

        return mpc, kept_gen_orig_idx, kept_br_idx

    # -- certificate extraction --------------------------------------------
    def extract(self, comps_st, variant="cert"):
        """Return (ref_dict, info) or (None, info) if the LP fails.

        ref_dict maps comp name -> ('>=', state), plus a 'sys' entry, matching
        the minimiser's output format. Costs one linprog solve.
        """
        from func_dcopt_py import _find_ref_buses
        from pypower.idx_bus import BUS_I
        from pypower.idx_gen import GEN_BUS, PMAX, PMIN
        from pypower.idx_brch import F_BUS, T_BUS, BR_X, TAP, RATE_A

        mpc, kept_gen, kept_br = self._reduce(comps_st)
        base = self.baseMVA
        bus, gen, branch = mpc['bus'], mpc['gen'], mpc['branch']
        nb_r, nl_r = bus.shape[0], branch.shape[0]
        is_orig = kept_gen < self.ng
        orig_rows = np.where(is_orig)[0]      # rows in reduced gen table
        load_rows = np.where(~is_orig)[0]
        G, L = len(orig_rows), len(load_rows)
        if nb_r == 0 or nl_r == 0 or G == 0:
            return None, {"fail": "degenerate network", "n_lp": 0}

        bus_map = {int(b): i for i, b in enumerate(bus[:, BUS_I].astype(int))}
        x_br = branch[:, BR_X].copy(); x_br[x_br == 0] = 1e-6
        tap = branch[:, TAP].copy(); tap[tap == 0] = 1.0
        b_br = 1.0 / (x_br * tap)
        f_bus = np.array([bus_map[int(b)] for b in branch[:, F_BUS]])
        t_bus = np.array([bus_map[int(b)] for b in branch[:, T_BUS]])
        Bf = np.zeros((nl_r, nb_r))
        Bbus = np.zeros((nb_r, nb_r))
        for l in range(nl_r):
            f, t, b = f_bus[l], t_bus[l], b_br[l]
            Bf[l, f] += b; Bf[l, t] -= b
            Bbus[f, f] += b; Bbus[f, t] -= b; Bbus[t, f] -= b; Bbus[t, t] += b
        rate = branch[:, RATE_A] / base
        rate[rate == 0] = 1e10
        ref_buses = _find_ref_buses(bus, branch, bus_map)

        pmax_r = gen[:, PMAX] / base           # reduced (state-scaled) caps
        pmin_r = gen[:, PMIN] / base
        req = (1.0 - self.threshold / 100.0) * self.totpf / base * (1 + 1e-6)

        if variant == "cert":
            # vars: [gseg (3 per orig gen) | Pl (loads) | Va | z]
            nv = 3 * G + L + nb_r + nl_r
            c = np.zeros(nv)
            lo = np.full(nv, 0.0); hi = np.full(nv, 0.0)
            for gi, row in enumerate(orig_rows):
                g_orig = kept_gen[row]
                name = f"vbus{self.gen_dic[g_orig]}"
                pm = self.pm_orig_pu[g_orig]
                caps = [0.4 * pm, 0.4 * pm, 0.2 * pm]        # state grid segments
                # convex PWL cost: marginal -log P(>= s) per p.u. of capacity used
                p1, p2, p3 = self._sf(name, 1), self._sf(name, 2), self._sf(name, 3)
                steps = [-np.log(max(p1, 1e-12)),
                         -np.log(max(p2 / max(p1, 1e-12), 1e-12)),
                         -np.log(max(p3 / max(p2, 1e-12), 1e-12))]
                avail = pmax_r[row]                          # seed-state cap
                acc = 0.0
                for k in range(3):
                    seg_cap = max(0.0, min(caps[k], avail - acc)); acc += seg_cap
                    j = 3 * gi + k
                    hi[j] = seg_cap
                    c[j] = steps[k] / caps[k] if caps[k] > 0 else 0.0
            for li, row in enumerate(load_rows):
                j = 3 * G + li
                lo[j], hi[j] = pmin_r[row], 0.0              # Pl in [-PD, 0]
            va0 = 3 * G + L
            for i in range(nb_r):
                lo[va0 + i], hi[va0 + i] = (0.0, 0.0) if i in ref_buses \
                    else (-np.inf, np.inf)
            z0 = va0 + nb_r
            for l in range(nl_r):
                name = f"br{kept_br[l] + 1}"
                lo[z0 + l], hi[z0 + l] = 0.0, rate[l]
                c[z0 + l] = -np.log(max(self._sf(name, 1), 1e-12)) / rate[l]

            A_eq = np.zeros((nb_r, nv)); b_eq = np.zeros(nb_r)
            for gi, row in enumerate(orig_rows):
                bi = bus_map[int(gen[row, GEN_BUS])]
                A_eq[bi, 3 * gi:3 * gi + 3] = 1.0
            for li, row in enumerate(load_rows):
                bi = bus_map[int(gen[row, GEN_BUS])]
                A_eq[bi, 3 * G + li] = 1.0
            A_eq[:, va0:va0 + nb_r] = -Bbus

            A_ub = np.zeros((2 * nl_r + 1, nv)); b_ub = np.zeros(2 * nl_r + 1)
            A_ub[:nl_r, va0:va0 + nb_r] = Bf
            A_ub[nl_r:2 * nl_r, va0:va0 + nb_r] = -Bf
            A_ub[:nl_r, z0:] = -np.eye(nl_r)
            A_ub[nl_r:2 * nl_r, z0:] = -np.eye(nl_r)
            A_ub[-1, 3 * G:3 * G + L] = 1.0                  # sum Pl <= -req
            b_ub[-1] = -req
        else:  # variant == "opf": the ordinary max-served-load solve
            nv = G + L + nb_r
            c = np.zeros(nv)
            c[G:G + L] = 1.0                                 # push Pl to -PD
            lo = np.full(nv, 0.0); hi = np.full(nv, 0.0)
            for gi, row in enumerate(orig_rows):
                hi[gi] = pmax_r[row]
            for li, row in enumerate(load_rows):
                lo[G + li], hi[G + li] = pmin_r[row], 0.0
            va0 = G + L
            for i in range(nb_r):
                lo[va0 + i], hi[va0 + i] = (0.0, 0.0) if i in ref_buses \
                    else (-np.inf, np.inf)
            A_eq = np.zeros((nb_r, nv)); b_eq = np.zeros(nb_r)
            for gi, row in enumerate(orig_rows):
                A_eq[bus_map[int(gen[row, GEN_BUS])], gi] = 1.0
            for li, row in enumerate(load_rows):
                A_eq[bus_map[int(gen[row, GEN_BUS])], G + li] = 1.0
            A_eq[:, va0:] = -Bbus
            A_ub = np.zeros((2 * nl_r, nv)); b_ub = np.concatenate([rate, rate])
            A_ub[:nl_r, va0:] = Bf
            A_ub[nl_r:, va0:] = -Bf

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=list(zip(lo, hi)), method='highs',
                      options={'presolve': True})
        self.n_solves += 1
        if not res.success:
            return None, {"fail": f"linprog: {res.message}", "n_lp": 1}

        # ---- read utilisations, round up to discrete states ----
        x = res.x
        if variant == "cert":
            pg = np.array([x[3 * gi:3 * gi + 3].sum() for gi in range(G)])
            pl = x[3 * G:3 * G + L]
            va = x[3 * G + L:3 * G + L + nb_r]
        else:
            pg = x[:G]; pl = x[G:G + L]; va = x[G + L:]
            served = -pl.sum()
            if served < req:                    # seed actually fails
                return None, {"fail": "below survival threshold", "n_lp": 1}
        flows = Bf @ va

        conds, alive = {}, set()
        for gi, row in enumerate(orig_rows):
            if pg[gi] <= FLOW_TOL:
                continue
            g_orig = kept_gen[row]
            bus_id = self.gen_dic[g_orig]
            pm = self.pm_orig_pu[g_orig]
            s_min = next(s for s in sorted(GEN_FRAC)
                         if GEN_FRAC[s] * pm >= pg[gi] - 1e-7)
            name = f"vbus{bus_id}"
            conds[name] = max(conds.get(name, 0), s_min)
        for li, row in enumerate(load_rows):
            if -pl[li] > FLOW_TOL:
                alive.add(int(gen[row, GEN_BUS]))
        for l in range(nl_r):
            if abs(flows[l]) > FLOW_TOL:
                conds[f"br{kept_br[l] + 1}"] = 1
                alive.add(int(branch[l, F_BUS])); alive.add(int(branch[l, T_BUS]))
        for bus_id in alive:
            name = f"vbus{bus_id}"
            conds[name] = max(conds.get(name, 0), 1)

        ref = {k: ('>=', v) for k, v in conds.items() if v > 0}
        ref['sys'] = ('>=', 1)
        return ref, {"n_lp": 1, "served_pu": float(-pl.sum())}

    # -- dual failure cut ----------------------------------------------------
    def extract_failure_cut(self, comps_st):
        """One transportation-relaxation solve at a failed state -> FailureCut.

        Returns (cut, info); cut is None when the relaxation still serves the
        threshold load (relaxation gap — e.g. a purely congestion-driven
        failure), in which case the caller should fall back to the box
        minimiser. Costs one linprog solve.
        """
        ss = self._system_state(comps_st)
        bus_alive = ss[:self.nb] != 1.0
        br_alive = (ss[self.nb:] != 1.0) \
            & bus_alive[self.br_f_pos] & bus_alive[self.br_t_pos]
        gen_frac = 1.0 - ss[self.gen_bus_pos]          # capacity fraction per gen
        nload = len(self.load_bus)
        ng, nl = self.ng, self.nl

        pg_up = self.pm_orig_pu * gen_frac
        srv_up = self.load_pd_pu * bus_alive[[self.bus_pos[b] for b in self.load_bus]]
        f_cap = self.rate_pu * br_alive

        nv = ng + nload + nl
        c = np.zeros(nv)
        c[ng:ng + nload] = -1.0                        # max served load
        A_eq = np.zeros((self.nb, nv)); b_eq = np.zeros(self.nb)
        for g in range(ng):
            A_eq[self.gen_bus_pos[g], g] = 1.0
        for d in range(nload):
            A_eq[self.bus_pos[self.load_bus[d]], ng + d] = -1.0
        for l in range(nl):
            A_eq[self.br_t_pos[l], ng + nload + l] += 1.0
            A_eq[self.br_f_pos[l], ng + nload + l] -= 1.0
        bounds = [(0.0, pg_up[g]) for g in range(ng)] \
            + [(0.0, srv_up[d]) for d in range(nload)] \
            + [(-f_cap[l], f_cap[l]) for l in range(nl)]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs',
                      options={'presolve': True})
        self.n_solves += 1
        if not res.success:
            return None, {"fail": f"linprog: {res.message}", "n_lp": 1}
        served = -res.fun
        if served > self.req0 - 1e-9:
            return None, {"fail": "relaxation gap", "served_pu": served, "n_lp": 1}

        # weak duality: served(x') <= W(x') for all x', with weights below
        lo, up = res.lower.marginals, res.upper.marginals
        clip = lambda a: np.where(np.abs(a) > 1e-9, a, 0.0)
        w_g = clip(-up[:ng]) * self.pm_orig_pu
        w_d = clip(-up[ng:ng + nload]) * self.load_pd_pu
        w_l = (clip(lo[ng + nload:]) - clip(up[ng + nload:])) * self.rate_pu

        # self-check (strong duality at the seed): W(seed) == served
        w_seed = float(w_g @ gen_frac + w_d @ bus_alive[
            [self.bus_pos[b] for b in self.load_bus]] + w_l @ br_alive)
        if not np.isclose(w_seed, served, atol=1e-5 * max(1.0, self.req0)):
            return None, {"fail": f"dual identity off: W={w_seed} vs {served}",
                          "n_lp": 1}

        frac_by_state = [GEN_FRAC[s] for s in sorted(GEN_FRAC)]
        gen_terms = [(f"vbus{self.gen_dic[g]}", float(w_g[g]), frac_by_state)
                     for g in range(ng) if w_g[g] > 1e-12]
        bus_terms = [(f"vbus{self.load_bus[d]}", float(w_d[d]))
                     for d in range(nload) if w_d[d] > 1e-12]
        line_terms = [(float(w_l[l]), f"br{l + 1}",
                       f"vbus{self.bus_dic[self.br_f_pos[l]]}",
                       f"vbus{self.bus_dic[self.br_t_pos[l]]}")
                      for l in range(nl) if w_l[l] > 1e-12]
        cut = FailureCut(bus_terms, gen_terms, line_terms, self.req0)
        return cut, {"n_lp": 1, "served_pu": served}


# ---------------------------------------------------------------------------
# Dual-guided cut refinement
# ---------------------------------------------------------------------------
def _best_states(probs_dict):
    return {n: max(int(k) for k in p) for n, p in probs_dict.items()}


def refine_failure_cut(cert_model, dcopt, seed_state, cut, best, *,
                       sys_surv_st=1, max_iters=2):
    """Dual-guided minimisation of a failure cut (~2-3 solves).

    A raw failed seed (from SuS or plain MC) carries incidentally degraded
    components; the LP solved on that degraded network can put dual weight on
    them, thinning the cut. The dual itself says which components matter — the
    cut's support. So: raise every off-support component to its best state
    (free — no search), confirm the cleaned state still fails (1 sfun call),
    and re-extract the cut there (1 LP). The refined cut is anchored at a
    near-minimal, higher-probability representative of the failure mode and
    still certifies the original seed (W is monotone and cleaned >= seed).
    Kept as-is when the cleaned state survives or hits a relaxation gap.

    Returns ``(cut, n_sfun, n_lp)``.
    """
    n_sfun = n_lp = 0
    cur = dict(seed_state)
    for _ in range(max_iters):
        supp = cut.support()
        cleaned = {n: (s if n in supp else best[n]) for n, s in cur.items()}
        if cleaned == cur:
            break
        _bo, sys_st, _ = dcopt(cleaned)
        n_sfun += 1
        if sys_st >= sys_surv_st:      # support alone doesn't fail: keep cut
            break
        new_cut, _info = cert_model.extract_failure_cut(cleaned)
        n_lp += 1
        if new_cut is None:            # relaxation gap at the cleaned state
            break
        cut, cur = new_cut, cleaned
    return cut, n_sfun, n_lp


def refined_cut_generator(cert_model, dcopt, *, sys_surv_st=1, max_iters=2):
    """``cut_generator`` for ``run_ref_extraction_by_mcs`` with dual-guided
    refinement. The refinement's dcopt re-checks are lumped into ``n_lp``
    (same cost class as the LP solves)."""
    best = _best_states(cert_model.probs_dict)

    def gen(comps_st):
        cut, info = cert_model.extract_failure_cut(comps_st)
        n_lp = int((info or {}).get("n_lp", 1))
        if cut is None:
            return None, {**(info or {}), "n_lp": n_lp}
        cut, ns, nl = refine_failure_cut(
            cert_model, dcopt, comps_st, cut, best,
            sys_surv_st=sys_surv_st, max_iters=max_iters)
        return cut, {"n_lp": n_lp + ns + nl}

    return gen


# ---------------------------------------------------------------------------
# SuS -> cut pipeline
# ---------------------------------------------------------------------------
def sus_failure_cuts(cert_model, dcopt, probs, row_names, *, sys_surv_st=1,
                     n_per_level=1000, p0=0.1, n_runs=1, max_levels=10,
                     n_flip_mean=5.0, n_workers=1, verbose=True):
    """Subset-Simulation-driven failure-cut generation.

    Walks toward the failure boundary with discrete Subset Simulation
    (``rsr.subset_sim``, Au & Beck 2001; severity = blackout size, so SuS
    concentrates on the *most probable* failure modes), then converts every
    distinct failed state it finds into a dual failure cut (one
    transportation-relaxation solve each; see :class:`FailureCut`). States
    already certified by a cut collected earlier are skipped, so heavily
    correlated chain samples cost nothing extra.

    Independent repetitions (``n_runs``) restart SuS from fresh prior samples
    to diversify the failure modes found. Returns ``(cuts, stats)``; feed
    ``cuts`` into ``run_ref_extraction_by_mcs(lower_cuts=...)`` so the
    certified mass counts toward p_lower from round one.

    Args:
        cert_model: :class:`CertModel` for the same network/threshold.
        dcopt: raw DC-OPF system function ``comps_st -> (blackout_%, sys_st, _)``
            (the *severity* matters here, so pass the underlying model, not
            the demo's binarised sfun).
        probs: (n_var, n_state) categorical prior (any device; moved to CPU).
        row_names: component names matching ``probs`` rows.
        n_workers: CPU processes for the SuS sfun evaluations (fork pool).
    """
    import multiprocessing as mp
    from rsr.subset_sim import subset_sim_search, set_worker_state

    probs = probs.detach().cpu()
    name_pos = {n: i for i, n in enumerate(row_names)}
    n_state = probs.shape[1]

    set_worker_state(dcopt, row_names, sys_surv_st)   # before fork
    pool = None
    if n_workers > 1:
        pool = mp.get_context('fork').Pool(n_workers)

    cuts, seen = [], set()
    best = _best_states(cert_model.probs_dict)
    stats = {"n_sfun": 0, "n_lp": 0, "n_failed_states": 0, "n_unique": 0,
             "n_skipped_certified": 0, "n_gaps": 0, "n_refined": 0,
             "levels": []}
    try:
        for run in range(n_runs):
            res = subset_sim_search(
                probs, dcopt, row_names, sys_surv_st,
                n_per_level=n_per_level, p0=p0, max_levels=max_levels,
                severity_sign=+1, n_flip_mean=n_flip_mean, pool=pool,
                verbose=verbose)
            stats["n_sfun"] += res["n_sfun_calls"]
            stats["levels"].append(res["n_levels"])
            failed = res["failed_states"]                 # (M, n_var) int
            stats["n_failed_states"] += int(failed.shape[0])

            for i in range(failed.shape[0]):
                row = failed[i]
                key = tuple(int(v) for v in row.tolist())
                if key in seen:
                    continue
                seen.add(key)
                stats["n_unique"] += 1
                if cuts and any(bool(c.certify(row.unsqueeze(0), name_pos,
                                               n_state)[0]) for c in cuts):
                    stats["n_skipped_certified"] += 1
                    continue
                cst = {row_names[k]: int(row[k]) for k in range(len(row_names))}
                cut, info = cert_model.extract_failure_cut(cst)
                stats["n_lp"] += int((info or {}).get("n_lp", 1))
                if cut is None:
                    stats["n_gaps"] += 1
                else:
                    # dual-guided refinement: re-anchor the cut at a cleaned
                    # (off-support comps at best) representative of its mode
                    # (~2-3 extra solves). Often a no-op: zero-weight comps are
                    # already unconditioned by the half-space, so the re-
                    # extracted dual tends to coincide with the original.
                    cut, ns, nl = refine_failure_cut(
                        cert_model, dcopt, cst, cut, best,
                        sys_surv_st=sys_surv_st)
                    stats["n_sfun"] += ns
                    stats["n_lp"] += nl
                    stats["n_refined"] += int(nl > 0)   # re-anchored at least once
                    cuts.append(cut)
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    if verbose:
        print(f"[SuS->cuts] {n_runs} run(s), levels={stats['levels']}, "
              f"{stats['n_sfun']} sfun calls; "
              f"{stats['n_failed_states']} failed states "
              f"({stats['n_unique']} unique, "
              f"{stats['n_skipped_certified']} already certified) -> "
              f"{len(cuts)} cuts ({stats['n_refined']} refined), "
              f"{stats['n_gaps']} relaxation gaps, "
              f"{stats['n_lp']} LP solves")
    return cuts, stats
