# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pruning-set gate of the backward benchmark harness and sweep (test helper).

No rocKE configuration is timed unless it passed, in the timing process, the
pruning set of :func:`tests.sdpa.bwd_kernel_cases.perf_prune_cases` for the
problem class of its main kernel, through the float64 oracle and the same
integrity checks as the numeric tests (sentinels, canaries, padding rows).

The harness injects this provider by name (``--gate
tests._attention_bwd_perf_gate:gate``); the benchmarks layer never imports the
tests. Protocol:

* ``prepare([(config, plan)], cache_dir)``: computes the float64 oracle of
  every pruning case those configurations need, once per job, and stores it
  in ``cache_dir`` (pickles; atomic writes).
* ``check(arch, config, plan, run_cache, cache_dir) -> [problem, ...]``: runs
  every pruning case of the plan's main-kernel class with the configuration's
  knobs and runtime values, on the kernel cache ``run_cache`` (the code
  objects the caller times), and returns the problems (empty: passed). It
  also fails when a pruning case resolves another main kernel than the timed
  one (``gate_kernel_mismatch``) or when a timed kernel was exercised by no
  pruning case (``gate_coverage``).
* ``static_prune(arch, spec, hsaco) -> (token | None, record)``: the
  resource rules of ``tests/_attention_bwd_resources.py`` on a compiled main
  kernel (used by the sweep before timing).
* ``static_prune_dq(arch, dq_spec, hsaco) -> (token | None, record)``: the
  same rules on the split dQ kernel of a ``dq_mode = "split"`` candidate.

Dense plans (``plan.route == "dense"``, a ``"dense"`` stage spec instead of
``"main"``): the dense kernel bakes the problem shape, so no pruning case
runs the timed dense code object itself (the timed shape is checked in the
timing process against the reference). The pruning set is the batched set
of the dense spec's class (head size, dtype, band or none, dK / dV mode,
``g_split``), keeping the cases the dense route serves on the plan's arch;
each runs on the dense route with the configuration's knobs (its own dense
kernel per shape), and must resolve a dense spec equal to the timed one in
every field but the problem shape, and under split dQ a dense dQ spec equal
to the timed one alike (``gate_kernel_mismatch`` otherwise).
Coverage asks that every other timed kernel (prep, converts) is exercised and
that at least one case ran the timed dense configuration. ``prepare`` also
compiles those per-shape dense kernels once per job into a code-object cache
beside the oracles (``<cache_dir>/gate-hsaco``, unless
``ROCKE_BWD_HSACO_CACHE`` names one), so a timing attempt only loads them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
import pickle
from pathlib import Path
from unittest import mock

from benchmarks.common.attention_bwd_rocke_arm import REQUEST_HINTS

from ._attention_bwd_harness import (
    _REF_CACHE,
    ThdStorage,
    narrow_inputs,
    reference,
    run_case,
)
from .sdpa.bwd_cases import compare_bwd
from .sdpa.bwd_kernel_cases import perf_prune_cases, request_from_case

# Narrow-access re-layout of the pruning cases for stage_vec = 1 classes:
# every tensor base 1 element (2 bytes) off a 16-byte boundary.
_NARROW_VIEW = "bshd"
_NARROW_BASE = 1
# The dense route switch of a configuration's runtime values (like the
# request hints, REQUEST_HINTS): never a tuning value.
_ROUTE = "route"
_HSACO_ENV = "ROCKE_BWD_HSACO_CACHE"
# Launch stages of the shape-baked dense kernels (main, split dQ).
_DENSE_STAGES = ("dense", "dense_dq")
# Dense spec fields that follow the problem shape (the rest is the
# configuration a pruning case must reproduce).
_DENSE_PROBLEM = frozenset(
    {
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "batch",
        "causal",
        "causal_bottom_right",
        "dtype",
        "layout",
        "sliding_window",
        "window_left",
        "window_right",
        "ragged",
        "name",
    }
)


def is_dense(plan) -> bool:
    """Whether ``plan`` runs the dense main kernel (``"dense"`` stage)."""
    return "main" not in plan.specs and "dense" in plan.specs


def dense_identity(spec) -> tuple:
    """The configuration part of a dense spec (every non-problem field)."""
    return (type(spec).__name__,) + tuple(
        (f.name, getattr(spec, f.name))
        for f in dataclasses.fields(spec)
        if f.name not in _DENSE_PROBLEM
    )


def dense_dq_identity(spec) -> tuple | None:
    """The configuration part of a dense split dQ spec (``None``: no split):
    every field but the problem shape (its dQ tuning values whole)."""
    if spec is None:
        return None
    return dense_identity(spec)


def cases_for(plan, runtime=None) -> tuple:
    """The pruning cases of ``plan``'s main-kernel class (dense plans: the
    cases of the dense spec's class the dense route serves on its arch)."""
    g_split = int(dict(runtime or {}).get("g_split", 1))
    if not is_dense(plan):
        spec = plan.specs["main"]
        return perf_prune_cases(
            head_size=spec.head_size,
            dtype=spec.dtype,
            mask_class=spec.mask_class,
            seq_mode=spec.seq_mode,
            dkv_mode=spec.dkv_mode,
            stage_vec=spec.stage_vec,
            g_split=g_split,
        )
    from kernels.common.attention_bwd_plan import dense_bwd_eligibility

    spec = plan.specs["dense"]
    cases = perf_prune_cases(
        head_size=spec.head_size,
        dtype=spec.dtype,
        mask_class="band" if spec.masked else "none",
        seq_mode="batched",
        dkv_mode=spec.dkv_mode,
        stage_vec=8,
        g_split=g_split,
    )
    return tuple(
        c for c in cases if dense_bwd_eligibility(request_from_case(c), plan.arch)[0]
    )


def _split_runtime(cfg) -> tuple[dict, dict]:
    """``(tuning values, request hints)`` of a configuration (knobs and its
    runtime values; the route switch dropped)."""
    runtime = dict(getattr(cfg, "runtime", None) or {})
    hints = {k: runtime.pop(k) for k in REQUEST_HINTS if k in runtime}
    runtime.pop(_ROUTE, None)
    return {**dict(getattr(cfg, "knobs", None) or {}), **runtime}, hints


def _device_hints(hints: dict) -> dict:
    if hints.get("num_cus") == "device":  # the rocKE arm's DEVICE_NUM_CUS
        from rocke.runtime.hip_module import get_device_num_cus

        hints = dict(hints, num_cus=int(get_device_num_cus(0)))
    return hints


def dense_case_plan(arch, cfg, case, plan=None):
    """The dense plan of pruning case ``case`` under configuration ``cfg``.

    With the timed ``plan`` the case takes the device hints that plan was
    built with (:func:`case_request`: the XCD count of its remap, the grid
    of a persistent kernel), whether the configuration names them or the
    arm supplied them by default; without it, the configuration's own hints.
    """
    from kernels.common.attention_bwd_plan import AttnBwdPolicy, FEATURE_GROUPS
    from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

    values, hints = _split_runtime(cfg)
    if plan is not None:
        hints = case_request(plan)
    req = request_from_case(case)
    if hints:
        req = dataclasses.replace(req, **_device_hints(hints))

    def tuning(_req, _arch, _route, _pclass):
        return dict(values)

    policy = AttnBwdPolicy(
        declared_groups=frozenset(FEATURE_GROUPS),
        waive_unverified=True,
        tuning=tuning,
    )
    return attn_dense_bwd_plan(req, arch, policy=policy)


@contextlib.contextmanager
def _gate_code_objects(cache_dir):
    """The on-disk code-object cache of the dense pruning kernels: the
    caller's ``ROCKE_BWD_HSACO_CACHE``, else ``<cache_dir>/gate-hsaco``."""
    if os.environ.get(_HSACO_ENV, "").strip() or cache_dir is None:
        yield
        return
    root = Path(cache_dir) / "gate-hsaco"
    root.mkdir(parents=True, exist_ok=True)
    with mock.patch.dict(os.environ, {_HSACO_ENV: str(root)}):
        yield


def case_request(plan) -> dict:
    """Request hints a pruning case needs to resolve ``plan``'s kernels.

    A configuration's ``num_cus`` / ``num_xcds`` describe the timed device,
    not a pruning case, and the arm fills either one in when the
    configuration leaves it out; so the hints come from the timed plan, never
    from the configuration. The XCD remap is rebuilt from the timed plan's
    XCD counts (main and split dQ; ``xcd_chunk`` values ``gx`` / ``2gx`` /
    ``4gx`` then resolve on the pruning grid). Mirror pairing takes
    ``num_cus = 1``: its fill guard is a no-gain rule (the paired decode
    visits every kv tile once for any grid), and the small pruning grids
    would never pass it, so the gate exercises the timed paired kernel as the
    device tests do. A persistent dense kernel takes ``num_cus`` = the timed
    plan's persistent grid (the same CTA count, looping over a small grid).
    """
    rt = plan.runtime
    out = {}
    xcds = int(rt.get("xcd_n") or 0) or int(rt.get("dq_xcd_n") or 0)
    if xcds:
        out["num_xcds"] = xcds
    if rt.get("pair"):
        out["num_cus"] = 1
    elif int(rt.get("num_persistent") or 0):
        out["num_cus"] = int(rt["num_persistent"])
    return out


def _oracle_path(cache_dir, case) -> Path:
    return Path(cache_dir) / f"prune-{case.id}.pkl"


class PruneGate:
    """The pruning-set gate (see the module docstring)."""

    def prepare(self, configs_and_plans, cache_dir) -> int:
        """Store the oracle of every needed pruning case; returns the count.

        Dense plans: their per-shape pruning kernels are compiled here too
        (once per job, see the module docstring)."""
        n = 0
        for cfg, plan in configs_and_plans:
            for case in cases_for(plan, getattr(cfg, "runtime", None)):
                path = _oracle_path(cache_dir, case)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}.tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump(reference(case), fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, path)
                n += 1
        for cfg, plan in configs_and_plans:
            if is_dense(plan):
                self._compile_dense(plan.arch, cfg, plan, cache_dir)
        return n

    @staticmethod
    def _compile_dense(arch, cfg, plan, cache_dir) -> None:
        """Compile (or find) every dense pruning kernel of ``cfg``; a case the
        dense plan refuses is left to :meth:`check`, which reports it."""
        from kernels.common.attention_bwd_run import attn_bwd_kernels

        with _gate_code_objects(cache_dir):
            for case in cases_for(plan, getattr(cfg, "runtime", None)):
                try:
                    p = dense_case_plan(arch, cfg, case, plan)
                except ValueError:
                    continue
                attn_bwd_kernels(p, cache={})

    @staticmethod
    def _load(case, cache_dir) -> None:
        if case.id in _REF_CACHE:
            return
        path = _oracle_path(cache_dir, case)
        if cache_dir is not None and path.exists():
            with open(path, "rb") as fh:
                _REF_CACHE[case.id] = pickle.load(fh)

    def check(self, arch, cfg, plan, run_cache, cache_dir) -> list:
        if is_dense(plan):
            with _gate_code_objects(cache_dir):
                return self._check_dense(arch, cfg, plan, run_cache, cache_dir)
        runtime = dict(getattr(cfg, "runtime", None) or {})
        # device hints are request fields, not knobs (see case_request)
        for k in REQUEST_HINTS:
            runtime.pop(k, None)
        overrides = {**dict(getattr(cfg, "knobs", None) or {}), **runtime}
        hints = case_request(plan)
        cases = cases_for(plan, runtime)
        if not cases:
            return ["prune: empty pruning set"]
        timed_main = plan.specs["main"].kernel_name()
        timed = {step.kernel for step in plan.launches}
        exercised = set()
        probs = []
        for case in cases:
            self._load(case, cache_dir)
            kw = {"request": dict(hints)} if hints else {}
            if plan.specs["main"].stage_vec == 1:
                if case.length_mode == "ragged":
                    kw["thd"] = ThdStorage(base_offset=_NARROW_BASE)
                else:
                    kw["inputs"], narrow = narrow_inputs(
                        case, _NARROW_VIEW, _NARROW_BASE
                    )
                    kw["request"] = {**narrow, **hints}
            try:
                run = run_case(
                    case,
                    arch=arch,
                    config="default",
                    overrides=overrides,
                    cache=run_cache,
                    **kw,
                )
            except Exception as ex:  # noqa: BLE001 - a gate failure, not a crash
                probs.append(f"{case.id}: error {type(ex).__name__}: {ex}"[:300])
                continue
            if run.plan.specs["main"].kernel_name() != timed_main:
                probs.append(f"{case.id}: gate_kernel_mismatch")
                continue
            exercised |= {step.kernel for step in run.plan.launches}
            found = compare_bwd(case, run.dq, run.dk, run.dv, run.ref) + list(
                run.integrity
            )
            if found:
                probs.append(f"{case.id}: {found[0]}"[:300])
        missing = sorted(timed - exercised)
        if missing and not probs:
            probs.append("gate_coverage: " + ",".join(missing))
        return probs

    def _check_dense(self, arch, cfg, plan, run_cache, cache_dir) -> list:
        """:meth:`check` of a dense plan (see the module docstring)."""
        cases = cases_for(plan, getattr(cfg, "runtime", None))
        if not cases:
            return ["prune: empty pruning set"]
        want = dense_identity(plan.specs["dense"])
        want_dq = dense_dq_identity(plan.specs.get("dense_dq"))
        # the dense kernels bake the shape (main, and the split dQ kernel)
        shaped = {s.kernel for s in plan.launches if s.stage in _DENSE_STAGES}
        timed = {s.kernel for s in plan.launches} - shaped
        exercised, ran_dense = set(), False
        probs = []
        for case in cases:
            self._load(case, cache_dir)
            try:
                p = dense_case_plan(arch, cfg, case, plan)
                run = run_case(
                    case,
                    arch=arch,
                    config="default",
                    cache=run_cache,
                    mutate_plan=lambda _general, p=p: p,
                )
            except Exception as ex:  # noqa: BLE001 - a gate failure, not a crash
                probs.append(f"{case.id}: error {type(ex).__name__}: {ex}"[:300])
                continue
            spec = run.plan.specs.get("dense")
            dq = dense_dq_identity(run.plan.specs.get("dense_dq"))
            if spec is None or dense_identity(spec) != want or dq != want_dq:
                probs.append(f"{case.id}: gate_kernel_mismatch")
                continue
            ran_dense = True
            exercised |= {s.kernel for s in run.plan.launches}
            found = compare_bwd(case, run.dq, run.dk, run.dv, run.ref) + list(
                run.integrity
            )
            if found:
                probs.append(f"{case.id}: {found[0]}"[:300])
        missing = sorted(timed - exercised)
        if not ran_dense:
            missing.append("dense")
        if missing and not probs:
            probs.append("gate_coverage: " + ",".join(missing))
        return probs

    def static_prune(self, arch, spec, hsaco):
        """Resource rules on a compiled main kernel: ``(token | None, record)``.

        Scratch, spills, the arch-VGPR / AGPR budgets, occupancy, LDS and the
        hot-loop AGPR-copy ceiling (production flavor only: a result from
        another flavor is not a gate result and prunes as ``flavor``).
        """
        from kernels.common.attention_bwd import attn_bwd_arch_facts

        from ._attention_bwd_resources import evaluate_resource_gate

        facts = attn_bwd_arch_facts(arch)
        res = evaluate_resource_gate(
            hsaco,
            arch=arch,
            lds_capacity_bytes=facts.lds_capacity_bytes,
            spec=spec,
        )
        rec = res.record()
        if not res.valid:
            return "flavor", rec
        return (res.failures[0] if res.failures else None), rec

    def static_prune_dq(self, arch, spec, hsaco):
        """Resource rules on a compiled split dQ kernel (``spec``: its resolved
        :class:`AttnBwdDqSpec`): ``(token | None, record)``.

        The rules of :meth:`static_prune`; with VGPR-form MFMA (``agpr_alloc
        = (0, 0)``) no key loop may read an AGPR (``agpr_copy_traffic``), as
        in the dQ kernel's resource gate test.
        """
        from kernels.common.attention_bwd import attn_bwd_arch_facts

        from ._attention_bwd_resources import evaluate_resource_gate

        facts = attn_bwd_arch_facts(arch)
        tile = {}
        if spec.agpr_alloc == (0, 0):
            tile = dict(block_m=spec.block_m, block_n=spec.block_n, waves=spec.waves)
        res = evaluate_resource_gate(
            hsaco, arch=arch, lds_capacity_bytes=facts.lds_capacity_bytes, **tile
        )
        rec = res.record()
        if not res.valid:
            return "flavor", rec
        fails = list(res.failures)
        if tile and any(x["reads"] for x in res.loops or ()):
            fails.append("agpr_copy_traffic")
        return (fails[0] if fails else None), rec


gate = PruneGate()
