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
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

from ._attention_bwd_harness import (
    _REF_CACHE,
    ThdStorage,
    narrow_inputs,
    reference,
    run_case,
)
from .sdpa.bwd_cases import compare_bwd
from .sdpa.bwd_kernel_cases import perf_prune_cases

# Narrow-access re-layout of the pruning cases for stage_vec = 1 classes:
# every tensor base 1 element (2 bytes) off a 16-byte boundary.
_NARROW_VIEW = "bshd"
_NARROW_BASE = 1


def cases_for(plan, runtime=None) -> tuple:
    """The pruning cases of ``plan``'s main-kernel class."""
    spec = plan.specs["main"]
    g_split = int(dict(runtime or {}).get("g_split", 1))
    return perf_prune_cases(
        head_size=spec.head_size,
        dtype=spec.dtype,
        mask_class=spec.mask_class,
        seq_mode=spec.seq_mode,
        dkv_mode=spec.dkv_mode,
        stage_vec=spec.stage_vec,
        g_split=g_split,
    )


def _oracle_path(cache_dir, case) -> Path:
    return Path(cache_dir) / f"prune-{case.id}.pkl"


class PruneGate:
    """The pruning-set gate (see the module docstring)."""

    def prepare(self, configs_and_plans, cache_dir) -> int:
        """Store the oracle of every needed pruning case; returns the count."""
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
        return n

    @staticmethod
    def _load(case, cache_dir) -> None:
        if case.id in _REF_CACHE:
            return
        path = _oracle_path(cache_dir, case)
        if cache_dir is not None and path.exists():
            with open(path, "rb") as fh:
                _REF_CACHE[case.id] = pickle.load(fh)

    def check(self, arch, cfg, plan, run_cache, cache_dir) -> list:
        runtime = dict(getattr(cfg, "runtime", None) or {})
        overrides = {**dict(getattr(cfg, "knobs", None) or {}), **runtime}
        cases = cases_for(plan, runtime)
        if not cases:
            return ["prune: empty pruning set"]
        timed_main = plan.specs["main"].kernel_name()
        timed = {step.kernel for step in plan.launches}
        exercised = set()
        probs = []
        for case in cases:
            self._load(case, cache_dir)
            kw = {}
            if plan.specs["main"].stage_vec == 1:
                if case.length_mode == "ragged":
                    kw["thd"] = ThdStorage(base_offset=_NARROW_BASE)
                else:
                    kw["inputs"], kw["request"] = narrow_inputs(
                        case, _NARROW_VIEW, _NARROW_BASE
                    )
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
