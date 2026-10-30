# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Pick a small kernel library from a one-shot skinny GEMM sweep (msgpack or YAML).

Reads sweep_skinny_gemm.py results where every screened configuration is listed
in each case's top entry (one-shot / full screening). Coverage uses the recorded
timeUs values, then every selected case is re-run on the GPU to check its
measured time and result before the selection is written. The checked kernels
are emitted for --code-object-version (for example cov4) into --kernels.

For each case the best time is the fastest valid USE_BETA1 entry. A configuration
covers a case when its timeUs is within --tolerance of that best. Entries marked
verified=failed are excluded.

Only problem records with useBeta true are kept (the USE_BETA1 kernel path). Top
entries with config useBeta false are skipped (no USE_BETA0 symbols).

Config identity is one emitted kernel symbol: bm, tn, lanesK, waves, prefetch,
upIters, partialSteps, group, numXcds, grid3d, blocks, mfma, lds, use_beta true.

Selection (greedy, up to --max-configs rounds):
  1. Among cases not yet assigned, count how often each config identity appears
     within tolerance of that case's sweep best (same per-case filter as below).
  2. Add the identity with the highest count to the library (ties: mean and
     worst slowdown, mean sweep rank, then identity tuple), then assign every
     case disjointly to its fastest within-tolerance config in the library so far.
  3. Repeat until the library is full, no identity helps remaining cases, or every
     case is assigned. Uncovered cases have no library config within tolerance.
  4. Improve the greedy result with coverage-increasing one-for-one swaps.

Example:

  python tests/tensilelite/bindings/select_skinny_gemm_from_sweep.py \\
      results.msgpack -o selection.yaml --kernels kernels/ \\
      --code-object-version cov4 --tolerance 0.10 --max-configs 14
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import NamedTuple

import yaml

from sweep_skinny_gemm_results import read_results

DEFAULT_TOLERANCE = 0.10
LABEL_KEYS = ("name", "m", "n", "k", "spk", "useBeta", "betaMode")


class ConfigIdentity(NamedTuple):
    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    up_iters: int
    partial_steps: int
    group: int
    xcds: int
    grid3d: bool
    blocks: int | None
    mfma: bool
    lds: bool
    use_beta: bool


@dataclass(frozen=True)
class SweepCase:
    index: int
    problem: dict
    best_us: float | None
    top: list

    @property
    def label(self) -> dict:
        return {key: self.problem[key] for key in LABEL_KEYS}

    def text(self) -> str:
        p = self.problem
        return f"{p['name']} m={p['m']} n={p['n']} k={p['k']} spk={p['spk']}"


@dataclass(frozen=True)
class QualifiedEntry:
    time_us: float
    cfg: dict
    kernel: str
    rank: int


@dataclass(frozen=True)
class Cover:
    identity: ConfigIdentity
    entry: QualifiedEntry


@dataclass(frozen=True)
class Validation:
    time_us: float
    limit_us: float


@dataclass(frozen=True)
class Selection:
    library: list[ConfigIdentity]
    tolerance_hits: Counter[ConfigIdentity]
    covers: dict[int, Cover]
    representatives: dict[ConfigIdentity, dict]


def identity_of(cfg: dict) -> ConfigIdentity:
    grid3d = bool(cfg["grid3d"])
    return ConfigIdentity(
        cfg["bm"],
        cfg["tn"],
        cfg["lanesK"],
        cfg["waves"],
        cfg["prefetch"],
        int(cfg["upIters"]),
        int(cfg["partialSteps"]),
        cfg["group"],
        cfg["numXcds"],
        grid3d,
        None if grid3d else cfg["blocks"],
        bool(cfg.get("mfma", False)),
        bool(cfg.get("lds", False)),
        True,
    )


def keep_problem(problem: dict) -> bool:
    return bool(problem.get("useBeta"))


def top_entry_allowed(item: dict) -> bool:
    return bool(item["config"].get("useBeta")) and item.get("verified") != "failed"


def is_use_beta_kernel_name(name: str) -> bool:
    return "_USE_BETA1" in name and "_USE_BETA0" not in name


def tie_break_key(identity: ConfigIdentity) -> tuple[int, ...]:
    return tuple(-1 if value is None else int(value) for value in identity)


def rounded(value: float) -> float:
    return round(float(value), 4)


def config_record(identity: ConfigIdentity) -> dict:
    record = {name: getattr(identity, name) for name in identity._fields}
    record["grid3d"] = bool(identity.grid3d)
    return record


def load_cases(path: str, only: str | None, limit: int | None) -> list[SweepCase]:
    document = read_results(path)
    meta = document.get("meta", {})
    if meta.get("oneShot") is not True:
        raise ValueError(
            "results must come from sweep_skinny_gemm.py --one-shot "
            "(meta.oneShot is not true)"
        )
    if meta.get("interrupted") or meta.get("aborted"):
        raise ValueError("results are from an interrupted or aborted sweep")
    if meta.get("casesDone") != meta.get("casesTotal"):
        raise ValueError(
            "results are incomplete "
            f"({meta.get('casesDone')} of {meta.get('casesTotal')} cases finished)"
        )
    entries = [
        entry
        for entry in document["results"]
        if entry.get("top") and keep_problem(entry["problem"])
    ]
    if only:
        entries = [entry for entry in entries if only in entry["problem"]["name"]]
    if limit is not None:
        entries = entries[:limit]
    cases = []
    for index, entry in enumerate(entries):
        top = sorted(
            (item for item in entry["top"] if top_entry_allowed(item)),
            key=lambda item: item["timeUs"],
        )
        best_us = min((item["timeUs"] for item in top), default=None)
        cases.append(SweepCase(index, entry["problem"], best_us, top))
    return cases


def validation_settings(
    path: str, iters: int | None, warmup: int | None
) -> tuple[int, int, int]:
    settings = read_results(path).get("meta", {}).get("settings", {})
    selected_iters = settings.get("iters") if iters is None else iters
    selected_warmup = settings.get("warmup") if warmup is None else warmup
    verify_full_max_elems = settings.get("verify_full_max_elems")
    if not isinstance(selected_iters, int) or selected_iters < 1:
        raise ValueError("results metadata has no valid settings.iters")
    if not isinstance(selected_warmup, int) or selected_warmup < 0:
        raise ValueError("results metadata has no valid settings.warmup")
    if not isinstance(verify_full_max_elems, int) or verify_full_max_elems < 1:
        raise ValueError(
            "results metadata has no valid settings.verify_full_max_elems"
        )
    return selected_iters, selected_warmup, verify_full_max_elems


def qualified_per_case(
    case: SweepCase, tolerance: float
) -> dict[ConfigIdentity, QualifiedEntry]:
    if case.best_us is None:
        return {}
    limit = case.best_us * (1 + tolerance)
    per_id: dict[ConfigIdentity, QualifiedEntry] = {}
    for item in case.top:
        time_us = item["timeUs"]
        if time_us > limit:
            break
        identity = identity_of(item["config"])
        entry = QualifiedEntry(time_us, item["config"], item["kernel"], item["rank"])
        known = per_id.get(identity)
        if known is None or time_us < known.time_us:
            per_id[identity] = entry
    return per_id


def build_qualified(
    cases: list[SweepCase], tolerance: float
) -> dict[int, dict[ConfigIdentity, QualifiedEntry]]:
    return {case.index: qualified_per_case(case, tolerance) for case in cases}


def tolerance_hits(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
) -> Counter:
    counts: Counter[ConfigIdentity] = Counter()
    for per_id in qualified.values():
        for identity in per_id:
            counts[identity] += 1
    return counts


def pick_next_identity(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    cases: list[SweepCase],
    case_indices: list[int],
    global_hits: Counter[ConfigIdentity],
    library: set[ConfigIdentity],
    min_config_cases: int,
) -> ConfigIdentity | None:
    entries: dict[ConfigIdentity, list[tuple[float, int]]] = {}
    for index in case_indices:
        best_us = cases[index].best_us
        if best_us is None:
            continue
        for identity, entry in qualified.get(index, {}).items():
            if identity in library or global_hits[identity] < min_config_cases:
                continue
            entries.setdefault(identity, []).append(
                (entry.time_us / best_us, entry.rank)
            )
    if not entries:
        return None
    return min(
        entries,
        key=lambda identity: (
            -len(entries[identity]),
            mean(ratio for ratio, _ in entries[identity]),
            max(ratio for ratio, _ in entries[identity]),
            mean(rank for _, rank in entries[identity]),
            tie_break_key(identity),
        ),
    )


def improve_library(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    cases: list[SweepCase],
    library: set[ConfigIdentity],
    global_hits: Counter[ConfigIdentity],
    min_config_cases: int,
) -> set[ConfigIdentity]:
    """Improve greedy coverage with deterministic, coverage-increasing swaps."""
    if not library:
        return library
    masks: dict[ConfigIdentity, int] = {}
    quality: dict[ConfigIdentity, tuple[float, float, float]] = {}
    measurements: dict[ConfigIdentity, list[tuple[float, int]]] = {}
    for case in cases:
        if case.best_us is None:
            continue
        bit = 1 << case.index
        for identity, entry in qualified.get(case.index, {}).items():
            if global_hits[identity] < min_config_cases:
                continue
            masks[identity] = masks.get(identity, 0) | bit
            measurements.setdefault(identity, []).append(
                (entry.time_us / case.best_us, entry.rank)
            )
    for identity, values in measurements.items():
        quality[identity] = (
            mean(ratio for ratio, _ in values),
            max(ratio for ratio, _ in values),
            mean(rank for _, rank in values),
        )

    while True:
        selected = sorted(library, key=tie_break_key)
        current_mask = 0
        for identity in selected:
            current_mask |= masks[identity]
        current_coverage = current_mask.bit_count()
        best: tuple[tuple, ConfigIdentity, ConfigIdentity] | None = None
        for removed in selected:
            base_mask = 0
            for identity in selected:
                if identity != removed:
                    base_mask |= masks[identity]
            for added, added_mask in masks.items():
                if added in library:
                    continue
                coverage = (base_mask | added_mask).bit_count()
                if coverage <= current_coverage:
                    continue
                key = (
                    -coverage,
                    *quality[added],
                    tie_break_key(added),
                    tie_break_key(removed),
                )
                if best is None or key < best[0]:
                    best = (key, removed, added)
        if best is None:
            return library
        _, removed, added = best
        library.remove(removed)
        library.add(added)


def fastest_winner(
    options: dict[ConfigIdentity, QualifiedEntry],
) -> tuple[ConfigIdentity, QualifiedEntry]:
    return min(
        options.items(),
        key=lambda pair: (pair[1].time_us, tie_break_key(pair[0])),
    )


def assign_disjoint(
    cases: list[SweepCase],
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    library: set[ConfigIdentity],
) -> dict[int, Cover]:
    """Disjoint assignment: prefer the case's fastest qualifier; else fastest in library."""
    winners: dict[int, ConfigIdentity] = {}
    for case in cases:
        per_id = qualified.get(case.index)
        if not per_id:
            continue
        identity, _ = fastest_winner(per_id)
        winners[case.index] = identity

    covers: dict[int, Cover] = {}
    for case in cases:
        per_id = qualified.get(case.index)
        if not per_id:
            continue
        identity = winners[case.index]
        if identity in library:
            covers[case.index] = Cover(identity, per_id[identity])
    for case in cases:
        if case.index in covers:
            continue
        per_id = qualified.get(case.index)
        if not per_id:
            continue
        options = {i: e for i, e in per_id.items() if i in library}
        if not options:
            continue
        identity, entry = fastest_winner(options)
        covers[case.index] = Cover(identity, entry)
    return covers


def representatives(
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]],
    library: set[ConfigIdentity],
) -> dict[ConfigIdentity, dict]:
    reps: dict[ConfigIdentity, dict] = {}
    for per_id in qualified.values():
        for identity, entry in per_id.items():
            if identity not in library:
                continue
            record = entry.cfg
            known = reps.get(identity)
            if known is None or entry.rank < known["_rank"]:
                reps[identity] = {**record, "_rank": entry.rank}
    for identity in list(reps):
        reps[identity] = {k: v for k, v in reps[identity].items() if k != "_rank"}
    return reps


def select_library(
    cases: list[SweepCase],
    tolerance: float,
    max_configs: int,
    min_config_cases: int,
) -> Selection:
    qualified = build_qualified(cases, tolerance)
    global_hits = tolerance_hits(qualified)
    library_set: set[ConfigIdentity] = set()
    covers: dict[int, Cover] = {}
    while True:
        while len(library_set) < max_configs:
            uncovered = [case.index for case in cases if case.index not in covers]
            if not uncovered:
                break
            identity = pick_next_identity(
                qualified,
                cases,
                uncovered,
                global_hits,
                library_set,
                min_config_cases,
            )
            if identity is None:
                break
            library_set.add(identity)
            covers = assign_disjoint(cases, qualified, library_set)
        library_set = improve_library(
            qualified, cases, library_set, global_hits, min_config_cases
        )
        covers = assign_disjoint(cases, qualified, library_set)
        assigned = {cover.identity for cover in covers.values()}
        if assigned == library_set:
            break
        # An identity with no assignments is redundant: every case it qualifies
        # for has a faster selected identity. Remove it and reuse the slot.
        library_set = assigned
        covers = assign_disjoint(cases, qualified, library_set)
    library = sorted(
        library_set,
        key=lambda identity: (-global_hits[identity], tie_break_key(identity)),
    )
    reps = representatives(qualified, library_set)
    return Selection(library, global_hits, covers, reps)


def validation_problem(case: SweepCase, cfg: dict):
    """Reconstructs the swept problem so the emitted symbol is re-run exactly."""
    import skinny_gemm_support as support

    align = support.alignUp(case.problem["k"], 4)
    mfma = bool(cfg.get("mfma", False))
    problem = support.makeProblem(
        case.problem["m"],
        case.problem["n"],
        case.problem["k"],
        spk=cfg["spk"],
        bm=cfg["bm"],
        tn=cfg["tn"],
        lanes_k=cfg["lanesK"],
        waves=cfg["waves"],
        pf=cfg["prefetch"],
        group=cfg["group"],
        xcds=cfg["numXcds"],
        up_scales=0 if mfma else int(cfg["upIters"] > 0),
        blocks=None if cfg["grid3d"] else cfg["blocks"],
        padding=(
            case.problem["ldA"] - align,
            case.problem["ldB"] - align,
            0,
        ),
        ldD=case.problem["ldD"] // 2,
        ldC=case.problem["ldC"] // 2,
        alpha=case.problem["alpha"],
        beta=case.problem["beta"],
        use_beta=case.problem["useBeta"],
        mfma=mfma,
        lds=bool(cfg.get("lds", False)),
    )
    if support.gemmKernelName(problem) != cfg["kernel"]:
        raise ValueError(
            f"{case.text()}: reconstructed kernel {support.gemmKernelName(problem)} "
            f"does not match selected kernel {cfg['kernel']}"
        )
    return problem


def code_object_version(text: str) -> int:
    import skinny_gemm_support as support

    token = text.strip().lower()
    if token.startswith("cov"):
        token = token[3:]
    try:
        value = int(token)
    except ValueError as err:
        allowed = ", ".join(f"cov{version}" for version in support.CODE_OBJECT_VERSIONS)
        raise argparse.ArgumentTypeError(f"must be one of {allowed}, got {text}") from err
    if value not in support.CODE_OBJECT_VERSIONS:
        allowed = ", ".join(f"cov{version}" for version in support.CODE_OBJECT_VERSIONS)
        raise argparse.ArgumentTypeError(f"must be one of {allowed}, got {text}")
    return value


def emit_kernels(
    cases: list[SweepCase], selection: Selection, code_object_version: int
) -> dict[str, tuple[str, bytes]]:
    """Assembles each selected GEMM, and the reduce when a selected case splits K."""
    import skinny_gemm_support as support

    problems: dict[str, object] = {}
    needs_reduce = False
    for index, cover in sorted(selection.covers.items()):
        cfg = {**cover.entry.cfg, "kernel": cover.entry.kernel}
        problem = validation_problem(cases[index], cfg)
        problems.setdefault(cover.entry.kernel, problem)
        needs_reduce = needs_reduce or problem.splits > 1
    emitted: dict[str, tuple[str, bytes]] = {}
    for name, problem in problems.items():
        asm = support.gemmAsm(problem, code_object_version, 0)
        emitted[name] = (asm, support.assemble(asm))
    if needs_reduce:
        asm = support.reduceAsm(code_object_version, 0)
        emitted[support.REDUCE_KERNEL_NAME] = (asm, support.assemble(asm))
    return emitted


def write_kernels(path: str, emitted: dict[str, tuple[str, bytes]]) -> None:
    os.makedirs(path, exist_ok=True)
    for name, (asm, hsaco) in sorted(emitted.items()):
        with open(os.path.join(path, f"{name}.s"), "w") as stream:
            stream.write(asm)
        with open(os.path.join(path, f"{name}.hsaco"), "wb") as stream:
            stream.write(hsaco)


def validate_selection(
    cases: list[SweepCase],
    selection: Selection,
    tolerance: float,
    iters: int,
    warmup: int,
    verify_full_max_elems: int,
    emitted: dict[str, tuple[str, bytes]],
) -> dict[int, Validation]:
    """Re-times and correctness-checks every selected case before output."""
    from contextlib import ExitStack

    import bench_skinny_gemm as bench
    import skinny_gemm_support as support
    from skinny_gemm_support import DeviceBuffer
    from sweep_skinny_gemm_flush import FastLlcFlush
    from sweep_skinny_gemm_inputs import OperandCache, Verifier, make_case_inputs

    if not selection.covers:
        return {}
    validations: dict[int, Validation] = {}
    operands = OperandCache()
    loaded: dict[str, support.LoadedKernel] = {}

    def kernel_for(name: str) -> support.LoadedKernel:
        kernel = loaded.get(name)
        if kernel is None:
            kernel = support.loadKernel(emitted[name][1], name)
            loaded[name] = kernel
        return kernel

    with FastLlcFlush() as flush:
        try:
            for index, cover in sorted(selection.covers.items()):
                case = cases[index]
                assert case.best_us is not None
                cfg = {**cover.entry.cfg, "kernel": cover.entry.kernel}
                problem = validation_problem(case, cfg)
                inputs = make_case_inputs(problem, operands)
                d_host, ws_host = support.makeOutputs(
                    problem.m, problem.n, problem.ldD, problem.splits
                )
                with ExitStack() as stack:
                    arrays = (
                        d_host,
                        inputs.a.raw,
                        inputs.b.raw,
                        inputs.a.swizzled,
                        inputs.b.swizzled,
                        ws_host,
                        inputs.c_bits,
                    )
                    d, a, b, scl_a, scl_b, ws, c = (
                        stack.enter_context(DeviceBuffer(array)) for array in arrays
                    )
                    kernel = kernel_for(cover.entry.kernel)
                    gemm_args = support.gemmArgs(problem, d, a, b, scl_a, scl_b, ws, c)
                    block = support.gemmBlock(problem)
                    if warmup:
                        bench.timeLaunches(
                            kernel, problem.grid, block, gemm_args, warmup, flush
                        )
                    gemm_ns = bench.timeLaunches(
                        kernel, problem.grid, block, gemm_args, iters, flush
                    )
                    if problem.splits > 1:
                        reducer = kernel_for(support.REDUCE_KERNEL_NAME)
                        reduce_grid, reduce_block = support.reduceLaunchDims(
                            problem.m, problem.n
                        )
                        reduce_args = support.reduceArgs(
                            d,
                            ws,
                            c,
                            problem.ldD,
                            problem.m,
                            problem.n,
                            problem.splits,
                            problem.ldC,
                            problem.alpha,
                            problem.beta,
                        )
                        if warmup:
                            bench.timeLaunches(
                                reducer,
                                reduce_grid,
                                reduce_block,
                                reduce_args,
                                warmup,
                                None,
                            )
                        reduce_ns = bench.timeLaunches(
                            reducer, reduce_grid, reduce_block, reduce_args, iters, None
                        )
                        time_us = min(
                            gemm + reduce for gemm, reduce in zip(gemm_ns, reduce_ns)
                        )
                    else:
                        time_us = min(gemm_ns)

                    d.upload(d_host)
                    ws.upload(ws_host)
                    support.launchKernel(kernel, problem.grid, block, gemm_args)
                    d_out, ws_out = d_host.copy(), ws_host.copy()
                    d.download(d_out)
                    ws.download(ws_out)
                    verifier = Verifier(
                        inputs,
                        (problem.m + problem.n) * problem.k <= verify_full_max_elems,
                    )
                    verifier.check_gemm(problem, d_out, ws_out)
                    if problem.splits > 1:
                        support.launchKernel(
                            reducer, reduce_grid, reduce_block, reduce_args
                        )
                        d.download(d_out)
                        verifier.check_reduced(problem, d_out)

                measured_us = time_us * 1e-3
                limit_us = case.best_us * (1 + tolerance)
                if measured_us > limit_us:
                    raise ValueError(
                        f"{case.text()}: {cover.entry.kernel} measured "
                        f"{measured_us:.4f} us, outside {limit_us:.4f} us tolerance limit"
                    )
                validations[index] = Validation(measured_us, limit_us)
        finally:
            for kernel in loaded.values():
                kernel.close()
    return validations


def case_assignment_record(
    case: SweepCase, cover: Cover, validation: Validation
) -> dict:
    assert case.best_us is not None
    record = dict(case.label)
    record.update(
        kernel=cover.entry.kernel,
        timeUs=rounded(cover.entry.time_us),
        bestUs=rounded(case.best_us),
        ratio=rounded(cover.entry.time_us / case.best_us),
        rank=cover.entry.rank,
        validation={
            "timeUs": rounded(validation.time_us),
            "limitUs": rounded(validation.limit_us),
            "ratio": rounded(validation.time_us / case.best_us),
            "correct": True,
            "withinTolerance": True,
        },
    )
    return record


def selected_record(
    identity: ConfigIdentity,
    hits: Counter[ConfigIdentity],
    cases: list[SweepCase],
    covers: dict[int, Cover],
    representatives: dict[ConfigIdentity, dict],
    validations: dict[int, Validation],
) -> dict:
    assigned = [i for i, cover in covers.items() if cover.identity == identity]
    kernels = sorted({covers[i].entry.kernel for i in assigned})
    return {
        "config": config_record(identity),
        "sweepConfig": representatives.get(identity),
        "toleranceHits": hits[identity],
        "num_cases": len(assigned),
        "kernels": kernels,
        "cases": [
            case_assignment_record(cases[i], covers[i], validations[i])
            for i in sorted(assigned)
        ],
    }


def validate_selected_rows(selected: list[dict]) -> None:
    for index, row in enumerate(selected):
        kernels = row["kernels"]
        if len(kernels) != 1:
            raise ValueError(
                f"selected[{index}] has {len(kernels)} kernels, expected 1: {kernels}"
            )
        kernel = kernels[0]
        if not is_use_beta_kernel_name(kernel):
            raise ValueError(f"selected[{index}] kernel is not USE_BETA1: {kernel}")
        for case in row["cases"]:
            if case.get("kernel") != kernel:
                raise ValueError(
                    f"selected[{index}] case kernel {case.get('kernel')!r} "
                    f"!= row kernel {kernel!r}"
                )


def uncovered_record(
    case: SweepCase,
    qualified: dict[ConfigIdentity, QualifiedEntry],
    library: set[ConfigIdentity],
    tolerance: float,
) -> dict:
    record = dict(case.label)
    record["bestUs"] = None if case.best_us is None else rounded(case.best_us)
    if not qualified:
        record["reason"] = "no valid USE_BETA1 configurations in sweep top"
        record["sweepBest"] = None
        record["bestInLibrary"] = None
        return record
    assert case.best_us is not None
    identity, entry = fastest_winner(qualified)
    record["sweepBest"] = {
        "config": config_record(identity),
        "timeUs": rounded(entry.time_us),
        "ratio": rounded(entry.time_us / case.best_us),
        "withinTolerance": entry.time_us <= case.best_us * (1 + tolerance),
        "inLibrary": identity in library,
    }
    in_lib = {i: e for i, e in qualified.items() if i in library}
    if in_lib:
        lib_id, lib_entry = fastest_winner(in_lib)
        record["bestInLibrary"] = {
            "config": config_record(lib_id),
            "timeUs": rounded(lib_entry.time_us),
            "ratio": rounded(lib_entry.time_us / case.best_us),
            "withinTolerance": lib_entry.time_us <= case.best_us * (1 + tolerance),
        }
    else:
        record["bestInLibrary"] = None
    if in_lib:
        record["reason"] = "within tolerance but lost disjoint assignment"
    elif record["sweepBest"]["withinTolerance"]:
        record["reason"] = (
            "sweep best not in library; no library config within tolerance"
        )
    else:
        record["reason"] = "no configuration within tolerance"
    return record


def write_selection_yaml(
    path: str,
    results_path: str,
    cases: list[SweepCase],
    selection: Selection,
    tolerance: float,
    max_configs: int,
    min_config_cases: int,
    only: str | None,
    limit_cases: int | None,
    validations: dict[int, Validation],
    validation_iters: int,
    validation_warmup: int,
    kernels_path: str,
    code_object_version: int,
) -> None:
    qualified = build_qualified(cases, tolerance)
    library_set = set(selection.library)
    uncovered = [
        uncovered_record(case, qualified.get(case.index, {}), library_set, tolerance)
        for case in cases
        if case.index not in selection.covers
    ]
    selected_rows = [
        selected_record(
            identity,
            selection.tolerance_hits,
            cases,
            selection.covers,
            selection.representatives,
            validations,
        )
        for identity in selection.library
    ]
    validate_selected_rows(selected_rows)
    document = {
        "meta": {
            "tool": "select_skinny_gemm_from_sweep.py",
            "results": os.path.abspath(results_path),
            "tolerance": tolerance,
            "max_configs": max_configs,
            "min_config_cases": min_config_cases,
            "use_beta_only": True,
            "only": only,
            "limit_cases": limit_cases,
            "num_cases": len(cases),
            "num_selected": len(selection.library),
            "num_covered": len(selection.covers),
            "num_uncovered": len(uncovered),
            "codeObjectVersion": code_object_version,
            "kernels": os.path.abspath(kernels_path),
            "validation": {
                "iters": validation_iters,
                "warmup": validation_warmup,
                "codeObjectVersion": code_object_version,
                "correct": True,
                "withinTolerance": True,
            },
        },
        "selected": selected_rows,
        "uncovered": uncovered,
    }
    with open(path, "w") as stream:
        yaml.safe_dump(
            document, stream, sort_keys=False, width=1000, default_flow_style=None
        )


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be at least 0, got {value}")
    return value


def non_negative_finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError(
            f"must be a finite non-negative number, got {text}"
        )
    return value


def make_parser() -> argparse.ArgumentParser:
    import skinny_gemm_support as support

    parser = argparse.ArgumentParser(
        description="Select skinny GEMM kernels from a one-shot sweep results file."
    )
    parser.add_argument(
        "results",
        help="msgpack or YAML results from sweep_skinny_gemm.py (--one-shot)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="selection.yaml",
        help="output YAML path (default: selection.yaml)",
    )
    parser.add_argument(
        "--kernels",
        required=True,
        help="directory for each selected kernel's assembly (.s) and code object (.hsaco)",
    )
    parser.add_argument(
        "--code-object-version",
        type=code_object_version,
        default=support.DEFAULT_CODE_OBJECT_VERSION,
        help="code object version used to emit, time, and verify the kernels "
        f"(cov4, cov5, or cov6; default cov{support.DEFAULT_CODE_OBJECT_VERSION})",
    )
    parser.add_argument(
        "--tolerance",
        type=non_negative_finite_float,
        default=DEFAULT_TOLERANCE,
        help="allowed relative slowdown over each case's best time (default 0.10)",
    )
    parser.add_argument(
        "--max-configs",
        type=positive_int,
        required=True,
        help="maximum number of kernel configurations in the library",
    )
    parser.add_argument(
        "--min-config-cases",
        type=non_negative_int,
        default=0,
        help="drop configs with fewer tolerance hits before applying --max-configs",
    )
    parser.add_argument("--only", help="keep cases whose name contains this substring")
    parser.add_argument(
        "--limit-cases",
        type=non_negative_int,
        help="keep the first N cases after --only",
    )
    parser.add_argument(
        "--validate-iters",
        type=positive_int,
        help="timed GPU launches per selected case (default: sweep settings.iters)",
    )
    parser.add_argument(
        "--validate-warmup",
        type=non_negative_int,
        help="warmup GPU launches per selected case (default: sweep settings.warmup)",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        cases = load_cases(args.results, args.only, args.limit_cases)
        validation_iters, validation_warmup, verify_full_max_elems = validation_settings(
            args.results, args.validate_iters, args.validate_warmup
        )
    except (OSError, UnicodeError, ValueError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    if not cases:
        print("error: no cases with top timings in results", file=sys.stderr)
        return 2
    selection = select_library(
        cases, args.tolerance, args.max_configs, args.min_config_cases
    )
    try:
        emitted = emit_kernels(cases, selection, args.code_object_version)
    except (OSError, RuntimeError, ValueError) as err:
        print(f"error: kernel emission failed: {err}", file=sys.stderr)
        return 1
    try:
        validation = validate_selection(
            cases,
            selection,
            args.tolerance,
            validation_iters,
            validation_warmup,
            verify_full_max_elems,
            emitted,
        )
    except (AssertionError, OSError, RuntimeError, ValueError) as err:
        print(f"error: selection validation failed: {err}", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    write_kernels(args.kernels, emitted)
    write_selection_yaml(
        args.output,
        args.results,
        cases,
        selection,
        args.tolerance,
        args.max_configs,
        args.min_config_cases,
        args.only,
        args.limit_cases,
        validation,
        validation_iters,
        validation_warmup,
        args.kernels,
        args.code_object_version,
    )
    print(
        f"selected {len(selection.library)} configurations covering "
        f"{len(selection.covers)} of {len(cases)} cases -> {args.output}; "
        f"cov{args.code_object_version} kernels -> {args.kernels}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
