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

Config identity is one launch parameterization: bm, tn, lanesK, waves, prefetch,
upIters, group, numXcds, grid3d, blocks, mfma, lds, use_beta true.
--max-configs bounds the number of distinct identities. Identities that differ
only by blocks share a kernel symbol, so fewer kernel binaries may be emitted.

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
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import mean
from typing import NamedTuple

import yaml

from sweep_skinny_gemm_results import msgpack, read_results

DEFAULT_TOLERANCE = 0.10
DEFAULT_VALIDATE_ROUNDS = 3
# Sweep settings may be too small for a stable replay.
MIN_VALIDATE_WARMUP = 10
MIN_VALIDATE_ITERS = 30
# bf16 has 7 explicit mantissa bits and the kernel and the reference both
# round fp32 to bf16, so they differ by at most one ulp, which is at most
# 2^-7 relative. The verifier's atol floor still covers values near zero.
BF16_EPS = 2.0**-7
DEFAULT_VALIDATE_MARGIN = 0.05
# Everything reading a malformed or foreign results file may raise.
INPUT_ERRORS: tuple[type[Exception], ...] = (
    OSError,
    UnicodeError,
    ValueError,
    KeyError,
    TypeError,
    RuntimeError,
    yaml.YAMLError,
    *((msgpack.exceptions.UnpackException,) if msgpack is not None else ()),
)
LABEL_KEYS = ("name", "m", "n", "k", "spk", "useBeta", "betaMode")


class ConfigIdentity(NamedTuple):
    bm: int
    tn: int
    lanes_k: int
    waves: int
    pf: int
    up_iters: int
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
        spk = "k" if p["spk"] is None else p["spk"]
        return f"{p['name']} m={p['m']} n={p['n']} k={p['k']} spk={spk}"


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
    reference_us: float
    limit_us: float
    correct: bool
    within_tolerance: bool


@dataclass(frozen=True)
class Selection:
    library: list[ConfigIdentity]
    tolerance_hits: Counter[ConfigIdentity]
    covers: dict[int, Cover]
    representatives: dict[ConfigIdentity, dict]
    qualified: dict[int, dict[ConfigIdentity, QualifiedEntry]]


def identity_of(cfg: dict) -> ConfigIdentity:
    grid3d = bool(cfg["grid3d"])
    return ConfigIdentity(
        cfg["bm"],
        cfg["tn"],
        cfg["lanesK"],
        cfg["waves"],
        cfg["prefetch"],
        int(cfg["upIters"]),
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
    return {name: getattr(identity, name) for name in identity._fields}


def read_document(path: str) -> dict:
    document = read_results(path)
    if not isinstance(document, dict) or not isinstance(
        document.get("results"), list
    ):
        raise ValueError(f"{path}: not a sweep results document (no results list)")
    meta = document.get("meta", {})
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: results meta is not a mapping")
    return document


def positive_time(item: dict) -> float:
    time_us = item["timeUs"]
    if not isinstance(time_us, (int, float)) or not time_us > 0:
        raise ValueError(f"sweep entry has non-positive timeUs: {time_us!r}")
    return time_us


def load_cases(path: str, only: str | None, limit: int | None) -> list[SweepCase]:
    document = read_document(path)
    meta = document.get("meta", {})
    if meta.get("oneShot") is not True:
        raise ValueError(
            "results must come from sweep_skinny_gemm.py --one-shot "
            "(meta.oneShot is not true)"
        )
    if meta.get("interrupted") or meta.get("aborted"):
        raise ValueError("results are from an interrupted or aborted sweep")
    cases_done = meta.get("casesDone")
    cases_total = meta.get("casesTotal")
    if cases_done is None or cases_total is None or cases_done != cases_total:
        raise ValueError(
            "results are incomplete "
            f"({cases_done} of {cases_total} cases finished)"
        )
    entries = [
        entry for entry in document["results"] if keep_problem(entry["problem"])
    ]
    if only:
        entries = [entry for entry in entries if only in entry["problem"]["name"]]
    if limit is not None:
        entries = entries[:limit]
    cases = []
    for index, entry in enumerate(entries):
        top = sorted(
            (item for item in entry.get("top") or [] if top_entry_allowed(item)),
            key=positive_time,
        )
        best_us = top[0]["timeUs"] if top else None
        cases.append(SweepCase(index, entry["problem"], best_us, top))
    return cases


def validation_settings(
    path: str, iters: int | None, warmup: int | None
) -> tuple[int, int, int]:
    """Explicit CLI values are used as is; otherwise the sweep settings are
    raised to the MIN_VALIDATE_* floors."""
    settings = read_document(path).get("meta", {}).get("settings", {})
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
    if iters is None:
        selected_iters = max(selected_iters, MIN_VALIDATE_ITERS)
    if warmup is None:
        selected_warmup = max(selected_warmup, MIN_VALIDATE_WARMUP)
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
    """Improve greedy coverage with deterministic, coverage-increasing swaps.

    Returns a new set; the argument is left untouched."""
    library = set(library)
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
    return Selection(library, global_hits, covers, reps, qualified)


def validation_problem(case: SweepCase, cfg: dict):
    """Reconstructs the swept problem so the emitted symbol is re-run exactly."""
    import skinny_gemm_support as support

    align = support.alignUp(case.problem["k"], 4)
    mfma = bool(cfg.get("mfma", False))
    # The kernel name tells the variant, so old and new result files rebuild the
    # same symbol.
    spk = cfg["spk"] if cfg["kernel"].endswith("_USE_SPLIT_K1") else None
    problem = support.makeProblem(
        case.problem["m"],
        case.problem["n"],
        case.problem["k"],
        spk=spk,
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


@contextmanager
def correctness_errors(case: SweepCase, kernel: str):
    """Reports a failed correctness assertion as a ValueError for main()."""
    try:
        yield
    except AssertionError as err:
        raise ValueError(
            f"{case.text()}: {kernel} failed the correctness check (D must "
            f"match the reference within the bf16 machine epsilon {BF16_EPS} "
            f"relative): {err}"
        ) from err


def validate_selection(
    cases: list[SweepCase],
    selection: Selection,
    tolerance: float,
    iters: int,
    warmup: int,
    verify_full_max_elems: int,
    code_object_version: int,
    emitted: dict[str, tuple[str, bytes]],
    rounds: int,
    margin: float,
) -> dict[int, Validation]:
    """Checks selected performance against a same-session replayed sweep best.

    Selection admits kernels up to the tolerance, so a check at the same
    tolerance fails on replay noise for every case near that edge; margin is
    the extra relative slowdown the replay may show on top of the tolerance."""
    from contextlib import ExitStack

    import bench_skinny_gemm as bench
    import skinny_gemm_support as support
    from skinny_gemm_support import DeviceBuffer
    from sweep_skinny_gemm_flush import FastLlcFlush
    from sweep_skinny_gemm_inputs import OperandCache, Verifier, make_case_inputs

    if not selection.covers:
        return {}
    validations: dict[int, Validation] = {}
    failures: list[str] = []
    operands = OperandCache()
    loaded: dict[str, support.LoadedKernel] = {}

    def kernel_for(name: str, problem) -> support.LoadedKernel:
        kernel = loaded.get(name)
        if kernel is None:
            hsaco = emitted.get(name, (None, None))[1]
            if hsaco is None:
                asm = support.gemmAsm(problem, code_object_version, 0)
                hsaco = support.assemble(asm)
            kernel = support.loadKernel(hsaco, name)
            loaded[name] = kernel
        return kernel

    def time_problem(problem, kernel, d, a, b, scl_a, scl_b, ws, c, flush) -> float:
        gemm_args = support.gemmArgs(problem, d, a, b, scl_a, scl_b, ws, c)
        block = support.gemmBlock(problem)
        if warmup:
            bench.timeLaunches(kernel, problem.grid, block, gemm_args, warmup, flush)
        gemm_ns = bench.timeLaunches(
            kernel, problem.grid, block, gemm_args, iters, flush
        )
        if problem.splits == 1:
            return mean(gemm_ns) * 1e-3
        reducer = kernel_for(support.REDUCE_KERNEL_NAME, problem)
        reduce_grid, reduce_block = support.reduceLaunchDims(problem.m, problem.n)
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
                reducer, reduce_grid, reduce_block, reduce_args, warmup, None
            )
        reduce_ns = bench.timeLaunches(
            reducer, reduce_grid, reduce_block, reduce_args, iters, None
        )
        return mean(gemm + reduce for gemm, reduce in zip(gemm_ns, reduce_ns)) * 1e-3

    with FastLlcFlush() as flush:
        try:
            for index, cover in sorted(selection.covers.items()):
                case = cases[index]
                assert case.best_us is not None
                selected_cfg = {**cover.entry.cfg, "kernel": cover.entry.kernel}
                selected_problem = validation_problem(case, selected_cfg)
                best = case.top[0]
                best_problem = validation_problem(
                    case, {**best["config"], "kernel": best["kernel"]}
                )
                inputs = make_case_inputs(selected_problem, operands)
                d_host, ws_host = support.makeOutputs(
                    selected_problem.m,
                    selected_problem.n,
                    selected_problem.ldD,
                    selected_problem.splits,
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
                    selected_kernel = kernel_for(cover.entry.kernel, selected_problem)
                    best_kernel = kernel_for(best["kernel"], best_problem)
                    buffers = (d, a, b, scl_a, scl_b, ws, c)
                    same = selected_problem == best_problem
                    reference_samples: list[float] = []
                    time_samples: list[float] = []
                    # Alternating the order each round keeps slow clock drift
                    # from favoring whichever kernel is timed first. Every
                    # round contributes the same iters, so the mean across
                    # rounds is the mean across all its launches, matching
                    # the sweep's own mean-of-launches timeUs.
                    for round_index in range(rounds):
                        order = ("best", "selected")
                        if round_index % 2:
                            order = order[::-1]
                        for which in order:
                            if which == "best":
                                reference_samples.append(
                                    time_problem(
                                        best_problem, best_kernel, *buffers, flush
                                    )
                                )
                            elif not same:
                                time_samples.append(
                                    time_problem(
                                        selected_problem,
                                        selected_kernel,
                                        *buffers,
                                        flush,
                                    )
                                )
                    reference_us = mean(reference_samples)
                    time_us = reference_us if same else mean(time_samples)

                    d.upload(d_host)
                    ws.upload(ws_host)
                    selected_args = support.gemmArgs(
                        selected_problem, d, a, b, scl_a, scl_b, ws, c
                    )
                    selected_block = support.gemmBlock(selected_problem)
                    support.launchKernel(
                        selected_kernel,
                        selected_problem.grid,
                        selected_block,
                        selected_args,
                    )
                    d_out, ws_out = d_host.copy(), ws_host.copy()
                    d.download(d_out)
                    ws.download(ws_out)
                    verifier = Verifier(
                        inputs,
                        (selected_problem.m + selected_problem.n) * selected_problem.k
                        <= verify_full_max_elems,
                        d_rtol=BF16_EPS,
                    )
                    with correctness_errors(case, cover.entry.kernel):
                        verifier.check_gemm(selected_problem, d_out, ws_out)
                        if selected_problem.splits > 1:
                            reducer = kernel_for(
                                support.REDUCE_KERNEL_NAME, selected_problem
                            )
                            reduce_grid, reduce_block = support.reduceLaunchDims(
                                selected_problem.m, selected_problem.n
                            )
                            reduce_args = support.reduceArgs(
                                d,
                                ws,
                                c,
                                selected_problem.ldD,
                                selected_problem.m,
                                selected_problem.n,
                                selected_problem.splits,
                                selected_problem.ldC,
                                selected_problem.alpha,
                                selected_problem.beta,
                            )
                            support.launchKernel(
                                reducer, reduce_grid, reduce_block, reduce_args
                            )
                            d.download(d_out)
                            verifier.check_reduced(selected_problem, d_out)

                if not reference_us > 0:
                    raise ValueError(
                        f"{case.text()}: replayed best time {reference_us} us is not positive"
                    )
                limit_us = reference_us * (1 + tolerance + margin)
                within_tolerance = time_us <= limit_us
                if not within_tolerance:
                    failures.append(
                        f"{case.text()}: {cover.entry.kernel} measured "
                        f"{time_us:.4f} us, outside {limit_us:.4f} us limit "
                        f"(ratio {time_us / reference_us:.4f}, sweep ratio "
                        f"{cover.entry.time_us / case.best_us:.4f}) from replayed "
                        f"best {best['kernel']} ({reference_us:.4f} us)"
                    )
                # The verifier raises on a mismatch, so reaching here means correct.
                validations[index] = Validation(
                    time_us, reference_us, limit_us, True, within_tolerance
                )
        finally:
            for kernel in loaded.values():
                kernel.close()
    if failures:
        shown = "\n  ".join(failures[:20])
        more = f"\n  ... and {len(failures) - 20} more" if len(failures) > 20 else ""
        raise ValueError(
            f"{len(failures)} of {len(selection.covers)} selected cases exceed "
            f"tolerance {tolerance} + validation margin {margin}:\n  {shown}{more}"
        )
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
            "replayedBestUs": rounded(validation.reference_us),
            "limitUs": rounded(validation.limit_us),
            "ratio": rounded(validation.time_us / validation.reference_us),
            "correct": validation.correct,
            "withinTolerance": validation.within_tolerance,
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
    validation_rounds: int,
    validation_margin: float,
) -> None:
    qualified = selection.qualified
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
                "performanceReference": "replayed case-best kernel",
                "rounds": validation_rounds,
                "margin": validation_margin,
                "correct": all(v.correct for v in validations.values()),
                "withinTolerance": all(
                    v.within_tolerance for v in validations.values()
                ),
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


def at_least_two_int(text: str) -> int:
    value = int(text)
    if value < 2:
        raise argparse.ArgumentTypeError(
            f"must be at least 2 so the timing order alternates, got {value}"
        )
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
        help="timed GPU launches per selected case "
        f"(default: max(sweep settings.iters, {MIN_VALIDATE_ITERS}))",
    )
    parser.add_argument(
        "--validate-warmup",
        type=non_negative_int,
        help="warmup GPU launches per selected case "
        f"(default: max(sweep settings.warmup, {MIN_VALIDATE_WARMUP}))",
    )
    parser.add_argument(
        "--validate-rounds",
        type=at_least_two_int,
        default=DEFAULT_VALIDATE_ROUNDS,
        help="alternating timing rounds of the selected and the replayed best "
        "kernel per case, averaged over all their launches "
        f"(default {DEFAULT_VALIDATE_ROUNDS})",
    )
    parser.add_argument(
        "--validate-margin",
        type=non_negative_finite_float,
        default=DEFAULT_VALIDATE_MARGIN,
        help="replay slowdown allowed on top of --tolerance, since selection "
        f"already uses the full tolerance (default {DEFAULT_VALIDATE_MARGIN})",
    )
    return parser


def main() -> int:
    args = make_parser().parse_args()
    try:
        cases = load_cases(args.results, args.only, args.limit_cases)
        validation_iters, validation_warmup, verify_full_max_elems = validation_settings(
            args.results, args.validate_iters, args.validate_warmup
        )
    except INPUT_ERRORS as err:
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
            args.code_object_version,
            emitted,
            args.validate_rounds,
            args.validate_margin,
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
        args.validate_rounds,
        args.validate_margin,
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
