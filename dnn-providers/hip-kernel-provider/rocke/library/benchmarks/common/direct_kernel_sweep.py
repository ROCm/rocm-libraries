# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The AOT cache for the direct convolution kernels.

Direct conv splits its parameters differently from implicit GEMM. Batch,
spatial extents and the group count are kernargs, so one binary serves any
image; the filter geometry (``KH``/``KW``), ``stride``, ``PAD`` and the
per-group channel counts (``cpg``/``kpg``) shape the unrolled MFMA chain and
the LDS row layout, so they are *capabilities* baked into the binary. A cache
therefore holds one binary per (capability, tuning knob) pair, and
``--run-from-cache`` offers a binary only to problems with exactly its
capabilities.

Which capabilities to build is a product decision. :data:`DIRECT_CAPABILITIES`
follows the set the reference direct-conv library covers on CDNA and drops
what rocke's kernels cannot compute (see the table's comments).

The non-grouped (``groups == 1``) forward kernel bakes less: its channel counts
are kernargs as well, so its capabilities record ``cpg = kpg = 0`` (runtime)
and one binary serves every ``C``/``K`` its tile divides. The wgrad kernel is
built the same way (``runtime_channels``): one binary per filter, padding and
tuning knob serves every per-group channel count its knobs fit. So is the
runtime-channel MFMA dgrad pipeline (``direct_dgrad_mfma_rtc``), which serves
dense as well as grouped problems: its compute pass keeps the weights of its
K-atoms in registers, so what it bakes is the *number* of K-atoms and M-tiles
(knobs ``k_atoms`` / ``m_tiles``), and the exact channel counts are kernargs.

The run side is split in two. :func:`direct_plans` is pure CPU: it picks the
cached kernels that fit a problem, re-validates each spec against the real
shape, and returns a launch plan -- binaries, signatures, grids and the
buffer each kernarg slot binds to. The benchmark only allocates memory and
launches the plan.
"""

from __future__ import annotations

import collections
import hashlib
import json
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from benchmarks.common.kernel_cache import KernelCache, KernelIdentity
from benchmarks.common.kernel_sweep import BuildJob, compile_jobs
from kernels.common.conv_abi import conv_direct_transform_signature
from kernels.conv.direct_aot_jobs import (  # noqa: F401 (the grid and builder, re-exported)
    BLOCK_GROUPS,
    BLOCK_Q,
    DGRAD_4C_BLOCK_GROUPS,
    DGRAD_4C_BLOCK_Q,
    DGRAD_4C_FORMS,
    DGRAD_BLOCK_H,
    DGRAD_BLOCK_Q,
    DGRAD_FUSED_BLOCK_H,
    DGRAD_FUSED_BLOCK_H_LARGE,
    DGRAD_FUSED_CHANNELS,
    DGRAD_FUSED_EXIT_BLOCK_H,
    DGRAD_FUSED_FILTERS,
    DGRAD_FUSED_LDS_PAD,
    DGRAD_FUSED_WPE,
    DGRAD_FUSED_WPE_MAX_VGPRS,
    DGRAD_RTC_LOOP_MAX_ATOMS,
    DGRAD_RTC_M_TILES,
    DGRAD_RTC_MAX_KPG,
    DGRAD_RTC_VGPR_FILL,
    DGRAD_WAVES,
    DIRECT_CAPABILITIES,
    DIRECT_DIRECTIONS,
    DIRECT_DTYPE,
    DIRECT_DTYPES,
    DOUBLE_BUFFER,
    DW_BLOCK_W_DGRAD,
    DW_BLOCK_W_FWD,
    DW_BLOCK_WAVES,
    DW_COL_BLOCK_H,
    DW_COL_BLOCK_W,
    DW_TILED_BLOCK_H,
    DW_TILED_BLOCK_W,
    DW_TILED_BLOCK_WAVES,
    DW_TILED_UNROLL,
    DW_WIN_MFMA_FILTERS,
    DW_WIN_UNROLL_BUDGET,
    WGRAD_HPB,
    WGRAD_MFMA_K,
    WGRAD_WAVES_C_WIDE,
    WGRAD_WAVES_FIXED,
    WGRAD_WAVES_K_WIDE,
    DirectCaps,
    _caps_jobs,
    _caps_of,
    _dgrad_fused_rule_block_groups,
    _helper_identity,
    _split_reduce_identity,
    count_direct_jobs,
    dgrad_rtc_k_atoms,
    dgrad_rtc_spills,
    dgrad_rtc_vgprs,
    direct_job_for_identity,
    direct_kernel_for_job,
    probe_problem,
    wgrad_waves,
)
from kernels.conv.direct_variants import (  # noqa: F401 (make_spec, validate_spec re-exported)
    DIRECT_DGRAD_REORGANIZE as _REORGANIZE,
    DIRECT_DGRAD_REORGANIZE_RTC as _REORGANIZE_RTC,
    DIRECT_DGRAD_TRANSPOSE as _TRANSPOSE,
    DIRECT_DGRAD_TRANSPOSE_RTC as _TRANSPOSE_RTC,
    DIRECT_HALO_VARIANTS as _HALO_VARIANTS,
    DIRECT_DGRAD_FPROP_SINGLE as _DGRAD_FPROP_SINGLE,
    DIRECT_MFMA_DGRAD as _MFMA_DGRAD,
    DIRECT_SPLIT_REDUCE as _SPLIT_REDUCE,
    DIRECT_WGRAD_FOLD as _WGRAD_FOLD,
    DIRECT_WGRAD_FOLDED as _WGRAD_FOLDED,
    direct_dgrad_weight_transforms,
    direct_wgrad_fold_knobs,
    direct_dgrad_workspaces,
    direct_entry_wave_reason,
    direct_spec_for_problem,
    make_direct_spec as make_spec,
    validate_direct_spec as validate_spec,
)


def _caps_worker(payload):
    return _caps_jobs(*payload)


def enumerate_direct_jobs(
    *,
    arch: str,
    target,
    directions: Sequence[str],
    dtype: str = DIRECT_DTYPE,
    jobs: int = 1,
    log=None,
    log_every_s: float = 5.0,
) -> List[BuildJob]:
    """Every direct kernel worth caching for ``arch``, pre-validated.

    ``directions`` takes the benchmark's names, ``fwd``, ``dgrad`` and
    ``wgrad``. The MFMA dgrad entries pull in their weight-transform helpers.

    The grid is walked one (variant, capability) cell at a time; with
    ``jobs > 1`` the cells run in a process pool. Cells are merged in order,
    so the result (including its order) does not depend on ``jobs``. With
    ``log`` set a progress line is emitted at most every ``log_every_s``
    seconds, as for the implicit-GEMM enumeration.
    """
    for d in directions:
        if d not in DIRECT_DIRECTIONS:
            raise ValueError(
                f"direct conv has no {d!r} kernel; expected one of {DIRECT_DIRECTIONS}"
            )
    if dtype not in DIRECT_DTYPES:
        raise ValueError(f"direct conv builds {DIRECT_DTYPES}, not {dtype!r}")
    wave_size = target.wave_size
    jobs = max(1, int(jobs))
    payloads = [
        (arch, wave_size, variant, direction, caps, dtype)
        for variant, (direction, caps_list) in DIRECT_CAPABILITIES.items()
        if direction in directions
        for caps in caps_list
    ]

    total = 0
    if log is not None:
        total = count_direct_jobs(directions, arch, dtype)
        log(f"  {total} candidates to check with {jobs} process(es)")

    n_raw = n_valid = 0
    started = last_log = time.perf_counter()
    results: List[Optional[List[Tuple[str, BuildJob]]]] = [None] * len(payloads)

    def _done(idx: int, cell_raw: int, cell_out) -> None:
        nonlocal n_raw, n_valid, last_log
        results[idx] = cell_out
        n_raw += cell_raw
        n_valid += len(cell_out)
        if log is None:
            return
        now = time.perf_counter()
        if now - last_log >= log_every_s:
            last_log = now
            pct = 100.0 * n_raw / total if total else 0.0
            log(
                f"  enumerating: {n_raw}/{total} candidates checked "
                f"({pct:.1f}%), ~{n_valid} valid ({now - started:.0f}s)"
            )

    if jobs <= 1:
        for idx, payload in enumerate(payloads):
            _done(idx, *_caps_jobs(*payload))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(_caps_worker, payload): idx
                for idx, payload in enumerate(payloads)
            }
            for fut in as_completed(futures):
                _done(futures[fut], *fut.result())

    seen = set()
    per_direction: Dict[str, List[BuildJob]] = {d: [] for d in directions}
    for payload, cell_out in zip(payloads, results):
        for key, job in cell_out:
            if key not in seen:
                seen.add(key)
                per_direction[payload[3]].append(job)
    # Round-robin across directions, as the implicit-GEMM enumeration does, so
    # a --limit smoke build covers every requested direction.
    out: List[BuildJob] = []
    streams = [iter(per_direction[d]) for d in directions]
    while streams:
        still_running = []
        for stream in streams:
            job = next(stream, None)
            if job is not None:
                out.append(job)
                still_running.append(stream)
        streams = still_running
    if log is not None:
        log(
            f"  enumerated {n_raw} candidates -> {len(out)} unique valid "
            f"({time.perf_counter() - started:.0f}s)"
        )
    return out


def compile_all_direct(
    *,
    cache: KernelCache,
    arch: str,
    target,
    directions: Sequence[str],
    jobs: int = 1,
    limit: Optional[int] = None,
    log=print,
    dtype: str = DIRECT_DTYPE,
) -> int:
    """Populate ``cache`` with every direct kernel in :data:`DIRECT_CAPABILITIES`
    for operand dtype ``dtype``."""
    log(
        f"AOT compile-all: enumerating direct variants for {arch}/{dtype} "
        f"({', '.join(directions)})"
    )
    all_jobs = enumerate_direct_jobs(
        arch=arch,
        target=target,
        directions=directions,
        dtype=dtype,
        jobs=jobs,
        log=log,
    )
    rc = compile_jobs(
        cache=cache,
        all_jobs=all_jobs,
        build=direct_kernel_for_job,
        arch=arch,
        dtype=dtype,
        directions=[f"direct_{d}" for d in directions],
        jobs=jobs,
        limit=limit,
        log=log,
    )
    if "fwd" in directions:
        prune_halo_entries(cache, all_jobs, log=log)
    return rc


def _elf_code_facts(hsaco: bytes) -> Optional[Tuple[bytes, int]]:
    """``(code digest, scratch bytes)`` of an ELF64 code object, or None for
    anything else.

    The code is the ``.text`` section followed by every kernel descriptor
    (``<kernel>.kd``): what the GPU executes and how it is launched, so two
    binaries with equal digests run identically whatever their symbol names.
    Only its SHA-256 is kept, so a sweep's worth of facts stays small.
    ``scratch`` is the largest descriptor's private segment size -- non-zero
    when the kernel spills.
    """
    if hsaco[:5] != b"\x7fELF\x02" or len(hsaco) < 64:
        return None
    (shoff,) = struct.unpack_from("<Q", hsaco, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", hsaco, 0x3A)
    secs = [
        struct.unpack_from("<IIQQQQII", hsaco, shoff + i * shentsize)
        for i in range(shnum)
    ]
    # (name, type, flags, addr, offset, size, link, info)
    strtab_off = secs[shstrndx][4]

    def cstr(off: int) -> str:
        return hsaco[off : hsaco.index(b"\0", off)].decode()

    text = b""
    kds = []
    for name_off, sh_type, _, _, off, size, link, _ in secs:
        if cstr(strtab_off + name_off) == ".text":
            text = hsaco[off : off + size]
        if sh_type != 2:  # SHT_SYMTAB
            continue
        names = secs[link][4]
        for sym in range(off, off + size, 24):
            st_name, _, _, shndx, value, st_size = struct.unpack_from(
                "<IBBHQQ", hsaco, sym
            )
            if not cstr(names + st_name).endswith(".kd") or shndx >= shnum:
                continue
            _, _, _, addr, sec_off, _, _, _ = secs[shndx]
            at = sec_off + value - addr
            kd = hsaco[at : at + st_size]
            # Bytes 16..24 locate the entry point relative to the descriptor:
            # file layout, not behaviour, so they are left out.
            kds.append(kd[:16] + kd[24:])
    if not kds:
        return None
    scratch = max(struct.unpack_from("<I", kd, 4)[0] for kd in kds)
    return hashlib.sha256(text + b"".join(sorted(kds))).digest(), scratch


def prune_halo_entries(cache: KernelCache, jobs: Sequence[BuildJob], log=print) -> int:
    """Drop the cached grouped-halo and split-C forward entries no sweep should
    be offered (the dense unsplit grid is left as it is).

    Two kinds, both only visible in the compiled code: binaries that spill to
    scratch (never the best of their shape), and ``waves_per_eu=3`` binaries
    whose code equals their ``waves_per_eu=None`` twin's -- the hint changed
    nothing, so the twin already is that kernel. The binaries stay in the blob
    store: a later ``--compile-all`` re-links those entries without compiling
    and this drops them again. Returns the number of entries dropped.
    """
    wanted = {
        j.identity
        for j in jobs
        if j.identity.direction == "direct_fwd"
        and (
            j.identity.algorithm == "direct_grouped_halo"
            or (
                j.identity.algorithm == "direct_nongrouped"
                and json.loads(j.identity.knobs).get("split_c", 1) > 1
            )
        )
    }
    present = {
        ident: path for ident, path in cache.list_all("direct_fwd") if ident in wanted
    }
    facts = {
        ident: _elf_code_facts(path.read_bytes()) for ident, path in present.items()
    }
    drop: Dict[KernelIdentity, str] = {}
    twins: Dict[KernelIdentity, Dict[object, KernelIdentity]] = {}
    for ident, fact in facts.items():
        if fact is not None and fact[1] > 0:
            drop[ident] = "spills"
        knobs = json.loads(ident.knobs)
        we = knobs.pop("waves_per_eu", None)
        key = replace(ident, knobs=json.dumps(knobs, sort_keys=True))
        twins.setdefault(key, {})[we] = ident
    for pair in twins.values():
        hinted, plain = pair.get(3), pair.get(None)
        if hinted is None or plain is None or hinted in drop:
            continue
        a, b = facts[hinted], facts[plain]
        if a is not None and b is not None and a[0] == b[0]:
            drop[hinted] = "same code as waves_per_eu=None"
    for ident in drop:
        cache.discard(ident)
    if drop:
        reasons = collections.Counter(drop.values())
        log(
            f"AOT compile-all: dropped {len(drop)} of {len(present)} grouped-halo "
            "and split-C forward entries ("
            + ", ".join(f"{n} {why}" for why, n in sorted(reasons.items()))
            + ")"
        )
    return len(drop)


# ---------------------------------------------------------------------------
# Running a problem out of the cache (CPU half)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectStep:
    """One kernel launch of a plan.

    ``buffers`` names the buffers bound to the A, B and D kernarg slots (B is
    ``None`` for the two-operand weight transforms). ``conv_args`` is ``None``
    for those too; a conv kernel takes its extents from it. ``scalars`` are
    the transforms' remaining kernargs (name, value), if any.
    """

    kernel_name: str
    hsaco_path: Path
    signature: list
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    buffers: Tuple[str, Optional[str], str]
    conv_args: object = None
    scalars: Tuple[Tuple[str, int], ...] = ()
    # Per A/B/D slot, ``(byte offset, byte size)`` of the window of its
    # buffer the step binds, or None for the whole buffer: a batch-split
    # launch sees only its images (see _batch_chunks).
    views: Tuple[Optional[Tuple[int, int]], ...] = ()


@dataclass(frozen=True)
class DirectPlan:
    """Everything needed to run one cached kernel (or pipeline) on a problem.

    Buffer names: ``x``/``w``/``y`` for forward, ``dy``/``w``/``dx`` for
    dgrad, ``dy``/``x``/``dw`` (fp32) for wgrad, plus the scratch buffers in
    ``workspaces`` (name -> bytes).
    """

    identity: KernelIdentity
    steps: Tuple[DirectStep, ...]
    workspaces: Dict[str, int] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.identity.label()


def _conv_step(identity, hsaco_path, meta, spec, problem, direction, buffers):
    from kernels.common.conv_abi import conv_direct_args_signature
    from kernels.common.conv_args import ConvArgs
    from kernels.common.conv_direct_dgrad_wmma import DirectDgradWmmaSpec
    from kernels.common.conv_direct_grouped import direct_launch_geometry
    from kernels.common.conv_direct_nongrouped import DirectNongroupedConvSpec

    if isinstance(spec, (DirectNongroupedConvSpec, DirectDgradWmmaSpec)):
        grid, block = spec.grid(), (spec.threads_per_block, 1, 1)
    elif hasattr(spec, "launch_geometry"):
        grid, block = spec.launch_geometry()
    else:
        grid, block = direct_launch_geometry(spec)
    return DirectStep(
        kernel_name=meta["kernel_name"],
        hsaco_path=hsaco_path,
        signature=conv_direct_args_signature(identity.dtype_a, direction=direction),
        grid=grid,
        block=block,
        buffers=buffers,
        conv_args=ConvArgs.from_problem(problem, direction=direction),
    )


def _batch_chunks(problem) -> Optional[List[int]]:
    """Images per launch for a forward problem the LDS-halo kernels can only
    run in pieces, or None when one launch serves it.

    Those kernels address every operand through one 32-bit buffer resource and
    redirect masked accesses past ``_OOB_BASE``, so no operand may reach it
    (a 768x768x256 bf16 batch of 8 is 2.4 GB). The images are independent, so
    the batch is split into launches whose activation and output windows each
    fit; the chunks are balanced rather than greedy.
    """
    from kernels.common.conv_direct_nongrouped import _OOB_BASE

    in_img = 2 * problem.H * problem.W * problem.total_c
    out_img = 2 * problem.Ho * problem.Wo * problem.total_k
    per_launch = _OOB_BASE // max(in_img, out_img)
    if problem.N <= per_launch or per_launch < 1:
        return None
    n_steps = -(-problem.N // per_launch)
    base, extra = divmod(problem.N, n_steps)
    return [base + (i < extra) for i in range(n_steps)]


def plan_for(
    cache: KernelCache,
    identity: KernelIdentity,
    hsaco_path: Path,
    meta: dict,
    problem,
    arch: str,
) -> Tuple[Optional[DirectPlan], str]:
    """Launch plan for one cached kernel on ``problem``, or ``(None, why)``.

    The capability match (:meth:`KernelCache.supports_problem`) is necessary
    but not sufficient: the knobs' own shape rules are re-validated on the
    real problem by
    :func:`~kernels.conv.direct_variants.direct_spec_for_problem`, so a
    binary is only offered where a fresh build would accept the shape too.
    """
    from kernels.common import conv_direct_grouped as dc

    if not meta.get("kernel_name"):
        return None, "cache entry has no kernel_name (rebuild the cache)"
    variant = identity.algorithm
    why = direct_entry_wave_reason(variant, identity.wave_size, meta)
    if why is not None:
        return None, why
    knobs = json.loads(identity.knobs) if identity.knobs else {}
    if variant in _HALO_VARIANTS and identity.direction == "direct_fwd":
        chunks = _batch_chunks(problem)
        if chunks is not None:
            if knobs.get("split_c", 1) > 1:
                return None, "a batch-split launch does not take split-C kernels"
            return _batch_split_plan(
                cache, identity, hsaco_path, meta, problem, arch, knobs, chunks
            )
    spec, why = direct_spec_for_problem(
        variant, problem, knobs, arch, identity.wave_size
    )
    if spec is None:
        return None, why

    if identity.direction == "direct_fwd" and getattr(spec, "split_c", 1) > 1:
        # Split-C: the conv pass writes fp32 partial sums into a workspace,
        # and the reduction folds them into y. Both take the forward ABI.
        from kernels.common.conv_args import ConvArgs

        reduce_id = _split_reduce_identity(
            identity.arch, identity.wave_size, spec.split_c, identity.dtype_a
        )
        entry = cache.get(reduce_id)
        if entry is None:
            return None, f"missing cached {_SPLIT_REDUCE} helper (rebuild the cache)"
        reduce_spec = make_spec(_SPLIT_REDUCE, problem, {"split_c": spec.split_c})
        conv = _conv_step(
            identity, hsaco_path, meta, spec, problem, "fwd", ("x", "w", "ws_split")
        )
        fold = DirectStep(
            kernel_name=entry[1]["kernel_name"],
            hsaco_path=cache.hsaco_path(reduce_id),
            signature=conv.signature,
            grid=reduce_spec.grid(problem),
            block=(reduce_spec.threads, 1, 1),
            buffers=("ws_split", "w", "y"),
            conv_args=ConvArgs.from_problem(problem, direction="fwd"),
        )
        return (
            DirectPlan(identity, (conv, fold), {"ws_split": spec.workspace_bytes}),
            "ok",
        )
    if identity.direction == "direct_fwd":
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "fwd", ("x", "w", "y")
        )
        return DirectPlan(identity, (step,)), "ok"
    if identity.direction == "direct_wgrad" and variant in _WGRAD_FOLDED:
        # The pass stores one partial slot per partition into the scratch
        # ``ws``; the fold sums them into dW (written whole, no zeroing).
        from kernels.common.conv_direct_dw_wgrad import (
            DW_WGRAD_FOLD_ARGS,
            dw_wgrad_fold_signature,
        )

        caps = _caps_of(identity)
        fold_knobs = direct_wgrad_fold_knobs(variant, caps.KH, caps.KW, caps.cpg)
        fold_id = _helper_identity(
            identity.arch,
            identity.wave_size,
            _WGRAD_FOLD,
            caps,
            fold_knobs,
            identity.dtype_a,
        )
        entry = cache.get(fold_id)
        if entry is None:
            return None, "missing cached wgrad fold (rebuild the cache)"
        fold_spec = make_spec(_WGRAD_FOLD, problem, fold_knobs)
        channels = (
            spec.fold_channels() if hasattr(spec, "fold_channels") else problem.groups
        )
        grid, block = fold_spec.launch_geometry(channels)
        main = _conv_step(
            identity, hsaco_path, meta, spec, problem, "wgrad", ("dy", "x", "ws")
        )
        fold = DirectStep(
            kernel_name=entry[1]["kernel_name"],
            hsaco_path=cache.hsaco_path(fold_id),
            signature=dw_wgrad_fold_signature(fold_spec),
            grid=grid,
            block=block,
            buffers=("ws", None, "dw"),
            scalars=tuple(zip(DW_WGRAD_FOLD_ARGS, (spec.n_partitions(), channels))),
        )
        return DirectPlan(identity, (main, fold), {"ws": spec.workspace_bytes()}), "ok"
    if identity.direction == "direct_wgrad":
        # dY rides A, X rides B, and D is the fp32 dW the kernel accumulates
        # into atomically (the caller zeroes it).
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "wgrad", ("dy", "x", "dw")
        )
        return DirectPlan(identity, (step,)), "ok"
    if variant in _DGRAD_FPROP_SINGLE:
        # A single forward kernel over the transposed problem that reads the
        # original weights itself: forward ABI, transposed extents.
        step = _conv_step(
            identity, hsaco_path, meta, spec, spec.problem, "fwd", ("dy", "w", "dx")
        )
        return DirectPlan(identity, (step,)), "ok"
    if variant not in _MFMA_DGRAD:
        step = _conv_step(
            identity, hsaco_path, meta, spec, problem, "dgrad", ("dy", "w", "dx")
        )
        return DirectPlan(identity, (step,)), "ok"

    # MFMA pipeline: W -> W_T (transpose), optionally W_T -> W_coa
    # (reorganize), then a forward pass over the transposed problem.
    caps = _caps_of(identity)
    runtime_channels = variant == "direct_dgrad_mfma_rtc"
    helpers = direct_dgrad_weight_transforms(knobs, variant)
    # The runtime-channel transforms take the original problem's per-group
    # channel counts as kernargs.
    scalars = (
        tuple(zip(dc.DIRECT_WEIGHT_TRANSFORM_RTC_ARGS, (problem.cpg, problem.kpg)))
        if runtime_channels
        else ()
    )
    steps = []
    for helper, helper_knobs, src, dst in helpers:
        helper_id = _helper_identity(
            identity.arch,
            identity.wave_size,
            helper,
            caps,
            helper_knobs,
            identity.dtype_a,
        )
        entry = cache.get(helper_id)
        if entry is None:
            return None, f"missing cached {helper} helper (rebuild the cache)"
        helper_path = cache.hsaco_path(helper_id)
        helper_spec = make_spec(helper, problem, helper_knobs)
        grid, block = dc.direct_launch_geometry(helper_spec)
        steps.append(
            DirectStep(
                kernel_name=entry[1]["kernel_name"],
                hsaco_path=helper_path,
                signature=conv_direct_transform_signature(
                    identity.dtype_a, runtime_channels
                ),
                grid=grid,
                block=block,
                buffers=(src, None, dst),
                scalars=scalars,
            )
        )
    weights = helpers[-1][3]
    # The compute pass is a *forward* kernel: forward ABI, transposed extents.
    steps.append(
        _conv_step(
            identity, hsaco_path, meta, spec, spec.problem, "fwd", ("dy", weights, "dx")
        )
    )
    workspaces = direct_dgrad_workspaces(problem, knobs)
    return DirectPlan(identity, tuple(steps), workspaces), "ok"


def _batch_split_plan(
    cache, identity, hsaco_path, meta, problem, arch, knobs, chunks
) -> Tuple[Optional[DirectPlan], str]:
    """One forward launch per batch chunk (see :func:`_batch_chunks`)."""
    from kernels.common.conv_direct_nongrouped import tile_w_candidates

    widths = tile_w_candidates(
        problem.Wo, make_spec(identity.algorithm, problem, knobs).atom_tile
    )
    if knobs["tile_w"] not in widths:
        return None, f"tile_w={knobs['tile_w']} does not fit Wo={problem.Wo}"
    in_img = 2 * problem.H * problem.W * problem.total_c
    out_img = 2 * problem.Ho * problem.Wo * problem.total_k
    steps = []
    n0 = 0
    for n in chunks:
        sub = replace(problem, N=n)
        spec = make_spec(identity.algorithm, sub, knobs)
        ok, why = validate_spec(identity.algorithm, spec, arch)
        if not ok:
            return None, why
        step = _conv_step(identity, hsaco_path, meta, spec, sub, "fwd", ("x", "w", "y"))
        views = ((n0 * in_img, n * in_img), None, (n0 * out_img, n * out_img))
        steps.append(replace(step, views=views))
        n0 += n
    return DirectPlan(identity, tuple(steps)), "ok"


def direct_plans(
    cache: KernelCache, problem, direction: str, arch: str
) -> Tuple[List[DirectPlan], List[Tuple[KernelIdentity, str]]]:
    """Launch plans for every cached kernel that can run ``problem``.

    Returns ``(plans, rejected)``; ``rejected`` pairs each capability-matching
    kernel that failed the per-shape re-validation with the reason, so an
    empty plan list is diagnosable.
    """
    plans: List[DirectPlan] = []
    rejected: List[Tuple[KernelIdentity, str]] = []
    for identity, hsaco_path, meta in cache.compatible(
        problem, direction=f"direct_{direction}"
    ):
        # A binary only serves the operand dtype it was built for.
        if identity.dtype_a != getattr(problem, "dtype", DIRECT_DTYPE):
            continue
        # The non-grouped family only exists for one group; offering it to a
        # grouped problem would bury the grouped kernels' reasons in noise.
        if identity.algorithm == "direct_nongrouped" and problem.groups != 1:
            continue
        # ... and the grouped LDS-halo kernel only for more than one; on a
        # dense problem it is the non-grouped kernel with extra divides.
        if identity.algorithm == "direct_grouped_halo" and problem.groups == 1:
            continue
        plan, why = plan_for(cache, identity, hsaco_path, meta, problem, arch)
        if plan is None:
            rejected.append((identity, why))
        else:
            plans.append(plan)
    return plans, rejected


def launch_values(
    step: DirectStep, ptrs: Dict[str, int], sizes: Dict[str, int]
) -> dict:
    """Kernarg values of ``step`` with its buffers bound to ``ptrs``/``sizes``."""
    a, b, d = step.buffers

    def bind(slot: int, name: str) -> Tuple[int, int]:
        view = step.views[slot] if step.views else None
        if view is None:
            return ptrs[name], sizes[name]
        return ptrs[name] + view[0], view[1]

    (a_ptr, a_bytes), (d_ptr, d_bytes) = bind(0, a), bind(2, d)
    if step.conv_args is None:
        return {
            "A": a_ptr,
            "D": d_ptr,
            "A_bytes": a_bytes,
            "D_bytes": d_bytes,
            **dict(step.scalars),
        }
    b_ptr, b_bytes = bind(1, b)
    return step.conv_args.to_launch_values(
        a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes
    )
