# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""ASM-backed coalescing orchestrator -- the automation that grounds the pure model on the compiled kernel.

The analysis layer (:mod:`.analysis.coalescing`) is GENERIC and pure: given a distribution + strides +
direction + dtype it reports the b128-IDEAL vectorization + cache-line fusion. This module is the glue that
(a) DERIVES that descriptor from a recorded kernel transaction -- direction, strides, dtype, distribution, and
the arch cache-line size -- so a human never hand-transcribes it (minimizing the error surface), and (b) reads
the ACHIEVED per-lane width straight out of ``llvm-objdump`` and HARD-GATES model-vs-ASM. A gap is fatal, never
a warning: an achieved width below the ideal is exactly the signal that surfaced the C-store b64/b128 defect,
and it means a bug in the model/viz OR in codegen -- the human must look. Reuses :func:`analyze_hsaco`
(``rocke.analysis.isa``) for the disassembly and :func:`assert_asm_backed` for the gate; it invents neither.

Scope note (honest): a GLOBAL STORE is unambiguous in a GEMM (only C stores to global), so the C-store gate is
exact. Distinguishing an A-load from a B-load in the same family from mnemonics alone is NOT possible, so the
load path reports the family's widths and flags the WORST (min) against ideal -- it cannot yet attribute a
width to a specific operand. That limitation is stated, not hidden.
"""

from __future__ import annotations

import re

from .analysis.coalescing import analyze_coalescing, assert_asm_backed

__all__ = [
    "ARCH_LINE_BYTES", "line_bytes_for", "achieved_widths", "report_for_transaction",
    "gate_report", "gate_transaction", "gate_recorded_store",
]

# Arch cache-line size (BYTES) -- the granularity a coalesced access fills. NEVER assumed: an unknown arch
# raises. Values are the L2 line size for the CDNA parts we have measured against; re-verify before adding a
# new arch (do not assume gfx90a carries over -- the user's standing rule).
ARCH_LINE_BYTES = {
    "gfx90a": 128,   # MI210/MI250 (CDNA2)
    "gfx942": 128,   # MI300 (CDNA3)
}


def line_bytes_for(arch: str) -> int:
    """Arch cache-line size in bytes -- REQUIRED, never assumed. Raises on an unknown arch so a wrong line
    size can't silently corrupt the fused/scattered verdict."""
    if arch not in ARCH_LINE_BYTES:
        raise KeyError(f"no cache-line size registered for arch {arch!r}; add it to ARCH_LINE_BYTES and "
                       f"re-verify before use (do not assume another arch's value carries over)")
    return ARCH_LINE_BYTES[arch]


# mnemonic width token -> bytes moved per lane. dword ladder (global/buffer/flat) + b-ladder (ds/LDS).
_WIDTH_BYTES = {
    "dword": 4, "dwordx2": 8, "dwordx3": 12, "dwordx4": 16,
    "b8": 1, "b16": 2, "b32": 4, "b64": 8, "b96": 12, "b128": 16,
    "short": 2, "byte": 1, "ubyte": 1, "sshort": 2, "ushort": 2,
}
_MNEMONIC_RE = re.compile(r"\b((?:global|buffer|flat)_(?:load|store)|ds_(?:read|write))_([a-z0-9]+)\b")


def _dtype_bits(name: str) -> int:
    n = name.lower()
    if "16" in n:
        return 16
    if "32" in n:
        return 32
    if "64" in n:
        return 64
    if "8" in n:
        return 8
    raise ValueError(f"cannot infer dtype bits from {name!r}")


def achieved_widths(objdump_text, *, direction, space, dtype_bits):
    """Per-lane vector widths (in ELEMENTS) the compiler ACTUALLY emitted for the ``direction``/``space`` access
    family, as a ``{width_elems: count}`` histogram read from ``llvm-objdump`` text. ``direction`` is
    ``"load"``/``"store"``; ``space`` is ``"global"`` (global/buffer/flat) or ``"lds"`` (ds). The byte width of
    each op comes from its mnemonic suffix (``dwordx4`` -> 16B, ``b64`` -> 8B); elements = bytes / dtype-bytes.
    Ambiguity is not invented: this counts the whole family, so on a shared family (e.g. A+B global loads) the
    caller must interpret the histogram, not assume a single operand."""
    ebytes = max(1, dtype_bits // 8)
    verb = "store" if direction == "store" else "load"
    hist: dict[int, int] = {}
    for m in _MNEMONIC_RE.finditer(objdump_text):
        head, suffix = m.group(1), m.group(2)
        is_lds = head.startswith("ds_")
        if (space == "lds") != is_lds:            # wrong memory space for this family
            continue
        if is_lds:
            if ("write" in head) != (verb == "store"):
                continue
        elif verb not in head:                    # global/buffer/flat: match load vs store
            continue
        wbytes = _WIDTH_BYTES.get(suffix)
        if wbytes is None:
            continue
        elems = max(1, wbytes // ebytes)
        hist[elems] = hist.get(elems, 0) + 1
    return hist


_OPERAND_DIMS = {"a": ("M", "K"), "b": ("K", "N"), "c": ("M", "N")}   # natural axes per GEMM operand


def _dims_for_transaction(txn, pipeline):
    """Best-effort axis NAMES for a recorded transaction, matched by ENCODING IDENTITY against the MMA op's
    a/b/c operand encodings (A=(M,K), B=(K,N), C=(M,N)). Returns ``None`` when it can't attribute the
    transaction to an operand -- the caller must then pass ``dims`` explicitly (never guessed here)."""
    enc = txn.encoding
    for op in pipeline.ops:
        if op.kind != "mma":
            continue
        for slot, ref in (("a", op.a_enc), ("b", op.b_enc), ("c", op.c_enc)):
            if enc is ref or enc == ref:
                return _OPERAND_DIMS[slot]
    return None


def report_for_transaction(txn, *, arch, dims=None, pipeline=None, n_lanes=64):
    """Build a :class:`CoalescingReport` from a recorded ``PipelineTransaction`` -- the descriptor is DERIVED,
    not hand-typed: ``direction`` = the verb (load/store), ``strides`` = the tensor's element strides,
    ``dtype_bits`` from the dtype, the distribution from the transaction's encoding (via ``RegisterMapper``),
    and ``line_bytes`` from ``arch``. ``dims`` (axis names) are matched from the MMA operand encodings when a
    ``pipeline`` is given, else must be supplied -- never guessed."""
    from .register_mapper import RegisterMapper

    if txn.strides is None:
        raise ValueError(f"transaction {txn.kind}/{txn.space_name} carries no strides (a register-only fill?)")
    if dims is None:
        dims = _dims_for_transaction(txn, pipeline) if pipeline is not None else None
    if dims is None:
        raise ValueError("axis names (dims) could not be attributed from the MMA operands; pass dims= "
                         "explicitly -- they are never guessed")
    strides = tuple(int(s) for s in txn.strides)
    rm = RegisterMapper(txn.encoding)
    fwd = {(lane, reg): rm.matrix_coordinates(lane, reg)
           for lane in range(n_lanes) for reg in range(rm.num_vector_items)}
    return analyze_coalescing(fwd, dims, strides, _dtype_bits(txn.dtype_name),
                              direction=txn.kind, line_bytes=line_bytes_for(arch))


def gate_report(report, objdump_text, *, space):
    """HARD-GATE a coalescing ``report`` against the compiled ASM. ``space`` is the memory space the modelled
    access targets (``"global"``/``"lds"``) -- needed to pick the ASM instruction family (``report.direction``
    and ``report.dtype_bits`` supply the rest). Returns ``(report, achieved_hist, note)``; RAISES (via
    :func:`assert_asm_backed`) when the compiler's WORST (min) width in the family is below the b128-ideal the
    layout supports. An EMPTY family is itself a discrepancy -- the render claims an access the kernel never
    emits -- and is raised, never passed. This is the pure gate; :func:`gate_transaction` derives the report
    first, then calls this."""
    hist = achieved_widths(objdump_text, direction=report.direction, space=space,
                           dtype_bits=report.dtype_bits)
    if not hist:
        raise AssertionError(f"ASM shows NO {report.direction}/{space} access for a modelled transaction "
                             f"-- the render claims an access the kernel never emits; investigate the "
                             f"model/recording OR the codegen")
    achieved = min(hist)                                  # worst per-lane width the compiler settled for
    _ok, note = assert_asm_backed(report, achieved)       # raises on under-/over-shoot
    return report, hist, note


def gate_transaction(txn, objdump_text, *, arch, dims=None, pipeline=None, n_lanes=64):
    """DERIVE the coalescing report for a recorded transaction, then HARD-GATE it against ``objdump_text`` via
    :func:`gate_report`. Returns ``(report, achieved_hist, note)`` or RAISES on a model-vs-ASM gap."""
    report = report_for_transaction(txn, arch=arch, dims=dims, pipeline=pipeline, n_lanes=n_lanes)
    return gate_report(report, objdump_text, space=txn.space)


def gate_recorded_store(pipeline, kernel, *, arch, dims=("M", "N"), objdump="llvm-objdump"):
    """End-to-end C-store gate: compile ``kernel`` -> HSACO, disassemble, and gate the recorded GLOBAL STORE
    (unambiguous in a GEMM: only C stores to global) against the model. Returns ``(report, achieved_hist,
    note)`` or RAISES on a model-vs-ASM gap. Requires the compile toolchain (comgr + llvm-objdump); it is the
    thin machine-specific seam -- everything above it is pure and unit-tested offline."""
    import tempfile
    from pathlib import Path

    from rocke.analysis.isa import analyze_hsaco
    from rocke.helpers.compile import compile_kernel

    stores = [t for t in pipeline.transactions if t.kind == "store" and t.space == "global"]
    if not stores:
        raise ValueError("no recorded GLOBAL store to gate (is this a GEMM epilogue?)")
    txn = stores[0]
    artifact = compile_kernel(kernel, arch=arch)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / f"{artifact.kernel_name}.hsaco"
        p.write_bytes(artifact.hsaco)
        analysis = analyze_hsaco(p, objdump=objdump, keep_text=True)
    return gate_transaction(txn, analysis.objdump_text, arch=arch, dims=dims, pipeline=pipeline)
