################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""End-to-end real-kernel reproducer for the cross-subiter Pack -> MFMA artifact.

Companion to:
  - bead `rocm-libraries-bwfr` and the investigation memo
    `Tensile/Components/CROSS_SUBITER_ALU_FP_INVESTIGATION.md` (the production
    failure list at memo section 3.2 enumerates the 10 unit tests that fail
    when the section 7.3 carve-out is neutralized);
  - bead `rocm-libraries-bwfr-companion` and the visual minimal-repro memo
    `Tensile/Components/CROSS_SUBITER_ALU_FP_MINIMAL_REPRO.md`;
  - the synthetic 3-instruction test
    `Tensile/Tests/unit/test_cross_subiter_pack_artifact.py` (commit 7c1295fbbda).

This script drives the SAME real production kernel that
`test_ScheduleCapture.py::TestRealKernelCapture::test_tf32_4x4_tn_capture_shape`
builds. That kernel is a 128x128x32 TF32 4x4 emulation kernel (TN layout,
DepthU=32, MI=16x16x32 with a 4x4 wave tile per group, PGR=2 PLR=1, DTL=1)
with `F32XdlMathOp='X'` enabling F32 emulation through bf16 packs. The
PackA3[N] / PackB3[N] cross-subiter artifact (768 edges per the bwfr memo
section 3.2) is reproduced at full production scale here.

What this script does:

1. Constructs the canonical kernel config (the exact dict the 10 production
   tests use; see ``CANONICAL_KERNEL_CONFIG`` below).
2. Builds the kernel TWICE through `KernelWriterAssembly._getKernelSource`:
   a. With `UseCustomMainLoopSchedule=1` (production CMS build): the writer
      auto-activates `_captureDefaultSchedule` (KernelWriter.py:4704), which
      stashes a SHADOW default-side `FourPartCapture` AND the real CMS-side
      `FourPartCapture`, then runs `compare_graphs` as the production gate
      (KernelWriter.py:5307). The emitted `.s` text is written to
      `repro_cross_subiter_cms.s`.
   b. With `UseCustomMainLoopSchedule=0` (default codegen path): the writer
      emits the default scheduler's assembly with no CMS macro replacement.
      The emitted `.s` text is written to `repro_cross_subiter_default.s`.
3. Re-runs `compare_graphs` on the captured pair WITH the section-7.3 carve-out
   monkey-patched off, so the artifact edges surface as
   `OrderInvertedFailure`s instead of being suppressed (mirrors the probe
   technique from `Tests/scratch/run_with_carveout_off.py` referenced in
   bwfr memo section 3.2).
4. Prints edge counts, missing-/extra-edge previews, the scratch vgprs
   participating in the artifact edges, and a summary line.

Run as:

    PYTHONPATH=projects/hipblaslt/tensilelite \\
        python projects/hipblaslt/tensilelite/Tensile/Components/repro_cross_subiter_artifact.py

The script depends on the same `isa_infrastructure` fixture machinery used by
the `TestRealKernelCapture` integration tests (probes the system compiler for
gfx950 ISA caps, ~3.8s on first call). All assembly emit path code is the
production path — no synthetic shortcuts.
"""

import os
import shutil
import sys
from collections import Counter


# =============================================================================
# Canonical kernel config — the dict the 10 production tests pass to
# `_make_solution(...)`. Sourced verbatim from
# `Tensile/Tests/unit/test_ScheduleCapture.py::TestRealKernelCapture::
# test_tf32_4x4_tn_capture_shape` (the first test in bwfr memo section 3.2's
# list, and the most-cited reproducer).
#
# Why this kernel:
#   - F32XdlMathOp='X' triggers the bf16-pack-based TF32 emulation path that
#     allocates the v133-style scratch vgpr range bwfr documents.
#   - 4x4 wave tile + DepthU=32 means inner-unroll subiters drive the
#     PackA3[N] / PackB3[N] cross-subiter pattern (4 subiters per inner unroll
#     in the production trace).
#   - PGR=2 PLR=1 + DTL=1 + TN: the registered CMS schedule
#     `_get_schedule_128x128x32_TF32` is the canonical CMS schedule for this
#     shape (lives under `Tensile/Components/CustomSchedule/gfx950/`).
#   - All 10 tests in the bwfr list use this exact dict, so this is the
#     single canonical kernel; no judgment call needed between alternatives.
# =============================================================================
CANONICAL_KERNEL_CONFIG = {
    'ProblemType': {
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    'MatrixInstruction': [16, 16, 32, 1, 1, 4, 4, 2, 2],
    'DepthU': 32, 'PrefetchGlobalRead': 2, 'PrefetchLocalRead': 1,
    'DirectToLds': 1, 'TransposeLDS': 1, 'LocalReadVectorWidth': 4,
    'GlobalReadVectorWidthA': 4, 'GlobalReadVectorWidthB': 4,
    'UseCustomMainLoopSchedule': 1, 'ExpandPointerSwap': 0,
    'SourceSwap': 1, 'StreamK': 0,
}


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_S_PATH = os.path.join(HERE, "repro_cross_subiter_default.s")
CMS_S_PATH = os.path.join(HERE, "repro_cross_subiter_cms.s")


def _make_isa_infrastructure():
    """Mirror of the `isa_infrastructure` fixture from
    `Tensile/Tests/unit/conftest.py`. Probes gfx950 ISA caps via the system
    compiler and constructs an Assembler. Returns (isaInfoMap, asm).
    """
    from Tensile.Common import IsaVersion
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Component import Assembler

    compiler = shutil.which('amdclang++') or shutil.which('clang++')
    assembler_bin = shutil.which('amdclang') or shutil.which('clang')
    if not compiler:
        raise RuntimeError("No C++ compiler (amdclang++/clang++) found for "
                           "ISA capability probing.")
    if not assembler_bin:
        raise RuntimeError("No assembler binary (amdclang/clang) found.")

    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)
    asm = Assembler(assembler_bin, 'V5')
    return isaInfoMap, asm


def _build_cms_kernel(isaInfoMap, asm):
    """Build the canonical kernel with `UseCustomMainLoopSchedule=1`. The
    writer auto-activates `_captureDefaultSchedule` (KernelWriter.py:4704),
    so both the default-side shadow capture and the CMS-side real capture
    are populated. Returns (writer, kernel_source_string).
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(HERE), 'Tests', 'unit'))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    config = dict(CANONICAL_KERNEL_CONFIG)
    config['UseCustomMainLoopSchedule'] = 1
    solution = _make_solution(config, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    source = writer._getKernelSource(solution)
    return writer, source


def _build_default_kernel(isaInfoMap, asm):
    """Build the canonical kernel with `UseCustomMainLoopSchedule=0`.

    The default codegen path (no CMS macro substitution) emits the SIA3
    scheduler's assembly directly. Returns the rocisa-emitted `.s` string.
    """
    sys.path.insert(0, os.path.join(
        os.path.dirname(HERE), 'Tests', 'unit'))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig

    config = dict(CANONICAL_KERNEL_CONFIG)
    config['UseCustomMainLoopSchedule'] = 0
    solution = _make_solution(config, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    source = writer._getKernelSource(solution)
    return writer, source


def _resource_label(res, name_to_idx=None):
    """Render a resource (RegisterContainer or MemoryRegion) for display.

    Production Pack/MFMA operands are emitted SYMBOLICALLY (e.g.
    `vgprValuPackA_X0_I0+3`) so `regIdx` is `-1`. The body's `name_to_idx`
    table (populated by the writer's RegSet directives, harvested at
    KernelWriter.py:5255) resolves the bare symbolic name to a numeric base
    — that's the same path build_dataflow_graph Phase 2 uses to collapse
    symbolic + numeric refs to the same physical register byte-key. Reproduce
    the same resolution here so the repro can show the user the actual
    `v133`-style numeric index participating in each artifact edge.
    """
    if not hasattr(res, "regType"):
        return repr(res)
    rt = getattr(res, "regType", None)
    ridx = getattr(res, "regIdx", -1)
    rnum = getattr(res, "regNum", 1) or 1
    if ridx is not None and ridx >= 0:
        return f"{rt}{ridx}" + (f":{ridx + rnum - 1}" if rnum > 1 else "")
    rname_obj = getattr(res, "regName", None)
    if rname_obj is None:
        return f"{rt}<?>"
    name = getattr(rname_obj, "name", "?")
    base_off = (rname_obj.getTotalOffsets()
                if hasattr(rname_obj, "getTotalOffsets") else 0)
    if name_to_idx:
        bare = name[4:] if name.startswith(("vgpr", "sgpr")) else name
        resolved = name_to_idx.get(bare)
        if resolved is not None:
            num = resolved + base_off
            return (f"{rt}{num}" + (f":{num + rnum - 1}" if rnum > 1 else "")
                    + f"  [from sym {name}+{base_off}]")
    return f"{rt}<sym:{name}+{base_off}>"


def _resolve_vgpr_index(res, name_to_idx=None):
    """Return the numeric vgpr index of `res` if resolvable, else None."""
    if getattr(res, "regType", None) != "v":
        return None
    ridx = getattr(res, "regIdx", -1)
    if ridx is not None and ridx >= 0:
        return ridx
    rname_obj = getattr(res, "regName", None)
    if rname_obj is None or not name_to_idx:
        return None
    name = getattr(rname_obj, "name", "")
    base_off = (rname_obj.getTotalOffsets()
                if hasattr(rname_obj, "getTotalOffsets") else 0)
    bare = name[4:] if name.startswith(("vgpr", "sgpr")) else name
    resolved = name_to_idx.get(bare)
    if resolved is None:
        return None
    return resolved + base_off


def _name_to_idx_for(graph, body_label):
    """Look up the body-local name_to_idx table from the dataflow graph,
    so symbolic operands can be resolved to numeric vgpr indices for
    display.
    """
    cap = graph.captures.get(body_label)
    if cap is None:
        return None
    return getattr(cap, "name_to_idx", None) or None


def _format_node(node, position_label="@"):
    """Render a GraphNode as `Category[loop_index] @ pos=<pos>` for diff display."""
    cat = node.category or "?"
    body = getattr(node, "body_label", "?")
    pos = node.position
    return f"{cat} ({body}) {position_label}{pos}"


def _format_edge(edge, graph=None):
    """Render a DataflowEdge as `Producer --[resource]--> Consumer` line."""
    prod = _format_node(edge.producer, position_label="prod_pos=")
    cons = _format_node(edge.consumer, position_label="cons_pos=")
    n2i = (_name_to_idx_for(graph, edge.producer.body_label)
           if graph is not None else None)
    res_str = _resource_label(edge.resource, name_to_idx=n2i)
    return f"{prod}  --[{res_str}, {edge.edge_kind}]-->  {cons}"


def _scratch_vgpr_indices(edges, graph=None):
    """Return a Counter of (resolved) vgpr indices that appear as edge.resource."""
    counts = Counter()
    for e in edges:
        n2i = (_name_to_idx_for(graph, e.producer.body_label)
               if graph is not None else None)
        idx = _resolve_vgpr_index(e.resource, name_to_idx=n2i)
        if idx is not None:
            counts[idx] += 1
    return counts


def _looks_like_cross_subiter_artifact(edge):
    """Heuristic: an edge from a `PackA*` or `PackB*` producer to an `MFMA`
    consumer with producer category subiter index >= 1 (i.e. PackA1/2/3 or
    PackB1/2/3) is a cross-subiter Pack -> MFMA artifact candidate.
    """
    pcat = (edge.producer.category or "")
    ccat = (edge.consumer.category or "")
    if not (pcat.startswith("PackA") or pcat.startswith("PackB")):
        return False
    if ccat != "MFMA":
        return False
    suffix = pcat[len("PackA"):] if pcat.startswith("PackA") else pcat[len("PackB"):]
    if not suffix:
        return False
    digits = ""
    for ch in suffix:
        if ch.isdigit():
            digits += ch
        else:
            break
    if not digits:
        return False
    return int(digits) >= 1


def main():
    print("=" * 78)
    print("Cross-subiter Pack -> MFMA artifact: real-kernel end-to-end repro")
    print("=" * 78)
    print()
    print("Canonical kernel config (verbatim from "
          "test_ScheduleCapture.py::TestRealKernelCapture::"
          "test_tf32_4x4_tn_capture_shape):")
    for k, v in CANONICAL_KERNEL_CONFIG.items():
        print(f"  {k!r}: {v!r},")
    print()

    print("Probing ISA capabilities (gfx950)...")
    isaInfoMap, asm = _make_isa_infrastructure()
    print("  done.")
    print()

    # ---- Build the CMS kernel: emits CMS `.s` AND populates both captures.
    print("Building canonical kernel with UseCustomMainLoopSchedule=1 ...")
    cms_writer, cms_source = _build_cms_kernel(isaInfoMap, asm)
    print(f"  CMS build emitted {len(cms_source):,} characters of assembly.")
    with open(CMS_S_PATH, "w") as fh:
        fh.write(cms_source)
    print(f"  Wrote {CMS_S_PATH}")
    print()

    # The shadow default-side capture and real CMS-side capture from the
    # SAME CMS kernel build are what `compare_graphs` runs over in
    # production (KernelWriter.py:5305-5307). Stash them now before any
    # subsequent kernel build resets `_capture_context`.
    default_cap = cms_writer._last_default_capture
    cms_cap = cms_writer._last_cms_capture
    if default_cap is None or cms_cap is None:
        raise RuntimeError(
            "CMS build did not populate both captures: "
            f"default_cap={default_cap is not None}, cms_cap={cms_cap is not None}. "
            "Auto-activation in KernelWriter.py:4704 should have set "
            "_captureDefaultSchedule for this CMS kernel."
        )
    print(f"  Captures populated: default_cap.num_mfma={default_cap.num_mfma}, "
          f"cms_cap.num_mfma={cms_cap.num_mfma}")
    print()

    # ---- Build the default-codegen kernel: emits default `.s`.
    print("Building canonical kernel with UseCustomMainLoopSchedule=0 ...")
    _default_writer, default_source = _build_default_kernel(isaInfoMap, asm)
    print(f"  Default build emitted {len(default_source):,} characters of "
          f"assembly.")
    with open(DEFAULT_S_PATH, "w") as fh:
        fh.write(default_source)
    print(f"  Wrote {DEFAULT_S_PATH}")
    print()

    # ---- Run compare_graphs against the captured pair.
    from Tensile.Components.CMSValidator import (
        GraphNode, build_dataflow_graph, compare_graphs,
    )

    print("Building dataflow graphs...")
    ref_graph = build_dataflow_graph(default_cap)
    subj_graph = build_dataflow_graph(cms_cap)
    print(f"  default-side (reference) graph: {len(ref_graph.edges):,} edges, "
          f"{len(ref_graph.nodes):,} nodes")
    print(f"  CMS-side (subject)    graph: {len(subj_graph.edges):,} edges, "
          f"{len(subj_graph.nodes):,} nodes")
    print()

    # In production the section-7.3 carve-out absorbs the artifact edges,
    # so `compare_graphs` returns []. To SURFACE the artifact edges we
    # neutralize the carve-out's `GraphNode.subiter()` predicate (post-nn0
    # the per-byte resolver-side predicate is a method on GraphNode; both
    # mirrored carve-out sites in `diagnose_missing_edge` and
    # `_alu_cross_subiter_passthrough` consult it). With `subiter()`
    # forced to return 0, the `producer.subiter(nmps) != consumer.subiter(nmps)`
    # gate fails, the carve-out branch is skipped, and the
    # OrderInvertedFailure path fires.
    print("Running compare_graphs WITH carve-out engaged (production "
          "behavior) ...")
    failures_with_carveout = compare_graphs(ref_graph, subj_graph)
    print(f"  -> {len(failures_with_carveout)} failures (production behavior; "
          f"carve-out absorbs artifact edges).")
    print()

    print("Running compare_graphs with section-7.3 carve-out NEUTRALIZED ...")
    # Pre-flight guard #1: confirm the patch target exists. If a future
    # refactor renames or relocates `GraphNode.subiter` (it migrated from
    # a free `_node_subiter(node, nmps)` function in pre-nn0 vlt to a
    # method on GraphNode post-nn0; the OLD patch silently no-op'd in
    # this script for ~1 week before the carveout-verify pinning test
    # caught it), this assertion fires loudly instead of letting the
    # script silently report 0 failures.
    if not hasattr(GraphNode, "subiter"):
        raise RuntimeError(
            "Cannot neutralize section-7.3 carve-out: GraphNode.subiter "
            "not found. The carve-out's predicate has likely been "
            "refactored. Update the patch target to whatever method the "
            "carve-out now consults (grep for 'cross_subiter_alu_artifact' "
            "and the diagnose_missing_edge early-return for current sites)."
        )
    original_subiter = GraphNode.subiter
    GraphNode.subiter = lambda self, nmps: 0
    try:
        # Pre-flight guard #2: confirm the patch took effect on actual
        # graph nodes. Catches scenarios where the attribute exists but
        # the carve-out consults a different surface (e.g. a free
        # function shadowed by the method, or a per-instance override).
        # `subj_graph.nodes` is a dict keyed by node identity, not a list.
        nodes_iter = iter(subj_graph.nodes.values())
        probe_node = next(nodes_iter, None)
        if probe_node is not None:
            probe_result = probe_node.subiter(1)
            if probe_result != 0:
                raise RuntimeError(
                    f"GraphNode.subiter patch did not take effect on a "
                    f"real node: probe returned {probe_result!r} instead "
                    f"of 0. The carve-out neutralization is silently "
                    f"broken — fix the patch target before relying on "
                    f"this script's output."
                )
        failures_neutralized = compare_graphs(ref_graph, subj_graph)
    finally:
        GraphNode.subiter = original_subiter
    print(f"  -> {len(failures_neutralized)} failures surfaced when the "
          f"carve-out is off.")
    # Post-flight guard #3: the canonical TF32 4x4 kernel is expected to
    # produce ~768 OrderInvertedFailure instances when the carve-out is
    # off (per bwfr memo section 3.2 and the carveout-verify pinning
    # test at test_cross_subiter_alu_carveout_real_kernel.py). If the
    # neutralized count is suspiciously low (especially 0), the patch
    # didn't propagate into the actual carve-out code paths — fail
    # loudly rather than print a misleading "0 failures" line.
    if len(failures_neutralized) < 100:
        raise RuntimeError(
            f"Expected ~768 OrderInvertedFailure instances when the "
            f"section-7.3 carve-out is neutralized; got "
            f"{len(failures_neutralized)}. Either (a) the canonical "
            f"kernel under test has changed shape, or (b) the patch "
            f"didn't reach the actual carve-out code path. See "
            f"test_cross_subiter_alu_carveout_real_kernel.py for the "
            f"pinned-down ground truth."
        )
    print()

    # ---- Edge-set diff using the same edge_keys() machinery the validator
    # ---- uses, so the missing/extra edge counts here match what the
    # ---- production gating-assertion path observes.
    ref_keys = ref_graph.edge_keys()
    subj_keys = subj_graph.edge_keys()
    missing_keys = ref_keys - subj_keys
    extra_keys = subj_keys - ref_keys

    print("Edge-set diff (ref = default-side, subj = CMS-side):")
    print(f"  total ref edges:   {len(ref_keys):,}")
    print(f"  total subj edges:  {len(subj_keys):,}")
    print(f"  ref - subj (missing in subj): {len(missing_keys):,}")
    print(f"  subj - ref (extra in subj):   {len(extra_keys):,}")
    print()

    # Map missing/extra edge keys back to DataflowEdge objects for display.
    # Per EMISSION_ORDINAL_DESIGN.md §4.2, role positions come from
    # `_role(node)` (rocisa-derived), matching `DataflowGraph.edge_keys`.
    from Tensile.Components.CMSValidator import _role
    ref_edges_by_key = {
        (_role(e.producer), e.producer.position, e.src_operand_slot,
         _role(e.consumer), e.consumer.position, e.sink_operand_slot,
         e.edge_kind, e.intra_operand_byte_offset): e
        for e in ref_graph.edges
    }
    subj_edges_by_key = {
        (_role(e.producer), e.producer.position, e.src_operand_slot,
         _role(e.consumer), e.consumer.position, e.sink_operand_slot,
         e.edge_kind, e.intra_operand_byte_offset): e
        for e in subj_graph.edges
    }
    missing_edges = [ref_edges_by_key[k] for k in missing_keys
                     if k in ref_edges_by_key]
    extra_edges = [subj_edges_by_key[k] for k in extra_keys
                   if k in subj_edges_by_key]

    print(f"First 20 missing edges (in default-side but not CMS-side):")
    for e in missing_edges[:20]:
        print(f"  {_format_edge(e, graph=ref_graph)}")
    if len(missing_edges) > 20:
        print(f"  ... and {len(missing_edges) - 20:,} more.")
    print()

    print(f"First 20 extra edges (in CMS-side but not default-side):")
    for e in extra_edges[:20]:
        print(f"  {_format_edge(e, graph=subj_graph)}")
    if len(extra_edges) > 20:
        print(f"  ... and {len(extra_edges) - 20:,} more.")
    print()

    # ---- Identify the scratch vgprs participating in the artifact edges.
    artifact_missing_edges = [e for e in missing_edges
                              if _looks_like_cross_subiter_artifact(e)]
    artifact_vgpr_counts = _scratch_vgpr_indices(artifact_missing_edges,
                                                 graph=ref_graph)
    print(f"Scratch vgprs participating in cross-subiter Pack -> MFMA "
          f"artifact edges:")
    if artifact_vgpr_counts:
        # Sorted by frequency so the most-used scratch vgprs appear first.
        for vidx, count in artifact_vgpr_counts.most_common():
            print(f"  v{vidx}: {count} artifact edge(s)")
    else:
        print("  (no v-typed edges identified among artifact missing edges)")
    print()

    # ---- Summary line.
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{len(artifact_missing_edges):,} cross-subiter PackA[N]/PackB[N] "
          f"artifact edges found in default-side graph that aren't in "
          f"CMS-side graph.")
    print()
    print(f"Total missing edges (any kind): {len(missing_keys):,}")
    print(f"OrderInvertedFailure count when carve-out off: "
          f"{sum(1 for f in failures_neutralized if type(f).__name__ == 'OrderInvertedFailure')}")
    print(f"Production carve-out absorbs to: {len(failures_with_carveout)} "
          f"failure(s).")
    print()
    print(f"Default-side .s : {DEFAULT_S_PATH} "
          f"({os.path.getsize(DEFAULT_S_PATH):,} bytes)")
    print(f"CMS-side .s     : {CMS_S_PATH} "
          f"({os.path.getsize(CMS_S_PATH):,} bytes)")


if __name__ == "__main__":
    main()
