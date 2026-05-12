"""One-shot diagnostic: dump assembly windows for the cross-subiter ALU
carve-out artifact failures surfaced by
`test_real_kernel_neutralized_carveout_surfaces_768_pack3_mfma_failures`.

Dumps ONLY the FIRST Pack3 -> MFMA artifact failure (whichever shape
appears first in `compare_graphs`'s output). For that one failure, prints:
  - producer
  - all instructions between producer and consumer in the SUBJECT (CMS)
    stream of the same body, sorted by stream position
  - consumer
  - the same window (between the matching producer and consumer) from the
    REFERENCE (default) stream, side-by-side

Re-uses the existing test fixtures (`real_kernel_graphs`, conftest's
`isa_infrastructure`) so the kernel build path is identical to the test.

Side-channel mechanism: monkey-patch `cms_node_label` to stash the actual
`GraphNode` reference on the resulting `FailureNodeLabel` as `_dump_node`,
so we can recover both producer and consumer GraphNodes after
`compare_graphs` returns.
"""

import os
import sys

import pytest


# Reuse the canonical config + the module-scoped real_kernel_graphs fixture.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_cross_subiter_alu_carveout_real_kernel import (  # noqa: E402
    real_kernel_graphs,
    CANONICAL_KERNEL_CONFIG,
)


WINDOW_CAP = 50
HEAD_TAIL = 25


def _render_asm(node) -> str:
    """Render a GraphNode's underlying instruction as a single ASM-like line.

    `WrappedInstruction.canonical_str` strips trailing comments and
    collapses whitespace.
    """
    from Tensile.Components.ScheduleCapture import WrappedInstruction
    inst = node.rocisa_inst
    return WrappedInstruction.canonical_str(inst)


def _slice_window(graph, body_label: str, lo_pos, hi_pos):
    """Return the list of GraphNodes in `graph` belonging to `body_label`
    whose position is STRICTLY between `lo_pos` and `hi_pos`, sorted by
    position. SchedulePosition's `<` is lex over `(loop_index, stream_index)`.
    """
    nodes_in_body = [
        n for n in graph.nodes.values() if n.body_label == body_label
    ]
    nodes_in_body.sort(key=lambda n: n.position)
    return [n for n in nodes_in_body if lo_pos < n.position < hi_pos]


def _print_window(label: str, producer, consumer, between, prefix: str = ""):
    """Pretty-print a producer -> ... -> consumer window.

    Renders node labels via `render_node_label` (CMSValidator) so PRE_LOOP
    and POST_LOOP entries that share `mfma_index=-1` are visually
    distinguishable in the dump output (rocm-libraries-aprv).
    """
    from Tensile.Components.CMSValidator import render_node_label
    print(f"\n{prefix}--- {label} window (body={producer.body_label}) ---")
    print(f"{prefix}FIRST  pos={producer.position}  cat={producer.category}  name={render_node_label(producer)}")
    print(f"{prefix}            asm: {_render_asm(producer)}")
    n_between = len(between)
    if n_between > WINDOW_CAP:
        head = list(enumerate(between[:HEAD_TAIL]))
        omitted = n_between - 2 * HEAD_TAIL
        tail_start = n_between - HEAD_TAIL
        tail = [(tail_start + j, b) for j, b in enumerate(between[-HEAD_TAIL:])]
        groups = [head, [("...", omitted)], tail]
    else:
        groups = [list(enumerate(between))]
    for grp in groups:
        for entry in grp:
            if entry[0] == "...":
                print(f"{prefix}    ... {entry[1]} omitted ...")
                continue
            idx, n = entry
            print(
                f"{prefix}  [{idx:3d}] pos={n.position} cat={n.category:14s}"
                f" body={n.body_label} name={render_node_label(n)!r}"
            )
            print(f"{prefix}         asm: {_render_asm(n)}")
    print(f"{prefix}LAST   pos={consumer.position}  cat={consumer.category}  name={render_node_label(consumer)}")
    print(f"{prefix}            asm: {_render_asm(consumer)}")


def _dump_full_stream(graph, out_path):
    """Write every node in `graph` as one ASM line per node, grouped by
    body, sorted by stream position within each body. Header comments
    annotate each body.

    Sourced from each LoopBodyCapture's `_graph_nodes` sidecar (attached
    by build_dataflow_graph at CMSValidator.py:1792), which retains the
    SCHEDULER-CONTROL nodes (SWait/SBarrier/SNop/SSetPrior) that
    `graph.nodes` filters out via _NO_DATAFLOW_IDENTITY_CATEGORIES. Using
    the sidecar here gives the true emission stream without changing the
    validator's data-flow graph contents.

    `_graph_nodes` is already in stream order; we sort defensively anyway.

    Per-node `name=` rendering uses `render_node_label` so PRE_LOOP and
    POST_LOOP entries with the shared `mfma_index=-1` slot are
    distinguishable as `@PRE-1.X` vs `@POST-1.X` instead of the colliding
    `@-1.X` (rocm-libraries-aprv).
    """
    from Tensile.Components.CMSValidator import render_node_label
    captures = graph.captures or {}
    body_order = []
    by_body = {}
    for body_label, cap in captures.items():
        nodes = list(getattr(cap, "_graph_nodes", []) or [])
        if not nodes:
            continue
        nodes.sort(key=lambda n: n.position)
        by_body[body_label] = nodes
        body_order.append(body_label)
    body_order.sort(key=lambda b: by_body[b][0].position)

    lines = []
    for body in body_order:
        nodes = by_body[body]
        n_total = len(nodes)
        n_dataflow = sum(1 for n in nodes if n.identity in graph.nodes)
        n_control = n_total - n_dataflow
        lines.append(
            f"// ===== BODY {body} ({n_total} nodes total: "
            f"{n_dataflow} data-flow + {n_control} scheduler-control) ====="
        )
        for n in nodes:
            in_graph = n.identity in graph.nodes
            graph_tag = "" if in_graph else "  [NOT-IN-GRAPH]"
            lines.append(
                f"// pos={n.position} cat={n.category} name={render_node_label(n)}{graph_tag}\n"
                f"{_render_asm(n)}"
            )
        lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def test_dump_carveout_assembly_windows(isa_infrastructure, monkeypatch):
    """Not a real assertion test — runs only to dump representative windows
    via `print`. Pytest still surfaces the prints when invoked with -s.

    Builds the canonical TF32 4x4 TN kernel from scratch (does NOT reuse
    `real_kernel_graphs` fixture) so we can capture the full assembly
    text returned by `_getKernelSource` and write it alongside the per-
    body graph dumps.

    Writes four files in the same folder as this script:
      - `kernel_cms.s`     : raw Tensile assembly with `UseCustomMainLoopSchedule=1`
                             — the CMS path.  This is the kernel the assembler
                             actually consumes in production for this tile.
      - `kernel_default.s` : raw Tensile assembly with `UseCustomMainLoopSchedule=0`
                             — bypasses the CMS dispatch entirely so the per-tile
                             schedule registration (and its `kernel["UsePLRPack"] = True`
                             mutation, `kernel["UseMFMAF32XEmulation"] = True`,
                             schedule slot directives, etc.) never fires.  This is
                             closer to a "true Tensilelite default" kernel — see
                             `rocm-libraries-2lzd` for why the shadow-default
                             capture inside the CMS build does NOT give you this
                             (the shadow shares the CMS-mutated `kernel` dict).
      - `cms.s`            : per-body GraphNode-rendered stream for the CMS
                             capture (subj graph), including scheduler-control
                             nodes filtered out of `graph.nodes`.
      - `default.s`        : same shape for the shadow default capture.

    The `cms.s`/`default.s` graph dumps come from the SAME single CMS build —
    that's what the validator operates on.  `kernel_cms.s`/`kernel_default.s`
    are two SEPARATE Tensile builds: the first matches the CMS build (and so
    matches what `cms.s`/`default.s` are derived from); the second is an
    independent build with CMS turned off, for side-by-side comparison
    against a true non-CMS Tensile emit.
    """
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CMSValidator import (
        GraphNode, compare_graphs, build_dataflow_graph, render_node_label,
    )
    import Tensile.Components.CMSValidator as cms_mod

    _isa, isaInfoMap, asm = isa_infrastructure

    here = os.path.dirname(os.path.abspath(__file__))
    kernel_cms_path = os.path.join(here, "kernel_cms.s")
    kernel_default_path = os.path.join(here, "kernel_default.s")
    cms_path = os.path.join(here, "cms.s")
    default_path = os.path.join(here, "default.s")

    # --- Build #1: CMS path. Capture asm + both shadow/CMS captures. ---
    cms_config = dict(CANONICAL_KERNEL_CONFIG)
    cms_config['UseCustomMainLoopSchedule'] = 1
    cms_solution = _make_solution(cms_config, asm, isaInfoMap)
    cms_writer = KernelWriterAssembly(asm, DebugConfig())
    cms_asm_text = cms_writer._getKernelSource(cms_solution)
    default_cap = cms_writer._last_default_capture
    cms_cap = cms_writer._last_cms_capture
    assert default_cap is not None and cms_cap is not None, (
        "CMS-path kernel build did not populate default/cms captures."
    )
    ref_graph = build_dataflow_graph(default_cap)
    subj_graph = build_dataflow_graph(cms_cap)

    with open(kernel_cms_path, "w") as f:
        f.write(cms_asm_text)
    print(f"\nWrote raw CMS Tensile asm     -> {kernel_cms_path} ({len(cms_asm_text)} chars)")

    # --- Build #2: default path. UseCustomMainLoopSchedule=0 bypasses the
    # CMS dispatch entirely (no per-tile schedule registration fires, no
    # UsePLRPack/UseMFMAF32XEmulation mutation), so we get a true non-CMS
    # Tensile emit for the same problem shape.  We do NOT need its
    # captures (default-only build won't auto-activate the shadow path
    # anyway); we only need the asm text.
    default_config = dict(CANONICAL_KERNEL_CONFIG)
    default_config['UseCustomMainLoopSchedule'] = 0
    default_solution = _make_solution(default_config, asm, isaInfoMap)
    default_writer = KernelWriterAssembly(asm, DebugConfig())
    default_asm_text = default_writer._getKernelSource(default_solution)

    with open(kernel_default_path, "w") as f:
        f.write(default_asm_text)
    print(f"Wrote raw default Tensile asm -> {kernel_default_path} ({len(default_asm_text)} chars)")

    # Per-body GraphNode-rendered streams (data-flow + scheduler-control
    # via `cap._graph_nodes` sidecar). Done BEFORE neutralizing the
    # carve-out so the dumps reflect production graphs.
    _dump_full_stream(subj_graph, cms_path)
    _dump_full_stream(ref_graph, default_path)
    print(f"Wrote full CMS stream     -> {cms_path}")
    print(f"Wrote full default stream -> {default_path}")

    # Neutralize the carve-out by collapsing every node into subiter 0.
    monkeypatch.setattr(GraphNode, "subiter", lambda self, nmps: 0)

    # Side-channel: monkey-patch cms_node_label to stash the GraphNode on
    # each FailureNodeLabel so we can recover them after compare_graphs.
    original_cms_node_label = cms_mod.cms_node_label

    def labeled(node, body_capture):
        lab = original_cms_node_label(node, body_capture)
        # FailureNodeLabel is a regular @dataclass (no frozen); direct
        # setattr works.
        try:
            object.__setattr__(lab, "_dump_node", node)
        except Exception:
            setattr(lab, "_dump_node", node)
        return lab

    monkeypatch.setattr(cms_mod, "cms_node_label", labeled)

    failures = compare_graphs(ref_graph, subj_graph)
    print(f"\n=== Total failures (carve-out neutralized): {len(failures)} ===")

    pack_failures = []
    for f in failures:
        if type(f).__name__ != "OrderInvertedFailure":
            continue
        p_cat = (getattr(f.producer, "category", "") or "")
        c_cat = (getattr(f.consumer, "category", "") or "")
        p_body = getattr(f.producer, "body_label", None)
        c_body = getattr(f.consumer, "body_label", None)
        if (
            (p_cat.startswith("PackA3") or p_cat.startswith("PackB3"))
            and c_cat == "MFMA"
            and p_body == c_body
        ):
            pack_failures.append(f)
    print(f"Pack3->MFMA artifact failures: {len(pack_failures)}")

    # Dump ONLY the first Pack3 -> MFMA artifact failure.
    targets = [("first Pack3 -> MFMA artifact", lambda f: True)]

    for tag, predicate in targets:
        chosen = next((f for f in pack_failures if predicate(f)), None)
        print("\n" + "=" * 78)
        print(f"=== TARGET: {tag} ===")
        print("=" * 78)
        if chosen is None:
            print(f"  (no matching failure found for {tag})")
            continue

        subj_p = getattr(chosen.producer, "_dump_node", None)
        subj_c = getattr(chosen.consumer, "_dump_node", None)
        if subj_p is None or subj_c is None:
            print(f"  ERROR: side-channel _dump_node missing on FailureNodeLabel")
            continue

        # Look up the same nodes in the reference graph by identity (the
        # nodes-dict key). Pack3 producers DO appear in the reference graph
        # because the carve-out only neutralizes the OrderInverted-gate
        # path; both graphs are still built from real captures. We rely on
        # cross-graph identity equality (TaggedInstruction.identity_for ->
        # `(class_tag, loop_index, canonical_render)`).
        ref_p = ref_graph.nodes.get(subj_p.identity)
        ref_c = ref_graph.nodes.get(subj_c.identity)

        print(f"\n  Producer (subj):")
        print(f"    identity = {subj_p.identity}")
        print(f"    category = {subj_p.category}  body = {subj_p.body_label}  name = {render_node_label(subj_p)}")
        print(f"    position = {subj_p.position}")
        print(f"    asm      = {_render_asm(subj_p)}")
        print(f"  Consumer (subj):")
        print(f"    identity = {subj_c.identity}")
        print(f"    category = {subj_c.category}  body = {subj_c.body_label}  name = {render_node_label(subj_c)}")
        print(f"    position = {subj_c.position}")
        print(f"    asm      = {_render_asm(subj_c)}")
        print(f"  default_producer_position = {chosen.default_producer_position}")
        print(f"  default_consumer_position = {chosen.default_consumer_position}")

        # SUBJECT (CMS) side: order-inverted, so producer.position should be
        # AFTER consumer.position in subject stream.
        body = subj_p.body_label
        if subj_p.position > subj_c.position:
            lo, hi = subj_c.position, subj_p.position
            subj_first, subj_last = subj_c, subj_p
            subj_dir = "consumer FIRST -> producer LAST (order-inverted, the artifact)"
        else:
            lo, hi = subj_p.position, subj_c.position
            subj_first, subj_last = subj_p, subj_c
            subj_dir = "producer FIRST -> consumer LAST (order preserved)"

        between_subj = _slice_window(subj_graph, body, lo, hi)
        print(f"\n  [SUBJECT/CMS]  {subj_dir}")
        _print_window(
            label=f"SUBJECT {tag}",
            producer=subj_first,
            consumer=subj_last,
            between=between_subj,
            prefix="    ",
        )

        # REFERENCE (default) side.
        if ref_p is None or ref_c is None:
            print(f"\n  WARN: could not resolve ref producer/consumer by identity "
                  f"(ref_p={'present' if ref_p else 'MISSING'}, "
                  f"ref_c={'present' if ref_c else 'MISSING'})")
            continue

        if ref_p.position < ref_c.position:
            rlo, rhi = ref_p.position, ref_c.position
            ref_first, ref_last = ref_p, ref_c
            ref_dir = "producer FIRST -> consumer LAST (expected order)"
        else:
            rlo, rhi = ref_c.position, ref_p.position
            ref_first, ref_last = ref_c, ref_p
            ref_dir = "consumer FIRST -> producer LAST (UNEXPECTED in reference)"

        between_ref = _slice_window(ref_graph, ref_p.body_label, rlo, rhi)
        print(f"\n  [REFERENCE/default]  {ref_dir}")
        _print_window(
            label=f"REFERENCE {tag}",
            producer=ref_first,
            consumer=ref_last,
            between=between_ref,
            prefix="    ",
        )

    # Don't fail — this is a dump tool.
    assert True
