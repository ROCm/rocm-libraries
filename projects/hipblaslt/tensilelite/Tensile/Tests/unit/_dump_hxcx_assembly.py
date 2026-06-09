"""Standalone dumper for rocm-libraries-hxcx investigation.

Builds the BPG#11 TF32 4x4 TN canonical kernel through the same production
path that test_cross_subiter_alu_carveout_real_kernel.py uses, then writes:

  hxcx_artifacts/kernel.s              — full assembly text
  hxcx_artifacts/validator_failures.txt — TimingTooCloseFailure entries with
                                           per-instruction details (producer
                                           identity, consumer identity,
                                           reported cycle gap, required gap)
  hxcx_artifacts/prologue_slice.s      — just the prologue body's emitted
                                           instructions (the failing region)
  hxcx_artifacts/edge_annotations.txt  — for each failing edge: producer
                                           SchedulePosition, consumer
                                           SchedulePosition, body label,
                                           rendered canonical text

Run with the worktree as cwd and rocisa freshly installed:

  cd <worktree>/projects/hipblaslt/tensilelite
  pip install -e ./rocisa
  python Tensile/Tests/unit/_dump_hxcx_assembly.py

Artifacts land in hxcx_artifacts/ in the cwd.
"""

import os
import sys
import pathlib

# Tree-shadowing guard (replicated from test_capture_pipeline_checks.py).
def _assert_tensile_tree_matches_test_tree():
    import Tensile.KernelWriter as _kw
    test_tree = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    kw_tree = os.path.abspath(
        os.path.join(os.path.dirname(_kw.__file__), "..")
    )
    if test_tree != kw_tree:
        raise RuntimeError(
            f"Tensile package loaded from a different tree than this script. "
            f"test_tree={test_tree!r}, kw_tree={kw_tree!r}. "
            f"Fix: `cd {test_tree}` before running."
        )


_assert_tensile_tree_matches_test_tree()


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
    'UseMFMAF32XEmulation': True, 'UsePLRPack': True,
}


def _build_isa_infrastructure():
    """Replicate the `isa_infrastructure` pytest fixture (conftest.py:32)."""
    import shutil
    from Tensile.Common import IsaVersion
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Component import Assembler

    compiler = shutil.which('amdclang++') or shutil.which('clang++')
    assembler_bin = shutil.which('amdclang') or shutil.which('clang')
    assert compiler, "No C++ compiler found for ISA capability probing"
    assert assembler_bin, "No assembler binary found"
    isaInfoMap = makeIsaInfoMap([IsaVersion(9, 5, 0)], compiler)
    asm = Assembler(assembler_bin, 'V5')
    return None, isaInfoMap, asm


def main():
    out_dir = pathlib.Path("hxcx_artifacts").resolve()
    out_dir.mkdir(exist_ok=True)

    print(f"Writing artifacts to {out_dir}")

    isa, isaInfoMap, asm = _build_isa_infrastructure()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import KernelWriterAssembly, DebugConfig
    from Tensile.Components.CMSValidator import (
        build_dataflow_graph,
        validate_edge_wait_coverage,
        compare_graphs,
    )

    config = dict(CANONICAL_KERNEL_CONFIG)
    solution = _make_solution(config, asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())

    # Skip the inline wait-coverage / compare_graphs assertion so the assembly
    # still gets emitted even when the timing violation we're investigating
    # would otherwise abort kernelBody (KernelWriter.py:6311).
    writer._capture_skip_internal_validate = True

    print("Building kernel via _getKernelSource (validator-assert skipped) ...")
    src = writer._getKernelSource(solution)

    asm_path = out_dir / "kernel.s"
    asm_path.write_text(src)
    print(f"  -> {asm_path} ({len(src)} bytes, {src.count(chr(10))} lines)")

    cms_cap = writer._last_cms_capture
    default_cap = writer._last_default_capture
    assert cms_cap is not None, "CMS capture not populated"
    assert default_cap is not None, "Default capture not populated"

    # Render each FourPartCapture as a line-numbered, body-ordered instruction
    # listing. The SHADOW (default) schedule is NOT emitted as assembly text by
    # the kernel writer; this listing is the apples-to-apples way to read it
    # next to the CMS schedule. Body order follows the unrolled timeline:
    # PRO -> ML_PREV (=ML iter0) -> ML (=ML iter1) -> NGL -> NLL.
    def _dump_capture_listing(cap, label, path):
        bodies = [
            ("PRO", cap.prologue),
            ("ML_PREV", cap.main_loop_prev),
            ("ML", cap.main_loop),
            ("NGL", cap.n_gl),
            ("NLL", cap.n_ll),
        ]
        lines = []
        lines.append(f"# {label} schedule capture listing for "
                     f"BPG#11 TF32 4x4 TN (CANONICAL_KERNEL_CONFIG)")
        lines.append(f"# Body order = unrolled timeline: PRO -> ML_PREV(iter0) "
                     f"-> ML(iter1) -> NGL -> NLL")
        lines.append(f"# Columns: <file_line> | body | body_idx | "
                     f"mfma_index | seq | canonical_render")
        lines.append("#" + "=" * 78)
        # main_loop / main_loop_prev / n_gl / n_ll may be dicts keyed by
        # codepath, or a LoopBodyCapture, or None. Normalize to an ordered
        # list of LoopBodyCapture.
        def _as_bodies(b):
            if b is None:
                return []
            if hasattr(b, "instructions"):
                return [b]
            if isinstance(b, dict):
                return [b[k] for k in sorted(b.keys())]
            if isinstance(b, (list, tuple)):
                out = []
                for x in b:
                    out.extend(_as_bodies(x))
                return out
            return []
        for body_label, raw in bodies:
            sub_bodies = _as_bodies(raw)
            if not sub_bodies:
                lines.append("")
                lines.append(f"### body {body_label}: (empty / absent)")
                continue
            for cp_i, lbc in enumerate(sub_bodies):
                lines.append("")
                tag = body_label if len(sub_bodies) == 1 else f"{body_label}[codepath={cp_i}]"
                lines.append(f"### body {tag}: {len(lbc.instructions)} instructions")
                for body_idx, ti in enumerate(lbc.instructions):
                    slot = getattr(ti, "slot", None)
                    mfma_index = getattr(slot, "mfma_index", "?")
                    seq = getattr(slot, "sequence", "?")
                    wrapped = getattr(ti, "wrapped", None)
                    inst = getattr(wrapped, "rocisa_inst", None)
                    try:
                        render = wrapped.canonical_str(inst) if wrapped is not None else str(ti)
                    except Exception:
                        render = repr(ti)
                    # file_line is the 1-based line number IN THIS listing file
                    # (filled after the fact below).
                    lines.append(f"PLACEHOLDER | {tag} | {body_idx} | "
                                 f"{mfma_index} | {seq} | {render}")
        # Now stamp real file-line numbers in place of PLACEHOLDER.
        numbered = []
        for i, ln in enumerate(lines, start=1):
            if ln.startswith("PLACEHOLDER | "):
                numbered.append(f"{i:5d} | " + ln[len("PLACEHOLDER | "):])
            else:
                numbered.append(f"{i:5d} | {ln}" if not ln.startswith("#") and ln.strip() and not ln.startswith("###") else f"{i:5d} | {ln}")
        path.write_text("\n".join(numbered) + "\n")
        return len(lines)

    cms_listing = out_dir / "cms_capture_listing.txt"
    shadow_listing = out_dir / "shadow_capture_listing.txt"
    n_cms = _dump_capture_listing(cms_cap, "CMS (subject)", cms_listing)
    n_shadow = _dump_capture_listing(default_cap, "SHADOW (reference / default)", shadow_listing)
    print(f"  -> {cms_listing} ({n_cms} lines)")
    print(f"  -> {shadow_listing} ({n_shadow} lines)")

    subj_graph = build_dataflow_graph(cms_cap)
    ref_graph = build_dataflow_graph(default_cap)

    print("Running validate_edge_wait_coverage on subj (CMS) graph ...")
    wait_failures = validate_edge_wait_coverage(subj_graph)
    print(f"  -> {len(wait_failures)} TimingTooCloseFailure entries")

    failures_path = out_dir / "validator_failures.txt"
    edge_annot_path = out_dir / "edge_annotations.txt"
    with failures_path.open("w") as fout, edge_annot_path.open("w") as eout:
        fout.write(f"# {len(wait_failures)} TimingTooCloseFailure entries from "
                   f"validate_edge_wait_coverage(subj_graph)\n")
        fout.write(f"# Kernel: BPG#11 TF32 4x4 TN (CANONICAL_KERNEL_CONFIG)\n\n")
        for i, f in enumerate(wait_failures):
            fout.write(f"=== Failure {i} ===\n{f.format()}\n\n")
            # Edge annotation if available
            edge = getattr(f, "edge", None)
            if edge is not None:
                eout.write(f"=== Failure {i} edge ===\n")
                for attr in ("producer_body_label", "consumer_body_label",
                             "producer_iter_index", "consumer_iter_index",
                             "producer_unrolled_position",
                             "consumer_unrolled_position",
                             "producer_write_byte_key",
                             "consumer_read_byte_key",
                             "edge_kind"):
                    val = getattr(edge, attr, None)
                    eout.write(f"  {attr}={val!r}\n")
                p = edge.producer
                c = edge.consumer
                p_inst = getattr(getattr(p, "tagged_inst", None), "wrapped", None)
                c_inst = getattr(getattr(c, "tagged_inst", None), "wrapped", None)
                if p_inst is not None:
                    eout.write(f"  producer.canonical_render={p_inst.canonical_str(p_inst.rocisa_inst)!r}\n")
                if c_inst is not None:
                    eout.write(f"  consumer.canonical_render={c_inst.canonical_str(c_inst.rocisa_inst)!r}\n")
                p_slot = getattr(getattr(p, "tagged_inst", None), "slot", None)
                c_slot = getattr(getattr(c, "tagged_inst", None), "slot", None)
                eout.write(f"  producer.slot={p_slot!r}\n")
                eout.write(f"  consumer.slot={c_slot!r}\n")
                eout.write(f"  producer.position={getattr(p, 'position', None)!r}\n")
                eout.write(f"  consumer.position={getattr(c, 'position', None)!r}\n")
                p_src = getattr(getattr(p, "tagged_inst", None), "source_module_id", None)
                c_src = getattr(getattr(c, "tagged_inst", None), "source_module_id", None)
                eout.write(f"  producer.source_module_id={p_src!r}\n")
                eout.write(f"  consumer.source_module_id={c_src!r}\n")
                eout.write("\n")
    print(f"  -> {failures_path}")
    print(f"  -> {edge_annot_path}")

    # Try to slice out the prologue. The prologue body emits before any
    # main-loop label, so a simple heuristic: lines from start until the first
    # label that looks like a loop entry.
    src_lines = src.splitlines()
    prologue_end = None
    for i, line in enumerate(src_lines):
        ls = line.strip()
        if (ls.startswith("label_") and "main_loop" in ls.lower()) or \
           ls in ("openLoopL", "label_openLoop:") or \
           "TopOfMainLoop" in ls or "openLoop_unroll" in ls:
            prologue_end = i
            break
    if prologue_end is None:
        # Fallback: just take the first ~2000 lines
        prologue_end = min(2000, len(src_lines))
        print(f"  (could not locate main-loop label; using first {prologue_end} lines as prologue slice)")
    prologue_path = out_dir / "prologue_slice.s"
    prologue_path.write_text("\n".join(src_lines[:prologue_end]))
    print(f"  -> {prologue_path} ({prologue_end} lines)")

    # Also dump compare_graphs output for context
    cg_failures = compare_graphs(ref_graph, subj_graph)
    cg_path = out_dir / "compare_graphs_failures.txt"
    with cg_path.open("w") as fout:
        fout.write(f"# {len(cg_failures)} compare_graphs failures\n\n")
        for i, f in enumerate(cg_failures):
            fout.write(f"=== Failure {i} ===\n{f.format()}\n\n")
    print(f"  -> {cg_path} ({len(cg_failures)} failures)")

    print("\nDone. Inspect hxcx_artifacts/ in cwd.")


if __name__ == "__main__":
    main()
