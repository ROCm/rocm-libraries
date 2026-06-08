"""Permanent regression guard: emission_ordinal invariance under UsePLRPack.

Guards the safety property that makes Option E (rocm-libraries-56e3) correct:
within each (canonical_render, source_module_id) group, instruction twins
emitted by SHADOW and CMS have semantically equivalent (reads, writes) sets
at every ordinal position. If an ordinal flip ever swaps semantically distinct
operations, this test emits a counter-example and fails.

For TF32+UsePLRPack fixtures, enumerate same-(canonical_render,
source_module_id) groups within each body in SHADOW vs CMS. For each
group with multiplicity > 1, compare the physical TaggedInstructions at
the i-th canonical-sort position and check whether the (reads, writes)
sets are semantically equivalent across SHADOW and CMS.

If they are always equivalent: Option E is SAFE (emission_ordinal flips
between SHADOW and CMS, if they occur, swap semantically interchangeable
operations). The tested invariant is `groups_rw_differ == 0`.

If they ever differ: Option E is UNSAFE — emit a counter-example and fail.

See `56e3_OPTION_E_EMPIRICAL_PROBE.md` for the original design rationale.
"""

import os
import sys

import pytest


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
            f"Tensile package loaded from a different tree. "
            f"test_tree={test_tree!r}, kw_tree={kw_tree!r}."
        )


_assert_tensile_tree_matches_test_tree()


_BPG_11_TF32_4X4_TN = dict(
    ProblemType={
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    MatrixInstruction=[16, 16, 32, 1, 1, 4, 4, 2, 2],
    DepthU=32, PrefetchGlobalRead=2, PrefetchLocalRead=1,
    DirectToLds=1, TransposeLDS=1, LocalReadVectorWidth=4,
    GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4,
    UseCustomMainLoopSchedule=1, ExpandPointerSwap=0,
    SourceSwap=1, StreamK=0,
    UseMFMAF32XEmulation=True, UsePLRPack=True,
)

_OPLB_TF32_6X8_TN = dict(
    ProblemType={
        'OperationType': 'GEMM', 'DataType': 'S', 'DestDataType': 'S',
        'F32XdlMathOp': 'X', 'TransposeA': True, 'TransposeB': False,
        'UseBeta': True, 'Batched': True,
    },
    MatrixInstruction=[16, 16, 32, 1, 1, 6, 8, 2, 2],
    DepthU=32, PrefetchGlobalRead=2, PrefetchLocalRead=1,
    DirectToLds=1, TransposeLDS=1, LocalReadVectorWidth=4,
    GlobalReadVectorWidthA=4, GlobalReadVectorWidthB=4,
    UseCustomMainLoopSchedule=1, ExpandPointerSwap=0,
    SourceSwap=1, StreamK=0,
    UseMFMAF32XEmulation=True, UsePLRPack=True,
)


def _build_shadow_cms_pair(kernel_config, asm, isaInfoMap):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cms_test_utils import _make_solution
    from Tensile.KernelWriterAssembly import (
        KernelWriterAssembly, DebugConfig,
    )

    solution = _make_solution(dict(kernel_config), asm, isaInfoMap)
    writer = KernelWriterAssembly(asm, DebugConfig())
    try:
        writer._getKernelSource(solution)
    except Exception:
        pass

    return writer._last_default_capture, writer._last_cms_capture


def _iter_bodies(fpc):
    """Yield (body_label, codepath_or_None, LoopBodyCapture) tuples."""
    if fpc.prologue is not None:
        yield ("PRO", None, fpc.prologue)
    for label, body_dict in [
        ("ML-1", fpc.main_loop_prev),
        ("ML", fpc.main_loop),
        ("NGL", fpc.n_gl),
        ("NLL", fpc.n_ll),
    ]:
        if body_dict is None:
            continue
        if isinstance(body_dict, dict):
            for cp, body in body_dict.items():
                if body is None:
                    continue
                yield (label, cp, body)
        else:
            # LoopBodyCapture directly
            yield (label, None, body_dict)


def _render_register(r):
    """Stringify a rocisa RegisterContainer by its content fields."""
    if r is None:
        return "None"
    parts = []
    for attr in ("regType", "regIdx", "regNum"):
        try:
            v = getattr(r, attr)
            parts.append(f"{attr}={v!r}")
        except Exception:
            pass
    # regName is an optional<RegName>; render its name/offsets if present.
    try:
        rn = getattr(r, "regName", None)
        if rn is None:
            parts.append("regName=None")
        else:
            try:
                name = getattr(rn, "name", None)
            except Exception:
                name = None
            try:
                offsets = list(getattr(rn, "offsets", []) or [])
            except Exception:
                offsets = []
            parts.append(f"regName=(name={name!r},offsets={offsets!r})")
    except Exception:
        pass
    return "|".join(parts) if parts else repr(r)


def _readwrite_signature(ti):
    """Return a hashable signature of an instruction's reads & writes."""
    reads = tuple(sorted(_render_register(r) for r in (ti.wrapped.reads or ())))
    writes = tuple(sorted(_render_register(w) for w in (ti.wrapped.writes or ())))
    return (reads, writes)


def test_56e3_probe(isa_infrastructure):
    from Tensile.Components.ScheduleCapture import WrappedInstruction

    _isa, isaInfoMap, asm = isa_infrastructure

    findings_lines = []
    findings_lines.append("# 56e3 Option E Empirical Probe Findings\n")

    overall_total_groups = 0
    overall_groups_with_mult_gt_1 = 0
    overall_groups_rw_differ = 0
    overall_groups_rw_match = 0
    overall_groups_multiplicity_differs = 0
    overall_mult_differs_with_source = 0
    counter_examples = []

    for fixture_id, config in [
        ("bpg11-tf32-4x4-tn", _BPG_11_TF32_4X4_TN),
        ("oplb-tf32-6x8-tn", _OPLB_TF32_6X8_TN),
    ]:
        findings_lines.append(f"\n## Fixture: {fixture_id}\n")
        try:
            shadow, cms = _build_shadow_cms_pair(config, asm, isaInfoMap)
        except Exception as exc:
            findings_lines.append(f"  BUILD FAILED: {exc!r}\n")
            continue

        if shadow is None or cms is None:
            findings_lines.append(
                f"  Captures missing: shadow={shadow is not None} cms={cms is not None}\n")
            continue

        # Build a dict keyed by (body_label, codepath) for both sides.
        shadow_bodies = {(lbl, cp): body for (lbl, cp, body) in _iter_bodies(shadow)}
        cms_bodies = {(lbl, cp): body for (lbl, cp, body) in _iter_bodies(cms)}

        common_keys = sorted(set(shadow_bodies) & set(cms_bodies),
                             key=lambda x: (str(x[0]), x[1] if x[1] is not None else -1))

        for body_key in common_keys:
            label, cp = body_key
            s_body = shadow_bodies[body_key]
            c_body = cms_bodies[body_key]
            s_insts = getattr(s_body, 'instructions', None)
            c_insts = getattr(c_body, 'instructions', None)
            if not s_insts or not c_insts:
                continue

            s_sorted = sorted(s_insts, key=lambda t: (t.slot.mfma_index, t.slot.sequence))
            c_sorted = sorted(c_insts, key=lambda t: (t.slot.mfma_index, t.slot.sequence))

            from collections import defaultdict
            s_groups = defaultdict(list)
            c_groups = defaultdict(list)
            for ti in s_sorted:
                key = (WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst),
                       ti.source_module_id)
                s_groups[key].append(ti)
            for ti in c_sorted:
                key = (WrappedInstruction.canonical_str(ti.wrapped.rocisa_inst),
                       ti.source_module_id)
                c_groups[key].append(ti)

            body_total = 0
            body_mult_gt_1 = 0
            body_rw_match = 0
            body_rw_differ = 0
            body_mult_differs = 0

            for key, s_list in s_groups.items():
                body_total += 1
                overall_total_groups += 1
                c_list = c_groups.get(key, [])
                if len(s_list) < 2 and len(c_list) < 2:
                    continue
                body_mult_gt_1 += 1
                overall_groups_with_mult_gt_1 += 1

                if len(s_list) != len(c_list):
                    body_mult_differs += 1
                    overall_groups_multiplicity_differs += 1
                    if key[1] is not None:
                        overall_mult_differs_with_source += 1
                    # Always capture mult_differs with non-None source_module_id
                    # (those are the dangerous ones if any).
                    if key[1] is not None or len(counter_examples) < 5:
                        counter_examples.append({
                            "fixture": fixture_id, "body": label, "cp": cp,
                            "kind": "multiplicity_differs",
                            "render": key[0][:120],
                            "source_module_id": key[1],
                            "shadow_n": len(s_list),
                            "cms_n": len(c_list),
                            "shadow_slots": [(t.slot.mfma_index, t.slot.sequence) for t in s_list],
                            "cms_slots": [(t.slot.mfma_index, t.slot.sequence) for t in c_list],
                        })
                    continue

                # Same multiplicity — compare position-by-position
                any_rw_differ = False
                for i, (s_ti, c_ti) in enumerate(zip(s_list, c_list)):
                    s_sig = _readwrite_signature(s_ti)
                    c_sig = _readwrite_signature(c_ti)
                    if s_sig != c_sig:
                        any_rw_differ = True
                        if len(counter_examples) < 10:
                            counter_examples.append({
                                "fixture": fixture_id, "body": label, "cp": cp,
                                "kind": "rw_differ_at_ordinal",
                                "ordinal": i,
                                "render": key[0][:120],
                                "source_module_id": key[1],
                                "shadow_reads": s_sig[0],
                                "shadow_writes": s_sig[1],
                                "cms_reads": c_sig[0],
                                "cms_writes": c_sig[1],
                                "shadow_slot": (s_ti.slot.mfma_index, s_ti.slot.sequence),
                                "cms_slot": (c_ti.slot.mfma_index, c_ti.slot.sequence),
                                "shadow_other_rws": [_readwrite_signature(t) for t in s_list],
                                "cms_other_rws": [_readwrite_signature(t) for t in c_list],
                            })
                if any_rw_differ:
                    body_rw_differ += 1
                    overall_groups_rw_differ += 1
                else:
                    body_rw_match += 1
                    overall_groups_rw_match += 1

            findings_lines.append(
                f"  body={label!r} cp={cp!r}: total_groups={body_total} "
                f"mult>1={body_mult_gt_1} rw_match={body_rw_match} "
                f"rw_differ={body_rw_differ} mult_differs={body_mult_differs}\n"
            )

    findings_lines.append("\n## Overall counts\n")
    findings_lines.append(f"  total_groups={overall_total_groups}\n")
    findings_lines.append(f"  groups_with_multiplicity>1={overall_groups_with_mult_gt_1}\n")
    findings_lines.append(f"  groups_rw_match (Option E safe)={overall_groups_rw_match}\n")
    findings_lines.append(f"  groups_rw_differ (Option E UNSAFE)={overall_groups_rw_differ}\n")
    findings_lines.append(f"  groups_multiplicity_differs={overall_groups_multiplicity_differs}\n")
    findings_lines.append(f"  mult_differs_with_NON_NONE_source={overall_mult_differs_with_source}\n")

    findings_lines.append("\n## Counter-examples\n")
    if not counter_examples:
        findings_lines.append("  (none)\n")
    else:
        for ce in counter_examples:
            findings_lines.append(f"  - {ce}\n")

    output = "".join(findings_lines)
    # Write the report into the test directory so we can read it.
    out_path = "/tmp/56e3_probe_findings.txt"
    with open(out_path, "w") as f:
        f.write(output)
    print(output)
    print(f"\n[PROBE] wrote {out_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
