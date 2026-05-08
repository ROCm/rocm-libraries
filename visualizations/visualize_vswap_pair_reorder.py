"""Render the two graphs from `test_vswap_pair_reorder_detected` to PNG.

Run from anywhere:

    python <worktree>/visualizations/visualize_vswap_pair_reorder.py

Outputs land next to this script:

    vswap_pair_reorder_ref.png   — REF: VSwap(v0,v1); VSwap(v1,v2)
    vswap_pair_reorder_subj.png  — SUBJ: VSwap(v1,v2); VSwap(v0,v1)

Inspect the PNGs to verify operand-slot identity makes the reorder
visible: the shared register `v1` lands at different operand-slots in
REF vs SUBJ, which the new edge identity (rocm-libraries-wx9.3 phase 3,
memo §6.1 step 1) surfaces as an `OrderInvertedFailure`.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE_ROOT = HERE.parent
TENSILELITE = WORKTREE_ROOT / "projects" / "hipblaslt" / "tensilelite"
sys.path.insert(0, str(TENSILELITE))
sys.path.insert(0, str(TENSILELITE / "Tensile" / "Tests" / "unit"))

from rocisa.container import vgpr
from rocisa.instruction import VSwapB32

from Tensile.Components.ScheduleCapture import (
    BODY_LABEL_ML, BODY_LABEL_ML_PREV, BODY_LABEL_NGL, BODY_LABEL_NLL,
    FourPartCapture, SLOT_KIND_MFMA, SlotKey,
    TaggedInstruction, WrappedInstruction,
)
from Tensile.Components.CMSValidator import (
    _DEFAULT_CDNA4_ARCH_PROFILE,
    build_dataflow_graph,
)
from Tensile.Components.CMSValidatorVisualization import visualize_dataflow_graph
from dataflow_fixtures import make_mfma, make_capture


def _wrap(ml_capture):
    """Same minimal `_wrap` as the test: fill the other 3 bodies with a
    single filler MFMA so `build_dataflow_graph` sees non-empty bodies.
    """
    _FILLER = {
        BODY_LABEL_ML_PREV: (200, 204, 208),
        BODY_LABEL_NGL:     (220, 224, 228),
        BODY_LABEL_NLL:     (240, 244, 248),
    }

    def _f(label):
        c, a, b = _FILLER[label]
        return make_capture(label, [make_mfma(
            c_dst_start=c, a_src_start=a, b_src_start=b, slot=0)])

    return FourPartCapture(
        main_loop={0: ml_capture},
        main_loop_prev={0: _f(BODY_LABEL_ML_PREV)},
        n_gl={0: _f(BODY_LABEL_NGL)},
        n_ll={0: _f(BODY_LABEL_NLL)},
        num_mfma=1, num_codepaths=1, source="cms",
        arch_profile=_DEFAULT_CDNA4_ARCH_PROFILE,
    )


def _tag(inst, *, category, mfma_index, sequence):
    return TaggedInstruction(
        wrapped=WrappedInstruction(inst),
        category=category,
        slot=SlotKey(subiter=0, slot_kind=SLOT_KIND_MFMA,
                     mfma_index=mfma_index, sequence=sequence),
    )


# REF: sw1=VSwap(v0,v1) at sub=0, sw2=VSwap(v1,v2) at sub=1
sw1 = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
sw2 = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
ref_cap = make_capture(BODY_LABEL_ML, [
    _tag(sw1, category="PackA0", mfma_index=0, sequence=0),
    _tag(sw2, category="PackA0", mfma_index=0, sequence=1),
])

# SUBJ: sw2=VSwap(v1,v2) at sub=0, sw1=VSwap(v0,v1) at sub=1 (literally swapped)
sw1b = VSwapB32(dst=vgpr(0, 1), src=vgpr(1, 1))
sw2b = VSwapB32(dst=vgpr(1, 1), src=vgpr(2, 1))
subj_cap = make_capture(BODY_LABEL_ML, [
    _tag(sw2b, category="PackA0", mfma_index=0, sequence=0),
    _tag(sw1b, category="PackA0", mfma_index=0, sequence=1),
])

g_ref = build_dataflow_graph(_wrap(ref_cap))
g_subj = build_dataflow_graph(_wrap(subj_cap))

ref_path = str(HERE / "vswap_pair_reorder_ref.png")
subj_path = str(HERE / "vswap_pair_reorder_subj.png")
visualize_dataflow_graph(
    g_ref, ref_path,
    title="VSwap pair-reorder REF — sw1=VSwap(v0,v1) at sub=0, sw2=VSwap(v1,v2) at sub=1",
)
visualize_dataflow_graph(
    g_subj, subj_path,
    title="VSwap pair-reorder SUBJ — sw2=VSwap(v1,v2) at sub=0, sw1=VSwap(v0,v1) at sub=1 (swapped)",
)
print(f"REF graph nodes: {len(g_ref.nodes)} edges: {len(g_ref.edges)} -> {ref_path}")
print(f"SUBJ graph nodes: {len(g_subj.nodes)} edges: {len(g_subj.edges)} -> {subj_path}")
