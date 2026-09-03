"""Regression tests for the LDS bank-conflict model + its gates (`rocke.helpers.tiling.lds_conflict`).

CPU-only (no GPU / container): they exercise the write-port simulator and the two GATES that wrap it.
The gates previously tripped on a case the validation corpus never covered -- a **b128 store into a
256-wide tile**, whose K-alias is **depth 16** (the corpus only had depth-8 cases: b64/b32 at
tile_free=128, b128 at tile_free=128). The MODEL was correct at depth 16; the gates were coded/tuned
against depth 8. These tests pin the depth-16 regime so selftest + pytest guard it going forward.
"""
import pytest

from rocke.helpers.tiling import lds_conflict as lc
from rocke.helpers.tiling.kernels.tiling_gemm_crc_demo.crc_interleaved_gemm import _macro_coop_descs_crc
from rocke.helpers.tiling.kernels.tiling_gemm_interleaved_demo import _wave_descs_interleaved, _transpose_desc

# The winner CRC A local store: macro tile_m=256, tile_k=32, 16 waves; wave read m_sub=4,n_sub=4,k_sub=2.
TILE_FREE, WTAG, DEPTH16 = 256, "b128", 16
FIX_STRIDE_DW = (TILE_FREE + 16) // 2      # pad +16 f16 -> 136 dwords (the HW-verified conflict-free pad)


def _a_descs():
    a_coop = _macro_coop_descs_crc(TILE_FREE, 32, 16)
    a_wave = _wave_descs_interleaved(4, 4, 2)[0]
    return lc.ProbeDescs.from_coop(a_coop, a_wave, transpose=_transpose_desc)


def _sim_pad0():
    """The write-port simulator on the pad0 address map -> conflicts/access (the value the GPU confirmed)."""
    a = lc.GFX90A
    acc, vw, datum = lc.store_datum(_a_descs().coop_store, TILE_FREE, a, (TILE_FREE, 1), "f16")
    r = lc.simulate(acc, arch=a, dtype_bytes=2)
    return r["BC"] / (r["IDX"] - r["BC"]) if (r["IDX"] - r["BC"]) else 0.0


def _hw_stub(pad, mode="store"):
    """Emulate the GPU `measure` with the VALIDATED predictors (matches the live rocprof run): pad0 from the
    write-port sim (3.0, == HW), any padded stride from the depth-aware stripe rule (conflict-free -> 0.0, ==
    HW). NOTE: `simulate()` is NOT the fix predictor -- at a padded stride it disagrees with the stripe rule
    and the GPU, and `analyze_store` never sims a padded stride (it uses the stripe rule + the GPU)."""
    if pad == 0:
        ca = _sim_pad0()
    else:
        ca = 0.0 if lc.is_conflict_free((TILE_FREE + pad) // 2, WTAG, lc.GFX90A, pad0_depth=DEPTH16) else 1.0
    return {"conflicts_per_access": ca, "max_abs_diff": 0.0}


def test_selftest_still_reproduces_the_corpus():
    """The write-port model still reproduces every gfx90a validation-corpus measurement (the mechanism gate)."""
    assert lc.selftest(lc.GFX90A) is True


def test_model_predicts_the_depth16_a_store_conflict():
    """The model (unchanged) predicts 3.0 conflicts/access for the depth-16 b128 A store -- matching the GPU
    (BC=811008, IDX=1081344 -> 3.0); the depth-aware stripe rule predicts pad+16 conflict-free (== GPU BC=0).
    Guards the model at depth 16."""
    assert _sim_pad0() == pytest.approx(3.0, abs=1e-9)
    assert lc.is_conflict_free(FIX_STRIDE_DW, WTAG, lc.GFX90A, pad0_depth=DEPTH16)


def test_gate_tolerates_live_ratio_noise_but_rejects_a_real_mismatch():
    """FIX 1: the HW gate compares a LIVE whole-run ratio -- it must accept sub-percent counter noise
    (3.000 sim vs 3.005 measured) yet still reject a real model error (whole conflict factors apart)."""
    lc.gate({"conflicts_per_access": 3.0}, {"conflicts_per_access": 3.0052})          # noise -> pass
    with pytest.raises(lc.ConflictModelError):
        lc.gate({"conflicts_per_access": 3.0}, {"conflicts_per_access": 3.5})         # real mismatch -> raise


def test_conflict_free_rule_is_depth_aware_for_b128_into_a_wide_tile():
    """FIX 2: a b128 store into a 256-wide tile K-aliases at DEPTH 16, so the stripe rule MUST be told the
    pad0 depth. With it, pad+16 (stride 136 dw) is conflict-free (== the GPU: BC=0); the depth-8 DEFAULT
    spuriously says otherwise -- which is exactly what mis-rejected the fix in the render gate."""
    a = lc.GFX90A
    assert lc.predict_pad_sweep(FIX_STRIDE_DW, WTAG, a, pad0_depth=DEPTH16) == 0.0
    assert lc.is_conflict_free(FIX_STRIDE_DW, WTAG, a, pad0_depth=DEPTH16)
    assert lc.predict_pad_sweep(FIX_STRIDE_DW, WTAG, a) != 0.0     # depth-agnostic default: WRONG here


def test_the_two_predictors_split_by_design_do_not_try_to_unify_them():
    """DESIGN CONTRACT (LDS Expert): the conflict-FREE verdict is a half-stripe PARITY property. The
    address-map `simulate` (naive bank = dword mod NB histogram) is STRUCTURALLY BLIND to parity, so it
    reproduces the magnitude of CONFLICTED pads but can NEVER reach 0 at the parity-resolved pads (here a
    depth-16 b128 store keeps a spurious residual at pad16). The DEPTH-AWARE stripe rule is the conflict-free
    predictor (matches the GPU: pad16 -> BC=0). Do NOT 'fix' simulate to return 0 -- the split is intentional;
    `analyze_store`/`render` gate the fix on `is_conflict_free`, never on simulate at a padded stride."""
    a = lc.GFX90A
    acc, _vw, _d = lc.store_datum(_a_descs().coop_store, TILE_FREE, a, (TILE_FREE + 16, 1), "f16")
    r = lc.simulate(acc, arch=a, dtype_bytes=2)
    sim_ca = r["BC"] / (r["IDX"] - r["BC"]) if (r["IDX"] - r["BC"]) else 0.0
    assert sim_ca > 0.0, "simulate is expected to be parity-blind here (structural), NOT a conflict-free oracle"
    assert lc.is_conflict_free(FIX_STRIDE_DW, WTAG, a, pad0_depth=DEPTH16), "stripe rule IS the fix oracle (== GPU)"


def test_analyze_store_runs_both_gates_end_to_end_at_depth16(tmp_path):
    """FIX 1 + FIX 2 together, on the exact tripping geometry, CPU-only: a stub `measure` returns the
    simulator's own prediction so the HW gate passes trivially, exercising recommend_pad + the render's
    depth-aware GATE 2. Previously this raised `A fix pad16 not conflict-free by the validated rule`."""
    descs = _a_descs()
    rep = lc.analyze_store(
        descs, tile_free=TILE_FREE, wtag=WTAG, operand_label="A", dims_label="M", tile_k=32, n_waves=16,
        macro_label="macro 256x256, waves 4x4, tile_k=32", strides=(TILE_FREE, 1),
        measure=_hw_stub, verify_fix=True,
        render_to=str(tmp_path / "lds_conflict_A_store.png"))
    assert rep.gate_passed and rep.fix_verified_hw
    assert rep.conflicts_per_access == pytest.approx(3.0, abs=1e-9)
    assert rep.fix_pad == 16
    assert rep.located["nway"] == DEPTH16 and rep.located["bank"] == 0
    import pathlib
    assert pathlib.Path(rep.png).exists() and pathlib.Path(rep.png).stat().st_size > 0
