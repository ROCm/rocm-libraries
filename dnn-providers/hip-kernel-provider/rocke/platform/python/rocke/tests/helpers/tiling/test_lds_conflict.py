"""Regression tests for the LDS bank-conflict model + its gates (`rocke.helpers.tiling.lds_conflict`).

CPU-only (no GPU / container): they exercise the write-port simulator and the two GATES that wrap it.
The gates previously tripped on a case the validation corpus never covered -- a **b128 store into a
256-wide tile**, whose K-alias is **depth 16** (the corpus only had depth-8 cases: b64/b32 at
tile_free=128, b128 at tile_free=128). The MODEL was correct at depth 16; the gates were coded/tuned
against depth 8. These tests pin the depth-16 regime so selftest + pytest guard it going forward.
"""
import pathlib

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
    acc, vw, datum = lc.store_datum(_a_descs().coop_store, TILE_FREE, a, (TILE_FREE, 1), "f16",
                                    origin=(0, 0), lds_swizzle=False)
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
    # The depth-agnostic default used to answer here -- WRONGLY (it assumed depth 8). A silently wrong
    # answer is worse than none, so omitting the depth is now a hard error instead of a guess.
    with pytest.raises(ValueError, match="pad0_depth is required"):
        lc.predict_pad_sweep(FIX_STRIDE_DW, WTAG, a, pad0_depth=None)


def test_the_two_predictors_split_by_design_do_not_try_to_unify_them():
    """DESIGN CONTRACT (LDS Expert): the conflict-FREE verdict is a half-stripe PARITY property. The
    address-map `simulate` (naive bank = dword mod NB histogram) is STRUCTURALLY BLIND to parity, so it
    reproduces the magnitude of CONFLICTED pads but can NEVER reach 0 at the parity-resolved pads (here a
    depth-16 b128 store keeps a spurious residual at pad16). The DEPTH-AWARE stripe rule is the conflict-free
    predictor (matches the GPU: pad16 -> BC=0). Do NOT 'fix' simulate to return 0 -- the split is intentional;
    `analyze_store`/`render` gate the fix on `is_conflict_free`, never on simulate at a padded stride."""
    a = lc.GFX90A
    acc, _vw, _d = lc.store_datum(_a_descs().coop_store, TILE_FREE, a, (TILE_FREE + 16, 1), "f16",
                                  origin=(0, 0), lds_swizzle=False)
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
        descs, mode="investigate", tile_free=TILE_FREE, wtag=WTAG, arch=lc.GFX90A,
        kernel_label="CRC", operand_label="A", dims_label="M", tile_k=32, n_waves=16,
        macro_label="macro 256x256, waves 4x4, tile_k=32", strides=(TILE_FREE, 1),
        dtype_name="f16", origin=(0, 0), lds_swizzle=False,
        measure=_hw_stub, verify_fix=True,
        render_to=str(tmp_path / "lds_conflict_A_store.png"))
    assert rep.gate_passed and rep.fix_verified_hw
    assert rep.verdict.startswith("VALIDATED")
    assert rep.conflicts_per_access == pytest.approx(3.0, abs=1e-9)
    assert rep.fix_pad == 16
    assert rep.located["nway"] == DEPTH16 and rep.located["bank"] == 0
    import pathlib
    assert pathlib.Path(rep.png).exists() and pathlib.Path(rep.png).stat().st_size > 0


def test_simulate_mode_needs_no_hardware_and_labels_itself(tmp_path):
    """SIMULATE mode on the same geometry: no `measure`, no GPU. It still runs both model gates and
    produces the same conflicts/access -- but the verdict says SIMULATED, so the number can never be
    read as a hardware result."""
    rep = lc.analyze_store(
        _a_descs(), mode="simulate", tile_free=TILE_FREE, wtag=WTAG, arch=lc.GFX90A,
        kernel_label="CRC", operand_label="A", dims_label="M", tile_k=32, n_waves=16,
        macro_label="macro 256x256, waves 4x4, tile_k=32", strides=(TILE_FREE, 1),
        dtype_name="f16", origin=(0, 0), lds_swizzle=False,
        render_to=str(tmp_path / "sim_A_store.png"))
    assert rep.measured is None and rep.gate_passed is False and rep.model_validated
    assert rep.verdict.startswith("SIMULATED")
    assert rep.conflicts_per_access == pytest.approx(3.0, abs=1e-9)
    assert rep.fix_pad == 16
    rows = rep.facts_table()
    assert "simulated" in rows["hard_facts_row"], "a simulated row must never look like counters"


def test_simulate_mode_full_stops_on_an_arch_with_no_validated_model():
    """The headline rule: an arch we have not validated is a STOP, never an extrapolation from
    gfx90a's constants."""
    with pytest.raises(ValueError, match="no validated LDS model"):
        lc.analyze_store(
            _a_descs(), mode="simulate", tile_free=TILE_FREE, wtag=WTAG, arch="gfx942",
            kernel_label="CRC", operand_label="A", dims_label="M", tile_k=32, n_waves=16,
            macro_label="macro 256x256, waves 4x4, tile_k=32", strides=(TILE_FREE, 1),
            dtype_name="f16", origin=(0, 0), lds_swizzle=False)


def test_the_two_modes_do_not_blur_into_each_other():
    """investigate without hardware, or simulate WITH it, are both caller confusion -- refuse rather
    than silently downgrade/upgrade the provenance of the answer."""
    common = dict(mode="investigate", tile_free=TILE_FREE, wtag=WTAG, arch=lc.GFX90A,
                  kernel_label="CRC", operand_label="A", dims_label="M", tile_k=32, n_waves=16,
                  macro_label="macro 256x256, waves 4x4, tile_k=32", strides=(TILE_FREE, 1),
                  dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                          dwords_per_lane=2)
    with pytest.raises(lc.ConflictModelError, match="investigate mode requires"):
        lc.analyze_store(_a_descs(), **common)
    with pytest.raises(ValueError, match="simulate mode takes no"):
        lc.analyze_store(_a_descs(), **{**common, "mode": "simulate", "measure": _hw_stub})


def test_render_refuses_a_measured_figure_without_a_measurement(tmp_path):
    """The provenance fork is enforced, not advisory: asking for a MEASURED figure with no measured
    number must refuse rather than quietly draw the simulator's value under a 'MEASURED' title."""
    store = _transpose_desc(_macro_coop_descs_crc(128, 16, 8))
    with pytest.raises(lc.ConflictModelError, match="refusing to render"):
        lc.render_conflict_3panel(str(tmp_path / "nope.png"), store_desc=store, tile_free=128,
                                  wtag="b64", subject_pad=0, measured_cpa=None,
                                  fix_pad=16, fix_label="pad+16 -> 0", provenance="measured",
                                  arch=lc.GFX90A, kernel_label="CRC", operand_label="A",
                                  dims_label="M", macro_label="macro 128x256, waves 2x4, tile_k=16",
                                  strides=(128, 1), dtype_name="f16", origin=(0, 0),
                                  lds_swizzle=False)


def test_render_simulated_is_watermarked_and_needs_no_hardware(tmp_path):
    """SIMULATE mode: no measured number at all, gated only on selftest(arch), and the figure carries
    the SIMULATED watermark so a stray PNG can never be read as a hardware result."""
    store = _transpose_desc(_macro_coop_descs_crc(128, 16, 8))
    p = lc.render_conflict_3panel(str(tmp_path / "conf_sim.png"), store_desc=store, tile_free=128,
                                  wtag="b64", subject_pad=0, fix_pad=16, fix_label="pad+16 -> 0",
                                  provenance="simulated", arch=lc.GFX90A, kernel_label="CRC",
                                  operand_label="A", dims_label="M",
                                  macro_label="macro 128x256, waves 2x4, tile_k=16",
                                  strides=(128, 1), dtype_name="f16", origin=(0, 0),
                                  lds_swizzle=False)
    assert pathlib.Path(p).stat().st_size > 0


# --------------------------------------------------------------------------------------------------
# READ-side. The write-port constants do NOT apply to reads. These tests pin the two tiers: a COST when
# the arch has a validated read model and the access is in envelope, and GEOMETRY-with-a-refusal
# otherwise -- the honesty gate, not just a feature.
# --------------------------------------------------------------------------------------------------
def test_read_datum_handles_a_sub_dword_read_that_store_datum_silently_drops():
    """The MMA-operand read is vw=1 f16 -- HALF a dword. store_datum's `vw // per_dword` floors to 0
    and yields an EMPTY map (a silent no-analysis). read_datum must map it."""
    descs = _a_descs()
    _acc, vw, store_shaped = lc.store_datum(descs.wave_read, TILE_FREE, lc.GFX90A, (TILE_FREE, 1),
                                            "f16", origin=(0, 0), lds_swizzle=False)
    assert vw == 1 and store_shaped == {}, "guard: this IS the sub-dword case store_datum drops"

    _acc, vw, datum = lc.read_datum(descs.wave_read, TILE_FREE, lc.GFX90A, (TILE_FREE, 1), "f16",
                                    origin=(0, 0), lds_swizzle=False)
    assert datum, "read_datum must not silently return an empty map"
    assert all(0 <= bank < lc.GFX90A.NB for (_k, _f, _d, bank) in datum.values())


def test_analyze_read_refuses_a_cost_for_an_arch_with_no_read_model():
    """gfx90a's read model ships registered. Any OTHER arch has no measured read port, so a read there
    is geometry only and asking for a cost must RAISE rather than extrapolate gfx90a's constants."""
    rep = lc.analyze_read(_a_descs(), tile_free=TILE_FREE, arch=lc.GFX90A, operand_label="A",
                          strides=(TILE_FREE, 1), dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                          dwords_per_lane=2)
    assert rep.max_nway >= 1
    saved = lc._READ_MODELS.pop("gfx90a", None)          # simulate an unmeasured arch
    try:
        bare = lc.analyze_read(_a_descs(), tile_free=TILE_FREE, arch=lc.GFX90A, operand_label="A",
                               strides=(TILE_FREE, 1), dtype_name="f16", origin=(0, 0),
                               lds_swizzle=False, dwords_per_lane=2)
        assert bare.sim is None and not bare.model_validated
        assert "GEOMETRY ONLY" in bare.verdict and "no validated read-port model" in bare.verdict
        assert "n/a (geometry only)" in bare.facts_table()
        with pytest.raises(lc.ConflictModelError, match="no conflicts/access"):
            _ = bare.conflicts_per_access
    finally:
        # UNCONDITIONAL restore: `if saved is not None` skips the restore whenever the pop returned
        # None, leaving the global deleted for every later test in the process.
        lc._READ_MODELS["gfx90a"] = saved if saved is not None else True


def test_analyze_read_origin_is_load_bearing_not_defaulted():
    """Each wave reads a different slice; a defaulted origin would analyze another wave's access."""
    with pytest.raises(TypeError, match="origin"):
        lc.analyze_read(_a_descs(), tile_free=TILE_FREE, arch=lc.GFX90A, operand_label="A",
                        strides=(TILE_FREE, 1), dtype_name="f16", lds_swizzle=False,
                        dwords_per_lane=2)


def test_analyze_read_full_stops_on_an_arch_with_no_model():
    with pytest.raises(ValueError, match="no validated LDS model"):
        lc.analyze_read(_a_descs(), tile_free=TILE_FREE, arch="gfx942", operand_label="A",
                        strides=(TILE_FREE, 1), dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                          dwords_per_lane=2)


# --------------------------------------------------------------------------------------------------
# The gfx90a READ-port model. Validated on hardware (rocprofv3, n_reads slope design). The read port
# is structurally DIFFERENT from the write port: no port-width cap, no write-combine, and phases
# SERIALIZE rather than pipeline -- so conflicts/access = max_depth - 1.
# --------------------------------------------------------------------------------------------------
def _read_descs(m_sub, tile_free=256):
    coop = _macro_coop_descs_crc(tile_free, 16, 16)
    return lc.ProbeDescs.from_coop(coop, _wave_descs_interleaved(m_sub, m_sub, 1)[0],
                                   transpose=_transpose_desc)


def test_read_model_reproduces_its_measured_corpus():
    lc.register_read_model("gfx90a")
    assert lc.selftest(lc.GFX90A, verbose=False)


@pytest.mark.parametrize("m_sub,pad,measured_cpa", [
    (2, 8, 0.0), (4, 8, 1.0), (8, 8, 3.0), (16, 8, 7.0),      # depth ladder 1/2/4/8
    (2, 0, 1.0), (2, 2, 1.0), (2, 4, 1.0), (2, 6, 1.0),       # depth 2, banks_used 16->28
])
def test_analyze_read_reproduces_hardware_end_to_end(m_sub, pad, measured_cpa):
    """From the real descriptors through the address map to conflicts/access, against rocprof."""
    lc.register_read_model("gfx90a")
    r = lc.analyze_read(_read_descs(m_sub), tile_free=256, arch=lc.GFX90A, operand_label="A",
                        strides=(256 + pad, 1), dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                          dwords_per_lane=2)
    assert r.conflicts_per_access == pytest.approx(measured_cpa, abs=1e-9)
    assert r.conflicts_per_access == r.max_nway - 1, "read cost is max_depth - 1"


def test_cost_depends_on_max_depth_ONLY_not_on_banks_used():
    """The measurement that makes MAX a fact rather than an assumption: banks_used sweeps 16->28 and
    the mean depth falls 2.00->1.14, while the cost stays put. Refutes mean-depth and
    accesses-per-bank rules, not just one competitor."""
    lc.register_read_model("gfx90a")
    seen = {}
    for pad in (0, 2, 4, 6):
        r = lc.analyze_read(_read_descs(2), tile_free=256, arch=lc.GFX90A, operand_label="A",
                            strides=(256 + pad, 1), dtype_name="f16", origin=(0, 0),
                            lds_swizzle=False, dwords_per_lane=2)
        banks = len(r.detail[(0, 0)])
        seen[banks] = r.conflicts_per_access
        assert r.max_nway == 2
    assert len(seen) == 4, f"expected 4 distinct banks_used, got {sorted(seen)}"
    assert len(set(seen.values())) == 1, f"cost must not vary with banks_used: {seen}"


def test_selftest_BLOCKS_a_corpus_that_cannot_discriminate_max_from_mean():
    """A uniform-only corpus reproduces max, mean and accesses-per-bank identically, so it cannot
    validate the rule. The gate must refuse it rather than trust whoever wrote it."""
    lc.register_read_model("gfx90a")
    corpus = lc._VALIDATION_CORPUS["gfx90a"]
    full = list(corpus["read_hists"])
    try:
        corpus["read_hists"] = [r for r in full if len(r[1]) == 1]     # uniform rows only
        assert not lc.selftest(lc.GFX90A, verbose=False)
        corpus["read_hists"] = [r for r in full if max(r[1]) != 1]     # no conflict-free floor
        assert not lc.selftest(lc.GFX90A, verbose=False)
        corpus["read_hists"] = []
        with pytest.raises(lc.ConflictModelError, match="NO read corpus"):
            lc.selftest(lc.GFX90A, verbose=False)
    finally:
        corpus["read_hists"] = full


def test_read_datum_models_the_MERGED_instruction_not_the_declared_vw():
    """The emit declares vw=1; the backend merges adjacent elements into ds_read2_b32. Counting
    per-element accesses would double the phase count and halve the per-instruction footprint."""
    r = lc.analyze_read(_read_descs(2), tile_free=256, arch=lc.GFX90A, operand_label="A",
                        strides=(264, 1), dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                          dwords_per_lane=2)
    assert r.n_instr == 2, "m_sub=2 issues 2 x ds_read2_b32 per read (disassembled)"
    assert r.footprint_dwords == 128, "64 lanes x 2 dwords per instruction"


def test_out_of_envelope_refuses_a_number_rather_than_extrapolating():
    """The read model is validated ONLY at 2 dwords/lane, so a caller declaring any other width must
    be REFUSED rather than priced with the 2-dword rule. NOTE the width is an ISA property that is NOT
    derivable from the address map (a lane with 8 contiguous dwords still issues 4 x ds_read2_b32), so
    it is a caller assertion -- this pins the refusal, not a detection. The assertion is unconditional
    on purpose: guarding it with `if not r.in_envelope` would let it pass the moment the gate stops
    firing."""
    lc.register_read_model("gfx90a")
    descs = lc.ProbeDescs.from_coop(_macro_coop_descs_crc(256, 32, 16),
                                    _wave_descs_interleaved(4, 4, 2)[0], transpose=_transpose_desc)
    r = lc.analyze_read(descs, tile_free=256, arch=lc.GFX90A, operand_label="A", strides=(264, 1),
                        dtype_name="f16", origin=(0, 0), lds_swizzle=False, dwords_per_lane=4)
    assert not r.in_envelope
    assert "validated only at 2" in r.out_of_envelope_reason
    assert r.sim is None and "GEOMETRY ONLY" in r.verdict
    with pytest.raises(lc.ConflictModelError):
        _ = r.conflicts_per_access


def test_dwords_per_lane_is_required_and_checked_against_the_map():
    """It comes from the disassembly; the emit's declared vw cannot tell you. Omitting it is a
    TypeError, and a value inconsistent with the address map is rejected rather than rounded."""
    descs = _read_descs(2)
    with pytest.raises(TypeError, match="dwords_per_lane"):
        lc.analyze_read(descs, tile_free=256, arch=lc.GFX90A, operand_label="A", strides=(264, 1),
                        dtype_name="f16", origin=(0, 0), lds_swizzle=False)
    with pytest.raises(ValueError, match="not a whole number"):
        lc.analyze_read(descs, tile_free=256, arch=lc.GFX90A, operand_label="A", strides=(264, 1),
                        dtype_name="f16", origin=(0, 0), lds_swizzle=False, dwords_per_lane=3)


# --------------------------------------------------------------------------------------------------
# The machinery that decides whether the model is APPLIED AT ALL. Mutation testing found these
# unpinned: deleting the import-time registration, or the registration gate, left the suite green.
# --------------------------------------------------------------------------------------------------
def test_gfx90a_read_model_is_registered_at_import():
    """The headline behaviour: callers get COST without opting in.

    This MUST run in a fresh interpreter. Asserting on the in-process module is vacuous, because
    sibling tests call `register_read_model` themselves -- so the state under test is repaired by
    whoever ran first, and deleting the module-scope registration leaves the suite green."""
    import os
    import subprocess
    import sys
    probe = (
        "from rocke.helpers.tiling import lds_conflict as lc;"
        "assert lc.READ_MODEL_REGISTRATION_ERROR is None, lc.READ_MODEL_REGISTRATION_ERROR;"
        "assert 'gfx90a' in lc._READ_MODELS, 'read model NOT registered at import';"
        "print('ok')"
    )
    # Propagate this interpreter's sys.path: a bare subprocess gets neither the test runner's
    # rootdir insertion nor an exported PYTHONPATH, so `rocke` is unimportable from any cwd but
    # platform/python -- which surfaces as a FALSE "not registered at import" rather than an
    # ImportError. The test would be red on any CI runner and green only on a shell that happens
    # to export PYTHONPATH.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"fresh import did not register the read model:\n{r.stderr}"


def test_a_corpus_regression_does_not_break_the_module_import():
    """Registration failure must NOT raise at import: this module is imported by the recorder, the
    pipeline renderer and layout_optimizer, none of which touch the read port. An unimportable module
    is a worse failure than an unpriced read, so the error is recorded in a sentinel instead."""
    import importlib
    corpus = lc._VALIDATION_CORPUS["gfx90a"]
    saved = list(corpus["read_hists"])
    try:
        corpus["read_hists"] = []                      # the regression
        with pytest.raises(lc.ConflictModelError, match="NO read corpus"):
            lc.register_read_model("gfx90a")
        # A failed RE-registration must RESTORE the model that was already validated, not destroy it.
        assert "gfx90a" in lc._READ_MODELS, "a failed re-registration wiped a working model"
    finally:
        corpus["read_hists"] = saved
        importlib.reload(lc)                            # restore module state for later tests


def test_register_read_model_refuses_an_arch_with_no_corpus():
    """gfx942 has no LDS model at all, so `arch_lds` refuses first with ValueError -- an earlier and
    stricter stop than the read-corpus check. Either way registration must not stick."""
    with pytest.raises(ValueError, match="no validated LDS model"):
        lc.register_read_model("gfx942")
    assert "gfx942" not in lc._READ_MODELS


# --------------------------------------------------------------------------------------------------
# Probe guardrails. These need NO GPU -- they are build-time refusals -- and the device buffer
# overrun that motivated this work would have been caught by the first one in 0.1s.
# --------------------------------------------------------------------------------------------------
def _probe_descs(tf, tk, nw, m_sub=4, k_sub=2):
    return lc.ProbeDescs.from_coop(_macro_coop_descs_crc(tf, tk, nw),
                                   _wave_descs_interleaved(m_sub, m_sub, k_sub)[0],
                                   transpose=_transpose_desc)


def _build(descs, mode, tf, tk, nw, wf):
    from rocke.core.ir import F16
    return lc.build_probe(descs, mode, tile_free=tf, tile_k=tk, n_waves=nw, warp_free=wf,
                          lds_pad=0, lds_swizzle=False, dtype=F16, wave_size=lc.GFX90A.WAVE)


def test_read_probe_refuses_the_geometry_that_overran_the_device_buffer():
    """tile_k=16 with a k_sub=2 read descriptor: the fragment addresses 32 rows of a 16-row window.
    This ran on real hardware before the guard existed."""
    with pytest.raises(ValueError, match="OVERRUN"):
        _build(_probe_descs(128, 16, 8), "read", 128, 16, 8, 64)


def test_probe_guard_catches_compensating_per_axis_errors():
    """A product check (rows*cols == elems) passes when two axes are wrong in opposite directions.
    The guard is per-axis for exactly this reason."""
    with pytest.raises(ValueError, match="addresses"):
        _build(_probe_descs(256, 16, 16), "read", 256, 16, 16, 128)


def test_probe_builds_at_a_matched_geometry():
    """Guard must not be a blanket refusal -- the measured geometry still builds."""
    assert _build(_probe_descs(256, 32, 16), "read", 256, 32, 16, 64) is not None
    assert _build(_probe_descs(256, 32, 16), "store", 256, 32, 16, 64) is not None


@pytest.mark.parametrize("bad", [0, -1, True, 2.0, "2"])
def test_dwords_per_lane_rejects_non_positive_ints(bad):
    """Without the type guard, 0 raises ZeroDivisionError from deep inside instead of saying why."""
    with pytest.raises(ValueError, match="positive int"):
        lc.analyze_read(_read_descs(2), tile_free=256, arch=lc.GFX90A, operand_label="A",
                        strides=(264, 1), dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                        dwords_per_lane=bad)


def test_a_failed_FIRST_registration_does_not_stick():
    """The mirror of the restore case: if the arch was not registered before, a failed attempt must
    leave it unregistered rather than half-on."""
    corpus = lc._VALIDATION_CORPUS["gfx90a"]
    saved_rows, saved_model = list(corpus["read_hists"]), lc._READ_MODELS.pop("gfx90a", None)
    try:
        corpus["read_hists"] = []
        with pytest.raises(lc.ConflictModelError):
            lc.register_read_model("gfx90a")
        assert "gfx90a" not in lc._READ_MODELS
    finally:
        corpus["read_hists"] = saved_rows
        if saved_model is not None:
            lc._READ_MODELS["gfx90a"] = saved_model


def test_register_arch_does_not_wipe_an_existing_read_corpus():
    """register_arch used to rewrite the corpus entry wholesale, silently deleting read_hists /
    read_provenance measured separately."""
    before = len(lc._VALIDATION_CORPUS["gfx90a"]["read_hists"])
    saved = dict(lc._VALIDATION_CORPUS["gfx90a"])
    try:
        lc.register_arch(lc.GFX90A, saved["hists"], saved["pad_sweep"])
        assert len(lc._VALIDATION_CORPUS["gfx90a"]["read_hists"]) == before
        assert "read_provenance" in lc._VALIDATION_CORPUS["gfx90a"]
    finally:
        lc._VALIDATION_CORPUS["gfx90a"] = saved


def test_broadcast_is_detected_and_pushes_the_read_out_of_envelope():
    """The broadcast arm of the envelope gate was completely untested -- two mutants (disabling the
    detection, and removing the arm) both survived. No shipped descriptor broadcasts, so drive it
    through the same code path with a datum that does."""
    import types
    real = lc.read_datum

    def broadcasting(*a, **k):
        acc, vw, datum = real(*a, **k)
        # put two lanes of half-wave 0 on the SAME dword at phase 0 -> a broadcast, not a conflict
        if (0, 0) in datum and (1, 0) in datum:
            datum[(1, 0)] = datum[(0, 0)]
        return acc, vw, datum

    lc.read_datum = broadcasting
    try:
        r = lc.analyze_read(_read_descs(2), tile_free=256, arch=lc.GFX90A, operand_label="A",
                            strides=(264, 1), dtype_name="f16", origin=(0, 0), lds_swizzle=False,
                            dwords_per_lane=2)
        assert not r.in_envelope, "a broadcasting read must not be priced"
        assert "broadcast" in r.out_of_envelope_reason
        assert r.sim is None
        with pytest.raises(lc.ConflictModelError):
            _ = r.conflicts_per_access
    finally:
        lc.read_datum = real


@pytest.mark.parametrize("rows,expect", [
    ([("d1", {1: 32}, 2, 128, 4, 0), ("d2", {2: 16}, 2, 128, 8, 4),
      ("d2b", {2: 20}, 2, 128, 8, 4), ("d4", {4: 8}, 2, 128, 16, 12),
      ("d8", {8: 4}, 2, 128, 32, 28)], "discriminate"),
    ([("d2", {2: 16}, 2, 128, 8, 4), ("d2nu", {1: 16, 2: 8}, 2, 128, 8, 4),
      ("d4", {4: 8}, 2, 128, 16, 12), ("d8", {8: 4}, 2, 128, 32, 28)], "depth==1"),
    ([("d1", {1: 32}, 2, 128, 4, 0), ("d2nu", {1: 16, 2: 8}, 2, 128, 8, 4)], "distinct max_depth"),
    ([("d1", {1: 32}, 2, 128, 4, 0), ("d2", {2: 16}, 2, 128, 8, 4),
      ("d2nu", {1: 16, 2: 8}, 2, 128, 8, 4), ("d4", {4: 8}, 2, 128, 16, 12),
      ("d8p4", {8: 4}, 4, 128, 32, 28)], "n_phases"),
])
def test_each_structural_minimum_is_separately_enforced(rows, expect, capsys):
    """Asserting only `not selftest(...)` lets the four minima mask each other -- three could be
    deleted and the test would stay green. Pin WHICH check fires for each corpus."""
    corpus = lc._VALIDATION_CORPUS["gfx90a"]
    saved = list(corpus["read_hists"])
    try:
        corpus["read_hists"] = rows
        assert not lc.selftest(lc.GFX90A, verbose=True)
        out = capsys.readouterr().out
        assert expect in out, f"expected the {expect!r} minimum to fire; got:\n{out}"
    finally:
        corpus["read_hists"] = saved


def test_footprint_is_gated_not_merely_documented():
    """`productive` is measured at exactly one value (every corpus row is 128 dwords), so a different
    footprint extrapolates the DENOMINATOR of conflicts/access. Documenting that in the SOT is not the
    same as gating it -- a gate that cannot fire is not a gate.

    Driven through the real path by making two lanes of DIFFERENT half-waves share a dword, which
    lowers the footprint WITHOUT tripping the broadcast check (that check is per-(half-wave, phase))."""
    descs = lc.ProbeDescs.from_coop(_macro_coop_descs_crc(256, 16, 16),
                                    _wave_descs_interleaved(2, 2, 1)[0], transpose=_transpose_desc)
    kw = dict(tile_free=256, arch=lc.GFX90A, operand_label="A", strides=(264, 1), dtype_name="f16",
              origin=(0, 0), lds_swizzle=False, dwords_per_lane=2)
    baseline = lc.analyze_read(descs, **kw)
    assert baseline.in_envelope and baseline.footprint_dwords == 128

    real = lc.read_datum

    def shrunk(*a, **k):
        acc, vw, datum = real(*a, **k)
        for ph in range(2):
            if (0, ph) in datum and (32, ph) in datum:
                datum[(32, ph)] = datum[(0, ph)]
        return acc, vw, datum

    lc.read_datum = shrunk
    try:
        r = lc.analyze_read(descs, **kw)
        assert r.footprint_dwords != 128
        assert not r.in_envelope, "a footprint the corpus never measured must not be priced"
        assert "footprint" in r.out_of_envelope_reason
        with pytest.raises(lc.ConflictModelError):
            _ = r.conflicts_per_access
    finally:
        lc.read_datum = real
