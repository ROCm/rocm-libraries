# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests of the pure-Python parts of the skinny GEMM sweep: no GPU, no bindings."""

import os
import stat
import types

import numpy as np
import pytest
import yaml

import skinny_gemm_support as support
import sweep_skinny_gemm_config as config
import sweep_skinny_gemm_inputs as inputs_module
import sweep_skinny_gemm_smi as smi
import sweep_skinny_gemm_worker as worker

HERE = os.path.dirname(os.path.abspath(__file__))


def parse(text: str) -> config.SweepInput:
    return config.parse_input(yaml.safe_load(text))


def test_range_expansion():
    assert config.expand_range("r", {"min": 1, "max": 8, "step": 3}) == [1, 4, 7]
    assert config.expand_range("r", {"min": 1, "max": 16, "mul": 2}) == [1, 2, 4, 8, 16]
    with pytest.raises(config.SweepConfigError, match="exactly one"):
        config.expand_range("r", {"min": 1, "max": 8})
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        config.expand_range("r", {"min": 1, "max": 8, "stp": 1})


def test_unknown_keys_and_types_are_errors():
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweeep: {}}]")
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweep: {bn: 1}}]")
    with pytest.raises(config.SweepConfigError, match="unknown key"):
        parse("settings: {iter: 3}\nproblems: [{name: a, m: 1, n: 2, k: 3}]")
    with pytest.raises(config.SweepConfigError, match="expected true or false"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweep: {grid3d: 1}}]")
    with pytest.raises(config.SweepConfigError, match="expected an integer"):
        parse("problems: [{name: a, m: 1, n: 2, k: 3, sweep: {bm: true}}]")


def test_shapes_transpose_and_spk():
    text = (
        "problems: [{name: a, m: [1, 2], n: 8, k: 512, transpose: true, "
        "spk: [k, 256], useBeta: false}]"
    )
    cases = parse(text).cases
    assert {(c.m, c.n) for c in cases} == {(1, 8), (2, 8), (8, 1), (8, 2)}
    assert {c.spk for c in cases if (c.m, c.n) == (1, 8)} == {256, 512}


def test_ld_defaults_and_validation():
    case = parse("problems: [{name: a, m: 5, n: 8, k: 4100}]").cases[0]
    assert (case.ld_a, case.ld_d) == (4100, 16)
    with pytest.raises(config.SweepConfigError, match="multiple of 4"):
        parse("problems: [{name: a, m: 5, n: 8, k: 64, ldA: 66}]")


def test_sweep_list_merge_and_dedupe():
    text = """
defaults:
  sweep:
    - {bm: [1, 2], grid3d: true}
    - {bm: 1, grid3d: false, group: [1, 2], numXcds: 1, blocks: [4, 100000]}
problems:
  - {name: a, m: 4, n: 4096, k: 1024, sweep: {tn: [1, 2]}}
"""
    case = parse(text).cases[0]
    assert len(case.sweeps) == 2
    expanded = config.expand_case(case)
    # 2 bm * 2 tn on the 3-D grid; persistent: 2 groups * 2 tn * 2 capped blocks.
    assert len(expanded.problems) == 4 + 8
    persistent = [p for p in expanded.problems if not p.grid3d]
    tiles = {p.tn: support.countTiles(4, 4096, 1024, 1024, 1, p.bn) for p in persistent}
    assert all(p.blocks <= tiles[p.tn] for p in persistent)


def test_grid3d_forces_plain_order_and_duplicates_collapse():
    text = """
problems:
  - {name: a, m: 4, n: 4096, k: 1024,
     sweep: {lanesK: 64, group: [1, 2, 4], numXcds: [1, 8], grid3d: true}}
"""
    expanded = config.expand_case(parse(text).cases[0])
    assert len(expanded.problems) == 1
    assert expanded.problems[0].group == expanded.problems[0].xcds == 1


def test_constraint_failures_are_recorded():
    text = "problems: [{name: a, m: 4, n: 4096, k: 1024, sweep: {lanesK: [32, 64]}}]"
    expanded = config.expand_case(parse(text).cases[0])
    assert [f.stage for f in expanded.failures] == ["constraint"]
    assert expanded.swept == 2


def test_derived_knobs_in_record():
    text = (
        "problems: [{name: a, m: 1, n: 4096, k: 4100, "
        "sweep: {lanesK: 16, upScales: 1}}]"
    )
    record = config.config_record(config.expand_case(parse(text).cases[0]).problems[0])
    assert record["partialSteps"] == 1 and record["upIters"] == 5


def test_default_inputs_parse():
    for name in ("sweep_skinny_gemm_shapes.yaml", "sweep_skinny_gemm_smoke.yaml"):
        assert config.load_input(os.path.join(HERE, name)).cases


def fake_smi(directory, fail: bool) -> str:
    """A rocm-smi that logs its arguments; with fail every set request fails."""
    log = os.path.join(directory, "calls.log")
    script = os.path.join(directory, "rocm-smi")
    with open(script, "w") as stream:
        stream.write(
            f"""#!/bin/sh
echo "$@" >> {log}
case "$*" in
  *showperflevel*) echo '{{"card1": {{"Performance Level": "auto"}}}}'; exit 0;;
  *showsclkrange*) echo '{{"card1": {{"Valid sclk range": "500Mhz - 2400Mhz"}}}}'
    exit 0;;
esac
{"echo 'sudo: a password is required' >&2; exit 1" if fail else "exit 0"}
"""
        )
    os.chmod(script, os.stat(script).st_mode | stat.S_IEXEC)
    return log


def calls(log: str) -> list[str]:
    with open(log) as stream:
        return stream.read().splitlines()


def test_stabilizer_restores_only_successful_locks(tmp_path, monkeypatch):
    log = fake_smi(str(tmp_path), fail=False)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with smi.GpuStabilizer([1], require_stable=False) as stabilizer:
        assert stabilizer.summary()["status"] == "locked"
    assert "-d 1 --setperfdeterminism 2400" in calls(log)
    assert "-d 1 --resetperfdeterminism" in calls(log)


def test_stabilizer_failed_lock_restores_nothing(tmp_path, monkeypatch):
    log = fake_smi(str(tmp_path), fail=True)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    with smi.GpuStabilizer([1], require_stable=False) as stabilizer:
        assert stabilizer.summary()["status"] == "unavailable"
    assert not any("reset" in call for call in calls(log))
    assert not any(call.endswith("--setperflevel auto") for call in calls(log))
    with pytest.raises(smi.SweepAbort):
        with smi.GpuStabilizer([1], require_stable=True):
            pass


def small_problem(spk: int | None, use_beta: bool, beta: float) -> support.Problem:
    case = parse(
        f"""
problems:
  - name: a
    m: 5
    n: 300
    k: 1000
    spk: {"k" if spk is None else spk}
    useBeta: {str(use_beta).lower()}
    alpha: 0.75
    beta: {beta}
    sweep: {{bm: 2, tn: 1, lanesK: 16, waves: 1, prefetch: 1, upScales: 0}}
"""
    ).cases[0]
    return config.expand_case(case).problems[0]


def make_verifier(p: support.Problem, full: bool) -> inputs_module.Verifier:
    cache = inputs_module.OperandCache()
    return inputs_module.Verifier(inputs_module.make_case_inputs(p, cache), full)


def d_bits_of(p, expected: np.ndarray) -> np.ndarray:
    """D bits holding the bf16 truncation of expected [m, n] in the layout of D."""
    bits, _ = support.makeOutputs(p.m, p.n, p.ldD, 1)
    index = np.arange(p.n)[:, None] * p.ldD + np.arange(p.m)
    bits[index] = support.truncateToBf16(expected.T)
    return bits


def split_workspace(p, verifier) -> np.ndarray:
    """The workspace [splits, n, m] of exact per-split partial products."""
    a = verifier.inputs.a.dequant(slice(None)).astype(np.float64)
    b = verifier.inputs.b.dequant(slice(None)).astype(np.float64)
    ws = np.empty((p.splits, p.n, p.m), dtype=np.float32)
    for split in range(p.splits):
        low, high = split * p.spk, min(p.k, (split + 1) * p.spk)
        ws[split] = (a[:, low:high] @ b[:, low:high].T).T
    return ws.ravel()


def test_dense_product_matches_support_reference():
    a = inputs_module.generate_operand(3, 4100, 4100, 1, 0)
    b = inputs_module.generate_operand(300, 4100, 4104, 2, 1)
    codes_a, codes_b = a.codes().T, b.codes().T
    expected = support.referenceProduct(
        support.dequant(codes_a, a.scales), support.dequant(codes_b, b.scales)
    )
    np.testing.assert_allclose(
        inputs_module.dense_product(a, b), expected, rtol=1e-4, atol=1e-2
    )
    assert (a.raw.size, b.swizzled.size) == (
        support.operandExtent(3, 4100, 4100),
        support.scaleExtent(300, 4100),
    )


@pytest.mark.parametrize("full", [True, False])
@pytest.mark.parametrize("use_beta, beta", [(False, 0.0), (True, 0.5)])
def test_verifier_single_split(full, use_beta, beta):
    p = small_problem(None, use_beta, beta)
    verifier = make_verifier(p, full)
    dense = make_verifier(p, True)
    expected = support.singleSplitD(p, dense.support_inputs())
    d_bits = d_bits_of(p, expected)
    verifier.check_gemm(p, d_bits, np.zeros(1, dtype=np.float32))
    count = len(verifier.sample(p)[0])
    assert verifier.label(p) == ("full" if full else f"sampled({count})")
    rows, cols = verifier.sample(p)
    corrupt = d_bits.copy()
    corrupt[cols[3] * p.ldD + rows[3]] ^= 0x4000
    with pytest.raises(AssertionError):
        verifier.check_gemm(p, corrupt, np.zeros(1, dtype=np.float32))
    padded = d_bits.copy()
    padded[p.m] = 0
    with pytest.raises(AssertionError, match="padding"):
        verifier.check_gemm(p, padded, np.zeros(1, dtype=np.float32))


@pytest.mark.parametrize("full", [True, False])
def test_verifier_split_k_workspace_and_reduce(full):
    p = small_problem(256, True, 0.5)
    assert p.splits == 4
    verifier = make_verifier(p, full)
    ws = split_workspace(p, verifier)
    d_unused = np.zeros(1, dtype=np.uint16)
    verifier.check_gemm(p, d_unused, ws)
    rows, cols = verifier.sample(p)
    corrupt = ws.copy()
    # The last split, which holds the tail: only its own slice is wrong.
    corrupt[3 * p.m * p.n + cols[5] * p.m + rows[5]] += 100.0
    with pytest.raises(AssertionError, match=None if full else "split 3"):
        verifier.check_gemm(p, d_unused, corrupt)
    expected = support.expectedD(p, make_verifier(p, True).support_inputs())
    d_bits = d_bits_of(p, expected)
    verifier.check_reduced(p, d_bits)
    d_bits[cols[7] * p.ldD + rows[7]] ^= 0x4000
    with pytest.raises(AssertionError):
        verifier.check_reduced(p, d_bits)


def test_sample_positions_cover_ends_and_tile_boundaries():
    p = small_problem(None, False, 0.0)
    rows, cols = inputs_module.sample_positions(p)
    assert set(rows.tolist()) == set(range(p.m))
    assert {0, p.n - 1, p.bn - 1, p.bn} <= set(cols.tolist())
    assert (rows < p.m).all() and (cols < p.n).all()


def test_operand_generation_is_deterministic_and_cached():
    cache = inputs_module.OperandCache()
    first = cache.get(inputs_module.ROLE_A, 70, 520, 524)
    assert cache.get(inputs_module.ROLE_A, 70, 520, 524) is first
    again = inputs_module.generate_operand(
        70, 520, 524, inputs_module.INPUT_SEED, inputs_module.ROLE_A
    )
    assert (first.raw == again.raw).all() and (first.swizzled == again.swizzled).all()
    assert not np.isnan(support.decodeE4M3FN(first.codes())).any()


def test_wrong_result_is_dropped_and_next_promoted(monkeypatch):
    p = config.expand_case(
        parse(
            "problems: [{name: a, m: 4, n: 64, k: 256, sweep: {tn: [1, 2, 4]}}]"
        ).cases[0]
    ).problems
    job = worker.Job(
        0,
        [(support.gemmKernelName(q), q) for q in p],
        "",
        {"top": 2, "verify_full_max_elems": 10**9},
        True,
    )

    def check(job, buffers, problem, name, verifier):
        assert problem.tn != 1, "D mismatch"

    monkeypatch.setattr(worker, "check_result", check)
    buffers = types.SimpleNamespace(inputs=None)
    timings = {i: worker.Timing([1000 * (i + 1)], []) for i in range(3)}
    outcome = {"ranked": [], "failures": []}
    worker.pick_winners(job, buffers, timings, timings, outcome)
    assert [w["index"] for w in outcome["ranked"]] == [1, 2]
    assert outcome["failures"][0]["status"] == "wrong result"
    assert all(w["verified"] == "full" for w in outcome["ranked"])
    job.verify = False
    outcome = {"ranked": [], "failures": []}
    worker.pick_winners(job, buffers, timings, timings, outcome)
    assert {w["verified"] for w in outcome["ranked"]} == {"no"}


def test_a_and_b_differ_when_shape_and_ld_are_equal():
    cache = inputs_module.OperandCache()
    a = cache.get(inputs_module.ROLE_A, 64, 512, 512)
    b = cache.get(inputs_module.ROLE_B, 64, 512, 512)
    assert a is not b
    assert (a.raw != b.raw).any() and (a.scales != b.scales).any()
    again = inputs_module.OperandCache().get(inputs_module.ROLE_B, 64, 512, 512)
    assert (again.raw == b.raw).all() and (again.swizzled == b.swizzled).all()
    square = config.expand_case(
        parse("problems: [{name: a, m: 64, n: 64, k: 512}]").cases[0]
    ).problems[0]
    assert square.m == square.n and square.ldA == square.ldB
    inputs = inputs_module.make_case_inputs(square, cache)
    assert (inputs.a.raw != inputs.b.raw).any()


def alpha_beta(text: str, seed: int | None = None) -> dict:
    """(alpha, beta) of every case, by problem and beta mode."""
    cases = config.parse_input(yaml.safe_load(text), seed).cases
    return {
        (c.name, c.m, c.n, c.k, c.spk, c.use_beta, c.beta_mode): (c.alpha, c.beta)
        for c in cases
    }


SHAPES = "{name: a, m: [1, 2, 4], n: [64, 128], k: 512, transpose: true, useBeta: true}"


def test_random_alpha_beta_are_deterministic_and_independent_of_order():
    forward = alpha_beta(f"problems: [{SHAPES}]")
    assert forward == alpha_beta(f"problems: [{SHAPES}]")
    assert alpha_beta(f"problems: [{SHAPES}]", seed=7) != forward
    # Entries listed in another order, alone or with other entries around.
    extra = "{name: b, m: 3, n: 64, k: 512}"
    shuffled = alpha_beta(f"problems: [{extra}, {SHAPES}]")
    assert {key: shuffled[key] for key in forward} == forward
    alone = "problems: [{name: a, m: 4, n: 128, k: 512, useBeta: true}]"
    for key, value in alpha_beta(alone).items():
        assert forward[key] == value


def test_random_alpha_beta_ranges():
    text = (
        "problems: [{name: a, m: [1, 2, 3, 4, 5, 6, 7, 8], n: [64, 96, 128, 160], "
        "k: 512, useBeta: true}]"
    )
    values = alpha_beta(text, seed=1)
    alphas = [a for a, _ in values.values()]
    betas = [b for (*_, mode), (_, b) in values.items() if mode == "random"]
    assert all(0.25 <= a <= 2.0 and abs(a - 1) >= 0.05 for a in alphas)
    assert all(0.25 <= abs(b) <= 2.0 and abs(abs(b) - 1) >= 0.05 for b in betas)
    assert any(b < 0 for b in betas) and any(b > 0 for b in betas)


def test_default_covers_both_use_beta_and_both_beta_modes():
    values = alpha_beta("problems: [{name: a, m: 4, n: 64, k: 512}]")
    # The kernel that never reads C runs once; the useBeta kernel in two modes.
    assert sorted((key[5], key[6]) for key in values) == [
        (False, "none"),
        (True, "random"),
        (True, "zero"),
    ]
    zero = values[("a", 4, 64, 512, 512, True, "zero")]
    random = values[("a", 4, 64, 512, 512, True, "random")]
    assert zero[1] == 0.0 and random[1] not in (0.0, 1.0)
    # The modes of a problem share alpha, which is drawn once per problem.
    assert zero[0] == random[0]
    assert values[("a", 4, 64, 512, 512, False, "none")][1] == 0.0


def test_split_k_makes_beta_matter_without_use_beta():
    text = "problems: [{name: a, m: 4, n: 64, k: 512, spk: 256, useBeta: false}]"
    values = alpha_beta(text)
    assert sorted(key[6] for key in values) == ["random", "zero"]


def test_beta_modes_override_and_pinned_beta():
    only_zero = "{name: a, m: 4, n: 64, k: 512, useBeta: true, betaModes: [zero]}"
    assert [k[6] for k in alpha_beta(f"problems: [{only_zero}]")] == ["zero"]
    pinned = "{name: a, m: 4, n: 64, k: 512, alpha: 0.5, beta: 0.5}"
    values = alpha_beta(f"problems: [{pinned}]")
    assert values[("a", 4, 64, 512, 512, True, "pinned")] == (0.5, 0.5)
    assert values[("a", 4, 64, 512, 512, False, "pinned")] == (0.5, 0.5)
    assert len(values) == 2
    zero = "{name: a, m: 4, n: 64, k: 512, useBeta: true, beta: 0.0}"
    zeros = alpha_beta(f"problems: [{zero}]")
    assert list(zeros) == [("a", 4, 64, 512, 512, True, "zero")]
    assert list(zeros.values())[0][1] == 0.0
    ranged = (
        "{name: a, m: 4, n: 64, k: 512, useBeta: true, "
        "beta: {random: {min: 3, max: 4}}}"
    )
    values = alpha_beta(f"problems: [{ranged}]")
    assert values[("a", 4, 64, 512, 512, True, "zero")][1] == 0.0
    assert 3 <= abs(values[("a", 4, 64, 512, 512, True, "random")][1]) <= 4


def test_alpha_beta_spec_errors():
    base = "problems: [{name: a, m: 4, n: 64, k: 512, beta: %s}]"
    bads = ("{rnd: 1}", "{random: {min: 0, max: 1}}", "{random: {min: 2, max: 1}}")
    for bad in (*bads, "fixed"):
        with pytest.raises(config.SweepConfigError):
            config.parse_input(yaml.safe_load(base % bad))


def test_resolved_alpha_beta_are_in_the_problem_record():
    case = config.parse_input(yaml.safe_load(f"problems: [{SHAPES}]")).cases[0]
    record = case.record()
    assert (record["alpha"], record["beta"]) == (case.alpha, case.beta)
    assert record["betaMode"] == case.beta_mode == "zero"
    assert case.alpha != 1.0 and case.beta == 0.0
    random = config.parse_input(yaml.safe_load(f"problems: [{SHAPES}]")).cases[1]
    assert random.beta_mode == "random" and random.beta != 0.0


def test_beta_zero_poisons_c():
    cache = inputs_module.OperandCache()
    p = small_problem(None, True, 0.0)
    c = support.bf16ToFloat32(inputs_module.make_case_inputs(p, cache).c_bits)
    index = np.arange(p.n)[:, None] * p.ldC + np.arange(p.m)
    assert np.isnan(c[index]).all()
