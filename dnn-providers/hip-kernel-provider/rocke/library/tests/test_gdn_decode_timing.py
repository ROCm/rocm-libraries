# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host tests (CPU tensors) of the GDN/KDA decode driver's shared timing and
gate helpers: the cold-memory rotation size, the warm-mode warning, and the
fp32 gate the sweep and benchmarks score every configuration with, through
every tool path that gates (``check``, the GDN benchmark, ``--bench``).

Needs torch (not a GPU): ``gdn_decode`` imports torch at module level and every
test builds CPU tensors, so the module skips as a whole where torch is missing.
What those tensors feed is torch-free and tested everywhere, in
``dispatch/gdn/test_gate_rules.py`` (the gate's pass test and the rotation
arithmetic) and in the sweep and benchmark tests (what a tool does with a
failing verdict); this module checks the driver's tensor code on real tensors."""

from __future__ import annotations

import dataclasses as dc
import math

import pytest

torch = pytest.importorskip("torch")

from builders.gfx950.gdn import gdn_decode as g  # noqa: E402
from kernels.gfx950.gdn_decode import GdnDecodeSpec  # noqa: E402

PLAIN = GdnDecodeSpec(num_k_heads=4, num_v_heads=8)
F32 = dc.replace(PLAIN, gate_kind="kda", state_dtype="f32")
FUSED = GdnDecodeSpec(
    num_k_heads=4,
    num_v_heads=4,
    num_warps=4,
    blocks_per_v_dim=1,
    fuse_conv=True,
    fuse_out_norm=True,
)


@pytest.mark.parametrize("spec", [PLAIN, F32, FUSED], ids=["bf16", "f32", "fused"])
@pytest.mark.parametrize("batch", [1, 5, 64])
@pytest.mark.parametrize("disjoint", [False, True])
def test_rotation_guarantees_what_the_label_prints(spec, batch, disjoint):
    """One cycle touches at least ``rotate_mb`` MiB, and a set is reused after
    at least ``rotate_mb`` MiB less one set: the two claims ``cold_copies``
    and the timing label make."""
    inp = g.make_inputs(spec, batch, device="cpu", disjoint_writes=disjoint)
    touched = g.touched_bytes(spec, inp, batch)
    for rotate_mb in (1, 16, 1024):
        copies = g.cold_copies(spec, inp, batch, rotate_mb)
        target = rotate_mb * 2**20
        assert copies >= 2
        assert copies * touched >= target
        assert (copies - 1) * touched >= target - touched
        # ...and not grossly more: one copy fewer would fall short (or 2 is the floor).
        assert copies == 2 or (copies - 1) * touched < target
    assert g.cold_copies(spec, inp, batch, 0) == 1
    with pytest.raises(ValueError, match="rotate_mb"):
        g.cold_copies(spec, inp, batch, -1)


def test_touched_bytes_counts_each_state_slot_once():
    batch = 6
    in_place = g.make_inputs(PLAIN, batch, device="cpu", disjoint_writes=False)
    disjoint = g.make_inputs(PLAIN, batch, device="cpu", disjoint_writes=True)
    page = in_place["state"][0].numel() * in_place["state"].element_size()
    out = batch * PLAIN.num_v_heads * PLAIN.head_v_dim * 2
    assert g.touched_bytes(PLAIN, in_place, batch) == batch * page + out
    assert g.touched_bytes(PLAIN, disjoint, batch) == 2 * batch * page + out


def test_touched_bytes_counts_the_conv_state_pages_of_a_fused_spec():
    batch = 6
    in_place = g.make_inputs(FUSED, batch, device="cpu", disjoint_writes=False)
    disjoint = g.make_inputs(FUSED, batch, device="cpu", disjoint_writes=True)
    state, conv = in_place["state"], in_place["conv_state"]
    pages = state[0].numel() * state.element_size()
    pages += conv[0].numel() * conv.element_size()
    out = batch * FUSED.num_v_heads * FUSED.head_v_dim * 2
    assert g.touched_bytes(FUSED, in_place, batch) == batch * pages + out
    assert g.touched_bytes(FUSED, disjoint, batch) == 2 * batch * pages + out


def test_value_sets_are_independent_copies():
    batch = 3
    inp = g.make_inputs(FUSED, batch, device="cpu")
    sets = g.cold_value_sets(FUSED, inp, batch, rotate_mb=1)
    assert len(sets) == g.cold_copies(FUSED, inp, batch, 1)
    for name, value in sets[0].items():
        if torch.is_tensor(value):
            assert value.data_ptr() != sets[1][name].data_ptr(), name
            if name in inp:
                assert value.data_ptr() != inp[name].data_ptr(), name
                assert torch.equal(value, inp[name]), name
    assert len(g.cold_value_sets(FUSED, inp, batch, rotate_mb=0)) == 1


@pytest.mark.parametrize("rotate_mb", [0, 1])
def test_cold_value_sets_reject_an_out_of_range_index(rotate_mb):
    """The first copy's validation covers the rest, so a bad index must be
    refused before any value set is returned (also with one warm copy)."""
    batch = 3
    inp = g.make_inputs(PLAIN, batch, device="cpu", disjoint_writes=False)
    inp["write_indices"][0] = inp["state"].shape[0]
    with pytest.raises(ValueError, match="out of range"):
        g.cold_value_sets(PLAIN, inp, batch, rotate_mb=rotate_mb)


@pytest.mark.parametrize("rotate_mb", [0, 1])
def test_cold_value_sets_reject_a_duplicated_write_index(rotate_mb):
    batch = 3
    inp = g.make_inputs(PLAIN, batch, device="cpu")
    inp["write_indices"][1] = inp["write_indices"][0]
    with pytest.raises(ValueError, match="unique across active"):
        g.cold_value_sets(PLAIN, inp, batch, rotate_mb=rotate_mb)


def test_warm_timing_says_so():
    assert g.timing_label(0).startswith("WARNING")
    assert "WARM" in g.timing_label(0)
    label = g.timing_label(1024)
    assert "cold memory" in label and "one cycle touches >= 1024 MiB" in label
    with pytest.raises(ValueError, match="rotate_mb"):
        g.timing_label(-1024)


def test_a_rotation_below_the_cold_target_is_not_labelled_cold():
    for rotate_mb in (1, 8, g.ROTATE_MB - 1):
        label = g.timing_label(rotate_mb)
        assert label.startswith("WARNING") and "partly warm" in label, rotate_mb
        assert "cold memory" not in label, rotate_mb
    assert "cold memory" in g.timing_label(g.ROTATE_MB)
    assert "cold memory" in g.timing_label(2 * g.ROTATE_MB)


def test_labels_claim_only_what_the_rotation_guarantees():
    """The label names the guarantee (reuse distance), says the caches are not
    measured, and mentions no flag only one tool has."""
    for rotate_mb in (0, 1, g.ROTATE_MB):
        label = g.timing_label(rotate_mb)
        assert "--no-device" not in label, rotate_mb
        assert "model a decode step" not in label, rotate_mb
        assert "reads HBM" not in label, rotate_mb
    cold = g.timing_label(g.ROTATE_MB)
    assert "reused only after" in cold and "not measured here" in cold
    assert "not comparable with cold ones" in g.timing_label(1)


def test_capture_cold_refuses_value_sets_of_another_batch():
    """The grid comes from ``batch`` and the kernel has no batch bound: a value
    set prepared for another batch must be refused before any launch."""
    sets = [{"batch_size": 4}, {"batch_size": 8}]
    with pytest.raises(ValueError, match="different batch"):
        g.capture_cold(PLAIN, 4, sets, launcher=object())


def _exact_launch(spec, inp):
    """What a correct kernel leaves: reference output, written slots updated,
    every other slot untouched."""
    ref = g.reference(spec, inp)
    written = inp["write_indices"].long()
    values = {"out": ref[0].to(torch.bfloat16), "state": inp["state"].clone()}
    values["state"][written] = ref[1].to(inp["state"].dtype)
    if spec.fuse_conv:
        values["conv_state"] = inp["conv_state"].clone()
        values["conv_state"][written] = ref[2].to(inp["conv_state"].dtype)
    return ref, values


@pytest.mark.parametrize("spec", [PLAIN, FUSED], ids=["plain", "fused"])
def test_gate_passes_a_correct_launch_and_rejects_nan_and_misplaced_writes(spec):
    batch = 4
    inp = g.make_inputs(spec, batch, device="cpu")
    ref, values = _exact_launch(spec, inp)
    assert g.gate(spec, inp, values, ref).passed

    nan = dict(values, out=values["out"].clone())
    nan["out"].view(-1)[0] = float("nan")
    verdict = g.gate(spec, inp, nan, ref)
    assert math.isnan(verdict.worst) and not verdict.passed

    # One ulp in a slot the kernel was not told to write: below any
    # tolerance, still a misplaced write.
    spare = int(inp["read_indices"][-1]) + batch + 1
    pools = ("state", "conv_state") if spec.fuse_conv else ("state",)
    for pool in pools:
        spilled = dict(values, **{pool: values[pool].clone()})
        bits = spilled[pool][spare].view(-1).view(torch.int16)
        bits[0] ^= 1  # the lowest mantissa bit of one element
        verdict = g.gate(spec, inp, spilled, ref)
        assert verdict.spill and verdict.worst <= g.TOL and not verdict.passed, pool

    wrong = dict(values, state=values["state"].clone())
    wrong["state"][inp["write_indices"].long()[0]] += 1.0
    assert not g.gate(spec, inp, wrong, ref).passed


def test_gate_reports_a_wrong_written_conv_state_slot_as_the_conv_error():
    """The conv state a fused kernel writes is scored against the reference
    like the state: a wrong value in a written slot fails the gate, and the
    reported cause is ``conv_err`` (the output and the state stay exact)."""
    spec, batch = FUSED, 4
    inp = g.make_inputs(spec, batch, device="cpu")
    ref, values = _exact_launch(spec, inp)
    good = g.gate(spec, inp, values, ref)
    assert good.passed and good.conv_err <= g.TOL
    wrong = dict(values, conv_state=values["conv_state"].clone())
    wrong["conv_state"][inp["write_indices"].long()[0]] += 1.0
    verdict = g.gate(spec, inp, wrong, ref)
    assert not verdict.passed and not verdict.spill
    assert verdict.conv_err > g.TOL
    assert verdict.out_err <= g.TOL and verdict.state_err <= g.TOL
    assert verdict.worst == verdict.conv_err


def _nan_state_launch(spec, inp, launcher, batch):
    """A ``run_values`` stand-in: the output is exact, one written state
    element is NaN. A finite output must not hide it."""
    _, values = _exact_launch(spec, inp)
    values["state"][inp["write_indices"].long()[0]].view(-1)[0] = float("nan")
    return values


@pytest.fixture
def nan_state_kernel(monkeypatch):
    """Every launch through the driver writes a NaN state with a finite output
    (on CPU tensors), and timing anything is a test failure."""
    real = g.make_inputs

    def cpu_inputs(spec, batch, seed=0, **kw):
        kw.pop("device", None)
        return real(spec, batch, seed=seed, device="cpu", **kw)

    def timed(*args, **kw):
        raise AssertionError("a configuration that failed the gate was timed")

    monkeypatch.setattr(g, "make_inputs", cpu_inputs)
    monkeypatch.setattr(g, "launcher_for", lambda spec, arch=None: object())
    monkeypatch.setattr(g, "run_values", _nan_state_launch)
    for name in ("device_us", "eager_us", "bench"):
        monkeypatch.setattr(g, name, timed)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)


def test_check_fails_a_nan_state_behind_a_finite_output(nan_state_kernel):
    verdict = g.check(PLAIN, 2)
    assert math.isfinite(verdict.out_err) and verdict.out_err <= g.TOL
    assert math.isnan(verdict.state_err) and not verdict.passed


def test_gdn_benchmark_fails_and_never_times_a_nan_state(nan_state_kernel, capsys):
    from benchmarks.gfx950.gdn import benchmark_gdn_decode

    assert benchmark_gdn_decode.main(["--batches", "1", "--no-device"]) == 1
    out = capsys.readouterr().out
    assert "FAIL max_err=nan" in out


def test_driver_bench_fails_and_never_times_a_nan_state(nan_state_kernel, capsys):
    assert g.main(["--batches", "1,2", "--bench"]) == 1
    out = capsys.readouterr().out
    assert out.count("state_err=nan") == 2 and "2 batches failed" in out


@pytest.mark.parametrize("batches", ["", "1,,2", "a", "1,x", "0", "-3,1"])
def test_driver_refuses_malformed_batches_as_a_usage_error(batches, capsys):
    """A bad ``--batches`` is a usage error (exit 2) naming the flag, not a
    ``ValueError`` traceback."""
    with pytest.raises(SystemExit) as err:
        g.main(["--batches", batches])
    assert err.value.code == 2
    assert "--batches" in capsys.readouterr().err
