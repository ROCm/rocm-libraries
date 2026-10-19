# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only contract for the optional unified split-KV (3D) LSE output."""

import importlib
import sys
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest

from kernels.common import attention_unified as au
from rocke.core.lower_llvm import lower_kernel_to_llvm

_ARCHES = ("gfx942", "gfx950")


def _reduce_module(arch):
    return importlib.import_module(f"kernels.{arch}.attention_tiled_3d")


def _reduce_spec(arch, **overrides):
    mod = _reduce_module(arch)
    fields = dict(
        head_size=128,
        num_query_heads=32,
        num_kv_heads=8,
        dtype="bf16",
        num_segments=16,
    )
    fields.update(overrides)
    return mod.UnifiedAttentionReduceTiledSpec(**fields)


def _reduce_ir(arch, spec):
    mod = _reduce_module(arch)
    kernel = mod.build_unified_attention_reduce_tiled(spec, arch=arch)
    return lower_kernel_to_llvm(kernel, arch=arch)


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_reduce_spec_rejects_non_boolean_emit_lse(arch, bad):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        _reduce_spec(arch, emit_lse=bad)


@pytest.mark.parametrize("arch", _ARCHES)
def test_reduce_lse_is_a_distinct_kernel_with_one_more_pointer(arch):
    off = _reduce_spec(arch)
    on = _reduce_spec(arch, emit_lse=True)
    assert on.kernel_name() == off.kernel_name() + "_lse"
    off_ir = _reduce_ir(arch, off)
    on_ir = _reduce_ir(arch, on)
    assert "lse_ptr" not in off_ir
    assert "lse_ptr" in on_ir
    assert "@llvm.log2.f32" in on_ir


def _problem(**overrides):
    """Decode over a long paged context: routes to split-KV (3D)."""
    fields = dict(
        total_q=4,
        num_seqs=4,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        block_size=16,
        max_seqlen_q=1,
        max_seqlen_k=4096,
        dtype="bf16",
    )
    fields.update(overrides)
    return au.UnifiedAttentionProblem(**fields)


@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_problem_rejects_non_boolean_emit_lse(bad):
    with pytest.raises(ValueError, match="emit_lse must be a bool"):
        _problem(emit_lse=bad)


@pytest.mark.parametrize("arch", _ARCHES)
def test_split_kv_gate_admits_lse(arch):
    ok, why = au.supports_native_unified_attention_3d_tiled(
        _problem(emit_lse=True), arch
    )
    assert ok, why


@pytest.mark.parametrize("arch", _ARCHES)
def test_tiled_2d_gate_declines_lse(arch):
    ok, why = au.supports_native_unified_attention_tiled(_problem(emit_lse=True), arch)
    assert not ok and "LSE" in why


def test_split_kv_gate_declines_lse_off_cdna3_cdna4():
    ok, why = au.supports_native_unified_attention_3d_tiled(
        _problem(emit_lse=True), "gfx1250"
    )
    assert not ok and "LSE" in why


@pytest.mark.parametrize("arch", _ARCHES)
def test_split_kv_gate_declines_lse_with_sinks(arch):
    ok, why = au.supports_native_unified_attention_3d_tiled(
        _problem(emit_lse=True, use_sinks=True), arch
    )
    assert not ok and "sink" in why


class _BypassSpec:
    """Minimal stand-in for an explicit tuning spec that sets the
    ``allow_unsupported`` escape hatch."""

    allow_unsupported = True


@pytest.mark.parametrize("arch", _ARCHES)
def test_allow_unsupported_cannot_bypass_lse_eligibility(arch):
    """allow_unsupported skips coverage gates, never the LSE output contract:
    with sinks, keyless rows would silently drop the sink."""
    ok, why = au._explicit_path_supported(
        _problem(emit_lse=True, use_sinks=True), _BypassSpec(), "3d", arch
    )
    assert not ok and "sink" in why
    ok, why = au._explicit_path_supported(
        _problem(emit_lse=True), _BypassSpec(), "2d", arch
    )
    assert not ok and "LSE" in why
    # The bypass still works for everything else.
    ok, _ = au._explicit_path_supported(
        _problem(use_sinks=True), _BypassSpec(), "3d", arch
    )
    assert ok


@pytest.mark.parametrize("arch", _ARCHES)
def test_cache_key_distinguishes_lse_and_keeps_default(arch):
    off = au._tiled_3d_cache_key(_problem(), arch)
    on = au._tiled_3d_cache_key(_problem(emit_lse=True), arch)
    assert on[: len(off)] == off and on[len(off) :] == ("lse",)


def test_reduce_signature_appends_lse_last():
    off = [p["name"] for p in au._reduce_signature("bf16")]
    on = [p["name"] for p in au._reduce_signature("bf16", emit_lse=True)]
    assert on == off + ["lse_ptr"]


def test_validator_contract_without_gpu():
    with pytest.raises(ValueError, match="caller-owned lse"):
        au.validate_unified_lse(
            _problem(emit_lse=True),
            None,
            q=None,
            k=None,
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            block_table=None,
        )
    with pytest.raises(ValueError, match="requires problem.emit_lse"):
        au.validate_unified_lse(
            _problem(),
            object(),
            q=None,
            k=None,
            v=None,
            out=None,
            cu_seqlens_q=None,
            seqused_k=None,
            block_table=None,
        )


# ---------------------------------------------------------------------------
# Public runner (run_unified_attention_torch) with the launch mocked out: what
# reaches the reduce launch, and which requests are refused before any launch.
# The buffer validator has its own tests above; it is stubbed here so the
# routing contract is checked without a GPU.
# ---------------------------------------------------------------------------


@pytest.fixture
def runner(monkeypatch):
    state = SimpleNamespace(red_vals=None)

    def empty(shape, *, dtype, device):
        return SimpleNamespace(shape=shape, device=device)

    torch = SimpleNamespace(
        float32="float32",
        empty=empty,
        cuda=SimpleNamespace(
            device=lambda device: nullcontext(),
            stream=lambda stream: nullcontext(),
            ExternalStream=lambda stream, device: SimpleNamespace(cuda_stream=stream),
            current_stream=lambda device=None: SimpleNamespace(cuda_stream=17),
            is_current_stream_capturing=lambda: False,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(au, "validate_unified_lse", lambda *a, **k: None)
    monkeypatch.setattr(au, "_resolve_attention_arch", lambda: "gfx942")
    monkeypatch.setattr(au, "_enable_3d_graph_replay", lambda problem, arch: False)
    monkeypatch.setattr(au, "_get_3d_pipeline", lambda *a, **k: SimpleNamespace())

    def launch(prepared, seg_vals, red_vals, stream, *, capturing):
        state.red_vals = dict(red_vals)

    monkeypatch.setattr(au, "_launch_3d_pipeline", launch)

    # Any 2D or scalar launch for an LSE request is a silent output drop.
    def never(*a, **k):
        raise AssertionError("an LSE request reached a path that writes no LSE")

    for name in ("_run_2d_graphed", "_get_2d_launcher", "_get_scalar_launcher"):
        monkeypatch.setattr(au, name, never)
    # Make the 2D graph fast path eligible, so a refusal placed after it fails.
    monkeypatch.setattr(au, "_enable_2d_graph_replay", lambda problem, arch: True)
    return state


def _run(problem, **overrides):
    args = dict(
        problem=problem,
        q=SimpleNamespace(device="cuda:0"),
        k=SimpleNamespace(shape=(64,)),
        v=object(),
        out=object(),
        cu_seqlens_q=object(),
        seqused_k=object(),
        softmax_scale=0.125,
        block_table=SimpleNamespace(stride=lambda dim: 256),
        softcap=0.0,
    )
    args.update(overrides)
    return au.run_unified_attention_torch(**args)


def test_runner_passes_the_lse_buffer_to_the_reduce_launch(runner):
    lse = object()
    _run(_problem(emit_lse=True), lse=lse)
    assert runner.red_vals["lse_ptr"] is lse


def test_runner_without_lse_launches_the_unchanged_reduce(runner):
    _run(_problem())
    assert "lse_ptr" not in runner.red_vals


@pytest.mark.parametrize("backend", ["auto", "tiled"])
def test_runner_refuses_lse_on_a_2d_routed_shape_before_any_2d_path(runner, backend):
    prefill = _problem(total_q=256, num_seqs=1, max_seqlen_q=256, max_seqlen_k=256)
    assert prefill.select_path() == "2d"
    with pytest.raises(NotImplementedError, match="LSE"):
        _run(replace(prefill, emit_lse=True), lse=object(), backend=backend)
    assert runner.red_vals is None


@pytest.mark.parametrize(
    "top_left,hinted", [(False, True), (True, False)], ids=["servable", "mask"]
)
def test_auto_refusal_names_backend_3d_only_when_split_kv_serves(
    runner, top_left, hinted
):
    """``auto`` routes this prefill to 2D, which writes no LSE. The refusal
    points at backend='3d' only when split-KV accepts the problem: gfx942
    split-KV declines the top-left mask, so there the hint would be wrong."""
    prefill = _problem(
        total_q=256,
        num_seqs=1,
        max_seqlen_q=256,
        max_seqlen_k=256,
        emit_lse=True,
        causal_top_left=top_left,
    )
    assert prefill.select_path() == "2d"
    with pytest.raises(NotImplementedError, match="LSE") as refused:
        _run(prefill, lse=object())
    assert ("backend='3d'" in str(refused.value)) is hinted
    assert runner.red_vals is None


def test_runner_reports_why_split_kv_declined_an_lse_request(runner):
    """``auto`` routes this decode to split-KV, whose gfx942 gate declines the
    top-left mask. The refusal must name that reason, not blame the 2D path."""
    top_left = _problem(emit_lse=True, causal_top_left=True)
    assert top_left.select_path() == "3d"
    with pytest.raises(NotImplementedError, match="split-KV.*top-left") as refused:
        _run(top_left, lse=object())
    assert "2D" not in str(refused.value)
    assert runner.red_vals is None


def test_runner_refuses_lse_for_strided_kv(runner):
    with pytest.raises(NotImplementedError, match="strided"):
        _run(_problem(emit_lse=True), lse=object(), kv_layout="strided")
    assert runner.red_vals is None


def _tuning_spec(*, reduce_emit_lse):
    spec = SimpleNamespace(
        arch="gfx942",
        path="3d",
        kernel_spec=SimpleNamespace(num_segments=16),
        reduce_spec=SimpleNamespace(emit_lse=reduce_emit_lse),
        compile_backend="llvm",
        allow_unsupported=False,
        cache_key=lambda: ("tuning", "gfx942"),
        build=lambda arch: None,
        launch_grid=lambda problem: (1, 1, 1),
        launch_block=lambda: (64, 1, 1),
    )
    spec.with_num_kv_blocks = lambda n: spec
    return spec


@pytest.mark.parametrize(
    "problem_lse,spec_lse", [(True, False), (False, True)], ids=["drop", "extra"]
)
def test_runner_rejects_a_tuning_spec_that_disagrees_on_lse(
    runner, problem_lse, spec_lse
):
    """drop: the spec's reduce writes no LSE for an LSE request. extra: the
    spec's reduce takes an lse_ptr the launch would not bind."""
    with pytest.raises(ValueError, match="disagrees with problem.emit_lse"):
        _run(
            _problem(emit_lse=problem_lse),
            lse=object() if problem_lse else None,
            tuning_spec=_tuning_spec(reduce_emit_lse=spec_lse),
        )
    assert runner.red_vals is None


def test_runner_refuses_lse_for_an_explicit_2d_tuning_spec(runner):
    """A 2D spec cannot write LSE: refused as NotImplementedError, the same
    error the backend paths raise, not a spec/problem-disagreement ValueError."""
    spec = _tuning_spec(reduce_emit_lse=False)
    spec.path, spec.reduce_spec = "2d", None
    with pytest.raises(NotImplementedError, match="LSE"):
        _run(_problem(emit_lse=True), lse=object(), tuning_spec=spec)
    assert runner.red_vals is None


def test_runner_rejects_a_strided_tuning_spec_whose_reduce_emits_lse(runner):
    """The strided branch returns before the paged LSE agreement check: an
    LSE-off strided launch with a reduce built to take lse_ptr must be refused,
    not launched with an argument list one pointer short."""
    problem = _problem()
    segment, reduce = au._strided_3d_specs_from_problem(problem, arch="gfx942")
    spec = _tuning_spec(reduce_emit_lse=False)
    spec.kernel_spec = segment
    spec.reduce_spec = replace(reduce, emit_lse=True)
    with pytest.raises(ValueError, match="strided reduce_spec cannot emit LSE"):
        _run(problem, kv_layout="strided", block_table=None, tuning_spec=spec)
    assert runner.red_vals is None
