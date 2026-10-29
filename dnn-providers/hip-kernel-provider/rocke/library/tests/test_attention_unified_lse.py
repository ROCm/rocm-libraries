# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-only contract for the optional unified (split-KV and tiled 2D) LSE output."""

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


def _prefill_problem(arch, **overrides):
    """Single-sequence causal prefill (Sq = Sk = 256, D128 bf16, GQA 32/8).

    ``select_path()`` routes it to tiled 2D, and the ``arch`` tiled gate serves
    it with LSE off, so any LSE refusal seen in a test is the LSE rule's."""
    problem = _problem(
        total_q=256, num_seqs=1, max_seqlen_q=256, max_seqlen_k=256, **overrides
    )
    assert problem.select_path() == "2d"
    ok, why = au.supports_native_unified_attention_tiled(
        replace(problem, emit_lse=False), arch
    )
    assert ok, why
    return problem


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
def test_tiled_2d_gate_admits_lse_for_prefill(arch):
    ok, why = au.supports_native_unified_attention_tiled(
        _prefill_problem(arch, emit_lse=True), arch
    )
    assert ok, why


@pytest.mark.parametrize("arch", _ARCHES)
def test_tiled_2d_gate_declines_lse_with_sinks(arch):
    ok, why = au.supports_native_unified_attention_tiled(
        _prefill_problem(arch, emit_lse=True, use_sinks=True), arch
    )
    assert not ok
    assert (
        "LSE output with attention sinks is not implemented on unified attention" in why
    )


def test_tiled_2d_gate_declines_lse_off_cdna3_cdna4():
    ok, why = au.supports_native_unified_attention_tiled(
        _problem(
            total_q=256, num_seqs=1, max_seqlen_q=256, max_seqlen_k=256, emit_lse=True
        ),
        "gfx1250",
    )
    assert not ok and "LSE output is not implemented on gfx1250" in why


@pytest.mark.parametrize("arch", _ARCHES)
def test_scalar_gate_refuses_lse(arch):
    """The scalar kernel writes no LSE: it must never be the auto tail for one."""
    problem = _prefill_problem(arch)
    ok, why = au.supports_native_unified_attention(problem, arch)
    assert ok, why
    ok, why = au.supports_native_unified_attention(
        replace(problem, emit_lse=True), arch
    )
    assert not ok and "scalar 2D" in why


def test_gfx942_4warp_cohort_is_ineligible_when_lse_is_on():
    """D256 bf16 causal prefill is the 4-warp GQA cohort with LSE off. With LSE
    on it falls to the default gfx942 builder, which writes LSE."""
    d256 = _problem(
        total_q=256, num_seqs=1, max_seqlen_q=256, max_seqlen_k=256, head_size=256
    )
    assert au._gfx942_4warp_route(d256, "gfx942") is not None
    assert au._gfx942_4warp_route(replace(d256, emit_lse=True), "gfx942") is None
    swa = _problem(
        total_q=256,
        num_seqs=1,
        max_seqlen_q=256,
        max_seqlen_k=256,
        sliding_window=64,
    )
    assert au._gfx942_4warp_route(swa, "gfx942") is not None
    assert au._gfx942_4warp_route(replace(swa, emit_lse=True), "gfx942") is None


def test_gfx942_4warp_builder_refuses_an_lse_spec():
    """The last gate: a spec built directly (no runner, no dispatcher) with
    emit_lse=True must not become a 4-warp kernel that has no LSE store."""
    from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa

    d256 = _problem(
        total_q=256, num_seqs=1, max_seqlen_q=256, max_seqlen_k=256, head_size=256
    )
    spec = au._tiled_spec_from_problem(d256, "gfx942")
    assert "lse" not in build_gfx942_4warp_gqa(spec, arch="gfx942").name
    with pytest.raises(NotImplementedError, match="4-warp GQA builder"):
        build_gfx942_4warp_gqa(replace(spec, emit_lse=True), arch="gfx942")


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
        _problem(emit_lse=True, use_sinks=True), _BypassSpec(), "2d", arch
    )
    assert not ok and "sink" in why
    ok, why = au._explicit_path_supported(
        _problem(emit_lse=True), _BypassSpec(), "2d", "gfx1250"
    )
    assert not ok and "gfx1250" in why
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


@pytest.mark.parametrize("arch", _ARCHES)
def test_tiled_spec_carries_lse_only_when_requested(arch):
    off = au._tiled_spec_from_problem(_prefill_problem(arch), arch)
    on = au._tiled_spec_from_problem(_prefill_problem(arch, emit_lse=True), arch)
    assert off.emit_lse is False and on.emit_lse is True
    assert replace(on, emit_lse=False) == off
    assert on.kernel_name() == off.kernel_name() + "_lse"


@pytest.mark.parametrize("arch", _ARCHES)
def test_tiled_cache_key_keeps_lse_off_and_suffixes_lse_on(arch):
    off = au._tiled_cache_key(_prefill_problem(arch), arch)
    on = au._tiled_cache_key(_prefill_problem(arch, emit_lse=True), arch)
    assert off[-1] != "lse"
    assert on == off + ("lse",)


def test_tiled_cache_key_rekeys_the_gfx942_4warp_cohort_under_lse():
    """D256 bf16 prefill is a 4-warp cohort with LSE off; with LSE on the
    default builder serves it, so resolved entries change, not just the suffix."""
    d256 = _problem(
        total_q=256, num_seqs=1, max_seqlen_q=256, max_seqlen_k=256, head_size=256
    )
    off = au._tiled_cache_key(d256, "gfx942")
    on = au._tiled_cache_key(replace(d256, emit_lse=True), "gfx942")
    assert on[-1] == "lse" and on[:-1] != off


def test_attn_signature_appends_lse_last():
    kw = dict(include_bt_stride=True, include_qq_bias_stride=True)
    off = [p["name"] for p in au._attn_signature("bf16", **kw)]
    on = [p["name"] for p in au._attn_signature("bf16", emit_lse=True, **kw)]
    assert on == off + ["lse_ptr"]
    assert off[-1] == "qq_bias_stride_0"
    # The additive-bias params come after qq_bias_stride_0; lse_ptr
    # stays last, after them, matching the gfx950 kernel's declaration order.
    kw["include_additive_bias"] = True
    off = [p["name"] for p in au._attn_signature("bf16", **kw)]
    on = [p["name"] for p in au._attn_signature("bf16", emit_lse=True, **kw)]
    assert on == off + ["lse_ptr"]
    assert off[-1] == "additive_bias_sq_stride"


def _values(problem, **extra):
    return au._attn_values(
        problem=problem,
        q=1,
        k=2,
        v=3,
        out=4,
        cu_seqlens_q=5,
        seqused_k=6,
        softmax_scale=0.125,
        block_table=7,
        softcap=0.0,
        sinks=0,
        bt_stride=16,
        include_bt_stride=True,
        include_qq_bias_stride=True,
        kv_addr_limit=None,
        **extra,
    )


def test_attn_values_bind_lse_only_when_requested():
    lse = object()
    assert "lse_ptr" not in _values(_prefill_problem("gfx950"))
    on = _values(_prefill_problem("gfx950", emit_lse=True), lse=lse)
    assert on["lse_ptr"] is lse
    assert list(on)[-1] == "lse_ptr"


@pytest.mark.parametrize("emit_lse", [False, True])
def test_2d_launcher_signature_follows_the_problem(monkeypatch, emit_lse):
    """A cached HSACO stands in for the compile; only the launcher is built."""
    problem = _prefill_problem("gfx942", emit_lse=emit_lse)
    key = au._tiled_cache_key(problem, "gfx942")
    monkeypatch.setattr(au, "_2D_LAUNCHERS", {})
    monkeypatch.setattr(au, "_ATTN_TILED_CACHE", {key: (b"", "kernel")})
    monkeypatch.setattr(au, "KernelLauncher", lambda **kw: SimpleNamespace(**kw))
    launcher = au._get_2d_launcher(problem, key, arch="gfx942")
    names = [p["name"] for p in launcher.signature]
    assert (names[-1] == "lse_ptr") is emit_lse
    assert names.count("lse_ptr") == int(emit_lse)


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
    state = SimpleNamespace(red_vals=None, vals2d=None, graphed=0, validated=[])

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

    # Record what the shared LSE buffer validator was given (it stays a stub:
    # the routing contract is checked without a GPU).
    monkeypatch.setattr(
        au, "validate_unified_lse", lambda *a, **k: state.validated.append(k)
    )

    # The scalar kernel writes no LSE: any scalar launch here is an output drop.
    def never(*a, **k):
        raise AssertionError("a request reached the scalar path, which writes no LSE")

    monkeypatch.setattr(au, "_get_scalar_launcher", never)

    # The tiled 2D launch records the bound arguments and the signature it was
    # built with.
    def get_2d_launcher(problem, key, *, arch, tuning_spec=None):
        sig = [
            p["name"]
            for p in au._attn_signature(
                problem.dtype,
                include_bt_stride=True,
                include_qq_bias_stride=True,
                include_additive_bias=problem.use_additive_bias,
                emit_lse=problem.emit_lse,
            )
        ]
        state.sig2d, state.key2d = sig, key

        def launch(vals, *, config):
            state.vals2d = dict(vals)

        return launch

    monkeypatch.setattr(au, "_get_2d_launcher", get_2d_launcher)
    # develop's tiled launch passes meta.kv_addr_limit to _attn_values (A:6333).
    monkeypatch.setattr(
        au,
        "_get_2d_launch_meta",
        lambda *a, **k: SimpleNamespace(
            grid=(1, 1, 1), block=(64, 1, 1), kv_addr_limit=None
        ),
    )

    # The gfx942 2D graph replays a cache hit before any gate, and its key does
    # not encode emit_lse: an LSE request must never reach it.
    def graphed(problem, **kw):
        assert not problem.emit_lse, "an LSE request reached the 2D graph"
        state.graphed += 1
        return "graphed"

    monkeypatch.setattr(au, "_run_2d_graphed", graphed)
    # Make the 2D graph fast path eligible, so a missing bypass fails.
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
def test_runner_launches_lse_on_a_2d_routed_shape_without_the_graph(runner, backend):
    lse = object()
    _run(_prefill_problem("gfx942", emit_lse=True), lse=lse, backend=backend)
    assert runner.vals2d["lse_ptr"] is lse
    assert runner.sig2d[-1] == "lse_ptr"
    assert runner.key2d[-1] == "lse"
    assert runner.graphed == 0 and runner.red_vals is None


def test_runner_without_lse_keeps_the_2d_graph(runner):
    assert _run(_prefill_problem("gfx942"), backend="auto") == "graphed"
    assert runner.graphed == 1 and runner.vals2d is None


def test_auto_lse_refusal_names_why_both_lse_paths_declined(runner):
    """gfx942 tiled 2D and split-KV both decline the top-left mask; the scalar
    tail writes no LSE, so the refusal names both reasons and nothing launches."""
    prefill = _problem(
        total_q=256,
        num_seqs=1,
        max_seqlen_q=256,
        max_seqlen_k=256,
        emit_lse=True,
        causal_top_left=True,
    )
    assert prefill.select_path() == "2d"
    with pytest.raises(NotImplementedError, match="LSE") as refused:
        _run(prefill, lse=object())
    assert "tiled 2D:" in str(refused.value)
    assert runner.red_vals is None and runner.vals2d is None


def test_runner_reports_why_split_kv_declined_an_lse_request(runner):
    """``auto`` routes this decode to split-KV, whose gfx942 gate declines the
    top-left mask. The refusal must name that reason, not blame the 2D path."""
    top_left = _problem(emit_lse=True, causal_top_left=True)
    assert top_left.select_path() == "3d"
    with pytest.raises(NotImplementedError, match="split-KV.*top-left") as refused:
        _run(top_left, lse=object())
    assert "tiled 2D:" in str(refused.value)
    assert runner.vals2d is None
    assert runner.red_vals is None


def _gfx942_d256_bs32_prefill(**overrides):
    """gfx942 D256 bf16 prefill at block size 32: ``select_path()`` is 2D.
    LSE off, the 4-warp builder serves it; LSE on, the default tiled 2D spec
    exceeds the LDS budget."""
    problem = _problem(
        total_q=2048,
        num_seqs=1,
        head_size=256,
        block_size=32,
        max_seqlen_q=2048,
        max_seqlen_k=2048,
        **overrides,
    )
    assert problem.select_path() == "2d"
    return problem


def test_auto_hands_an_lse_request_tiled_2d_declines_to_split_kv(runner):
    problem = _gfx942_d256_bs32_prefill(emit_lse=True)
    ok, why = au.supports_native_unified_attention_tiled(problem, "gfx942")
    assert not ok and "LDS budget" in why
    lse = object()
    _run(problem, lse=lse)
    assert runner.red_vals["lse_ptr"] is lse
    assert runner.vals2d is None and runner.graphed == 0


def test_auto_without_lse_keeps_tiled_2d_for_that_shape(runner):
    assert _run(_gfx942_d256_bs32_prefill(), backend="auto") == "graphed"
    assert runner.red_vals is None


def test_tiled_backend_still_refuses_an_lse_request_its_gate_declines(runner):
    """Only ``auto`` may hand the request to split-KV; ``tiled`` was asked for
    the 2D kernel and refuses with its gate's reason."""
    with pytest.raises(NotImplementedError, match="LDS budget"):
        _run(_gfx942_d256_bs32_prefill(emit_lse=True), lse=object(), backend="tiled")
    assert runner.red_vals is None and runner.vals2d is None


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


def _tuning_spec_2d(*, kernel_emit_lse, builder_kind="tiled"):
    spec = _tuning_spec(reduce_emit_lse=False)
    spec.path, spec.reduce_spec, spec.builder_kind = "2d", None, builder_kind
    spec.kernel_spec = SimpleNamespace(emit_lse=kernel_emit_lse, kv_storage_dtype=None)
    return spec


@pytest.mark.parametrize(
    "problem_lse,spec_lse", [(True, False), (False, True)], ids=["drop", "extra"]
)
def test_runner_rejects_a_2d_tuning_spec_that_disagrees_on_lse(
    runner, problem_lse, spec_lse
):
    with pytest.raises(ValueError, match="disagrees with problem.emit_lse"):
        _run(
            _prefill_problem("gfx942", emit_lse=problem_lse),
            lse=object() if problem_lse else None,
            tuning_spec=_tuning_spec_2d(kernel_emit_lse=spec_lse),
        )
    assert runner.vals2d is None


def test_runner_launches_an_agreeing_2d_tuning_spec_with_lse(runner):
    lse = object()
    _run(
        _prefill_problem("gfx942", emit_lse=True),
        lse=lse,
        tuning_spec=_tuning_spec_2d(kernel_emit_lse=True),
    )
    assert runner.vals2d["lse_ptr"] is lse and runner.graphed == 0


def test_runner_refuses_lse_for_a_gfx942_4warp_tuning_spec(runner):
    spec = _tuning_spec_2d(kernel_emit_lse=True, builder_kind="gfx942_4warp_gqa")
    with pytest.raises(NotImplementedError, match="4-warp"):
        _run(_prefill_problem("gfx942", emit_lse=True), lse=object(), tuning_spec=spec)
    assert runner.vals2d is None


def test_runner_validates_the_2d_lse_buffer_with_the_shared_overlap_set(runner):
    """The LSE buffer must not overlap the additive bias either. The
    runner calls validate_unified_lse (_check_lse_buffer) once, before any
    path is chosen, so the tiled 2D path reuses it rather than adding its own."""
    lse = object()
    _run(_prefill_problem("gfx942", emit_lse=True), lse=lse)
    (checked,) = runner.validated
    assert "additive_bias" in checked
    assert runner.vals2d["lse_ptr"] is lse


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
