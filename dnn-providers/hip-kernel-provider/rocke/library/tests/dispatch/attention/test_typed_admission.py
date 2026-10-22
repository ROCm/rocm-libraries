# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Raw attention values must fail before selection or normalization."""

from dataclasses import asdict, replace
from fractions import Fraction
from types import SimpleNamespace

import pytest
from dispatch.attention import AttentionRequest, dispatch_attention
from dispatch.attention import bindings
from dispatch.attention.common import _problem, _request_errors, _tuning_problem
from dispatch.attention.tuning_specs import (
    ExplicitAttention2DConfig,
    ExplicitAttention3DConfig,
    make_explicit_attention_2d_spec,
    make_explicit_attention_3d_specs,
    resolve_tile_policy,
)
from kernels.common import attention_unified as au


class _Coercible:
    def __int__(self):
        raise AssertionError("raw value was coerced")

    def __index__(self):
        raise AssertionError("raw value used index conversion")

    def __bool__(self):
        raise AssertionError("raw value used truthiness")

    def __str__(self):
        raise AssertionError("raw value used string conversion")


BAD_INTS = (
    True,
    False,
    32.0,
    32.9,
    "32",
    Fraction(32),
    float("nan"),
    float("inf"),
    None,
    _Coercible(),
)
BAD_BOOLS = (0, 1, 0.0, 1.0, "false", None, _Coercible())
REQUEST_INTS = (
    "batch",
    "nhead_q",
    "nhead_k",
    "seqlen_q",
    "seqlen_k",
    "hdim_q",
    "hdim_v",
    "sliding_window",
    "kv_block_size",
    "num_cus",
    "target_ctas",
)
PROBLEM_INTS = (
    "total_q",
    "num_seqs",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "block_size",
    "max_seqlen_q",
    "max_seqlen_k",
    "sliding_window",
    "num_cus",
    "target_ctas",
    "num_kv_blocks",
    "waves_per_eu",
)


def _req(arch="gfx950"):
    return AttentionRequest(
        batch=3,
        nhead_q=8,
        nhead_k=2,
        seqlen_q=1,
        seqlen_k=33,
        hdim_q=64,
        hdim_v=64,
        arch=arch,
        kv_block_size=32,
        algorithm="paged_decode_t32",
    )


def _poison(obj, field, value):
    # Model input bypassing a frozen dataclass's constructor.
    clone = replace(obj)
    object.__setattr__(clone, field, value)
    return clone


@pytest.mark.parametrize("field", REQUEST_INTS)
def test_request_integer_fields_before_extraction(field, subtests):
    req = _req()
    for value in BAD_INTS:
        with subtests.test(field=field, value=repr(value)):
            with pytest.raises(ValueError, match=field):
                replace(req, **{field: value})
            bad = _poison(req, field, value)
            for use in (
                lambda: bad.dims(),
                lambda: bad.features(),
                lambda: bad.normalized(),
                lambda: _request_errors(bad),
                lambda: _problem(bad),
                lambda: _tuning_problem(bad),
                lambda: dispatch_attention(bad),
            ):
                with pytest.raises(ValueError, match=field):
                    use()


@pytest.mark.parametrize(
    "field", ["use_sinks", "use_fp8", "fp8_fnuz", "use_alibi", "use_qq_bias"]
)
def test_request_boolean_fields(field, subtests):
    for value in BAD_BOOLS:
        with subtests.test(value=repr(value)), pytest.raises(ValueError, match=field):
            replace(_req(), **{field: value})


@pytest.mark.parametrize(
    "field", ["arch", "op", "dtype", "algorithm", "spec_id", "tuning_id", "kv_layout"]
)
def test_request_string_fields(field, subtests):
    for value in (1, True, None, _Coercible()):
        if field == "arch" and value is None:
            assert replace(_req(), arch=None).arch == ""
            continue
        with subtests.test(value=repr(value)), pytest.raises(ValueError, match=field):
            replace(_req(), **{field: value})


@pytest.mark.parametrize("field", PROBLEM_INTS)
def test_problem_integer_fields_at_direct_and_factory_gates(
    field, monkeypatch, subtests
):
    p = _problem(_req())
    monkeypatch.setattr(
        au,
        "_resolve_attention_arch",
        lambda: pytest.fail("invalid problem reached selection"),
    )
    for value in BAD_INTS:
        if field == "waves_per_eu" and value is None:
            continue
        with subtests.test(field=field, value=repr(value)):
            with pytest.raises(ValueError, match=field):
                replace(p, **{field: value})
            bad = _poison(p, field, value)
            for use in (
                lambda: make_explicit_attention_3d_specs(
                    bad, ExplicitAttention3DConfig(8, "32"), arch="gfx950"
                ),
                lambda: make_explicit_attention_2d_spec(
                    bad, ExplicitAttention2DConfig(1, 16, "1x"), arch="gfx950"
                ),
                lambda: au.run_unified_attention_torch(
                    problem=bad,
                    q=None,
                    k=None,
                    v=None,
                    out=None,
                    cu_seqlens_q=None,
                    seqused_k=None,
                    block_table=None,
                    softmax_scale=1.0,
                    softcap=0.0,
                ),
                lambda: au.prepare_unified_attention_torch(
                    problem=bad,
                    q=None,
                    k=None,
                    v=None,
                    out=None,
                    cu_seqlens_q=None,
                    seqused_k=None,
                    block_table=None,
                    softmax_scale=1.0,
                    softcap=0.0,
                ),
            ):
                with pytest.raises(ValueError, match=field):
                    use()


@pytest.mark.parametrize(
    "field", ["use_sinks", "use_alibi", "use_qq_bias", "use_fp8", "fp8_fnuz"]
)
def test_problem_boolean_fields(field, subtests):
    for value in BAD_BOOLS:
        with subtests.test(value=repr(value)), pytest.raises(ValueError, match=field):
            replace(_problem(_req()), **{field: value})


@pytest.mark.parametrize("field", ["dtype", "q_dtype", "compile_backend", "clamp_arch"])
def test_problem_string_fields(field, subtests):
    for value in (1, True, _Coercible()):
        with subtests.test(value=repr(value)), pytest.raises(ValueError, match=field):
            replace(_problem(_req()), **{field: value})


def test_problem_softcap_types(subtests):
    for value in (True, False, "0", None, _Coercible()):
        with subtests.test(value=repr(value)), pytest.raises(
            ValueError, match="softcap"
        ):
            replace(_problem(_req()), softcap=value)


@pytest.mark.parametrize("path", ["2d", "3d"])
def test_explicit_config_types_and_standalone_tile_resolver(path, subtests):
    base = (
        ExplicitAttention3DConfig(8, "32")
        if path == "3d"
        else ExplicitAttention2DConfig(1, 16, "1x")
    )
    names = (
        ("num_segments", "waves_per_eu")
        if path == "3d"
        else ("num_warps", "block_m_per_warp", "waves_per_eu")
    )
    for field in names:
        for value in BAD_INTS:
            if field == "waves_per_eu" and value is None:
                continue
            with subtests.test(field=field, value=repr(value)):
                with pytest.raises(ValueError, match=field):
                    replace(base, **{field: value})
                bad = _poison(base, field, value)
                factory = (
                    make_explicit_attention_3d_specs
                    if path == "3d"
                    else make_explicit_attention_2d_spec
                )
                with pytest.raises(ValueError, match=field):
                    factory(_problem(_req()), bad, arch="gfx950")
    for value in BAD_INTS:
        with subtests.test(value=repr(value)), pytest.raises(
            ValueError, match="block_size"
        ):
            resolve_tile_policy(value, "32", path=path)
    for value in (32, True, None, _Coercible()):
        with subtests.test(value=repr(value)), pytest.raises(
            ValueError, match="tile_policy"
        ):
            resolve_tile_policy(32, value, path=path)
    assert resolve_tile_policy(64, " HALF ", path=path) == 32


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_binding_raw_comparisons_and_retarget_count(arch, subtests):
    req = _req(arch)
    p = _problem(req)
    spec = dispatch_attention(req).spec
    for value in BAD_INTS:
        with subtests.test(value=repr(value)):
            with pytest.raises(ValueError, match="kv_block_size"):
                bindings.validate_tuning_attention_contract(
                    _poison(req, "kv_block_size", value), p, spec
                )
            with pytest.raises(ValueError, match="block_size"):
                bindings.validate_tuning_attention_contract(
                    req, _poison(p, "block_size", value), spec
                )
            with pytest.raises(ValueError, match="num_kv_blocks"):
                spec.with_num_kv_blocks(value)
    for count in (0, 1, 32):
        assert spec.with_num_kv_blocks(count).num_kv_blocks == count


@pytest.mark.parametrize("change", ["arch", "path", "segments", "type", "geometry"])
def test_binding_rechecks_returned_retarget(change, monkeypatch):
    req = replace(_req("gfx942"), kv_block_size=64) if change == "geometry" else _req()
    p = _problem(req)
    spec = dispatch_attention(req).spec
    returned = spec
    if change == "arch":
        returned = replace(spec, arch="gfx1250")
    elif change == "path":
        returned = replace(spec, path="2d")
    elif change == "geometry":
        returned = replace(
            spec, kernel_spec=replace(spec.kernel_spec, tile_size_override=16)
        )
    elif change == "segments":
        returned = replace(spec, reduce_spec=replace(spec.reduce_spec, num_segments=16))
    else:
        returned = replace(
            spec, reduce_spec=_poison(spec.reduce_spec, "num_segments", 8.0)
        )
    structural = SimpleNamespace(
        **{
            name: getattr(spec, name)
            for name in ("arch", "path", "kernel_spec", "reduce_spec", "fp8_fnuz")
        }
    )
    structural.with_num_kv_blocks = lambda count: returned
    monkeypatch.setattr(
        bindings, "validate_tuning_attention_tensors", lambda *a, **k: 32
    )
    with pytest.raises(
        ValueError, match="retargeting|num_segments|segment count|geometry"
    ):
        bindings.bind_tuning_attention_torch(
            req, structural, {"problem": p, "q": object()}
        )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950", "gfx1250"])
def test_valid_shared_request_and_default_semantics(arch):
    req = replace(
        _req(arch),
        algorithm="auto",
        kv_block_size=16,
        num_cus=0,
        target_ctas=0,
        seqlen_q=512,
        seqlen_k=512,
    )
    result = dispatch_attention(req)
    assert result.spec is not None
    assert req.dims()["kv_block_size"] == 16
    from rocke.helpers.split_k import _ARCH_NUM_CUS

    if arch in _ARCH_NUM_CUS:
        assert _tuning_problem(req).num_cus == _ARCH_NUM_CUS[arch]
    else:
        with pytest.raises(ValueError, match="pass num_cus"):
            _tuning_problem(req)
        assert _tuning_problem(replace(req, num_cus=120)).num_cus == 120
    p = _problem(req)
    assert p.waves_per_eu is None and p.num_kv_blocks == 0
    assert au.UnifiedAttentionProblem(**asdict(p)) == p


@pytest.mark.parametrize(
    "field",
    ["num_warps", "block_m_per_warp", "num_segments", "waves_per_eu", "tile_policy"],
)
def test_explicit_configs_reject_missing_consumed_fields(field):
    config = (
        ExplicitAttention3DConfig(8, "32")
        if field == "num_segments"
        else ExplicitAttention2DConfig(1, 16, "1x")
    )
    raw = SimpleNamespace(**asdict(config))
    delattr(raw, field)
    factory = (
        make_explicit_attention_3d_specs
        if field == "num_segments"
        else make_explicit_attention_2d_spec
    )
    with pytest.raises(ValueError, match=field):
        factory(_problem(_req()), raw, arch="gfx950")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_direct_config_knobs_require_types(arch, subtests):
    req = _req(arch)
    # The concrete config is downstream of the documented tuning-axis boundary.
    for value in BAD_BOOLS:
        with subtests.test(value=repr(value)), pytest.raises(
            ValueError, match="use_invariant_hoist"
        ):
            make_explicit_attention_3d_specs(
                _problem(req),
                ExplicitAttention3DConfig(
                    8, "32", knobs=(("use_invariant_hoist", value),)
                ),
                arch=arch,
            )


class _IntSubclass(int):
    pass


def test_integer_subclasses_and_optional_defaults_remain_valid():
    req = replace(_req(), kv_block_size=_IntSubclass(32), num_cus=0, target_ctas=0)
    assert dispatch_attention(req).spec.kernel_spec.block_size == 32
    p = _problem(req)
    assert (
        replace(
            p, waves_per_eu=None, q_dtype=None, compile_backend=None, clamp_arch=None
        ).waves_per_eu
        is None
    )
    assert ExplicitAttention3DConfig(8, " 32 ", waves_per_eu=None).num_segments == 8


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_shared_2d_config_knobs_reject_untyped_values(arch, subtests):
    problem = replace(
        _problem(_req(arch)),
        total_q=1024,
        max_seqlen_q=1024,
        max_seqlen_k=1024,
        num_seqs=1,
        head_size=128,
        block_size=16,
    )
    int_knob = "ring_depth" if arch == "gfx942" else "sched_barrier_mask"
    for name, value in (
        ("use_register_pv", 1),
        ("use_register_pv", "false"),
        (int_knob, True),
        (int_knob, 8.0),
    ):
        config = ExplicitAttention2DConfig(2, 16, "4x", knobs=((name, value),))
        with subtests.test(name=name, value=repr(value)), pytest.raises(
            ValueError, match=name
        ):
            make_explicit_attention_2d_spec(problem, config, arch=arch)


@pytest.mark.parametrize("kind", ["request", "problem"])
def test_missing_raw_fields_reject_before_conversion(kind, subtests):
    obj = _req() if kind == "request" else _problem(_req())
    from kernels.common._attention_validation import (
        validate_attention_request_types,
        validate_attention_problem_types,
    )

    validate = (
        validate_attention_request_types
        if kind == "request"
        else validate_attention_problem_types
    )
    for field in asdict(obj):
        if kind == "request" and field in ("mask_type", "tuning_knobs"):
            continue  # Dedicated enum/container contracts validate these fields.
        raw = SimpleNamespace(**asdict(obj))
        delattr(raw, field)
        with subtests.test(field=field), pytest.raises(ValueError, match=field):
            validate(raw)


def test_request_softcap_before_selection(subtests):
    req = _req()
    for value in (True, False, "0", None, _Coercible(), -1, float("nan"), float("inf")):
        with subtests.test(value=repr(value)):
            with pytest.raises(ValueError, match="softcap"):
                replace(req, softcap=value)
            bad = _poison(req, "softcap", value)
            for use in (bad.features, bad.normalized, lambda: _problem(bad)):
                with pytest.raises(ValueError, match="softcap"):
                    use()


@pytest.mark.parametrize("field", ["window_left", "window_right"])
def test_explicit_window_types_before_selection(field, subtests):
    req = _req()
    for value in BAD_INTS:
        if value is None:
            continue
        with subtests.test(value=repr(value)), pytest.raises(ValueError, match=field):
            replace(req, **{field: value})
    assert getattr(replace(req, **{field: None}), field) is None


@pytest.mark.parametrize(
    "field",
    [
        "use_softcap",
        "use_attn_bias",
        "emit_lse",
        "window_bottom_right",
        "runtime_head_dims",
    ],
)
def test_extended_request_flags_before_selection(field, subtests):
    for value in BAD_BOOLS:
        with subtests.test(value=repr(value)), pytest.raises(ValueError, match=field):
            replace(_req(), **{field: value})


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_paged_selection_uses_explicit_arch(arch, monkeypatch):
    req = replace(_req(arch), arch=arch.upper() + ":xnack-", num_cus=120)
    monkeypatch.setattr(
        au, "_resolve_attention_arch", lambda: pytest.fail("selection read host arch")
    )
    spec = dispatch_attention(req).spec
    assert spec.arch == arch
    assert spec.kernel_spec.num_segments == au._num_segments(_problem(req), arch)


@pytest.mark.parametrize("changes", [{"softcap": 2.0}, {"use_softcap": True}])
def test_softcap_feature_accepts_both_request_forms(changes):
    assert "softcap" in replace(_req(), **changes).features()


def test_paged_softcap_flag_requires_numeric_value():
    from dispatch.attention.paged_decode import make_candidate

    candidate = make_candidate()
    assert not candidate.admits(replace(_req(), use_softcap=True))[0]
    req = replace(_req(), softcap=2.0)
    assert req.use_softcap
    assert candidate.admits(req)[0]
    assert candidate.select_spec(req).kernel_spec.has_softcap


def test_paged_binding_checks_problem_before_reading_fields(monkeypatch):
    from dispatch.attention.paged_decode import make_candidate

    req = _req()
    spec = dispatch_attention(req).spec

    class PoisonProblem:
        def __getattr__(self, name):
            pytest.fail(f"read unvalidated problem.{name}")

    with pytest.raises(TypeError, match="UnifiedAttentionProblem"):
        make_candidate().bind_torch(req, spec, {"problem": PoisonProblem()})


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("page", [1, 16, 32, 64])
def test_paged_binding_checks_compiled_window(arch, page):
    from dispatch.attention.paged_decode import make_candidate

    req = replace(_req(arch), kv_block_size=page)
    spec = dispatch_attention(req).spec
    bad = replace(spec, kernel_spec=replace(spec.kernel_spec, sliding_window=17))
    with pytest.raises(ValueError, match="kernel_spec.sliding_window"):
        make_candidate().bind_torch(req, bad, {"problem": _problem(req)})
