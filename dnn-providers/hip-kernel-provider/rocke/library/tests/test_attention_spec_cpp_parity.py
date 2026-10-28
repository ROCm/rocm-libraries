# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The C++ tiled spec-from-problem hooks build the same spec Python does.

``rocke_engine.<arch>_attn_tiled_spec_from_problem`` runs the C++ selectors and
spec assembly; ``attention_unified._tiled_spec_from_problem`` is the reference.
Every field both specs carry must match, and where Python raises (spec
validation, LDS budget) C++ must raise the same exception type."""

from __future__ import annotations

import itertools
from dataclasses import asdict, replace

import pytest
from kernels.common.attention_unified import (
    UnifiedAttentionProblem,
    _tiled_3d_spec_from_problem,
    _tiled_spec_from_problem,
)

# Every HIPDNN_GFX942_* knob the gfx942 selectors read, one setting each.
_GFX942_ENVS = (
    {},
    {"HIPDNN_GFX942_FLASH_WIDE": "0"},
    {"HIPDNN_GFX942_FLASH_WIDE": "2"},
    {"HIPDNN_GFX942_FLASH_MLIM": "d64"},
    {"HIPDNN_GFX942_K_SLICED_RING": "0"},
    {"HIPDNN_GFX942_K_SLICE_HD": "8"},
    {"HIPDNN_GFX942_K_LDSSEQ": " Ck "},
    {"HIPDNN_GFX942_BF16_CFVST": "0"},
    {"HIPDNN_GFX942_D128_SMALLTILE_DK": "0"},
)
_3D_ENVS = ({}, {"HIPDNN_GFX942_3D_WKV": " Off "})


@pytest.fixture(scope="module")
def engine():
    return pytest.importorskip("rocke_engine")


@pytest.fixture
def clean_env(monkeypatch):
    for name in {k for env in _GFX942_ENVS + _3D_ENVS for k in env}:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _problems_2d():
    for (
        dtype,
        head_size,
        block_size,
        (nq, nkv),
        num_seqs,
        max_seqlen_q,
        sliding_window,
        extra,
    ) in itertools.product(
        ("fp16", "bf16"),
        (64, 128, 256),
        (16, 32, 64),
        ((8, 8), (32, 8), (64, 8)),
        (1, 4),
        (1, 512, 2048),
        (0, 256, 4096),
        (None, "use_sinks", "softcap", "use_alibi", "use_fp8", "num_kv_blocks"),
    ):
        kw = {}
        if extra == "softcap":
            kw["softcap"] = 30.0
        elif extra == "num_kv_blocks":
            kw["num_kv_blocks"] = 1 << 22  # past the i32 KV buffer range
        elif extra is not None:
            kw[extra] = True
        yield UnifiedAttentionProblem(
            dtype=dtype,
            head_size=head_size,
            block_size=block_size,
            num_query_heads=nq,
            num_kv_heads=nkv,
            num_seqs=num_seqs,
            total_q=num_seqs * max_seqlen_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max(max_seqlen_q, 2048),
            sliding_window=sliding_window,
            **kw,
        )


def _outcome(fn, *args):
    try:
        return fn(*args), None
    except (ValueError, RuntimeError, NotImplementedError) as e:
        return None, type(e)


def _check_spec(engine, arch, problems, keys=None, dim="2d"):
    if dim == "3d":
        hook = getattr(engine, f"{arch}_attn_tiled_3d_spec_from_problem")
        py_fn = _tiled_3d_spec_from_problem
    else:
        hook = getattr(engine, f"{arch}_attn_tiled_spec_from_problem")
        py_fn = _tiled_spec_from_problem
    for p in problems:
        py, py_err = _outcome(py_fn, p, arch)
        cpp, cpp_err = _outcome(hook, asdict(p), arch)
        assert py_err == cpp_err, (p, py_err, cpp_err)
        if py is None:
            continue
        py = asdict(py)
        shared = py.keys() & cpp.keys() if keys is None else keys
        diff = {k: (cpp[k], py[k]) for k in shared if cpp[k] != py[k]}
        assert not diff, (p, diff)


@pytest.mark.parametrize(
    "env", _GFX942_ENVS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items())
)
def test_gfx942_2d_spec_matches_python(engine, clean_env, env):
    for name, value in env.items():
        clean_env.setenv(name, value)
    _check_spec(engine, "gfx942", _problems_2d())


def test_gfx942_2d_hook_serves_only_gfx942(engine):
    p = asdict(next(_problems_2d()))
    with pytest.raises(NotImplementedError, match="gfx942"):
        engine.gfx942_attn_tiled_spec_from_problem(p, "gfx950")
    with pytest.raises(NotImplementedError, match="target_ctas"):
        engine.gfx942_attn_tiled_spec_from_problem(dict(p, target_ctas=256), "gfx942")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_2d_hook_honours_waves_per_eu_pin(engine, clean_env, arch):
    problems = [
        replace(p, waves_per_eu=wpe)
        for p in itertools.islice(_problems_2d(), 0, None, 37)
        for wpe in (1, 3)
    ]
    # gfx950 still differs from Python on fields the pin does not touch.
    _check_spec(engine, arch, problems, keys={"waves_per_eu"})
    p = replace(next(_problems_2d()), waves_per_eu=3)
    spec = getattr(engine, f"{arch}_attn_tiled_spec_from_problem")(asdict(p), arch)
    assert spec["waves_per_eu"] == 3


def _problems_3d():
    for (
        dtype,
        head_size,
        block_size,
        (nq, nkv),
        num_seqs,
        max_seqlen_q,
        max_seqlen_k,
        sliding_window,
        (num_cus, target_ctas),
        extra,
    ) in itertools.product(
        ("fp16", "bf16"),
        (64, 128, 256),
        (16, 32),
        ((32, 8), (8, 8)),
        (1, 16),
        (1, 8),
        (2048, 16384, 65536),
        (0, 256),
        ((120, 0), (304, 0), (600, 0), (304, 2048)),
        (None, "use_fp8", "use_sinks", "num_kv_blocks", "kv_2gib", "waves_per_eu"),
    ):
        kw = {}
        if extra == "num_kv_blocks":
            kw["num_kv_blocks"] = 1 << 22  # past the i32 KV buffer range
        elif extra == "kv_2gib":
            # Past the 0x7FFF0000-byte buffer bound but within 2^31 bytes, which
            # only the gfx942 3D wide and fp8 K/V loaders read with i32 offsets.
            kw["num_kv_blocks"] = (1 << 31) // (block_size * nkv * head_size * 2)
        elif extra == "waves_per_eu":
            kw["waves_per_eu"] = 3
        elif extra is not None:
            kw[extra] = True
        yield UnifiedAttentionProblem(
            dtype=dtype,
            head_size=head_size,
            block_size=block_size,
            num_query_heads=nq,
            num_kv_heads=nkv,
            num_seqs=num_seqs,
            total_q=num_seqs * max_seqlen_q,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            sliding_window=sliding_window,
            num_cus=num_cus,
            target_ctas=target_ctas,
            **kw,
        )


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "env", _3D_ENVS, ids=lambda e: ",".join(f"{k}={v}" for k, v in e.items())
)
def test_3d_spec_matches_python(engine, clean_env, arch, env):
    for name, value in env.items():
        clean_env.setenv(name, value)
    _check_spec(engine, arch, _problems_3d(), dim="3d")


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_3d_hook_rejects_other_arch_and_unported_fields(engine, arch):
    hook = getattr(engine, f"{arch}_attn_tiled_3d_spec_from_problem")
    p = asdict(next(_problems_3d()))
    other = "gfx950" if arch == "gfx942" else "gfx942"
    with pytest.raises(NotImplementedError, match=arch):
        hook(p, other)
    for field, value in (
        ("use_additive_bias", True),
        ("causal_top_left", True),
        ("right_bound", -1),
    ):
        with pytest.raises(NotImplementedError, match=field):
            hook(dict(p, **{field: value}), arch)
    with pytest.raises(ValueError):
        hook(dict(p, num_kv_heads=3), arch)
