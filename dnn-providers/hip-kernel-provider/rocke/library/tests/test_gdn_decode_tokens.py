# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU contract of the multi-token GDN/KDA decode step (``num_tokens``).

``num_tokens = T`` chains T recurrence steps of one sequence in one launch
(the MTP verify step), writing each token's output to ``out[b, t]`` and its
post-state to ``write_indices[b, t]``. The single-token kernel must not move:
``T = 1`` keeps its name and its IR (the golden test pins the IR).
"""

from __future__ import annotations

import dataclasses as dc

import pytest

from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode, request_errors
from kernels.gfx950.gdn_decode import (
    MAX_TOKENS,
    GdnDecodeSpec,
    build_gdn_decode,
    is_valid_spec,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

ATOM_GEOMETRY = dict(num_k_heads=16, num_v_heads=48, state_dtype="f32")


def _ll(spec):
    return _lower_kernel_to_llvm_python(build_gdn_decode(spec), arch="gfx950")


def test_single_token_name_and_ir_are_unchanged():
    spec = GdnDecodeSpec(**ATOM_GEOMETRY)
    assert dc.replace(spec, num_tokens=1).kernel_name() == spec.kernel_name()
    assert "_t1" not in spec.kernel_name()
    assert _ll(dc.replace(spec, num_tokens=1)) == _ll(spec)


@pytest.mark.parametrize("T", [2, 3, 4, MAX_TOKENS])
def test_multi_token_is_its_own_kernel(T):
    spec = GdnDecodeSpec(**ATOM_GEOMETRY, num_tokens=T)
    assert is_valid_spec(spec) == (True, "")
    assert spec.kernel_name().endswith(f"_t{T}_l2")
    one = _ll(GdnDecodeSpec(**ATOM_GEOMETRY))
    many = _ll(spec)
    assert many != one
    # One output store and one snapshot per token: T x the single-token count.
    assert many.count("store") > one.count("store")


@pytest.mark.parametrize("bad", [0, -1, MAX_TOKENS + 1])
def test_num_tokens_range(bad):
    ok, why = is_valid_spec(GdnDecodeSpec(num_tokens=bad))
    assert not ok and "num_tokens must be in" in why


@pytest.mark.parametrize("bad", [True, 2.0, "2", None])
def test_num_tokens_type(bad):
    ok, why = is_valid_spec(GdnDecodeSpec(num_tokens=bad))
    assert not ok and "num_tokens must be an integer" in why


@pytest.mark.parametrize(
    "over",
    [
        dict(simple=True),
        dict(fuse_conv=True, num_k_heads=16, num_v_heads=16, blocks_per_v_dim=1),
        dict(
            fuse_conv=True,
            fuse_out_norm=True,
            num_k_heads=16,
            num_v_heads=16,
            blocks_per_v_dim=1,
        ),
    ],
)
def test_multi_token_paths_not_yet_implemented(over):
    ok, why = is_valid_spec(GdnDecodeSpec(num_tokens=2, **over))
    assert not ok and why.startswith("NOT_YET_IMPLEMENTED: num_tokens > 1")


@pytest.mark.parametrize("gate_kind", ["gdn", "kda"])
@pytest.mark.parametrize(
    "tile_knobs",
    [
        dict(num_warps=2),
        dict(num_warps=1),
        dict(num_warps=8, out_lds=True),
        dict(num_warps=1, out_lds=True),
        dict(num_warps=16, stream_rows=True, dpp_reduce=True, xcd_remap=True),
    ],
)
def test_out_norm_chains(gate_kind, tile_knobs):
    """fuse_out_norm with num_tokens > 1: its own kernel, one gated norm per
    token (one output store per token, packed or per row)."""
    heads = dict(num_k_heads=16, num_v_heads=48 if gate_kind == "gdn" else 16)
    base = dict(
        heads,
        state_dtype="f32",
        gate_kind=gate_kind,
        fuse_out_norm=True,
        blocks_per_v_dim=1,
        norm_gate_act="sigmoid",
        **tile_knobs,
    )
    spec = GdnDecodeSpec(**base, num_tokens=4)
    assert is_valid_spec(spec) == (True, "")
    assert "_t4" in spec.kernel_name()
    many, one = _ll(spec), _ll(GdnDecodeSpec(**base))
    assert many != one
    # every token's output reaches `out` through the norm's stores
    assert many.count("store") > one.count("store")


@pytest.mark.parametrize("gate_kind", ["gdn", "kda"])
@pytest.mark.parametrize(
    "knobs",
    [
        dict(),
        dict(dpp_reduce=True, interleave_cols=True),
        dict(stream_rows=True, blocks_per_v_dim=1, num_warps=2),
        dict(xcd_remap=True, state_load_hint="streaming", state_store_hint="streaming"),
        dict(waves_per_eu=4),
    ],
)
def test_knobs_build_with_tokens(gate_kind, knobs):
    spec = GdnDecodeSpec(**ATOM_GEOMETRY, gate_kind=gate_kind, num_tokens=4, **knobs)
    if gate_kind == "kda":
        spec = dc.replace(spec, num_v_heads=16)
    assert is_valid_spec(spec) == (True, "")
    assert _ll(spec)


def test_request_carries_num_tokens_to_the_spec():
    req = GdnDecodeRequest(batch=3, arch="gfx950", num_tokens=4, **ATOM_GEOMETRY)
    assert request_errors(req) == []
    spec = dispatch_gdn_decode(req).spec.kernel_spec
    assert spec.num_tokens == 4
    # auto runs the chain's own row (AUTO_DEFAULTS_MULTI_TOKEN), so only the
    # tile and knobs may differ from the single-token pick
    one = dispatch_gdn_decode(dc.replace(req, num_tokens=1)).spec.kernel_spec
    knobs = ("num_warps", "warp_threads_k", "blocks_per_v_dim", "waves_per_eu")
    knobs += ("dpp_reduce", "interleave_cols", "stream_rows", "xcd_remap")
    keep = {k: getattr(one, k) for k in knobs}
    assert dc.replace(spec, num_tokens=1, **keep) == one


@pytest.mark.parametrize(
    "bad, msg",
    [
        (0, "num_tokens must be in"),
        (MAX_TOKENS + 1, "num_tokens must be in"),
        (True, "num_tokens must be an int"),
        (2.0, "num_tokens must be an int"),
    ],
)
def test_request_rejects_bad_num_tokens(bad, msg):
    errs = request_errors(GdnDecodeRequest(batch=1, arch="gfx950", num_tokens=bad))
    assert any(msg in e for e in errs), errs


def test_request_rejects_conv_multi_token():
    errs = request_errors(
        GdnDecodeRequest(
            batch=1,
            arch="gfx950",
            num_tokens=2,
            fuse_conv=True,
            num_k_heads=16,
            num_v_heads=16,
        )
    )
    assert any(e.startswith("NOT_YET_IMPLEMENTED: num_tokens > 1") for e in errs)


def test_request_accepts_out_norm_multi_token():
    req = GdnDecodeRequest(
        batch=2,
        arch="gfx950",
        num_tokens=4,
        fuse_out_norm=True,
        norm_gate_act="sigmoid",
        **ATOM_GEOMETRY,
    )
    assert request_errors(req) == []
    spec = dispatch_gdn_decode(req).spec.kernel_spec
    assert spec.num_tokens == 4 and spec.fuse_out_norm
    assert spec.blocks_per_v_dim == 1
