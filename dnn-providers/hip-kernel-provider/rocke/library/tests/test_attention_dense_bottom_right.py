# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Focused CPU/spec/IR coverage for dense bottom-right causal attention."""

from __future__ import annotations

import dataclasses

import pytest

from kernels.common.attention_dense_spec import (
    DENSE_TILE_GEOMETRIES,
    attention_dense_cache_key,
)
from kernels.gfx950.attention_dense import (
    Gfx950AttentionDenseSpec,
    build_attention_dense,
    supports_attention_dense,
)


_GEOMETRIES = tuple(DENSE_TILE_GEOMETRIES)


def _spec(*, geometry: str = "default", **over) -> Gfx950AttentionDenseSpec:
    tile = DENSE_TILE_GEOMETRIES[geometry]
    kw = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=1024,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_m=int(tile["block_m"]),
        block_n=int(tile["block_n"]),
    )
    kw.update(over)
    return Gfx950AttentionDenseSpec(**kw)


def _lowered_kernel(kernel) -> str:
    """Lower without a toolchain and remove the identity-only symbol difference."""
    from rocke.helpers.compile import _lower_llvm_via_backend

    llvm = _lower_llvm_via_backend(kernel, arch="gfx950", backend="python", spec=None)
    return llvm.replace(kernel.name, "KERNEL")


def _lowered_body(spec: Gfx950AttentionDenseSpec) -> str:
    return _lowered_kernel(build_attention_dense(spec, arch="gfx950"))


def _walk_ops(region):
    for op in region.ops:
        yield op
        for child in op.regions:
            yield from _walk_ops(child)


def _const_i32(value) -> int:
    op = value.op
    assert op is not None and op.name == "arith.constant"
    assert op.attrs["ity"] == "i32"
    return int(op.attrs["value"])


@pytest.mark.parametrize(
    "over,match",
    [
        ({"causal": False}, "requires causal=True"),
        ({"seqlen_q": 1024, "seqlen_kv": 512}, "seqlen_q <= seqlen_kv"),
    ],
)
def test_bottom_right_common_invariants(over, match):
    with pytest.raises(ValueError, match=match):
        _spec(causal_bottom_right=True, **over)


@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("ragged", [False, True], ids=["aligned", "ragged"])
def test_cross_length_bottom_right_accepts_both_geometries(geometry, ragged):
    tile = DENSE_TILE_GEOMETRIES[geometry]
    if ragged:
        sq, skv = 197, 400
    else:
        sq = 2 * int(tile["block_m"])
        skv = sq + 4 * int(tile["block_n"])

    spec = _spec(
        geometry=geometry,
        seqlen_q=sq,
        seqlen_kv=skv,
        ragged=ragged,
        causal_bottom_right=True,
    )
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why
    assert spec.block_m == int(tile["block_m"])
    assert spec.block_n == int(tile["block_n"])


def test_ragged_non_causal_cross_length_is_gfx950_cross_attention():
    # Full attention between different lengths is valid on gfx950: the key-pad mask and
    # the guarded store handle the ragged tails, the diagonal plays no part.
    for sq, sk in ((197, 400), (400, 197)):
        spec = _spec(seqlen_q=sq, seqlen_kv=sk, ragged=True, causal=False)
        ok, why = supports_attention_dense(spec, arch="gfx950")
        assert ok, why


def test_ragged_non_causal_cross_length_stays_rejected_on_gfx942():
    from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec

    tile = DENSE_TILE_GEOMETRIES["default"]
    with pytest.raises(ValueError, match="self-attention only"):
        Gfx942AttentionDenseSpec(
            batch=1,
            seqlen_q=197,
            seqlen_kv=400,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=False,
            dtype="bf16",
            block_m=int(tile["block_m"]),
            block_n=int(tile["block_n"]),
            ragged=True,
        )


@pytest.mark.parametrize("sq,sk", [(1000, 1050), (1050, 1000)])
@pytest.mark.parametrize("persistent", [False, True])
def test_ragged_cross_length_top_left_accepted(sq, sk, persistent):
    spec = _spec(seqlen_q=sq, seqlen_kv=sk, ragged=True, persistent=persistent)
    assert not spec.causal_bottom_right
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why


_PAGED = {"paged": True, "block_size": 16, "num_kv_blocks": 64, "dtype": "fp16"}


@pytest.mark.parametrize(
    "feature,over",
    [
        ("persistent", {"persistent": True}),
        ("sliding_window", {"sliding_window": 64}),
        # The single-sequence paged body has no shifted diagonal.
        ("paged=True unless varlen", _PAGED),
        ("right_bound", {"varlen": True, "right_bound": 64}),
        ("seqlen_q <= seqlen_kv", {**_PAGED, "varlen": True, "seqlen_q": 2048}),
    ],
)
def test_gfx950_rejects_each_unimplemented_bottom_right_mode(feature, over):
    with pytest.raises(ValueError, match=feature):
        _spec(causal_bottom_right=True, **over)


# Varlen bottom-right: the spec lengths are maxima and each sequence's diagonal
# shift kv_len - q_len is read at run time.
_VARLEN_LAYOUTS = {"contiguous": {"varlen": True}, "paged": {**_PAGED, "varlen": True}}


@pytest.mark.parametrize("geometry", _GEOMETRIES)
@pytest.mark.parametrize("layout", sorted(_VARLEN_LAYOUTS))
def test_varlen_bottom_right_is_accepted(geometry, layout):
    spec = _spec(geometry=geometry, causal_bottom_right=True, **_VARLEN_LAYOUTS[layout])
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why
    assert "_br_" in spec.kernel_name() and "_varlen" in spec.kernel_name()
    top_left = _spec(geometry=geometry, **_VARLEN_LAYOUTS[layout], seqlen_kv=512)
    assert attention_dense_cache_key(spec, arch="gfx950") != attention_dense_cache_key(
        top_left, arch="gfx950"
    )


def test_bottom_right_varlen_keeps_its_baked_maxima():
    """Varlen bakes its maxima, so the bottom-right body keeps per-shape identity."""
    spec = _spec(causal_bottom_right=True, varlen=True)
    assert not spec.runtime_shape
    assert dataclasses.replace(spec, seqlen_kv=2496).kernel_name() != spec.kernel_name()


def test_paged_varlen_top_left_allows_unequal_maxima():
    """Top-left starts every sequence's diagonal at its first key, whatever its
    lengths."""
    spec = _spec(**_PAGED, varlen=True)
    assert spec.seqlen_q != spec.seqlen_kv and not spec.causal_bottom_right


def test_gfx942_supports_paged_varlen_bottom_right():
    from kernels.gfx942.attention_dense import Gfx942AttentionDenseSpec
    from kernels.gfx942.attention_dense import (
        supports_attention_dense as supports_gfx942,
    )

    ok, why = supports_gfx942(
        Gfx942AttentionDenseSpec(
            batch=3,
            seqlen_q=512,
            seqlen_kv=1024,
            num_query_heads=4,
            num_kv_heads=1,
            head_size=128,
            causal=True,
            varlen=True,
            causal_bottom_right=True,
            **_PAGED,
        ),
        arch="gfx942",
    )
    assert ok, why


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"varlen": True}, id="contiguous"),
        pytest.param({"varlen": True, "emit_lse": True}, id="contiguous_lse"),
        pytest.param({**_PAGED, "num_kv_blocks": 2048, "varlen": True}, id="paged"),
    ],
)
def test_varlen_bottom_right_bakes_no_diagonal(over):
    """With maxima Sq=2048, Skv=8192 a baked shift would be the constant 6144; the
    static bottom-right body has it, the varlen body must not."""
    import re

    shape = {"batch": 4, "seqlen_q": 2048, "seqlen_kv": 8192}
    baked = _lowered_body(_spec(causal_bottom_right=True, **shape))
    runtime = _lowered_body(_spec(causal_bottom_right=True, **shape, **over))
    assert re.search(r"\b6144\b", baked)
    assert not re.search(r"\b6144\b", runtime)


@pytest.mark.parametrize(
    "over",
    [
        pytest.param({"varlen": True}, id="contiguous"),
        pytest.param({**_PAGED, "varlen": True}, id="paged"),
    ],
)
def test_varlen_bottom_right_shifts_by_the_sequence_lengths(over):
    """The shift is kv_len - q_len from the per-sequence lengths: it changes the
    body even where the maxima are equal (each sequence may still differ)."""
    spec = _spec(causal_bottom_right=True, seqlen_kv=512, **over)
    kernel = build_attention_dense(spec, arch="gfx950")
    top_left = _lowered_body(_spec(seqlen_kv=512, **over))
    assert _lowered_kernel(kernel) != top_left
    # The masked diagonal tail starts at max(1, qb * n_per + shift // block_n).
    loops = [
        op
        for op in _walk_ops(kernel.body)
        if op.name == "scf.for" and op.attrs.get("iv") == "%nt"
    ]
    assert len(loops) == 1
    lower = loops[0].operands[0].op
    assert lower is not None and lower.name == "arith.select"
    diag_start = lower.operands[2].op
    assert diag_start is not None and diag_start.name == "arith.add"
    tiles = diag_start.operands[1].op
    assert tiles is not None and tiles.name == "arith.div"
    shift, block_n = tiles.operands
    assert shift.op is not None and shift.op.name == "arith.sub"
    assert _const_i32(block_n) == spec.block_n


def test_paged_varlen_bottom_right_keeps_the_i64_page_addressing():
    """The shift moves the diagonal, not the K/V addresses: with
    ``use_i64_kv_addr`` the bottom-right body has the top-left body's per-page
    resources, page bases and block-table lookups."""
    import re

    over = {**_PAGED, "varlen": True, "seqlen_kv": 512, "use_i64_kv_addr": True}
    spec = _spec(causal_bottom_right=True, **over)
    assert supports_attention_dense(spec, arch="gfx950") == (True, "")
    bottom_right, top_left = _lowered_body(spec), _lowered_body(_spec(**over))
    assert bottom_right != top_left
    for pattern in (
        r"@llvm\.amdgcn\.make\.buffer\.rsrc\.",
        r"getelementptr inbounds i8, ptr addrspace\(1\) %[kv]_ptr, i64 ",
        r"getelementptr inbounds i32, ptr addrspace\(1\) %block_tables",
    ):
        count = len(re.findall(pattern, bottom_right))
        assert count > 0 and count == len(re.findall(pattern, top_left)), pattern


@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_sinks_remain_valid_on_nonpersistent_bottom_right(geometry):
    spec = _spec(
        geometry=geometry,
        causal_bottom_right=True,
        use_sinks=True,
        persistent=False,
    )
    ok, why = supports_attention_dense(spec, arch="gfx950")
    assert ok, why
    assert "_br" in spec.kernel_name()
    assert "sinks" in spec.kernel_name()


def test_bottom_right_has_distinct_symbol_and_cache_identity():
    top_left = _spec()
    bottom_right = _spec(causal_bottom_right=True)

    assert "_br" not in top_left.kernel_name()
    assert "_br" in bottom_right.kernel_name()
    assert top_left.kernel_name() != bottom_right.kernel_name()
    assert attention_dense_cache_key(
        top_left, arch="gfx950"
    ) != attention_dense_cache_key(bottom_right, arch="gfx950")


def test_different_bottom_right_offsets_do_not_share_compiled_identity():
    first = _spec(seqlen_q=256, seqlen_kv=512, causal_bottom_right=True)
    second = _spec(seqlen_q=512, seqlen_kv=1024, causal_bottom_right=True)

    # The diagonal is baked into each body, so reusing either binary is incorrect.
    assert _lowered_body(first) != _lowered_body(second)
    assert attention_dense_cache_key(first, arch="gfx950") != attention_dense_cache_key(
        second, arch="gfx950"
    )
    assert first.kernel_name() != second.kernel_name()


@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_moving_bottom_right_changes_the_lowered_body(geometry):
    top_left = _spec(geometry=geometry)
    bottom_right = _spec(geometry=geometry, causal_bottom_right=True)
    assert _lowered_body(top_left) != _lowered_body(bottom_right)


@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_equal_length_bottom_right_emits_no_add_zero(geometry):
    tile = DENSE_TILE_GEOMETRIES[geometry]
    seqlen = 2 * int(tile["block_m"])
    top_left = _spec(geometry=geometry, seqlen_q=seqlen, seqlen_kv=seqlen)
    bottom_right = _spec(
        geometry=geometry,
        seqlen_q=seqlen,
        seqlen_kv=seqlen,
        causal_bottom_right=True,
    )

    # Exact body identity proves both diagonal additions are omitted when offset=0.
    assert "_br" in bottom_right.kernel_name()
    assert _lowered_body(top_left) == _lowered_body(bottom_right)


def test_equal_length_dispatch_normalizes_to_top_left_body():
    from dispatch.attention import AttentionRequest, attention_tuning_spec

    def dispatched(mask_type):
        return attention_tuning_spec(
            AttentionRequest(
                batch=1,
                nhead_q=4,
                nhead_k=1,
                seqlen_q=512,
                seqlen_k=512,
                hdim_q=128,
                hdim_v=128,
                arch="gfx950",
                mask_type=mask_type,
                dtype="bf16",
            ),
            "gfx950_dense_grid",
        ).kernel_spec

    top_left = dispatched(1)
    bottom_right_request = dispatched(2)
    assert bottom_right_request.causal_bottom_right is False
    assert bottom_right_request.kernel_name() == top_left.kernel_name()
    assert _lowered_body(bottom_right_request) == _lowered_body(top_left)


@pytest.mark.parametrize("geometry", _GEOMETRIES)
def test_ragged_ir_uses_ceiled_shift_and_kv_bound(geometry):
    """An offset landing mid-tile must still visit the partial final KV tile."""
    sq, skv = 197, 400
    spec = _spec(
        geometry=geometry,
        seqlen_q=sq,
        seqlen_kv=skv,
        ragged=True,
        causal_bottom_right=True,
    )
    kernel = build_attention_dense(spec, arch="gfx950")
    loops = [
        op
        for op in _walk_ops(kernel.body)
        if op.name == "scf.for" and op.attrs.get("iv") == "%nt"
    ]
    assert len(loops) == 1

    upper = loops[0].operands[1]
    assert upper.op is not None and upper.op.name == "arith.select"
    unclamped, kv_bound = upper.op.operands[1:]
    assert unclamped.op is not None and unclamped.op.name == "arith.add"

    offset = skv - sq
    expected_span = (spec.block_m + offset + spec.block_n - 1) // spec.block_n
    expected_kv_tiles = (skv + spec.block_n - 1) // spec.block_n
    assert offset % spec.block_n
    assert _const_i32(unclamped.op.operands[1]) == expected_span
    assert _const_i32(kv_bound) == expected_kv_tiles

    llvm = _lowered_kernel(kernel)
    bound_line = next(
        line for line in llvm.splitlines() if line.strip().startswith(upper.name + " =")
    )
    assert bound_line.endswith(f"i32 {expected_kv_tiles}")
    assert f"icmp slt i32 %nt, {upper.name}" in llvm


def test_sinks_and_shift_each_change_the_lowered_body():
    variants = [
        _spec(),
        _spec(use_sinks=True),
        _spec(causal_bottom_right=True),
        _spec(causal_bottom_right=True, use_sinks=True),
    ]
    assert len({_lowered_body(spec) for spec in variants}) == 4
