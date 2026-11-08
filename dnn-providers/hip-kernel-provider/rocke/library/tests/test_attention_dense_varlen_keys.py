# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks for a packed varlen dense-attention launch without the LSE output.

The packed body zeroes no keyless row and has no handling for an empty KV sequence
without LSE, so the launcher rejects both instead of returning wrong values. Needs
torch only for the offset tensors; no GPU.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch required for the offset tensors")

from kernels.common._lse_store import validate_varlen_keys  # noqa: E402
from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES  # noqa: E402
from kernels.gfx950.attention_dense import (  # noqa: E402
    Gfx950AttentionDenseSpec,
    run_attention_dense_torch,
)


def _spec(**over) -> Gfx950AttentionDenseSpec:
    tile = DENSE_TILE_GEOMETRIES["default"]
    kw = dict(
        batch=2,
        seqlen_q=512,
        seqlen_kv=256,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=128,
        causal=True,
        dtype="bf16",
        block_m=int(tile["block_m"]),
        block_n=int(tile["block_n"]),
        varlen=True,
    )
    kw.update(over)
    return Gfx950AttentionDenseSpec(**kw)


def _offsets(*lengths):
    """int32 cumulative offsets for the given per-sequence lengths."""
    out = [0]
    for n in lengths:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


def test_accepts_nonempty_sequences():
    validate_varlen_keys(_spec(), _offsets(256, 128), _offsets(256, 128))


def test_no_window_allows_q_longer_than_kv():
    """Top-left causal rows past kv_len still see every key, so none is keyless."""
    validate_varlen_keys(_spec(), _offsets(512, 256), _offsets(256, 256))


def test_window_accepts_sequences_that_all_keep_a_key():
    """q_len - kv_len = 127 < W: the last row (q = 382) still sees key 255."""
    validate_varlen_keys(
        _spec(sliding_window=128), _offsets(383, 256), _offsets(256, 256)
    )


@pytest.mark.parametrize("kv_lengths", [(256, 0), (0, 256)])
def test_rejects_an_empty_kv_sequence(kv_lengths):
    with pytest.raises(ValueError, match="must be nonempty"):
        validate_varlen_keys(_spec(), _offsets(256, 256), _offsets(*kv_lengths))


def test_window_rejects_a_sequence_that_leaves_rows_without_a_key():
    """The second sequence has q_len - kv_len = 128 = W: its last row sees no key."""
    with pytest.raises(ValueError, match="no key"):
        validate_varlen_keys(
            _spec(sliding_window=128), _offsets(256, 384), _offsets(256, 256)
        )


@pytest.mark.parametrize(
    "q_lengths,kv_lengths", [((256,), (256, 256)), ((256, 256), (256,))]
)
def test_rejects_offsets_of_the_wrong_length(q_lengths, kv_lengths):
    with pytest.raises(ValueError, match=r"batch\+1 entries"):
        validate_varlen_keys(_spec(), _offsets(*q_lengths), _offsets(*kv_lengths))


def test_launcher_runs_the_check_when_lse_is_off():
    """The public launcher reaches the check before it touches the device."""
    spec = _spec(seqlen_q=256, seqlen_kv=256)
    q = torch.zeros(512, 4, 128, dtype=torch.bfloat16)
    k = torch.zeros(256, 1, 128, dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    out = torch.empty_like(q)
    with pytest.raises(ValueError, match="must be nonempty"):
        run_attention_dense_torch(
            spec=spec,
            q=q,
            k=k,
            v=v,
            out=out,
            scale=0.1,
            cu_seqlens_q=_offsets(256, 256),
            cu_seqlens_kv=_offsets(256, 0),
        )
