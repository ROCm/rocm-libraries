# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""PagedBuffers: the P2a/P2b device bindings, checked on CPU tensors."""

import pytest

torch = pytest.importorskip("torch")

from hipdnn_vllm.attention.metadata import PagedBounds, PagedBuffers  # noqa: E402

B, M = 8, 6
BOUNDS = PagedBounds(max_num_seqs=B, max_decode_query_len=1, max_query_len=64, max_model_len=96)


def _runner_buffers():
    """Persistent buffers shaped like the model runner's: [B+1], [B], [B, M]."""
    qsl = torch.full((B + 1,), 999, dtype=torch.int32)  # stale tail
    seq_lens = torch.full((B,), 777, dtype=torch.int32)
    table = torch.arange(B * M, dtype=torch.int32).view(B, M)
    return qsl, seq_lens, table


def test_refresh_widens_runner_views_and_pads_offsets():
    qsl, seq_lens, table = _runner_buffers()
    n = 3
    qsl[: n + 1] = torch.tensor([0, 5, 6, 9], dtype=torch.int32)
    seq_lens[:n] = torch.tensor([40, 7, 12], dtype=torch.int32)
    buffers = PagedBuffers(BOUNDS, "cpu")
    p = buffers.refresh(qsl[: n + 1], seq_lens[:n], table[:n])
    assert p is not None
    assert p.ragged_offset.tolist() == [0, 5, 6, 9, 9, 9, 9, 9, 9]
    assert p.seq_len_q.tolist() == [5, 1, 3, 0, 0, 0, 0, 0]
    # Zero-copy: widened over the runner's own storage.
    assert p.seq_len_kv.shape == (B,) and p.seq_len_kv.data_ptr() == seq_lens.data_ptr()
    assert p.page_table.shape == (B, M) and p.page_table.data_ptr() == table.data_ptr()
    assert p.page_table.stride() == (M, 1)
    # A later in-place runner update is visible without another refresh.
    seq_lens[0] = 41
    assert int(p.seq_len_kv[0]) == 41


def test_refresh_reuses_buffers_across_steps():
    qsl, seq_lens, table = _runner_buffers()
    buffers = PagedBuffers(BOUNDS, "cpu")
    qsl[:3] = torch.tensor([0, 1, 2], dtype=torch.int32)
    first = buffers.refresh(qsl[:3], seq_lens[:2], table[:2])
    qsl[:6] = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    second = buffers.refresh(qsl[:6], seq_lens[:5], table[:5])
    assert first.ragged_offset.data_ptr() == second.ragged_offset.data_ptr()
    assert first.seq_len_q.data_ptr() == second.seq_len_q.data_ptr()
    assert second.seq_len_q.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_refresh_copies_sources_that_cannot_be_widened():
    n = 2
    qsl = torch.tensor([0, 4, 5], dtype=torch.int32)
    seq_lens = torch.tensor([30, 9], dtype=torch.int32)  # own storage: only 2 rows
    table = torch.tensor([[3, 1, 0, 0, 0, 0], [2, 5, 4, 0, 0, 0]], dtype=torch.int32)
    buffers = PagedBuffers(BOUNDS, "cpu")
    p = buffers.refresh(qsl, seq_lens, table)
    assert p.seq_len_kv.tolist() == [30, 9, 0, 0, 0, 0, 0, 0]
    assert p.seq_len_kv.data_ptr() != seq_lens.data_ptr()
    assert p.page_table.shape == (B, M)
    assert p.page_table[:n].tolist() == table.tolist()


@pytest.mark.parametrize(
    "rows,ok",
    [(B, True), (B + 1, False)],
)
def test_refresh_rejects_batches_over_the_bound(rows, ok):
    qsl = torch.arange(rows + 1, dtype=torch.int32)
    seq_lens = torch.ones(rows, dtype=torch.int32)
    table = torch.zeros(rows, M, dtype=torch.int32)
    assert (PagedBuffers(BOUNDS, "cpu").refresh(qsl, seq_lens, table) is not None) is ok


def test_refresh_rejects_missing_block_table_and_wrong_dtypes():
    qsl = torch.arange(3, dtype=torch.int32)
    seq_lens = torch.ones(2, dtype=torch.int32)
    table = torch.zeros(2, M, dtype=torch.int32)
    buffers = PagedBuffers(BOUNDS, "cpu")
    assert buffers.refresh(qsl, seq_lens, None) is None
    assert buffers.refresh(qsl, seq_lens, table.long()) is None
    assert buffers.refresh(qsl.long(), seq_lens, table) is None
