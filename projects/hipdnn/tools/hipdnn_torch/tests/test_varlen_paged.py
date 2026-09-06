# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Pure-CPU, provider-free tests for the paged varlen override.

Two things are worth pinning here, and neither needs a GPU:

  * **the gate** -- ``VarlenSdpaOverride`` owns its applicability rules rather than
    mirroring cuDNN's (which refuses paged+causal and refuses GQA outright, both of
    which the tiled engine serves). The tests below pin what it declines *and*
    that it does not decline GQA or causal.
  * **the graph mapping** -- the override re-describes torch's packed/paged operands
    as the rank-4 strided views hipDNN wants. The committed integration bundle
    ``SdpaFwd/paged/bf16/hd128_page64_gqa4/Small`` is a geometry the tiled engine
    already accepts, so it is used directly as the oracle: the built graph must
    reproduce its dims, strides, dtypes and attributes exactly. That check is what
    proves the mapping without a device -- a stub frontend records the calls.
"""

import json
import pathlib
from types import SimpleNamespace

import pytest

from hipdnn_torch.varlen import (
    _K_UID,
    _O_UID,
    _PT_K_UID,
    _PT_V_UID,
    _Q_UID,
    _SKV_UID,
    _SQ_UID,
    _V_UID,
    VarlenSdpaOverride,
)

# The committed bundle the tiled engine's own CTest target runs
# (`*SdpaFwd_paged*`). Located relative to this file so the test moves with the
# tree rather than hardcoding a checkout path.
_BUNDLE = (
    pathlib.Path(__file__).resolve().parents[5]
    / "dnn-providers/integration-tests/integration-test-bundles/quick"
    / "SdpaFwd/paged/bf16/hd128_page64_gqa4/Small/Small.json"
)


class _FakeDtype:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"<dtype {self.name}>"


BF16 = _FakeDtype("bf16")
F16 = _FakeDtype("f16")
I32 = _FakeDtype("i32")
INT8 = _FakeDtype("int8")  # not in dtype_map -> "not graph-mappable"


def _contig_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


class _FakeTensor:
    """Minimal tensor supporting the views the override takes (unsqueeze/permute),
    tracked symbolically so dims+strides can be asserted without a real torch."""

    _next_ptr = 0x1000

    def __init__(self, shape, dtype=BF16, is_cuda=True, stride=None, ptr=None):
        self._shape = tuple(shape)
        self.device = "cuda:0"  # only ever passed through to _execute
        self.dtype = dtype
        self.is_cuda = is_cuda
        self._stride = tuple(stride) if stride is not None else _contig_strides(shape)
        if ptr is None:
            _FakeTensor._next_ptr += 0x1000
            ptr = _FakeTensor._next_ptr
        self._ptr = ptr

    @property
    def shape(self):
        return self._shape

    def dim(self):
        return len(self._shape)

    def stride(self, i=None):
        return self._stride if i is None else self._stride[i]

    def data_ptr(self):
        return self._ptr

    def numel(self):
        n = 1
        for s in self._shape:
            n *= s
        return n

    # -- the views the override takes; all share the base pointer (zero copy) --
    def unsqueeze(self, axis):
        assert axis == 0
        return _FakeTensor(
            (1, *self._shape),
            self.dtype,
            self.is_cuda,
            (self.numel(), *self._stride),
            self._ptr,
        )

    def permute(self, *axes):
        return _FakeTensor(
            tuple(self._shape[a] for a in axes),
            self.dtype,
            self.is_cuda,
            tuple(self._stride[a] for a in axes),
            self._ptr,
        )

    def contiguous(self):
        return self

    def to(self, dtype):
        return _FakeTensor(self._shape, dtype, self.is_cuda, self._stride, self._ptr)

    def tolist(self):
        return list(self._values)


# --------------------------------------------------------------------------- #
# Stub hipDNN frontend: records the graph the override builds                  #
# --------------------------------------------------------------------------- #
class _StubTensor:
    def __init__(self):
        self.d = {}

    def set_name(self, v):
        self.d["name"] = v
        return self

    def set_dim(self, v):
        self.d["dims"] = list(v)
        return self

    def set_stride(self, v):
        self.d["strides"] = list(v)
        return self

    def set_data_type(self, v):
        self.d["data_type"] = v
        return self

    def set_uid(self, v):
        self.d["uid"] = v
        return self

    def set_output(self, v):
        self.d["output"] = v
        return self


class _StubAttrs:
    def __init__(self):
        self.calls = []

    def _rec(self, name, val):
        self.calls.append((name, val))

    def set_attn_scale(self, v):
        self._rec("attn_scale_value", v)

    def set_causal_mask(self, v):
        self._rec("causal_mask", v)

    def set_paged_attention_k_table(self, t):
        self._rec("page_table_k", t)

    def set_paged_attention_v_table(self, t):
        self._rec("page_table_v", t)

    def set_paged_attention_max_seq_len_kv(self, v):
        self._rec("max_seq_len_kv", v)

    def set_seq_len_q(self, t):
        self._rec("seq_len_q", t)

    def set_seq_len_kv(self, t):
        self._rec("seq_len_kv", t)

    def set_diagonal_band_left_bound(self, v):
        self._rec("left_bound", v)

    def set_diagonal_band_right_bound(self, v):
        self._rec("right_bound", v)


class _StubGraph:
    def __init__(self):
        self.tensors = []
        self.attrs = None

    def set_io_data_type(self, v):
        self.io = v

    def set_compute_data_type(self, v):
        self.compute = v

    def tensor(self, t):
        self.tensors.append(t)
        return t

    def sdpa(self, q, k, v, attrs):
        self.attrs = attrs
        return [_StubTensor(), None]


def _fake_torch():
    def diff(t):
        vals = t._values
        out = _FakeTensor((len(vals) - 1,), I32)
        out._values = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        return out

    def empty(shape, dtype=None, device=None):
        return _FakeTensor(shape, dtype)

    return SimpleNamespace(
        diff=diff, empty=empty, int32=I32, bfloat16=BF16, float16=F16
    )


def _make():
    ov = VarlenSdpaOverride()
    ov.state = SimpleNamespace(
        torch=_fake_torch(),
        hipdnn=SimpleNamespace(
            Graph=_StubGraph,
            Tensor=_StubTensor,
            SdpaAttributes=_StubAttrs,
            DataType=SimpleNamespace(
                FLOAT="float", BFLOAT16="bfloat16", HALF="float16", INT32="int32"
            ),
        ),
        dtype_map={BF16: "bfloat16", F16: "float16", I32: "int32"},
        select_mode="force",
    )
    ov._installed = True
    return ov


def _seq(values, dtype=I32):
    t = _FakeTensor((len(values),), dtype)
    t._values = list(values)
    return t


# --------------------------------------------------------------------------- #
# The gate                                                                     #
# --------------------------------------------------------------------------- #
_Q = _FakeTensor((64, 16, 128))
_BT = _FakeTensor((2, 32), I32)
_SK = _seq([1024, 2048])


def test_gate_accepts_paged_gqa():
    """cuDNN's own gate refuses GQA outright; ours must not -- the tiled engine
    serves paged+GQA, and deferring would hand us far less than we can serve."""
    ok, reason = _make()._gate(_Q, _BT, _SK, None)
    assert ok, reason


def test_gate_declines_non_cuda():
    q = _FakeTensor((64, 16, 128), is_cuda=False)
    ok, reason = _make()._gate(q, _BT, _SK, None)
    assert not ok and "cuda" in reason


def test_gate_declines_unmappable_dtype():
    q = _FakeTensor((64, 16, 128), INT8)
    ok, reason = _make()._gate(q, _BT, _SK, None)
    assert not ok and "not graph-mappable" in reason


def test_gate_declines_dense_varlen():
    """No block_table is not a paged workload at all -- this override does not
    claim it, and the reason must say so rather than implying a capability gap."""
    ok, reason = _make()._gate(_Q, None, _SK, None)
    assert not ok and "dense varlen" in reason


def test_gate_declines_block_table_without_lengths():
    ok, reason = _make()._gate(_Q, _BT, None, None)
    assert not ok and "seqused_k" in reason


def test_gate_declines_num_splits():
    ok, reason = _make()._gate(_Q, _BT, _SK, 4)
    assert not ok and "num_splits" in reason


@pytest.mark.parametrize("d", [48, 96, 512])
def test_gate_declines_unbaked_head_size(d):
    q = _FakeTensor((64, 16, d))
    ok, reason = _make()._gate(q, _BT, _SK, None)
    assert not ok and "head_size" in reason


@pytest.mark.parametrize("shape", [(16, 128), (1, 64, 16, 128)])
def test_gate_declines_wrong_rank(shape):
    ok, reason = _make()._gate(_FakeTensor(shape), _BT, _SK, None)
    assert not ok and "rank" in reason


# --------------------------------------------------------------------------- #
# The mapping, against the committed bundle                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle not found: {_BUNDLE}")
def test_graph_reproduces_committed_paged_bundle():
    """The override's graph must match a geometry the engine already accepts.

    Every dim, stride and dtype is compared against the committed bundle rather
    than against numbers restated here, so this fails if either side drifts."""
    bundle = json.loads(_BUNDLE.read_text())
    want = {t["name"]: t for t in bundle["tensors"]}
    want_attrs = bundle["nodes"][0]["attributes"]

    # The bundle's exact workload, expressed the way torch hands it to varlen_attn.
    q = _FakeTensor((64, 16, 128))
    k = _FakeTensor((128, 64, 4, 128))
    v = _FakeTensor((128, 64, 4, 128))
    cu_seq_q = _seq([0, 32, 64])
    seqused_k = _seq([1024, 2048])
    block_table = _FakeTensor((2, 32), I32)

    ov = _make()
    built = {}

    def _capture(key, build, describe):
        built["graph"] = build()
        return {"graph": built["graph"], "ws": 0, "engine": "stub"}

    ov._cached_graph = _capture
    ov._execute = lambda entry, variant, device: built.update(variant=variant)

    def _never(*a, **kw):
        raise AssertionError(f"fell back to native: {ov.fallback_reasons()}")

    out = ov._call(
        _never,
        q,
        k,
        v,
        cu_seq_q,
        None,
        32,
        2048,
        scale=want_attrs["attn_scale_value"],
        window_size=(-1, 0),
        seqused_k=seqused_k,
        block_table=block_table,
    )

    got = {t.d["name"]: t.d for t in built["graph"].tensors}
    for name in (
        "Q",
        "K",
        "V",
        "PAGE_TABLE_K",
        "PAGE_TABLE_V",
        "SEQ_LEN_Q",
        "SEQ_LEN_KV",
    ):
        assert got[name]["dims"] == want[name]["dims"], name
        assert got[name]["strides"] == want[name]["strides"], name
        assert got[name]["data_type"] == want[name]["data_type"], name

    recorded = dict(built["graph"].attrs.calls)
    assert recorded["attn_scale_value"] == pytest.approx(want_attrs["attn_scale_value"])
    assert recorded["max_seq_len_kv"] == want_attrs["max_seq_len_kv"]
    assert recorded["left_bound"] == want_attrs["left_bound"]
    assert recorded["right_bound"] == want_attrs["right_bound"]

    # Output follows native's packed contract, not the graph's rank-4 view.
    assert out.shape == (64, 16, 128)


@pytest.mark.skipif(not _BUNDLE.exists(), reason=f"bundle not found: {_BUNDLE}")
def test_operands_are_bound_without_copying():
    """Q/K/V reach the engine as strided views of the caller's own buffers. If a
    permute ever materialises, the variant pack stops matching the base pointer."""
    q = _FakeTensor((64, 16, 128))
    k = _FakeTensor((128, 64, 4, 128))
    v = _FakeTensor((128, 64, 4, 128))

    ov = _make()
    built = {}
    ov._cached_graph = lambda key, build, describe: {
        "graph": build(),
        "ws": 0,
        "engine": "stub",
    }
    ov._execute = lambda entry, variant, device: built.update(variant=variant)
    ov._call(
        lambda *a, **kw: None,
        q,
        k,
        v,
        _seq([0, 32, 64]),
        None,
        32,
        2048,
        scale=0.088,
        window_size=(-1, 0),
        seqused_k=_seq([1024, 2048]),
        block_table=_FakeTensor((2, 32), I32),
    )

    variant = built["variant"]
    assert variant[_Q_UID] == q.data_ptr()
    assert variant[_K_UID] == k.data_ptr()
    assert variant[_V_UID] == v.data_ptr()
    # Both page tables are driven by torch's single block_table.
    assert variant[_PT_K_UID] == variant[_PT_V_UID]
    for uid in (_O_UID, _SQ_UID, _SKV_UID):
        assert uid in variant


def test_offsets_become_lengths():
    """cu_seq_q is cumulative offsets (num_seqs+1); hipDNN wants per-sequence
    lengths. Getting this backwards is the one real arithmetic error available
    here, and it would not fail loudly -- a wrong-but-plausible length vector
    computes the wrong attention rather than raising."""
    ov = _make()
    built = {}

    def _capture(key, build, describe):
        built["graph"] = build()
        return {"graph": built["graph"], "ws": 0, "engine": "stub"}

    ov._cached_graph = _capture
    ov._execute = lambda entry, variant, device: None
    ov._call(
        lambda *a, **kw: None,
        _FakeTensor((96, 16, 128)),
        _FakeTensor((128, 64, 4, 128)),
        _FakeTensor((128, 64, 4, 128)),
        _seq([0, 16, 48, 96]),  # ragged offsets -> lengths 16, 32, 48
        None,
        48,
        2048,
        scale=0.088,
        window_size=(-1, 0),
        seqused_k=_seq([100, 200, 300]),
        block_table=_FakeTensor((3, 32), I32),
    )

    got = {t.d["name"]: t.d for t in built["graph"].tensors}
    # Three sequences, not the four entries the offset vector carries.
    assert got["SEQ_LEN_Q"]["dims"] == [3]
    assert got["SEQ_LEN_KV"]["dims"] == [3]
    assert got["SEQ_LEN_Q"]["data_type"] == "int32"
    assert got["SEQ_LEN_KV"]["data_type"] == "int32"


def test_num_seqs_mismatch_falls_back():
    """A q/kv sequence-count disagreement is a caller error we must not paper
    over: the page table would be indexed per-sequence with the wrong count."""
    ov = _make()
    ov._cached_graph = lambda key, build, describe: {
        "graph": build(),
        "ws": 0,
        "engine": "stub",
    }
    ov._execute = lambda entry, variant, device: None
    sentinel = object()
    out = ov._call(
        lambda *a, **kw: sentinel,
        _FakeTensor((64, 16, 128)),
        _FakeTensor((128, 64, 4, 128)),
        _FakeTensor((128, 64, 4, 128)),
        _seq([0, 32, 64]),  # 2 sequences
        None,
        32,
        2048,
        scale=0.088,
        window_size=(-1, 0),
        seqused_k=_seq([100, 200, 300]),  # 3 sequences
        block_table=_FakeTensor((3, 32), I32),
    )
    assert out is sentinel
    assert any("num_seqs mismatch" in r for r in ov.fallback_reasons())
