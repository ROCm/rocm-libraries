# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the ATen-level paged route.

These need a real ``torch`` (the dispatcher is the thing under test) but **not** a
GPU or a provider: the route is registered on the CPU key, and ``run_paged`` is
stubbed, so what is exercised is the registration and argument plumbing rather
than the kernel.

Two of these pin bugs that were real, not hypothetical:

  * ``test_native_fallback_does_not_recurse`` -- a fallback that simply re-calls
    the op re-enters this very kernel and recurses until the stack dies.
  * ``test_key_tensor_reaches_run_paged`` -- the op's second positional argument
    is the K tensor, which is easy to shadow with the dispatch key of the same
    name; the result is a K tensor that is silently not a tensor.
"""

import pytest

torch = pytest.importorskip("torch")

from hipdnn_torch.varlen import VarlenSdpaOverride  # noqa: E402
from hipdnn_torch.varlen_aten import AtenVarlenRoute  # noqa: E402

_OP = "_flash_attention_forward"

_HAS_PAGED_OP = hasattr(torch.ops.aten, _OP) and {
    "block_table",
    "seqused_k",
}.issubset({a.name for a in getattr(torch.ops.aten, _OP).default._schema.arguments})

pytestmark = pytest.mark.skipif(
    not _HAS_PAGED_OP,
    reason=f"torch {torch.__version__} has no paged aten::{_OP} (needs >= 2.12)",
)


class _Recorder(VarlenSdpaOverride):
    """Stands in for the real mapping: records what the ATen kernel forwarded and
    declines, so the fallback path is exercised too."""

    def __init__(self, result=None):
        super().__init__()
        self.seen = None
        self._result = result

    def run_paged(self, query, key, value, cu_seq_q, max_k, **kw):
        self.seen = {
            "query": query,
            "key": key,
            "value": value,
            "max_k": max_k,
            "window": kw.get("window"),
            "block_table": kw.get("block_table"),
            "seqused_k": kw.get("seqused_k"),
        }
        return self._result


def _cpu_route(override):
    """Install the route on the CPU key. The routing logic is key-agnostic; CPU is
    used so the test needs no device."""
    route = AtenVarlenRoute(override)
    override.state = type("S", (), {"torch": torch})()

    op = torch.ops.aten._flash_attention_forward
    key_set = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)

    def _impl(
        query,
        key_t,
        value,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        dropout_p,
        is_causal,
        return_debug_mask,
        scale=None,
        window_size_left=None,
        window_size_right=None,
        seqused_k=None,
        alibi_slopes=None,
        block_table=None,
        num_splits=None,
    ):
        def _native():
            with torch._C._ExcludeDispatchKeyGuard(key_set):
                return op(
                    query,
                    key_t,
                    value,
                    cum_seq_q,
                    cum_seq_k,
                    max_q,
                    max_k,
                    dropout_p,
                    is_causal,
                    return_debug_mask,
                    scale=scale,
                    window_size_left=window_size_left,
                    window_size_right=window_size_right,
                    seqused_k=seqused_k,
                    alibi_slopes=alibi_slopes,
                    block_table=block_table,
                    num_splits=num_splits,
                )

        left = -1 if window_size_left is None else int(window_size_left)
        right = (
            0
            if is_causal
            else (-1 if window_size_right is None else int(window_size_right))
        )
        out = override.run_paged(
            query,
            key_t,
            value,
            cum_seq_q,
            max_k,
            scale=scale,
            window=(left, right),
            seqused_k=seqused_k,
            block_table=block_table,
            num_splits=num_splits,
            census_key="aten",
        )
        if out is None:
            return _native()
        lse = torch.empty(
            (int(query.shape[1]), int(query.shape[0])), dtype=torch.float32
        )
        rng = torch.empty(2, dtype=torch.uint64)
        empty = torch.empty(0, dtype=query.dtype)
        return out, lse, rng, empty, empty

    lib = torch.library.Library("aten", "IMPL")
    lib.impl(_OP, _impl, "CPU")
    route._lib = lib
    route._installed = True
    return route


@pytest.fixture(autouse=True)
def _release_registrations():
    """A torch.library registration lives until its Library is collected, and a
    second registration on the same key raises rather than replacing. Force the
    release between tests so each one installs into a clean dispatcher."""
    yield
    import gc

    gc.collect()


def _operands():
    q = torch.zeros(64, 16, 128)
    k = torch.zeros(128, 64, 4, 128)
    v = torch.zeros(128, 64, 4, 128)
    cu = torch.tensor([0, 32, 64], dtype=torch.int32)
    sk = torch.tensor([1024, 2048], dtype=torch.int32)
    bt = torch.zeros(2, 32, dtype=torch.int32)
    return q, k, v, cu, sk, bt


def test_paged_arguments_reach_run_paged():
    """A raw dispatcher call must arrive with its page table and KV lengths
    intact -- that is the entire reason this route exists alongside the wrapper."""
    rec = _Recorder(result=None)
    route = _cpu_route(rec)
    assert route.installed
    q, k, v, cu, sk, bt = _operands()

    with pytest.raises(Exception):  # declines -> native, which has no CPU kernel
        torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu,
            None,
            32,
            2048,
            0.0,
            False,
            False,
            scale=0.088,
            window_size_left=-1,
            window_size_right=0,
            seqused_k=sk,
            block_table=bt,
        )

    assert rec.seen is not None, "dispatcher call never reached the hipDNN route"
    assert rec.seen["block_table"] is bt
    assert rec.seen["seqused_k"] is sk
    assert rec.seen["max_k"] == 2048


def test_key_tensor_reaches_run_paged():
    """The K operand must arrive as the tensor, not shadowed by the dispatch key
    of the same name."""
    rec = _Recorder(result=None)
    route = _cpu_route(rec)
    assert route.installed
    q, k, v, cu, sk, bt = _operands()

    with pytest.raises(Exception):
        torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu,
            None,
            32,
            2048,
            0.0,
            False,
            False,
            seqused_k=sk,
            block_table=bt,
        )

    assert torch.is_tensor(rec.seen["key"])
    assert rec.seen["key"].shape == k.shape


def test_causal_maps_to_bottom_of_band():
    """is_causal must reach the mapping as the (-1, 0) window the committed paged
    bundle carries, not as an unbounded band."""
    rec = _Recorder(result=None)
    route = _cpu_route(rec)
    assert route.installed
    q, k, v, cu, sk, bt = _operands()

    with pytest.raises(Exception):
        torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu,
            None,
            32,
            2048,
            0.0,
            True,
            False,
            seqused_k=sk,
            block_table=bt,
        )
    assert rec.seen["window"] == (-1, 0)


def test_native_fallback_does_not_recurse():
    """Declining must reach the kernel we displaced. Re-calling the op instead
    re-enters this kernel; the symptom is a stack overflow, not a wrong number."""
    calls = {"n": 0}

    class _Counting(_Recorder):
        def run_paged(self, *a, **kw):
            calls["n"] += 1
            if calls["n"] > 5:
                raise RecursionError("native fallback re-entered the hipDNN kernel")
            return super().run_paged(*a, **kw)

    rec = _Counting(result=None)
    route = _cpu_route(rec)
    assert route.installed
    q, k, v, cu, sk, bt = _operands()

    with pytest.raises(Exception) as excinfo:
        torch.ops.aten._flash_attention_forward(
            q,
            k,
            v,
            cu,
            None,
            32,
            2048,
            0.0,
            False,
            False,
            seqused_k=sk,
            block_table=bt,
        )
    assert not isinstance(excinfo.value, RecursionError)
    assert calls["n"] == 1, f"kernel re-entered {calls['n']} times"


def test_served_call_returns_the_ops_five_tuple():
    """When the graph serves the call, the op's return contract still applies:
    five values, output first."""
    out = torch.zeros(64, 16, 128)
    rec = _Recorder(result=out)
    route = _cpu_route(rec)
    assert route.installed
    q, k, v, cu, sk, bt = _operands()

    result = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        cu,
        None,
        32,
        2048,
        0.0,
        False,
        False,
        seqused_k=sk,
        block_table=bt,
    )
    assert len(result) == 5
    assert result[0] is out
    assert result[1].shape == (16, 64)  # logsumexp is [H, total_q]


def test_install_refuses_a_torch_without_paged_arguments():
    """On an older wheel the op exists but lacks block_table/seqused_k.
    Registering anyway would silently never see a page table."""

    class _OldTorch:
        __version__ = "2.11.0"

        class ops:
            class aten:
                pass

    route = AtenVarlenRoute(VarlenSdpaOverride())
    route.override.state = type("S", (), {"torch": _OldTorch})()
    with pytest.raises(ImportError, match=_OP):
        route.install()
