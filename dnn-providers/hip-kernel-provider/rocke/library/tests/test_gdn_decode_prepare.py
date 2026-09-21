# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side input-validation guard for the GDN decode kernel, without a GPU.

``prepare`` rejects inputs that would make the kernel read or write outside the
state pool. The decode kernel bounds-checks nothing on device beyond the ``-1``
skip sentinel, so this host guard *is* the memory-safety contract. The checks
raise before any launch, so they are pure host logic that runs on a CPU box --
which is exactly where a "this guard must not be silently dropped" regression
test belongs, rather than behind the on-device ``gpu`` gate.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="torch required (CPU build is fine)")

from builders.gfx950.gdn.gdn_decode import make_inputs, prepare
from kernels.gfx950.gdn_decode import GdnDecodeSpec

DEVICE = "cpu"


def test_out_of_range_index_is_rejected():
    """An index past the pool depth is an OOB access; prepare() must refuse it.

    ``-1`` stays legal (skip); any other out-of-pool value is rejected before a
    launch can touch it, for both the read and the write index.
    """
    spec = GdnDecodeSpec()
    batch = 8
    pool_depth = make_inputs(spec, batch, device=DEVICE)["state"].shape[0]

    for name, bad in (
        ("read_indices", pool_depth),  # == depth: the first OOB slot
        ("write_indices", pool_depth + 5),
        ("read_indices", -2),  # below the -1 skip sentinel
        ("write_indices", -2),
    ):
        inp = make_inputs(spec, batch, device=DEVICE)
        inp[name][0] = bad
        with pytest.raises(ValueError, match="out of range"):
            prepare(spec, inp, batch)


def test_duplicate_active_write_index_is_rejected():
    """Two live sequences cannot race to update one recurrent-state page."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["write_indices"][1] = inp["write_indices"][0]
    with pytest.raises(ValueError, match="unique across active sequences"):
        prepare(spec, inp, batch)


def test_inactive_mismatched_lane_does_not_reserve_a_write_index():
    """A lane with either negative index is inactive and cannot claim a page."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][1] = -1
    inp["write_indices"][1] = inp["write_indices"][0]
    prepare(spec, inp, batch)  # must not raise: lane 1 is inactive


def test_the_skip_sentinel_is_accepted():
    """``-1`` marks an idle continuous-batching slot and must pass the guard."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][1::2] = -1
    inp["write_indices"][1::2] = -1
    prepare(spec, inp, batch)  # must not raise


def test_wrong_state_head_dims_are_rejected():
    """A state pool whose head dims disagree with the spec is a shape bug, and
    the check is sync-free so it runs regardless of the value-range flag."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    # Drop a slice of the K dim so the pool no longer matches the spec.
    inp["state"] = inp["state"][..., :-8].contiguous()
    with pytest.raises(ValueError, match="head dims"):
        prepare(spec, inp, batch, validate_indices=False)


def test_wrong_state_dtype_is_rejected():
    """A pool whose element type disagrees with the spec must be refused here.

    ``bf16`` and ``f16`` are both 16 bits, so a mismatched pool has the right
    shape *and* the right byte size: every address the kernel computes is
    identical and nothing faults. The kernel is compiled with a fixed pointer
    element type and gets no dtype tag at runtime, so it simply decodes the
    bits under the wrong rule -- and decode writes that value back into the
    pool, compounding it over the whole generation. Nothing on device can
    catch it, which is why the host guard must.
    """
    spec = GdnDecodeSpec()
    assert spec.state_dtype == "bf16", "test assumes the default spec state dtype"
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["state"] = inp["state"].to(torch.float16)
    # Same shape, same byte count -- only the element type differs.
    with pytest.raises(ValueError, match="state dtype"):
        prepare(spec, inp, batch, validate_indices=False)


def test_validate_indices_flag_skips_the_range_check():
    """The value-range check reads the index extrema (a device sync on GPU), so
    it is flag-gated for the hot path.

    With it off, prepare() does not inspect the values and an out-of-pool index
    slips past; the sync-free shape checks still run. This pins the flag
    contract so the default-on guard cannot be silently lost.
    """
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["read_indices"][0] = inp["state"].shape[0]  # OOB, but unchecked
    prepare(spec, inp, batch, validate_indices=False)


def test_mis_shaped_input_tensor_is_rejected():
    """Every address is computed from SPEC dims, and the kernel emits no buffer
    descriptor -- there is no `num_records` to clamp an over-reach -- so a
    tensor whose real shape is smaller than the spec says is an out-of-bounds
    read with nothing between it and other allocations.
    """
    spec = GdnDecodeSpec()
    batch = 8
    for name in ("query", "key", "value", "a", "b", "dt_bias", "A_log"):
        inp = make_inputs(spec, batch, device=DEVICE)
        inp[name] = inp[name][..., :-1]  # one element short on the last axis
        with pytest.raises(ValueError, match=f"{name} must be"):
            prepare(spec, inp, batch)


def test_wrong_input_dtype_is_rejected():
    """The kernel is compiled against a fixed element type; handing it other
    bytes reinterprets them rather than converting."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    inp["query"] = inp["query"].to(torch.float32)
    with pytest.raises(ValueError, match="query dtype"):
        prepare(spec, inp, batch)


def test_non_contiguous_input_is_rejected():
    """The doc promises row-major and every offset assumes it. A transposed
    view has the right shape and the wrong memory order, so the kernel would
    read the right INDEX out of the wrong ADDRESS -- silently."""
    spec = GdnDecodeSpec()
    batch = 8
    inp = make_inputs(spec, batch, device=DEVICE)
    # Same shape, non-unit strides: transpose two axes and transpose back via
    # a view that keeps the permuted layout.
    inp["value"] = inp["value"].transpose(2, 3).contiguous().transpose(2, 3)
    assert not inp["value"].is_contiguous()
    with pytest.raises(ValueError, match="contiguous"):
        prepare(spec, inp, batch)
