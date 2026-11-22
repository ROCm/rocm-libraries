# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The MoE sort groups every (token, top-k slot) pair under its expert on the GPU.

Given router output ``TopkIds`` / ``TopkWeights`` of shape ``(tokens, topk)``,
both MoE sort paths -- the three-launch histogram -> scan -> scatter chain and
the single-kernel persistent sort -- write:

* ``Counts[e]``: how many pairs route to expert ``e`` (0 for an expert no token
  picked);
* ``Offsets[e]``: the exclusive prefix sum of ``Counts``, where expert ``e``'s
  run starts;
* ``SortedTokenIds`` / ``SortedTopkIds`` / ``SortedWeights``: expert ``e``'s
  pairs, as (token, slot, weight), in ``[Offsets[e], Offsets[e] + Counts[e])``.

The scatter claims slots with atomics, so the order inside one expert's run is
not fixed; the reference compares each run as a set of (token, slot, weight)
triples, built in numpy from the routing table alone. The configurations cover
experts that receive no pair, expert 0 picked by every other token, more pairs
than one block, and expert counts below and above the wave size (the scan's
one-wave and LDS paths).
"""

from __future__ import annotations

import numpy as np
import pytest
from rocke.helpers.compile import compile_kernel
from rocke.instances.common.moe_sorting import (
    MoeSortingSpec,
    build_moe_sort_histogram,
    build_moe_sort_persistent,
    build_moe_sort_scan,
    build_moe_sort_scatter,
    is_valid_spec,
    moe_sort_histogram_grid,
    moe_sort_histogram_signature,
    moe_sort_persistent_grid,
    moe_sort_persistent_signature,
    moe_sort_scan_grid,
    moe_sort_scan_signature,
    moe_sort_scatter_grid,
    moe_sort_scatter_signature,
)
from rocke.tests.gpu import current_device_arch_in, launch_with_buffers

# The arches this test has run green on; add one once it has.
_RUN_ARCHES = ("gfx90a", "gfx942", "gfx950")

_SPECS = [
    # Decode-sized: 16 pairs over 64 experts, so most experts get none.
    MoeSortingSpec(tokens=2, topk=8, experts=64, block_size=64),
    # More pairs than one block, experts above the wave size (LDS scan).
    MoeSortingSpec(tokens=300, topk=4, experts=128, block_size=256),
    MoeSortingSpec(tokens=97, topk=2, experts=8, block_size=128),
]


def _routing(spec: MoeSortingSpec) -> tuple[np.ndarray, np.ndarray]:
    """Distinct experts per token, skewed so expert 0 takes many pairs and the
    top quarter of the experts none."""
    rng = np.random.default_rng(spec.tokens * 1000 + spec.experts)
    reachable = max(spec.topk, (3 * spec.experts) // 4)
    ids = np.empty((spec.tokens, spec.topk), dtype=np.int32)
    for token in range(spec.tokens):
        rest = rng.choice(np.arange(1, reachable), spec.topk - 1, replace=False)
        ids[token] = (
            np.concatenate([[0], rest])
            if token % 2
            else rng.choice(reachable, spec.topk, replace=False)
        )
    weights = rng.uniform(0.0, 1.0, ids.shape).astype(np.float32)
    return ids, weights


def _check(spec, ids, weights, out):
    counts, offsets = out["Counts"], out["Offsets"]
    tokens, slots, sorted_weights = (
        out["SortedTokenIds"],
        out["SortedTopkIds"],
        out["SortedWeights"],
    )
    want_counts = np.bincount(ids.ravel(), minlength=spec.experts)
    np.testing.assert_array_equal(counts, want_counts)
    np.testing.assert_array_equal(offsets, np.cumsum(want_counts) - want_counts)
    assert (want_counts == 0).any(), "the routing leaves no expert empty"
    for expert in range(spec.experts):
        run = slice(offsets[expert], offsets[expert] + counts[expert])
        got = sorted(zip(tokens[run], slots[run], sorted_weights[run]))
        token_ids, slot_ids = np.nonzero(ids == expert)
        want = sorted(zip(token_ids, slot_ids, weights[token_ids, slot_ids]))
        assert got == want, f"expert {expert}: got {got[:4]}..., want {want[:4]}..."


def _outputs(spec: MoeSortingSpec) -> dict[str, np.ndarray]:
    pairs = spec.total_pairs
    return {
        "Offsets": np.full(spec.experts, -1, np.int32),
        "Counts": np.full(spec.experts, -1, np.int32),
        "SortedTokenIds": np.full(pairs, -1, np.int32),
        "SortedTopkIds": np.full(pairs, -1, np.int32),
        "SortedWeights": np.full(pairs, np.nan, np.float32),
    }


def _launch(kernel, arch, signature, buffers, grid, spec, scalars):
    launch_with_buffers(
        compile_kernel(kernel, arch=arch),
        signature,
        buffers,
        grid=grid,
        block=(spec.block_size, 1, 1),
        scalars=scalars,
    )


@pytest.mark.parametrize("spec", _SPECS, ids=lambda spec: spec.kernel_name("all"))
def test_three_launch_sort_groups_pairs_by_expert_on_gpu(spec):
    arch = current_device_arch_in(_RUN_ARCHES, subject="moe_sorting")
    assert is_valid_spec(spec, arch) == (True, "ok")
    ids, weights = _routing(spec)
    out = _outputs(spec)
    hist = np.zeros(spec.experts, np.int32)
    experts = {"num_experts": spec.experts}
    _launch(
        build_moe_sort_histogram(spec, arch),
        arch,
        moe_sort_histogram_signature(spec),
        {"TopkIds": ids, "Hist": hist},
        moe_sort_histogram_grid(spec),
        spec,
        {"num_pairs": spec.total_pairs, **experts},
    )
    _launch(
        build_moe_sort_scan(spec, arch),
        arch,
        moe_sort_scan_signature(spec),
        {"Hist": hist, "Offsets": out["Offsets"], "Counts": out["Counts"]},
        moe_sort_scan_grid(spec),
        spec,
        experts,
    )
    _launch(
        build_moe_sort_scatter(spec, arch),
        arch,
        moe_sort_scatter_signature(spec),
        {
            "TopkIds": ids,
            "TopkWeights": weights,
            "Offsets": out["Offsets"],
            "Counter": np.zeros(spec.experts, np.int32),
            "SortedTokenIds": out["SortedTokenIds"],
            "SortedTopkIds": out["SortedTopkIds"],
            "SortedWeights": out["SortedWeights"],
        },
        moe_sort_scatter_grid(spec),
        spec,
        {"tokens": spec.tokens, "topk": spec.topk, **experts},
    )
    _check(spec, ids, weights, out)


@pytest.mark.parametrize("spec", _SPECS, ids=lambda spec: spec.kernel_name("all"))
def test_persistent_sort_groups_pairs_by_expert_on_gpu(spec):
    arch = current_device_arch_in(_RUN_ARCHES, subject="moe_sorting")
    assert is_valid_spec(spec, arch) == (True, "ok")
    ids, weights = _routing(spec)
    out = _outputs(spec)
    _launch(
        build_moe_sort_persistent(spec, arch),
        arch,
        moe_sort_persistent_signature(spec),
        {"TopkIds": ids, "TopkWeights": weights, **out},
        moe_sort_persistent_grid(spec),
        spec,
        {"tokens": spec.tokens, "topk": spec.topk, "num_experts": spec.experts},
    )
    _check(spec, ids, weights, out)
