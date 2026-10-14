# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for MLA decode (gfx942 and gfx950).

The CI-collectable form of the MLA decode correctness gate: the dispatched
decode kernel(s) on the local device against the fp64 reference, through the
same harness as ``builders.mla.verify_mla_decode``. It covers the latent kernel
single-pass and split-KV (including more segments than key tiles, which leaves
segments empty), the whole op with its two ``torch.bmm`` projections, and -- on
gfx942 -- the prefill ``absorbed_query`` kernel as an independent GPU oracle.

Specs come from the dispatch factory, never hand-rolled. Every test is marked
``gpu`` and skips unless the device is gfx942 or gfx950 with ROCm torch:

    HIP_VISIBLE_DEVICES=0 python -m pytest -m gpu tests/test_mla_decode_numeric.py
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="ROCm torch required")

_ARCHES = ("gfx942", "gfx950")


def _device_arch() -> str | None:
    """The device's gfx target (``gcnArchName``, not the product name)."""
    if not torch.cuda.is_available():
        return None
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0].lower()
    return arch if arch in _ARCHES else None


ARCH = _device_arch()

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        ARCH is None, reason="needs a gfx942 or gfx950 GPU with ROCm torch"
    ),
]


def _run(name: str, mode: str, num_segs, seed: int = 0) -> bool:
    from builders.mla import verify_mla_decode as v

    sc = v.SCENARIOS[name]
    # The prefill oracle needs a clean cache (see verify_mla_decode --mode
    # oracle); decode itself is always checked against a NaN-poisoned one.
    p = v.make_problem(
        heads=sc.heads,
        q_lens=sc.q_lens,
        k_lens=sc.k_lens,
        seed=seed,
        poison_unused=mode != "oracle",
    )
    return v.run_case(name, p, mode=mode, num_segs=num_segs, arch=ARCH)


@pytest.mark.parametrize(
    "scenario",
    ["ragged_h128", "ragged_h64", "no_query_h128", "zero_keys_h64", "long_h128"],
)
def test_latent_single_pass_and_split(scenario):
    # 64 segments exceeds most tile counts here: trailing segments are empty.
    assert _run(scenario, "latent", ["auto", "1", "7", "64"])


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("scenario", ["ragged_h128", "zero_keys_h128", "no_query_h128"])
def test_whole_op(scenario, seed):
    assert _run(scenario, "full", ["auto", "1", "7"], seed=seed)


def test_prefill_kernel_as_gpu_oracle():
    if ARCH != "gfx942":
        pytest.skip("the prefill absorbed_query kernel is gfx942-only")
    assert _run("ragged_h128", "oracle", ["auto", "1"])
