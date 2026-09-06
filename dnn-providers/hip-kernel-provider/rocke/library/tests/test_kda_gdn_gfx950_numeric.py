# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the GDN mode of the KDA chunkwise prefill kernel.

GDN is KDA with a scalar softplus forget-gate on the split (raw-prep) path. Each
case is checked against a token-serial float64 oracle fed the broadcast scalar
decay, with q/k expanded from Hk key-heads to Hv value-heads. Covers MHA
(kv_group=1) and GQA (kv_group=2), the gate range to -5, and +/- initial state.

Marked ``gpu`` and skipped off a gfx950.
"""

from __future__ import annotations

import pytest


def _gpu_ready():
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    return "gfx950" in torch.cuda.get_device_properties(0).gcnArchName.lower()


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)


@pytest.mark.gpu
@requires_gfx950_gpu
@pytest.mark.parametrize("Hv,Hk", [(4, 4), (8, 4), (32, 8)])
@pytest.mark.parametrize("gate_low", [-0.5, -5.0])
@pytest.mark.parametrize("with_h0", [False, True])
def test_gdn_split_parity(Hv, Hk, gate_low, with_h0):
    from builders.gfx950.kda.gdn_prefill import check_gdn

    worst = check_gdn(2, Hv, Hk, 256, 128, 128, gate_low=gate_low, with_h0=with_h0)
    assert (
        worst < 3e-2
    ), f"GDN split Hv{Hv}/Hk{Hk} gate{gate_low} h0={with_h0}: rel {worst}"


@pytest.mark.gpu
@requires_gfx950_gpu
@pytest.mark.parametrize(
    "batch,expected_value_splits",
    [
        (8, 8),  # BH = 8*8  = 64  -> value_splits=8
        (16, 2),  # BH = 16*8 = 128 -> value_splits=2
        (32, 1),  # BH = 32*8 = 256 -> value_splits=1
    ],
)
def test_gdn_prefill_dispatched_value_splits(batch, expected_value_splits):
    """The dispatcher's tuned value_splits table launches correct kernels.

    Holds a validated GQA shape (Hv=8/Hk=4, kv_group=2) fixed and varies the
    batch so BH sweeps the value_splits bands; each split half's spec comes from
    ``dispatch_gdn_prefill`` (not the builder) and is checked against the same
    float64 oracle. Isolating the table from head-shape effects: see the
    documented Hv32/Hk8 GQA gather gap in the vault.
    """
    import dataclasses

    from builders.gfx950.kda.gdn_prefill import check_gdn
    from dispatch.gdn import GdnPrefillRequest, dispatch_gdn_prefill

    Hv, Hk, T, DK, DV = 8, 4, 256, 128, 128
    scan_req = GdnPrefillRequest(
        batch=batch,
        seqlen=T,
        arch="gfx950",
        num_k_heads=Hk,
        num_v_heads=Hv,
        head_k_dim=DK,
        head_v_dim=DV,
        algorithm="chunk_scan",
    )
    scan = dispatch_gdn_prefill(scan_req).spec
    prep = dispatch_gdn_prefill(
        dataclasses.replace(scan_req, algorithm="chunk_prep")
    ).spec
    assert scan.value_splits == expected_value_splits
    assert prep.gate_kind == "gdn" and prep.kv_group == Hv // Hk

    worst = check_gdn(batch, Hv, Hk, T, DK, DV, specs=(scan, prep))
    assert worst < 3e-2, (
        f"dispatched GDN prefill B{batch} vs{expected_value_splits} "
        f"(Hv{Hv}/Hk{Hk}): rel {worst}"
    )
