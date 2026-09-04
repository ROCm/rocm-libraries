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
@pytest.mark.parametrize("Hv,Hk", [(4, 4), (8, 4)])
@pytest.mark.parametrize("gate_low", [-0.5, -5.0])
@pytest.mark.parametrize("with_h0", [False, True])
def test_gdn_split_parity(Hv, Hk, gate_low, with_h0):
    from builders.gfx950.kda.gdn_prefill import check_gdn

    worst = check_gdn(2, Hv, Hk, 256, 128, 128, gate_low=gate_low, with_h0=with_h0)
    assert worst < 3e-2, (
        f"GDN split Hv{Hv}/Hk{Hk} gate{gate_low} h0={with_h0}: rel {worst}"
    )
