# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host checks for the dense runners' optional-output error boundary."""

from dataclasses import replace

import pytest

from kernels.gfx950 import attention_dense as gfx950_dense

_RUNNERS = {
    "gfx950": (gfx950_dense.Gfx950AttentionDenseSpec, gfx950_dense),
}


def _spec(arch):
    spec_type, _ = _RUNNERS[arch]
    return spec_type(
        batch=1,
        seqlen_q=256,
        seqlen_kv=256,
        num_query_heads=4,
        num_kv_heads=1,
        head_size=128,
        dtype="bf16",
        emit_lse=True,
    )


@pytest.mark.parametrize("arch", sorted(_RUNNERS))
def test_missing_lse_output_rejected_before_input_access(arch):
    """Presence validation must fail before compilation or dereferencing inputs."""
    with pytest.raises(ValueError):
        _RUNNERS[arch][1].run_attention_dense_torch(
            spec=_spec(arch),
            q=None,
            k=None,
            v=None,
            out=None,
            scale=0.5,
        )


@pytest.mark.parametrize("arch", sorted(_RUNNERS))
def test_lse_output_rejected_when_not_requested(arch):
    with pytest.raises(ValueError):
        _RUNNERS[arch][1].run_attention_dense_torch(
            spec=replace(_spec(arch), emit_lse=False),
            q=None,
            k=None,
            v=None,
            out=None,
            lse=object(),
            scale=0.5,
        )
