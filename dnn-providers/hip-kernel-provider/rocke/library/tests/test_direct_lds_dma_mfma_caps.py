# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one capability gate of the LDS-DMA MFMA direct conv kernels.

``direct_lds_dma_mfma_caps_reason`` (``kernels.common.conv_direct_grouped``) is
what fwd pp / flex, dgrad pp and the tiled and gather wgrad call to decide
whether an arch can run them: wave64, the transposing LDS read (dgrad pp and the
wgrads only), the
16-byte LDS DMA, and ``mfma_f32_16x16x32``. Each stand-in target below is
gfx950 with exactly one of those taken away, so every check is the deciding
refusal in its own case; a check dropped from the helper fails one case.
The C++ twin is covered by ``conv_direct_pp_mfma_host.cpp``.

CPU only.
"""

from __future__ import annotations

from dataclasses import replace
from unittest import mock

import pytest

from kernels.common.conv_direct_dgrad_pp_mfma import (
    DgradPpMfmaSpec,
    dgrad_pp_mfma_problem,
    dgrad_pp_mfma_reason,
)
from kernels.common.conv_direct_fwd_flex_mfma import (
    FwdFlexMfmaSpec,
    fwd_flex_mfma_reason,
)
from kernels.common.conv_direct_fwd_pp_mfma import FwdPpMfmaSpec, fwd_pp_mfma_reason
from kernels.common.conv_direct_grouped import (
    DirectConvProblem,
    direct_lds_dma_mfma_caps_reason,
)
from kernels.common.conv_direct_wgrad_gather import (
    WgradGatherSpec,
    is_valid_wgrad_gather_spec,
)
from kernels.common.conv_direct_wgrad_tiled import (
    DirectWgradTiledSpec,
    is_valid_wgrad_tiled_spec,
)
from rocke.core.arch import ArchTarget


class _NoMfma16x16x32:
    """A catalog without the 16x16x32 MFMA (everything else answers as gfx950)."""

    def __init__(self, inner):
        self._inner = inner

    def has_shape(self, **kw):
        if (kw.get("m"), kw.get("n"), kw.get("k")) == (16, 16, 32):
            return False
        return self._inner.has_shape(**kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _standins():
    t = ArchTarget.from_gfx("gfx950")
    assert t.wave_size == 64 and t.memory.has_ds_read_tr
    assert t.async_lds_max_dwords >= 4
    return {
        "wave32": replace(t, wave_size=32),
        "no_ds_read_tr": replace(t, memory=replace(t.memory, has_ds_read_tr=False)),
        "async_2_dwords": replace(t, memory=replace(t.memory, async_lds_max_dwords=2)),
        "no_mfma_16x16x32": replace(t, mma=_NoMfma16x16x32(t.mma)),
    }


def _want(family, arch, dtype="bf16"):
    ab = "bf16" if dtype == "bf16" else "f16"
    return {
        "wave32": f"{family} is a wave64 MFMA kernel",
        "no_ds_read_tr": f"{family} needs ds_read_b64_tr_b16 (absent on {arch})",
        "async_2_dwords": f"{family} needs the 16-byte LDS DMA (absent on {arch})",
        "no_mfma_16x16x32": f"missing mfma_f32_16x16x32_{ab} on {arch}",
    }


def _p(C=96, K=136, dtype="bf16"):
    return DirectConvProblem(
        N=2,
        H=37,
        W=41,
        groups=1,
        cpg=C,
        kpg=K,
        KH=3,
        KW=3,
        PAD=1,
        stride=1,
        dtype=dtype,
    )


def test_gfx950_has_every_capability():
    t = ArchTarget.from_gfx("gfx950")
    for tr in (False, True):
        for bf16 in (False, True):
            assert (
                direct_lds_dma_mfma_caps_reason(
                    t, "gfx950", family="f", bf16=bf16, needs_ds_read_tr=tr
                )
                is None
            )


@pytest.mark.parametrize("needs_tr", [False, True])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("name", list(_standins()))
def test_helper_each_capability_decides(name, dtype, needs_tr):
    t = _standins()[name]
    got = direct_lds_dma_mfma_caps_reason(
        t, "gfx950", family="k", bf16=dtype == "bf16", needs_ds_read_tr=needs_tr
    )
    if name == "no_ds_read_tr" and not needs_tr:
        assert got is None
    else:
        assert got == _want("k", "gfx950", dtype)[name]


def test_wave_size_of_the_kernel_is_gated_too():
    t = ArchTarget.from_gfx("gfx950")
    assert direct_lds_dma_mfma_caps_reason(
        t, "gfx950", family="k", bf16=True, wave_size=32
    ) == ("k is a wave64 MFMA kernel")


def _callers():
    """(label, reason(arch) -> str | None, family, needs_ds_read_tr)."""
    fwd = FwdPpMfmaSpec(problem=_p())
    dgrad = DgradPpMfmaSpec(problem=dgrad_pp_mfma_problem(_p(136, 96)))
    flex = FwdFlexMfmaSpec(problem=_p(192, 136))
    wgrad = DirectWgradTiledSpec(problem=_p(64, 64))
    gather = WgradGatherSpec(
        problem=DirectConvProblem(
            N=2, H=16, W=32, groups=1, cpg=3, kpg=24, stride=2, dtype="bf16"
        )
    )

    def ok_or_why(valid):
        return lambda a: (lambda ok, why: None if ok else why)(*valid(a))

    return (
        ("fwd_pp", lambda a: fwd_pp_mfma_reason(fwd, a), "fwd_pp_mfma", False),
        ("dgrad_pp", lambda a: dgrad_pp_mfma_reason(dgrad, a), "dgrad_pp_mfma", True),
        ("fwd_flex", lambda a: fwd_flex_mfma_reason(flex, a), "fwd_flex_mfma", False),
        (
            "wgrad_tiled",
            ok_or_why(lambda a: is_valid_wgrad_tiled_spec(wgrad, a)),
            "wgrad_tiled",
            True,
        ),
        (
            "wgrad_gather",
            ok_or_why(lambda a: is_valid_wgrad_gather_spec(gather, a)),
            "wgrad_gather",
            True,
        ),
    )


@pytest.mark.parametrize("name", list(_standins()))
@pytest.mark.parametrize("caller", [c[0] for c in _callers()])
def test_every_caller_uses_the_shared_gate(caller, name):
    """Each kernel's validator refuses a stand-in exactly as the helper does."""
    label, reason, family, needs_tr = next(c for c in _callers() if c[0] == caller)
    standin = _standins()[name]
    with mock.patch.object(ArchTarget, "from_gfx", return_value=standin):
        got = reason("gfx950")
    if name == "no_ds_read_tr" and not needs_tr:
        # Not a capability of this kernel: the gate lets it through.
        assert got is None or "ds_read_b64_tr_b16" not in got
    else:
        assert got == _want(family, "gfx950")[name]


@pytest.mark.parametrize("caller", [c[0] for c in _callers()])
def test_real_gfx950_passes_every_caller(caller):
    _, reason, _, _ = next(c for c in _callers() if c[0] == caller)
    assert reason("gfx950") is None


@pytest.mark.parametrize(
    "arch, want",
    [
        ("gfx942", "{f} needs ds_read_b64_tr_b16 (absent on gfx942)"),
        ("gfx1100", "{f} is a wave64 MFMA kernel"),
    ],
)
@pytest.mark.parametrize("caller", ["wgrad_tiled", "wgrad_gather"])
def test_wgrad_callers_on_real_arches(caller, arch, want):
    """The C++ host test ``conv_direct_pp_mfma_host`` pins the same texts."""
    _, reason, family, _ = next(c for c in _callers() if c[0] == caller)
    assert reason(arch) == want.format(f=family)
