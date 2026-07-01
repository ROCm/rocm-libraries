# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared parity-harness infrastructure (arch selection, compile, Result).

Both the platform extended-kernel harness (parity_extended_kernels) and the
library attention parity harness (builders.common.parity_fmha_extended) import
from this module.  It is pure-platform: it imports only from ``rocke.*`` and
stdlib — never from ``kernels``.

Consuming modules are expected to:

1. Import this module as ``import rocke.examples.common._parity_harness_common
   as _phc``.
2. Keep a local ``_ARCH`` module variable that is set in their ``main()`` from
   ``--arch`` / ``_default_arch()``.
3. After setting local ``_ARCH``, sync it back: ``_phc._ARCH = _ARCH``.  This
   keeps ``_phc._compile`` and ``_phc._require_ocp_fp8_arch`` in sync with the
   same arch as builder calls that pass ``arch=_ARCH`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch  # noqa: E402 (torch must be imported after HIP init in calling code)

from rocke.helpers import compile_kernel  # noqa: E402
from rocke.runtime.launcher import KernelLauncher, LaunchConfig  # noqa: F401,E402


# ---------------------------------------------------------------------------
# Arch selection
# ---------------------------------------------------------------------------


def _default_arch() -> str:
    """Return the running device's gfx arch, falling back to ``gfx950``."""
    try:
        from rocke.runtime.hip_module import get_device_arch

        return get_device_arch() or "gfx950"
    except Exception:  # noqa: BLE001 - no device / import issue: fall back
        return "gfx950"


# Target gfx arch for all compiles.  Consuming modules mutate this via
# ``_phc._ARCH = _ARCH`` in their ``main()`` before running any cases.
_ARCH = "gfx950"


def _compile(kernel):
    """Compile *kernel* for the harness-selected arch (``_ARCH``)."""
    return compile_kernel(kernel, arch=_ARCH)


def _require_ocp_fp8_arch(case: str) -> None:
    """Raise a gfx950-only SKIP for OCP-fp8 (e4m3fn) parity cases on gfx942.

    These cases dequantise the KV / operand tensors with the hardware
    ``v_cvt_f32_fp8`` family and compare against a torch ``float8_e4m3fn``
    (OCP, exp-bias 7) reference. On CDNA4 (gfx950 / MI350) ``cvt_f32_fp8``
    decodes the byte as OCP e4m3fn, matching torch bit-for-bit. On CDNA3
    (gfx942 / MI300) the *same* intrinsic decodes the byte as the legacy
    AMD ``e4m3fnuz`` format (exp-bias 8, 0x80 == NaN), so the hardware and
    the OCP torch reference disagree (and 0x80 bytes surface as NaN). The
    MFMA atom and the kernel itself build + run fine on gfx942 -- this is
    purely an fp8 *byte-format* mismatch, so the OCP-reference parity check
    is legitimately gfx950-only. (i8 / i4 sage variants stay green on both:
    they feed an f32 codebook and apply the identical fp8 round-trip in
    kernel + reference, so no native-byte interpretation is involved.)
    """
    if _ARCH != "gfx950":
        raise NotImplementedError(
            f"{case}: OCP fp8e4m3fn dequant parity is gfx950-only; gfx942 "
            "cvt_f32_fp8 decodes bytes as legacy e4m3fnuz (bias 8, 0x80=NaN), "
            "which does not match the torch float8_e4m3fn (OCP) reference"
        )


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    passed: bool
    max_abs_diff: float
    rel_max: float
    range_min: float
    range_max: float
    note: str = ""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _summarise(O, O_ref, *, tol: float, note: str = "") -> Result:
    diff = (O.float() - O_ref.float()).abs()
    max_d = float(diff.max().item())
    ref_max = float(O_ref.abs().max().item())
    rel = max_d / (ref_max + 1e-9)
    O_min = float(O.min().item())
    O_max = float(O.max().item())
    # Sanity: O must be non-trivial when ref is non-trivial.
    if max(abs(O_min), abs(O_max)) < 0.001 and ref_max > 0.01:
        return Result(
            name="",
            passed=False,
            max_abs_diff=max_d,
            rel_max=rel,
            range_min=O_min,
            range_max=O_max,
            note=f"output is trivially zero (ref range ~{ref_max:.3f})",
        )
    return Result(
        name="",
        passed=(max_d <= tol),
        max_abs_diff=max_d,
        rel_max=rel,
        range_min=O_min,
        range_max=O_max,
        note=note,
    )


def _launch(launcher, args, *, grid, block=(64, 1, 1)):
    """Launch with wave64 block-size by default (the FMHA / sage / sparse
    kernels distribute the head-dim axis across the wave64). Per-kernel
    overrides pass ``block=(spec.block_size, 1, 1)`` explicitly when the
    kernel uses thread-id distribution at a different granularity (e.g.
    appendkv / moe_gather scatter kernels)."""
    launcher(args, config=LaunchConfig(grid=grid, block=block))
    torch.cuda.synchronize()
