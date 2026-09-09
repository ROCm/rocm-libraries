# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Device feature rules applied to a HIP properties snapshot.

These rules describe hardware support. Compiler support and kernel eligibility
are checked separately by compilation and dispatch code.
"""

from __future__ import annotations

from dataclasses import dataclass

from .device_info import DeviceInfo


@dataclass(frozen=True)
class DeviceCapabilities:
    """Known support for individual device features.

    Each field is ``True`` for supported, ``False`` for unsupported, or ``None``
    when no rule covers the device. Callers requiring a feature should check
    ``is True`` so unknown support does not enable it.

    ``has_tdm_multicast`` describes TDM multicast, independently of ordinary
    TDM loads, workgroup clusters, and cluster barriers.
    ``has_fp4_wmma_32x16`` describes FP4 WMMA with physical instruction dimensions
    32 by 16, independently of logical dimensions after operand swapping.
    """

    has_tdm_multicast: bool | None = None
    has_fp4_wmma_32x16: bool | None = None


# Matches hipBLASLt's gfx1250 revision rules. Source links and policy scope are
# documented in dsl_docs/runtime/comgr_and_hipmodule.md.
_CAPABILITIES = {
    ("gfx1250", 0): DeviceCapabilities(
        has_tdm_multicast=False, has_fp4_wmma_32x16=False
    ),
    ("gfx1250", 1): DeviceCapabilities(has_tdm_multicast=True, has_fp4_wmma_32x16=True),
}


def infer_device_capabilities(info: DeviceInfo) -> DeviceCapabilities:
    """Apply architecture and ASIC revision rules to an existing snapshot.

    Makes no HIP calls and does not modify ``info``. Missing identity values,
    unlisted architectures, and unlisted revisions produce unknown support.
    Target suffixes are handled by ``DeviceInfo.base_arch``; the exact target ID
    and compiler target remain available on ``info``.
    """
    return _CAPABILITIES.get((info.base_arch, info.asic_revision), DeviceCapabilities())


__all__ = ["DeviceCapabilities", "infer_device_capabilities"]
