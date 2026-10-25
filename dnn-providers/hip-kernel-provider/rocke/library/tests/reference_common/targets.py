# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Physical processor identity for independently qualified reference cohorts."""

from __future__ import annotations


def processor_from_target_id(target_id: str | None) -> str | None:
    """Remove profiles/features without replacing the processor by a catalog row."""
    from rocke.core.arch import compiler_target_from_target_id

    return (
        compiler_target_from_target_id(target_id).split(":", 1)[0]
        if target_id is not None
        else None
    )


def get_device_processor() -> str | None:
    """Query the hardware independently of the kernel's generic compile target."""
    from rocke.runtime.hip_module import get_device_target_id

    return processor_from_target_id(get_device_target_id())
