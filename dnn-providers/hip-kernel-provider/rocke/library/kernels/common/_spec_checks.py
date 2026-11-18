# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side checks shared by the attention spec and problem dataclasses."""


def require_bool(name: str, value: object) -> None:
    """Refuse a flag that is not exactly ``bool``.

    ``bool("false")`` is ``True``, and ``1`` or ``"yes"`` would silently change a
    kernel's identity, so a flag is checked, never coerced. ``type(value)`` rather
    than ``isinstance`` keeps the check exact.
    """
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool, got {type(value).__name__}")
