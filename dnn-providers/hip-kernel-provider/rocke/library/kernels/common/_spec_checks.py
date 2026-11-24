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


# Largest paged KV cache, in bytes, that i32 addressing reads correctly on the
# tiled and dense builders. They bound K/V buffer loads at num_records=0x7FFF0000,
# so a load at or past that offset returns zero.
I32_KV_BUFFER_BYTES = 0x7FFF_0000


def kv_cache_needs_i64_addr(num_kv_blocks: int, block_bytes: int) -> bool:
    """True when ``num_kv_blocks`` blocks of ``block_bytes`` each need i64 addressing.

    The single threshold for the spec builders, the dispatcher's tuning specs
    and the launch guard, so they cannot disagree. ``num_kv_blocks <= 0`` means
    the cache size is unknown, so assume small.
    """
    return num_kv_blocks > 0 and num_kv_blocks * block_bytes > I32_KV_BUFFER_BYTES
