# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Generate stable 64-bit solution IDs for hipBLASLt."""

from __future__ import annotations

import secrets
import time

"""
Solution ID is a 64-bit integer that is used to uniquely identify a solution.
It is a combination of the current time in milliseconds since the SolutionUID
epoch and an OS-backed random 24-bit integer.

| 40 bits | 24 bits |
|---------|---------|
|  Time   | Random  |
"""

EPOCH_MS = 1_767_225_600_000  # 2026-01-01 00:00:00 UTC
MS_MASK = (1 << 40) - 1
RANDOM_BITS = 24
RANDOM_MASK = (1 << RANDOM_BITS) - 1
RANDOM_SHIFT = RANDOM_BITS
BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
BASE62_LOOKUP = {character: index for index, character in enumerate(BASE62_ALPHABET)}
BASE62_PREFIX = "0u"
UINT64_MAX = (1 << 64) - 1


def encode_solution_uid(solution_uid: int) -> str:
    """Encode a 64-bit solution UID in canonical ``0u`` base62 form.

    Args:
        solution_uid: Unsigned 64-bit solution identifier.

    Returns:
        Canonical unpadded base62 representation prefixed with ``0u``.

    Raises:
        ValueError: If ``solution_uid`` is outside the unsigned 64-bit range.
    """
    if solution_uid < 0 or solution_uid > UINT64_MAX:
        raise ValueError(
            f"SolutionUID ({solution_uid}) is out of range for uint64 "
            f"(0..{UINT64_MAX})"
        )

    if solution_uid == 0:
        return f"{BASE62_PREFIX}0"

    digits = []
    value = solution_uid
    while value:
        value, remainder = divmod(value, len(BASE62_ALPHABET))
        digits.append(BASE62_ALPHABET[remainder])
    return BASE62_PREFIX + "".join(reversed(digits))


def decode_solution_uid(encoded_uid: str) -> int:
    """Decode a canonical ``0u`` or ``0U`` base62 solution UID.

    Args:
        encoded_uid: Canonical prefixed, unpadded base62 representation.

    Returns:
        Unsigned 64-bit solution identifier.

    Raises:
        TypeError: If ``encoded_uid`` is not a string.
        ValueError: If the representation is malformed, non-canonical, or
            outside the unsigned 64-bit range.
    """
    if not isinstance(encoded_uid, str):
        raise TypeError("SolutionUID representation must be a string")
    if len(encoded_uid) < 3 or encoded_uid[:2] not in {"0u", "0U"}:
        raise ValueError("SolutionUID must start with 0u or 0U")

    payload = encoded_uid[2:]
    if len(payload) > 1 and payload[0] == "0":
        raise ValueError("SolutionUID base62 payload must not contain leading zeros")

    value = 0
    for character in payload:
        digit = BASE62_LOOKUP.get(character)
        if digit is None:
            raise ValueError(f"Invalid SolutionUID base62 character: {character!r}")
        if value > (UINT64_MAX - digit) // len(BASE62_ALPHABET):
            raise ValueError("SolutionUID is out of range for uint64")
        value = value * len(BASE62_ALPHABET) + digit
    return value


def _ms_since_epoch(now_ms: int) -> int:
    """Convert Unix milliseconds to ms since :data:`EPOCH_MS`.

    Args:
        now_ms: Unix timestamp in milliseconds.

    Returns:
        Milliseconds since the configured epoch.

    Raises:
        ValueError: If the value does not fit in 40 bits.
    """
    ms = now_ms - EPOCH_MS
    if ms < 0 or ms > MS_MASK:
        raise ValueError(
            f"ms since epoch ({ms}) is out of range for 40 bits (0..{MS_MASK})"
        )
    return ms


def generate_solution_id(*, now_ms: int | None = None) -> int:
    """Generate a 64-bit solution ID.

    Layout: ``(ms_since_epoch << 24) | random24`` where ``ms_since_epoch`` uses
    40 bits and ``random24`` uses ``secrets.randbits(24)``.

    Args:
        now_ms: Optional fixed Unix timestamp in milliseconds for testing.
            When omitted, ``time.time()`` is used.

    Returns:
        A 64-bit unsigned solution identifier.

    Raises:
        ValueError: If the ms-since-epoch value does not fit in 40 bits.
    """
    if now_ms is None:
        now_ms = int(time.time() * 1000)

    ms = _ms_since_epoch(now_ms)
    random24 = secrets.randbits(RANDOM_BITS)
    return (ms << RANDOM_SHIFT) | random24


def decode_solution_id(solution_id: int) -> tuple[int, int]:
    """Split a solution ID into its timestamp and random components.

    Args:
        solution_id: 64-bit solution identifier from :func:`generate_solution_id`.

    Returns:
        Tuple of ``(ms_since_epoch, random24)``.
    """
    return solution_id >> RANDOM_SHIFT, solution_id & RANDOM_MASK


def read_solution_uid(solution: dict) -> int:
    """Read a solution UID from YAML state, generating one if absent.

    Args:
        solution: Solution dictionary from logic YAML.

    Returns:
        64-bit solution UID.
    """
    if "SolutionUID" in solution:
        return decode_solution_uid(solution["SolutionUID"])
    return generate_solution_id()


def ensure_solution_uid(solution: dict) -> None:
    """Assign ``SolutionUID`` when the solution has no UID field yet.

    Args:
        solution: Solution dictionary to update in place.

    Raises:
        ValueError: Propagated from :func:`generate_solution_id` on time overflow.
    """
    if "SolutionUID" not in solution:
        solution["SolutionUID"] = encode_solution_uid(generate_solution_id())


def regenerate_solution_uid(solution: dict) -> int:
    """Replace ``SolutionUID`` with a newly generated value.

    Args:
        solution: Solution dictionary to update in place.

    Returns:
        The newly assigned 64-bit solution UID.

    Raises:
        ValueError: Propagated from :func:`generate_solution_id` on time overflow.
    """
    uid = generate_solution_id()
    solution["SolutionUID"] = encode_solution_uid(uid)
    return uid
