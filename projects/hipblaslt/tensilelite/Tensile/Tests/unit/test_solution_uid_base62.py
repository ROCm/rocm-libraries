# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for the canonical SolutionUID base62 representation."""

from __future__ import annotations

import pytest

from Tensile.Common.SolutionIdGen import (
    UINT64_MAX,
    decode_solution_uid,
    encode_solution_uid,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("solution_uid", "encoded_uid"),
    [
        (0, "0u0"),
        (1, "0u1"),
        (9, "0u9"),
        (10, "0uA"),
        (35, "0uZ"),
        (36, "0ua"),
        (61, "0uz"),
        (62, "0u10"),
        (UINT64_MAX, "0uLygHa16AHYF"),
    ],
)
def test_solution_uid_base62_round_trip(
    solution_uid: int,
    encoded_uid: str,
) -> None:
    """Encode and decode representative unsigned 64-bit values.

    Args:
        solution_uid: Numeric UID under test.
        encoded_uid: Expected canonical representation.

    Returns:
        None.

    Raises:
        AssertionError: If either codec direction produces an unexpected value.
    """
    assert encode_solution_uid(solution_uid) == encoded_uid
    assert decode_solution_uid(encoded_uid) == solution_uid


def test_decode_solution_uid_accepts_uppercase_prefix() -> None:
    """Accept ``0U`` as a read-only alias for the canonical ``0u`` prefix.

    Returns:
        None.

    Raises:
        AssertionError: If the uppercase prefix is rejected or changes the value.
    """
    assert decode_solution_uid("0UABC") == decode_solution_uid("0uABC")
    assert encode_solution_uid(decode_solution_uid("0UABC")) == "0uABC"


@pytest.mark.parametrize("encoded_uid", ["", "0", "0u", "u1", "0x1", "1"])
def test_decode_solution_uid_rejects_missing_or_invalid_prefix(
    encoded_uid: str,
) -> None:
    """Reject representations without a complete supported prefix.

    Args:
        encoded_uid: Invalid representation under test.

    Returns:
        None.

    Raises:
        AssertionError: If an invalid representation is accepted.
    """
    with pytest.raises(ValueError, match="0u or 0U"):
        decode_solution_uid(encoded_uid)


@pytest.mark.parametrize("encoded_uid", ["0u00", "0u01", "0u0ABC"])
def test_decode_solution_uid_rejects_leading_zeros(encoded_uid: str) -> None:
    """Reject non-canonical payloads with leading zero digits.

    Args:
        encoded_uid: Non-canonical representation under test.

    Returns:
        None.

    Raises:
        AssertionError: If a leading-zero representation is accepted.
    """
    with pytest.raises(ValueError, match="leading zeros"):
        decode_solution_uid(encoded_uid)


@pytest.mark.parametrize("encoded_uid", ["0u+", "0u_", "0u/", "0uA-B"])
def test_decode_solution_uid_rejects_invalid_characters(encoded_uid: str) -> None:
    """Reject characters outside the frozen base62 alphabet.

    Args:
        encoded_uid: Invalid representation under test.

    Returns:
        None.

    Raises:
        AssertionError: If a non-base62 character is accepted.
    """
    with pytest.raises(ValueError, match="base62 character"):
        decode_solution_uid(encoded_uid)


def test_solution_uid_codec_rejects_uint64_overflow() -> None:
    """Reject integer and text values larger than ``uint64``.

    Returns:
        None.

    Raises:
        AssertionError: If either codec accepts an overflowing value.
    """
    with pytest.raises(ValueError, match="out of range"):
        encode_solution_uid(UINT64_MAX + 1)
    with pytest.raises(ValueError, match="out of range"):
        decode_solution_uid("0uLygHa16AHYG")


def test_solution_uid_codec_rejects_negative_and_non_string_inputs() -> None:
    """Reject negative integers and non-string encoded representations.

    Returns:
        None.

    Raises:
        AssertionError: If either invalid input is accepted.
    """
    with pytest.raises(ValueError, match="out of range"):
        encode_solution_uid(-1)
    with pytest.raises(TypeError, match="must be a string"):
        decode_solution_uid(1)  # type: ignore[arg-type]
