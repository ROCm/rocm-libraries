# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The record a parity emitter prints for the HIP source identity test.

``test_hip_source_identity.py`` holds the Python and C++ HIP lowerings to the
same output for every kernel the parity emitters build. The emitter's
``hip-case`` mode prints one case per kernel: its target, its ck.dsl.ir/v1
serialization -- the C++ engine's input -- and the Python engine's outcome for
the kernel the builder made. The Python outcome is taken in the emitter's
process from the built kernel; the C++ engine reads the serialization, so a
difference between the outcomes comes from the two lowerings or from what the
serialization does not carry. The serialization must also read back through
the Python parser to the same text (``require_serialization_round_trip``), so
the Python parser reads every kernel the C++ engine is given.

One case, with the byte counts of the UTF-8 text that follows the header line::

    hip-case <arch> <serialization bytes> source <source bytes>
    <serialization><source>

or, for a refusal::

    hip-case <arch> <serialization bytes> refusal <error type> <message bytes>
    <serialization><message>

A config that builds several kernels prints one case per kernel, back to back.
"""

from __future__ import annotations

import dataclasses

from ..core.ir import KernelDef
from ..core.ir_serialize import parse, serialize
from .hip_outcomes import HipRefusal, python_hip_outcome_of_kernel

__all__ = [
    "HipIdentityCase",
    "SerializationRoundTripError",
    "hip_case_text",
    "parse_hip_cases",
    "require_serialization_round_trip",
]

_CASE_TAG = "hip-case"


@dataclasses.dataclass(frozen=True)
class HipIdentityCase:
    """One kernel to lower with the C++ engine, and the Python engine's outcome."""

    arch: str
    serialized_kernel: str
    python_outcome: str | HipRefusal


class SerializationRoundTripError(RuntimeError):
    """The Python parser does not read a kernel's serialization back to the
    same text. Not a refusal type, so the identity test reports it as a failure
    of the config that built the kernel."""


def require_serialization_round_trip(kernel_name: str, serialization: str) -> None:
    """Raise :class:`SerializationRoundTripError` unless the Python parser reads
    ``serialization`` and serializes it back to the same text."""
    try:
        read_back = serialize(parse(serialization))
    except Exception as error:  # any parser failure is the finding
        raise SerializationRoundTripError(
            f"require_serialization_round_trip failed - kernel={kernel_name!r}, "
            f"the Python parser raised {type(error).__name__}: {error}, expected "
            "it to read the serialization"
        ) from error
    if read_back != serialization:
        line_number, written, read = next(
            (number, written, read)
            for number, (written, read) in enumerate(
                zip(
                    serialization.splitlines() + [""],
                    read_back.splitlines() + [""],
                ),
                start=1,
            )
            if written != read
        )
        raise SerializationRoundTripError(
            f"require_serialization_round_trip failed - kernel={kernel_name!r}, "
            f"line {line_number} reads back as {read!r}, expected {written!r}"
        )


def hip_case_text(kernel: KernelDef, arch: str) -> str:
    """The ``hip-case`` record of ``kernel`` lowered for ``arch``. Its
    serialization must read back through the Python parser to the same text
    (:func:`require_serialization_round_trip`)."""
    _require_one_word("arch", arch)
    serialization = serialize(kernel)
    require_serialization_round_trip(kernel.name, serialization)
    outcome = python_hip_outcome_of_kernel(kernel, arch)
    if isinstance(outcome, HipRefusal):
        _require_one_word("error_type_name", outcome.error_type_name)
        kind, body = f"refusal {outcome.error_type_name}", outcome.message
    else:
        kind, body = "source", outcome
    header = (
        f"{_CASE_TAG} {arch} {_utf8_size(serialization)} {kind} {_utf8_size(body)}\n"
    )
    return header + serialization + body


def _require_one_word(name: str, value: str) -> None:
    """A header field must be one word, or the parser would split it."""
    if value.split() != [value]:
        raise ValueError(
            f"hip_case_text failed - {name}={value!r}, expected one word with no "
            "whitespace"
        )


def _utf8_size(text: str) -> int:
    return len(text.encode("utf-8"))


def parse_hip_cases(output: bytes) -> list[HipIdentityCase]:
    """The cases of one emitter config's ``hip-case`` ``output``, in order."""
    cases = []
    position = 0
    while position < len(output):
        header_end = output.find(b"\n", position)
        if header_end < 0:
            header_end = len(output)
        header = output[position:header_end].decode("utf-8", "replace")
        fields = header.split()
        refusal = len(fields) == 6 and fields[3] == "refusal"
        source = len(fields) == 5 and fields[3] == "source"
        counts = fields[2:3] + fields[-1:]
        if (
            fields[:1] != [_CASE_TAG]
            or not (refusal or source)
            or not all(count.isascii() and count.isdigit() for count in counts)
        ):
            raise ValueError(
                f"parse_hip_cases failed - header={header[:80]!r} at byte "
                f"{position}, expected '{_CASE_TAG} <arch> <bytes> source <bytes>' "
                f"or '{_CASE_TAG} <arch> <bytes> refusal <error type> <bytes>'"
            )
        serialization_end = header_end + 1 + int(fields[2])
        case_end = serialization_end + int(fields[-1])
        if case_end > len(output):
            raise ValueError(
                f"parse_hip_cases failed - header={header[:80]!r} at byte "
                f"{position} counts {case_end - header_end - 1} bytes, expected at "
                f"most the {len(output) - header_end - 1} that follow it"
            )
        try:
            serialized_kernel = output[header_end + 1 : serialization_end].decode()
            body = output[serialization_end:case_end].decode()
        except UnicodeDecodeError as error:
            raise ValueError(
                f"parse_hip_cases failed - header={header[:80]!r} at byte "
                f"{position}, expected UTF-8 text after it: {error}"
            ) from error
        cases.append(
            HipIdentityCase(
                arch=fields[1],
                serialized_kernel=serialized_kernel,
                python_outcome=HipRefusal(fields[4], body) if refusal else body,
            )
        )
        position = case_end
    return cases
