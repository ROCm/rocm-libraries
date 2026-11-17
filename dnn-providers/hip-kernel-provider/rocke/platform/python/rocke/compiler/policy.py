# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Canonical compiler policy resolved from measured COMGR behavior.

The capability report is evidence: several historical forms may compile. This
module is the one policy boundary that turns those observations into independent
lowering choices. Python and C++ lowering must consume the serialized policy,
not interpret raw observations separately.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .model import (
    CAPABILITY_BUFFER_RESOURCE,
    CAPABILITY_FP8_PACKING,
    CAPABILITY_LAYOUT,
    CAPABILITY_NAMED_BARRIER,
    CAPABILITY_PROCESSOR_NAMES,
    CAPABILITY_SCALED_MFMA,
    EvidenceReport,
)

__all__ = ["CompilerPolicy", "PolicyResolutionError", "resolve_policy"]

_SCHEMA = "rocke.compiler_policy/v1"
_LAYOUT_PREFERENCE = ("L3", "L2", "L1", "L0")
_BUFFER_RESOURCE_PREFERENCE = (
    "p8_p1_i64",
    "p8_p1_i32",
    "p1_i64",
    "p1_i32",
)
_FP8_PACKING_PREFERENCE = ("i64", "v2i32")
_NAMED_BARRIER_PREFERENCE = ("as15", "as3")
_SCALED_MFMA_PREFERENCE = ("current_9", "legacy_11")


class PolicyResolutionError(ValueError):
    """COMGR evidence cannot produce a complete, deterministic policy."""


def _canonical_json(document: dict[str, object]) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def _sha256(document: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(document).encode()).hexdigest()


@dataclass(frozen=True)
class CompilerPolicy:
    """Independent compiler decisions consumed by kernel lowering.

    Optional intrinsic fields are ``None`` when the selected target/compiler
    cannot lower that family. A lowering that needs one must reject the kernel;
    it must not infer a fallback from an LLVM version.
    """

    target_triple: str
    requested_processor: str
    effective_processor: str
    data_layout: str
    buffer_resource_abi: str
    fp8_packed_operand_type: str | None
    named_barrier_address_space: int | None
    scaled_mfma_call_abi: str | None
    evidence_fingerprint: str

    def __post_init__(self) -> None:
        if self.target_triple != "amdgcn-amd-amdhsa":
            raise ValueError(
                "compiler policy target_triple="
                f"{self.target_triple!r}, expected 'amdgcn-amd-amdhsa'"
            )
        for field, value in (
            ("requested_processor", self.requested_processor),
            ("effective_processor", self.effective_processor),
        ):
            if not isinstance(value, str):
                raise TypeError(f"compiler policy {field}={value!r}, expected a string")
            if re.fullmatch(r"gfx[0-9a-z-]+", value) is None:
                raise ValueError(
                    f"compiler policy {field}={value!r}, expected a gfx identifier"
                )
        _validate_choice("data_layout", self.data_layout, _LAYOUT_PREFERENCE)
        _validate_choice(
            "buffer_resource_abi",
            self.buffer_resource_abi,
            _BUFFER_RESOURCE_PREFERENCE,
        )
        _validate_optional_choice(
            "fp8_packed_operand_type",
            self.fp8_packed_operand_type,
            _FP8_PACKING_PREFERENCE,
        )
        if self.named_barrier_address_space not in {None, 3, 15}:
            raise ValueError(
                "compiler policy named_barrier_address_space="
                f"{self.named_barrier_address_space!r}, expected 3, 15, or null"
            )
        _validate_optional_choice(
            "scaled_mfma_call_abi",
            self.scaled_mfma_call_abi,
            _SCALED_MFMA_PREFERENCE,
        )
        _validate_digest("evidence_fingerprint", self.evidence_fingerprint)

    def _document(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "target_triple": self.target_triple,
            "requested_processor": self.requested_processor,
            "effective_processor": self.effective_processor,
            "data_layout": self.data_layout,
            "buffer_resource_abi": self.buffer_resource_abi,
            "fp8_packed_operand_type": self.fp8_packed_operand_type,
            "named_barrier_address_space": self.named_barrier_address_space,
            "scaled_mfma_call_abi": self.scaled_mfma_call_abi,
            "evidence_fingerprint": self.evidence_fingerprint,
        }

    @property
    def policy_sha256(self) -> str:
        """Digest used to bind caches and both lowerers to this policy."""
        return _sha256(self._document())

    def to_json(self) -> str:
        """Return the canonical policy manifest, including its digest."""
        document = {**self._document(), "policy_sha256": self.policy_sha256}
        return json.dumps(document, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> CompilerPolicy:
        """Validate and load a canonical policy manifest."""
        try:
            document = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"compiler policy is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise TypeError(
                f"compiler policy is {type(document).__name__}, expected an object"
            )
        expected = {
            "schema",
            "target_triple",
            "requested_processor",
            "effective_processor",
            "data_layout",
            "buffer_resource_abi",
            "fp8_packed_operand_type",
            "named_barrier_address_space",
            "scaled_mfma_call_abi",
            "evidence_fingerprint",
            "policy_sha256",
        }
        if set(document) != expected:
            raise ValueError(
                "compiler policy fields differ - "
                f"found={sorted(document)}, expected={sorted(expected)}"
            )
        if document["schema"] != _SCHEMA:
            raise ValueError(
                "compiler policy schema differs - "
                f"schema={document['schema']!r}, expected={_SCHEMA!r}"
            )
        digest = document.pop("policy_sha256")
        if not isinstance(digest, str) or digest != _sha256(document):
            raise ValueError("compiler policy digest does not match its contents")
        policy = cls(
            target_triple=_required_string(document, "target_triple"),
            requested_processor=_required_string(document, "requested_processor"),
            effective_processor=_required_string(document, "effective_processor"),
            data_layout=_choice(document, "data_layout", _LAYOUT_PREFERENCE),
            buffer_resource_abi=_choice(
                document, "buffer_resource_abi", _BUFFER_RESOURCE_PREFERENCE
            ),
            fp8_packed_operand_type=_optional_choice(
                document, "fp8_packed_operand_type", _FP8_PACKING_PREFERENCE
            ),
            named_barrier_address_space=_optional_address_space(document),
            scaled_mfma_call_abi=_optional_choice(
                document, "scaled_mfma_call_abi", _SCALED_MFMA_PREFERENCE
            ),
            evidence_fingerprint=_digest(document, "evidence_fingerprint"),
        )
        if policy.policy_sha256 != digest:
            raise ValueError("compiler policy normalization changed its digest")
        return policy


def _validate_choice(field: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ValueError(
            f"compiler policy {field}={value!r}, expected one of {choices}"
        )


def _validate_optional_choice(
    field: str, value: str | None, choices: tuple[str, ...]
) -> None:
    if value is not None:
        _validate_choice(field, value, choices)


def _validate_digest(field: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"compiler policy {field}={value!r}, expected a string")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(
            f"compiler policy {field}={value!r}, expected a SHA-256 digest"
        )


def _required_string(document: dict[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise TypeError(
            f"compiler policy {field}={value!r}, expected a non-empty string"
        )
    return value


def _digest(document: dict[str, object], field: str) -> str:
    value = _required_string(document, field)
    _validate_digest(field, value)
    return value


def _choice(document: dict[str, object], field: str, choices: tuple[str, ...]) -> str:
    value = document.get(field)
    if not isinstance(value, str):
        raise TypeError(f"compiler policy {field}={value!r}, expected a string")
    _validate_choice(field, value, choices)
    return value


def _optional_choice(
    document: dict[str, object], field: str, choices: tuple[str, ...]
) -> str | None:
    value = document.get(field)
    if value is None:
        return None
    return _choice(document, field, choices)


def _optional_address_space(document: dict[str, object]) -> int | None:
    value = document.get("named_barrier_address_space")
    if value is None:
        return None
    if value not in {3, 15}:
        raise ValueError(
            "compiler policy named_barrier_address_space="
            f"{value!r}, expected 3, 15, or null"
        )
    return int(value)


def _observations(report: EvidenceReport, capability: str) -> list[dict[str, object]]:
    try:
        observations = report["capabilities"][capability]["value"]["observations"]
    except (KeyError, TypeError) as exc:
        raise PolicyResolutionError(
            f"compiler capability report omitted {capability}"
        ) from exc
    if not isinstance(observations, list) or not all(
        isinstance(entry, dict) for entry in observations
    ):
        raise PolicyResolutionError(f"{capability} observations are malformed")
    return observations


def _select(
    report: EvidenceReport,
    capability: str,
    preference: tuple[str, ...],
    *,
    required: bool,
) -> str | None:
    outcomes = {
        entry.get("candidate"): entry.get("outcome")
        for entry in _observations(report, capability)
    }
    selected = None
    for candidate in preference:
        outcome = outcomes.get(candidate)
        if outcome == "accepted":
            selected = candidate
            break
        if outcome in {"timeout", "error"}:
            raise PolicyResolutionError(
                f"{capability} candidate {candidate} has unknown outcome {outcome}"
            )
    if selected is None and required:
        raise PolicyResolutionError(
            f"{capability} accepted none of the policy candidates {preference}"
        )
    return selected


def resolve_policy(report: EvidenceReport) -> CompilerPolicy:
    """Resolve measured evidence into one canonical lowering policy."""
    try:
        probe = report["probe"]
        target_triple = probe["target_triple"]
        requested = probe["processor"]
        effective = probe["effective_processor"]
        fingerprint = probe["fingerprint"]
    except (KeyError, TypeError) as exc:
        raise PolicyResolutionError(
            "compiler capability report envelope is malformed"
        ) from exc
    processor_observations = _observations(report, CAPABILITY_PROCESSOR_NAMES)
    processor_outcome = next(
        (
            entry.get("outcome")
            for entry in processor_observations
            if entry.get("family") == effective
        ),
        None,
    )
    if processor_outcome in {"timeout", "error"}:
        raise PolicyResolutionError(
            f"COMGR processor={effective!r} has unknown outcome {processor_outcome}"
        )
    if processor_outcome != "accepted":
        raise PolicyResolutionError(
            f"COMGR accepted neither processor={requested!r} nor effective={effective!r}"
        )
    named_barrier = _select(
        report,
        CAPABILITY_NAMED_BARRIER,
        _NAMED_BARRIER_PREFERENCE,
        required=False,
    )
    return CompilerPolicy(
        target_triple=str(target_triple),
        requested_processor=str(requested),
        effective_processor=str(effective),
        data_layout=str(
            _select(report, CAPABILITY_LAYOUT, _LAYOUT_PREFERENCE, required=True)
        ),
        buffer_resource_abi=str(
            _select(
                report,
                CAPABILITY_BUFFER_RESOURCE,
                _BUFFER_RESOURCE_PREFERENCE,
                required=True,
            )
        ),
        fp8_packed_operand_type=_select(
            report,
            CAPABILITY_FP8_PACKING,
            _FP8_PACKING_PREFERENCE,
            required=False,
        ),
        named_barrier_address_space=(
            int(named_barrier.removeprefix("as")) if named_barrier else None
        ),
        scaled_mfma_call_abi=_select(
            report,
            CAPABILITY_SCALED_MFMA,
            _SCALED_MFMA_PREFERENCE,
            required=False,
        ),
        evidence_fingerprint=str(fingerprint),
    )
