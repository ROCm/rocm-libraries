# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher data contracts shared by operator families.

This module is intentionally operator-agnostic. Op-specific request types,
algorithm names, ABI versions, and candidate factories belong in their family
modules (for example, :mod:`rocke.dispatch.gemm`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence, Tuple


def stable_json_hash(payload: Mapping[str, Any], *, n: int = 16) -> str:
    """Stable short SHA256 over JSON-serializable dispatcher payloads."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:n]


@dataclass(frozen=True)
class OperatorRequest:
    """Base marker for normalized framework requests.

    Concrete operator families should subclass this and return a stable,
    JSON-serializable dictionary from :meth:`normalized`. That normalized
    payload is what feeds request hashes and benchmark/cache identity.
    """

    def normalized(self) -> dict:
        return {}


@dataclass(frozen=True)
class KernelId:
    """Stable identity shared by caches, manifests, benchmarks, and frameworks."""

    op: str
    family: str
    candidate: str
    algorithm: str
    spec_id: str
    arch: str
    abi_version: str
    request_hash: str
    spec_hash: str

    @property
    def cache_key(self) -> str:
        return (
            f"{self.op}:{self.family}:{self.candidate}:{self.arch}:"
            f"{self.algorithm}:{self.spec_id}:{self.abi_version}:"
            f"{self.request_hash}:{self.spec_hash}"
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class KernelCandidate:
    """One selectable implementation family for an operator request."""

    name: str
    family: str
    algorithm: str
    spec_id: str
    abi_version: str
    priority: int
    supports: Callable[[OperatorRequest], Tuple[bool, str]]
    select_spec: Callable[[OperatorRequest], Any]
    signature: Callable[[Any], Sequence[dict]]
    grid: Callable[[Any, OperatorRequest], Tuple[int, int, int]]
    block: Callable[[Any], Tuple[int, int, int]]
    sweep_space: Callable[[OperatorRequest], Sequence[Any]]


Ranker = Callable[
    [OperatorRequest, Sequence[KernelCandidate]], Sequence[KernelCandidate]
]


class CandidateRegistry:
    """Simple in-process candidate registry.

    Mirrors the CK dispatcher shape at Python scale: candidates are registered
    once, then filtered by support predicates and selected by explicit
    ``algorithm`` / ``spec_id`` request fields or by priority for ``auto``.
    """

    def __init__(self, family: str):
        self.family = family
        self._candidates = {}

    def register(self, candidate: KernelCandidate) -> None:
        if candidate.name in self._candidates:
            raise ValueError(f"duplicate candidate {candidate.name!r}")
        if candidate.family != self.family:
            raise ValueError(
                f"candidate family {candidate.family!r} != registry {self.family!r}"
            )
        self._candidates[candidate.name] = candidate

    def candidates(self) -> Tuple[KernelCandidate, ...]:
        return tuple(
            sorted(self._candidates.values(), key=lambda c: (c.priority, c.name))
        )

    def supported(self, request: OperatorRequest) -> Tuple[KernelCandidate, ...]:
        return tuple(c for c in self.candidates() if c.supports(request)[0])

    def select(
        self, request: OperatorRequest, *, ranker: Ranker | None = None
    ) -> KernelCandidate:
        supported = self.supported(request)
        if supported:
            ranked = (
                tuple(ranker(request, supported)) if ranker is not None else supported
            )
            if not ranked:
                raise ValueError("ranker returned no candidates")
            ranked_names = {c.name for c in supported}
            for candidate in ranked:
                if candidate.name not in ranked_names:
                    raise ValueError(
                        f"ranker returned unsupported candidate {candidate.name!r}"
                    )
            return ranked[0]
        reasons = []
        for candidate in self.candidates():
            ok, why = candidate.supports(request)
            if not ok:
                reasons.append(f"{candidate.name}: {why}")
        joined = "; ".join(reasons) if reasons else "no candidates registered"
        raise ValueError(f"no candidate supports request: {joined}")

    def extend(self, candidates: Iterable[KernelCandidate]) -> None:
        for candidate in candidates:
            self.register(candidate)


@dataclass(frozen=True)
class DispatchResult:
    """Dispatcher answer for one request."""

    request: OperatorRequest
    candidate: KernelCandidate
    spec: Any
    kernel_id: KernelId
    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    signature: Tuple[dict, ...]
    explanation: Tuple[str, ...]
