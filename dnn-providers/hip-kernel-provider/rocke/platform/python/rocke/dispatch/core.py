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

from ..core.arch import known_arches


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

    def dims(self) -> Mapping[str, int]:
        """Every gateable integer quantity, derived ones included.

        This is the vocabulary :class:`ShapeRange` and :class:`DimRelation`
        constrain. Families are free to expose quantities that are computed
        rather than stored -- attention's ``total_q``, conv's ``Ho``/``Wo`` --
        because those are what kernels actually branch on. Returning ``{}``
        means the family has not adopted capability gating yet.
        """
        return {}

    def features(self) -> frozenset[str]:
        """Optional behaviors this request needs, as a set of names.

        A candidate declares the features it can serve, so a feature the
        request needs but the candidate never declared is a rejection rather
        than a kernel that silently ignores it.
        """
        return frozenset()


@dataclass(frozen=True)
class ShapeRange:
    """One bound, applied to a dimension or broadcast across a set of them.

    ``dims`` is a single name or a set of names sharing the bound. Conv's
    paired dimensions -- (Hi, Wi), (Y, X), (stride_h, stride_w) -- are the
    common case for the set form.
    """

    dims: str | frozenset[str]
    min: int | None = None
    max: int | None = None
    multiple_of: int | None = None
    allowed: Tuple[int, ...] | None = None

    def names(self) -> Tuple[str, ...]:
        """Sorted: a set is unordered, and messages must be reproducible."""
        if isinstance(self.dims, str):
            return (self.dims,)
        return tuple(sorted(self.dims))

    def check(self, dims: Mapping[str, int]) -> Tuple[bool, str]:
        for name in self.names():
            if name not in dims:
                return False, f"dim {name!r} not provided (have {sorted(dims)})"
            value = int(dims[name])
            if self.allowed is not None and value not in self.allowed:
                return False, f"{name}={value} not in {self.allowed}"
            if self.min is not None and value < self.min:
                return False, f"{name}={value} < min {self.min}"
            if self.max is not None and value > self.max:
                return False, f"{name}={value} > max {self.max}"
            if self.multiple_of and value % self.multiple_of:
                return False, f"{name}={value} not a multiple of {self.multiple_of}"
        return True, "ok"

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"dims": list(self.names())}
        for field in ("min", "max", "multiple_of"):
            value = getattr(self, field)
            if value is not None:
                payload[field] = value
        if self.allowed is not None:
            payload["allowed"] = list(self.allowed)
        return payload


_DIM_RELATION_OPS = {
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "multiple_of": lambda a, b: b != 0 and a % b == 0,
}


@dataclass(frozen=True)
class DimRelation:
    """A constraint between two dimensions, or a dimension and a literal.

    Deliberately data rather than a callable: a callable could not be
    serialized into a coverage manifest, diffed across releases, or rendered
    into documentation, which is the whole point of declaring coverage.
    """

    lhs: str
    op: str
    rhs: str | int

    def __post_init__(self):
        if self.op not in _DIM_RELATION_OPS:
            raise ValueError(
                f"unknown DimRelation op {self.op!r}; "
                f"expected one of {sorted(_DIM_RELATION_OPS)}"
            )

    def check(self, dims: Mapping[str, int]) -> Tuple[bool, str]:
        for key in (self.lhs, self.rhs):
            if isinstance(key, str) and key not in dims:
                return False, f"dim {key!r} not provided (have {sorted(dims)})"
        a = int(dims[self.lhs])
        b = int(dims[self.rhs]) if isinstance(self.rhs, str) else int(self.rhs)
        if _DIM_RELATION_OPS[self.op](a, b):
            return True, "ok"
        return False, f"{self.lhs}={a} {self.op} {self.rhs}={b} violated"

    def as_dict(self) -> dict[str, Any]:
        return {"lhs": self.lhs, "op": self.op, "rhs": self.rhs}


@dataclass(frozen=True)
class Capability:
    """What a candidate was built for, as data rather than code.

    Answers coverage questions without executing a request, and serves as a
    cheap prefilter before predicates run. An empty tuple means unconstrained,
    with one exception: ``arches`` fails closed, so a capability that declares
    no architecture matches nothing. :meth:`CandidateRegistry.register` rejects
    that case up front rather than letting it surface as a silent no-match.

    Capability is a conservative *superset* of what ``_supports()`` accepts. A
    constraint it cannot express stays in the predicate; the direction that
    must never invert is capability accepting less than the predicate does.
    """

    arches: Tuple[str, ...] = ()
    dtypes: Tuple[str, ...] = ()
    layouts: Tuple[str, ...] = ()
    shapes: Tuple[ShapeRange, ...] = ()
    relations: Tuple[DimRelation, ...] = ()
    supports_features: frozenset[str] = frozenset()
    requires_features: frozenset[str] = frozenset()

    def dim_names(self) -> frozenset[str]:
        """Every dimension this capability refers to, for registration checks."""
        names = {name for rng in self.shapes for name in rng.names()}
        for relation in self.relations:
            names.add(relation.lhs)
            if isinstance(relation.rhs, str):
                names.add(relation.rhs)
        return frozenset(names)

    def check(self, request: OperatorRequest) -> Tuple[bool, str]:
        normalized = request.normalized()

        def canonical(field: str) -> str:
            value = normalized.get(field, getattr(request, field, ""))
            return str(value)

        arch = canonical("arch")
        if arch not in self.arches:
            return False, f"arch {arch!r} not in {self.arches}"
        if self.dtypes and canonical("dtype").lower() not in self.dtypes:
            return False, f"dtype {canonical('dtype')!r} not in {self.dtypes}"
        if self.layouts and canonical("layout").upper() not in self.layouts:
            return False, f"layout {canonical('layout')!r} not in {self.layouts}"

        dims = request.dims()
        for constraint in self.shapes + self.relations:
            ok, why = constraint.check(dims)
            if not ok:
                return False, why

        features = request.features()
        missing = self.requires_features - features
        if missing:
            return False, f"requires features {sorted(missing)}"
        unsupported = features - self.supports_features
        if unsupported:
            return False, f"cannot serve features {sorted(unsupported)}"
        return True, "ok"

    def as_dict(self) -> dict[str, Any]:
        return {
            "arches": list(self.arches),
            "dtypes": list(self.dtypes),
            "layouts": list(self.layouts),
            "shapes": [rng.as_dict() for rng in self.shapes],
            "relations": [rel.as_dict() for rel in self.relations],
            "supports_features": sorted(self.supports_features),
            "requires_features": sorted(self.requires_features),
        }


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
    def compile_key(self) -> str:
        """Identity of the compiled binary: arch, ABI, and spec only.

        Problem-independent by construction, so every request that selects the
        same spec shares one compile. This is the key an HSACO cache wants.
        """
        return f"{self.arch}:{self.abi_version}:{self.spec_hash}"

    @property
    def selection_key(self) -> str:
        """Identity of the routing decision, including the problem.

        Tuning records, dispatch logs, and benchmark rows index by this, since
        the request is precisely what they need to tell apart.
        """
        return (
            f"{self.op}:{self.family}:{self.candidate}:{self.arch}:"
            f"{self.algorithm}:{self.spec_id}:{self.abi_version}:"
            f"{self.request_hash}:{self.spec_hash}"
        )

    @property
    def cache_key(self) -> str:
        """Deprecated alias for :attr:`selection_key`.

        The name predates the split and reads like a compile-cache key, which
        it is not: keying a compile cache on it recompiles per shape. The value
        is unchanged from before the split so existing benchmark records stay
        comparable. New code should say which key it means.
        """
        return self.selection_key

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProblemBinding:
    """Everything a launcher needs to run one dispatched kernel once.

    Selection answers *which* kernel; a binding answers *how to call it* for a
    concrete request: the launch geometry, the packed argument buffer, a
    numeric reference, and the roofline denominators.

    The callables take the HIP ``Runtime`` as a parameter rather than closing
    over one, so this module keeps its CPU-only import cost and a binding can
    be built, inspected, and unit-tested on a machine with no GPU. That is also
    the shape the manifest runner's problem builders already use, so an adapter
    can delegate here instead of re-deriving geometry from manifest fields.
    """

    grid: Tuple[int, int, int]
    block: Tuple[int, int, int]
    make_args: Callable[[Any], Tuple[bytes, Tuple[int, ...]]]
    """``make_args(rt) -> (packed_args, device_ptrs)``; allocates and uploads."""
    check: Callable[[Any, Tuple[int, ...]], Tuple[float, int, int]]
    """``check(rt, ptrs) -> (max_abs_diff, bad_count, total)``; a no-op returns
    ``(0.0, 0, total)`` when the binding was built without verification."""
    flop: float
    bytes_moved: float

    def as_problem_builder(self) -> tuple:
        """Adapt to the manifest runner's positional problem-builder tuple."""
        return (
            self.make_args,
            self.grid,
            self.block,
            self.flop,
            self.bytes_moved,
            self.check,
        )


@dataclass(frozen=True)
class KernelCandidate:
    """One selectable implementation family for an operator request."""

    name: str
    family: str
    algorithm: str
    spec_id: str
    abi_version: str
    priority: int
    _supports: Callable[[OperatorRequest], Tuple[bool, str]]
    """The residual predicate: everything ``capability`` cannot express as data.

    Underscored because it is not a complete eligibility answer and calling it
    alone silently skips the arch and dtype gates that moved into
    ``capability``. :meth:`admits` is the public verdict; a bare ``_supports``
    call at a non-registry call site should read as the violation it is.
    """

    select_spec: Callable[[OperatorRequest], Any]
    signature: Callable[[Any], Sequence[dict]]
    grid: Callable[[Any, OperatorRequest], Tuple[int, int, int]]
    block: Callable[[Any], Tuple[int, int, int]]
    sweep_space: Callable[[OperatorRequest], Sequence[Any]]
    capability: Capability | None = None
    """Declared coverage. Required by :meth:`CandidateRegistry.register`.

    Still typed optional so a candidate can be constructed and inspected
    standalone in a test, but a registry will not accept ``None``: an
    undeclared candidate is invisible to ``for_arch`` and ``coverage``, and
    answering "what runs on gfx1250?" has to stay a lookup rather than a probe.
    """

    build: Callable[[Any, str], Any] | None = None
    """``build(spec, arch) -> KernelDef``; the candidate's real IR builder.

    Typed loosely because ``KernelDef`` lives in :mod:`rocke.core` and importing
    it here would drag the IR layer into every dispatch import. What matters is
    the contract: the spec ``select_spec`` returns is exactly what this accepts,
    so a selection can be compiled without a per-family call site.
    """

    bind: Callable[[Any, bool], ProblemBinding] | None = None
    """``bind(result, verify) -> ProblemBinding``; optional.

    Optional because a candidate is useful for selection alone, and the
    families differ in how much host-side setup they need. Where it is
    provided it is the single definition of the launch contract: the geometry
    it reports comes from this candidate's own ``grid``/``block``, so a runner
    cannot drift from the dispatcher the way a hand-written adapter can.
    """

    def built(self, spec: Any, arch: str) -> Any:
        """Build this candidate's IR for ``spec`` on ``arch``."""
        if self.build is None:
            raise NotImplementedError(
                f"candidate {self.name!r} ({self.family}) declares no build(); "
                "it can be selected but not compiled through the generic "
                "path. Point build at its real builder to close that gap."
            )
        return self.build(spec, arch)

    def bound(self, result: Any, *, verify: bool = False) -> ProblemBinding:
        """Bind ``result`` to a runnable problem, or explain what is missing."""
        if self.bind is None:
            raise NotImplementedError(
                f"candidate {self.name!r} ({self.family}) declares no bind(); "
                "it can be selected but not launched through the generic "
                "runner. Give it a bind to close that gap."
            )
        return self.bind(result, verify)

    def admits(self, request: OperatorRequest) -> Tuple[bool, str]:
        """Full eligibility verdict: capability prefilter, then predicate.

        The only eligibility question a caller should ask. Registered
        candidates keep their arch and dtype gates in ``capability``, so
        ``_supports`` carries only the residual checks and is not a complete
        answer on its own -- an RDNA-only WMMA candidate's predicate happily
        accepts a CDNA target, because rule 1 of ARCHITECTURE.md 6.2 removed
        the arch check it used to duplicate.
        """
        if self.capability is not None:
            ok, why = self.capability.check(request)
            if not ok:
                return False, f"capability: {why}"
        return self._supports(request)


Ranker = Callable[
    [OperatorRequest, Sequence[KernelCandidate]], Sequence[KernelCandidate]
]


class CandidateRegistry:
    """Simple in-process candidate registry.

    Mirrors the CK dispatcher shape at Python scale: candidates are registered
    once, then filtered by support predicates and selected by explicit
    ``algorithm`` / ``spec_id`` request fields or by priority for ``auto``.
    """

    def __init__(
        self,
        family: str,
        *,
        dim_vocabulary: Iterable[str] | None = None,
        require_build: bool = False,
        require_binding: bool = False,
    ) -> None:
        self.family = family
        self.dim_vocabulary = (
            None if dim_vocabulary is None else frozenset(dim_vocabulary)
        )
        self.require_build = require_build
        """Whether this family refuses to register a candidate it cannot build.

        Separate from ``require_binding`` because the two are reachable at
        different times: building needs only a spec and a builder, which every
        platform family already has, while binding additionally needs a
        declared args signature and launch geometry.
        """
        self.require_binding = require_binding
        """Whether this family refuses to register a candidate it cannot launch.

        A per-family ratchet rather than a global rule, because ``bind`` is
        executable behavior and not, like ``capability``, a declaration that is
        always available to make. A family turns this on once it has backfilled
        its candidates; from then on a new candidate cannot rejoin the
        unlaunchable set by omission. See ARCHITECTURE.md 5.2.
        """
        self._candidates = {}

    def register(self, candidate: KernelCandidate) -> None:
        if candidate.name in self._candidates:
            raise ValueError(f"duplicate candidate {candidate.name!r}")
        if candidate.family != self.family:
            raise ValueError(
                f"candidate family {candidate.family!r} != registry {self.family!r}"
            )
        self._validate_capability(candidate)
        self._validate_build(candidate)
        self._validate_binding(candidate)
        self._candidates[candidate.name] = candidate

    def _validate_build(self, candidate: KernelCandidate) -> None:
        if self.require_build and candidate.build is None:
            raise ValueError(
                f"{candidate.name!r} declares no build, and family "
                f"{self.family!r} requires one: a candidate this family "
                "registers must be compilable, not merely selectable. Point "
                "build at the builder whose spec type select_spec returns."
            )

    def _validate_binding(self, candidate: KernelCandidate) -> None:
        if self.require_binding and candidate.bind is None:
            raise ValueError(
                f"{candidate.name!r} declares no bind, and family "
                f"{self.family!r} requires one: every candidate it registers "
                "must be launchable, not merely selectable. Give it a bind "
                "returning a ProblemBinding (see ARCHITECTURE.md 7.5), or if "
                "this candidate genuinely cannot be launched, that is a reason "
                "not to register it here."
            )

    def _validate_capability(self, candidate: KernelCandidate) -> None:
        """Reject a capability that cannot mean what its author intended.

        These fire at import time, which is the point: a candidate with no arch
        gate or a misspelled dimension name would otherwise sit dormant until
        some request happened to reach it.
        """
        capability = candidate.capability
        if capability is None:
            raise ValueError(
                f"{candidate.name!r} declares no capability; every registered "
                "candidate must say what it covers (see ARCHITECTURE.md 5.1)"
            )
        if not capability.arches:
            raise ValueError(
                f"{candidate.name!r} declares no arch coverage; set "
                "arches=(...) (see ARCHITECTURE.md 5.1)"
            )
        unknown_arches = set(capability.arches) - set(known_arches())
        if unknown_arches:
            raise ValueError(
                f"{candidate.name!r} declares unknown arches "
                f"{sorted(unknown_arches)}"
            )
        if self.dim_vocabulary is None:
            return
        unknown_dims = capability.dim_names() - self.dim_vocabulary
        if unknown_dims:
            raise ValueError(
                f"{candidate.name!r} constrains unknown dims "
                f"{sorted(unknown_dims)}; {self.family} provides "
                f"{sorted(self.dim_vocabulary)}"
            )

    def candidates(self) -> Tuple[KernelCandidate, ...]:
        return tuple(
            sorted(self._candidates.values(), key=lambda c: (c.priority, c.name))
        )

    def get(self, name: str) -> KernelCandidate:
        """Return the candidate registered under ``name``.

        Raises ``ValueError`` naming the registered candidates, because the
        usual cause is a stale or misspelled identifier and the fix is knowing
        what was available instead.
        """
        try:
            return self._candidates[name]
        except KeyError:
            raise ValueError(
                f"unknown candidate {name!r}; registered: {sorted(self._candidates)}"
            ) from None

    def resolve(self, kernel_id: KernelId) -> KernelCandidate:
        """Return the candidate a previously issued ``kernel_id`` names.

        The ABI check is what makes a persisted tuning result safe to replay: an
        id minted by an older build fails loudly here instead of binding to a
        candidate whose kernarg layout has changed underneath it.
        """
        candidate = self.get(kernel_id.candidate)
        if candidate.abi_version != kernel_id.abi_version:
            raise ValueError(
                f"ABI mismatch for {kernel_id.candidate!r}: id has "
                f"{kernel_id.abi_version}, registry has {candidate.abi_version}"
            )
        return candidate

    def coverage(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest of what this registry holds.

        Answers "what is dispatchable?" without a request, so CI can diff the
        surface instead of reading source. Ordering follows :meth:`candidates`,
        so the manifest is stable across processes.
        """
        return {
            "family": self.family,
            "requires_build": self.require_build,
            "requires_binding": self.require_binding,
            "candidates": [
                {
                    "name": c.name,
                    "algorithm": c.algorithm,
                    "spec_id": c.spec_id,
                    "abi_version": c.abi_version,
                    "priority": c.priority,
                    # Whether this candidate can be compiled and launched, not
                    # just chosen. Queryable for the same reason coverage is:
                    # "can I run this?" should be a lookup, not a call that
                    # might raise.
                    "buildable": c.build is not None,
                    "bindable": c.bind is not None,
                    "capability": (
                        None if c.capability is None else c.capability.as_dict()
                    ),
                }
                for c in self.candidates()
            ],
        }

    def for_arch(self, arch: str) -> Tuple[KernelCandidate, ...]:
        """Return the candidates declaring ``arch``, without needing a request.

        A candidate that has not declared a capability is excluded: it has made
        no claim about ``arch``, and guessing one from its predicate would
        require a request, which is exactly what this avoids.
        """
        return tuple(
            c
            for c in self.candidates()
            if c.capability is not None and arch in c.capability.arches
        )

    def supported(self, request: OperatorRequest) -> Tuple[KernelCandidate, ...]:
        return tuple(c for c in self.candidates() if c.admits(request)[0])

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
            ok, why = candidate.admits(request)
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

    def build(self) -> Any:
        """Build the IR for this selection.

        ``dispatch_gemm_fp16(req).build()`` replaces the per-family
        ``build_kernel(result)`` call sites, so "compile whatever dispatch
        chose" can be written once over ``dispatch_*_all``.
        """
        return self.candidate.built(self.spec, self.request.arch)

    def bind(self, *, verify: bool = False) -> ProblemBinding:
        """Turn this selection into a runnable problem.

        The call site for anything that wants to execute what the dispatcher
        chose: ``dispatch_gemm_fp16(req).bind(verify=True)``.
        """
        return self.candidate.bound(self, verify=verify)
