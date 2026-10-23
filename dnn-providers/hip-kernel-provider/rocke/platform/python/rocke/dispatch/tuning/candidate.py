# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The opt-in tuned candidate every family builds the same way.

Admitted only when the request names both its ``algorithm`` and ``spec_id``.
Its configuration comes from the request's ``tuning_knobs`` (rebuilt
directly), else its ``tuning_id`` (``auto``: the space's default; otherwise a
search of the space). ``support`` and ``select_spec`` share one resolution per
request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Optional, Sequence, Tuple

from ..core import Capability, KernelCandidate, OperatorRequest, normalize_selector
from .identity import key_of
from .space import KnobSpace, Verdict
from .walk import current_sweep_level


def explicitly_pinned(req, algorithm: str, spec_id: str) -> bool:
    """Whether ``req`` names this opt-in candidate: both ``algorithm`` and
    ``spec_id`` must match (sweeps pin both through ``opt_in_probe``)."""
    return normalize_selector(req.spec_id) == normalize_selector(
        spec_id
    ) and normalize_selector(req.algorithm) == normalize_selector(algorithm)


def resolve_pinned(
    req,
    *,
    default: Callable[[], Verdict],
    canonicalize: Callable[[dict], Verdict],
    find: Callable[[str], Optional[Any]],
) -> Verdict:
    """The configuration ``req`` pins on an admitted candidate.

    The recorded knobs (possibly empty: the default configuration) are rebuilt
    directly and must reproduce the id's ``config_key``; the display stem of
    the id is not compared. Only when no knobs were recorded and the empty set
    does not reproduce the key does a bare id fall back to searching the
    production set (bounded; full-space ids need their knobs).
    A pin that does not resolve is refused with the reason; the candidate never
    substitutes another configuration.
    """
    wanted = req.tuning_id.strip() or "auto"
    knobs = dict(req.tuning_knobs)
    if wanted == "auto" and not knobs:
        return default()
    spec, why = canonicalize(knobs)
    if wanted == "auto":
        return spec, why
    key = key_of(wanted)
    if spec is not None and spec.config_key == key:
        return spec, "ok"
    if knobs:
        if spec is None:
            return None, why
        return None, (
            f"stored tuning_id {wanted!r} is stale: tuning_knobs canonicalize to "
            f"tuning_id {spec.tuning_id!r} (config_key {spec.config_key!r}), not "
            f"config_key {key!r}; the knobs, the defaults they are relative to, "
            "or the identity schema changed; re-sweep/revalidate and replace the "
            "stored tuning_id; no fallback was selected"
        )
    found = find(wanted)
    if found is None:
        current = (
            ""
            if spec is None
            else f"; this candidate's current default is {spec.tuning_id!r}"
        )
        return None, (
            f"unknown tuning_id {wanted!r}: no production spec has config_key "
            f"{key!r}{current}. A full-space id must be pinned with its "
            "tuning_knobs; an id generated under older defaults or an older "
            "identity schema must be re-swept/revalidated and replaced. No "
            "fallback was selected"
        )
    return found, "ok"


def make_tuned_candidate(
    *,
    name: str,
    family: str,
    algorithm: str,
    spec_id: str,
    abi_version: str,
    priority: int,
    capability: Capability,
    space: KnobSpace,
    base: Callable[[Any], Any],
    request_errors: Callable[[Any], Sequence[str]],
    signature: Callable[[Any], Sequence[Any]],
    build: Callable[[Any, str], Any],
    bind_torch: Callable[..., Any],
    grid: Callable[[Any, Any], Tuple[int, int, int]],
    block: Callable[[Any], Tuple[int, int, int]] = lambda spec: spec.launch_block(),
    precheck: Optional[Callable[[Any], Tuple[bool, str]]] = None,
) -> KernelCandidate:
    """An opt-in candidate over ``space``.

    ``base(req)`` is the input the space builds from for one request (raise
    ``ValueError`` when the request cannot have one). It must be a pure function
    of the request's hash/equality-visible fields: the result is memoized and
    shared by pin resolution, sweeps and sampling. Device or process state must
    first be normalized into the request. ``precheck(req)`` is any request-level
    gate to run before resolving (a backend coverage check). The remaining
    callables are the family's build, signature, grid and Torch binding for its
    tuned spec.
    """

    @lru_cache(maxsize=32)
    def resolve_base(req):
        try:
            return base(req), "ok"
        except ValueError as e:
            return None, str(e)

    @lru_cache(maxsize=32)
    def resolve(req) -> Verdict:
        b, why = resolve_base(req)
        if b is None:
            return None, why
        return resolve_pinned(
            req,
            default=lambda: space.default(b),
            canonicalize=lambda knobs: space.canonicalize(b, knobs),
            find=lambda wanted: space.find(b, wanted),
        )

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = request_errors(req)
        if errors:
            return False, "; ".join(errors)
        if not explicitly_pinned(req, algorithm, spec_id):
            return False, f"opt-in: pin algorithm={algorithm!r} and spec_id={spec_id!r}"
        if precheck is not None:
            ok, why = precheck(req)
            if not ok:
                return False, why
        spec, why = resolve(req)
        return spec is not None, why

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return resolve(req)[0]

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        b, _why = resolve_base(req)
        return space.stream(b, current_sweep_level())

    def sample(req: OperatorRequest, n: int, seed: int):
        if not candidate.admits(req)[0]:
            return ()
        b, _why = resolve_base(req)
        return space.sample(b, n, seed)

    candidate = KernelCandidate(
        name=name,
        family=family,
        algorithm=algorithm,
        spec_id=spec_id,
        abi_version=abi_version,
        priority=priority,
        capability=capability,
        _supports=support,
        select_spec=select,
        signature=signature,
        grid=grid,
        block=block,
        sweep_space=sweep,
        sample_space=sample,
        build=build,
        bind_torch=bind_torch,
        opt_in=True,
    )
    return candidate
