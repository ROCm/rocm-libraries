# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Conformance kit: the contract every tuned candidate must keep.

A family's test calls :func:`assert_tuning_contract` for a representative set
of its candidates and requests. It goes through the public candidate API only
(``admits`` / ``select_spec`` / ``sweep_space`` / ``sample_space`` / ``grid``
/ ``block``), so it checks what callers and sweeps actually see.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Mapping, Optional, Sequence, Tuple

from ..core import KernelCandidate, pin_to_spec, spec_identity
from .identity import names_variant
from .spec import TunableRequest, TunedSpec
from .walk import sweep_level


class TuningContractError(AssertionError):
    pass


def _fail(candidate: KernelCandidate, message: str):
    raise TuningContractError(f"{candidate.name}: {message}")


def _select(candidate, req, what: str):
    """``candidate.select_spec(req)``, a refusal reported as a contract error."""
    try:
        return candidate.select_spec(req)
    except ValueError as e:
        _fail(candidate, f"{what}: refused ({e})")


def _pinned(candidate, req, tuning_id="auto", knobs=()):
    return replace(
        req,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        tuning_id=tuning_id,
        tuning_knobs=dict(knobs),
    )


def _streams(candidate, req, sample: int, seed: int):
    with sweep_level("production"):
        production = list(candidate.sweep_space(req))
    with sweep_level("full"):
        sampled = list(candidate.sample_space(req, sample, seed)) if sample else []
    return production, sampled


def _check_spec(candidate, spec):
    if not isinstance(spec, TunedSpec):
        _fail(candidate, f"{type(spec).__name__} does not satisfy TunedSpec")
    if not spec.tuning_id.endswith(f"@{spec.config_key}"):
        _fail(candidate, f"{spec.tuning_id!r} does not end in its config_key")
    if not names_variant(spec.tuning_id, spec.variant_id):
        _fail(candidate, f"{spec.tuning_id!r} does not name variant {spec.variant_id!r}")
    if list(spec.knobs) != sorted(spec.knobs):
        _fail(candidate, f"{spec.tuning_id!r} knobs are not sorted pairs")
    if replace(spec) != spec or spec_identity(replace(spec)) != spec_identity(spec):
        _fail(candidate, f"{spec.tuning_id!r} identity is not stable")


def assert_tuning_contract(
    candidate: KernelCandidate,
    requests: Sequence[TunableRequest],
    *,
    other_requests: Sequence[TunableRequest] = (),
    default_knobs: Optional[Mapping[str, object]] = None,
    refused_knobs: Iterable[Mapping[str, object]] = (),
    max_replays: int = 6,
    sample: int = 6,
    seed: int = 0,
) -> dict:
    """Raise :class:`TuningContractError` unless ``candidate`` keeps the contract.

    For each request (unpinned; the kit pins it):

    - the pinned ``auto`` request is admitted, and its spec is the first
      production spec;
    - every production and sampled spec satisfies :class:`TunedSpec`, and ids
      are unique within each stream;
    - ``(tuning_id, knobs)`` and knobs alone reselect the same spec, a bare id
      does too, and knobs paired with another spec's id are refused;
    - the request :func:`~rocke.dispatch.core.pin_to_spec` stores reselects
      the spec, and the candidate's ``grid`` / ``block`` accept it;
    - ``default_knobs`` (knobs set to their default values) give the default
      spec, and each of ``refused_knobs`` is refused;
    - on every one of ``other_requests`` where the same knobs stay canonical
      (none is dropped as inert there), the ``config_key`` is unchanged;
    - a pin whose id differs from the spec's only in the display stem still
      reselects the spec.

    Returns counts of what was checked.
    """
    counts = {"requests": 0, "specs": 0, "replays": 0, "portable": 0}
    for req in requests:
        counts["requests"] += 1
        auto = _pinned(candidate, req)
        ok, why = candidate.admits(auto)
        if not ok:
            _fail(candidate, f"pinned auto request refused: {why}")
        default = _select(candidate, auto, "auto")
        production, sampled = _streams(candidate, auto, sample, seed)
        if not production:
            _fail(candidate, "production stream is empty for an admitted request")
        if production[0] != default:
            _fail(candidate, "auto does not select the first production spec")
        for stream in (production, sampled):
            ids = [s.tuning_id for s in stream]
            if len(ids) != len(set(ids)):
                _fail(candidate, "a stream repeats a tuning_id")
            for spec in stream:
                _check_spec(candidate, spec)
        counts["specs"] += len(production) + len(sampled)

        replays = production[:max_replays] + sampled[: max(1, max_replays // 2)]
        for spec in replays:
            by_both = _pinned(candidate, req, spec.tuning_id, spec.knobs)
            if _select(candidate, by_both, f"replaying {spec.tuning_id!r}") != spec:
                _fail(candidate, f"(id, knobs) do not replay {spec.tuning_id!r}")
            if _select(candidate, _pinned(candidate, req, knobs=spec.knobs), f"knobs of {spec.tuning_id!r}") != spec:
                _fail(candidate, f"knobs alone do not replay {spec.tuning_id!r}")
            restemmed = f"{spec.variant_id}_wpe0@{spec.config_key}"
            if _select(candidate, _pinned(candidate, req, restemmed, spec.knobs),
                       f"restemmed {spec.tuning_id!r}") != spec:
                _fail(candidate, f"a display-stem change refuses {spec.tuning_id!r}")
            stored = pin_to_spec(auto, candidate, spec)
            if _select(candidate, stored, f"stored request for {spec.tuning_id!r}") != spec:
                _fail(candidate, f"stored request does not reselect {spec.tuning_id!r}")
            candidate.grid(spec, stored)
            candidate.block(spec)
            counts["replays"] += 1
        for spec in production[:2]:
            if _select(candidate, _pinned(candidate, req, spec.tuning_id), f"bare id {spec.tuning_id!r}") != spec:
                _fail(candidate, f"bare id does not resolve {spec.tuning_id!r}")
        mismatched = next(
            (
                (a, b)
                for a, b in zip(replays, replays[1:], strict=False)
                if a.knobs != b.knobs
            ),
            None,
        )
        if mismatched is not None:
            a, b = mismatched
            if candidate.admits(_pinned(candidate, req, a.tuning_id, b.knobs))[0]:
                _fail(candidate, "knobs that do not reproduce the id are admitted")

        if default_knobs:
            pinned = _pinned(candidate, req, knobs=default_knobs)
            if _select(candidate, pinned, f"default knobs {dict(default_knobs)}") != default:
                _fail(candidate, f"{dict(default_knobs)} did not canonicalize to the default")
        for knobs in refused_knobs:
            if candidate.admits(_pinned(candidate, req, knobs=knobs))[0]:
                _fail(candidate, f"{dict(knobs)} should be refused")

        for other in other_requests:
            for spec in production[:max_replays]:
                pinned = _pinned(candidate, other, knobs=spec.knobs)
                if not candidate.admits(pinned)[0]:
                    continue
                moved = _select(candidate, pinned, "portable knobs")
                if moved.knobs != spec.knobs:
                    continue  # some knob is inert on this problem and was dropped
                if moved.config_key != spec.config_key:
                    _fail(candidate, f"config_key of {spec.tuning_id!r} depends on the problem")
                counts["portable"] += 1
    return counts


def representative(
    candidates: Iterable[KernelCandidate], prefixes: Sequence[str]
) -> Tuple[KernelCandidate, ...]:
    """The first candidate whose name starts with each prefix, for a test that
    cannot afford to check every registered variant."""
    pool = tuple(candidates)
    chosen = []
    for prefix in prefixes:
        match = next((c for c in pool if c.name.startswith(prefix)), None)
        if match is None:
            raise LookupError(f"no candidate named {prefix!r}*")
        chosen.append(match)
    return tuple(chosen)
