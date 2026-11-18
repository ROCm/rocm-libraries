# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Routing of the attention backward among its families (selection policy).

A request the general predicate accepts goes to exactly one family:

    attn_bwd_support (general predicate) -> decline, or OK
    OK -> dense-eligible and its dense key in the tuned dense pack -> dense plan
          else -> general plan (the general tuned geometry)
    rocKE unified request -> attn_bwd_unified_support -> unified plan

The decision reads request fields and the tuned tables only (no device
query). The kernels own the predicate
(:func:`kernels.common.attention_bwd_plan.attn_bwd_support`), the
eligibility (:func:`~kernels.common.attention_bwd_plan.dense_bwd_eligibility`)
and both plans; this module only chooses.

:func:`dense_policy` turns a tuned dense table (``{(arch, dense key):
knobs}``, :data:`~.dense_tuning.DENSE_BWD_TUNED` by default) into the
policy's ``dense_pack`` and a tuning hook that answers the dense route from
the table and every other route from the base policy. Forced routes
(``route="general"`` / ``"dense"``) serve the gate runs that time both routes
on the same cell; a forced route keeps the request's own predicate verdict.

Serving: the routed policy serves every request the predicate accepts (and the
plan builds). The per-arch performance status of a class (``verified``,
``unverified``, ``below_baseline``; :data:`ARCH_TUNING`'s
``GENERAL_PERF_STATUS`` / ``DENSE_PERF_STATUS``, empty by policy, so every
class reads ``unverified``) is information only: the plan records it
(``AttnBwdPlan.perf_status``) and :func:`attention_bwd_declaration` publishes
it per class, and neither the route nor the knobs depend on it.

Every per-arch table entry is checked when this module is imported
(:func:`.tuned_tables.check_arch_tables`): its knobs are registered values
(not ``not_built``, ``gated`` or known wrong) and plan on a representative
request of its key; a bad entry fails the import.

The candidates, the ranker and the library-level entry points over this
routing are :mod:`.registry`.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Optional

from kernels.common.attention_bwd_plan import (
    AttnBwdPlan,
    AttnBwdPolicy,
    AttnBwdRequest,
    _dense_key,
    attn_bwd_declaration,
    attn_bwd_perf_class,
    attn_bwd_plan,
    attn_bwd_route,
    attn_bwd_support,
    shipped_policy,
)
from kernels.common.attention_bwd_unified_plan import (
    AttnBwdUnifiedPolicy,
    AttnBwdUnifiedRequest,
    attn_bwd_unified_plan,
    attn_bwd_unified_support,
)
from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

from . import gfx942_tuning, gfx950_tuning, tuned_tables
from .dense_tuning import DENSE_BWD_TUNED, dense_axis

__all__ = [
    "ARCH_TUNING",
    "BWD_ROUTES",
    "arch_tuned_policy",
    "attention_bwd_declaration",
    "attention_bwd_route",
    "dense_policy",
    "plan_attention_bwd",
    "plan_attention_bwd_unified",
    "routed_policy",
]

BWD_ROUTES = ("general", "dense")
# Per-arch tuned tables of the general family (GENERAL_TUNED_TABLE,
# GENERAL_PERF_STATUS) and of the dense family (DENSE_TUNED, joined into
# DENSE_BWD_TUNED; DENSE_PERF_STATUS): {arch: module}.
ARCH_TUNING = {mod.ARCH: mod for mod in (gfx942_tuning, gfx950_tuning)}


def _check_tuned_tables() -> None:
    """Every per-arch table entry is a registered, plannable configuration
    (raised at import; :func:`.tuned_tables.check_arch_tables`)."""
    for mod in ARCH_TUNING.values():
        tuned_tables.check_arch_tables(mod, dense_axis=dense_axis)


_check_tuned_tables()


def arch_tuned_policy(base: AttnBwdPolicy) -> AttnBwdPolicy:
    """``base`` with the per-arch general tuned tables (:data:`ARCH_TUNING`)
    joined into its ``tuned_table``, and the per-arch general and dense
    performance status into its ``perf_status`` (a base entry of the same key
    wins). Empty per-arch tables leave ``base`` as it is."""
    table, status = {}, {}
    for mod in ARCH_TUNING.values():
        table.update(mod.GENERAL_TUNED_TABLE)
        status.update(mod.GENERAL_PERF_STATUS)
        status.update(mod.DENSE_PERF_STATUS)
    if not table and not status:
        return base
    return dataclasses.replace(
        base,
        tuned_table={**table, **dict(base.tuned_table)},
        perf_status={**status, **dict(base.perf_status)},
    )


def dense_policy(
    base: AttnBwdPolicy, table: Optional[Mapping[tuple, Mapping[str, Any]]] = None
) -> AttnBwdPolicy:
    """``base`` with the dense pack and dense tuning of ``table``.

    ``table`` maps ``(arch, dense key)`` to the dense knobs of that key
    (default: the shipped tuned table). Every other route keeps the base
    policy's tuning.
    """
    table = DENSE_BWD_TUNED if table is None else table
    base_tuning = base.tuning

    def tuning(req, arch, route, perf_class):
        if route == "dense":
            # a key outside the table (a forced dense route) takes the
            # dense defaults
            return dict(table.get((arch, _dense_key(req)), {}))
        return base_tuning(req, arch, route, perf_class)

    return dataclasses.replace(
        base, dense_pack=frozenset(base.dense_pack) | frozenset(table), tuning=tuning
    )


def routed_policy() -> AttnBwdPolicy:
    """The shipped policy with the per-arch general tuned tables, the per-arch
    performance status and the shipped tuned dense table."""
    return dense_policy(arch_tuned_policy(shipped_policy()))


def attention_bwd_route(
    req: AttnBwdRequest, arch: str, *, policy: Optional[AttnBwdPolicy] = None
) -> tuple[str, str]:
    """``(route, tag)`` of an accepted request (see the module docstring)."""
    policy = routed_policy() if policy is None else policy
    return attn_bwd_route(req, arch, policy=policy)


def plan_attention_bwd(
    req: AttnBwdRequest,
    arch: str,
    *,
    policy: Optional[AttnBwdPolicy] = None,
    route: Optional[str] = None,
) -> AttnBwdPlan:
    """The plan of exactly one family for ``req`` on ``arch``.

    ``route=None`` routes by :func:`attention_bwd_route`; ``"general"`` /
    ``"dense"`` force a route (a forced dense route still needs a
    dense-eligible request; its knobs come from the policy's dense tuning).
    Raises ``ValueError`` when the predicate declines.
    """
    policy = routed_policy() if policy is None else policy
    verdict = attn_bwd_support(req, arch, policy=policy)
    if not verdict.ok:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    chosen, _tag = attn_bwd_route(req, arch, policy=policy)
    if route is not None:
        if route not in BWD_ROUTES:
            raise ValueError(f"route={route!r} not in {BWD_ROUTES}")
        chosen = route
    if chosen == "dense":
        return attn_dense_bwd_plan(req, arch, policy=policy)
    # General route: the plan's own routing would label a packed key "dense";
    # a forced general route plans without the dense pack and keeps the
    # request's verdict above (taken under the full policy, its opt-in
    # performance gate included), so the gate is not applied a second time
    # to the general class.
    general_policy = policy
    if attn_bwd_route(req, arch, policy=policy)[0] == "dense":
        general_policy = dataclasses.replace(
            policy, dense_pack=frozenset(), perf_gate=False
        )
    plan = attn_bwd_plan(req, arch, policy=general_policy)
    if plan.route != "general":
        raise AssertionError("general plan reports another route")
    return dataclasses.replace(
        plan, perf_class=attn_bwd_perf_class(req, arch, "general")
    )


def attention_bwd_declaration(
    arch: str, *, policy: Optional[AttnBwdPolicy] = None
) -> dict:
    """The backward declaration of ``arch`` under the routed policy
    (:func:`routed_policy`): served classes and the performance status of
    every general class and every dense-pack key."""
    policy = routed_policy() if policy is None else policy
    return attn_bwd_declaration(arch, policy=policy)


def plan_attention_bwd_unified(
    req: AttnBwdUnifiedRequest,
    arch: str,
    *,
    policy: Optional[AttnBwdUnifiedPolicy] = None,
) -> AttnBwdPlan:
    """The unified family's plan for a rocKE unified (paged KV) request.

    A unified request is its own entry point (THD queries over a paged KV
    cache, not a hipDNN request): ``attn_bwd_unified_support`` declines or the
    unified plan serves it (route ``"unified"``). Raises ``ValueError`` with
    the reason code when the predicate declines.
    """
    verdict = attn_bwd_unified_support(req, arch, policy=policy)
    if not verdict.ok:
        raise ValueError(f"{verdict.code}: {verdict.detail}")
    return attn_bwd_unified_plan(req, arch, policy=policy)
