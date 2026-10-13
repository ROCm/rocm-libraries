# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned tables of the attention backward on gfx950 (selection policy).

The gfx950 entries of the two tuned tables the backward routing reads
(:func:`~.routing.routed_policy`):

* :data:`GENERAL_TUNED_TABLE`: ``{tuning key: knobs}`` of the general family,
  keyed by :func:`kernels.common.attention_bwd_plan.attn_bwd_tuning_key` (the
  performance class, which names the arch, plus the kv-tile and grid-fill
  buckets; ``ANY_BUCKET`` matches any bucket). The plan takes the most
  specific match (``AttnBwdPolicy.tuned_table``).
* :data:`DENSE_TUNED`: ``{(arch, dense key): knobs}`` of the dense family,
  keyed by :func:`kernels.common.attention_bwd_plan._dense_key`; a request
  routes to the dense family only when its exact key is here (and its route
  is eligible). Merged into :data:`~.dense_tuning.DENSE_BWD_TUNED`.
* :data:`GENERAL_PERF_STATUS`: ``{performance class: status}`` of the general
  family (``AttnBwdPolicy.perf_status``).

The tables are empty: an entry is added only for a key whose winner passed the
sweep's confirm stage (a paired win over the current choice on every cell of
the key, a clean A/A control of each arm, the pruning-set gate on); the sweep
writes such entries (``attention_bwd_sweep.py --emit-dense-table``) and
:func:`dense_entries` / :func:`general_entries` turn them into table entries,
checking each one (gfx950, the key's shape, knobs that plan on the key) before
it is pasted here. With empty tables routing and planning are unchanged.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

__all__ = [
    "ARCH",
    "DENSE_TUNED",
    "GENERAL_PERF_STATUS",
    "GENERAL_TUNED_TABLE",
    "check_dense_entry",
    "check_general_entry",
    "dense_entries",
    "general_entries",
]

ARCH = "gfx950"

# {attn_bwd_tuning_key(req, "gfx950"): knobs}: empty until a key's winner
# passes the confirm stage.
GENERAL_TUNED_TABLE: Mapping[tuple, Mapping[str, Any]] = {}
# {("gfx950", dense key): dense knobs}: empty until a key's dense winner
# passes the confirm stage.
DENSE_TUNED: Mapping[tuple, Mapping[str, Any]] = {}
# {general performance class: status}
GENERAL_PERF_STATUS: Mapping[tuple, str] = {}

# Length of a dense key (kernels.common.attention_bwd_plan._dense_key).
_DENSE_KEY_LEN = 9


def _tuple(x):
    """JSON lists back to the tuples of a key (nested)."""
    if isinstance(x, list):
        return tuple(_tuple(v) for v in x)
    return x


def _knobs(raw: Mapping[str, Any]) -> dict:
    return {k: _tuple(v) for k, v in dict(raw).items()}


def check_dense_entry(key: tuple, knobs: Mapping[str, Any], req=None) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a gfx950 dense entry:
    ``key = ("gfx950", dense key)`` and, given the key's request ``req``, the
    dense knobs plan on it (the dense family's own rules)."""
    if not (isinstance(key, tuple) and len(key) == 2 and key[0] == ARCH):
        raise ValueError(f"dense key {key!r} is not a {ARCH} key")
    if not (isinstance(key[1], tuple) and len(key[1]) == _DENSE_KEY_LEN):
        raise ValueError(f"dense key {key[1]!r} is not a dense key")
    if req is not None:
        from kernels.common.attention_bwd_plan import (
            FEATURE_GROUPS,
            AttnBwdPolicy,
            _dense_key,
        )
        from kernels.common.attention_dense_bwd_plan import attn_dense_bwd_plan

        if _dense_key(req) != key[1]:
            raise ValueError(f"request key {_dense_key(req)!r} is not {key[1]!r}")
        # planned as the routing plans it: the entry is the dense tuning
        values = dict(knobs)
        policy = AttnBwdPolicy(
            declared_groups=frozenset(FEATURE_GROUPS),
            waive_unverified=True,
            tuning=lambda *_a: dict(values),
        )
        attn_dense_bwd_plan(req, ARCH, policy=policy)


def check_general_entry(key: tuple, knobs: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``key`` is a gfx950 general tuning key
    (``(arch, route, ...)``, the general route) and ``knobs`` a mapping of
    names to values."""
    if not (isinstance(key, tuple) and len(key) >= 3):
        raise ValueError(f"tuning key {key!r} is not a tuning key")
    if key[0] != ARCH or key[1] != "general":
        raise ValueError(f"tuning key {key!r} is not a {ARCH} general key")
    if not isinstance(knobs, Mapping) or not all(isinstance(k, str) for k in knobs):
        raise ValueError(f"knobs {knobs!r} are not a name -> value mapping")


def dense_entries(doc: Mapping | Iterable, requests: Mapping | None = None) -> dict:
    """``{(arch, dense key): knobs}`` of the sweep's dense table document
    (``{"entries": [{"arch", "key", "knobs", ...}]}`` or the entry list),
    each checked by :func:`check_dense_entry` (``requests``: ``{dense key:
    request}`` to plan the knobs on)."""
    entries = doc["entries"] if isinstance(doc, Mapping) else list(doc)
    out = {}
    for e in entries:
        key = (e["arch"], _tuple(e["key"]))
        knobs = _knobs(e["knobs"])
        req = None if requests is None else requests.get(key[1])
        check_dense_entry(key, knobs, req)
        if key in out and out[key] != knobs:
            raise ValueError(f"dense key {key!r} has two entries")
        out[key] = knobs
    return out


def general_entries(doc: Mapping | Iterable) -> dict:
    """``{tuning key: knobs}`` of a general table document (``{"entries":
    [{"key", "knobs"}]}`` or the entry list), each checked by
    :func:`check_general_entry`."""
    entries = doc["entries"] if isinstance(doc, Mapping) else list(doc)
    out = {}
    for e in entries:
        key = _tuple(e["key"])
        knobs = _knobs(e["knobs"])
        check_general_entry(key, knobs)
        if key in out and out[key] != knobs:
            raise ValueError(f"tuning key {key!r} has two entries")
        out[key] = knobs
    return out
