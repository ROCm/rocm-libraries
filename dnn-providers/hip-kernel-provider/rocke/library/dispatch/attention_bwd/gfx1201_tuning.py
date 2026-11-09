# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned tables of the attention backward on gfx1201 (selection policy).

The gfx1201 entries of the tuned tables the backward routing reads
(:func:`~.routing.routed_policy`), in the layout of :mod:`.gfx942_tuning`
(its docstring states the entry rule and the regeneration steps; they apply
here unchanged):

* :data:`GENERAL_TUNED_TABLE`: ``{tuning key: knobs}`` of the general family,
  keyed by :func:`kernels.common.attention_bwd_plan.attn_bwd_tuning_key`. An
  entry is the confirmed winner of the sweep
  (``benchmarks/common/attention_bwd_sweep.py --arch gfx1201``) on every
  measured cell of its key, run on a gfx1201 device.
* :data:`DENSE_TUNED`: always empty. The dense family is not built on RDNA
  (the sweep's ``FAMILY_ARCHS["dense"]``), so no request routes dense here.
* :data:`GENERAL_PERF_STATUS` / :data:`DENSE_PERF_STATUS`: always empty. An
  RDNA class reads ``degraded`` whatever these tables hold
  (``AttnBwdPolicy.status_of``), and RDNA is never declined on performance.

Every entry is checked when the package is imported
(:func:`.tuned_tables.check_arch_tables`, from :mod:`.routing`) and on every
representative request of its key in the tests.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from . import tuned_tables

__all__ = [
    "ARCH",
    "DENSE_PERF_STATUS",
    "DENSE_TUNED",
    "GENERAL_PERF_STATUS",
    "GENERAL_TUNED_TABLE",
    "check_general_entry",
    "general_entries",
]

ARCH = "gfx1201"

# {attn_bwd_tuning_key(req, "gfx1201"): knobs}
GENERAL_TUNED_TABLE: Mapping[tuple, Mapping[str, Any]] = {}
# the dense family is not built on RDNA
DENSE_TUNED: Mapping[tuple, Mapping[str, Any]] = {}
# RDNA classes read degraded (AttnBwdPolicy.status_of): no status entries
GENERAL_PERF_STATUS: Mapping[tuple, str] = {}
DENSE_PERF_STATUS: Mapping[tuple, str] = {}


def check_general_entry(key: tuple, knobs: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a gfx1201 general entry
    whose knobs plan on the key (:func:`.tuned_tables.check_general_entry`)."""
    tuned_tables.check_general_entry(ARCH, key, knobs)


def general_entries(doc: Mapping | Iterable) -> dict:
    """Checked gfx1201 general entries of a general table document
    (:func:`.tuned_tables.general_entries`)."""
    return tuned_tables.general_entries(ARCH, doc)
