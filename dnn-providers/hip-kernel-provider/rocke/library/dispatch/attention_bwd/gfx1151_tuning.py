# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tuned tables of the attention backward on gfx1151 (selection policy).

The gfx1151 entries of the tuned tables the backward routing reads
(:func:`~.routing.routed_policy`), in the layout of :mod:`.gfx942_tuning`.

* :data:`GENERAL_TUNED_TABLE`: ``{tuning key: knobs}`` of the general family,
  keyed by :func:`kernels.common.attention_bwd_plan.attn_bwd_tuning_key` (with
  the device's ``num_cus``).

**Entry rule (RDNA).** On the RDNA targets the rocKE backward is slower than
PyTorch's own backward on every measured cell, so the CDNA rule (an entry wins
the bench's confirm stage against the competitor on every cell of its key)
admits no entry. Here rocKE is the only hipDNN backward, so an entry is the
best rocKE configuration instead: a finalist of the sweep's ``runtime`` stage
(``benchmarks/common/attention_bwd_sweep.py --arch gfx1151``, primary general
slices) that, timed with the bench on every dense census cell of its key
together with the shipped plan (``--rocke-configs`` holding ``shipped`` and the
finalists, ``--aa-control all``, the pruning-set gate on), on every such cell

1. is correct (its arm completes and passes the pruning-set gate),
2. has a clean A/A control (its twin is not judged different from it either
   way), and
3. is not slower than the shipped plan (the paired verdict against it is
   ``pass``, or ``ambiguous`` with a time ratio within ``TIE_EPS``).

Of the finalists that pass every cell of a key, the one with the lowest
geometric-mean time ratio against the shipped plan is the entry, and only when
that ratio is below 1. Keys whose cells disagree or have no qualifying
finalist are left out (they keep the defaults). The same timing records the
gap to PyTorch's default backward (information only; no claim is made). Head
size 32 is not swept (the census has no d = 32 cell), so it keeps the defaults
as on CDNA.

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

ARCH = "gfx1151"

# {attn_bwd_tuning_key(req, "gfx1151"): knobs}
GENERAL_TUNED_TABLE: Mapping[tuple, Mapping[str, Any]] = {}
# the dense family is not built on RDNA
DENSE_TUNED: Mapping[tuple, Mapping[str, Any]] = {}
# RDNA classes read degraded (AttnBwdPolicy.status_of): no status entries
GENERAL_PERF_STATUS: Mapping[tuple, str] = {}
DENSE_PERF_STATUS: Mapping[tuple, str] = {}


def check_general_entry(key: tuple, knobs: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` unless ``(key, knobs)`` is a gfx1151 general entry
    whose knobs plan on the key (:func:`.tuned_tables.check_general_entry`)."""
    tuned_tables.check_general_entry(ARCH, key, knobs)


def general_entries(doc: Mapping | Iterable) -> dict:
    """Checked gfx1151 general entries of a general table document
    (:func:`.tuned_tables.general_entries`)."""
    return tuned_tables.general_entries(ARCH, doc)
