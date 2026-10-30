# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The engine's drift fallback, mirrored: which cells of an MLREC_v3 model a
kernel pool serves degraded (shared/tilewright/src/tilewright/rank.cpp,
drift_degraded). The drift is the kernel pool's: the library a model is
served from no longer holds every kernel of its whitelist tier, unlike the
bench canary's drift (lib/bench_log.py) or a config's (lib/pinning.py).

A model served with the drift fallback (serving flag bit 10, lib/mlrec.py)
declares `keep_ppm`, a share in millionths. Of a cell's whitelisted view keys
the model lists as trained (every whitelisted view key without the
trained-key gate), each counted once, `total` is their number and `present`
the number of which the pool holds a kernel; the cell is degraded when
`total > 0` and `present * 10**6 < keep_ppm * total`, in integers. Nothing
else enters: not the problem, the context, the order of the pool or its
other kernels. A pool that holds every trained whitelisted key degrades no
cell, whatever it adds.

A degraded cell ranks its whitelisted kernels and the cascade over its other
kernels as one tier (README.md of the engine). The fallback is set at deploy
only (`deploy.drift_fallback`, stages/stage06_deploy_weights.py).
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import features as fs
from . import mlrec

_U64 = (1 << 64) - 1


def keep_ppm(keep: Any) -> int:
    """The share `keep` (in (0, 1]) as the engine's keep_ppm, exactly:
    ValueError unless `keep` times 10**6 is a whole number from 1 to
    mlrec.DRIFT_KEEP_MAX."""
    if isinstance(keep, bool):
        raise ValueError(f"the drift fallback's keep must be a number, got {keep!r}")
    try:
        ppm = Decimal(str(keep)) * mlrec.DRIFT_KEEP_MAX
    except (InvalidOperation, ValueError):
        raise ValueError(
            f"the drift fallback's keep must be a number, got {keep!r}"
        ) from None
    if (
        not ppm.is_finite()
        or ppm != ppm.to_integral_value()
        or not (1 <= ppm <= mlrec.DRIFT_KEEP_MAX)
    ):
        raise ValueError(
            f"the drift fallback's keep must be a share in (0, 1] in whole "
            f"millionths, got {keep!r}"
        )
    return int(ppm)


def model_keep_ppm(parsed: Mapping[str, Any]) -> Optional[int]:
    """The keep_ppm a parsed model (lib/mlrec.py read_model or read_header)
    is served with, or None without the drift fallback."""
    keep = (parsed.get("serving") or {}).get("drift_keep_ppm")
    return None if keep is None else int(keep)


def pool_hashes(
    kernels: Iterable[Mapping[str, Any]], spec: fs.AttributeSpec
) -> List[int]:
    """The view-key hashes of a pool's kernels (pool dicts with the config
    fields and `attributes`) under the attribute spec `spec`."""
    return [
        fs.view_key_hash(
            fs.v3_view_key(fs.config_kwargs_from_row(k), k.get("attributes"), spec)
        )
        for k in kernels
    ]


def _trained(parsed: Mapping[str, Any]) -> Optional[set]:
    serving = parsed.get("serving") or {}
    keys = parsed.get("trained_view_keys")
    if not serving.get("trained_key_gate") or keys is None:
        return None
    return {int(h) & _U64 for h in keys}


def cell_counts(
    parsed: Mapping[str, Any], pool: Iterable[int]
) -> List[Tuple[int, int]]:
    """Per cell of a parsed model (lib/mlrec.py read_model, file order, which
    is the engine's cell index): (present, total) for the pool whose
    view-key hashes are `pool`."""
    held = {int(h) & _U64 for h in pool}
    trained = _trained(parsed)
    out: List[Tuple[int, int]] = []
    for cell in parsed["cells"]:
        listed = {int(h) & _U64 for h in cell.get("view_key_hashes") or []}
        if trained is not None:
            listed &= trained
        out.append((len(listed & held), len(listed)))
    return out


def degraded_cells(
    parsed: Mapping[str, Any], pool: Iterable[int], keep: Optional[int] = None
) -> List[int]:
    """The cells (indices, ascending) a pool with view-key hashes `pool`
    serves degraded: at the model's own keep_ppm, or at `keep` (keep_ppm)
    when given; none for a model without the fallback and no `keep`."""
    keep = model_keep_ppm(parsed) if keep is None else int(keep)
    if keep is None:
        return []
    if not 1 <= keep <= mlrec.DRIFT_KEEP_MAX:
        raise ValueError(f"keep_ppm {keep} is outside 1 to {mlrec.DRIFT_KEEP_MAX}")
    return [
        i
        for i, (present, total) in enumerate(cell_counts(parsed, pool))
        if total > 0 and present * mlrec.DRIFT_KEEP_MAX < keep * total
    ]


def missing_keys(
    parsed: Mapping[str, Any], pool: Iterable[int]
) -> Dict[int, List[int]]:
    """Per cell index with any: the trained whitelisted view-key hashes the
    pool `pool` holds no kernel of, ascending."""
    held = {int(h) & _U64 for h in pool}
    trained = _trained(parsed)
    out: Dict[int, List[int]] = {}
    for i, cell in enumerate(parsed["cells"]):
        listed = {int(h) & _U64 for h in cell.get("view_key_hashes") or []}
        if trained is not None:
            listed &= trained
        gone = sorted(listed - held)
        if gone:
            out[i] = gone
    return out


def drift_summary(
    parsed: Mapping[str, Any], pool: Sequence[int], keep: Optional[int] = None
) -> Dict[str, Any]:
    """What the engine's `[TILEWRIGHT_DIAG DRIFT]` line reports for the pool:
    degraded cells, cells with a whitelist, keep_ppm and the trained
    whitelisted keys the pool lacks, summed over the degraded cells."""
    keep = model_keep_ppm(parsed) if keep is None else int(keep)
    counts = cell_counts(parsed, pool)
    cells = degraded_cells(parsed, pool, keep) if keep is not None else []
    return {
        "keep_ppm": keep,
        "degraded": cells,
        "cells_with_whitelist": sum(
            1 for c in parsed["cells"] if c.get("view_key_hashes")
        ),
        "missing": sum(counts[i][1] - counts[i][0] for i in cells),
    }


__all__: Sequence[str] = (
    "cell_counts",
    "degraded_cells",
    "drift_summary",
    "keep_ppm",
    "missing_keys",
    "model_keep_ppm",
    "pool_hashes",
)
