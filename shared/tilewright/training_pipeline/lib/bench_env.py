# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Environment for every hipblaslt-bench run the pipeline makes.

`TENSILE_PREDICTION_LIB=1` routes `MasterSolutionLibrary::findTopSolutions`
through `ProblemPredictionLibrary`, the library type the models are trained
for and deployed into.

Benches never inherit `TENSILE_USE_TILEWRIGHT` or any `TILEWRIGHT_*` variable
from the caller. A training bench that ran with a previously deployed model
switched on would order and skip-screen candidates with that model instead of
Origami, so its labels and stage04's `is_origami_pick` column would be wrong.
A caller that wants tilewright on (stage07's ML-on runs) passes the variables
explicitly through `bench_env_with(..., tilewright={...})`.

Benches do not inherit `HIPBLASLT_TENSILE_LIBPATH` either: hipBLASLt loads
that directory verbatim instead of the library subtree matching the device's
ASIC revision. A caller that needs it (stage07) passes it in `extra`.
"""
from __future__ import annotations

import os
from typing import Dict, Mapping, Optional

TILEWRIGHT_ENABLE_VAR = "TENSILE_USE_TILEWRIGHT"
_TILEWRIGHT_PREFIX = "TILEWRIGHT_"
LIBPATH_VAR = "HIPBLASLT_TENSILE_LIBPATH"

REQUIRED_BENCH_ENV: Dict[str, str] = {
    "TENSILE_PREDICTION_LIB": "1",
    "HSA_HOTSWAP_DISABLE": "1",
}


def is_tilewright_var(name: str) -> bool:
    """True for `TENSILE_USE_TILEWRIGHT` and every `TILEWRIGHT_*` name."""
    return name == TILEWRIGHT_ENABLE_VAR or name.startswith(_TILEWRIGHT_PREFIX)


def bench_env_with(
    extra: Optional[Mapping[str, object]] = None,
    *,
    tilewright: Optional[Mapping[str, object]] = None,
) -> Dict[str, str]:
    """Return the environment for one hipblaslt-bench process.

    Starts from a copy of `os.environ` without the tilewright variables and
    `HIPBLASLT_TENSILE_LIBPATH`, then applies `REQUIRED_BENCH_ENV`, then
    `extra` (which may override the required knobs), then `tilewright`.

    `extra` must not name a tilewright variable and `tilewright` may name only
    tilewright variables; either mistake raises ValueError, so turning
    tilewright on always reads as such at the call site, e.g.
    `bench_env_with({"TENSILE_DB": "0x10000"},
    tilewright={"TENSILE_USE_TILEWRIGHT": "1", "TILEWRIGHT_DIAG": "1"})`."""
    extra_env = {str(k): str(v) for k, v in (extra or {}).items()}
    tw_env = {str(k): str(v) for k, v in (tilewright or {}).items()}
    misplaced = sorted(k for k in extra_env if is_tilewright_var(k))
    if misplaced:
        raise ValueError(
            f"tilewright variables {misplaced} must be passed through "
            f"bench_env_with(tilewright=...), not `extra`"
        )
    foreign = sorted(k for k in tw_env if not is_tilewright_var(k))
    if foreign:
        raise ValueError(
            f"bench_env_with(tilewright=...) accepts only "
            f"{TILEWRIGHT_ENABLE_VAR} and {_TILEWRIGHT_PREFIX}* names; got {foreign}"
        )
    env = {
        k: v
        for k, v in os.environ.items()
        if not is_tilewright_var(k) and k != LIBPATH_VAR
    }
    env.update(REQUIRED_BENCH_ENV)
    env.update(extra_env)
    env.update(tw_env)
    return env
