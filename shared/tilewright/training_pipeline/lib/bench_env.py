# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Environment for every hipblaslt-bench run the pipeline makes.

`TENSILE_PREDICTION_LIB=1` routes `MasterSolutionLibrary::findTopSolutions`
through `ProblemPredictionLibrary`, the library type the models are trained
for and deployed into. Production sets no such switch: TensileLite then
searches every row of the problem's library files in their order, and an
Equality row of the problem's size or a GridBased row ahead of the Prediction
row answers first. Stage 7's production-route run benches that way
(`bench_env_with(..., prediction_lib=False)`), to see which row serves the
problems the model was meant for.

Benches never inherit `TENSILE_USE_TILEWRIGHT` or any `TILEWRIGHT_*` or
`TENSILE_TILEWRIGHT_*` variable from the caller. A training bench that ran
with a previously deployed model switched on would order and skip-screen
candidates with that model instead of Origami, so its labels and stage04's
`is_origami_pick` column would be wrong. `TENSILE_TILEWRIGHT_BANK` decides
which kernels tilewright ranks and `TENSILE_TILEWRIGHT_PREPARE` when it
builds its per-cell data, so inherited values would make stage07's pick
parity and selection times follow the caller's shell. A caller that wants
tilewright on (stage07's ML-on runs) passes the variables explicitly through
`bench_env_with(..., tilewright={...})`.

Benches do not inherit `HIPBLASLT_TENSILE_LIBPATH` either: hipBLASLt loads
that directory verbatim instead of the library subtree matching the device's
ASIC revision. A caller that needs it (stage07) passes it in `extra`.

Nor do they inherit TensileLite's launch overrides (`is_launch_override`):
`TENSILE_FIXED_*`, `TENSILE_PERSISTENT_*`, `TENSILE_STREAMK_*`, the legacy
hybrid force mode `TENSILE_STREAMK5_FORCE_MODE`, `TENSILE_DISABLE_STAGGERU`,
`TENSILE_PRECISE_SM_TARGET` and `TENSILE_AUTO_GSU_ALGO`. They fix how the
kernel a bench times is launched (its workgroup mapping, stagger,
persistent grid, Stream-K decomposition, hybrid schedule, split-K, the
rounding of the CU budget) or print the launch choices into the bench's
output, so a label taken under one measures another launch than the
runtime makes, and a forced dynamic schedule would also measure another
execution context than the exclusive one the models serve. A config that
means to bench under one sets it in `runtime.env`; platform.json and the
deploy record say which were set (lib/launch_settings.py). The pipeline
itself sets none of them: an execution context's schedule reaches its
benches as hipblaslt-bench's `--hybrid_assignment_policy`
(lib/contexts.py), which the hybrid force modes (HYBRID_FORCE_VARS)
override, so the config check warns about those apart.

A run sets variables of its own through the config's `runtime.env`
(`runtime_env`), such as the switch that makes a device report its strict
stepping (lib/arch.py): the orchestrator puts them in the environment of
every stage, and the stages that bench pass them to every bench and
co-tenant. They cannot name a variable the pipeline sets or removes itself
(`managed_var`). platform.json records them as the benches had them
(`recorded_env`).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .arch import STRICT_SWITCH

TILEWRIGHT_ENABLE_VAR = "TENSILE_USE_TILEWRIGHT"
PREDICTION_LIB_VAR = "TENSILE_PREDICTION_LIB"
TILEWRIGHT_BANK_VAR = "TENSILE_TILEWRIGHT_BANK"
_TILEWRIGHT_PREFIXES: Tuple[str, ...] = ("TILEWRIGHT_", "TENSILE_TILEWRIGHT_")
LIBPATH_VAR = "HIPBLASLT_TENSILE_LIBPATH"
LAUNCH_OVERRIDE_PREFIXES: Tuple[str, ...] = (
    "TENSILE_FIXED_",
    "TENSILE_PERSISTENT_",
    "TENSILE_STREAMK_",
)
LAUNCH_OVERRIDE_VARS: Tuple[str, ...] = (
    "TENSILE_STREAMK5_FORCE_MODE",
    "TENSILE_DISABLE_STAGGERU",
    "TENSILE_PRECISE_SM_TARGET",
    "TENSILE_AUTO_GSU_ALGO",
)
HYBRID_FORCE_VARS: Tuple[str, ...] = (
    "TENSILE_PERSISTENT_HYBRID_FORCE_MODE",
    "TENSILE_STREAMK5_FORCE_MODE",
)

REQUIRED_BENCH_ENV: Dict[str, str] = {
    PREDICTION_LIB_VAR: "1",
    "HSA_HOTSWAP_DISABLE": "1",
}
# The runtime's device lists; the pipeline points every bench at its GPU.
DEVICE_LIST_VARS: Tuple[str, ...] = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_tilewright_var(name: str) -> bool:
    """True for `TENSILE_USE_TILEWRIGHT` and every `TILEWRIGHT_*` and
    `TENSILE_TILEWRIGHT_*` name."""
    return name == TILEWRIGHT_ENABLE_VAR or name.startswith(_TILEWRIGHT_PREFIXES)


def is_launch_override(name: str) -> bool:
    """True for every variable of TensileLite's launch-override family:
    `TENSILE_FIXED_*`, `TENSILE_PERSISTENT_*` and `TENSILE_STREAMK_*`, and
    LAUNCH_OVERRIDE_VARS."""
    return name in LAUNCH_OVERRIDE_VARS or name.startswith(LAUNCH_OVERRIDE_PREFIXES)


def launch_overrides(env: Mapping[str, str]) -> Dict[str, str]:
    """{name: value} of the launch overrides in a bench environment, by
    name."""
    return {k: env[k] for k in sorted(env) if is_launch_override(k)}


def managed_var(name: str) -> bool:
    """True for a variable the pipeline sets or removes in a bench's
    environment itself. Launch overrides are not: `runtime.env` may set
    them."""
    return (
        is_tilewright_var(name)
        or name == LIBPATH_VAR
        or name in REQUIRED_BENCH_ENV
        or name in DEVICE_LIST_VARS
    )


def runtime_env_errors(value: Any, where: str = "runtime.env") -> List[str]:
    """What is wrong with a `runtime.env` value: a mapping of variable names
    to non-empty strings or integers, none of them `managed_var`."""
    if value is None:
        return []
    if not isinstance(value, Mapping):
        return [f"{where}: expected a mapping of variable names to values"]
    errors: List[str] = []
    for name, v in value.items():
        if not isinstance(name, str) or not _ENV_NAME_RE.match(name):
            errors.append(f"{where}: {name!r} is not an environment variable name")
            continue
        if managed_var(name):
            errors.append(
                f"{where}.{name}: the pipeline sets this variable itself; drop it"
            )
        if isinstance(v, bool) or not isinstance(v, (str, int)) or v == "":
            errors.append(
                f"{where}.{name}: expected a non-empty string or an integer, "
                f"got {v!r}"
            )
    return errors


def runtime_env(cfg: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    """The config's `runtime.env` as strings: the variables every stage,
    bench and co-tenant of the run gets."""
    env = ((cfg or {}).get("runtime") or {}).get("env") or {}
    return {str(k): str(v) for k, v in env.items()}


def runtime_env_text(cfg: Optional[Mapping[str, Any]]) -> str:
    """`runtime_env` as `NAME=value` pairs; empty without one."""
    return ", ".join(f"{k}={v}" for k, v in runtime_env(cfg).items())


def recorded_env(
    cfg: Optional[Mapping[str, Any]], environ: Mapping[str, str]
) -> Dict[str, Optional[str]]:
    """{name: its value in `environ`, None when unset} of the variables
    `runtime.env` names and of the strict-mode switch (lib/arch.py), which
    decides the library subtree a device loads."""
    return {n: environ.get(n) for n in sorted({STRICT_SWITCH, *runtime_env(cfg)})}


def bench_env_with(
    extra: Optional[Mapping[str, object]] = None,
    *,
    tilewright: Optional[Mapping[str, object]] = None,
    prediction_lib: bool = True,
) -> Dict[str, str]:
    """Return the environment for one hipblaslt-bench process.

    Starts from a copy of `os.environ` without the tilewright variables,
    `HIPBLASLT_TENSILE_LIBPATH` and the launch overrides, then applies
    `REQUIRED_BENCH_ENV`, then `extra` (which may override the required
    knobs; the stages pass the config's runtime.env in it, the only way a
    launch override reaches a bench), then `tilewright`. With
    `prediction_lib` false the environment holds no `TENSILE_PREDICTION_LIB`
    at all, the caller's included, as production runs.

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
        accepted = [TILEWRIGHT_ENABLE_VAR] + [f"{p}*" for p in _TILEWRIGHT_PREFIXES]
        raise ValueError(
            f"bench_env_with(tilewright=...) accepts only {accepted}; got {foreign}"
        )
    env = {
        k: v
        for k, v in os.environ.items()
        if not is_tilewright_var(k) and k != LIBPATH_VAR and not is_launch_override(k)
    }
    env.update(REQUIRED_BENCH_ENV)
    env.update(extra_env)
    env.update(tw_env)
    if not prediction_lib:
        env.pop(PREDICTION_LIB_VAR, None)
    return env
