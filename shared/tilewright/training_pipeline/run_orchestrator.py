# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Run the tilewright training pipeline for one config.

Modes:

    initial   round_0 only: seed shapes for every grid cell, bench, train.
    active    one more round on top of the complete rounds already in
              --run-dir (round_k, k = number of complete rounds). An
              incomplete round_k is resumed.
    full      round_0 plus active rounds up to `full.total_rounds`. Complete
              rounds are skipped, so re-running the same command resumes.
    validate  stages 7 and 8 against an existing --run-dir; no training.

Resume:

    --from-round R   full: re-run round R and every later round (rounds
                     before R must be complete). validate: take round R's
                     trained models instead of the latest round's.
    --from-stage N   re-run the target round from logical stage N
                     (1, 2, 3, 4, 4.5, 5, 6; validate mode: 7 or 8). Earlier
                     stages must have completed. Without it a resumed round
                     restarts at its first stage that did not complete.

A stage that is re-run gets a fresh output directory: the previous attempt's
directory and log move to `<round>/stale/`. A completed stage writes
`.stage_complete.json` into its directory, a completed round writes
`.round_complete.json`.

Each invocation writes its expanded config to `<run-dir>/configs/` and passes
that snapshot to every stage, so later edits of the config file cannot leak
into a running or resumed pipeline. An invocation holds an exclusive lock on
`<run-dir>/.lock`; a second one on the same run directory exits with 2.

Stages run in their own process group. SIGINT, SIGTERM or SIGHUP stops the
running stage (SIGINT to its group, SIGKILL after 30 s) and exits with
128 + the signal number; the same command resumes the round.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from lib import ui  # noqa: E402

RUNS = THIS_DIR / "runs"
STAGE_MARKER = ".stage_complete.json"
ROUND_MARKER = ".round_complete.json"
LOCK_NAME = ".lock"
FEATURE_DIMS = (55, 12, 37)
# stage05: outputs written, but a cell failed or a requested cell was not
# trained.
STAGE05_CELLS_NOT_TRAINED = 3


class ConfigError(Exception):
    def __init__(self, errors: List[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


class Interrupted(KeyboardInterrupt):
    """SIGTERM or SIGHUP, handled like SIGINT."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = int(signum)


# ── config schema ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Spec:
    kind: str
    required: bool = False
    default: Any = None
    nullable: bool = False
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    exclusive_minimum: bool = False
    choices: Optional[Tuple[Any, ...]] = None
    pattern: Optional[str] = None
    item: Optional["Spec"] = None
    min_len: int = 0
    children: Optional[Mapping[str, "Spec"]] = None


def _int(**kw: Any) -> Spec:
    return Spec("int", **kw)


def _num(**kw: Any) -> Spec:
    return Spec("num", **kw)


def _bool(**kw: Any) -> Spec:
    return Spec("bool", **kw)


def _str(**kw: Any) -> Spec:
    return Spec("str", **kw)


def _list(item: Spec, **kw: Any) -> Spec:
    return Spec("list", item=item, **kw)


def _section(children: Mapping[str, Spec], **kw: Any) -> Spec:
    return Spec("section", children=children, **kw)


_DTYPE = r"^[a-z0-9_]+$"
_PATH_NAME = r"^[A-Za-z0-9_.+:-]+$"
EXCLUDE_KEYS = (
    "min_m",
    "max_m",
    "min_n",
    "max_n",
    "min_k",
    "max_k",
    "min_batch",
    "max_batch",
    "min_mn",
    "max_mn",
)

SCHEMA = _section(
    {
        "config_id": _str(required=True, pattern=r"^[A-Za-z0-9_.-]+$"),
        "arch": _str(required=True, pattern=r"^gfx[0-9a-z]+(-[a-z]+)?(:[a-z0-9+-]+)*$"),
        "hipblaslt": _section(
            {
                "a_type": _str(required=True, pattern=_DTYPE),
                "b_type": _str(required=True, pattern=_DTYPE),
                "c_type": _str(required=True, pattern=_DTYPE),
                "d_type": _str(required=True, pattern=_DTYPE),
                "scale_type": _str(required=True, pattern=_DTYPE),
                "compute_type": _str(required=True, pattern=_DTYPE),
                "scaleA": _int(minimum=0),
                "scaleB": _int(minimum=0),
                "transA": _str(required=True, choices=("T", "N")),
                "transB": _str(required=True, choices=("T", "N")),
                "library_stem": _str(
                    required=True, pattern=r"^TensileLibrary_[A-Za-z0-9_.-]+$"
                ),
            },
            required=True,
        ),
        "hardware": _section(
            {
                "n_cu": _int(minimum=1),
                "lds_bytes": _int(minimum=1),
                "l2_bytes": _int(minimum=1),
            }
        ),
        "runtime": _section(
            {
                "devices": _list(_int(minimum=0), required=True, min_len=1),
                "rocm_libraries_root": _str(required=True),
                "build_dir": _str(),
            },
            required=True,
        ),
        "seed": _section(
            {
                "target_per_cell": _int(default=1600, minimum=1),
                "shapes_per_cell": _int(default=150, minimum=1),
                "seed": _int(default=42),
                "n_strata": _int(default=1, minimum=1),
                "exclude": _list(
                    _section({key: _int(minimum=0) for key in EXCLUDE_KEYS})
                ),
            }
        ),
        "probe": _section(
            {
                "first_line_iters": _int(minimum=1),
                "first_line_cold_iters": _int(minimum=0),
                "other_lines_iters": _int(minimum=1),
                "other_lines_cold_iters": _int(minimum=0),
                "duration_us": _num(minimum=0, exclusive_minimum=True),
                "skip_ratio_mode": _str(choices=("sigmoid", "bucket")),
                "skip_ratio_min": _num(minimum=0, maximum=1),
                "skip_ratio_max": _num(minimum=0, maximum=1),
                "skip_ratio_center_us": _num(minimum=0, exclusive_minimum=True),
                "skip_ratio_k": _num(minimum=0, exclusive_minimum=True),
                "fast_us_threshold": _num(minimum=0),
                "ratio_light": _num(minimum=0, maximum=1),
                "ratio_heavy": _num(minimum=0, maximum=1),
            }
        ),
        "bench": _section(
            {
                "blocks_per_gpu": _int(default=20, minimum=1),
                "stall_timeout_s": _num(default=300.0, minimum=0),
                "startup_grace_s": _num(default=900.0, minimum=0),
                "warmup_pass": _bool(),
                "skip_slow_solution_ratio": _num(minimum=0, maximum=1),
                "rotating_mb": _num(minimum=0),
                "rotating_target_blocks": _int(minimum=1),
                "device_memory_gib": _num(minimum=0, exclusive_minimum=True),
                "adaptive": _section(
                    {
                        "enabled": _bool(default=False),
                        "warmup_time": _num(minimum=0),
                        "sample_time": _num(minimum=0),
                        "measure_time": _num(minimum=0),
                        "max_measure_time": _num(minimum=0),
                        "min_iters": _int(minimum=0),
                        "max_iters": _int(minimum=0),
                        "noise_threshold": _num(minimum=0),
                        "stability_threshold": _num(minimum=0),
                        "stability_window": _int(minimum=0),
                        "stability_interval": _int(minimum=0),
                    }
                ),
            }
        ),
        "train": _section(
            {
                "workers": _int(default=1, minimum=1),
                "threads_per_worker": _int(default=0, minimum=0),
                "epochs": _int(default=200, minimum=1),
                "device": _str(default="cpu"),
                "eval_sample_frac": _num(
                    default=1.0, minimum=0, maximum=1, exclusive_minimum=True
                ),
                "smart_k": _int(default=10, minimum=1),
                "smart_k_max": _int(nullable=True, minimum=1),
                "smart_k_min": _int(default=4, minimum=1),
                "smart_k_saturation_eps": _num(default=0.005, minimum=0),
                "min_cell_gemms": _int(default=50, minimum=1),
                "validation_frac": _num(minimum=0, maximum=1),
            }
        ),
        "validate": _section(
            {
                "sel_eff_threshold": Spec(
                    "threshold", default="0.95", exclusive_minimum=True
                ),
                "split_floor": Spec("threshold", default="0.85"),
                "min_cell_gemms": _int(default=20, minimum=1),
                "split_min_improvement": _num(default=0.02),
                "split_after_attempts": _int(default=2, minimum=0),
                "max_split_depth": _int(default=6, minimum=0),
                "structural_split_min_octaves": _num(default=0.0, minimum=0),
                "structural_split_max_depth": _int(default=-1),
            }
        ),
        "full": _section({"total_rounds": _int(default=3, minimum=1)}),
        "deploy": _section(
            {
                "target_sel_eff": _num(
                    default=0.95,
                    nullable=True,
                    minimum=0,
                    maximum=1,
                    exclusive_minimum=True,
                ),
                "on_final_round": _bool(default=True),
                "stop_when_target_reached": _bool(default=False),
                "weight_dtype": _str(
                    default="bf16", choices=("fp32", "bf16", "int8", "int4")
                ),
                "rocm_libraries_to_deploy": _str(nullable=True),
                "build_dir": _str(),
                "targets": _list(Spec("target"), min_len=1),
                "apply_patch": _bool(default=False),
                "build_after_apply": _bool(default=False),
                "build_jobs": _int(nullable=True, minimum=1),
            }
        ),
        "stage07": _section(
            {
                "parity_sample_trained": _int(default=0, minimum=0),
                "bench_yamls": _list(_str(), default=[]),
                "timing_request_sizes": _list(_int(minimum=1), default=[1], min_len=1),
                "timing_repetitions": _int(default=3, minimum=1),
                "parity_min_match_rate": _num(default=1.0, minimum=0, maximum=1),
            }
        ),
        "stage08": _section(
            {
                "datasets": _list(
                    _section(
                        {
                            "name": _str(required=True, pattern=r"^[A-Za-z0-9_.-]+$"),
                            "path": _str(required=True),
                            "rocm_libraries_used": _str(),
                        }
                    ),
                    default=[],
                ),
                "tie_tolerance": _num(minimum=0),
                "bootstrap": _int(minimum=0),
            }
        ),
    }
)


def _type_name(v: Any) -> str:
    return "null" if v is None else type(v).__name__


def _check_value(spec: Spec, value: Any, where: str, errors: List[str]) -> None:
    if value is None:
        if not spec.nullable and spec.kind != "section":
            errors.append(f"{where}: must not be null")
        return
    kind = spec.kind
    if kind == "section":
        if not isinstance(value, dict):
            errors.append(f"{where}: expected a mapping, got {_type_name(value)}")
            return
        assert spec.children is not None
        for key in value:
            if key not in spec.children:
                errors.append(
                    f"{where}.{key}: unknown key" if where else f"{key}: unknown key"
                )
        for key, child in spec.children.items():
            path = f"{where}.{key}" if where else key
            if key in value:
                _check_value(child, value[key], path, errors)
            elif child.required:
                errors.append(f"{path}: required key is missing")
        return
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{where}: expected an integer, got {value!r}")
            return
    elif kind == "num":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"{where}: expected a number, got {value!r}")
            return
        if not math.isfinite(float(value)):
            errors.append(f"{where}: must be finite, got {value!r}")
            return
    elif kind == "bool":
        if not isinstance(value, bool):
            errors.append(f"{where}: expected true or false, got {value!r}")
        return
    elif kind == "str":
        if not isinstance(value, str) or not value:
            errors.append(f"{where}: expected a non-empty string, got {value!r}")
            return
        if spec.pattern and not re.match(spec.pattern, value):
            errors.append(f"{where}: {value!r} does not match {spec.pattern}")
    elif kind == "threshold":
        if value == "origami":
            return
        span = "(0, 1]" if spec.exclusive_minimum else "[0, 1] (0 disables)"
        try:
            v = float(value)
        except (TypeError, ValueError):
            errors.append(f"{where}: expected a number in {span} or 'origami'")
            return
        low_ok = v > 0.0 if spec.exclusive_minimum else v >= 0.0
        if isinstance(value, bool) or not (low_ok and v <= 1.0):
            errors.append(f"{where}: expected a number in {span} or 'origami'")
        return
    elif kind == "target":
        if isinstance(value, str):
            if not re.match(_PATH_NAME, value):
                errors.append(f"{where}: invalid target directory name {value!r}")
            return
        if isinstance(value, dict):
            unknown = sorted(set(value) - {"dir", "library_stem"})
            if unknown:
                errors.append(f"{where}: unknown keys {unknown}")
            d = value.get("dir")
            if not isinstance(d, str) or not re.match(_PATH_NAME, d):
                errors.append(f"{where}.dir: expected a directory name")
            stem = value.get("library_stem")
            if stem is not None and not (
                isinstance(stem, str) and stem.startswith("TensileLibrary_")
            ):
                errors.append(f"{where}.library_stem: expected a TensileLibrary_ stem")
            return
        errors.append(f"{where}: expected a directory name or {{dir, library_stem}}")
        return
    elif kind == "list":
        if not isinstance(value, list):
            errors.append(f"{where}: expected a list, got {_type_name(value)}")
            return
        if len(value) < spec.min_len:
            errors.append(f"{where}: needs at least {spec.min_len} entries")
        assert spec.item is not None
        for i, item in enumerate(value):
            _check_value(spec.item, item, f"{where}[{i}]", errors)
        return
    if spec.choices is not None and value not in spec.choices:
        errors.append(f"{where}: {value!r} is not one of {list(spec.choices)}")
    if spec.minimum is not None and isinstance(value, (int, float)):
        if value < spec.minimum or (spec.exclusive_minimum and value == spec.minimum):
            op = ">" if spec.exclusive_minimum else ">="
            errors.append(f"{where}: must be {op} {spec.minimum:g}, got {value!r}")
    if spec.maximum is not None and isinstance(value, (int, float)):
        if value > spec.maximum:
            errors.append(f"{where}: must be <= {spec.maximum:g}, got {value!r}")


def _spec_at(dotted: str) -> Spec:
    spec = SCHEMA
    for part in dotted.split("."):
        assert spec.children is not None and part in spec.children, dotted
        spec = spec.children[part]
    return spec


def cfg_get(cfg: Mapping[str, Any], dotted: str) -> Any:
    """Config value at `dotted`, or the schema default when absent."""
    spec = _spec_at(dotted)
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return spec.default
        cur = cur[part]
    return cur


def arch_base(arch: str) -> str:
    return str(arch).split(":", 1)[0]


def compiler_target(arch: str) -> str:
    """Name a Tensile library file carries for `arch`: ASIC revisions build
    under their base target's name (gfx1250v0 libraries end in _gfx1250)."""
    base = arch_base(arch)
    m = re.match(r"^(gfx[0-9a-f]+?)v\d+$", base)
    return m.group(1) if m else base


def deploy_targets(cfg: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(weights subdirectory name, index stem) for every deploy target."""
    arch = str(cfg["arch"])
    stem = str(cfg["hipblaslt"]["library_stem"])
    own_suffix = "_" + compiler_target(arch)
    out: List[Tuple[str, str]] = []
    for entry in cfg_get(cfg, "deploy.targets") or [arch_base(arch)]:
        if isinstance(entry, dict):
            d = str(entry["dir"])
            explicit = entry.get("library_stem")
        else:
            d, explicit = str(entry), None
        if explicit:
            out.append((d, str(explicit)))
            continue
        want = compiler_target(d)
        if want == compiler_target(arch):
            out.append((d, stem))
        elif stem.endswith(own_suffix):
            out.append((d, stem[: -len(own_suffix)] + "_" + want))
        else:
            raise ConfigError(
                [
                    f"deploy.targets: cannot derive the library stem for {d!r} "
                    f"from {stem!r}; give {{dir: {d}, library_stem: ...}}"
                ]
            )
    return out


def _split_rounds_possible(cfg: Mapping[str, Any]) -> bool:
    return (
        cfg_get(cfg, "validate.max_split_depth") > 0
        and cfg_get(cfg, "full.total_rounds") >= 3
    )


def validate_config(cfg: Any) -> Tuple[List[str], List[str]]:
    """(errors, warnings) for a loaded, env-expanded config."""
    errors: List[str] = []
    warnings: List[str] = []
    _check_value(SCHEMA, cfg, "", errors)
    if errors:
        return errors, warnings

    devices = cfg["runtime"]["devices"]
    if len(set(devices)) != len(devices):
        errors.append("runtime.devices: duplicate device indices")

    arch = cfg["arch"]
    stem = cfg["hipblaslt"]["library_stem"]
    if not stem.endswith("_" + compiler_target(arch)):
        warnings.append(
            f"hipblaslt.library_stem does not end in _{compiler_target(arch)}; "
            f"Tensile names {arch} libraries with that suffix"
        )
    try:
        targets = deploy_targets(cfg)
    except ConfigError as e:
        errors.extend(e.errors)
        targets = []
    dirs = [d for d, _ in targets]
    if len(set(dirs)) != len(dirs):
        errors.append("deploy.targets: duplicate target directories")

    adaptive = bool(cfg_get(cfg, "bench.adaptive.enabled"))
    if not adaptive:
        probe = cfg.get("probe") or {}
        need = [
            "first_line_iters",
            "first_line_cold_iters",
            "other_lines_iters",
            "other_lines_cold_iters",
            "duration_us",
            "skip_ratio_mode",
        ]
        if probe.get("skip_ratio_mode") == "bucket":
            need += ["fast_us_threshold", "ratio_light", "ratio_heavy"]
        for key in need:
            if key not in probe:
                errors.append(
                    f"probe.{key}: required when bench.adaptive is disabled "
                    f"(stage02 runs)"
                )
        if probe.get("skip_ratio_mode") == "sigmoid":
            for key in ("fast_us_threshold", "ratio_light", "ratio_heavy"):
                if key in probe:
                    warnings.append(f"probe.{key}: only read in bucket mode")

    target = cfg_get(cfg, "seed.target_per_cell")
    per_leaf = max(
        cfg_get(cfg, "seed.shapes_per_cell"), cfg_get(cfg, "train.min_cell_gemms")
    )
    train_min = cfg_get(cfg, "train.min_cell_gemms")
    val_min = cfg_get(cfg, "validate.min_cell_gemms")
    if train_min > target:
        errors.append(
            f"train.min_cell_gemms ({train_min}) > seed.target_per_cell "
            f"({target}): round_0 would train no cell"
        )
    if val_min > per_leaf:
        errors.append(
            f"validate.min_cell_gemms ({val_min}) exceeds the new GEMMs a leaf "
            f"gets per round ({per_leaf} = max(seed.shapes_per_cell, "
            f"train.min_cell_gemms)): no cell could be evaluated"
        )
    if _split_rounds_possible(cfg):
        if per_leaf < 2 * val_min:
            errors.append(
                f"split budget infeasible: a split needs validate.min_cell_gemms "
                f"({val_min}) new GEMMs on each side, but a leaf gets {per_leaf} "
                f"per round; raise seed.shapes_per_cell to >= {2 * val_min}, "
                f"lower validate.min_cell_gemms, or set validate.max_split_depth: 0"
            )
        last_split_round = cfg_get(cfg, "full.total_rounds") - 2
        if cfg_get(cfg, "validate.split_after_attempts") > last_split_round - 1:
            warnings.append(
                "validate.split_after_attempts leaves no round in which a split "
                "can happen (splits need that many prior retrains and are "
                "disabled in the final round)"
            )
    elif cfg_get(cfg, "validate.max_split_depth") > 0:
        warnings.append(
            "validate.max_split_depth > 0 but full.total_rounds < 3: the final "
            "round never splits, so no split can happen"
        )
    smax = cfg_get(cfg, "train.smart_k_max")
    if smax is not None and cfg_get(cfg, "train.smart_k_min") > smax:
        errors.append("train.smart_k_min > train.smart_k_max")

    deploy_root = cfg_get(cfg, "deploy.rocm_libraries_to_deploy")
    apply_patch = cfg_get(cfg, "deploy.apply_patch")
    build_after = cfg_get(cfg, "deploy.build_after_apply")
    if apply_patch and not deploy_root:
        errors.append("deploy.apply_patch requires deploy.rocm_libraries_to_deploy")
    if build_after and not apply_patch:
        warnings.append("deploy.build_after_apply is ignored without apply_patch")
    stage07_on = bool(cfg_get(cfg, "stage07.bench_yamls")) or (
        cfg_get(cfg, "stage07.parity_sample_trained") > 0
    )
    if stage07_on and not (apply_patch and build_after):
        warnings.append(
            "stage07 validates the model this run deploys and the bench loads "
            "it from the build tree: without deploy.apply_patch and "
            "deploy.build_after_apply it can only run after a manual deploy "
            "and model build (hipblaslt-tilewright-models)"
        )
    return errors, warnings


# ── config loading ──────────────────────────────────────────────────────────


_UNEXPANDED_VAR_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")


def expand_config_env(node: Any, missing: set) -> Any:
    """Expand ${VAR}, $VAR and a leading ~ in every string; record names
    that stay unexpanded in `missing`."""
    if isinstance(node, dict):
        return {k: expand_config_env(v, missing) for k, v in node.items()}
    if isinstance(node, list):
        return [expand_config_env(v, missing) for v in node]
    if isinstance(node, str) and ("$" in node or node.startswith("~")):
        names = set(_UNEXPANDED_VAR_RE.findall(node))
        missing.update(n for n in names if os.environ.get(n, "") == "")
        out = os.path.expanduser(os.path.expandvars(node))
        if "${" in out and not missing:
            missing.add(f"(unsupported syntax in {node!r})")
        return out
    return node


def load_config(path: Path) -> Tuple[Dict[str, Any], str]:
    """(expanded config, raw text). Raises ConfigError."""
    if not path.is_file():
        raise ConfigError([f"config not found: {path}"])
    raw = path.read_text()
    try:
        cfg = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError([f"{path}: invalid yaml: {e}"])
    if not isinstance(cfg, dict):
        raise ConfigError([f"{path}: top level must be a mapping"])
    missing: set = set()
    cfg = expand_config_env(cfg, missing)
    if missing:
        raise ConfigError(
            [
                f"{path} references unset or empty environment variable(s): "
                f"{', '.join(sorted(missing))}"
            ]
        )
    return cfg, raw


class _FlowListDumper(yaml.SafeDumper):
    pass


def _scalar_list_repr(dumper: yaml.SafeDumper, data: list) -> Any:
    flow = all(isinstance(x, (int, float, bool, str)) for x in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flow)


_FlowListDumper.add_representer(list, _scalar_list_repr)


def write_snapshot(run_dir: Path, cfg: Mapping[str, Any], raw: str, mode: str) -> Path:
    """Write `<run_dir>/configs/<ts>_<mode>.yaml` (expanded) next to the raw
    source text; never reuses an existing name."""
    out_dir = run_dir / "configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(1000):
        name = f"{stamp}_{mode}" + (f"_{n}" if n else "")
        snap = out_dir / f"{name}.yaml"
        try:
            fd = os.open(snap, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w") as f:
            yaml.dump(dict(cfg), f, Dumper=_FlowListDumper, sort_keys=False)
        (out_dir / f"{name}.source.yaml").write_text(raw)
        return snap
    raise RuntimeError(f"could not allocate a config snapshot name in {out_dir}")


# ── paths and preflight ─────────────────────────────────────────────────────


@dataclass
class Paths:
    bench_root: Path
    build_dir: Path
    bench_binary: Path
    library_dir: Path
    deploy_root: Optional[Path]
    deploy_build_dir: Optional[Path]

    @staticmethod
    def library_file(library_dir: Path, stem: str) -> Optional[Path]:
        for suffix in (".dat", ".dat.zlib", ".yaml"):
            p = library_dir / f"{stem}{suffix}"
            if p.is_file():
                return p
        return None


def _hipblaslt_build(root: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    return root / "projects" / "hipblaslt" / "build" / "release"


def derive_paths(cfg: Mapping[str, Any]) -> Paths:
    root = Path(cfg["runtime"]["rocm_libraries_root"])
    build = _hipblaslt_build(root, cfg_get(cfg, "runtime.build_dir"))
    deploy_root_s = cfg_get(cfg, "deploy.rocm_libraries_to_deploy")
    deploy_root = Path(deploy_root_s) if deploy_root_s else None
    deploy_build = (
        _hipblaslt_build(deploy_root, cfg_get(cfg, "deploy.build_dir"))
        if deploy_root
        else None
    )
    return Paths(
        bench_root=root,
        build_dir=build,
        bench_binary=build / "clients" / "hipblaslt-bench",
        library_dir=build / "Tensile" / "library" / arch_base(cfg["arch"]),
        deploy_root=deploy_root,
        deploy_build_dir=deploy_build,
    )


def _library_errors(library_dir: Path, stem: str, what: str) -> List[str]:
    if not library_dir.is_dir():
        return [
            f"{what} library directory is missing: {library_dir}. Build "
            f"hipBLASLt for this arch first (a gfx1250 build also produces "
            f"Tensile/library/gfx1250v0)."
        ]
    if Paths.library_file(library_dir, stem) is None:
        near = sorted(
            p.name
            for p in library_dir.glob("TensileLibrary_*Contraction*")
            if p.name.split("_")[-1].split(".")[0] == stem.split("_")[-1]
        )[:8]
        hint = ("; libraries there include: " + ", ".join(near)) if near else ""
        return [
            f"hipblaslt.library_stem {stem!r} has no .dat/.dat.zlib in "
            f"{library_dir}{hint}"
        ]
    return []


def _executable_errors(path: Path, what: str) -> List[str]:
    if not path.is_file():
        return [f"{what} not found: {path}"]
    if not os.access(path, os.X_OK):
        return [f"{what} is not executable: {path}"]
    return []


def _tilewright_errors() -> List[str]:
    try:
        from lib import evaluate

        tilewright = evaluate.tilewright_module()
    except Exception as e:
        return [
            f"the tilewright Python module does not import ({e!r}); install it "
            f"with `pip install shared/tilewright/python`"
        ]
    try:
        from lib import features
    except Exception as e:
        return [f"lib.features does not import: {e!r}"]
    have = getattr(tilewright, "feature_catalog_hash", None)
    if callable(have) and have() != features.feature_names_hash():
        return [
            f"feature catalog hash mismatch: tilewright module "
            f"{have()} vs lib/features.py {features.feature_names_hash()}"
        ]
    return []


def preflight(cfg: Mapping[str, Any], paths: Paths, mode: str) -> List[str]:
    stem = cfg["hipblaslt"]["library_stem"]
    errs: List[str] = []
    if mode != "validate":
        errs += _executable_errors(paths.bench_binary, "hipblaslt-bench")
        errs += _library_errors(paths.library_dir, stem, "bench")
        if cfg_get(cfg, "deploy.apply_patch"):
            root = paths.deploy_root
            if root is None or not (root / "shared" / "tilewright").is_dir():
                errs.append(
                    f"deploy.rocm_libraries_to_deploy is not a rocm-libraries "
                    f"checkout with shared/tilewright: {root}"
                )
            if cfg_get(cfg, "deploy.build_after_apply"):
                bd = paths.deploy_build_dir
                if bd is None or not (bd / "CMakeCache.txt").is_file():
                    errs.append(f"deploy build directory is not configured: {bd}")
    else:
        stage07_on = bool(cfg_get(cfg, "stage07.bench_yamls")) or (
            cfg_get(cfg, "stage07.parity_sample_trained") > 0
        )
        if stage07_on and paths.deploy_build_dir is not None:
            errs += _executable_errors(
                paths.deploy_build_dir / "clients" / "hipblaslt-bench",
                "deploy hipblaslt-bench",
            )
            errs += _library_errors(
                paths.deploy_build_dir / "Tensile" / "library" / arch_base(cfg["arch"]),
                stem,
                "deploy",
            )
        if cfg_get(cfg, "stage08.datasets"):
            errs += _library_errors(paths.library_dir, stem, "bench")
    errs += _tilewright_errors()
    return errs


# ── subprocess + logging ────────────────────────────────────────────────────


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_ORIG_STDOUT: Optional[Any] = None


class _Tee:
    """Writes to the terminal and, without ANSI escapes, to a log file."""

    def __init__(self, terminal: Any, logfile: Any) -> None:
        self.terminal = terminal
        self.logfile = logfile

    def write(self, data: str) -> int:
        try:
            self.terminal.write(data)
        except (OSError, ValueError):
            pass
        try:
            self.logfile.write(_ANSI_RE.sub("", data))
        except ValueError:
            pass
        return len(data)

    def flush(self) -> None:
        try:
            self.terminal.flush()
        except (OSError, ValueError):
            pass
        try:
            self.logfile.flush()
        except ValueError:
            pass

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def _signal_stage(proc: subprocess.Popen, sig: int) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def stop_stage(proc: subprocess.Popen, grace_s: float = 30.0) -> None:
    """SIGINT the stage's process group, SIGKILL it after `grace_s`."""
    _signal_stage(proc, signal.SIGINT)
    try:
        proc.wait(timeout=grace_s)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        _signal_stage(proc, signal.SIGKILL)
        proc.wait()


def run_logged(
    cmd: List[str], log_path: Path, env: Optional[Dict[str, str]] = None
) -> int:
    """Run `cmd` in its own process group, copying its output to `log_path`
    and the terminal (not to orchestrator.log). Returns the exit code
    (negative for a signal). An interrupt stops the stage (`stop_stage`) and
    is re-raised."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ui.info("run", " ".join(str(c) for c in cmd))
    t0 = time.time()
    term = _ORIG_STDOUT if _ORIG_STDOUT is not None else sys.stdout
    with log_path.open("w", buffering=1, errors="replace") as logf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            errors="replace",
            env=env,
            start_new_session=True,
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                try:
                    term.write(line)
                    term.flush()
                except (OSError, ValueError):
                    pass
                logf.write(line)
            rc = proc.wait()
        except KeyboardInterrupt:

            def _drain() -> None:
                try:
                    for line in proc.stdout:
                        logf.write(line)
                except (OSError, ValueError):
                    pass

            drain = threading.Thread(target=_drain, daemon=True)
            drain.start()
            stop_stage(proc)
            drain.join(timeout=5)
            raise
    el = time.time() - t0
    if rc == 0:
        ui.ok("run", f"completed in {ui.fmt_dur(el)}")
    else:
        ui.err("run", f"FAILED (rc={rc}) after {ui.fmt_dur(el)}; see {log_path}")
    return rc


def exit_code(rc: int) -> int:
    """Shell-style status for a child's return code (signal N -> 128+N)."""
    return 128 - rc if rc < 0 else rc


class Timing:
    """Per-(round, stage) wall-clock ledger of one invocation."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.t0 = time.time()

    def add(
        self, round_idx: Optional[int], stage: str, rc: int, elapsed: float
    ) -> None:
        self.rows.append(
            {"round": round_idx, "stage": stage, "rc": rc, "elapsed_s": elapsed}
        )

    def total(self) -> float:
        return time.time() - self.t0

    def print_summary(self) -> None:
        if not self.rows:
            return
        ui.banner("timing summary", ui.C.HEAD)
        width = max(12, *(len(r["stage"]) for r in self.rows))
        current: Any = object()
        for r in self.rows:
            if r["round"] != current:
                current = r["round"]
                ui.info("timing", "validate" if current is None else f"round_{current}")
            tag = "ok " if r["rc"] == 0 else f"rc={r['rc']}"
            print(
                f"   {r['stage']:<{width}}  {ui.fmt_dur(r['elapsed_s']):>8}  [{tag}]",
                flush=True,
            )
        ui.ok("timing", f"wall-clock total: {ui.fmt_dur(self.total())}")

    def write(self, run_dir: Path, mode: str) -> None:
        out = run_dir / "timing"
        out.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out / f"{stamp}_{mode}.json"
        n = 1
        while path.exists():
            path = out / f"{stamp}_{mode}_{n}.json"
            n += 1
        path.write_text(
            json.dumps(
                {"mode": mode, "total_s": self.total(), "rows": self.rows}, indent=2
            )
            + "\n"
        )


# ── stage directories ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageDef:
    number: float
    key: str
    script: str
    title: str


STAGE_DEFS: Dict[str, StageDef] = {
    s.key: s
    for s in (
        StageDef(1.0, "stage01", "stage01_generate_OOB_shapes.py", "generate shapes"),
        StageDef(2.0, "stage02", "stage02_probe.py", "probe"),
        StageDef(3.0, "stage03", "stage03_load_balance_offline_tuning.py", "bench"),
        StageDef(4.0, "stage04", "stage04_convert_to_enriched_dataset.py", "enrich"),
        StageDef(
            4.5,
            "stage04b",
            "stage04b_iterate_active_learning.py",
            "validate prior model",
        ),
        StageDef(5.0, "stage05", "stage05_train.py", "train"),
        StageDef(6.0, "stage06", "stage06_deploy_weights.py", "deploy"),
        StageDef(
            7.0,
            "stage07",
            "stage07_optional_validate_selection_time.py",
            "selection time + parity",
        ),
        StageDef(
            8.0,
            "stage08",
            "stage08_optional_validate_selection_efficiency.py",
            "selection efficiency",
        ),
    )
}
TRAINING_STAGE_NUMBERS = (1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0)


def stage_complete(stage_dir: Path) -> bool:
    return (stage_dir / STAGE_MARKER).is_file()


def mark_stage_complete(
    stage_dir: Path, cmd: Optional[List[str]], note: str = ""
) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "cmd": [str(c) for c in cmd] if cmd else None,
    }
    if note:
        rec["note"] = note
    (stage_dir / STAGE_MARKER).write_text(json.dumps(rec, indent=2) + "\n")


def move_aside(owner_dir: Path, *paths: Path) -> None:
    """Move stale stage outputs of a previous attempt into
    `<owner_dir>/stale/<stamp>/`."""
    present = [p for p in paths if p.exists() or p.is_symlink()]
    if not present:
        return
    owner = owner_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    dest = owner_dir / "stale" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for p in present:
        if p.resolve().parent != owner:
            raise RuntimeError(f"refusing to move {p}: not inside {owner_dir}")
        shutil.move(str(p), str(dest / p.name))
    ui.warn("stale", f"moved previous attempt's {[p.name for p in present]} to {dest}")


# ── run-dir discovery ───────────────────────────────────────────────────────


def round_index(path: Path) -> Optional[int]:
    m = re.fullmatch(r"round_(\d+)", path.name)
    return int(m.group(1)) if m else None


def round_dirs(run_dir: Path) -> List[Path]:
    if not run_dir.is_dir():
        return []
    out = [p for p in run_dir.iterdir() if p.is_dir() and round_index(p) is not None]
    return sorted(out, key=lambda p: round_index(p) or 0)


def round_complete(rd: Path) -> bool:
    return (rd / ROUND_MARKER).is_file()


def models_ready(rd: Path) -> bool:
    return (rd / "stage05" / "models.pt").is_file() and stage_complete(rd / "stage05")


def complete_prefix(run_dir: Path) -> List[Path]:
    """round_0..round_{k-1}, the longest run of complete rounds from 0."""
    out: List[Path] = []
    by_idx = {round_index(p): p for p in round_dirs(run_dir)}
    k = 0
    while k in by_idx and round_complete(by_idx[k]):
        out.append(by_idx[k])
        k += 1
    return out


def latest_models_dir(rounds: Iterable[Path]) -> Optional[Path]:
    for rd in sorted(rounds, key=lambda p: round_index(p) or 0, reverse=True):
        if models_ready(rd):
            return rd / "stage05"
    return None


def latest_deploy_record(rounds: Iterable[Path]) -> Optional[Path]:
    for rd in sorted(rounds, key=lambda p: round_index(p) or 0, reverse=True):
        p = rd / "stage06" / "deploy_record.json"
        if p.is_file() and stage_complete(rd / "stage06"):
            return p
    return None


def _read_json(path: Path) -> Optional[Any]:
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def held_out_sel_eff(decisions_path: Path) -> float:
    """n_eval-weighted geomean of `sel_eff_new` in one decisions.json."""
    d = _read_json(decisions_path)
    log_sum = w_sum = 0.0
    for c in (d or {}).get("per_cell") or []:
        if not isinstance(c, dict):
            continue
        se, n = c.get("sel_eff_new"), c.get("n_eval") or 0
        if isinstance(se, (int, float)) and se > 0 and n > 0:
            log_sum += math.log(float(se)) * float(n)
            w_sum += float(n)
    return math.exp(log_sum / w_sum) if w_sum > 0 else float("nan")


def _split_parents(run_dir: Path) -> set:
    parents: set = set()
    for rd in round_dirs(run_dir):
        d = _read_json(rd / "stage04b" / "splits.json")
        splits = (d or {}).get("splits") or {}
        if isinstance(splits, dict):
            parents.update(str(k) for k in splits)
    return parents


def aggregate_held_out(run_dir: Path) -> Dict[str, Any]:
    """Most recent held-out measurement per leaf cell across every round's
    stage04b/decisions.json, aggregated as n_eval-weighted geomeans for the
    model and the Origami baseline. Split parents are dropped: the shipped
    model no longer contains them."""
    parents = _split_parents(run_dir)
    latest: Dict[str, Dict[str, Any]] = {}
    rounds_seen: List[str] = []

    def _f(x: Any) -> Optional[float]:
        return (
            float(x)
            if isinstance(x, (int, float)) and not isinstance(x, bool)
            else None
        )

    for rd in round_dirs(run_dir):
        d = _read_json(rd / "stage04b" / "decisions.json")
        if d is None:
            continue
        rounds_seen.append(rd.name)
        for c in d.get("per_cell") or []:
            if not isinstance(c, dict) or not c.get("cell"):
                continue
            se, n = _f(c.get("sel_eff_new")), int(c.get("n_eval") or 0)
            if se is None or se <= 0 or n <= 0:
                continue
            latest[str(c["cell"])] = {
                "round": rd.name,
                "sel_eff_new": se,
                "n_eval": n,
                "origami_sel_eff_new": _f(c.get("origami_sel_eff_new")),
                "origami_n_eval": int(c.get("origami_n_eval") or 0),
                "winner_us_geomean": _f(c.get("winner_us_geomean")),
                "pick_us_geomean": _f(c.get("pick_us_geomean")),
                "origami_pick_us_geomean": _f(c.get("origami_pick_us_geomean")),
            }
    for p in parents:
        latest.pop(p, None)

    def _geo(pairs: Iterable[Tuple[Optional[float], int]]) -> float:
        ls = ws = 0.0
        for se, n in pairs:
            if se is not None and se > 0 and n > 0:
                ls += math.log(se) * n
                ws += n
        return math.exp(ls / ws) if ws > 0 else float("nan")

    return {
        "held_out_sel_eff": _geo(
            (e["sel_eff_new"], e["n_eval"]) for e in latest.values()
        ),
        "origami_held_out_sel_eff": _geo(
            (e["origami_sel_eff_new"], e["origami_n_eval"]) for e in latest.values()
        ),
        "n_cells": len(latest),
        "n_cells_with_origami": sum(
            1
            for e in latest.values()
            if e["origami_sel_eff_new"] is not None and e["origami_n_eval"] > 0
        ),
        "n_rounds_aggregated": len(rounds_seen),
        "rounds": rounds_seen,
        "per_cell": latest,
    }


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


# ── pipeline ────────────────────────────────────────────────────────────────


@dataclass
class RoundStep:
    index: int
    mode: str
    priors: List[Path]
    from_stage: Optional[float]
    is_final: bool


@dataclass
class RoundResult:
    rc: int
    held_out_sel_eff: float = float("nan")
    deployed: bool = False
    target_reached: bool = False


class Pipeline:
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        snapshot: Path,
        run_dir: Path,
        paths: Paths,
        stages_dir: Path,
        timing: Timing,
        force_deploy: bool,
    ) -> None:
        self.cfg = cfg
        self.snapshot = snapshot
        self.run_dir = run_dir
        self.paths = paths
        self.stages_dir = stages_dir
        self.timing = timing
        self.force_deploy = force_deploy
        self.arch = str(cfg["arch"])
        self.stem = str(cfg["hipblaslt"]["library_stem"])
        self.devices = [int(d) for d in cfg["runtime"]["devices"]]
        self.weight_dtype = str(cfg_get(cfg, "deploy.weight_dtype"))

    def get(self, dotted: str) -> Any:
        return cfg_get(self.cfg, dotted)

    # ── stage plumbing ──

    def run_stage(
        self,
        owner_dir: Path,
        key: str,
        args: List[str],
        round_idx: Optional[int],
        out_dir: Optional[Path] = None,
    ) -> int:
        sdef = STAGE_DEFS[key]
        out_dir = out_dir or owner_dir / key
        log = owner_dir / f"{key}.log"
        move_aside(owner_dir, out_dir, log)
        script = self.stages_dir / sdef.script
        cmd = [sys.executable, "-u", str(script), *args]
        ui.banner(f"Stage {sdef.number:g} -- {sdef.title}", ui.C.BLUE)
        t0 = time.time()
        rc = run_logged(cmd, log)
        self.timing.add(round_idx, key, rc, time.time() - t0)
        if rc == 0:
            mark_stage_complete(out_dir, cmd)
        return rc

    def _require_complete(self, rd: Path, key: str) -> bool:
        if stage_complete(rd / key):
            ui.warn("stage", f"{key} skipped (already complete)")
            return True
        ui.err(
            "stage",
            f"cannot skip {key}: {rd / key} has no completed output; "
            f"resume from an earlier stage",
        )
        return False

    def _round_stage_keys(self, mode: str) -> List[str]:
        keys = ["stage01"]
        if not self.get("bench.adaptive.enabled"):
            keys.append("stage02")
        keys += ["stage03", "stage04"]
        if mode == "active":
            keys.append("stage04b")
        keys.append("stage05")
        return keys

    def first_incomplete_stage(self, rd: Path, mode: str) -> float:
        for key in self._round_stage_keys(mode):
            if not stage_complete(rd / key):
                return STAGE_DEFS[key].number
        return 6.0

    # ── argument builders ──

    def _common(self) -> List[str]:
        return ["--config-yaml", str(self.snapshot)]

    def args_stage01(self, rd: Path, step: RoundStep) -> List[str]:
        args = [
            "--mode",
            step.mode,
            "--out-dir",
            str(rd / "stage01"),
            *self._common(),
            "--seed",
            str(self.get("seed.seed")),
            "--round-index",
            str(step.index),
            "--n-strata",
            str(self.get("seed.n_strata")),
        ]
        if step.mode == "initial":
            args += ["--target-per-cell", str(self.get("seed.target_per_cell"))]
        else:
            for p in step.priors:
                args += ["--prior-round-dir", str(p)]
            args += [
                "--shapes-per-cell",
                str(self.get("seed.shapes_per_cell")),
                "--min-cell-gemms",
                str(self.get("train.min_cell_gemms")),
            ]
        return args

    def args_stage02(self, rd: Path) -> List[str]:
        return [
            "--in-dir",
            str(rd / "stage01"),
            "--out-dir",
            str(rd / "stage02"),
            *self._common(),
            "--bench-binary",
            str(self.paths.bench_binary),
            "--devices",
            ",".join(str(d) for d in self.devices),
            "--stall-timeout-s",
            str(self.get("bench.stall_timeout_s")),
            "--startup-grace-s",
            str(self.get("bench.startup_grace_s")),
        ]

    def args_stage03(self, rd: Path) -> List[str]:
        in_dir = rd / ("stage01" if self.get("bench.adaptive.enabled") else "stage02")
        return [
            "--in-dir",
            str(in_dir),
            "--out-dir",
            str(rd / "stage03"),
            *self._common(),
            "--bench-binary",
            str(self.paths.bench_binary),
            "--devices",
            ",".join(str(d) for d in self.devices),
            "--blocks-per-gpu",
            str(self.get("bench.blocks_per_gpu")),
            "--stall-timeout-s",
            str(self.get("bench.stall_timeout_s")),
            "--startup-grace-s",
            str(self.get("bench.startup_grace_s")),
        ]

    def args_stage04(self, rd: Path) -> List[str]:
        return [
            "--log-dir",
            str(rd / "stage03" / "logs"),
            "--out-dir",
            str(rd / "stage04"),
            "--dat-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            "--progress-every",
            "60",
        ]

    def _model_eval_args(self) -> List[str]:
        return [
            *self._common(),
            "--arch",
            self.arch,
            "--hardware-device",
            str(self.devices[0]),
            "--library-dir",
            str(self.paths.library_dir),
            "--library-stem",
            self.stem,
            "--weight-dtype",
            self.weight_dtype,
        ]

    def per_leaf_budget(self) -> int:
        return max(
            int(self.get("seed.shapes_per_cell")), int(self.get("train.min_cell_gemms"))
        )

    def args_stage04b(self, rd: Path, step: RoundStep, models_dir: Path) -> List[str]:
        args = [
            "--models-dir",
            str(models_dir),
            "--enriched-csv-dir",
            str(rd / "stage04"),
            "--out-dir",
            str(rd / "stage04b"),
            *self._model_eval_args(),
            "--sel-eff-threshold",
            str(self.get("validate.sel_eff_threshold")),
            "--split-floor",
            str(self.get("validate.split_floor")),
            "--split-min-improvement",
            str(self.get("validate.split_min_improvement")),
            "--split-after-attempts",
            str(self.get("validate.split_after_attempts")),
            "--max-split-depth",
            str(self.get("validate.max_split_depth")),
            "--structural-split-min-octaves",
            str(self.get("validate.structural_split_min_octaves")),
            "--structural-split-max-depth",
            str(self.get("validate.structural_split_max_depth")),
            "--round-idx",
            str(step.index),
            "--min-cell-gemms",
            str(self.get("validate.min_cell_gemms")),
            "--train-min-cell-gemms",
            str(self.get("train.min_cell_gemms")),
            "--per-leaf-budget",
            str(self.per_leaf_budget()),
        ]
        if step.is_final:
            args.append("--final-round")
        for p in step.priors:
            args += ["--prior-round-dir", str(p)]
            dec = p / "stage04b" / "decisions.json"
            if dec.is_file():
                args += ["--prior-decisions", str(dec)]
        if len(step.priors) >= 2:
            pre = latest_models_dir(step.priors[:-1])
            if pre is not None:
                args += ["--prior-models", str(pre / "models.pt")]
        return args

    def args_stage05(
        self, rd: Path, step: RoundStep, only_cells: Optional[str]
    ) -> List[str]:
        args: List[str] = []
        for p in step.priors:
            if (p / "stage04").is_dir():
                args += ["--enriched-csv-dir", str(p / "stage04")]
        args += ["--enriched-csv-dir", str(rd / "stage04")]
        for p in step.priors:
            args += ["--prior-round-dir", str(p)]
        args += [
            "--output-dir",
            str(rd / "stage05"),
            *self._model_eval_args(),
            "--device",
            str(self.get("train.device")),
            "--smart-k",
            str(self.get("train.smart_k")),
            "--epochs",
            str(self.get("train.epochs")),
            "--eval-sample-frac",
            str(self.get("train.eval_sample_frac")),
            "--min-cell-gemms",
            str(self.get("train.min_cell_gemms")),
            "--train-workers",
            str(self.get("train.workers")),
            "--train-threads-per-worker",
            str(self.get("train.threads_per_worker")),
            "--seed",
            str(self.get("seed.seed")),
        ]
        vfrac = self.get("train.validation_frac")
        if vfrac is not None:
            args += ["--validation-frac", str(vfrac)]
        smax = self.get("train.smart_k_max")
        if smax is not None:
            args += [
                "--smart-k-max",
                str(smax),
                "--smart-k-min",
                str(self.get("train.smart_k_min")),
                "--smart-k-saturation-eps",
                str(self.get("train.smart_k_saturation_eps")),
            ]
        if only_cells is not None:
            args += ["--current-round-dir", str(rd), "--only-cells", only_cells]
        return args

    def args_stage06(self, rd: Path) -> List[str]:
        args = [
            "--train-dir",
            str(rd / "stage05"),
            *self._common(),
            "--out-dir",
            str(rd / "stage06"),
            "--library-stem",
            self.stem,
            "--weight-dtype",
            self.weight_dtype,
        ]
        if self.paths.deploy_root is not None:
            args += ["--rocm-libraries-to-deploy", str(self.paths.deploy_root)]
            if self.get("deploy.apply_patch"):
                args.append("--apply-patch")
                if self.get("deploy.build_after_apply"):
                    args.append("--build-after-apply")
                    if self.paths.deploy_build_dir is not None:
                        args += ["--build-dir", str(self.paths.deploy_build_dir)]
                    jobs = self.get("deploy.build_jobs")
                    if jobs is not None:
                        args += ["--build-jobs", str(jobs)]
        return args

    # ── one training round ──

    def run_round(self, step: RoundStep) -> RoundResult:
        rd = self.run_dir / f"round_{step.index}"
        rd.mkdir(parents=True, exist_ok=True)
        (rd / ROUND_MARKER).unlink(missing_ok=True)
        from_stage = (
            step.from_stage
            if step.from_stage is not None
            else self.first_incomplete_stage(rd, step.mode)
        )
        ui.banner(
            f"round_{step.index}  mode={step.mode}  from_stage={from_stage:g}"
            + ("  (final round)" if step.is_final else ""),
            ui.C.HEAD,
        )
        if step.mode == "active":
            if not step.priors:
                ui.err("round", "an active round needs at least one prior round")
                return RoundResult(2)
            ui.info("round", f"priors: {[p.name for p in step.priors]}")
        t_round = time.time()
        result = self._run_round_stages(rd, step, from_stage)
        self.timing.add(step.index, "(round total)", result.rc, time.time() - t_round)
        if result.rc == 0:
            (rd / ROUND_MARKER).write_text(
                json.dumps(
                    _json_safe(
                        {
                            "round": step.index,
                            "mode": step.mode,
                            "deployed": result.deployed,
                            "held_out_sel_eff": result.held_out_sel_eff,
                            "finished": datetime.now().isoformat(timespec="seconds"),
                        }
                    ),
                    indent=2,
                )
                + "\n"
            )
            ui.ok("round", f"round_{step.index} complete")
        else:
            ui.err("round", f"round_{step.index} FAILED (rc={result.rc})")
        return result

    def _stage_or_skip(
        self,
        rd: Path,
        key: str,
        from_stage: float,
        args_fn: Callable[[], List[str]],
        round_idx: int,
    ) -> int:
        if STAGE_DEFS[key].number < from_stage:
            return 0 if self._require_complete(rd, key) else 2
        return self.run_stage(rd, key, args_fn(), round_idx)

    def _run_round_stages(
        self, rd: Path, step: RoundStep, from_stage: float
    ) -> RoundResult:
        idx = step.index
        rc = self._stage_or_skip(
            rd, "stage01", from_stage, lambda: self.args_stage01(rd, step), idx
        )
        if rc:
            return RoundResult(exit_code(rc))

        if self.get("bench.adaptive.enabled"):
            if from_stage <= 2.0:
                move_aside(rd, rd / "stage02", rd / "stage02.log")
        else:
            rc = self._stage_or_skip(
                rd, "stage02", from_stage, lambda: self.args_stage02(rd), idx
            )
            if rc:
                return RoundResult(exit_code(rc))

        for key, fn in (
            ("stage03", lambda: self.args_stage03(rd)),
            ("stage04", lambda: self.args_stage04(rd)),
        ):
            rc = self._stage_or_skip(rd, key, from_stage, fn, idx)
            if rc:
                return RoundResult(exit_code(rc))

        retrain: Optional[str] = None
        if step.mode == "active":
            models_dir = latest_models_dir(step.priors)
            if models_dir is None:
                ui.err("round", "no prior round has a trained models.pt")
                return RoundResult(2)
            rc = self._stage_or_skip(
                rd,
                "stage04b",
                from_stage,
                lambda: self.args_stage04b(rd, step, models_dir),
                idx,
            )
            if rc:
                return RoundResult(exit_code(rc))
            path = rd / "stage04b" / "retrain_cells.txt"
            retrain = path.read_text().strip() if path.is_file() else ""

        if from_stage > 5.0:
            if not self._require_complete(rd, "stage05"):
                return RoundResult(2)
        elif retrain == "":
            rc = self._carry_forward(rd, step)
            if rc:
                return RoundResult(rc)
        else:
            rc = self.run_stage(
                rd, "stage05", self.args_stage05(rd, step, retrain), idx
            )
            if rc == STAGE05_CELLS_NOT_TRAINED:
                self._report_untrained_cells(rd)
            if rc:
                return RoundResult(exit_code(rc))

        return self._maybe_deploy(rd, step, from_stage)

    def _report_untrained_cells(self, rd: Path) -> None:
        metrics_path = rd / "stage05" / "metrics.json"
        metrics = _read_json(metrics_path) or {}
        failed = list(metrics.get("cells_failed") or [])
        missing = [str(r.get("cell")) for r in metrics.get("cells_not_trained") or []]
        ui.err(
            "round",
            f"stage05 did not train every requested cell (failed: {failed or '-'}; "
            f"not trained: {missing or '-'}; see {metrics_path}). {rd.name} stops "
            f"before deployment: fix the cause and run the same command to rerun "
            f"stage 5, or add --from-stage 4.5 to redo the decisions.",
        )

    def _carry_forward(self, rd: Path, step: RoundStep) -> int:
        src = latest_models_dir(step.priors)
        if src is None:
            ui.err("carry", "every cell passed but no prior round has models.pt")
            return 2
        splits = _read_json(rd / "stage04b" / "splits.json") or {}
        if splits.get("splits"):
            ui.err(
                "carry",
                "stage04b split cells but listed none to retrain; refusing to "
                "carry the unsplit model forward",
            )
            return 2
        dst = rd / "stage05"
        move_aside(rd, dst, rd / "stage05.log")
        dst.mkdir(parents=True)
        for name in ("models.pt", "cells.json"):
            if (src / name).is_file():
                shutil.copy2(src / name, dst / name)
        (dst / "carried_forward.json").write_text(
            json.dumps({"source": str(src)}, indent=2) + "\n"
        )
        mark_stage_complete(dst, None, note=f"carried forward from {src}")
        ui.ok("carry", f"every cell passed; carried {src} forward to {dst}")
        return 0

    def _maybe_deploy(
        self, rd: Path, step: RoundStep, from_stage: float
    ) -> RoundResult:
        held = float("nan")
        if step.mode == "active":
            held = held_out_sel_eff(rd / "stage04b" / "decisions.json")
        target = self.get("deploy.target_sel_eff")
        reached = target is not None and not math.isnan(held) and held >= target
        if not math.isnan(held):
            ui.info(
                "deploy",
                f"held-out sel_eff {held * 100:.2f}% (target "
                f"{'none' if target is None else f'{target * 100:.2f}%'})",
            )
        if step.mode == "initial":
            should = self.force_deploy
            if not should:
                ui.info("deploy", "round_0 deploys only with --deploy")
        else:
            should = (
                self.force_deploy
                or reached
                or (step.is_final and self.get("deploy.on_final_round"))
            )
        if not should:
            return RoundResult(0, held_out_sel_eff=held, target_reached=reached)
        reason = (
            "forced"
            if self.force_deploy
            else "target reached" if reached else "final round"
        )
        ui.info("deploy", f"deploying ({reason})")
        rc = self.run_stage(rd, "stage06", self.args_stage06(rd), step.index)
        if rc:
            return RoundResult(
                exit_code(rc), held_out_sel_eff=held, target_reached=reached
            )
        return RoundResult(
            0, held_out_sel_eff=held, deployed=True, target_reached=reached
        )

    # ── validation (stages 7 and 8) ──

    def stage07_enabled(self) -> bool:
        return bool(self.get("stage07.bench_yamls")) or (
            self.get("stage07.parity_sample_trained") > 0
        )

    def stage08_enabled(self) -> bool:
        return bool(self.get("stage08.datasets"))

    def run_validate(
        self,
        *,
        models_round: Path,
        rounds: List[Path],
        from_stage: float,
        require_deploy: bool = True,
    ) -> int:
        vdir = self.run_dir / "validate"
        vdir.mkdir(parents=True, exist_ok=True)
        ui.banner(f"validate  models from {models_round.name}", ui.C.HEAD)
        rcs: List[int] = []
        if self.stage07_enabled() and from_stage <= 7.0:
            if require_deploy or latest_deploy_record(rounds) is not None:
                rcs.append(self._run_stage07(vdir, rounds))
            else:
                ui.warn("validate", "stage 7 skipped: this run has deployed no model")
        elif self.stage07_enabled():
            ui.warn("validate", "stage 7 skipped (--from-stage 8)")
        if self.stage08_enabled():
            rcs.append(self._run_stage08(vdir, models_round))
        failed = [rc for rc in rcs if rc != 0]
        for rc in failed:
            ui.err("validate", f"a validation stage failed (rc={rc})")
        return max((exit_code(rc) for rc in failed), default=0)

    def _stage07_bench_yamls(self, vdir: Path) -> List[Path]:
        n = int(self.get("stage07.parity_sample_trained"))
        if n <= 0:
            return [Path(p) for p in self.get("stage07.bench_yamls")]
        out = vdir / f"stage07_trained_{n}.yaml"
        got, total = sample_trained_shapes(
            self.run_dir, n, self.cfg, out, int(self.get("seed.seed"))
        )
        if got == 0:
            ui.warn("validate", "no trained shapes to sample for stage 7")
            return [Path(p) for p in self.get("stage07.bench_yamls")]
        ui.ok("validate", f"stage 7: {got} of {total} trained shapes -> {out.name}")
        return [out]

    def _run_stage07(self, vdir: Path, rounds: List[Path]) -> int:
        record = latest_deploy_record(rounds)
        if record is None:
            ui.err(
                "validate",
                "stage 7 needs a model deployed by this run (round_*/stage06/"
                "deploy_record.json); none found",
            )
            return 2
        if self.paths.deploy_build_dir is None:
            ui.err("validate", "stage 7 needs deploy.rocm_libraries_to_deploy")
            return 2
        yamls = self._stage07_bench_yamls(vdir)
        if not yamls:
            ui.err("validate", "stage 7 has no bench yaml to run")
            return 2
        return self.run_stage(
            vdir, "stage07", self.args_stage07(vdir, record, yamls), None
        )

    def args_stage07(self, vdir: Path, record: Path, yamls: List[Path]) -> List[str]:
        assert self.paths.deploy_build_dir is not None
        args = [
            *self._common(),
            "--build-dir",
            str(self.paths.deploy_build_dir),
            "--deploy-record",
            str(record),
            "--out-dir",
            str(vdir / "stage07"),
            "--device",
            str(self.devices[0]),
        ]
        for y in yamls:
            args += ["--bench-yaml", str(y)]
        return args

    def _run_stage08(self, vdir: Path, models_round: Path) -> int:
        return self.run_stage(
            vdir, "stage08", self.args_stage08(vdir, models_round), None
        )

    def args_stage08(self, vdir: Path, models_round: Path) -> List[str]:
        args = [
            "--train-dir",
            str(models_round / "stage05"),
            *self._model_eval_args(),
            "--out-dir",
            str(vdir / "stage08"),
            "--devices",
            ",".join(str(d) for d in self.devices),
            "--blocks-per-gpu",
            str(self.get("bench.blocks_per_gpu")),
            "--min-cell-gemms",
            str(self.get("validate.min_cell_gemms")),
            "--bench-binary",
            str(self.paths.bench_binary),
        ]
        for flag, key in (
            ("--tie-tolerance", "stage08.tie_tolerance"),
            ("--bootstrap", "stage08.bootstrap"),
        ):
            if self.get(key) is not None:
                args += [flag, str(self.get(key))]
        hbl = self.cfg["hipblaslt"]
        for flag, key in (
            ("--filter-a-type", "a_type"),
            ("--filter-b-type", "b_type"),
            ("--filter-c-type", "c_type"),
            ("--filter-d-type", "d_type"),
            ("--filter-compute-type", "compute_type"),
            ("--filter-transA", "transA"),
            ("--filter-transB", "transB"),
        ):
            args += [flag, str(hbl[key])]
        for ds in self.get("stage08.datasets"):
            args += ["--dataset", f"{ds['name']}={ds['path']}"]
            if ds.get("rocm_libraries_used"):
                build = self.dataset_build_dir(str(ds["rocm_libraries_used"]))
                args += ["--dataset-build-dir", f"{ds['name']}={build}"]
        return args

    def dataset_build_dir(self, checkout: str) -> Path:
        """hipBLASLt build directory of the checkout a stage08 dataset was
        benched with: the bench build (runtime.build_dir) for the bench root,
        the checkout's default build directory otherwise."""
        root = Path(checkout)
        if root.resolve() == self.paths.bench_root.resolve():
            return self.paths.build_dir
        return _hipblaslt_build(root, None)


def sample_trained_shapes(
    run_dir: Path, n: int, cfg: Mapping[str, Any], out_path: Path, seed: int
) -> Tuple[int, int]:
    """Write a bench yaml of up to `n` distinct (M, N, K, batch) problems that
    benched successfully in any round (problems present in the enriched
    stage04 CSVs; stage01 shapes when no CSV exists). Returns (written,
    available)."""
    from lib.bench_yaml import parse_shapes_yaml, write_bench_yaml
    from lib.shapes import Shape

    seen: Dict[Tuple[int, int, int, int], None] = {}
    for rd in round_dirs(run_dir):
        for csv_path in sorted((rd / "stage04").glob("*.csv")):
            with csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                cols = {c.lower(): c for c in (reader.fieldnames or [])}
                keys = [cols.get(k) for k in ("m", "n", "k")]
                bkey = cols.get("batch_count") or cols.get("batch")
                if None in keys:
                    continue
                for row in reader:
                    try:
                        t = (
                            int(row[keys[0]]),
                            int(row[keys[1]]),
                            int(row[keys[2]]),
                            int(row[bkey]) if bkey else 1,
                        )
                    except (TypeError, ValueError):
                        continue
                    seen.setdefault(t, None)
    if not seen:
        for rd in round_dirs(run_dir):
            y = rd / "stage01" / "shapes.yaml"
            if y.is_file():
                for t in parse_shapes_yaml(y):
                    seen.setdefault(tuple(int(v) for v in t), None)
    shapes = sorted(seen)
    random.Random(seed).shuffle(shapes)
    pick = shapes[: max(0, n)]
    if pick:
        write_bench_yaml(
            [Shape(m=s[0], n=s[1], k=s[2], batch=s[3]) for s in pick], cfg, out_path
        )
    return len(pick), len(shapes)


# ── planning ────────────────────────────────────────────────────────────────


def plan_training(
    mode: str,
    run_dir: Path,
    total_rounds: int,
    from_round: Optional[int],
    from_stage: Optional[float],
) -> List[RoundStep]:
    """Round steps for initial/active/full. Raises ConfigError on an
    inconsistent request."""
    existing = {round_index(p): p for p in round_dirs(run_dir)}
    done = complete_prefix(run_dir)
    k = len(done)

    def priors(i: int) -> List[Path]:
        return [run_dir / f"round_{j}" for j in range(i)]

    if mode == "initial":
        if from_round not in (None, 0):
            raise ConfigError(["--mode initial only runs round_0"])
        if 0 in existing and round_complete(existing[0]):
            later = [i for i in existing if i and i > 0]
            if from_stage is None or later:
                raise ConfigError(
                    [
                        "round_0 is already complete"
                        + (" and later rounds depend on it" if later else "")
                        + "; use --mode active/full, or --from-stage to redo "
                        "round_0 of a run that has no later rounds"
                    ]
                )
        return [RoundStep(0, "initial", [], from_stage, True)]

    if mode == "active":
        if k == 0:
            raise ConfigError(
                [
                    f"--mode active needs a complete round_0 in {run_dir}; run "
                    f"--mode initial first"
                ]
            )
        if from_round not in (None, k):
            raise ConfigError(
                [f"--mode active runs round_{k} (the next round); got --from-round"]
            )
        stray = sorted(i for i in existing if i is not None and i > k)
        if stray:
            raise ConfigError(
                [
                    f"round_{k} is incomplete but later rounds exist "
                    f"({['round_%d' % i for i in stray]}); resume with --mode full "
                    f"--from-round {k}"
                ]
            )
        return [RoundStep(k, "active", priors(k), from_stage, k >= total_rounds - 1)]

    if from_round is not None:
        if from_round >= total_rounds:
            raise ConfigError(
                [f"--from-round {from_round} >= full.total_rounds ({total_rounds})"]
            )
        if from_round > k:
            raise ConfigError(
                [f"--from-round {from_round}: round_{k} is not complete yet"]
            )
        start = from_round
        start_stage = from_stage if from_stage is not None else 1.0
    else:
        if k >= total_rounds:
            return []
        start = k
        start_stage = from_stage
    steps = []
    for i in range(start, total_rounds):
        steps.append(
            RoundStep(
                i,
                "initial" if i == 0 else "active",
                priors(i),
                start_stage if i == start else 1.0,
                i == total_rounds - 1,
            )
        )
    return steps


# ── report ──────────────────────────────────────────────────────────────────


def emit_report(run_dir: Path) -> None:
    """Best-effort HTML report; never changes the exit status."""
    script = THIS_DIR / "scripts" / "make_report.py"
    out = run_dir / "report.html"
    try:
        subprocess.run(
            [sys.executable, str(script), str(run_dir), str(out)],
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as e:
        ui.warn("report", f"report generation failed: {e!r}")
        return
    if out.is_file():
        ui.ok("report", f"{out}")
    else:
        ui.warn("report", "report generation produced no file")


# ── main ────────────────────────────────────────────────────────────────────


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="tilewright training pipeline orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", type=Path, required=True, help="pipeline config yaml")
    ap.add_argument(
        "--mode", choices=("initial", "active", "full", "validate"), required=True
    )
    ap.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="run directory (default: runs/<config stem>_<timestamp>)",
    )
    ap.add_argument(
        "--deploy",
        action="store_true",
        help="deploy after every round, round_0 included",
    )
    ap.add_argument("--from-round", type=int, default=None, metavar="R")
    ap.add_argument(
        "--from-stage",
        type=float,
        default=None,
        metavar="N",
        choices=(1.0, 2.0, 3.0, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0),
    )
    ap.add_argument(
        "--skip-preflight",
        action="store_true",
        help="skip the checks of the bench, library and tilewright module",
    )
    ap.add_argument(
        "--stages-dir",
        type=Path,
        default=THIS_DIR / "stages",
        help="directory holding the stage scripts",
    )
    ap.add_argument("--no-report", action="store_true", help="do not write report.html")
    args = ap.parse_args(argv)
    if args.mode == "validate":
        if args.from_stage is not None and args.from_stage not in (7.0, 8.0):
            ap.error("--mode validate takes --from-stage 7 or 8")
    elif args.from_stage is not None and args.from_stage not in TRAINING_STAGE_NUMBERS:
        ap.error(f"--mode {args.mode} takes --from-stage 1, 2, 3, 4, 4.5, 5 or 6")
    return args


def _print_problems(tag: str, errors: List[str], warnings: List[str]) -> None:
    for w in warnings:
        ui.warn(tag, w)
    for e in errors:
        ui.err(tag, e)


def lock_run_dir(run_dir: Path) -> Optional[int]:
    """Descriptor holding an exclusive lock on `<run_dir>/.lock`, or None
    when another process holds it."""
    fd = os.open(run_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def _raise_interrupted(signum: int, _frame: Any) -> None:
    raise Interrupted(signum)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        cfg, raw = load_config(args.config)
    except ConfigError as e:
        _print_problems("cfg", e.errors, [])
        return 2
    errors, warnings = validate_config(cfg)
    _print_problems("cfg", errors, warnings)
    if errors:
        return 2

    run_dir = (
        args.run_dir
        or RUNS / f"{args.config.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ).resolve()
    if args.mode in ("active", "validate") and not run_dir.is_dir():
        ui.err("cfg", f"--mode {args.mode} needs an existing --run-dir: {run_dir}")
        return 2
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = lock_run_dir(run_dir)
    if lock is None:
        ui.err(
            "cfg",
            f"another orchestrator is using {run_dir} (it holds "
            f"{run_dir / LOCK_NAME}); wait for it to finish or pick another "
            f"--run-dir",
        )
        return 2
    previous: Dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        previous = {
            s: signal.signal(s, _raise_interrupted)
            for s in (signal.SIGTERM, signal.SIGHUP)
        }
    try:
        return _main_locked(args, cfg, raw, run_dir)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
        os.close(lock)


def _main_locked(
    args: argparse.Namespace, cfg: Dict[str, Any], raw: str, run_dir: Path
) -> int:
    global _ORIG_STDOUT
    log_fh = (run_dir / "orchestrator.log").open("a", buffering=1)
    log_fh.write(
        f"\n{'=' * 80}\n# {datetime.now().isoformat(timespec='seconds')}  "
        f"mode={args.mode}  config={args.config}\n# argv={sys.argv}\n{'=' * 80}\n"
    )
    _ORIG_STDOUT = sys.stdout
    orig_err = sys.stderr
    sys.stdout = _Tee(sys.stdout, log_fh)
    sys.stderr = _Tee(sys.stderr, log_fh)
    timing = Timing()
    try:
        return _main_logged(args, cfg, raw, run_dir, timing)
    except KeyboardInterrupt as e:
        signum = getattr(e, "signum", signal.SIGINT)
        ui.err(
            "interrupt",
            f"{signal.Signals(signum).name}: stopped; run the same command to "
            f"resume the interrupted round",
        )
        return 128 + signum
    finally:
        timing.print_summary()
        try:
            timing.write(run_dir, args.mode)
        except OSError as e:
            ui.warn("timing", f"could not write timing json: {e!r}")
        if not args.no_report:
            emit_report(run_dir)
        sys.stdout.flush()
        sys.stdout = _ORIG_STDOUT
        sys.stderr = orig_err
        _ORIG_STDOUT = None
        log_fh.close()


def _main_logged(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    raw: str,
    run_dir: Path,
    timing: Timing,
) -> int:
    ui.banner(f"tilewright pipeline  config={args.config.name}  mode={args.mode}")
    snapshot = write_snapshot(run_dir, cfg, raw, args.mode)
    paths = derive_paths(cfg)
    ui.info("cfg", f"config_id    : {cfg['config_id']}")
    ui.info("cfg", f"run_dir      : {run_dir}")
    ui.info("cfg", f"snapshot     : {snapshot}")
    ui.info("cfg", f"devices      : {cfg['runtime']['devices']}")
    ui.info("cfg", f"bench        : {paths.bench_binary}")
    ui.info("cfg", f"library      : {paths.library_dir}")
    ui.info("cfg", f"library stem : {cfg['hipblaslt']['library_stem']}")
    if not args.skip_preflight:
        errs = preflight(cfg, paths, args.mode)
        if errs:
            _print_problems("preflight", errs, [])
            return 2
        ui.ok("preflight", "bench, library and tilewright module found")

    pipeline = Pipeline(
        cfg=cfg,
        snapshot=snapshot,
        run_dir=run_dir,
        paths=paths,
        stages_dir=args.stages_dir.resolve(),
        timing=timing,
        force_deploy=args.deploy,
    )

    if args.mode == "validate":
        rounds = complete_prefix(run_dir)
        if not rounds:
            ui.err("validate", f"no complete round in {run_dir}")
            return 2
        if args.from_round is not None:
            pick = [r for r in rounds if round_index(r) == args.from_round]
            if not pick:
                ui.err("validate", f"round_{args.from_round} is not a complete round")
                return 2
            models_round = pick[0]
        else:
            models_round = rounds[-1]
        if not (pipeline.stage07_enabled() or pipeline.stage08_enabled()):
            ui.err(
                "validate",
                "nothing to validate: set stage07.parity_sample_trained, "
                "stage07.bench_yamls or stage08.datasets",
            )
            return 2
        considered = [
            r
            for r in rounds
            if (round_index(r) or 0) <= (round_index(models_round) or 0)
        ]
        return pipeline.run_validate(
            models_round=models_round,
            rounds=considered,
            from_stage=args.from_stage or 7.0,
        )

    try:
        steps = plan_training(
            args.mode,
            run_dir,
            int(pipeline.get("full.total_rounds")),
            args.from_round,
            args.from_stage,
        )
    except ConfigError as e:
        _print_problems("plan", e.errors, [])
        return 2
    if not steps:
        ui.ok("plan", "every planned round is already complete")
    deployed_any = False
    for step in steps:
        result = pipeline.run_round(step)
        if result.rc:
            return result.rc
        deployed_any = deployed_any or result.deployed
        if (
            result.target_reached
            and pipeline.get("deploy.stop_when_target_reached")
            and not step.is_final
        ):
            ui.ok("round", "deploy target reached; stopping early")
            break

    aggregated = aggregate_held_out(run_dir)
    (run_dir / "held_out_sel_eff_breakdown.json").write_text(
        json.dumps(_json_safe(aggregated), indent=2) + "\n"
    )
    if not math.isnan(aggregated["held_out_sel_eff"]):
        ui.ok(
            "done",
            f"held-out sel_eff (model, latest per leaf): "
            f"{aggregated['held_out_sel_eff'] * 100:.2f}% over "
            f"{aggregated['n_cells']} cells",
        )
        if not math.isnan(aggregated["origami_held_out_sel_eff"]):
            ui.ok(
                "done",
                f"held-out sel_eff (Origami)          : "
                f"{aggregated['origami_held_out_sel_eff'] * 100:.2f}% over "
                f"{aggregated['n_cells_with_origami']} cells",
            )

    rc = 0
    if steps and (pipeline.stage07_enabled() or pipeline.stage08_enabled()):
        rounds = complete_prefix(run_dir)
        if rounds:
            if (
                pipeline.stage07_enabled()
                and not deployed_any
                and latest_deploy_record(rounds) is not None
            ):
                ui.warn(
                    "validate",
                    "this invocation deployed nothing; stage 7 validates the "
                    "latest model this run deployed",
                )
            rc = pipeline.run_validate(
                models_round=rounds[-1],
                rounds=rounds,
                from_stage=7.0,
                require_deploy=False,
            )
    ui.banner(
        "done" if rc == 0 else "done (validation failed)",
        ui.C.GREEN if rc == 0 else ui.C.RED,
    )
    ui.ok("done", f"run_dir: {run_dir}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
