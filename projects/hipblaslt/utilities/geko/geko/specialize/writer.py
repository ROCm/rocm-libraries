# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``geko --specialize``: traced calls -> bespoke logic, ``calls.csv``, manifest, refusals.

Per call identity and run configuration: check the trace/build fingerprint,
resolve the fixed stock recipe (``recipe.py``), check the call's facts
(``facts.py``), then derive one bespoke single-solution logic with the target
TensileLite tree and check that its selection facts accept each call it serves.
Identical bespoke logic is written once and shared. A refused call is listed in ``refusals.json`` with its
reason and call count; the other calls are still generated and the exit status
is 1 whenever anything was refused. Usage errors exit 2 and generate nothing.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from geko.constants import SPECIALIZE_IDENTITY_FIELDS
from geko.specialize import facts as F
from geko.specialize.calls import UNKNOWN, CallIdentity, TraceError, configuration_name, load_calls
from geko.specialize.recipe import (
    ARCH_ISA, BaselineBuild, Recipe, Refused, UsageError, deriver, names, single_solution_logic,
    write_logic,
)

LOG_NAME = re.compile(r"^hipblaslt_\d+\.log$")
UNTESTED = {"gfx942": "untested-gfx942"}  # outputs for archs not validated on hardware
# A hand-written custom-kernel recipe cannot be specialized (bespoke custom kernels are not supported):
# the call is listed with this status and reason, never generated.
CUSTOM_STATUS = "custom-kernel-recipe-unsupported"
CUSTOM_REASON = "custom-kernel recipe unsupported"
# A model_run configuration (one with run.json) is specialization input only when it is a complete
# mask-128 analysis run: startup/greedy/performance, mask-16 API and unfinished runs are refused.
MODEL_RUN_REQUIRED = (("status", "complete"), ("mode", "analysis"))
# calls.csv columns: the call-manifest format (+ extra_outputs, api_path).
MANIFEST_COLUMNS = [
    "problem_id", "tier", "provenance", "source_path", "source_sha256", "source_lines",
    "model", "run", "derivation", "call_count", "transA", "transB", "M", "N", "K", "batch",
    "lda", "ldb", "ldc", "ldd", "strides", "a_type", "b_type", "c_type", "d_type",
    "compute_type", "alpha", "beta", "pointer_mode", "bias", "bias_type", "activation",
    "act_args", "sav", "recipe_mode", "stock_solution_index", "stock_kernel_name",
    "stock_logic_file", "stock_logic_index", "refusal_cases", "extra_outputs", "api_path",
]
# Identity fields whose calls.csv column has another spelling keep the logged text in <field>_raw.
RAW_COLUMNS = ("pointer_mode", "a_type", "b_type", "c_type", "d_type", "compute_type", "bias_type")
ABSENT = "unknown"  # calls.csv text of a field the record did not log
ASSERT_KEYS = F.ASSERT_KEYS
TCL_OPTIONS = ("--no-compress", "--disable-asm-comments", "--gfx1250v0")  # tcl_options.md (cmake build)
TCL_CXX = "/opt/rocm/bin/amdclang++"  # tcl_options.md --cxx-compiler
MAX_JOBS = 64  # CPU limit for one geko run
MANDATORY_KEYS = ("AssertSizeEqual", "BespokeNarrowing")  # never dropped, refused if not honoured
OUTPUT_FILES = ("calls.csv", "specialize_manifest.json", "refusals.json")
BUILD_RECORD = ".geko_specialize_build.json"  # what the last --build-library into OUT created


# --- inputs -------------------------------------------------------------------------------------

@dataclass
class Configuration:
    name: str
    logs: List[Path]
    run_json: Optional[Dict[str, Any]]
    route: Optional[Dict[str, Any]]
    fingerprint: str = "unverified"  # "verified" | "unverified" | "missing" | "mismatch"


def _configurations(workload: Path) -> List[Tuple[str, List[Path], Optional[Path]]]:
    """(name, process logs, run.json) per run configuration of --workload-log; the name is
    calls.configuration_name (run.json run_configuration, else the directory name)."""
    def logs_in(directory: Path) -> List[Path]:
        return sorted(p for p in directory.iterdir() if p.is_file() and LOG_NAME.match(p.name))

    if workload.is_file():
        run = workload.parent / "run.json"
        return [(configuration_name(workload.parent), [workload], run if run.is_file() else None)]
    if not workload.is_dir():
        raise UsageError(f"--workload-log {workload} does not exist")
    if logs_in(workload):
        run = workload / "run.json"
        return [(configuration_name(workload), logs_in(workload), run if run.is_file() else None)]
    out = []
    for sub in sorted(p for p in workload.iterdir() if p.is_dir()):
        run = sub / "run.json"
        logs = logs_in(sub)
        if logs or run.is_file():
            out.append((configuration_name(sub), logs, run if run.is_file() else None))
    if not out:
        raise UsageError(f"--workload-log {workload} holds no run-configuration directory with hipblaslt_<pid>.log")
    names = [name for name, _, _ in out]
    twice = sorted({name for name in names if names.count(name) > 1})
    if twice:
        raise UsageError(f"--workload-log {workload}: several directories are run configuration {twice[0]} "
                         "(run.json run_configuration, else the directory name); give each its own name")
    return out


def _route_documents(path: Path) -> Dict[str, Dict[str, Any]]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.json"))
    else:
        raise UsageError(f"--route-facts {path} does not exist")
    docs = {}
    for file in files:
        try:
            doc = json.loads(file.read_text())
        except json.JSONDecodeError as e:
            raise UsageError(f"--route-facts {file}: not JSON ({e})") from None
        if isinstance(doc, dict) and isinstance(doc.get("calls"), list):
            docs[str(doc.get("run_configuration") or file.stem)] = doc
    return docs


def _identity_key(identity: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((k, str(v)) for k, v in identity.items()))


def logged_identity(identity: CallIdentity) -> Dict[str, str]:
    """The identity as logged, UNKNOWN omitted (the identity a mask-128 trace consumer keys on)."""
    return {k: v for k, v in identity.items() if v is not UNKNOWN}


def _exclusions(path: Optional[str]) -> Dict[Tuple[Any, ...], Tuple[frozenset, List[Dict[str, Any]]]]:
    """(recipe key, identity key) -> (excluded facts, the entries naming them)."""
    if not path:
        return {}
    try:
        doc = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise UsageError(f"--exclude-controls {path}: {e}") from None
    out: Dict[Tuple[Any, ...], Tuple[frozenset, List[Dict[str, Any]]]] = {}
    for i, entry in enumerate(doc.get("exclusions") or []):
        try:
            recipe, ident, names_ = entry["recipe"], entry["identity"], list(entry["facts"])
            key = (str(recipe["source_file"]), str(recipe["sha256"]), int(recipe["in_file_index"]),
                   _identity_key(ident))
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise UsageError(f"--exclude-controls {path} entry {i}: needs recipe{{source_file, sha256, "
                             f"in_file_index}}, identity and facts ({e})") from None
        typed = {k: v for k, v in ident.items() if not isinstance(v, str)}
        if typed:
            raise UsageError(f"--exclude-controls {path} entry {i}: identity values must be strings exactly as "
                             f"logged; quote them in the YAML (e.g. beta: '0.000000'), got {typed}")
        if "AssertSizeEqual" in names_:
            raise UsageError(f"--exclude-controls {path} entry {i}: AssertSizeEqual is mandatory "
                             "and cannot be excluded")
        bad = [n for n in names_ if n not in F.EXPRESSIBLE]
        if bad:
            raise UsageError(f"--exclude-controls {path} entry {i}: {bad} are not expressible request facts "
                             f"{list(F.EXPRESSIBLE)}")
        facts_, entries = out.get(key, (frozenset(), []))
        out[key] = (facts_ | frozenset(names_), entries + [entry])
    return out


# --- bespoke logic -------------------------------------------------------------------------------

def _largest_divisor(size, registry):
    return max(r for r in registry if size % r == 0)


def _call_violations(sol, call):
    sizes = [call["M"], call["N"], call["batch"], call["K"]]
    lds = {"A": call["lda"], "B": call["ldb"], "C": call["ldc"], "D": call["ldd"]}
    checks = {"AssertSizeEqual": lambda s, v: s == v, "AssertSizeGreaterThan": lambda s, v: s > v,
              "AssertSizeLessThan": lambda s, v: s < v}
    out = []
    for key, size in zip(ASSERT_KEYS, (call["M"], call["N"], call["K"])):
        if size % sol[key]:
            out.append(f"{key}={sol[key]} (size {size})")
    for key, ok in checks.items():
        value = sol[key] if key in sol else -1
        for idx, v in (value.items() if isinstance(value, dict) else ()):
            if v != -1 and not (idx < len(sizes) and ok(sizes[idx], v)):
                out.append(f"{key}[{idx}]={v} (size {sizes[idx] if idx < len(sizes) else '?'})")
    for tc, ld in lds.items():
        key = f"AssertStride{tc}Equal"
        value = sol[key] if key in sol else -1
        for idx, v in (value.items() if isinstance(value, dict) else ()):
            actual = {0: 1, 1: ld}.get(idx)
            if v != -1 and actual != v:
                out.append(f"{key}[{idx}]={v} (stride {actual if actual is not None else 'unknown'})")
    for key in ("AssertAIGreaterThanEqual", "AssertAILessThanEqual"):
        if key in sol and sol[key] > 0:
            out.append(f"{key}={sol[key]} (AI bound present, not evaluated)")
    return out


def _derived_problem_type(sol):
    pt = sol["ProblemType"]
    return {"UseBeta": bool(pt["UseBeta"]), "ActivationType": pt["ActivationType"].value,
            "UseBias": int(pt["UseBias"]), "BiasDataTypeList": [t.value for t in pt["BiasDataTypeList"]],
            "UseScaleAlphaVec": int(pt["UseScaleAlphaVec"])}


def _selection_violations(bespoke: "Bespoke", call: Dict[str, Any]) -> List[str]:
    """Selection facts of a bespoke solution that reject one of its own calls.

    The host turns a bespoke solution's facts into selection predicates: exact sizes and
    assertions, beta zero (UseBeta: false), bias off, scale-alpha vector off, the bias type list,
    a single ActivationType, and, for UseBias != 0, a mandatory bias-presence check.
    A solution whose facts reject the call it was generated for can never serve that call."""
    got = _derived_problem_type(bespoke.solution)
    out = []
    if got["UseBias"] != 0 and call["bias_on"] is not True:
        logged = "not logged" if call["bias_on"] is UNKNOWN else "false"
        out.append(f"bias: UseBias={got['UseBias']} makes bias presence a selection fact of the bespoke "
                   f"solution, but the call's bias_vector is {logged}")
    if got["UseBias"] == 0 and call["bias_on"] is True:
        out.append("bias: UseBias=0 makes bias off a selection fact of the bespoke solution, but the call "
                   "has a bias vector")
    if (got["UseBias"] != 0 and call["bias_on"] is True and call["bias_type"] is not UNKNOWN
            and got["BiasDataTypeList"] and F.DTYPES[call["bias_type"]] not in got["BiasDataTypeList"]):
        out.append(f"bias type: BiasDataTypeList={got['BiasDataTypeList']} is a selection fact of the bespoke "
                   f"solution, but the call's bias_type is {call['bias_type']}")
    if not got["UseBeta"] and call["beta"] is not UNKNOWN and call["beta"] != 0.0:
        out.append(f"beta: UseBeta=false makes beta zero a selection fact of the bespoke solution, but the "
                   f"call's beta is {call['beta']}")
    if got["UseScaleAlphaVec"] == 0 and (call["sav"] == 1 or call["pointer_raw"] not in (0, UNKNOWN)):
        out.append("scale-alpha vector: UseScaleAlphaVec=0 is a selection fact of the bespoke solution, but the "
                   f"call uses one (scaleAlpha_vector {call['sav']}, pointer mode {call['pointer_raw']})")
    act, want = got["ActivationType"], call["activation"]
    if act not in ("all", "hipblaslt_all") and act != want and not (want is UNKNOWN and act == bespoke.stock_activation):
        out.append(f"activation: single ActivationType {act} is a selection fact of the bespoke solution, but the "
                   f"call's activation is {'not logged' if want is UNKNOWN else want}")
    out += [f"size: {v}" for v in _call_violations(bespoke.solution, call)]
    return out


def _derived_assertions(sol):
    out = {k: int(sol[k]) for k in ASSERT_KEYS}
    ase = sol["AssertSizeEqual"] if "AssertSizeEqual" in sol else -1
    out["AssertSizeEqual"] = ase if isinstance(ase, dict) and ase else None
    out["BespokeNarrowing"] = (sol["BespokeNarrowing"] if "BespokeNarrowing" in sol else False) is True
    return out


@dataclass
class Bespoke:
    logic: Dict[str, Any]
    solution_name: str
    kernel_name: str
    applied: List[Dict[str, Any]]
    not_applied: List[Dict[str, Any]]
    solution: Any  # the derived bespoke solution, whose facts the host selects on
    stock_activation: str  # ActivationType the stock solution itself derives


def build_bespoke(build: BaselineBuild, recipe: Recipe, call: Dict[str, Any], keep_beta: bool,
                  excluded: frozenset, scratch: Path) -> Bespoke:
    """The bespoke logic of one call on its fixed recipe."""
    from tensilelite.Common.ValidParameters import validParameters

    stock, _ = build.stock(recipe.path)
    stock_sol = stock["Solutions"][recipe.in_file_index]
    stock_pt = stock["ProblemType"]
    registry = {k: sorted(validParameters[k]) for k in ASSERT_KEYS}
    defaults = stock.get("DefaultSolution") or {}

    def stock_value(key):
        return stock_sol.get(key, defaults.get(key, 1))

    req_pt, not_applied = F.narrowing(call, stock_pt, keep_beta, excluded)
    if req_pt.get("UseBeta") is True:
        not_applied.append({"fact": "UseBeta", "requested": False, "kept": True,
                            "reason": "--keep-beta keeps UseBeta: true for the traced beta 0"})
    sizes = dict(zip(ASSERT_KEYS, (call["M"], call["N"], call["K"])))
    req_assert = {k: _largest_divisor(sizes[k], registry[k]) for k in ASSERT_KEYS}
    general = single_solution_logic(stock, stock_sol, (call["M"], call["N"], call["batch"], call["K"]))
    gsol = build.general_solution(recipe)
    gassert = _derived_assertions(gsol)
    violations = _call_violations(gsol, call)
    if violations:
        raise Refused("stock-rejects-call", "pre-filter: the stock solution's derived assertions reject the "
                      f"call: {violations}")
    for key in ASSERT_KEYS:
        req_assert[key] = max(req_assert[key], gassert[key])
        if key in excluded:
            req_assert[key] = gassert[key]
            not_applied.append({"fact": key, "reason": "excluded by --exclude-controls"})
    sol_changes = {k: req_assert[k] for k in ASSERT_KEYS if req_assert[k] != stock_value(k)}
    sol_changes["AssertSizeEqual"] = {0: call["M"], 1: call["N"], 2: call["batch"], 3: call["K"]}
    sol_changes["BespokeNarrowing"] = True

    def bespoke_logic(pt_changes):
        logic = copy.deepcopy(general)
        logic["ProblemType"].update(pt_changes)
        logic["Solutions"][0].update(sol_changes)
        return logic

    derive = deriver(build.arch)
    source = f"bespoke logic of {recipe.source_file}[{recipe.in_file_index}]"
    bsol = derive.derive_named(bespoke_logic(req_pt), scratch, "candidate.yaml", source)
    got = _derived_problem_type(bsol)
    if "ActivationType" in req_pt and call["activation"] != "none":
        if got["ActivationType"] != req_pt["ActivationType"]:
            not_applied.append({"fact": "ActivationType", "requested": req_pt["ActivationType"],
                                "kept": stock_pt["ActivationType"],
                                "reason": f"the target tree derives ActivationType={got['ActivationType']} "
                                          f"from ActivationType={req_pt['ActivationType']}"})
            del req_pt["ActivationType"], req_pt["Activation"]
    for key in ("UseBeta", "UseBias", "BiasDataTypeList", "UseScaleAlphaVec"):
        if key in req_pt and got[key] != req_pt[key]:
            not_applied.append({"fact": key, "requested": req_pt[key], "kept": stock_pt.get(key),
                                "reason": f"the target tree derives {key}={got[key]}"})
            del req_pt[key]
    candidate = _derived_assertions(bsol)
    for key in list(sol_changes):
        if candidate[key] != sol_changes[key]:
            if key in MANDATORY_KEYS:
                raise Refused("derivation", f"bespoke derivation: the target tree derives {key}={candidate[key]}, "
                              f"not the mandatory {sol_changes[key]}")
            not_applied.append({"fact": key, "requested": sol_changes[key], "kept": gassert[key],
                                "reason": f"the target tree derives {key}={candidate[key]} for this solution"})
            del sol_changes[key]
            if key in req_assert:
                req_assert[key] = gassert[key]

    bespoke = bespoke_logic(req_pt)
    bsol = derive.derive_named(bespoke, scratch, "bespoke.yaml", source)
    bassert = _derived_assertions(bsol)
    wrong = {k: (bassert[k], v) for k, v in {**req_assert, **sol_changes}.items() if bassert[k] != v}
    if wrong:
        raise Refused("bespoke-derivation", f"the target tree derives {wrong} (derived, written)")
    gpt, bpt = general["ProblemType"], bespoke["ProblemType"]
    gs, bs = general["Solutions"][0], bespoke["Solutions"][0]
    applied = [{"fact": k, "value": bpt[k]} for k in bpt if gpt.get(k) != bpt[k]]
    applied += [{"fact": k, "value": bs[k]} for k in bs if gs.get(k) != bs[k]]
    solution_name, kernel_name = names(bsol)
    return Bespoke(bespoke, solution_name, kernel_name, applied, not_applied, bsol,
                   _derived_problem_type(gsol)["ActivationType"])


# --- outputs ------------------------------------------------------------------------------------

@dataclass
class Unit:
    """One call identity in one run configuration."""

    cfg: Configuration
    identity: CallIdentity
    call_count: int
    observations: Dict[Tuple[Any, Any, Any], int]
    sources: Dict[Tuple[str, str], int]
    recipes: Dict[Tuple[Any, Any, Any], Any] = field(default_factory=dict)  # observation -> Recipe | Refused
    recipe: Optional[Recipe] = None
    route: Optional[Dict[str, Any]] = None
    refusal: Optional[Refused] = None
    status: str = "generated"
    bespoke: Optional[Bespoke] = None
    logic_file: str = ""
    not_applied: List[Dict[str, Any]] = field(default_factory=list)


def _text(identity: CallIdentity, field_: str) -> str:
    value = identity[field_]
    return ABSENT if value is UNKNOWN else str(value)


def _csv_rows(unit: Unit, arch: str, model: str) -> List[Dict[str, str]]:
    ident = unit.identity
    typed = ident.typed
    strides = [_text(ident, f"stride_{x}") for x in "abcd"]
    bias_on = typed("bias_vector")
    if bias_on is UNKNOWN:
        bias = ABSENT
    elif not bias_on:
        bias = "none"
    else:
        axis = unit.route.get("bias_axis") if unit.route and F.proven(unit.route, "bias_axis") else None
        bias = str(axis) if axis else ABSENT
    extra = [name for f_, name in (("use_e", "aux"), ("gradient", "gradient")) if typed(f_) is True]
    if typed("scaleC") not in (0, False, UNKNOWN) or typed("scaleD") not in (0, False, UNKNOWN):
        extra.append("scaled_output")
    if unit.route and unit.route.get("output_amax") is not False:
        extra.append("amax")
    route = str(unit.route.get("call_route")) if unit.route and unit.route.get("call_route") else ABSENT
    api_path = {F.ORDINARY_ROUTE: "gemm"}.get(route, "grouped" if "grouped" in route.lower()
                                               else "device_args" if "device" in route.lower() else route)
    act = _text(ident, "activation_type")
    pointer = typed("pointer_mode")
    base = {
        "tier": "A", "provenance": "traced",
        "source_path": ";".join(sorted(p for p, _ in unit.sources)),
        "source_sha256": ";".join(sha for _, sha in sorted(unit.sources)),
        "source_lines": "", "model": model, "run": unit.cfg.name, "derivation": "",
        "transA": _text(ident, "transA"), "transB": _text(ident, "transB"),
        "M": _text(ident, "M"), "N": _text(ident, "N"), "K": _text(ident, "K"), "batch": _text(ident, "batch_count"),
        "lda": _text(ident, "lda"), "ldb": _text(ident, "ldb"), "ldc": _text(ident, "ldc"), "ldd": _text(ident, "ldd"),
        "strides": "0" if all(s == "0" for s in strides) else "/".join(strides),
        "alpha": _text(ident, "alpha"), "beta": _text(ident, "beta"),
        "pointer_mode": ABSENT if pointer is UNKNOWN else F.POINTER_MODE_NAMES.get(pointer, ABSENT),
        "bias": bias,
        "bias_type": (F.manifest_type(_text(ident, "bias_type")) if typed("bias_type") is not UNKNOWN else ABSENT)
        if bias_on is True else "",
        "activation": act,
        "act_args": ";".join(_text(ident, f) for f in ("activation_arg0", "activation_arg1")) if act == "clamp" else "",
        "sav": ABSENT if typed("scaleAlpha_vector") is UNKNOWN else str(int(bool(typed("scaleAlpha_vector")))),
        "recipe_mode": "pinned-parent", "refusal_cases": "",
        "extra_outputs": ";".join(extra) or "none", "api_path": api_path,
        "run_configuration": unit.cfg.name, "arch": arch, "label": UNTESTED.get(arch, ""),
        "status": unit.status, "refusal": unit.refusal.reason if unit.refusal else "",
        "expected_kernel_name": unit.bespoke.kernel_name if unit.bespoke else "",
        "bespoke_logic_file": unit.logic_file,
        "trace_fingerprint": unit.cfg.fingerprint,
    }
    for k in ("a_type", "b_type", "c_type", "d_type", "compute_type"):
        base[k] = F.manifest_type(_text(ident, k)) if typed(k) is not UNKNOWN else ABSENT
    for k in SPECIALIZE_IDENTITY_FIELDS:
        base[k + "_raw" if k in RAW_COLUMNS else k] = _text(ident, k)
    rows = []
    for observation, count in sorted(unit.observations.items(), key=lambda o: tuple(map(str, o[0]))):
        recipe = unit.recipes.get(observation)
        recipe = recipe if isinstance(recipe, Recipe) else None
        key = json.dumps([unit.cfg.name, _identity_key(logged_identity(ident)), list(map(str, observation))])
        index, solution_name, kernel_name = (ABSENT if v is UNKNOWN else str(v) for v in observation)
        rows.append({**base,
                     "problem_id": f"{unit.cfg.name}_{hashlib.sha256(key.encode()).hexdigest()[:12]}",
                     "call_count": str(count),
                     "stock_solution_index": index, "stock_solution_name": solution_name,
                     "stock_kernel_name": kernel_name,
                     "stock_logic_file": recipe.source_file if recipe else "",
                     "stock_logic_sha256": recipe.sha256 if recipe else "",
                     "stock_logic_index": str(recipe.in_file_index) if recipe else ""})
    return rows


def _tree(out: Path) -> Tuple[set, set]:
    """(files, directories) below OUT, relative POSIX paths; symlinks count as files."""
    files, dirs = set(), set()
    for directory, names, leaves in os.walk(out):
        rel = Path(directory).relative_to(out)
        dirs.update((rel / n).as_posix() for n in names)
        files.update((rel / n).as_posix() for n in leaves)
    return files, dirs


def _check_exclusive(out: Path) -> None:
    """OUT/logic and OUT/library belong to geko alone. OUT (already resolved) must be a
    directory or absent, those roots real directories or absent (no symlinks), and any path under
    them that the last build's record does not list is a usage error, raised before anything changes."""
    if out.exists() and not out.is_dir():
        raise UsageError(f"--build-library {out}: not a directory")
    for name in ("logic", "library"):
        root = out / name
        if root.is_symlink() or (root.exists() and not root.is_dir()):
            raise UsageError(f"--build-library {out}: {name}/ must be a real directory geko created "
                             "(not a symlink or file), use a clean output directory")
    record = out / BUILD_RECORD
    owned: set = set()
    if record.is_file():
        try:
            data = json.loads(record.read_text())
            owned = set(data["files"]) | set(data["directories"])
        except (ValueError, KeyError, TypeError) as e:
            raise UsageError(f"--build-library {out}: unreadable {BUILD_RECORD} ({e})") from None
    files, dirs = _tree(out) if out.is_dir() else (set(), set())
    foreign = sorted(p for p in files | dirs
                     if p.split("/")[0] in ("logic", "library") and "/" in p and p not in owned)
    if foreign:
        raise UsageError(f"--build-library {out}: logic/ and library/ hold paths geko did not write "
                         f"(e.g. {foreign[0]}); they are geko-owned, use a clean output directory")


def _remove_owned(out: Path) -> None:
    """Remove what the previous --build-library into OUT created (its ownership record), then
    the directories it created if that left them empty. Everything else in OUT stays."""
    record = out / BUILD_RECORD
    if not record.is_file():
        return
    owned = json.loads(record.read_text())
    for rel in owned.get("files", []):
        path = out / rel
        if path.is_symlink() or path.is_file():
            path.unlink()
    for rel in sorted(owned.get("directories", []), key=lambda d: d.count("/"), reverse=True):
        path = out / rel
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    record.unlink()


def _merge_logic(build: BaselineBuild, bespoke: List[Path], out: Path) -> Path:
    """The TensileCreateLibrary input under OUT/logic: a byte-identical copy of the baseline's
    whole source logic tree in ``stock/`` and this run's bespoke files in ``bespoke/``."""
    logic = out / "logic"
    shutil.copytree(build.logic_root, logic / "stock", symlinks=False, dirs_exist_ok=True)
    (logic / "bespoke").mkdir(parents=True, exist_ok=True)
    for path in bespoke:
        shutil.copyfile(path, logic / "bespoke" / path.name)
    return logic


def _build_library(arch: str, logic_dir: Path, out: Path, jobs: int) -> Dict[str, Any]:
    """TensileCreateLibrary with the cmake build's options (tcl_options.md: --cxx-compiler,
    --no-compress, --disable-asm-comments, --gfx1250v0; keep-build-tmp off), --architecture from
    --arch. The build-local MatchTable.yaml TCL writes to its working directory is moved to
    OUT/library/MatchTable.yaml, next to the library/<arch> it indexes."""
    cxx = TCL_CXX if Path(TCL_CXX).exists() else (shutil.which("amdclang++") or TCL_CXX)
    cmd = [sys.executable, "-m", "tensilelite.TensileCreateLibrary", f"--architecture={arch}",
           f"--cxx-compiler={cxx}", f"--jobs={jobs}", *TCL_OPTIONS, str(logic_dir), str(out), "HIP"]
    if arch in UNTESTED:  # the label travels with the library tree
        (out / UNTESTED[arch].replace("untested", "UNTESTED")).write_text(
            f"{arch} output of geko --specialize: generated and built on CPU only, never validated on "
            f"{arch} hardware.\n")
    with open(out / "tcl.log", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=out).returncode
    table = out / "MatchTable.yaml"
    library = out / "library" / arch
    if rc == 0 and table.is_file() and library.is_dir():
        table.replace(out / "library" / "MatchTable.yaml")
    return {"command": cmd, "jobs": jobs, "out": str(out), "library": str(library), "exit": rc,
            "log": str(out / "tcl.log"), "matchtable": str(out / "library" / "MatchTable.yaml")}


def _build_owned(build: BaselineBuild, arch: str, bespoke: List[Path], out: Path, jobs: int,
                 timings: Dict[str, float]) -> Dict[str, Any]:
    """One merged build into OUT that owns exactly what it creates: the previous build's
    recorded outputs are removed first, and what this build adds is recorded for the next."""
    out.mkdir(parents=True, exist_ok=True)
    out = out.resolve()
    t0 = time.perf_counter()
    _remove_owned(out)
    before_files, before_dirs = _tree(out)
    logic = _merge_logic(build, bespoke, out)
    t1 = time.perf_counter()
    result = _build_library(arch, logic, out, jobs)
    timings["logic_merge"] = t1 - t0
    timings["tensile_create_library"] = time.perf_counter() - t1
    after_files, after_dirs = _tree(out)
    (out / BUILD_RECORD).write_text(json.dumps({
        "files": sorted(after_files - before_files - {BUILD_RECORD}),
        "directories": sorted(after_dirs - before_dirs)}, indent=1) + "\n")
    return result


def _default_jobs() -> int:
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    return max(1, min(cpus, MAX_JOBS))


# --- driver -------------------------------------------------------------------------------------

def _error(message: str) -> None:
    print(f"geko --specialize: {message}", file=sys.stderr)


def _invalidate(work: Path) -> None:
    """Remove only what an earlier --specialize run wrote here (its three output files and
    ``logic/<arch>/<arch>_bespoke__*.yaml``), so no stale output survives; user files stay."""
    for name in OUTPUT_FILES:
        (work / name).unlink(missing_ok=True)
    emptied = False
    for arch in ARCH_ISA:
        owned = list((work / "logic" / arch).glob(f"{arch}_bespoke__*.yaml"))
        for path in owned:
            path.unlink()
        # Drop a directory only if this removal emptied it; user directories stay.
        if owned and not any((work / "logic" / arch).iterdir()):
            (work / "logic" / arch).rmdir()
            emptied = True
    if emptied and not any((work / "logic").iterdir()):
        (work / "logic").rmdir()


def run(workload_log: str, route_facts: Optional[str], hipblaslt: str, arch: str, workdir: str,
        keep_beta: bool = False, exclude_controls: Optional[str] = None,
        build_library: Optional[str] = None, stock_only: bool = False, jobs: Optional[int] = None,
        logic_root: Optional[str] = None) -> int:
    """Run ``--specialize``; returns the exit status (0 ok, 1 refusals/errors, 2 usage)."""
    start = time.perf_counter()
    work = Path(workdir)
    try:
        if arch not in ARCH_ISA:
            raise UsageError(f"--arch {arch}: --specialize supports {sorted(ARCH_ISA)}")
        if jobs is not None and not 1 <= jobs <= MAX_JOBS:
            raise UsageError(f"--jobs {jobs}: TensileCreateLibrary jobs must be 1..{MAX_JOBS} "
                             f"(CPU limit {MAX_JOBS})")
        if build_library:
            out, here = Path(build_library).resolve(), work.resolve()
            if out == here or out in here.parents or here in out.parents:
                raise UsageError(f"--build-library {build_library} and --workdir {workdir} must be separate, "
                                 "non-nested directories (the merged build input would pick up the workdir)")
            _check_exclusive(out)
        exclusions = _exclusions(exclude_controls)
        try:
            import tensilelite.LibraryIO  # noqa: F401
        except ImportError as e:
            raise UsageError(f"TensileLite is not importable ({e}); put the source checkout's "
                             "projects/hipblaslt/tensilelite directory and its built rocisa package on "
                             "PYTHONPATH (see the README, Specialize setup)") from None
        build = BaselineBuild(Path(hipblaslt), arch, None, logic_root)
        configs_in = _configurations(Path(workload_log))
        # Without route facts every call's route is unknown: each call is refused, none assumed.
        routes = _route_documents(Path(route_facts)) if route_facts else {}
    except UsageError as e:
        _error(f"usage: {e}")
        return 2

    _invalidate(work)  # only after every usage check passed
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".specialize_", dir=work) as scratch:
        build.scratch = Path(scratch)
        try:
            return _run(build, configs_in, routes, arch, work, keep_beta, exclusions, build_library, stock_only,
                        jobs or _default_jobs(), start)
        except ValueError as e:  # whole-input refusal: a trace or run.json that is not valid input
            _invalidate(work)
            path = getattr(e, "path", None)
            entry = {"code": "whole-input", "reason": f"whole input refused: {e}", "file": str(path) if path else ""}
            label = UNTESTED.get(arch, "")
            doc = {"schema": 1, "arch": arch, **({"label": label} if label else {}), "refusals": [entry]}
            (work / "refusals.json").write_text(json.dumps(doc, indent=2) + "\n")
            _error(entry["reason"])
            return 1


def _narrowing_meet(routes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The narrowing facts of one identity over the configurations in which it is
    eligible (route and domain already checked per configuration). The bias axis counts only if
    every such configuration proves M; otherwise it stays unproven."""
    merged = dict(routes[0], evidence=dict(routes[0].get("evidence") or {}))
    if all(F.proven(r, "bias_axis") and r.get("bias_axis") == "M" for r in routes):
        merged["bias_axis"] = "M"
        merged["evidence"]["bias_axis"] = [e for r in routes for e in r["evidence"]["bias_axis"]]
    else:
        merged["bias_axis"] = None
        merged["evidence"]["bias_axis"] = []
    return merged


def _fingerprint_refusal(cfg: Configuration, build: BaselineBuild) -> Refused:
    """Refusal of a configuration whose run.json does not bind its trace to this build."""
    if cfg.fingerprint == "missing":
        return Refused("fingerprint-missing", "trace fingerprint missing in run.json: no library_fp "
                       f"to bind the trace to {build.library}")
    return Refused("fingerprint", "trace/build fingerprint mismatch: run.json library_fp "
                   f"{cfg.run_json['library_fp']} != {build.fingerprint} of {build.library}")


def _model_run_problem(run_path: Optional[Path], run_json: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a configuration with run.json is not specialization input (None if it is, or has no run.json)."""
    if run_json is None:
        return None
    wrong = [f"{key} {run_json.get(key)!r} (needs {want!r})" for key, want in MODEL_RUN_REQUIRED
             if run_json.get(key) != want]
    if not wrong:
        return None
    return (f"{run_path}: run.json {' and '.join(wrong)}; only a complete analysis (mask-128) model run is "
            "specialization input, so this configuration's calls were not read (not counted: call_count null)")


def _run(build, configs_in, routes, arch, work, keep_beta, exclusions, build_library, stock_only, jobs,
         start) -> int:
    label = UNTESTED.get(arch, "")
    configs: List[Configuration] = []
    units: List[Unit] = []
    errors: List[Dict[str, Any]] = []
    zero: List[Dict[str, Any]] = []
    for name, logs, run_path in configs_in:
        run_json = None
        if run_path:
            try:
                run_json = json.loads(run_path.read_text())
            except (OSError, json.JSONDecodeError) as e:
                raise TraceError(run_path, f"{run_path}: run.json is not valid JSON ({e})") from None
            if not isinstance(run_json, dict):
                raise TraceError(run_path, f"{run_path}: run.json is not a JSON object")
        cfg = Configuration(name, logs, run_json, routes.get(name))
        if run_json is not None:
            cfg.fingerprint = ("missing" if "library_fp" not in run_json
                               else "verified" if run_json["library_fp"] == build.fingerprint else "mismatch")
        configs.append(cfg)
        incomplete = _model_run_problem(run_path, run_json)
        if incomplete:
            # Its calls are deliberately not read, so their number is unknown (null), never 0.
            errors.append({"run_configuration": name, "call_count": None, "code": "run-incomplete",
                           "reason": incomplete})
            continue
        table = load_calls(logs, allow_empty=True)
        if not table:
            route = cfg.route
            per_process = (route or {}).get("per_process") or {}
            if cfg.fingerprint in ("mismatch", "missing"):  # zero calls do not waive the binding
                refusal = _fingerprint_refusal(cfg, build)
                errors.append({"run_configuration": name, "call_count": 0, "code": refusal.code,
                               "reason": refusal.reason})
            elif route and not ((route.get("runtime_route") or {}).get("proven") is True
                                and (route.get("runtime_route") or {}).get("evidence")):
                errors.append({"run_configuration": name, "call_count": 0, "code": "runtime-route-unproven",
                               "reason": "configuration logged no mask-128 call, but its route facts do not prove "
                                         "the runtime route (runtime_route.proven with evidence); "
                                         f"{(route.get('runtime_route') or {}).get('reason') or 'not recorded'}"})
            elif (route and route.get("zero_verified") is True and route.get("thin") is True
                    and route.get("matmul_count") == 0
                    and all(not (p or {}).get("rocblaslt_matmul") for p in per_process.values())):
                zero.append({"run_configuration": name, "reason": "thin configuration, zero calls proven by "
                             "route facts (zero_verified)"})
            else:
                errors.append({"run_configuration": name, "call_count": 0, "code": "zero-calls-unproven",
                               "reason": "configuration logged no mask-128 call and its route facts do not "
                                         "prove zero calls (zero_verified, thin, matmul_count 0)"})
            continue
        route_calls = {_identity_key(c.get("identity") or {}): c for c in (cfg.route or {}).get("calls", [])}
        for identity, entry in table.items():
            units.append(Unit(cfg, identity, entry.call_count, dict(entry.observations), dict(entry.sources),
                              route=route_calls.get(_identity_key(logged_identity(identity)))))

    # Fingerprints, then every observation through the MatchTable.
    recipes_of: Dict[CallIdentity, set] = defaultdict(set)
    for unit in units:
        if unit.cfg.fingerprint in ("mismatch", "missing"):
            unit.refusal = _fingerprint_refusal(unit.cfg, build)
            unit.status = "refused"
            continue
        for observation in unit.observations:
            try:
                unit.recipes[observation] = build.resolve(observation)
                recipes_of[unit.identity].add(unit.recipes[observation].key())
            except Refused as e:
                unit.recipes[observation] = e

    def refuse(unit, error):
        unit.refusal = error
        if unit.status != CUSTOM_STATUS:
            unit.status = "refused"

    # Recipe, then generation/domain eligibility per configuration with its own route facts.
    eligible: Dict[CallIdentity, List[Unit]] = defaultdict(list)
    for unit in units:
        if unit.refusal:
            continue
        try:
            failed = [r for r in unit.recipes.values() if isinstance(r, Refused)]
            if failed:
                raise failed[0]
            if len(recipes_of[unit.identity]) > 1:
                raise Refused("two-recipes", "two recipes for one call identity: "
                              f"{sorted(recipes_of[unit.identity])}")
            unit.recipe = next(iter(unit.recipes.values()))
            stops = (unit.cfg.route or {}).get("stops") or []
            if unit.route is None and stops:
                codes = ", ".join(sorted({str(s.get("code")) for s in stops}))
                raise Refused("route-facts-stopped", f"the route facts of configuration {unit.cfg.name} prove "
                              f"no call: the route-facts producer stopped ({codes}); see its stops")
            F.call_facts(unit.identity, unit.route)
            eligible[unit.identity].append(unit)
        except Refused as e:
            refuse(unit, e)
    # Narrowing facts are decided once per identity over its eligible configurations.
    for group in eligible.values():
        merged = _narrowing_meet([u.route for u in group])
        for unit in group:
            unit.route = merged

    derived: Dict[Tuple[Any, ...], Bespoke] = {}
    used_exclusions = set()
    for unit in (u for group in eligible.values() for u in group):
        try:
            call = F.call_facts(unit.identity, unit.route)
            if unit.recipe.custom:
                unit.status = CUSTOM_STATUS
                raise Refused("custom-kernel", CUSTOM_REASON)
            stock, _ = build.stock(unit.recipe.path)
            F.check_stock(call, stock["ProblemType"], stock["ProblemType"].get("Batched", True), keep_beta)
            if stock_only:
                unit.status = "stock-only"
                continue
            exclusion_key = (*unit.recipe.key(), _identity_key(logged_identity(unit.identity)))
            excluded = exclusions.get(exclusion_key, (frozenset(), []))[0]
            if exclusion_key in exclusions:
                used_exclusions.add(exclusion_key)
            key = (unit.recipe.key(), repr(sorted(call.items(), key=lambda kv: kv[0])), keep_beta, excluded)
            if key not in derived:
                derived[key] = build_bespoke(build, unit.recipe, call, keep_beta, excluded, build.scratch)
            # Self-reachability: the solution's selection facts must accept this identity.
            unreachable = _selection_violations(derived[key], call)
            if unreachable:
                raise Refused("unreachable", "the bespoke solution's selection facts reject its own call: "
                              + "; ".join(unreachable))
            unit.bespoke = derived[key]
        except Refused as e:
            refuse(unit, e)

    # One logic file per distinct bespoke logic, named by its content only: neither the
    # insertion order nor the recipe's source file renames or duplicates it.
    logic_dir = work / "logic" / arch
    for unit in units:
        if unit.bespoke:
            text = json.dumps(unit.bespoke.logic, sort_keys=True, default=str)
            suffix = f"__{label}" if label else ""
            unit.logic_file = f"{arch}_bespoke__{hashlib.sha256(text.encode()).hexdigest()[:16]}{suffix}.yaml"
            path = logic_dir / unit.logic_file
            if not path.exists():
                write_logic(path, unit.bespoke.logic)

    unused = [entry for key, (_, entries) in exclusions.items() if key not in used_exclusions for entry in entries]
    _write_outputs(build, configs, units, errors, zero, arch, work, keep_beta, stock_only, unused)
    timings = {"parse_specialize": time.perf_counter() - start, "logic_merge": 0.0, "tensile_create_library": 0.0}
    library_failed = False
    manifest_path = work / "specialize_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if build_library:
        # Merged library: the baseline's stock logic plus this run's bespoke files
        # (none with --stock-only); never anything else a user keeps under logic/<arch>.
        written = sorted({u.logic_file for u in units if u.logic_file})
        manifest["build"]["library"] = _build_owned(build, arch, [logic_dir / name for name in written],
                                                    Path(build_library), jobs, timings)
        manifest["build"]["library"].update(stock_logic=str(build.logic_root), bespoke_files=written)
        library_failed = manifest["build"]["library"]["exit"] != 0
        if library_failed:
            _error(f"TensileCreateLibrary failed; see {manifest['build']['library']['log']}")
    timings["total"] = time.perf_counter() - start
    manifest["timings_seconds"] = timings
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    refused = [u for u in units if u.refusal]
    for unit in refused:
        _error(f"refused ({unit.cfg.name}, {unit.call_count} calls): {unit.refusal.reason}")
    for entry in errors:
        _error(f"{entry['run_configuration']}: {entry['reason']}")
    for entry in unused:
        _error(f"warning: --exclude-controls entry matched no call: {entry}")
    return 1 if refused or errors or library_failed else 0


def _write_outputs(build, configs, units, errors, zero, arch, work, keep_beta, stock_only, unused_exclusions):
    label = UNTESTED.get(arch, "")
    models = {c.name: str(((c.run_json or {}).get("model") or {}).get("slug", "")) for c in configs}
    rows = [row for unit in units for row in _csv_rows(unit, arch, models[unit.cfg.name])]
    extra = [c for c in (rows[0] if rows else {}) if c not in MANIFEST_COLUMNS]
    if not rows:
        extra = ["run_configuration", "arch", "label", "status", "refusal"]
    with open(work / "calls.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS + extra)
        writer.writeheader()
        writer.writerows(rows)

    calls = []
    for unit in sorted(units, key=lambda u: (u.cfg.name, json.dumps(_identity_key(logged_identity(u.identity))))):
        if unit.status not in ("generated", CUSTOM_STATUS, "stock-only"):
            continue
        # An unsupported custom-kernel call has no expected bespoke kernel; consumers count it as fallback.
        pending = unit.status == CUSTOM_STATUS
        kernel = None if pending else unit.bespoke.kernel_name if unit.bespoke else unit.recipe.kernel_name
        calls.append({
            "identity": logged_identity(unit.identity), "run_configuration": unit.cfg.name,
            "call_count": unit.call_count, "kernel_name": kernel,
            "solution_name": None if pending else unit.bespoke.solution_name if unit.bespoke
            else unit.recipe.solution_name,
            "stock_kernel_name": unit.recipe.kernel_name,
            "status": unit.status,
            "recipe": unit.recipe.manifest(),
            "logic_file": unit.logic_file,
            "facts_applied": unit.bespoke.applied if unit.bespoke else [],
            "facts_not_applied": unit.bespoke.not_applied if unit.bespoke else [],
            "keep_beta": bool(keep_beta), **({"label": label} if label else {}),
        })
    grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for unit in units:
        if not unit.refusal:
            continue
        key = (_identity_key(logged_identity(unit.identity)), unit.refusal.code, unit.refusal.reason)
        entry = grouped.setdefault(key, {"identity": logged_identity(unit.identity), "code": unit.refusal.code,
                                         "reason": unit.refusal.reason, "call_count": 0,
                                         "run_configurations": []})
        entry["call_count"] += unit.call_count
        entry["run_configurations"].append(unit.cfg.name)
    refusals = [grouped[k] for k in sorted(grouped, key=lambda k: json.dumps(k))]
    refusals += [{"identity": None, **e} for e in errors]
    manifest = {
        "schema": 1, "arch": arch, **({"label": label} if label else {}),
        "stage": "keep-beta" if keep_beta else "B", "stock_only": bool(stock_only),
        "build": {"root": str(build.root), "matchtable": str(build.matchtable_path),
                  "source_logic": str(build.logic_root), "library": str(build.library),
                  "fingerprint": build.fingerprint,
                  "fingerprint_algorithm": "model_run.library_fp (sorted walk: rel\\0size\\0sha256\\0, "
                                           "symlink rel\\0L\\0target\\0)",
                  "trace_fingerprints": {c.name: c.fingerprint for c in configs}},
        "calls": calls,
        "zero_call_configurations": zero,
        "configuration_errors": errors,
        "unused_exclusions": unused_exclusions,
        "refusals": refusals,
    }
    (work / "specialize_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    doc = {"schema": 1, "arch": arch, **({"label": label} if label else {}), "refusals": refusals}
    (work / "refusals.json").write_text(json.dumps(doc, indent=2) + "\n")
