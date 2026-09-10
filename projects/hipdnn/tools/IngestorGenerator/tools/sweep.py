#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Measure ordered installed arms; resume only content-bound, fully gated phases."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from device_probe import ARCH_TOKEN, device_info


class ConfigError(ValueError):
    """Invalid declarative input; no measurement is authorized."""


class GateError(RuntimeError):
    """A required observation is missing or failed."""


class UniqueSafeLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in result:
            raise ConfigError(f"non-string or duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    def invalid(value):
        raise ValueError(f"non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(), parse_constant=invalid)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        if hasattr(os, "O_DIRECTORY"):
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    finally:
        temporary.unlink(missing_ok=True)


def _fields(value, required, optional=()):
    if not isinstance(value, dict):
        raise ConfigError("expected a YAML mapping")
    missing = set(required) - value.keys()
    unknown = value.keys() - set(required) - set(optional)
    if missing or unknown:
        raise ConfigError(
            f"missing keys {sorted(missing)}; unknown keys {sorted(unknown)}"
        )


def _count(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{label} must be an integer >= {minimum}")


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{label} must be a nonempty string")
    return value


def _path(value, base):
    path = Path(_text(value, "path"))
    return str((base / path).resolve())


#: Interpreters that make argv[0] a shell command line rather than a benchmark
#: executable. A wrapper reaches the driver either AS one of these or WITH one on
#: its shebang line, and both spellings hide the real command from the ledger.
SHELL_LAUNCHERS = {
    "sh",
    "bash",
    "dash",
    "zsh",
    "fish",
    "ksh",
    "csh",
    "tcsh",
    "cmd",
    "powershell",
    "pwsh",
}


def _launcher_name(path):
    """The interpreter argv[0] actually selects: itself, or its shebang's."""
    path = Path(path)
    if path.stem.lower() in SHELL_LAUNCHERS:
        return path.stem.lower()
    try:
        with path.open("rb") as handle:
            first = handle.readline(4096).decode("utf-8", errors="replace").strip()
    except OSError as exc:
        raise ConfigError(f"benchmark executable is unreadable: {exc}") from exc
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    if not parts:
        return None
    selected = parts[1] if len(parts) > 1 and Path(parts[0]).name == "env" else parts[0]
    return Path(selected).stem.lower()


def _argv(value, base):
    if not isinstance(value, list) or not value:
        raise ConfigError("argv must be a nonempty string list, not a shell command")
    for arg in value:
        _text(arg, "argv element")
    command = value[0]
    resolved = (
        str((base / command).resolve())
        if "/" in command or "\\" in command
        else shutil.which(command)
    )
    if not resolved or not Path(resolved).is_file() or not os.access(resolved, os.X_OK):
        raise ConfigError(f"executable not available: {command}")
    launcher = _launcher_name(Path(resolved).resolve())
    if launcher in SHELL_LAUNCHERS:
        raise ConfigError(
            f"shell launchers are not sweep executables: {command} selects {launcher}"
        )
    return [resolved, *value[1:]]


#: The complete top-level surface a sweep config declares. Module level so the
#: shipped example is checked against the parser rather than against a second list
#: that can drift from it.
REQUIRED_KEYS = (
    "sweep_root",
    "output_dir",
    "corpus_dir",
    "arch",
    "engine_name",
    "engine_ued_name",
    "corpora",
    "arms",
    "warmup_arm",
    "rounds",
    "min_served",
    "exclude_tensors",
    "benchmark",
    "correctness",
)
OPTIONAL_KEYS = ("probe_env",)


def load_config(path):
    path = Path(path).resolve()
    try:
        # nosec B506 -- UniqueSafeLoader subclasses yaml.SafeLoader and additionally
        # rejects duplicate and non-string keys, which yaml.safe_load accepts silently.
        # safe_load takes no Loader argument, so this is the only way to get both.
        config = yaml.load(path.read_text(), Loader=UniqueSafeLoader)  # nosec B506
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(str(exc)) from exc
    _fields(config, REQUIRED_KEYS, OPTIONAL_KEYS)
    base = path.parent
    for key in ("sweep_root", "output_dir", "corpus_dir"):
        config[key] = _path(config[key], base)
    for key in ("sweep_root", "corpus_dir"):
        if not Path(config[key]).is_dir():
            raise ConfigError(f"{key} must already exist: {config[key]}")
    output = Path(config["output_dir"])
    if output == Path(config["sweep_root"]) or not output.is_relative_to(
        config["sweep_root"]
    ):
        raise ConfigError("output_dir must be a dedicated child of sweep_root")
    if not re.fullmatch(ARCH_TOKEN, _text(config["arch"], "arch")):
        raise ConfigError("arch must be an exact gfx token")
    for key in ("engine_name", "engine_ued_name"):
        _text(config[key], key)
    for key in ("rounds", "min_served"):
        _count(config[key], key)
    inputs = [Path(config["corpus_dir"])]
    for key, fields, path_key, count_key in (
        ("corpora", ("name", "path", "expected_graphs"), "path", "expected_graphs"),
        (
            "arms",
            ("name", "install_tree", "expected_descriptors"),
            "install_tree",
            "expected_descriptors",
        ),
    ):
        rows = config[key]
        if not isinstance(rows, list) or not rows:
            raise ConfigError(f"{key} must be a nonempty ordered list")
        names = set()
        for row in rows:
            _fields(row, fields)
            name = _text(row["name"], "name")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in names:
                raise ConfigError(f"unsafe or duplicate {key} name: {name}")
            names.add(name)
            _count(row[count_key], count_key)
            row[path_key] = _path(row[path_key], base)
            root = Path(row[path_key])
            if not root.is_dir():
                raise ConfigError(f"input root does not exist: {root}")
            if key == "corpora" and not root.is_relative_to(config["corpus_dir"]):
                raise ConfigError("every corpus path must be inside corpus_dir")
            inputs.append(root)
    if config["warmup_arm"] is not None and config["warmup_arm"] not in {
        a["name"] for a in config["arms"]
    }:
        raise ConfigError("warmup_arm must name an arm or be explicit null")
    if any(config["min_served"] > c["expected_graphs"] for c in config["corpora"]):
        raise ConfigError("min_served exceeds corpus inventory")
    if any(
        output.is_relative_to(root) or root.is_relative_to(output) for root in inputs
    ):
        raise ConfigError("output and input roots must not overlap")
    excluded = config["exclude_tensors"]
    if excluded != "none":
        if not isinstance(excluded, list) or not excluded:
            raise ConfigError("exclude_tensors must be 'none' or a nonempty list")
        config["exclude_tensors"] = [
            _text(t, "excluded tensor").lower() for t in excluded
        ]
    benchmark = config["benchmark"]
    _fields(benchmark, ("argv", "warmup", "iters"))
    benchmark["argv"] = _argv(benchmark["argv"], base)
    # Phase flags are owned by this driver, so duplicate options cannot redirect evidence.
    reserved = {
        "--graph",
        "--plugin-path",
        "--warmup",
        "--iters",
        "-o",
        "--output",
        "--validate",
        "--engine",
        "--backend",
    }
    if any(arg.split("=", 1)[0] in reserved for arg in benchmark["argv"][1:]):
        raise ConfigError("benchmark.argv must not override phase-owned options")
    correctness = config["correctness"]
    _fields(correctness, ("enabled", "reference", "warmup", "iters"))
    if type(correctness["enabled"]) is not bool:
        raise ConfigError("correctness.enabled must be boolean")
    if correctness["enabled"]:
        _text(correctness["reference"], "correctness.reference")
    elif correctness["reference"] is not None and not isinstance(
        correctness["reference"], str
    ):
        raise ConfigError("correctness.reference must be a string or null")
    for row in (benchmark, correctness):
        _count(row["warmup"], "warmup", 0)
        _count(row["iters"], "iters")
    if config.get("probe_env") is not None:
        config["probe_env"] = _argv(config["probe_env"], base)
    config["config_dir"] = str(base)
    return config


def corpus_inventory(corpus, exclusions):
    root = Path(corpus["path"])
    inventory = []
    names = set()
    for path in sorted(root.rglob("*.json")):
        try:
            graph = read_json(path)
            if not isinstance(graph, dict) or not isinstance(
                graph.get("tensors"), list
            ):
                raise ValueError("expected a graph mapping with tensors")
            name = graph.get("name", path.stem)
            _text(name, "graph name")
            if name in names:
                raise ValueError(f"ambiguous duplicate graph name {name!r}")
            names.add(name)
            tensor_names = {str(t.get("name", "")).lower() for t in graph["tensors"]}
            if exclusions != "none" and tensor_names & set(exclusions):
                raise ValueError(
                    f"excluded tensor hazard: {sorted(tensor_names & set(exclusions))}"
                )
            semantic = {
                k: v
                for k, v in graph.items()
                if k not in ("name", "provenance") and not k.startswith("_")
            }
            inventory.append(
                {
                    "graph_name": name,
                    "source_path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "sha256": file_hash(path),
                    "semantic_identity": digest(semantic),
                    "provenance": graph.get("_provenance", graph.get("provenance", {})),
                }
            )
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ConfigError(f"{path}: {exc}") from exc
    if len(inventory) != corpus["expected_graphs"]:
        raise ConfigError(
            f"{root}: {len(inventory)} graphs, expected {corpus['expected_graphs']}"
        )
    return inventory


def tree_identity(root):
    root = Path(root)
    if not root.is_dir():
        raise GateError(f"identity root is missing: {root}")
    return {
        str(p.relative_to(root)): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
    }


def command_identity(argv, base):
    files = {argv[0]: file_hash(argv[0])}
    for arg in argv[1:]:
        path = Path(base) / arg
        if path.is_file():
            files[str(path.resolve())] = file_hash(path)
    return files


# Inspect the actual benchmark interpreter, without importing the GPU packages.
# Distribution files include shared objects; package roots include editable source.
_ENV_PROBE = """
import importlib.metadata as m, importlib.util as u, json, pathlib, sys
roots = set()
files = {str(pathlib.Path(sys.executable).resolve())}
for name in ('dnn_benchmarking', 'hipdnn_frontend', 'torch'):
    spec = u.find_spec(name)
    if spec:
        if spec.submodule_search_locations:
            roots.update(str(pathlib.Path(p).resolve()) for p in spec.submodule_search_locations)
        elif spec.origin:
            files.add(str(pathlib.Path(spec.origin).resolve()))
for dist in m.distributions():
    for entry in dist.files or ():
        p = pathlib.Path(dist.locate_file(entry))
        if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts:
            files.add(str(p.resolve()))
print(json.dumps({'python': sys.executable, 'roots': sorted(roots), 'files': sorted(files)}))
"""


def runtime_identity(config, env):
    command = Path(config["benchmark"]["argv"][0])
    interpreter = str(command)
    if not command.name.startswith("python"):
        with command.open("rb") as handle:
            shebang = handle.readline(4096).decode("utf-8", errors="replace").strip()
        if not shebang.startswith("#!"):
            raise GateError(
                "benchmark executable must expose its Python environment via a shebang"
            )
        parts = shebang[2:].split()
        if len(parts) == 2 and Path(parts[0]).name == "env":
            interpreter = shutil.which(parts[1], path=env.get("PATH"))
        elif len(parts) == 1:
            interpreter = parts[0]
        else:
            raise GateError("cannot identify benchmark interpreter from its shebang")
        if not interpreter or not Path(interpreter).name.startswith("python"):
            raise GateError("benchmark shebang does not select Python")
    result = subprocess.run(
        [interpreter, "-c", _ENV_PROBE],
        cwd=config["config_dir"],
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise GateError(
            f"benchmark environment identity failed: {result.stderr.strip()}"
        )
    observed = json.loads(result.stdout)
    files = {p: file_hash(p) for p in observed["files"]}
    for root in observed["roots"]:
        files.update(
            {str(Path(root) / p): sha for p, sha in tree_identity(root).items()}
        )
    return {"python": observed["python"], "files": files}


def phase_env(original, arm, attempt):
    env = original.copy()
    install = Path(arm["install_tree"])
    env.update(
        ROCM_PATH=str(install),
        LD_LIBRARY_PATH=str(install / "lib")
        + (
            os.pathsep + original["LD_LIBRARY_PATH"]
            if original.get("LD_LIBRARY_PATH")
            else ""
        ),
        HIPDNN_CACHE_DIR=str(attempt / "cache"),
        HIPDNN_LOG_FILE=str(attempt / "hipdnn.log"),
        HIPDNN_FORCE_BENCHMARKING="1",
        HIPDNN_LOG_LEVEL="info",
    )
    return env


def discover_engine(config, arm, env, destination):
    install = Path(arm["install_tree"])
    command = [
        str(install / "bin" / "hipdnn_list_engines"),
        "--plugin-dir",
        str(install / "lib" / "hipdnn_plugins" / "engines"),
    ]
    result = subprocess.run(
        command, cwd=config["config_dir"], env=env, text=True, capture_output=True
    )
    destination.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise GateError(f"engine discovery exited {result.returncode}")
    engines = re.findall(
        r"^  (.+) \((0x[0-9a-fA-F]+)\)$", result.stdout, flags=re.MULTILINE
    )
    selected = [
        int(engine_id, 16)
        for name, engine_id in engines
        if name == config["engine_ued_name"]
    ]
    if len(selected) != 1:
        raise GateError(
            "installed engine discovery did not uniquely identify engine_ued_name"
        )
    engine_id = selected[0]
    if config["engine_name"] not in (
        config["engine_ued_name"],
        f"engine_{engine_id:#x}",
    ):
        raise GateError(
            "engine_name does not match the installed name or exact engine-ID fallback"
        )
    return engine_id


def descriptor_count(arm):
    count = 0
    for path in Path(arm["install_tree"]).rglob("*.kdp.json"):
        doc = read_json(path)
        kernels = doc["kernelDescriptors"]
        if not isinstance(kernels, list):
            raise ValueError(f"invalid kernelDescriptors in {path}")
        count += len(kernels)
    return count


def _positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def signed64(engine_id):
    """The engine ID as the benchmark CLI spells it.

    Engine discovery prints unsigned hex; the benchmark's `--engine` takes a decimal
    int and its JSON reports the same value reinterpreted as signed int64. One
    identity, two spellings, converted here rather than in each caller.
    """
    return engine_id - (1 << 64) if engine_id >= (1 << 63) else engine_id


def evaluate_phase(
    config,
    arm,
    inventory,
    engine_id,
    kind,
    result_path,
    hip_log,
    command_exit,
    required_served=(),
):
    """Evaluate real graph-first dnn-benchmark JSON, independently of command status."""
    gates = {
        "command": command_exit == 0,
        "descriptors": False,
        "provenance": False,
        "parsed_inventory": False,
        "metadata": False,
        "served": False,
        "outcomes": False,
    }
    if kind == "correctness":
        gates["correctness"] = False
        gates["reference"] = False
    errors = []
    rows = {}
    ledger = []
    try:
        gates["descriptors"] = descriptor_count(arm) == arm["expected_descriptors"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"descriptor census: {exc}")
    plugin_dir = Path(arm["install_tree"]) / "lib" / "hipdnn_plugins" / "engines"
    try:
        loaded = re.findall(r"load plugin from \[([^\]]+)\]", Path(hip_log).read_text())
        gates["provenance"] = any(
            Path(p).resolve().parent == plugin_dir for p in loaded
        )
    except (OSError, UnicodeError) as exc:
        errors.append(f"plugin provenance: {exc}")
    metadata = None
    try:
        doc = read_json(result_path)
        graphs = doc["graphs"]
        if not isinstance(graphs, list):
            raise ValueError("graphs must be a list")
        for graph in graphs:
            name = graph["graph_name"]
            if (
                not isinstance(name, str)
                or name in rows
                or not isinstance(graph["results"], list)
            ):
                raise ValueError("duplicate graph name or invalid results list")
            rows[name] = graph
        gates["parsed_inventory"] = set(rows) == {g["graph_name"] for g in inventory}
        metadata = doc["metadata"]
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(f"result parse: {exc}")
        rows = {}
        metadata = None
    for source in inventory:
        name = source["graph_name"]
        entry = {
            **source,
            "expected_engine": config["engine_name"],
            "engine_id": engine_id,
            "outcome": "missing",
            "reason": "no exact engine result",
            "mean_ms": None,
            "correctness": None,
            "evidence": str(result_path),
        }
        graph = rows.get(name, {})
        candidates = []
        references = []
        for row in graph.get("results", []):
            if not isinstance(row, dict):
                entry.update(outcome="ambiguous", reason="non-mapping result row")
                continue
            if row.get("role", "engine") == "reference":
                references.append(row)
                continue
            if row.get("engine_name") == config["engine_name"]:
                candidates.append(row)
        # A reference row is only evidence when it names the requested provider and
        # actually ran: a silently skipped reference leaves every engine row with
        # tolerance_match null, which the suite counts as a pass.
        wanted = config["correctness"]["reference"]
        chosen = [
            r
            for r in references
            if r.get("provider") == wanted and r.get("status") == "success"
        ]
        entry["reference"] = {
            "expected_provider": wanted,
            "observed": [
                {
                    "provider": r.get("provider"),
                    "status": r.get("status"),
                    "skip_reason": r.get("skip_reason"),
                    "error_message": r.get("error_message"),
                    "warnings": r.get("warnings"),
                }
                for r in references
            ],
            "usable": len(chosen) == 1,
        }
        if len(candidates) > 1:
            entry.update(outcome="ambiguous", reason="duplicate exact engine rows")
        elif len(candidates) == 1:
            row = candidates[0]
            observed_id = row.get("engine_id")
            if (
                type(observed_id) is not int
                or (observed_id & ((1 << 64) - 1)) != engine_id
            ):
                entry.update(outcome="ambiguous", reason="engine name/ID disagreement")
            elif (
                row.get("plugin_path")
                and Path(row["plugin_path"]).resolve().parent != plugin_dir
            ):
                entry.update(
                    outcome="ambiguous",
                    reason="engine row came from a different plugin path",
                )
            elif row.get("status") == "success":
                mean = (row.get("gpu_kernel_stats") or {}).get("mean_ms")
                comparison = row.get("correctness")
                if not _positive(mean):
                    entry.update(
                        outcome="execution_error",
                        reason="missing or nonpositive finite GPU timing",
                    )
                else:
                    entry.update(
                        outcome="served",
                        reason="",
                        mean_ms=mean,
                        correctness=comparison,
                    )
            elif (
                row.get("status") == "skipped"
                and isinstance(row.get("skip_reason"), str)
                and row["skip_reason"].strip()
            ):
                entry.update(outcome="declined", reason=row["skip_reason"])
            else:
                entry.update(
                    outcome="execution_error",
                    reason=row.get("error_message")
                    or "invalid or failed engine result",
                )
        ledger.append(entry)
    served = {e["graph_name"] for e in ledger if e["outcome"] == "served"}
    gates["served"] = len(served) >= config["min_served"]
    gates["outcomes"] = all(e["outcome"] in ("served", "declined") for e in ledger)
    # Suite metadata is a second, independent readout of the same run: the suite
    # counts every engine-role row itself, and the phase command pins the engine, so
    # a nonzero fail/error count is this engine's. gpu_arch corroborates the device
    # the rocminfo gate already established; the suite reports "unknown" when it has
    # no torch to ask, so an absent readout is not evidence against the run.
    observed_arch = (metadata or {}).get("gpu_arch")
    gates["metadata"] = bool(metadata) and (
        metadata.get("total_graphs") == len(inventory)
        and metadata.get("fail_combinations") == 0
        and metadata.get("error_combinations") == 0
        and observed_arch in (None, "unknown", config["arch"])
    )
    if kind == "correctness":

        def compared(entry):
            result = entry["correctness"]
            return (
                isinstance(result, dict)
                and all(
                    result.get(k) is True
                    for k in ("passed", "execution_success", "tolerance_match")
                )
                and all(
                    type(value) in (int, float) and math.isfinite(value)
                    for key, value in result.items()
                    if key in ("max_abs_diff", "max_rel_diff", "rtol", "atol")
                )
            )

        gates["correctness"] = (
            bool(served)
            and set(required_served) <= served
            and all(compared(e) for e in ledger if e["outcome"] == "served")
        )
        gates["reference"] = bool(ledger) and all(
            e["reference"]["usable"] for e in ledger if e["outcome"] == "served"
        )
    return {
        "gates": gates,
        "errors": errors,
        "ledger": ledger,
        "served": sorted(served),
        "success": all(gates.values()),
    }


def phase_key(kind, corpus, arm, round_number=0):
    return {
        "kind": kind,
        "corpus": corpus["name"],
        "arm": arm["name"],
        "round": round_number,
    }


def phase_tag(key):
    return f"{key['kind']}__{key['corpus']}__{key['arm']}__r{key['round']}"


def phase_fingerprint(
    config, key, inventory, installed, runtime, device, environment, required
):
    return digest(
        {
            "config": config,
            "phase": key,
            "corpus": inventory,
            "install": installed,
            "runtime": runtime,
            "device": device,
            "environment": environment,
            "required_served": sorted(required),
            "command": command_identity(
                config["benchmark"]["argv"], config["config_dir"]
            ),
            "driver": {
                str(Path(__file__).resolve()): file_hash(__file__),
                str(Path(__file__).with_name("device_probe.py")): file_hash(
                    Path(__file__).with_name("device_probe.py")
                ),
            },
        }
    )


def resume_phase(
    sidecar, fingerprint, key, config, arm, inventory, engine_id, required
):
    try:
        record = read_json(sidecar)
        if (
            record["status"] != "success"
            or record["phase"] != key
            or record["fingerprint"] != fingerprint
        ):
            return None
        root = Path(config["output_dir"])
        for path, sha in record["evidence_hashes"].items():
            if not Path(path).resolve().is_relative_to(root) or file_hash(path) != sha:
                return None
        if not record["evidence_hashes"] or not all(record["gates"].values()):
            return None
        required_evidence = {
            record["result_path"],
            record["hip_log"],
            record["command_log"],
        }
        if not required_evidence <= record["evidence_hashes"].keys():
            return None
        checked = evaluate_phase(
            config,
            arm,
            inventory,
            engine_id,
            key["kind"],
            record["result_path"],
            record["hip_log"],
            record["command_exit"],
            required,
        )
        checked["gates"]["stable_inputs"] = True
        if not checked["success"] or checked["gates"] != record["gates"]:
            return None
        checked["ledger"] = [
            {**entry, "phase": key, "fingerprint": fingerprint}
            for entry in checked["ledger"]
        ]
        return {**record, **checked, "resumed": True}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_phase(
    config,
    key,
    corpus,
    arm,
    inventory,
    engine_id,
    stage,
    fingerprint,
    original_env,
    session,
    required,
    inputs_stable,
):
    output = Path(config["output_dir"])
    tag = phase_tag(key)
    sidecar = output / f"{tag}.complete.json"
    sidecar.unlink(missing_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix=tag + "-", dir=output / "attempts"))
    cache = attempt / "cache"
    cache.mkdir()
    env = phase_env(original_env, arm, attempt)
    result_path = attempt / "results.json"
    command_log = attempt / "command.log"
    hip_log = attempt / "hipdnn.log"
    counts = (
        config["correctness"] if key["kind"] == "correctness" else config["benchmark"]
    )
    # The engine is SELECTED, never left to per-graph ranking: an unranked engine
    # produces no row at all, which is indistinguishable from a graph the suite never
    # reached. Selecting it turns that into an explicit decline, and it makes the
    # suite's own exit status and combination counts refer to this engine alone.
    command = [
        *config["benchmark"]["argv"],
        "--graph",
        str(stage / "*.json"),
        "--plugin-path",
        str(Path(arm["install_tree"]) / "lib" / "hipdnn_plugins" / "engines"),
        "--engine",
        str(signed64(engine_id)),
        "--warmup",
        str(counts["warmup"]),
        "--iters",
        str(counts["iters"]),
        "-o",
        str(result_path),
    ]
    if key["kind"] == "correctness":
        command.extend(("--validate", config["correctness"]["reference"]))
    rc = -1
    try:
        with command_log.open("w") as log:
            try:
                rc = subprocess.run(
                    command,
                    cwd=config["config_dir"],
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ).returncode
            except OSError as exc:
                log.write(str(exc) + "\n")
        checked = evaluate_phase(
            config,
            arm,
            inventory,
            engine_id,
            key["kind"],
            result_path,
            hip_log,
            rc,
            required,
        )
        checked["gates"]["stable_inputs"] = inputs_stable()
        checked["success"] = all(checked["gates"].values())
        winners = attempt / "winners"
        for path in sorted(cache.rglob("winners.jsonl")):
            destination = winners / path.relative_to(cache)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
        ledger_path = attempt / "outcomes.json"
        ledger = [
            {**entry, "phase": key, "fingerprint": fingerprint}
            for entry in checked["ledger"]
        ]
        atomic_json(ledger_path, ledger)
        record = {
            "status": "success" if checked["success"] else "failed",
            "phase": key,
            "fingerprint": fingerprint,
            "session": session,
            "command": command,
            "command_exit": rc,
            "result_path": str(result_path),
            "hip_log": str(hip_log),
            "command_log": str(command_log),
            **checked,
            "ledger": ledger,
            "resumed": False,
        }
        evidence = [
            p for p in attempt.rglob("*") if p.is_file() and not p.is_relative_to(cache)
        ]
        record["evidence_hashes"] = {str(p): file_hash(p) for p in sorted(evidence)}
        atomic_json(attempt / "phase.json", record)
        if checked["success"]:
            atomic_json(sidecar, record)
        print(
            f"{tag}: {'PASS' if checked['success'] else 'FAIL'} {canonical(checked['gates'])}",
            flush=True,
        )
        return record
    finally:
        shutil.rmtree(cache)


def report(config, records, installed):
    """Separate coverage/correctness from paired timing, preserving corpus strata."""
    comparisons = []
    baseline = config["arms"][0]["name"]
    timed = {
        (r["phase"]["corpus"], r["phase"]["arm"], r["phase"]["round"]): r
        for r in records
        if r["phase"]["kind"] == "timing" and r["success"]
    }
    for corpus in config["corpora"]:
        for arm in config["arms"][1:]:
            for round_number in range(1, config["rounds"] + 1):
                a = timed.get((corpus["name"], baseline, round_number))
                b = timed.get((corpus["name"], arm["name"], round_number))
                if not a or not b:
                    continue
                left = {
                    e["graph_name"]: e for e in a["ledger"] if e["outcome"] == "served"
                }
                right = {
                    e["graph_name"]: e for e in b["ledger"] if e["outcome"] == "served"
                }
                groups = {}
                for name in sorted(left.keys() & right.keys()):
                    provenance = left[name]["provenance"]
                    source = (
                        provenance.get("source", "unspecified")
                        if isinstance(provenance, dict)
                        else "unspecified"
                    )
                    groups.setdefault(str(source), []).append(
                        (left[name]["mean_ms"], right[name]["mean_ms"])
                    )
                for source, pairs in groups.items():
                    comparisons.append(
                        {
                            "corpus": corpus["name"],
                            "source": source,
                            "arm": arm["name"],
                            "round": round_number,
                            "paired_graphs": len(pairs),
                            "geomean_ratio": math.exp(
                                sum(math.log(x / y) for x, y in pairs) / len(pairs)
                            ),
                            "time_weighted_ratio": sum(x for x, _ in pairs)
                            / sum(y for _, y in pairs),
                            "byte_identical_install": installed[baseline]
                            == installed[arm["name"]],
                        }
                    )
    drift = []
    for corpus in config["corpora"]:
        for arm in config["arms"]:
            first = timed.get((corpus["name"], arm["name"], 1))
            if not first:
                continue
            initial = {
                e["graph_name"]: e["mean_ms"]
                for e in first["ledger"]
                if e["outcome"] == "served"
            }
            for number in range(2, config["rounds"] + 1):
                later = timed.get((corpus["name"], arm["name"], number))
                if not later:
                    continue
                current = {
                    e["graph_name"]: e["mean_ms"]
                    for e in later["ledger"]
                    if e["outcome"] == "served"
                }
                shared = initial.keys() & current.keys()
                if shared:
                    drift.append(
                        {
                            "corpus": corpus["name"],
                            "arm": arm["name"],
                            "round": number,
                            "paired_graphs": len(shared),
                            "sum_time_vs_round1": sum(current[n] for n in shared)
                            / sum(initial[n] for n in shared),
                        }
                    )
    return {
        "timing_comparisons": comparisons,
        "round_drift": drift,
        "coverage": [
            {
                "phase": r["phase"],
                "counts": dict(Counter(e["outcome"] for e in r["ledger"])),
                "gates": r["gates"],
            }
            for r in records
        ],
        "numerical_correctness": [
            {
                "phase": r["phase"],
                "passed": r["gates"].get("correctness", False),
                "served": r["served"],
            }
            for r in records
            if r["phase"]["kind"] == "correctness"
        ],
    }


def execute(config):
    inventories = {
        c["name"]: corpus_inventory(c, config["exclude_tensors"])
        for c in config["corpora"]
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    lock = output / ".running.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise GateError(f"{lock} exists; its owner may still be running") from exc
    stage_root = None
    original = os.environ.copy()
    session = {
        "id": uuid.uuid4().hex,
        "host": socket.gethostname(),
        "job": original.get("SLURM_JOB_ID"),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        atomic_json(lock / "owner.json", session)
        (output / "summary.json").unlink(missing_ok=True)
        (output / "attempts").mkdir(exist_ok=True)
        # An absent or wrong-arch device is a required observation that failed, so
        # it declines the sweep (exit 1) rather than reporting a driver defect.
        try:
            device_text = device_info(
                config["arch"], cwd=config["config_dir"], env=original
            )
        except ValueError as exc:
            raise GateError(str(exc)) from exc
        device = {
            "host": socket.gethostname(),
            "arch": config["arch"],
            "rocminfo_sha256": hashlib.sha256(device_text.encode()).hexdigest(),
        }
        (output / "rocminfo.log").write_text(device_text)
        environment = {
            k: v
            for k, v in original.items()
            if k.startswith(
                ("HIP", "ROCM", "ROCR", "CUDA", "HSA", "PYTHON", "TORCH", "LD_")
            )
            or k in ("PATH", "VIRTUAL_ENV")
        }
        if config.get("probe_env"):
            probe = subprocess.run(
                config["probe_env"],
                cwd=config["config_dir"],
                env=original,
                capture_output=True,
                text=True,
            )
            (output / "environment-probe.log").write_text(probe.stdout + probe.stderr)
            if probe.returncode:
                raise GateError(f"environment probe exited {probe.returncode}")
            environment["probe"] = {
                "command": command_identity(config["probe_env"], config["config_dir"]),
                "output": probe.stdout,
            }
        installed = {
            a["name"]: tree_identity(a["install_tree"]) for a in config["arms"]
        }
        runtime = {}
        engines = {}
        for arm in config["arms"]:
            with tempfile.TemporaryDirectory(
                prefix="discovery-", dir=output
            ) as temporary:
                scratch = Path(temporary)
                (scratch / "cache").mkdir()
                env = phase_env(original, arm, scratch)
                runtime[arm["name"]] = runtime_identity(config, env)
                engines[arm["name"]] = discover_engine(
                    config, arm, env, output / (arm["name"] + ".engines.log")
                )
        atomic_json(
            output / "input-identities.json",
            {
                "session": session,
                "config": config,
                "device": device,
                "environment": environment,
                "installed": installed,
                "runtime": runtime,
                "corpora": inventories,
                "engines": engines,
            },
        )
        stage_root = Path(tempfile.mkdtemp(prefix="stage-", dir=output))
        stages = {}
        for corpus in config["corpora"]:
            stage = stage_root / corpus["name"]
            stage.mkdir()
            stages[corpus["name"]] = stage
            for index, graph in enumerate(inventories[corpus["name"]]):
                doc = read_json(graph["source_path"])
                doc["name"] = graph["graph_name"]
                atomic_json(stage / f"{index:08d}.json", doc)
        records = []
        expected = []

        def phase(kind, corpus, arm, number=0, required=(), resume=True):
            key = phase_key(kind, corpus, arm, number)
            expected.append(key)
            inventory = inventories[corpus["name"]]
            fingerprint = phase_fingerprint(
                config,
                key,
                inventory,
                installed[arm["name"]],
                runtime[arm["name"]],
                device,
                environment,
                required,
            )
            sidecar = output / (phase_tag(key) + ".complete.json")
            record = (
                resume_phase(
                    sidecar,
                    fingerprint,
                    key,
                    config,
                    arm,
                    inventory,
                    engines[arm["name"]],
                    required,
                )
                if resume
                else None
            )
            if record is None:

                def stable():
                    return (
                        corpus_inventory(corpus, config["exclude_tensors"]) == inventory
                        and tree_identity(arm["install_tree"]) == installed[arm["name"]]
                        and phase_fingerprint(
                            config,
                            key,
                            inventory,
                            installed[arm["name"]],
                            runtime[arm["name"]],
                            device,
                            environment,
                            required,
                        )
                        == fingerprint
                    )

                record = run_phase(
                    config,
                    key,
                    corpus,
                    arm,
                    inventory,
                    engines[arm["name"]],
                    stages[corpus["name"]],
                    fingerprint,
                    original,
                    session,
                    required,
                    stable,
                )
            else:
                print(
                    f"{phase_tag(key)}: SKIP (all current-input gates rechecked)",
                    flush=True,
                )
            records.append(record)
            return record

        if config["warmup_arm"] is not None:
            warmup = next(
                a for a in config["arms"] if a["name"] == config["warmup_arm"]
            )
            for corpus in config["corpora"]:
                if not phase("warmup", corpus, warmup, resume=False)["success"]:
                    atomic_json(
                        output / "summary.json",
                        {
                            "validated_complete": False,
                            "session": session,
                            **report(config, records, installed),
                        },
                    )
                    return 1
        for number in range(1, config["rounds"] + 1):
            for corpus in config["corpora"]:
                for arm in config["arms"]:
                    phase("timing", corpus, arm, number)
        if config["correctness"]["enabled"]:
            for corpus in config["corpora"]:
                for arm in config["arms"]:
                    required = {
                        name
                        for r in records
                        if r["phase"]["kind"] == "timing"
                        and r["phase"]["corpus"] == corpus["name"]
                        and r["phase"]["arm"] == arm["name"]
                        for name in r["served"]
                    }
                    phase("correctness", corpus, arm, required=required)
        complete = len(records) == len(expected) and {
            canonical(r["phase"]) for r in records if r["success"]
        } == {canonical(k) for k in expected}
        ledger = [entry for record in records for entry in record["ledger"]]
        atomic_json(output / "outcomes.json", ledger)
        summary = {
            "validated_complete": complete and config["correctness"]["enabled"],
            "timing_only_complete": complete and not config["correctness"]["enabled"],
            "session": session,
            "expected_phases": expected,
            "single_session_cohort": len(
                {r["session"]["id"] for r in records if r["phase"]["kind"] == "timing"}
            )
            == 1,
            "proof_boundary": "Phase gates are not the engine-pinned device integration test; resumed cohorts are not single-job comparisons.",
            **report(config, records, installed),
        }
        atomic_json(output / "summary.json", summary)
        return 0 if complete else 1
    finally:
        if stage_root is not None:
            shutil.rmtree(stage_root)
        shutil.rmtree(lock)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Declarative sweep YAML; relative paths use its directory",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        result = execute(config)
    except ConfigError as exc:
        print(f"INVALID CONFIG: {exc}", file=sys.stderr)
        return 2
    except GateError as exc:
        # A gate declined: the sweep ran, and the measurement it did not produce is
        # an ordinary outcome the caller reads off the marker.
        print(f"SWEEP_INCOMPLETE: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        # An operational failure is NOT an incomplete sweep. A KeyError or TypeError
        # is a defect in this driver and an OSError is a broken execution host;
        # rendering either as SWEEP_INCOMPLETE reads as a measured decline and lets
        # a harness carry on as though the corpus had simply been gated out.
        print(f"SWEEP ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("SWEEP ERROR: interrupted before the sweep could decide", file=sys.stderr)
        return 2
    print(
        "SWEEP_INCOMPLETE"
        if result
        else "SWEEP_DONE" if config["correctness"]["enabled"] else "SWEEP_TIMING_ONLY"
    )
    return result


if __name__ == "__main__":
    sys.exit(main())
