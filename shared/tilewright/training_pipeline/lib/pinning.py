# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""What a run is produced with, so that it never changes silently.

A pin has three parts (`PARTS`):

    code    the digest of the pipeline's Python files that produce a run's
            data and models (run_orchestrator.py, lib/ and stages/ without
            the stages that only validate a run, `EXCLUDED`), each file's
            own digest (`files`, so that a change names its files), the
            revision of the checkout holding them and whether its tilewright
            tree has uncommitted changes (both informational)
    engine  the tilewright module the stages import: its path, the digest of
            its files and the feature catalogs it computes
    config  the fingerprint of the config's pinned values, every key at the
            value the stages apply (`values`)

`<run dir>/pin.json` holds the pin the run was produced with, the stage whose
completion recorded it, and the history of the changes invocations accepted.
The orchestrator compares an invocation's pin with it before any stage runs
(`differences`), and its code and engine again before every stage (`drift`),
since stages start from the files on disk; which parts an invocation must
keep is the orchestrator's choice. The revision, the dirty flag and the
engine's path are recorded but not compared: the digests are what the stages
run. A file that cannot be read enters a digest as unreadable, so that it
reads as a change rather than an error.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .fs import write_json

RUN_PIN = "pin.json"
PARTS = ("code", "engine", "config")
CODE_DIRS = ("lib", "stages")
CODE_FILES = ("run_orchestrator.py",)
# Stages whose outputs are validation reports only: no data or model of a
# run comes from them, so a fix to one never needs --allow-code-change.
EXCLUDED = (
    "stages/stage07_optional_validate_selection_time.py",
    "stages/stage08_optional_validate_selection_efficiency.py",
)
ENGINE_MODULE = "tilewright"
GIT_TIMEOUT_S = 20.0
CHANGED_FILES_SHOWN = 8


def _file_digest(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as e:
        return f"unreadable ({type(e).__name__})"


def _digest(files: Mapping[str, str]) -> str:
    """One digest over {relative path: file digest}."""
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(files[rel].encode())
        h.update(b"\0")
    return h.hexdigest()


def _files_under(base: Path, suffix: str = "") -> List[Path]:
    """Regular files below `base` ending in `suffix`, outside __pycache__;
    an editor's dangling lock link is no file."""
    out: List[Path] = []
    for root, dirs, names in os.walk(base):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [Path(root) / n for n in sorted(names) if n.endswith(suffix)]
    return [p for p in out if p.is_file()]


def code_files(pipeline_dir: Path) -> List[Path]:
    pipeline_dir = Path(pipeline_dir)
    out = [pipeline_dir / f for f in CODE_FILES if (pipeline_dir / f).is_file()]
    for d in CODE_DIRS:
        out += _files_under(pipeline_dir / d, ".py")
    return [p for p in out if p.relative_to(pipeline_dir).as_posix() not in EXCLUDED]


def code_file_digests(pipeline_dir: Path) -> Dict[str, str]:
    """{path relative to `pipeline_dir`: digest} of the code part's files."""
    base = Path(pipeline_dir)
    return {p.relative_to(base).as_posix(): _file_digest(p) for p in code_files(base)}


def code_digest(pipeline_dir: Path) -> str:
    return _digest(code_file_digests(pipeline_dir))


def _git(args: List[str], cwd: Path) -> Optional[str]:
    env = dict(
        os.environ,
        GIT_OPTIONAL_LOCKS="0",
        GIT_NO_LAZY_FETCH="1",
        GIT_ALLOW_PROTOCOL="file",
        GIT_TERMINAL_PROMPT="0",
    )
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def code_state(pipeline_dir: Path) -> Dict[str, Any]:
    """The digest and file digests of the code part as it is on disk."""
    files = code_file_digests(pipeline_dir)
    return {"digest": _digest(files), "files": files}


def code_record(pipeline_dir: Path) -> Dict[str, Any]:
    """The code part of a pin; `revision` and `dirty` are None outside a git
    checkout."""
    pipeline_dir = Path(pipeline_dir).resolve()
    revision = _git(["rev-parse", "HEAD"], pipeline_dir)
    status = _git(
        [
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            str(pipeline_dir.parent),
        ],
        pipeline_dir,
    )
    return {
        **code_state(pipeline_dir),
        "revision": revision.strip() if revision else None,
        "dirty": None if status is None else bool(status.strip()),
    }


def engine_files() -> Optional[Dict[str, Any]]:
    """{"path", "sha256"} of the engine module the stages import (the one this
    interpreter finds); None when there is none."""
    try:
        spec = importlib.util.find_spec(ENGINE_MODULE)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    origin = Path(spec.origin)
    if spec.submodule_search_locations:
        base = origin.parent
        files = _files_under(base)
    else:
        base, files = origin.parent, [origin]
    digests = {p.relative_to(base).as_posix(): _file_digest(p) for p in files}
    return {"path": str(base), "sha256": _digest(digests)}


def engine_record() -> Dict[str, Any]:
    """The engine part of a pin."""
    files = engine_files() or {"path": None, "sha256": None}
    catalogs: Optional[List[str]] = None
    try:
        module = importlib.import_module(ENGINE_MODULE)
        names = getattr(module, "feature_catalog_hashes", None)
        if callable(names):
            catalogs = sorted(str(h) for h in names())
        else:
            one = getattr(module, "feature_catalog_hash", None)
            catalogs = [str(one())] if callable(one) else None
    except Exception:
        catalogs = None
    return {**files, "catalogs": catalogs}


def config_record(values: Mapping[str, Any]) -> Dict[str, Any]:
    """The config part of a pin over its pinned `values` ({dotted key:
    value})."""
    text = json.dumps(dict(values), sort_keys=True, separators=(",", ":"))
    return {
        "fingerprint": hashlib.sha256(text.encode()).hexdigest(),
        "values": dict(values),
    }


def make_pin(
    pipeline_dir: Path, config_values: Optional[Mapping[str, Any]]
) -> Dict[str, Any]:
    pin: Dict[str, Any] = {
        "code": code_record(pipeline_dir),
        "engine": engine_record(),
    }
    if config_values is not None:
        pin["config"] = config_record(config_values)
    return pin


def summary(pin: Mapping[str, Any]) -> Dict[str, Any]:
    """The pin without the config's values and the code's file digests, as
    stage markers record it."""
    out = {k: dict(v) for k, v in pin.items() if isinstance(v, Mapping)}
    if "config" in out:
        out["config"] = {"fingerprint": out["config"].get("fingerprint")}
    if "code" in out:
        out["code"].pop("files", None)
    return out


def _short(value: Any, width: int = 60) -> str:
    text = json.dumps(value, sort_keys=True)
    return text if len(text) <= width else text[: width - 3] + "..."


def _code_text(code: Mapping[str, Any]) -> str:
    rev = code.get("revision")
    where = f"revision {rev[:12]}" if rev else "no git revision"
    if code.get("dirty"):
        where += " with uncommitted changes"
    return f"{str(code.get('digest'))[:16]} ({where})"


def _changed_files(old: Mapping[str, Any], new: Mapping[str, Any]) -> str:
    """The files whose digests differ between two code parts, when both
    record them."""
    a, b = old.get("files"), new.get("files")
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return ""
    names = []
    for rel in sorted(set(a) | set(b)):
        if rel not in a:
            names.append(f"{rel} (new)")
        elif rel not in b:
            names.append(f"{rel} (gone)")
        elif a[rel] != b[rel]:
            names.append(rel + (f" ({b[rel]})" if b[rel].startswith("unread") else ""))
    more = len(names) - CHANGED_FILES_SHOWN
    shown = ", ".join(names[:CHANGED_FILES_SHOWN])
    return f"; files: {shown}" + (f" and {more} more" if more > 0 else "")


def differences(
    old: Mapping[str, Any], new: Mapping[str, Any], parts: Iterable[str] = PARTS
) -> List[str]:
    """What `new` changes against `old` in `parts`, one line each; config
    values are compared only when both pins have them."""
    parts = set(parts)
    out: List[str] = []
    oc, nc = old.get("code") or {}, new.get("code") or {}
    if "code" in parts and oc.get("digest") != nc.get("digest"):
        out.append(
            f"pipeline code: {_code_text(oc)} -> {_code_text(nc)}"
            + _changed_files(oc, nc)
        )
    oe, ne = old.get("engine") or {}, new.get("engine") or {}
    if "engine" in parts and oe.get("sha256") != ne.get("sha256"):
        out.append(
            f"engine module: {str(oe.get('sha256'))[:16]} ({oe.get('path')}) -> "
            f"{str(ne.get('sha256'))[:16]} ({ne.get('path')})"
        )
    if "engine" in parts and oe.get("catalogs") != ne.get("catalogs"):
        out.append(
            f"engine feature catalogs: {oe.get('catalogs')} -> {ne.get('catalogs')}"
        )
    if "config" in parts and "config" in old and "config" in new:
        ov = old["config"].get("values") or {}
        nv = new["config"].get("values") or {}
        for key in sorted(set(ov) | set(nv)):
            if ov.get(key, "<unset>") != nv.get(key, "<unset>"):
                out.append(
                    f"config {key}: {_short(ov.get(key, '<unset>'))} -> "
                    f"{_short(nv.get(key, '<unset>'))}"
                )
    return out


def refreshed(pin: Mapping[str, Any], pipeline_dir: Path) -> Dict[str, Any]:
    """`pin` with its code and engine files as they are on disk now."""
    out = dict(pin)
    out["code"] = dict(pin.get("code") or {}, **code_state(pipeline_dir))
    out["engine"] = dict(
        pin.get("engine") or {}, **(engine_files() or {"path": None, "sha256": None})
    )
    return out


def drift(
    pin: Mapping[str, Any], pipeline_dir: Path, parts: Iterable[str] = PARTS
) -> List[str]:
    """How the code and engine files on disk differ from `pin` now, in
    `parts`."""
    return differences(pin, refreshed(pin, pipeline_dir), set(parts) - {"config"})


def describe(pin: Mapping[str, Any]) -> str:
    """The pin in one line, for the config snapshot's head."""
    engine = pin.get("engine") or {}
    text = (
        f"pipeline code {_code_text(pin.get('code') or {})}; engine module "
        f"{engine.get('path') or 'not found'} (sha256 "
        f"{str(engine.get('sha256'))[:16]}, catalogs {engine.get('catalogs')})"
    )
    config = pin.get("config")
    if config:
        text += f"; pinned config {str(config.get('fingerprint'))[:16]}"
    return text


def read_run_pin(run_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((Path(run_dir) / RUN_PIN).read_text())
    except (OSError, ValueError):
        return None
    return (
        data if isinstance(data, dict) and isinstance(data.get("pin"), dict) else None
    )


def write_run_pin(
    run_dir: Path,
    pin: Mapping[str, Any],
    *,
    previous: Optional[Mapping[str, Any]] = None,
    accepted: Optional[List[str]] = None,
    first_stage: Optional[str] = None,
) -> None:
    """Record `pin` as the run's, first when `first_stage` (the run's
    stage, relative to it) completes: a run resumed from before pins were
    recorded is pinned from that stage on. With `accepted`, the changes an
    invocation accepted from `previous` (the earlier pin.json) join its
    history."""
    now = datetime.now().isoformat(timespec="seconds")
    history = list((previous or {}).get("history") or [])
    if accepted:
        history.append(
            {
                "accepted": now,
                "changes": list(accepted),
                "previous": summary((previous or {}).get("pin") or {}),
            }
        )
    rec: Dict[str, Any] = {"pin": dict(pin), "recorded": now}
    first = first_stage or (previous or {}).get("first_stage")
    if first:
        rec["first_stage"] = first
    write_json(Path(run_dir) / RUN_PIN, {**rec, "history": history})
