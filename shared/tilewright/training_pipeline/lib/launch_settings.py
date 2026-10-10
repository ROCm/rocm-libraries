# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""How the benches launched the kernels they timed, recorded with the
labels.

TensileLite decides at every launch how the kernel a bench times runs: its
workgroup mapping and chiplet remap, the stagger of its K loop, the
persistent grid, the Stream-K reduction and hybrid schedule, and split-K,
some of them with Origami's heuristics. A label therefore means "this kernel,
launched as the bench build launches it", and three things decide that launch
besides the kernel and the problem:

  overrides   the launch overrides (lib/bench_env.py `is_launch_override`)
              in the bench's environment: none unless the config's
              runtime.env sets them; `stripped` names those the stage's own
              environment had and the bench did not get
  heuristic   the launch-rule sources the bench build compiled
              (`heuristic`): the TensileLite and Origami files of
              LAUNCH_RULE_FILES, found through the build's
              compile_commands.json (else the source directory its
              CMakeCache.txt names), each with its sha256, and `digest`, a
              sha256 over those; `newer_than_build` names the files changed
              since the build compiled its library
  workspace   the workspace the bench lines pass (`workspace_bytes`), which
              caps Stream-K partials and so decides the grid of some
              launches

`bench_record` is what stages 3 and 7 write into platform.json (`launch`);
`read_record` reads it back next to a bench's logs, and `summarize` merges
records (or summaries) into what stages 4, 5 and 6 carry: every heuristic
digest, override and workspace their data was benched with, and how many
benches recorded no heuristic or one whose files changed after the build
(`comparable_digest`). Different digests mean that the launch-rule sources
differ, not necessarily the launches.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .bench_env import is_launch_override, launch_overrides

# (component, path under the component's source root). The TensileLite
# files hold the auto mapping, stagger, split-K, persistent grid, Stream-K
# reduction and hybrid choice and the defaults of the launch overrides; the
# Origami files the heuristics TensileLite calls and the hardware model they
# read. DefaultPersistentLaunch.hpp exists only in builds that factor the
# persistent launch rule out.
LAUNCH_RULE_FILES: Tuple[Tuple[str, str], ...] = (
    ("tensilelite", "src/ContractionSolution.cpp"),
    ("tensilelite", "include/Tensile/ContractionSolution.hpp"),
    ("tensilelite", "include/Tensile/ContractionProblem.hpp"),
    ("tensilelite", "include/Tensile/AMDGPU.hpp"),
    ("tensilelite", "src/AMDGPU.cpp"),
    ("tensilelite", "include/Tensile/DefaultPersistentLaunch.hpp"),
    ("origami", "src/origami/origami.cpp"),
    ("origami", "src/origami/streamk.cpp"),
    ("origami", "src/origami/heuristics.cpp"),
    ("origami", "src/origami/hardware.cpp"),
    ("origami", "include/origami/origami.hpp"),
    ("origami", "include/origami/streamk.hpp"),
    ("origami", "include/origami/heuristics.hpp"),
    ("origami", "include/origami/hardware.hpp"),
)
# A compiled source by which a build's compile_commands.json locates each
# component's source root, and how many directories up the root is.
_ROOT_MARKERS = {
    "tensilelite": ("tensilelite/src/ContractionSolution.cpp", 2),
    "origami": ("origami/src/origami/streamk.cpp", 3),
}
# The build's compiled library of each component, by glob under the build
# directory, the hipBLASLt library as the fallback.
_BUILT = {
    "tensilelite": "tensilelite/libtensilelite-host.so*",
    "origami": "origami/liborigami.a",
}
_HIPBLASLT_LIBRARY = "library/libhipblaslt.so*"

SCHEMA_FILE = "hipblaslt_common.yaml"
WORKSPACE_KEY = "user_allocated_workspace"
_WORKSPACE_LINE_RE = re.compile(rf"(?<![\w]){WORKSPACE_KEY}:\s*(\d+)")
_WORKSPACE_DEFAULT_RE = re.compile(rf"^\s+{WORKSPACE_KEY}:\s*(\d+)\s*$", re.MULTILINE)
_READ_BYTES = 1 << 20
# Summaries next to a bench's logs/ that hold its launch record: stage03's
# platform record and stage03b's summary of its merged logs.
RECORD_FILES = ("platform.json", "rebench_summary.json")
RECORD_KEY = "launch"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(_READ_BYTES), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def build_dir_of(
    bench_binary: Optional[Path], cfg: Optional[Mapping[str, Any]] = None
) -> Optional[Path]:
    """The hipBLASLt build a bench belongs to: the build whose `clients/`
    holds `bench_binary` (through symlinks); for a wrapper elsewhere, such
    as an execution context's, the config's `runtime.build_dir`, else
    `<runtime.rocm_libraries_root>/projects/hipblaslt/build/release`; None
    when neither names one."""
    if bench_binary is not None:
        real = Path(bench_binary).resolve()
        build = real.parent.parent
        if real.parent.name == "clients" and (build / "CMakeCache.txt").is_file():
            return build
    runtime = (cfg or {}).get("runtime") or {}
    if runtime.get("build_dir"):
        return Path(str(runtime["build_dir"]))
    if runtime.get("rocm_libraries_root"):
        root = Path(str(runtime["rocm_libraries_root"]))
        return root / "projects" / "hipblaslt" / "build" / "release"
    return None


def _roots_from_compile_commands(path: Path) -> Dict[str, Path]:
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    out: Dict[str, Path] = {}
    for e in entries if isinstance(entries, list) else ():
        f = str(e.get("file") or "") if isinstance(e, dict) else ""
        for comp, (suffix, up) in _ROOT_MARKERS.items():
            if comp not in out and f.replace("\\", "/").endswith("/" + suffix):
                out[comp] = Path(f).parents[up - 1]
    return out


def _roots_from_cmake_cache(path: Path) -> Dict[str, Path]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return {}
    m = re.search(r"^hipblaslt_SOURCE_DIR:\w+=(.+)$", text, re.MULTILINE)
    if m is None:
        return {}
    src = Path(m.group(1).strip())
    return {
        "tensilelite": src / "tensilelite",
        "origami": src.parent.parent / "shared" / "origami",
    }


def source_roots(build_dir: Path) -> Tuple[Dict[str, Path], Optional[str]]:
    """({component: its source root}, the build file that named them) of
    the components of LAUNCH_RULE_FILES the build compiled; ({}, None) when
    neither file names them."""
    for name, read in (
        ("compile_commands.json", _roots_from_compile_commands),
        ("CMakeCache.txt", _roots_from_cmake_cache),
    ):
        roots = read(Path(build_dir) / name)
        roots = {
            c: r
            for c, r in roots.items()
            if (r / _ROOT_MARKERS[c][0].split("/", 1)[1]).is_file()
        }
        if len(roots) == len(_ROOT_MARKERS):
            return roots, name
    return {}, None


def _built_mtime(build_dir: Path, component: str) -> Optional[float]:
    for pattern in (_BUILT[component], _HIPBLASLT_LIBRARY):
        times = [p.stat().st_mtime for p in Path(build_dir).glob(pattern)]
        if times:
            return max(times)
    return None


def heuristic(build_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    """The launch-rule sources of the build in `build_dir` (see the module
    docstring): `digest`, `files` ({component/path: sha256, or None for a
    file the source tree lacks}), `found_by` and `newer_than_build`; None
    when the build names no source tree holding them."""
    if build_dir is None:
        return None
    roots, found_by = source_roots(Path(build_dir))
    if not roots:
        return None
    files: Dict[str, Optional[str]] = {}
    newer: List[str] = []
    built = {c: _built_mtime(Path(build_dir), c) for c in roots}
    for comp, rel in LAUNCH_RULE_FILES:
        path = roots[comp] / rel
        name = f"{comp}/{rel}"
        if not path.is_file():
            files[name] = None
            continue
        files[name] = _sha256(path)
        if built[comp] is not None and path.stat().st_mtime > built[comp]:
            newer.append(name)
    h = hashlib.sha256()
    for name, digest in files.items():
        h.update(f"{name}\t{digest or '-'}\n".encode())
    return {
        "digest": "sha256:" + h.hexdigest(),
        "files": files,
        "found_by": found_by,
        "newer_than_build": newer,
    }


def schema_workspace_bytes(
    bench_binary: Optional[Path], build_dir: Optional[Path]
) -> Optional[int]:
    """The workspace a bench line without `user_allocated_workspace` passes:
    the default of the bench's argument schema (hipblaslt_common.yaml next
    to the bench, or in the build's `clients/`); None without one."""
    dirs: List[Path] = []
    if bench_binary is not None:
        dirs += [Path(bench_binary).parent, Path(bench_binary).resolve().parent]
    if build_dir is not None:
        dirs.append(Path(build_dir) / "clients")
    for d in dirs:
        try:
            text = (d / SCHEMA_FILE).read_text(errors="replace")
        except OSError:
            continue
        m = _WORKSPACE_DEFAULT_RE.search(text)
        if m is not None:
            return int(m.group(1))
    return None


def workspace_bytes(lines: Iterable[str], default: Optional[int]) -> Any:
    """The workspace the bench `lines` pass: their `user_allocated_workspace`,
    `default` for a line without one; one number when they agree, else the
    sorted numbers; None when none is known."""
    values = set()
    for line in lines:
        m = _WORKSPACE_LINE_RE.search(line)
        if m is not None:
            values.add(int(m.group(1)))
        elif default is not None:
            values.add(int(default))
    if not values:
        return None if default is None else int(default)
    return values.pop() if len(values) == 1 else sorted(values)


def bench_record(
    bench_binary: Optional[Path],
    cfg: Optional[Mapping[str, Any]],
    env: Mapping[str, str],
    environ: Mapping[str, str],
    lines: Sequence[str] = (),
) -> Dict[str, Any]:
    """The launch record of benches run as `bench_binary` with environment
    `env` from a stage whose environment is `environ`, on bench `lines`."""
    build = build_dir_of(bench_binary, cfg)
    return {
        "overrides": launch_overrides(env),
        "stripped": sorted(
            k for k in environ if is_launch_override(k) and k not in env
        ),
        "heuristic": heuristic(build),
        "workspace_bytes": workspace_bytes(
            lines, schema_workspace_bytes(bench_binary, build)
        ),
    }


def read_record(log_dir: Path) -> Optional[Dict[str, Any]]:
    """The launch record (or summary) a bench left next to its logs
    (`log_dir` is `<stage dir>/logs`, its summary one of RECORD_FILES), or
    None."""
    for name in RECORD_FILES:
        try:
            doc = json.loads((Path(log_dir).parent / name).read_text())
        except (OSError, ValueError):
            continue
        rec = doc.get(RECORD_KEY) if isinstance(doc, dict) else None
        if isinstance(rec, dict):
            return rec
    return None


def _workspaces(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, list):
        return [int(v) for v in value]
    return [int(value)]


def comparable_digest(heuristic: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The digest of a `heuristic`, None when it is unknown: no heuristic,
    or launch-rule files changed since the build compiled them
    (`newer_than_build`), which the digest describes as they are now."""
    if not isinstance(heuristic, Mapping) or not heuristic.get("digest"):
        return None
    if heuristic.get("newer_than_build"):
        return None
    return str(heuristic["digest"])


def digests(item: Optional[Mapping[str, Any]]) -> List[str]:
    """The heuristic digests of a record or summary; [] for None or a
    record whose heuristic is unknown (`comparable_digest`)."""
    if item is None:
        return []
    if "heuristics" in item:
        return [str(d) for d in item.get("heuristics") or ()]
    digest = comparable_digest(item.get("heuristic"))
    return [digest] if digest else []


def summarize(items: Iterable[Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    """What a set of benches ran with, from their records (`bench_record`),
    summaries (this function's output) or None for a bench that recorded
    none: `heuristics` (the distinct digests), `overrides` ({name: its
    distinct values}), `workspace_bytes` (the distinct sizes) and `unknown`
    (benches whose heuristic digest is unknown: `comparable_digest`)."""
    found: set = set()
    overrides: Dict[str, set] = {}
    workspaces: set = set()
    unknown = 0
    for item in items:
        if item is None:
            unknown += 1
            continue
        found.update(digests(item))
        if "heuristics" in item:
            unknown += int(item.get("unknown") or 0)
            for name, values in (item.get("overrides") or {}).items():
                overrides.setdefault(name, set()).update(values)
        else:
            unknown += 0 if digests(item) else 1
            for name, value in (item.get("overrides") or {}).items():
                overrides.setdefault(name, set()).add(str(value))
        workspaces.update(_workspaces(item.get("workspace_bytes")))
    return {
        "heuristics": sorted(found),
        "overrides": {k: sorted(v) for k, v in sorted(overrides.items())},
        "workspace_bytes": sorted(workspaces),
        "unknown": unknown,
    }


def short(digest: Optional[str]) -> str:
    """A heuristic digest as messages name it: its first 16 hex digits."""
    return str(digest or "none").split(":", 1)[-1][:16]


def describe_digests(item: Optional[Mapping[str, Any]]) -> str:
    """The heuristic digests of a record or summary, short, for messages."""
    return ", ".join(short(d) for d in digests(item)) or "unknown"


def describe(summary: Mapping[str, Any]) -> str:
    """A summary in one line, for logs."""
    parts = ["launch heuristic " + describe_digests(summary)]
    if summary.get("unknown"):
        parts.append(f"{summary['unknown']} bench(es) with an unknown heuristic")
    overrides = summary.get("overrides") or {}
    parts.append(
        "overrides "
        + (
            ", ".join(f"{k}={'|'.join(v)}" for k, v in overrides.items())
            if overrides
            else "none"
        )
    )
    ws = summary.get("workspace_bytes") or []
    parts.append(
        "workspace " + (", ".join(str(w) for w in ws) if ws else "unknown") + " B"
    )
    return "; ".join(parts)
