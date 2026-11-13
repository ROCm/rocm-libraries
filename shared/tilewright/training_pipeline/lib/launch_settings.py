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
  workspace   the workspace limit of the bench lines (`workspace_bytes`,
              hipblaslt-bench's `user_allocated_workspace`): the bench gives
              every kernel of a line the largest workspace the line's
              kernels ask for within it, none when none asks for one, or,
              with `workspace_per_process` (launch rule version 3), every
              launch the whole limit from one buffer per bench process; the
              workspace a launch is given caps its Stream-K partials and so
              decides the grid of some launches
  summation   the uniform summation order the benches launch with
              (`uniform_summation_order`): the value the lines pass (-1,
              hipBLASLt's default, leaves it to the handle), the handle's
              default in the build's hipBLASLt sources and the resulting
              `effective` mode, "on" or "off"; TensileLite then steers the
              grid of static Stream-K launches and leaves out kernels that
              cannot launch uniformly, which lib/hipblaslt_launch.py models
              from rule version 3 on
  rule        the launch rule version the run trains on (`launch_rule`)

`bench_record` is what stages 3 and 7 write into platform.json (`launch`);
`read_record` reads it back next to a bench's logs, and `summarize` merges
records (or summaries) into what stages 4, 5 and 6 carry: every heuristic
digest, override and workspace limit their data was benched with, and how many
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
# persistent launch rule out. The library headers hold which kernels a call
# under the uniform summation order may launch.
LAUNCH_RULE_FILES: Tuple[Tuple[str, str], ...] = (
    ("tensilelite", "src/ContractionSolution.cpp"),
    ("tensilelite", "include/Tensile/ContractionSolution.hpp"),
    ("tensilelite", "include/Tensile/ContractionProblem.hpp"),
    ("tensilelite", "include/Tensile/AMDGPU.hpp"),
    ("tensilelite", "src/AMDGPU.cpp"),
    ("tensilelite", "include/Tensile/DefaultPersistentLaunch.hpp"),
    ("tensilelite", "include/Tensile/SolutionLibrary.hpp"),
    ("tensilelite", "include/Tensile/PredictionLibrary.hpp"),
    ("tensilelite", "include/Tensile/ContractionTaskPredicates.hpp"),
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
PROCESS_WORKSPACE_KEY = "workspace_per_process"
_PROCESS_WORKSPACE_RE = re.compile(rf"(?<![\w]){PROCESS_WORKSPACE_KEY}:\s*(\w+)")
SUMMATION_ORDER_KEY = "uniform_summation_order"
_ORDER_LINE_RE = re.compile(rf"(?<![\w]){SUMMATION_ORDER_KEY}:\s*(-?\d+)")
_ORDER_DEFAULT_RE = re.compile(
    rf"^\s+{SUMMATION_ORDER_KEY}:\s*(-?\d+)\s*$", re.MULTILINE
)
# The hipBLASLt source that holds the handle's default uniform summation
# order, which a call leaving it unset (-1) inherits, relative to the
# hipBLASLt source root.
HANDLE_HEADER = "library/src/amd_detail/rocblaslt/src/include/handle.h"
_HANDLE_ORDER_RE = re.compile(rf"\b{SUMMATION_ORDER_KEY}\s*=\s*(-?\d+)\s*;")
# The hipBLASLt source whose handle constructor turns that default off on the
# architectures of contexts.SUMMATION_ORDER_OFF_ARCHS; a hipBLASLt whose
# constructor does not keeps the header's default on every architecture.
HANDLE_SOURCE = "library/src/amd_detail/rocblaslt/src/handle.cpp"
_HANDLE_OFF_ARCHS_RE = re.compile(
    r'find\("gfx11"\)[^;]*find\("gfx120"\)[^;]*'
    rf"\b{SUMMATION_ORDER_KEY}\s*=\s*0\s*;"
)
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


def _schema_dirs(bench_binary: Optional[Path], build_dir: Optional[Path]) -> List[Path]:
    """Where the argument schema of a bench can be: next to it, next to the
    binary it links to, and in the build's `clients/`."""
    dirs: List[Path] = []
    if bench_binary is not None:
        dirs += [Path(bench_binary).parent, Path(bench_binary).resolve().parent]
    if build_dir is not None:
        dirs.append(Path(build_dir) / "clients")
    return dirs


def schema_workspace_bytes(
    bench_binary: Optional[Path], build_dir: Optional[Path]
) -> Optional[int]:
    """The workspace limit of a bench line without `user_allocated_workspace`:
    the default of the bench's argument schema (hipblaslt_common.yaml next
    to the bench, or in the build's `clients/`); None without one."""
    for d in _schema_dirs(bench_binary, build_dir):
        try:
            text = (d / SCHEMA_FILE).read_text(errors="replace")
        except OSError:
            continue
        m = _WORKSPACE_DEFAULT_RE.search(text)
        if m is not None:
            return int(m.group(1))
    return None


def handle_summation_order(
    build_dir: Optional[Path], arch: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """The handle-level uniform summation order of the hipBLASLt build in
    `build_dir` on `arch`, from the `_rocblaslt_handle` of HANDLE_HEADER in
    the source tree it compiled (`source_roots`), turned off on an arch the
    constructor in HANDLE_SOURCE turns it off on: `default`, 0 or 1, None
    for a handle without one (a hipBLASLt from before the mode, which runs
    without it), and `newer_than_build`, whether either source changed since
    the build compiled its library; None when the build names no source
    tree."""
    if build_dir is None:
        return None
    roots, _ = source_roots(Path(build_dir))
    if "tensilelite" not in roots:
        return None
    path = roots["tensilelite"].parent / HANDLE_HEADER
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return None
    start = text.find("struct _rocblaslt_handle")
    body = text[start : text.find("\n};", start)] if start >= 0 else ""
    m = _HANDLE_ORDER_RE.search(body)
    default = int(m.group(1)) if m else None
    built = _built_mtime(Path(build_dir), "tensilelite")
    newer = built is not None and path.stat().st_mtime > built
    source = roots["tensilelite"].parent / HANDLE_SOURCE
    try:
        constructor = source.read_text(errors="replace")
        newer = newer or (built is not None and source.stat().st_mtime > built)
    except OSError:
        constructor = ""
    from .contexts import production_summation_order

    if default and arch and not production_summation_order(arch):
        if _HANDLE_OFF_ARCHS_RE.search(constructor):
            default = 0
    return {"default": default, "newer_than_build": newer}


def summation_order(
    lines: Sequence[str],
    bench_binary: Optional[Path],
    build_dir: Optional[Path],
    arch: Optional[str] = None,
) -> Dict[str, Any]:
    """The uniform summation order the bench `lines` launch with on `arch`:
    `lines`, the value they pass (their SUMMATION_ORDER_KEY, else the
    default of the bench's argument schema, else -1; the sorted values when
    they differ), `handle`, the build's `handle_summation_order` default on
    `arch`, and `effective`, "on" or "off" (the sorted modes when the lines
    differ), None when it depends on a handle default that is unknown or
    newer than the build."""
    default = -1
    for d in _schema_dirs(bench_binary, build_dir):
        try:
            m = _ORDER_DEFAULT_RE.search((d / SCHEMA_FILE).read_text(errors="replace"))
        except OSError:
            continue
        if m is not None:
            default = int(m.group(1))
            break
    values = set()
    for line in lines:
        m = _ORDER_LINE_RE.search(line)
        values.add(int(m.group(1)) if m else default)
    values = values or {default}
    handle = handle_summation_order(build_dir, arch)
    inherited: Optional[str] = None
    if handle is not None and not handle["newer_than_build"]:
        inherited = "on" if handle["default"] else "off"
    modes = {inherited if v == -1 else ("on" if v else "off") for v in values}
    effective: Any = None if None in modes else sorted(modes)
    return {
        "lines": values.pop() if len(values) == 1 else sorted(values),
        "handle": None if handle is None else handle["default"],
        "effective": effective[0] if effective and len(effective) == 1 else effective,
    }


def workspace_bytes(lines: Iterable[str], default: Optional[int]) -> Any:
    """The workspace limit of the bench `lines`: their
    `user_allocated_workspace`, `default` for a line without one; one number
    when they agree, else the sorted numbers; None when none is known."""
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


def workspace_per_process(lines: Iterable[str]) -> Any:
    """Whether the bench `lines` give every launch their whole workspace
    from one buffer per process (PROCESS_WORKSPACE_KEY true) rather than
    the largest one a line's kernels ask for: one bool when they agree, else
    [False, True]; False without lines."""
    values = set()
    for line in lines:
        m = _PROCESS_WORKSPACE_RE.search(line)
        values.add(m is not None and m.group(1).lower() in ("true", "1"))
    if not values:
        return False
    return values.pop() if len(values) == 1 else sorted(values)


def launch_rule(cfg: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The launch rule version a run of `cfg` trains on, None without a
    config."""
    if not cfg:
        return None
    from .hipblaslt_launch import rule_version

    return rule_version(str(cfg.get("arch") or ""))


def full_workspace_bytes(cfg: Optional[Mapping[str, Any]]) -> Optional[int]:
    """The full workspace the lines of a run of `cfg` give the round's own
    context under launch rule version 3 (lib/contexts.py `full_workspace`);
    None before that version, without a config, or for one that is no
    workspace."""
    if (launch_rule(cfg) or 0) < 3:
        return None
    from .contexts import full_workspace

    try:
        return full_workspace(cfg or {})
    except ValueError:
        return None


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
        "workspace_per_process": workspace_per_process(lines),
        "full_workspace_bytes": full_workspace_bytes(cfg),
        "uniform_summation_order": summation_order(
            lines, bench_binary, build, (cfg or {}).get("arch")
        ),
        "launch_rule": launch_rule(cfg),
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
    distinct values}), `workspace_bytes` (the distinct sizes), `unknown`
    (benches whose heuristic digest is unknown: `comparable_digest`) and,
    when any bench records them, `uniform_summation_order` (the distinct
    effective modes, "unknown" for a bench whose mode is unknown),
    `workspace_per_process` (the distinct mechanisms), `full_workspace_bytes`
    (the distinct full workspaces) and `launch_rules` (the distinct rule
    versions)."""
    found: set = set()
    overrides: Dict[str, set] = {}
    workspaces: set = set()
    orders: set = set()
    mechanisms: set = set()
    fulls: set = set()
    rules: set = set()
    unknown = 0
    for item in items:
        if item is None:
            unknown += 1
            continue
        fulls.update(_workspaces(item.get("full_workspace_bytes")))
        mechanism = item.get("workspace_per_process")
        if isinstance(mechanism, list):
            mechanisms.update(bool(v) for v in mechanism)
        elif isinstance(mechanism, bool):
            mechanisms.add(mechanism)
        rule = item.get("launch_rules", item.get("launch_rule"))
        if isinstance(rule, list):
            rules.update(int(v) for v in rule)
        elif isinstance(rule, int) and not isinstance(rule, bool):
            rules.add(rule)
        order = item.get("uniform_summation_order")
        if isinstance(order, dict):
            order = order.get("effective") or "unknown"
        if isinstance(order, list):
            orders.update(str(v) for v in order)
        elif isinstance(order, str):
            orders.add(order)
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
    out = {
        "heuristics": sorted(found),
        "overrides": {k: sorted(v) for k, v in sorted(overrides.items())},
        "workspace_bytes": sorted(workspaces),
        "unknown": unknown,
    }
    if orders:
        out["uniform_summation_order"] = sorted(orders)
    if mechanisms:
        out["workspace_per_process"] = sorted(mechanisms)
    if fulls:
        out["full_workspace_bytes"] = sorted(fulls)
    if rules:
        out["launch_rules"] = sorted(rules)
    return out


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
        "workspace limit " + (", ".join(str(w) for w in ws) if ws else "unknown") + " B"
    )
    orders = summary.get("uniform_summation_order")
    if orders:
        parts.append("uniform summation order " + ", ".join(orders))
    return "; ".join(parts)
