# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""I/O and safety: the `run://` surface, the flows allow-list, and reading live files.

Three boundaries live here, all of them load-bearing and all of them tested:

* **Run containment.** Every path a client names is resolved against one run
  directory and must still be inside it afterwards. Absolute paths, `..`
  segments and symlink escapes are refused.
* **The flows allow-list.** A caller may name a flow, or a path that stays
  inside the configured flows directory. Anything else is refused, because the
  alternative is pointing the server at an arbitrary YAML that runs arbitrary
  argv.
* **Reading a file that is being rewritten underneath us.** The engine replaces
  its manifest atomically, so a reader gets one whole document or the other --
  provided it opens, reads and closes immediately, and retries once when
  Windows reports the replace holding the handle.

Nothing here knows the name of a flow, a step, an output or an artifact. What a
run directory contains is discovered by walking it.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

from . import schema

#: Files are served as text and capped. A run directory holds evidence -- logs,
#: JSON, source -- and a client that wants all of a very large one is told where
#: it lives on disk instead.
READ_CAP = 1 << 20

#: How long to wait before the single retry of :func:`read_bytes`. `os.replace`
#: is atomic, but on Windows it can fail transiently while a reader holds the
#: destination, and the symmetric failure is a reader losing the race.
RETRY_DELAY_S = 0.05

#: How many run directories `resources/list` will describe beyond the ones this
#: session launched. Older runs stay readable by URI; only the listing is
#: bounded, so a runs tree with thousands of files cannot blow up a response.
LIST_LIMIT = 10

#: Extension-driven and indifferent to what a file means. A future flow writing
#: descriptors, CMake fragments or JUnit reports is typed correctly with no
#: change; an unknown extension is text, which is what everything in a run
#: directory is.
_MIME_BY_SUFFIX = {
    ".json": "application/json",
    ".md": "text/markdown",
    ".hip": "text/x-c++src",
    ".cpp": "text/x-c++src",
    ".cc": "text/x-c++src",
    ".h": "text/x-c++src",
    ".hpp": "text/x-c++src",
    ".cu": "text/x-c++src",
    ".cmake": "text/x-cmake",
    ".log": "text/plain",
    ".txt": "text/plain",
    ".xml": "application/xml",
}

#: Matched before the extension table so the build-system convention wins over
#: the generic `.txt` mapping.
_MIME_BY_NAME = {"CMakeLists.txt": "text/x-cmake"}

DEFAULT_MIME = "text/plain"

#: The engine's own scratch file for the atomic manifest replace. It is the one
#: thing a walk of a run directory must not offer, because it is a fragment of a
#: write that has not landed.
_MANIFEST_TMP = schema.MANIFEST_NAME + ".tmp"


class ResourceError(Exception):
    """A request named something the server will not serve."""


class RunNotFound(ResourceError):
    """No run directory answers to this id."""


class PathRefused(ResourceError):
    """A path resolved outside the directory it was resolved against."""


class FlowNotFound(ResourceError):
    """No flow answers to this name inside the flows directory."""


class FlowRefused(ResourceError):
    """A flow argument pointed outside the flows directory."""


# -- reading ----------------------------------------------------------------


def read_bytes(path: Path) -> bytes:
    """Read a whole file and let go of the handle immediately.

    The manifest is replaced atomically while we read it, so holding a handle is
    what turns a correct design into an intermittent `PermissionError` on
    Windows -- for the writer as much as for us. One retry covers that window;
    a second failure is a real one. Absence is not retried: `os.replace` never
    leaves the destination missing, so a file that is not there will not appear
    in fifty milliseconds.
    """
    for final in (False, True):
        try:
            with open(path, "rb") as handle:
                return handle.read()
        except FileNotFoundError:
            raise
        except OSError:
            if final:
                raise
            time.sleep(RETRY_DELAY_S)
    raise AssertionError("unreachable")


def read_text(path: Path, cap: int = READ_CAP) -> str:
    """File contents as text, truncated at `cap` bytes with a marker line.

    The marker names the absolute path rather than pretending the rest is
    unavailable: everything a run produced stays on disk.
    """
    raw = read_bytes(path)
    if len(raw) <= cap:
        return raw.decode("utf-8", errors="replace")
    head = raw[:cap].decode("utf-8", errors="replace")
    marker = (
        f"[truncated: {cap} of {len(raw)} bytes — read the file at "
        f"{Path(path).resolve()} for the rest]"
    )
    return head + ("" if head.endswith("\n") else "\n") + marker


def read_json(path: Path) -> dict[str, Any] | None:
    """A JSON document, or `None` when it is absent or unreadable.

    Absence is a state the caller reconciles (a worker that died before its
    first checkpoint leaves no manifest at all), not an error to raise through.
    """
    try:
        raw = read_bytes(Path(path))
    except OSError:
        return None
    try:
        document = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return None
    return document if isinstance(document, dict) else None


# -- run directories --------------------------------------------------------


def _is_listable(entry: Path) -> bool:
    """Dot-directories are infrastructure, never runs.

    The supervisor keeps its process records under one, and a scan that mistook
    a record for a run directory would serve process state as a run artifact.
    """
    return entry.is_dir() and not entry.name.startswith(".")


def find_run_dir(run_root: Path | str, run_id: str) -> Path | None:
    """The directory a run id names, found by scanning rather than remembering.

    Runs live at `<run-root>/<flow>/<runId>`. The flow level is scanned rather
    than assumed, because the caller knows the id and nothing else -- that is
    the whole point of an id.
    """
    root = Path(run_root)
    if not run_id or "/" in run_id or "\\" in run_id or run_id.startswith("."):
        return None
    try:
        children = sorted(root.iterdir())
    except OSError:
        return None
    for child in children:
        if not _is_listable(child):
            continue
        candidate = child / run_id
        if candidate.is_dir():
            return candidate.resolve()
    return None


def newest_run_dirs(run_root: Path | str, limit: int = LIST_LIMIT) -> list[Path]:
    """The most recently touched run directories, newest first."""
    root = Path(run_root)
    found: list[tuple[float, Path]] = []
    try:
        children = sorted(root.iterdir())
    except OSError:
        return []
    for child in children:
        if not _is_listable(child):
            continue
        try:
            runs = sorted(child.iterdir())
        except OSError:
            continue
        for run in runs:
            if not _is_listable(run):
                continue
            try:
                found.append((run.stat().st_mtime, run.resolve()))
            except OSError:
                continue
    found.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in found[:limit]]


def resolve_in_run(run_dir: Path | str, relative: str) -> Path:
    """A path inside one run directory, or a refusal.

    An absolute path is refused rather than quietly reinterpreted as a relative
    one: a caller that asked for a root path and silently got something else
    inside the run is worse off than one that was told no.

    `Path.resolve()` collapses `..` and follows symlinks, so the containment
    check after it covers traversal and link escapes with one comparison.
    """
    root = Path(run_dir).resolve()
    raw = str(relative).replace("\\", "/")
    if raw.startswith("/") or Path(raw).is_absolute() or Path(raw).drive:
        raise PathRefused(f"{relative!r} is absolute; run paths are relative")
    cleaned = raw.strip("/")
    if not cleaned:
        raise PathRefused(f"no path given inside {root}")
    candidate = Path(cleaned)
    resolved = (root / candidate).resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise PathRefused(f"{relative!r} resolves outside run directory {root}")
    return resolved


def walk_run_dir(run_dir: Path | str) -> list[str]:
    """Every file under a run directory, as POSIX-relative paths, sorted.

    A walk, not a lookup: whatever the flow wrote is here, whether that is
    source, a descriptor, a bundle, a build fragment or a tree of reports. The
    only exclusion is the manifest's write-in-progress scratch file, which is
    a fragment rather than an artifact.
    """
    root = Path(run_dir)
    if not root.is_dir():
        return []
    found: list[str] = []
    for directory, _, files in os.walk(root):
        base = Path(directory)
        for name in files:
            if name == _MANIFEST_TMP:
                continue
            try:
                relative = (base / name).relative_to(root)
            except ValueError:
                continue
            found.append(relative.as_posix())
    found.sort()
    return found


def mime_for(path: Path | str) -> str:
    name = Path(path).name
    if name in _MIME_BY_NAME:
        return _MIME_BY_NAME[name]
    return _MIME_BY_SUFFIX.get(Path(name).suffix.lower(), DEFAULT_MIME)


def resource_entries(
    run_id: str, run_dir: Path | str, flow_name: str | None = None
) -> Iterator[dict[str, Any]]:
    """One `resources/list` entry per file that exists under a run directory."""
    root = Path(run_dir)
    label = f"{flow_name} {run_id}" if flow_name else run_id
    for relative in walk_run_dir(root):
        try:
            size = (root / relative).stat().st_size
        except OSError:
            continue
        entry = {
            "uri": schema.format_run_uri(run_id, relative),
            "name": relative,
            "title": f"{label} — {relative}",
            "mimeType": mime_for(relative),
            "size": size,
        }
        if relative == schema.MANIFEST_NAME:
            entry["description"] = (
                "Authoritative structured run state. Rewritten atomically at "
                "every step transition."
            )
        yield entry


# -- the flows allow-list ---------------------------------------------------

#: Suffixes tried when a caller names a flow rather than a file. Bare names are
#: what `flow_list` publishes, so they are what a caller sends back.
_FLOW_SUFFIXES = (".yaml", ".yml")


def flow_files(flows_dir: Path | str) -> list[Path]:
    """Every flow file in the flows directory, in directory order."""
    directory = Path(flows_dir)
    try:
        entries = sorted(directory.iterdir())
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if entry.is_file() and entry.suffix.lower() in _FLOW_SUFFIXES
    ]


def resolve_flow(name_or_path: str, flows_dir: Path | str) -> Path:
    """The flow a caller named, resolved strictly inside the flows directory.

    This is the security boundary of the whole server. An agent step runs with
    `acceptEdits` against the checkout, so the set of runnable flows is exactly
    the set an operator put in one directory -- never a path a caller supplies.
    """
    directory = Path(flows_dir).resolve()
    text = str(name_or_path or "").strip().replace("\\", "/")
    if not text:
        raise FlowNotFound("no flow named")
    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive or text.startswith("//"):
        raise FlowRefused(
            f"{name_or_path!r} is an absolute path; name a flow inside {directory}"
        )
    attempts = [text] + [
        text + suffix for suffix in _FLOW_SUFFIXES if not text.endswith(suffix)
    ]
    for attempt in attempts:
        resolved = (directory / attempt).resolve()
        if not resolved.is_relative_to(directory):
            raise FlowRefused(
                f"{name_or_path!r} resolves outside the flows directory {directory}"
            )
        if resolved.is_file():
            return resolved
    raise FlowNotFound(f"no flow {name_or_path!r} in {directory}")
