# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Read and write skinny GEMM sweep result documents (YAML or msgpack).

The failed configurations of a sweep live next to the results in
<output stem>.failed.yaml: ONE top-level YAML list with a group per case,
appended case by case (successive one-element lists keep it a single list):

  - problem: {name: ..., m: ..., ...}
    failures:
      - {config: {...}, status: failed to build, stage: constraint, reason: ...}
"""

from __future__ import annotations

import os
import re
from typing import Any

import yaml

try:
    import msgpack
except (
    ImportError
):  # pragma: no cover - optional at import, required for msgpack output
    msgpack = None

# The C implementations are much faster on the large documents of a sweep.
Dumper = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
Loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)

FORMAT_YAML = "yaml"
FORMAT_MSGPACK = "msgpack"
MSGPACK_MAGIC = b"SGSW\x01"

YAML_HEADER = """\
# Skinny GEMM sweep results written by sweep_skinny_gemm.py.
# Times are microseconds: timeUs is the MEAN of the refined launches of gemm
# (+ reduce for split-K, per iteration), screeningUs the mean of the screening
# pass; minUs and p90Us (refine mode) are informational. Every ranked entry
# lists its launches in ns: gemmNs (and reduceNs for split-K) are the launches
# timeUs is the mean of, screeningGemmNs and screeningReduceNs (refine mode)
# those of screeningUs; reduce launches are warm (not flushed) and follow the
# GEMM launches. meta.timing tells how the events were attached and the flush
# size. verified is full (the
# whole result was compared with a dense host reference), sampled(N) (N output
# elements were compared, each over all its splits), failed (one-shot verify
# mismatch), or no (--no-verify). In refine+top mode a failing winner is dropped;
# in one-shot every timed configuration stays listed. phaseSeconds
# splits the wall time of a case on its GPU.
"""


# The per-launch time lists of a ranked entry, which a reader that only needs
# the summary statistics drops to keep its copy of the document small.
LAUNCH_LIST_KEYS = ("gemmNs", "reduceNs", "screeningGemmNs", "screeningReduceNs")


def without_launch_lists(entry: dict) -> dict:
    """The ranked entry without its per-launch time lists."""
    return {k: v for k, v in entry.items() if k not in LAUNCH_LIST_KEYS}


def output_format(path: str, explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    lower = path.lower()
    if lower.endswith(".msgpack") or lower.endswith(".mpack"):
        return FORMAT_MSGPACK
    return FORMAT_YAML


def read_results(path: str) -> dict[str, Any]:
    """Loads a sweep results document from YAML or msgpack."""
    with open(path, "rb") as stream:
        data = stream.read()
    if data.startswith(MSGPACK_MAGIC):
        if msgpack is None:
            raise RuntimeError("msgpack is not installed; pip install msgpack")
        return msgpack.unpackb(data[len(MSGPACK_MAGIC) :], raw=False)
    text = data.decode("utf-8")
    if text.startswith("#"):
        text = text[text.find("\n") + 1 :]
    return yaml.safe_load(text)


def dump_block(data: Any, stream) -> None:
    """Block YAML whose leaf mappings are on one line, to keep it scannable."""
    yaml.dump(
        data,
        stream,
        Dumper=Dumper,
        sort_keys=False,
        width=1000,
        default_flow_style=None,
    )


def failed_path(results_path: str) -> str:
    return os.path.splitext(results_path)[0] + ".failed.yaml"


def read_failures(results_path: str) -> list[dict]:
    """The per-case failure groups of the failed file of a results file."""
    path = failed_path(results_path)
    if not os.path.isfile(path):
        return []
    with open(path) as stream:
        document = yaml.load(stream, Loader=Loader)
    if document is not None and not isinstance(document, list):
        raise ValueError(f"{path}: expected a list of failure groups")
    return document or []


def process_state(pid: int) -> str | None:
    """The state letter of /proc/<pid>/stat, or None without /proc."""
    try:
        with open(f"/proc/{pid}/stat") as stream:
            text = stream.read()
    except OSError:
        return None
    # The command name may hold spaces and parentheses: the state follows the
    # last closing one.
    fields = text[text.rfind(")") + 1 :].split()
    return fields[0] if fields else None


def pid_alive(pid: int) -> bool:
    """False for a dead process and for a zombie that nobody has reaped yet:
    the pool executor fails its futures before it reaps a crashed process."""
    if process_state(pid) in ("Z", "X"):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def dead_process_files(
    directory: str, prefix: str = "", suffix: str = ".tmp"
) -> list[str]:
    """The names of the <name>.<pid><suffix> files whose process is dead (and is
    not this one): what a crash left behind."""
    pattern = re.compile(rf"^(.*)\.(\d+){re.escape(suffix)}$")
    stale = []
    for name in os.listdir(directory):
        match = pattern.match(name)
        if not name.startswith(prefix) or match is None:
            continue
        pid = int(match.group(2))
        if pid == os.getpid() or pid_alive(pid):
            continue
        stale.append(name)
    return stale


def remove_stale_temporaries(
    directory: str, prefix: str = "", suffix: str = ".tmp"
) -> int:
    """Removes the <name>.<pid><suffix> files of dead processes that a crash left
    behind; returns how many."""
    removed = 0
    for name in dead_process_files(directory, prefix, suffix):
        try:
            os.remove(os.path.join(directory, name))
        except FileNotFoundError:
            continue
        removed += 1
    return removed


def write_results(path: str, document: dict[str, Any], fmt: str) -> None:
    """Writes a sweep document atomically."""
    temporary = f"{path}.{os.getpid()}.tmp"
    if fmt == FORMAT_MSGPACK:
        if msgpack is None:
            raise RuntimeError("msgpack is not installed; pip install msgpack")
        meta = dict(document.get("meta") or {})
        meta["outputFormat"] = FORMAT_MSGPACK
        document = {**document, "meta": meta}
        packed = MSGPACK_MAGIC + msgpack.packb(document, use_bin_type=True)
        with open(temporary, "wb") as stream:
            stream.write(packed)
    else:
        meta = dict(document.get("meta") or {})
        meta["outputFormat"] = FORMAT_YAML
        document = {**document, "meta": meta}
        with open(temporary, "w") as stream:
            stream.write(YAML_HEADER)
            dump_block(document, stream)
    os.replace(temporary, path)
