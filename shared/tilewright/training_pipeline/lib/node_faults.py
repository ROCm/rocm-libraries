# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU faults that tell against the node a bench ran on, and the record of
the nodes they retired.

Two errors a bench process can report count against its node, whatever
line it ran:

  719  hipErrorLaunchFailure, a launch that failed: the stage stops the
       node's benches at once and marks the node;
  700  hipErrorIllegalAddress, a memory access fault: the stage keeps the
       runtime's record of it (`runtime_record`: the kernel being launched,
       the faulting address, the dispatch packet), then benches the line
       once more in a fresh process with RERUN_ENV. When the fault follows
       the line (the re-run faults again) the line is quarantined in the run
       and the node goes on; when it does not, the node is marked and its
       benches stop, as for 719.

`fault_of` reads which of them a bench's output reports, 719 before 700.

A marked node takes no bench of any run until it is reset (`NodeStore`):
the store (a directory outliving runs: `store_dir`) keeps each node's faults
in `<host>.json`, and a node is held (`NodeStore.held`) while it has a fault
newer than its reset marker `<host>.reset`, which only a person writes, once
the node was rebooted and checked (scripts/node_faults.py reset). Nothing
in the pipeline writes a reset marker.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LAUNCH_FAILURE = 719
ILLEGAL_ADDRESS = 700
# What a bench prints for each fault: HIP's error name and string, the code
# in the form hipblaslt-bench and HIP print it, and for 700 the HSA
# runtime's memory fault report.
_PATTERNS = {
    LAUNCH_FAILURE: re.compile(
        r"hipErrorLaunchFailure|unspecified launch failure"
        r"|\(719\)|\b(?:error|code|status)\W{0,3}719\b",
        re.IGNORECASE,
    ),
    ILLEGAL_ADDRESS: re.compile(
        r"hipErrorIllegalAddress|an illegal memory access was encountered"
        r"|Memory access fault by GPU|HSA_STATUS_ERROR_MEMORY_FAULT"
        r"|\(700\)|\b(?:error|code|status)\W{0,3}700\b",
        re.IGNORECASE,
    ),
}
# The environment of the re-run of a line that faulted with 700: every
# launch synchronous, so that the fault names its own launch, and TensileLite
# printing each kernel and its launch arguments.
RERUN_ENV = {"HIP_LAUNCH_BLOCKING": "1", "TENSILE_DB": "0x200040"}
# The store a run uses when it names none.
STORE_VAR = "TILEWRIGHT_NODE_FAULTS"
_KERNEL_RE = re.compile(
    r"(?:kernel(?:[ _]name)?|solution name)\s*[:=]\s*(\S+)", re.IGNORECASE
)
_ADDRESS_RE = re.compile(r"address\s*[:=]?\s*(0x[0-9a-f]+)", re.IGNORECASE)
_PACKET_RE = re.compile(r"packet|dispatch", re.IGNORECASE)
_RECORD_LINES = 20


def fault_of(text: str) -> Optional[int]:
    """The fault (LAUNCH_FAILURE or ILLEGAL_ADDRESS) a bench's output
    `text` reports, LAUNCH_FAILURE when it reports both; None for none."""
    for code in (LAUNCH_FAILURE, ILLEGAL_ADDRESS):
        if _PATTERNS[code].search(text):
            return code
    return None


def runtime_record(text: str) -> Dict[str, Any]:
    """What a bench's output `text` holds of its fault: `code`
    (`fault_of`), `kernel` (the last kernel or solution name it printed
    before the fault), `address` (the faulting address the runtime named),
    `packet` (the lines about the dispatch packet) and `lines` (the lines
    that report the fault), at most _RECORD_LINES each."""
    lines = text.splitlines()
    code = fault_of(text)
    at = next(
        (i for i, ln in enumerate(lines) if code and _PATTERNS[code].search(ln)),
        len(lines),
    )
    kernel = None
    for ln in lines[:at]:
        m = _KERNEL_RE.search(ln)
        if m:
            kernel = m.group(1).strip("'\",")
    address = None
    for ln in lines:
        m = _ADDRESS_RE.search(ln)
        if m:
            address = m.group(1)
            break
    return {
        "code": code,
        "kernel": kernel,
        "address": address,
        "packet": [ln.strip() for ln in lines if _PACKET_RE.search(ln)][:_RECORD_LINES],
        "lines": [ln.strip() for ln in lines if code and _PATTERNS[code].search(ln)][
            :_RECORD_LINES
        ],
    }


def node_name() -> str:
    """The name of this node, as the store keeps it."""
    return socket.gethostname().split(".", 1)[0] or "localhost"


def store_dir(configured: Optional[str] = None) -> Path:
    """The store of node faults: `configured`, else STORE_VAR, else
    `$XDG_STATE_HOME/tilewright/node_faults` (`~/.local/state` without
    it)."""
    if configured:
        return Path(configured).expanduser()
    if os.environ.get(STORE_VAR):
        return Path(os.environ[STORE_VAR]).expanduser()
    state = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(state) / "tilewright" / "node_faults"


class NodeStore:
    """The faults of the nodes in directory `root` and their reset
    markers."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def faults_path(self, node: str) -> Path:
        return self.root / f"{node}.json"

    def reset_path(self, node: str) -> Path:
        return self.root / f"{node}.reset"

    def faults(self, node: str) -> List[Dict[str, Any]]:
        """Every fault recorded for `node`, oldest first."""
        try:
            doc = json.loads(self.faults_path(node).read_text())
        except (OSError, ValueError):
            return []
        return [f for f in doc if isinstance(f, dict)] if isinstance(doc, list) else []

    def mark(self, node: str, fault: Mapping[str, Any]) -> Dict[str, Any]:
        """Record `fault` against `node` (with the time, `at`, and the
        epoch, `epoch`); returns the entry."""
        now = time.time()
        entry = {
            **dict(fault),
            "node": node,
            "epoch": now,
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.faults_path(node)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(self.faults(node) + [entry], indent=2) + "\n")
        os.replace(tmp, path)
        return entry

    def reset_epoch(self, node: str) -> Optional[float]:
        try:
            return self.reset_path(node).stat().st_mtime
        except OSError:
            return None

    def held(self, node: str) -> List[Dict[str, Any]]:
        """The faults that hold `node`: those after its last reset marker,
        none when it has no fault since."""
        reset = self.reset_epoch(node)
        return [
            f
            for f in self.faults(node)
            if reset is None or float(f.get("epoch", 0)) > reset
        ]

    def reset(self, node: str, note: str) -> Path:
        """Write the reset marker of `node` with `note` (who checked the
        node, after which reboot): its faults so far hold it no more."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.reset_path(node)
        path.write_text(
            json.dumps(
                {"note": note, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            )
            + "\n"
        )
        return path


def describe_hold(node: str, faults: List[Mapping[str, Any]], root: Path) -> str:
    """Why `node` takes no bench, for messages."""
    last = faults[-1]
    return (
        f"node {node} is held by GPU faults since its last reset ({len(faults)} kept), "
        f"the last error {last.get('code')} at {last.get('at')} "
        f"({last.get('why') or 'no reason recorded'}); once it was rebooted and "
        f"checked: scripts/node_faults.py reset {node} --store {root} --note '...'"
    )
