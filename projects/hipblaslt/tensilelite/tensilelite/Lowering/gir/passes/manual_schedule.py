# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Editable, block-local GIR schedules for SIA0 experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading

import yaml

from ....Common.Utilities import print1
from ..nodes import Mark, Mma, Move  # noqa: F401 -- kept for the Mark guard below
from ..render import inst_line
from .base import Pass


FORMAT = "tensilelite-gir-schedule-v1"


def schedule_file(root, kernel_name):
    """Deterministic, filesystem-safe schedule path for one full kernel name."""
    digest = hashlib.sha256(kernel_name.encode("utf-8")).hexdigest()[:16]
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", kernel_name)[:96].rstrip("_.-")
    return Path(root) / ("%s__%s.schedule.yaml" % (prefix or "kernel", digest))


def schedule_cache_tag(kernel_name, dump_path="", load_path=""):
    """Cache discriminator; edited schedules must rebuild cached GIR."""
    if load_path:
        path = schedule_file(load_path, kernel_name)
        if path.is_file():
            return "load:" + hashlib.sha256(path.read_bytes()).hexdigest()
        return "load:missing"
    if dump_path:
        return "dump:" + str(Path(dump_path).resolve())
    return ""


def _fingerprint(prog):
    """Fingerprint the unmodified GIR at the manual scheduling point."""
    records = []
    for block in prog.walk_rpo():
        records.append((
            block.label,
            tuple(repr(inst) for inst in block.body),
            tuple(repr(phi) for phi in block.phis),
            repr(block.term),
        ))
    payload = repr(tuple(records)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()





def _dump_text(prog, kernel_name):
    lines = [
        "# Reorder complete list items only; comments are informational.",
        "format: %s" % FORMAT,
        "kernel: %s" % json.dumps(kernel_name),
        "fingerprint: %s" % _fingerprint(prog),
        "blocks:",
    ]
    for block in prog.walk_rpo():
        label = block.label
        quoted = json.dumps(label)
        if not block.body:
            lines.append("  %s: []" % quoted)
            continue
        lines.append("  %s:" % quoted)
        for index, inst in enumerate(block.body):
            lines.append("    - %d # %s" % (index, inst_line(inst)))
    return "\n".join(lines) + "\n"


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        "%s.tmp.%d.%d" % (path.name, os.getpid(), threading.get_ident()))
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _read_schedule(path):
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("GIR schedule %s cannot be read: %s" % (path, exc)) from exc
    if not isinstance(document, dict):
        raise RuntimeError("GIR schedule %s must contain a mapping" % path)
    return document


class ManualSchedulePass(Pass):
    """Dump or apply one compact, ordinal-based schedule per kernel."""

    def __init__(self, kernel_name, *, dump_path="", load_path=""):
        if dump_path and load_path:
            raise ValueError("GIR schedule dump and load modes are mutually exclusive")
        self.kernel_name = str(kernel_name)
        self.dump_path = str(dump_path or "")
        self.load_path = str(load_path or "")

    def run(self, prog, am):
        if self.dump_path:
            path = schedule_file(self.dump_path, self.kernel_name)
            _write_atomic(path, _dump_text(prog, self.kernel_name))
            return
        if not self.load_path:
            return

        path = schedule_file(self.load_path, self.kernel_name)
        if not path.is_file():
            return                              # a directory may override only selected kernels
        document = _read_schedule(path)
        if document.get("format") != FORMAT:
            raise RuntimeError(
                "GIR schedule %s has format %r, expected %r"
                % (path, document.get("format"), FORMAT))
        if document.get("kernel") != self.kernel_name:
            raise RuntimeError(
                "GIR schedule %s names a different kernel\n  file:     %s\n  expected: %s"
                % (path, document.get("kernel"), self.kernel_name))
        actual_fingerprint = _fingerprint(prog)
        if document.get("fingerprint") != actual_fingerprint:
            raise RuntimeError(
                "GIR schedule %s is stale: GIR fingerprint is %s, file has %s"
                % (path, actual_fingerprint, document.get("fingerprint")))

        schedules = document.get("blocks")
        if not isinstance(schedules, dict):
            raise RuntimeError("GIR schedule %s has no blocks mapping" % path)
        expected_labels = set(prog.blocks)
        if set(schedules) != expected_labels:
            raise RuntimeError(
                "GIR schedule %s block set differs: expected %s, found %s"
                % (path, sorted(expected_labels), sorted(schedules)))

        changed = False
        for label, block in prog.blocks.items():
            order = schedules[label]
            if not isinstance(order, list) or any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in order):
                raise RuntimeError(
                    "GIR schedule %s block %s must be a list of integer instruction ids"
                    % (path, label))
            expected = list(range(len(block.body)))
            if sorted(order) != expected:
                raise RuntimeError(
                    "GIR schedule %s block %s must be an exact permutation of %s, found %s"
                    % (path, label, expected, order))
            if order != expected:
                original = list(block.body)
                block.body[:] = [original[index] for index in order]
                changed = True
        if changed:
            prog.bump()
        print1("%s loaded" % path)
