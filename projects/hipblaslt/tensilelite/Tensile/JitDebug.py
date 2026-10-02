# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Opt-in timing and progress records for the JIT generator command lines.

``--debug CATEGORIES`` (``timing``, ``progress`` or ``all``, comma-separated)
turns them on. With ``--debug-dir DIR``, which hipBLASLt passes when
HIPBLASLT_JIT_DEBUG asks for them, progress events are appended to
``DIR/events.jsonl`` as they happen and timing is written to ``DIR/timing.json``
at the end. Without it, progress is printed to stderr as it happens and timing
as one table at the end. Nothing else the command writes changes, and a failure
to record never fails the command.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

CATEGORIES = ("timing", "progress")


def parseCategories(value):
    """Return the categories a ``--debug`` value names; raise ValueError for others."""
    names = set()
    for token in value.split(","):
        token = token.strip().lower()
        if token == "all":
            names.update(CATEGORIES)
        elif token in CATEGORIES:
            names.add(token)
        elif token:
            raise ValueError(f"unknown category {token!r}; use timing, progress or all")
    if not names:
        raise ValueError("no category; use timing, progress or all")
    return frozenset(names)


def addArguments(parser):
    """Add the private ``--debug`` and ``--debug-dir`` options to an argparse parser."""
    import argparse

    def categories(value):
        try:
            return parseCategories(value)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from None

    parser.add_argument("--debug", type=categories, help=argparse.SUPPRESS)
    parser.add_argument("--debug-dir", dest="debugDir", help=argparse.SUPPRESS)


def fromArguments(parser, args, *, module, mode):
    """Remove the debug options from ``args`` and return their recorder."""
    categories = args.pop("debug")
    directory = args.pop("debugDir")
    if directory is not None and categories is None:
        parser.error("--debug-dir requires --debug")
    if categories is None:
        return NULL
    return Recorder(categories, directory, module=module, mode=mode)


class _Null:
    """The recorder when nothing is recorded: no clock reads, files or output."""

    timing = progress = False
    _span = contextlib.nullcontext({})

    def span(self, name, **fields):
        return self._span

    def event(self, kind, **fields):
        pass

    def request(self, **fields):
        pass

    def candidate(self, **fields):
        pass

    def bundle(self, rank):
        pass

    def published(self, count):
        pass

    def finish(self, status, errorType=None):
        pass


NULL = _Null()


def _describe(kind, fields):
    fields = {key: value for key, value in fields.items() if value is not None}
    if kind == "stage":
        words = [fields.pop("stage"), fields.pop("phase"), fields.pop("status", None)]
        if "ns" in fields:
            words.append(f"{fields.pop('ns') / 1e6:.1f} ms")
    else:
        words = [kind]
    words += [f"{key}={value}" for key, value in fields.items()]
    return "progress: " + " ".join(word for word in words if word is not None)


class Recorder:
    """Records inclusive spans, candidates and progress events for one command."""

    def __init__(self, categories, directory=None, *, module, mode):
        self.timing = "timing" in categories
        self.progress = "progress" in categories
        self._start = time.perf_counter_ns()
        self._cpu = time.process_time_ns()
        self._times = os.times()
        self._categories = sorted(categories)
        self._module = module
        self._mode = mode
        # Captured now: candidate derivation redirects sys.stderr.
        self._stderr = sys.stderr
        self._directory = Path(directory) if directory is not None else None
        self._events = None
        self._seq = 0
        self._spans = []
        self._open = []
        self._candidates = []
        self._rank = None
        self._published = 0
        self._request = {}
        if self._directory is not None:
            try:
                self._directory.mkdir(parents=True, exist_ok=True)
                if self.progress:
                    self._events = open(self._directory / "events.jsonl", "a", encoding="utf-8")
            except OSError:
                self._directory = None
                self.progress = self.timing = False

    def now(self):
        return time.perf_counter_ns()

    @contextlib.contextmanager
    def span(self, name, *, stage=True, rejects=(), **fields):
        """Time a block as an inclusive span; ``stage`` spans also emit start and end events.

        Exceptions of the ``rejects`` types end the span as ``rejected``, others as ``failed``.
        """
        record = {"id": len(self._spans) + 1, "parent": self._open[-1] if self._open else None,
                  "name": name, "rank": self._rank, **fields}
        self._spans.append(record)
        self._open.append(record["id"])
        if stage:
            self.event("stage", stage=name, phase="start", rank=self._rank)
        status = "ok"
        start = time.perf_counter_ns()
        try:
            yield record
        except rejects:
            status = "rejected"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            record["ns"] = time.perf_counter_ns() - start
            record["status"] = status
            self._open.pop()
            if stage:
                self.event("stage", stage=name, phase="end", status=status, rank=self._rank,
                           ns=record["ns"] if self.timing else None)

    def event(self, kind, **fields):
        """Append one progress event; fields that are None are left out."""
        if not self.progress:
            return
        self._seq += 1
        record = {"v": 1, "seq": self._seq, "pid": os.getpid(), "mono_ns": time.monotonic_ns(),
                  "kind": kind}
        record.update((key, value) for key, value in fields.items() if value is not None)
        try:
            if self._events is not None:
                self._events.write(json.dumps(record, separators=(",", ":"), default=str) + "\n")
                self._events.flush()
            else:
                print(_describe(kind, fields), file=self._stderr, flush=True)
        except (OSError, ValueError):
            self.progress = False

    def request(self, **fields):
        """Record what was asked for and emit the ``request`` event."""
        self._request = {key: value for key, value in fields.items() if value is not None}
        self.event("request", module=self._module, mode=self._mode, **self._request)

    def candidate(self, **fields):
        """Record one candidate the selection tried and emit its ``candidate`` event."""
        fields = {"rank": self._rank, **fields}
        if self.timing:
            self._candidates.append({key: value for key, value in fields.items()
                                     if value is not None})
        if not self.timing:
            fields.pop("ns", None)
        self.event("candidate", **fields)

    def bundle(self, rank):
        """Attribute the following spans and candidates to bundle ``rank``."""
        self._rank = rank

    def published(self, count):
        self._published = count

    def finish(self, status, errorType=None):
        """Emit ``done`` and write ``timing.json`` or the timing table. Never raises."""
        try:
            self.event("done", module=self._module, mode=self._mode, status=status,
                       bundles_published=self._published, error_type=errorType, **self._request)
            if self._events is not None:
                self._events.close()
                self._events = None
            if self.timing:
                self._writeTiming(status, errorType)
        except Exception:
            pass

    def _writeTiming(self, status, errorType):
        times = os.times()
        children = (times.children_user + times.children_system
                    - self._times.children_user - self._times.children_system)
        totals = {}
        for span in self._spans:
            totals[span["name"]] = totals.get(span["name"], 0) + span.get("ns", 0)
        document = {
            "v": 1,
            "producer": "python",
            "module": self._module,
            "mode": self._mode,
            "categories": self._categories,
            "clock": "perf_counter_ns",
            "status": status,
            "error_type": errorType,
            "bundles_published": self._published,
            "total_ns": time.perf_counter_ns() - self._start,
            "cpu_ns": time.process_time_ns() - self._cpu,
            "children_cpu_ns": round(children * 1e9),
            "cpu_threads": 1,
            "totals": totals,
            "spans": self._spans,
            "candidates": self._candidates,
        }
        if self._directory is None:
            self._printTiming(document)
            return
        descriptor, temporary = tempfile.mkstemp(prefix=".timing-", suffix=".json",
                                                 dir=self._directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(document, stream, separators=(",", ":"), default=str)
            os.replace(temporary, self._directory / "timing.json")
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

    def _printTiming(self, document):
        def ms(ns):
            return f"{ns / 1e6:10.1f} ms"

        lines = [f"timing: {self._module} {document['status']}: total {ms(document['total_ns'])},"
                 f" cpu {ms(document['cpu_ns'])}"]
        for span in document["spans"]:
            if span["parent"] is None:
                rank = "" if span["rank"] is None else f" [{span['rank']}]"
                lines.append(f"timing:   {span['name'] + rank:<20} {ms(span['ns'])} {span['status']}")
        print("\n".join(lines), file=self._stderr, flush=True)
