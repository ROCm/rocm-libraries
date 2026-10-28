# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Shrink a module while a check still reports its failure."""

import os
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class Reduction:
    """The smallest module check still reported as failing, how many times
    check ran, and whether the search finished before max_checks. functions,
    blocks, and lines are (before, after) counts.
    """

    module: object
    checks: int
    complete: bool
    functions: tuple
    blocks: tuple
    lines: tuple

    def __str__(self):
        counts = ", ".join(
            "%s %d -> %d" % (label, before, after)
            for label, (before, after) in (
                ("functions", self.functions),
                ("blocks", self.blocks),
                ("lines", self.lines),
            )
        )
        text = "%s after %s" % (counts, _amount(self.checks, "check"))
        return text if self.complete else text + ", stopped at max_checks"


class _OutOfChecks(Exception):
    pass


def _amount(count, noun):
    return "%d %s%s" % (count, noun, "" if count == 1 else "s")


def _counts(module):
    functions = module.functions()
    blocks = [(fn.name, block.name) for fn in functions for block in fn.blocks()]
    lines = sum(len(module.block_lines(fn, block)) for fn, block in blocks)
    return len(functions), len(blocks), lines


def _function(module, name):
    [function] = [fn for fn in module.functions() if fn.name == name]
    return function


def _shrink(items, keep):
    """Chunked delta debugging. keep(rest) tries the items in rest without the
    others and says whether that was kept. Chunks start as all of items and
    halve down to one item. Returns the items that stay.
    """
    chunk = len(items)
    while items and chunk > 0:
        start = 0
        while start < len(items):
            rest = items[:start] + items[start + chunk :]
            if keep(rest):
                items = rest
            else:
                start += chunk
        chunk //= 2
    return items


def _delete_blocks(module, function, names):
    """Delete the blocks in names, in whatever order lets each one go: a block
    another block in names branches to goes after that block does.
    """
    pending = list(names)
    while pending:
        refused = []
        for name in pending:
            try:
                module.delete_block(function, name)
            except ValueError:
                refused.append(name)
        if len(refused) == len(pending):
            raise ValueError("blocks %s are branch targets" % ", ".join(refused))
        pending = refused


class _Search:
    def __init__(self, module, check, max_checks, log):
        self.module = module.copy()
        self.check = check
        self.max_checks = max_checks
        self.log = log or (lambda line: None)
        self.checks = 0
        self.kept = 0
        self.baseline = set()

    def _fails(self, module):
        if self.max_checks is not None and self.checks >= self.max_checks:
            raise _OutOfChecks()
        self.checks += 1
        return bool(self.check(module.copy()))

    def start(self):
        if not self._fails(self.module):
            raise ValueError("check does not report the module as failing")
        self.baseline = {diagnostic.id for diagnostic in self.module.verify()}

    def _try(self, edit):
        """Make the edit on a copy, and keep the copy when it still fails."""
        candidate = self.module.copy()
        try:
            edit(candidate)
        except ValueError:
            return False
        if not {diagnostic.id for diagnostic in candidate.verify()} <= self.baseline:
            return False
        try:
            failing = self._fails(candidate)
        except _OutOfChecks:
            raise
        except Exception:
            return False
        if failing:
            self.module = candidate
            self.kept += 1
        return failing

    def _functions(self):
        callables = [fn.name for fn in self.module.functions()[1:]]
        left = len(callables)

        def keep(rest):
            nonlocal left

            def edit(candidate):
                for fn in candidate.functions()[1:]:
                    if fn.name not in rest:
                        candidate.delete_function(fn.name)

            if not self._try(edit):
                return False
            self.log(
                "removed %s; %d left"
                % (_amount(left - len(rest), "function"), len(rest) + 1)
            )
            left = len(rest)
            return True

        _shrink(callables, keep)

    def _blocks(self, function):
        names = [block.name for block in _function(self.module, function).blocks()[1:]]
        left = len(names)

        def keep(rest):
            nonlocal left

            def edit(candidate):
                blocks = _function(candidate, function).blocks()[1:]
                gone = [block.name for block in blocks if block.name not in rest]
                _delete_blocks(candidate, function, gone)

            if not self._try(edit):
                return False
            self.log(
                "removed %s of '%s'; %d left"
                % (_amount(left - len(rest), "block"), function, len(rest) + 1)
            )
            left = len(rest)
            return True

        _shrink(names, keep)

    def _lines(self, function, block):
        lines = self.module.block_lines(function, block)
        left = len(lines)

        def keep(rest):
            nonlocal left
            body = "\n".join(lines[i] for i in rest)
            if not self._try(
                lambda candidate: candidate.replace_block(function, block, body)
            ):
                return False
            self.log(
                "removed %s of ^%s in '%s'; %d left"
                % (_amount(left - len(rest), "line"), block, function, len(rest))
            )
            left = len(rest)
            return True

        _shrink(list(range(len(lines))), keep)

    def round(self):
        """One pass over every level. True when it kept a removal."""
        kept = self.kept
        self._functions()
        for fn in self.module.functions():
            self._blocks(fn.name)
        for fn in self.module.functions():
            for block in fn.blocks():
                self._lines(fn.name, block.name)
        return self.kept != kept


def reduce(module, check, *, max_checks=None, log=None):
    """Shrink module while check still reports its failure, as llvm-reduce does.

    check(candidate) returns true while the bug is present. Each call gets its
    own copy of a candidate, with no attached SSA, and may run passes on it. The
    first call is on the module as given: if it returns false, reduce raises
    ValueError, and if it raises, so does reduce. On a later candidate, an
    exception counts as not failing.

    Each round removes callable functions, then blocks other than a function's
    entry, then body lines, a chunk at a time down to one, until a round removes
    nothing. A candidate whose verify() reports a kind of problem the module did
    not have is skipped without a check. max_checks bounds the calls to check,
    and log, when given, gets one line per removal kept. The module given is not
    changed.
    """
    if max_checks is not None and max_checks < 1:
        raise ValueError("max_checks must be at least 1, not %r" % (max_checks,))
    search = _Search(module, check, max_checks, log)
    before = _counts(search.module)
    search.start()
    complete = True
    try:
        while search.round():
            pass
    except _OutOfChecks:
        complete = False
    functions, blocks, lines = zip(before, _counts(search.module))
    return Reduction(search.module, search.checks, complete, functions, blocks, lines)


def command_check(argv, *, form="stir", timeout=None):
    """A check that runs a command on each candidate, as llvm-reduce --test does.

    The candidate is written to a temporary candidate.stir with serialize(), or
    candidate.s with emit_asm() when form is "asm", and that path is the last
    argument. Exit status 0 means the candidate still fails. The command reads no
    input and its output is discarded, so a pass that aborts ends only that
    process. A command still running after timeout seconds is stopped and counts
    as not failing. One that cannot start raises OSError.
    """
    if form not in ("stir", "asm"):
        raise ValueError('form must be "stir" or "asm", not %r' % (form,))
    command = [argv] if isinstance(argv, str) else list(argv)
    name = "candidate.s" if form == "asm" else "candidate.stir"

    def check(module):
        text = module.emit_asm() if form == "asm" else module.serialize()
        with tempfile.TemporaryDirectory(prefix="stinkytofu-reduce-") as directory:
            path = os.path.join(directory, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            try:
                result = subprocess.run(
                    command + [path],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False
        return result.returncode == 0

    return check
