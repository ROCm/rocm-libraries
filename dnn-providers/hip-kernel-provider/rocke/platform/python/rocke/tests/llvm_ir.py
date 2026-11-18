# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Test helper: read the SSA definitions of a lowered kernel's LLVM IR.

A host test that checks what a kernel computes, rather than how its IR is
spelled, walks the definitions of the LLVM IR both engines emit: from a store
back to the loads and loop phis it is computed from. :class:`KernelIR` holds
those definitions by SSA name and answers the questions every such walk asks
(which instruction defines a value, with which operands, what a loop phi starts
with and carries round the loop). It also evaluates the integer
index arithmetic of an address for given work-item ids and loop counters, so a
test can check, for every lane, which element a load reads or a store writes.
A test adds the queries specific to its kernel in a subclass.

It reads the one kernel of a module as the rocke lowering prints it; it is not a
general LLVM IR parser. :func:`intrinsic_calls` finds the calls of one intrinsic.
"""

from __future__ import annotations

import re

import numpy as np

__all__ = ["KernelIR", "intrinsic_calls"]

_DEFINITION = re.compile(r"^\s*(%[\w.]+) = (.*)$")
_LABEL = re.compile(r"^([\w.]+):")
_CALLEE = re.compile(r"@([\w.]+)\((.*)\)")
_INCOMING = re.compile(r"\[\s*([^,\]]+?)\s*,\s*(%[\w.]+)\s*\]")
_LOOP_BOUND = re.compile(r"icmp slt i32 (%[\w.]+), (\d+)$")
_BUFFER_LOAD = "llvm.amdgcn.raw.ptr.buffer.load"
_BUFFER_RESOURCE = "llvm.amdgcn.make.buffer.rsrc"


def intrinsic_calls(llvm_text: str, intrinsic: str) -> list[str]:
    """The call lines of ``intrinsic`` (named with or without its ``@``) in ``llvm_text``, with or
    without LLVM's overload type suffixes (``.v8i32``, ``.f16``): another intrinsic that merely
    starts with the same name, such as its ``.tied`` twin, does not count."""
    name = re.escape(intrinsic.removeprefix("@"))
    pattern = re.compile(rf"@{name}(\.v?\d*(?:i|f|bf)\d+)*\(")
    return [
        line
        for line in llvm_text.splitlines()
        if "call" in line and pattern.search(line)
    ]


class KernelIR:
    """The SSA definitions of the one kernel in an LLVM IR text, by name.

    ``definitions`` maps each SSA name (``%v12``) to the text right of its
    ``=``; ``block_of`` maps it to the label of the block that defines it
    (``entry`` before the first label); ``lines`` is the text, line by line.
    """

    def __init__(self, llvm_text: str):
        self.lines = llvm_text.splitlines()
        self.definitions: dict[str, str] = {}
        self.block_of: dict[str, str] = {}
        block = "entry"
        for line in self.lines:
            label = _LABEL.match(line)
            if label:
                block = label.group(1)
            match = _DEFINITION.match(line)
            if match:
                self.definitions[match.group(1)] = match.group(2)
                self.block_of[match.group(1)] = block

    @staticmethod
    def callee(definition: str) -> str:
        """The function a ``call`` definition calls, without its ``@``."""
        return _CALLEE.search(definition).group(1)

    def opcode(self, name: str) -> str:
        """The instruction (``fmul``, ``add``, ...) or, for a call, the callee
        without its type suffix (``llvm.fma``, ``llvm.exp2``, ...); every MFMA
        intrinsic is ``mfma``."""
        text = self.definitions[name]
        if text.startswith("call "):
            callee = self.callee(text)
            if callee.startswith("llvm.amdgcn.mfma."):
                return "mfma"
            return callee.rsplit(".", 1)[0]
        return text.split()[0]

    def operands(self, name: str) -> list[str]:
        """The operand values, in order: an SSA name (``%x``) or a literal."""
        text = self.definitions[name]
        if text.startswith("call "):
            text = _CALLEE.search(text).group(2)
        else:
            text = text.split(None, 1)[1]
        values = []
        for item in text.split(","):
            item = item.split(" to ")[0]  # a cast: ``sext i32 %x to i64``
            values.append(item.split()[-1])
        return values

    def find(self, opcode: str) -> list[str]:
        """Every value whose :meth:`opcode` is ``opcode``, in IR order."""
        return [n for n in self.definitions if self.opcode(n) == opcode]

    def copied(self, name: str) -> str:
        """The value ``name`` copies: a ``bitcast`` to its own type, as the
        lowering emits for loop-carried values, is followed to its source. A
        literal or a value that is not a copy is returned as it is."""
        while name in self.definitions and self.opcode(name) == "bitcast":
            (name,) = self.operands(name)
        return name

    def incoming(self, phi: str) -> tuple[str, str]:
        """A loop phi's (initial value, value carried round the loop, with
        copies followed)."""
        assert self.opcode(phi) == "phi", self.definitions[phi]
        (initial, entry), (carried, latch) = _INCOMING.findall(self.definitions[phi])
        assert entry != latch, self.definitions[phi]
        return initial, self.copied(carried)

    def line_of(self, name: str) -> int:
        """The line number of ``name``'s definition."""
        return next(
            i
            for i, line in enumerate(self.lines)
            if line.strip().startswith(name + " = ")
        )

    def is_buffer_load(self, name: str) -> bool:
        """Whether ``name`` is a raw buffer load (any width)."""
        return self.opcode(name) == _BUFFER_LOAD

    def buffer_param(self, resource: str) -> str:
        """The pointer parameter a buffer resource is made from (``%C``)."""
        assert self.opcode(resource).startswith(_BUFFER_RESOURCE), resource
        return self.operands(resource)[0]

    def buffer_load_access(self, load: str) -> tuple[str, str, str]:
        """(pointer parameter, lane byte offset, scalar byte offset) of a raw
        buffer load: the offsets are SSA names or literals, for
        :meth:`evaluate`."""
        assert self.is_buffer_load(load), self.definitions[load]
        resource, lane_offset, scalar_offset, _ = self.operands(load)
        return self.buffer_param(resource), lane_offset, scalar_offset

    def call_named(self, callee: str) -> str:
        """The value defined by the one call of ``callee`` (for example
        ``llvm.amdgcn.workitem.id.x``)."""
        (name,) = [
            n
            for n, text in self.definitions.items()
            if text.startswith("call ") and self.callee(text) == callee
        ]
        return name

    def loop_counter(self, phi: str) -> tuple[str, int, int]:
        """(counter, first value, end) of the loop whose header defines
        ``phi``: the header's i32 phi that its ``icmp slt`` compares with a
        constant bound, the value it starts with, and that bound."""
        header = self.block_of[phi]
        header_i32_phis = {
            n
            for n, text in self.definitions.items()
            if self.block_of[n] == header and text.startswith("phi i32 ")
        }
        ((counter, end),) = [
            (m.group(1), int(m.group(2)))
            for m in map(_LOOP_BOUND.search, self.definitions.values())
            if m and m.group(1) in header_i32_phis
        ]
        first, _ = self.incoming(counter)
        return counter, int(first), end

    def evaluate(
        self, name: str, inputs: dict[str, np.ndarray | int]
    ) -> np.ndarray | np.int64:
        """Evaluate an i32 index value with numpy, given values (scalars or
        arrays that broadcast) for the work-item ids and loop counters it
        reads, keyed by SSA name. Every value here is a non-negative index, so
        ``sdiv`` / ``srem`` are floor division and remainder; a
        ``readfirstlane`` passes its operand through, which is right for the
        wave-uniform values the lowering reads that way. An instruction this
        does not know fails the test."""
        if not name.startswith("%"):
            return np.int64(name)
        if name in inputs:
            return inputs[name]
        op = self.opcode(name)
        args = [self.evaluate(v, inputs) for v in self.operands(name)]
        if op in ("sext", "zext", "trunc", "llvm.amdgcn.readfirstlane"):
            return args[0]
        if op == "add":
            return args[0] + args[1]
        if op == "sub":
            return args[0] - args[1]
        if op == "mul":
            return args[0] * args[1]
        if op == "shl":
            return args[0] << args[1]
        if op == "or":
            return args[0] | args[1]
        if op in ("sdiv", "srem"):
            assert (np.asarray(args[0]) >= 0).all() and args[1] > 0
            return args[0] // args[1] if op == "sdiv" else args[0] % args[1]
        raise AssertionError(f"not modelled: {self.definitions[name]}")
