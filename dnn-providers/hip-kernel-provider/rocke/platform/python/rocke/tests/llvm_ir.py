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

from ..core.ir import fits_i32

__all__ = ["KernelIR", "intrinsic_calls"]

_DEFINITION = re.compile(r"^\s*(%[\w.]+) = (.*)$")
_LABEL = re.compile(r"^([\w.]+):")
_CALLEE = re.compile(r"@([\w.]+)\((.*)\)")
_INCOMING = re.compile(r"\[\s*([^,\]]+?)\s*,\s*(%[\w.]+)\s*\]")
_LOOP_BOUND = re.compile(r"icmp slt i32 (%[\w.]+), (\d+)$")
_BUFFER_LOAD = "llvm.amdgcn.raw.ptr.buffer.load"
_BUFFER_RESOURCE = "llvm.amdgcn.make.buffer.rsrc"


def _all_fit_i32(values: np.ndarray | np.int64) -> bool:
    """Whether every element of ``values`` is a signed i32 (:func:`rocke.core.ir.fits_i32`)."""
    array = np.asarray(values)
    return array.size == 0 or (
        fits_i32(int(array.min())) and fits_i32(int(array.max()))
    )


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

    def header_i32_phis(self, phi: str) -> set[str]:
        """The i32 phis of the loop header that defines ``phi``: the loop's
        counter among them, whatever its bound."""
        header = self.block_of[phi]
        return {
            name
            for name, text in self.definitions.items()
            if self.block_of[name] == header and text.startswith("phi i32 ")
        }

    def loop_counter(self, phi: str) -> tuple[str, int, int]:
        """(counter, first value, end) of the loop whose header defines
        ``phi``: the header's i32 phi that its ``icmp slt`` compares with a
        constant bound, the value it starts with, and that bound."""
        header_i32_phis = self.header_i32_phis(phi)
        ((counter, end),) = [
            (match.group(1), int(match.group(2)))
            for match in map(_LOOP_BOUND.search, self.definitions.values())
            if match and match.group(1) in header_i32_phis
        ]
        first, _ = self.incoming(counter)
        return counter, int(first), end

    def loads_feeding(self, name: str, element: int) -> set[str]:
        """The loads (global or buffer) that element ``element`` of ``name``
        is computed from, through float and vector arithmetic; the walk stops
        at loads, phis and MFMA results."""
        found, stack, seen = set(), [(name, element)], set()
        while stack:
            current, lane = stack.pop()
            if (current, lane) in seen or not current.startswith("%"):
                continue
            seen.add((current, lane))
            op = self.opcode(current)
            if (
                op == "load"
                or self.is_buffer_load(current)
                or (op == "extractelement" and self.vector_load_element(current))
            ):
                found.add(current)
            elif op == "insertelement":
                vector, scalar, index = self.operands(current)
                stack.append((scalar, 0) if int(index) == lane else (vector, lane))
            elif op in ("fmul", "fadd", "llvm.fma", "llvm.fmuladd", "bitcast"):
                stack.extend((operand, lane) for operand in self.operands(current))
            elif op not in ("phi", "mfma"):
                raise AssertionError(f"not modelled: {self.definitions[current]}")
        return found

    def vector_load_element(self, name: str) -> tuple[str, int] | None:
        """(buffer load, element index) when ``name`` extracts a constant
        element of a vector buffer load; ``None`` otherwise."""
        vector, index = self.operands(name)
        vector = self.copied(vector)
        if self.is_buffer_load(vector) and index.isdigit():
            return vector, int(index)
        return None

    def address_offset(self, address: str) -> tuple[str, str]:
        """(pointer parameter, element offset) of a ``getelementptr`` load or
        store address. A base that is itself a ``getelementptr`` of the
        parameter (a slice of it, such as one expert's weights) is followed to
        the parameter; the offset is the one within that slice. The slice's own
        offset is dropped, not added: it is often computed from loaded data
        (an expert id read from memory), which :meth:`evaluate` cannot follow,
        so a check built on this offset does not see a wrong slice base."""
        assert self.opcode(address) == "getelementptr", self.definitions[address]
        _, base, offset = self.operands(address)
        while base in self.definitions:
            assert self.opcode(base) == "getelementptr", self.definitions[base]
            base = self.operands(base)[1]
        return base, offset

    def read_location(
        self, load: str, inputs: dict[str, np.ndarray | int]
    ) -> tuple[str, np.ndarray | np.int64]:
        """(pointer parameter, element index) a 4-byte load reads, the index
        evaluated for ``inputs``: from a global load's ``getelementptr``
        (:meth:`address_offset`, so within the slice of the parameter the load
        addresses), or from a buffer load's resource (its base
        pointer) and its per-lane and scalar byte offsets, plus 4 bytes per
        element for an element of a vector buffer load."""
        element = 0
        if self.opcode(load) == "extractelement":
            load, element = self.vector_load_element(load)
        if self.opcode(load) == "load":
            (address,) = [v for v in self.operands(load) if v.startswith("%")]
            param, offset = self.address_offset(address)
            return param, self.evaluate(offset, inputs)
        param, lane_offset, scalar_offset = self.buffer_load_access(load)
        byte = (
            self.evaluate(lane_offset, inputs)
            + self.evaluate(scalar_offset, inputs)
            + 4 * element
        )
        assert (np.asarray(byte) % 4 == 0).all(), byte
        return param, byte // 4

    def evaluate(
        self, name: str, inputs: dict[str, np.ndarray | int]
    ) -> np.ndarray | np.int64:
        """Evaluate an i32 index value with numpy, given values (scalars or
        arrays that broadcast) for the work-item ids and loop counters it
        reads, keyed by SSA name. ``sdiv`` / ``srem`` by a positive divisor
        round toward zero, as LLVM's do, so an index below zero (a reversed
        view addresses back from its pointer) divides as on the GPU; a
        ``readfirstlane`` passes its operand through, which is right for the
        wave-uniform values the lowering reads that way. An ``nsw`` ``add``,
        ``sub`` or ``mul`` of i32 values whose result leaves the i32 range is
        poison in LLVM, and a literal operand of one outside the range is
        another number to LLVM; either fails the test
        (:func:`rocke.core.ir.fits_i32`). An instruction this does not know
        fails the test too."""
        if not name.startswith("%"):
            return np.int64(name)
        if name in inputs:
            return inputs[name]
        op = self.opcode(name)
        args = [self.evaluate(v, inputs) for v in self.operands(name)]
        if op in ("sext", "zext", "trunc", "llvm.amdgcn.readfirstlane"):
            return args[0]
        if op in ("add", "sub", "mul"):
            left, right = args
            result = {"add": left + right, "sub": left - right, "mul": left * right}[op]
            definition = self.definitions[name]
            if definition.startswith(f"{op} nsw i32 "):
                # Operands too: LLVM reads a literal outside i32 as another number.
                for value in (left, right, result):
                    assert _all_fit_i32(
                        value
                    ), f"{name} = {definition} leaves i32: {value}"
            return result
        if op == "shl":
            return args[0] << args[1]
        if op == "or":
            return args[0] | args[1]
        if op in ("sdiv", "srem"):
            dividend, divisor = args
            assert divisor > 0, self.definitions[name]
            quotient = np.sign(dividend) * (np.abs(dividend) // divisor)
            return quotient if op == "sdiv" else dividend - quotient * divisor
        raise AssertionError(f"not modelled: {self.definitions[name]}")
