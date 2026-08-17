# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Code-composition IR nodes (Module, KernelBody, Label, TextBlock, etc.).

Module.to_stinky_asm() is the left-path entry point: walks the item tree,
builds a LogicalModule, and routes through lower_logical_module for emission.
"""

from __future__ import annotations

import copy as _copy
import re as _re
from typing import Any, Iterable, List, Optional, Sequence

from ._dummy import make_dummy_class
from .base import (
    Item,
    isaToGfx as _isa_to_gfx,
    outputNoComment as _outputNoComment,
    setVgprIdx as _set_vgpr_idx,
    setVgprMsb as _set_vgpr_msb,
)
from .enum import SignatureValueKind as _SVK
from .instruction import (
    Instruction as _Instruction,
    MacroInstruction as _MacroInstruction,
)

_P = "rocisa.code"


# ---------------------------------------------------------------------------
# Wait-instruction post-processing
# ---------------------------------------------------------------------------
# stinkytofu's Python LogicalModule path does not support SWaitCntData
# modifiers, so the gfx12+ legalization pass cannot split s_waitcnt into
# individual s_wait_loadcnt / s_wait_dscnt etc.  As a workaround, we encode
# the target wait type in the comment of SWaitCnt logical instructions via
# NUL-prefixed markers and post-process the emitted assembly text here.

from .instruction import (
    _WAIT_MARKER_LOADCNT,
    _WAIT_MARKER_STORECNT,
    _WAIT_MARKER_DSCNT,
    _WAIT_MARKER_KMCNT,
)

_WAIT_MARKER_MAP = {
    _WAIT_MARKER_LOADCNT: "s_wait_loadcnt",
    _WAIT_MARKER_STORECNT: "s_wait_storecnt",
    _WAIT_MARKER_DSCNT: "s_wait_dscnt",
    _WAIT_MARKER_KMCNT: "s_wait_kmcnt",
}

_WAIT_MARKER_RE = _re.compile(
    r"^(\s*)s_waitcnt\s+(\d+)\s*//\s*\x00@W([LSDK])@(.*)$"
)

_MARKER_CODE_TO_OPCODE = {
    "L": "s_wait_loadcnt",
    "S": "s_wait_storecnt",
    "D": "s_wait_dscnt",
    "K": "s_wait_kmcnt",
}


def _postprocess_wait_markers(asm: str) -> str:
    """Replace marker-annotated ``s_waitcnt N`` lines with gfx12+ waits."""
    lines = asm.split("\n")
    out: List[str] = []
    for line in lines:
        m = _WAIT_MARKER_RE.match(line)
        if m:
            indent = m.group(1)
            count = m.group(2)
            code = m.group(3)
            comment = m.group(4) or ""
            opcode = _MARKER_CODE_TO_OPCODE[code]
            new_line = f"{indent}{opcode} {count}"
            if comment:
                pad = max(1, 51 - len(new_line))
                new_line += " " * pad + "// " + comment
            out.append(new_line)
        else:
            out.append(line)
    return "\n".join(out)


_VCMPX_RE = _re.compile(
    r"^(\s*)(v_cmpx_\w+)\s+(.+?)(\s*(//.*?))?$"
)

_SBARRIER_RE = _re.compile(
    r"^(\s*)s_barrier\s*(//.+)?$"
)

_CARRY_RE = _re.compile(
    r"^(\s*)(v_(?:add|sub)_co(?:_ci)?_u32)\s+(.+?)(\s*(//.*?))?$"
)


def _postprocess_vcmpx(asm: str) -> str:
    """Expand ``v_cmpx_*`` into ``v_cmp_* + s_mov_b32 exec_lo`` for gfx1250."""
    lines = asm.split("\n")
    out: List[str] = []
    for line in lines:
        m = _VCMPX_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent = m.group(1)
        mnemonic = m.group(2)
        operands_str = m.group(3)
        comment = m.group(5) or ""

        cmp_mnemonic = mnemonic.replace("_cmpx_", "_cmp_", 1)
        operands = [op.strip() for op in operands_str.split(",")]

        if len(operands) >= 3:
            dst = operands[0]
            srcs = ", ".join(operands[1:])
            if dst in ("exec_lo", "exec_hi", "exec"):
                dst = "vcc_lo"
        else:
            dst = "vcc_lo"
            srcs = ", ".join(operands)

        cmp_line = f"{indent}{cmp_mnemonic} {dst}, {srcs}"
        mov_line = f"{indent}s_mov_b32 exec_lo, {dst}"
        if comment:
            cmp_pad = max(1, 51 - len(cmp_line))
            cmp_line += " " * cmp_pad + comment
            mov_pad = max(1, 51 - len(mov_line))
            mov_line += " " * mov_pad + comment
        out.append(cmp_line)
        out.append(mov_line)
    return "\n".join(out)


def _postprocess_sbarrier(asm: str) -> str:
    """Expand ``s_barrier`` into ``s_barrier_signal -1 + s_barrier_wait -1`` for gfx1250."""
    lines = asm.split("\n")
    out: List[str] = []
    for line in lines:
        m = _SBARRIER_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent = m.group(1)
        comment_part = m.group(2) or ""
        sig_line = f"{indent}s_barrier_signal -1"
        wait_line = f"{indent}s_barrier_wait -1"
        if comment_part:
            comment_text = comment_part
            pad = max(1, 51 - len(wait_line))
            wait_line += " " * pad + comment_text
        out.append(sig_line)
        out.append(wait_line)
    return "\n".join(out)


def _postprocess_carry(asm: str) -> str:
    """Insert ``vcc_lo`` carry operand into ``v_add_co_u32``/``v_sub_co_u32``/``v_add_co_ci_u32``."""
    lines = asm.split("\n")
    out: List[str] = []
    for line in lines:
        m = _CARRY_RE.match(line)
        if not m:
            out.append(line)
            continue
        indent = m.group(1)
        mnemonic = m.group(2)
        operands_str = m.group(3)
        comment = m.group(5) or ""

        operands = [op.strip() for op in operands_str.split(",")]

        is_ci = "_ci_" in mnemonic
        if is_ci and len(operands) == 3:
            # v_add_co_ci_u32 dst, src0, src1 → dst, vcc_lo, src0, src1, vcc_lo
            new_ops = f"{operands[0]}, vcc_lo, {operands[1]}, {operands[2]}, vcc_lo"
        elif not is_ci and len(operands) == 3:
            # v_add_co_u32 / v_sub_co_u32 dst, src0, src1 → dst, vcc_lo, src0, src1
            new_ops = f"{operands[0]}, vcc_lo, {operands[1]}, {operands[2]}"
        else:
            out.append(line)
            continue

        new_line = f"{indent}{mnemonic} {new_ops}"
        if comment:
            pad = max(1, 51 - len(new_line))
            new_line += " " * pad + comment
        out.append(new_line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Post-process: restore s_delay_alu from SNop placeholders
# ---------------------------------------------------------------------------

_DELAY_ALU_PLACEHOLDER_RE = _re.compile(
    r"^(\s*)s_nop 0\s+//\s*DELAY_ALU:(.+?)(?:\s*<This is \d+-cycle>)?$"
)


def _postprocess_delay_alu_placeholder(asm: str) -> str:
    """Restore s_delay_alu instructions from SNop(0) placeholders.

    During stinkytofu lowering, SDelayAlu instructions are emitted as
    ``s_nop 0  // DELAY_ALU:instid0(...)`` because the Python binding cannot
    properly lower SDelayAlu (SDelayAluData assertion).  This pass restores
    the original ``s_delay_alu`` text from the placeholder comments.
    """
    lines = asm.split("\n")
    out: List[str] = []
    for line in lines:
        m = _DELAY_ALU_PLACEHOLDER_RE.match(line)
        if m:
            indent = m.group(1)
            alu_operands = m.group(2)
            out.append(f"{indent}s_delay_alu {alu_operands}")
        else:
            out.append(line)
    return "\n".join(out)


class _PostProcessModule:
    """Wraps a StinkyAsmModule and post-processes emitAssembly() output.

    Applies transforms:
    1. Wait-marker expansion (s_waitcnt → s_wait_loadcnt/etc)
    2. VCmpX expansion (v_cmpx_* → v_cmp_* + s_mov_b32 exec_lo)
    3. s_delay_alu placeholder restoration (SNop → s_delay_alu)
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Any, insert_delay_alu: bool = False,
                 set_directives: str = "") -> None:
        self._inner = inner

    def runOptimizationPipeline(self) -> None:
        self._inner.runOptimizationPipeline()

    def emitAssembly(self) -> str:
        asm = self._inner.emitAssembly()
        asm = _postprocess_wait_markers(asm)
        asm = _postprocess_vcmpx(asm)
        asm = _postprocess_sbarrier(asm)
        asm = _postprocess_carry(asm)
        asm = asm.replace("+-", "-")
        return asm

    def getSetDirectives(self) -> str:
        """Legacy accessor — .set directives are now emitted inline."""
        return ""

    def getName(self) -> str:
        return self._inner.getName()

    def setOutputName(self, name: str) -> None:
        self._inner.setOutputName(name)

    def getOutputName(self) -> str:
        return self._inner.getOutputName()

    def setOutputDir(self, directory: str) -> None:
        self._inner.setOutputDir(directory)

    def getOutputDir(self) -> str:
        return self._inner.getOutputDir()

    def getMetaDataU64(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner.getMetaDataU64(*args, **kwargs)


# ---------------------------------------------------------------------------
# TextBlock -- raw text leaf node.
# ---------------------------------------------------------------------------
#
# Mirrors ``rocisa::TextBlock`` (code.hpp:133-160 + code.cpp:140-155). The
# binding boils down to:
#   - ctor ``TextBlock(text) : Item(text), text(text)``       -- name == text
#   - ``toString()`` returns ``text`` unless ``outputNoComment`` is set
#   - ``__getstate__`` / ``__setstate__`` round-trip ``(name, text)``
#   - ``prettyPrint`` inherits ``Item::prettyPrint`` -- ``indent + className
#     + " " + toString()`` with NO trailing newline (we now actually
#     inherit it via the Python ``Item`` base class rather than re-
#     declaring it -- matches the C++ inheritance shape one-for-one)
# We mirror every one of these points so an adapter swap is byte-identical
# at both the asm-emit layer (``Module.toString``) and the debug-dump layer
# (``Module.prettyPrint``).
class TextBlock(Item):
    """Raw text leaf; mirror of ``rocisa::TextBlock`` (code.hpp:133-160).

    Created internally by ``Module.addSpaceLine`` / ``Module.addComment*``
    and also constructed directly by KernelWriter for inline asm / macro
    snippets. Has no logicalIR counterpart -- ``Module.to_stinky_asm``
    silently skips TextBlock items.

    Production-build comment suppression: ``rocIsa.getOutputOptions().
    outputNoComment = True`` causes ``toString()`` to return ``""`` for
    every TextBlock (rocisa code.hpp:154-159 -- the flag is a blanket
    suppressor, not just for ``// foo`` comments; inline-asm TextBlocks
    are also stripped). We read the flag through ``base.outputNoComment()``
    so the adapter stays decoupled from the rocisa C++ singleton.

    Inheritance: subclass of ``Item`` -- ``name`` / ``parent`` come from
    the Item base (its ``__slots__`` provides them; redeclaring them
    here would TypeError). ``__str__`` / ``prettyPrint`` are inherited
    too (Item's defaults match TextBlock's required format byte-for-
    byte: ``"{indent}TextBlock {toString()}"`` via
    ``type(self).__name__``).
    """

    __slots__ = ("text",)

    def __init__(self, text: str = ""):
        # rocisa C++: ``TextBlock(text) : Item(text), text(text)`` -- the
        # base ``Item(name)`` ctor stores ``name = text``. Match exactly so
        # ``Module.findNamedItem`` / ``removeItemsByName`` see identical
        # behaviour on both backends. (Default text="" still gives name="",
        # so the common addComment / addSpaceLine path is unchanged.)
        super().__init__(name=text)
        self.text: str = text

    def toString(self) -> str:
        # rocisa code.hpp:154-159 -- the ``outputNoComment`` flag blanket-
        # suppresses TextBlock output regardless of whether the text is a
        # comment or an inline-asm fragment. Production builds rely on
        # this to strip every human-readable annotation in one pass.
        if _outputNoComment():
            return ""
        return self.text

    def __deepcopy__(self, memo):
        # rocisa binding lambda (code.cpp:143-148) copies via the public
        # ctor and then patches ``name`` separately. Since our ctor already
        # sets ``name = text``, replicate the same patch path so callers
        # that mutate ``name`` post-construction (rare but legal) survive
        # the clone. ``__new__`` skips ``__init__`` so we set the Item-
        # inherited slots (name / parent) explicitly here.
        tb = TextBlock.__new__(TextBlock)
        tb.name = self.name
        tb.text = self.text
        tb.parent = None  # cloned subtree gets re-parented by Module.__deepcopy__
        memo[id(self)] = tb
        return tb

    def __getstate__(self) -> tuple:
        # rocisa code.cpp:149-150 -- ``make_tuple(name, text)``.
        return (self.name, self.text)

    def __setstate__(self, state: tuple) -> None:
        # rocisa code.cpp:151-154 -- rebuild via ``TextBlock(text)`` (which
        # sets name=text), then patch ``name`` from the stored value. The
        # two-step matters when a TextBlock was renamed after construction.
        # ``name`` / ``parent`` are Item-inherited slots; they exist on
        # the unpickled instance even though ``__init__`` wasn't called.
        name, text = state
        self.name = name
        self.text = text
        self.parent = None

    def __repr__(self) -> str:
        return f"TextBlock({self.text!r})"


# ---------------------------------------------------------------------------
# Comment formatters -- low-stakes approximations of rocisa's format.hpp.
# ---------------------------------------------------------------------------
#
# rocisa C++ has slash() / slash50() / block() / blockNewLine() / block3Line()
# in format.hpp. They control human-readable comment layout, NOT instruction
# emit, so byte-parity is not on the vertical-slice critical path; KernelWriter
# only uses them for headers / dividers. We keep the formats simple and
# distinct (so a comment never silently merges with adjacent text) and document
# the byte-parity caveat. Lift these into a dedicated module + tighten formats
# once a KernelWriter diff exposes a mismatch.
def _slash(comment: str) -> str:
    """``// COMMENT`` on its own line. Used by ``Module.addComment``."""
    return f"// {comment}\n"


def _slash50(comment: str) -> str:
    """``// COMMENT`` aligned to col 50 -- mirrors instruction-line layout."""
    return f"{'':<50} // {comment}\n"


def _block(comment: str) -> str:
    """Single-line block comment ``/* COMMENT */`` — matches native format.hpp::block()."""
    return f"/* {comment} */\n"


def _block_newline(comment: str) -> str:
    """Same as ``_block`` but with a leading blank line."""
    return "\n" + _block(comment)


def _block_3line(comment: str) -> str:
    """Star-bar block comment — matches native format.hpp::block3Line().

    Produces: newline + star-bar + per-line ``/* text (padded to 38) */`` + star-bar.
    """
    bar = "/" + "*" * 42 + "/"
    lines = comment.split("\n")
    out = "\n" + bar + "\n"
    for line in lines:
        out += f"/* {line:<38} */\n"
    out += bar + "\n"
    return out


def _format_endif_str(instr: str, comment: str) -> str:
    """Format an instruction line with an optional trailing comment.

    Used by ``ValueEndif.toString`` for the ``.endif [// <comment>]``
    rendering. Layout rules:

      * ``comment`` empty OR ``_outputNoComment()`` returns True ->
        ``"{instr}\\n"`` with no padding.
      * Otherwise: ``instr`` is right-padded with spaces to width 50
        (``max(0, 50 - len(instr))`` spaces), then ``" // {comment}\\n"``
        is appended. Padding width 50 matches the column where rocisa
        instruction lines align their trailing ``// ...`` notes.

    Currently used only by ``ValueEndif``; Phase 5 (assembly emit)
    will need the full surface (including an ``outputInlineAsm``
    branch that wraps the instruction string in ``"...\\n\\t"`` for
    inline-asm output). When that lands, lift this into a public
    ``format.py`` module; for now keeping it private to ``code.py``
    keeps the surface area minimal.
    """
    if not comment or _outputNoComment():
        return instr + "\n"
    padding = " " * max(0, 50 - len(instr))
    return f"{instr}{padding} // {comment}\n"


def _to_hex_parity(num: int) -> str:
    """Lowercase hex (no ``0x`` prefix) mirroring rocisa's ``std::hex``
    cast over an ``int64_t``.

    Used by ``ValueSet.toString`` for the ``format == 1`` branch.
    Rocisa builds the payload as ``"0x" + std::hex(value + offset)``
    where the operand is ``int64_t`` and ``std::hex`` prints the raw
    two's-complement bits with no minus sign. Python's ``f"{-1:x}"``
    would emit ``"-1"`` instead, diverging for negative inputs. Mask
    to 64 bits to recover byte-parity.

    Non-negative values are unaffected (the mask is a no-op for any
    int64_t-representable non-negative value).
    """
    if num < 0:
        num &= 0xFFFFFFFFFFFFFFFF
    return f"{num:x}"


# ---------------------------------------------------------------------------
# Preprocessor conditional blocks -- ValueIf / ValueElseIf / ValueEndif.
# ---------------------------------------------------------------------------
#
# Mirror of rocisa's ``ValueIf`` / ``ValueElseIf`` / ``ValueEndif``.
# These produce the GNU assembler preprocessor directives ``.if`` /
# ``.elseif`` / ``.endif`` that KernelWriter uses to gate macro /
# kernel-text sections at assemble time (CustomSchedule.py:448-508
# chains them; KernelWriterAssembly.py:1827 uses a single ValueEndif
# for the "overflowed resources" guard).
#
# Parity notes:
#   * ``Item.name`` is set to the CLASS NAME ("ValueIf" / ... ) rather
#     than to the condition expression -- matches rocisa's ctors which
#     pass ``Item("ValueIf")``. This means
#     ``findNamedItem("ValueIf")`` matches every ValueIf node in a
#     Module; KernelWriter doesn't rely on that today but the parity
#     keeps any future searcher behaviour identical.
#   * Subclasses of ``Item`` -- ``__str__`` / ``prettyPrint`` /
#     ``countType`` / ``countExactType`` / 7 cap-proxy methods all
#     come from Item's defaults. We override only ``toString``,
#     ``__deepcopy__``, ``__getstate__``, ``__setstate__`` -- the
#     same four overrides rocisa's nanobind binding wires up
#     explicitly.
#   * ValueIf / ValueElseIf store a ``value`` (the condition
#     expression); ValueEndif stores a ``comment`` and uses
#     ``_format_endif_str`` to byte-match rocisa's ``formatStr``
#     padding semantics.

class ValueIf(Item):
    """``.if <value>`` directive; mirror of ``rocisa::ValueIf``."""

    __slots__ = ("value",)

    def __init__(self, value: str):
        # ``Item.name`` is the literal "ValueIf" (class name), NOT the
        # condition expression -- matches rocisa's ctor.
        super().__init__(name="ValueIf")
        self.value: str = value

    def toString(self) -> str:
        # Raw ``.if`` + value + newline; no padding / comment support
        # (the condition expression IS the trailing payload on this
        # line).
        return f".if {self.value}\n"

    def __deepcopy__(self, memo):
        # Copy ctor -- a fresh ValueIf with the same value.
        clone = ValueIf(self.value)
        memo[id(self)] = clone
        return clone

    def __getstate__(self) -> str:
        # Pickle just the value string.
        return self.value

    def __setstate__(self, state: str) -> None:
        # Rebuild as ``ValueIf(value)``. Have to populate Item-
        # inherited slots manually since ``__init__`` isn't called
        # by the pickle machinery.
        self.name = "ValueIf"
        self.parent = None
        self.value = state

    def __repr__(self) -> str:
        return f"ValueIf({self.value!r})"


class ValueElseIf(Item):
    """``.elseif <value>`` directive; mirror of ``rocisa::ValueElseIf``."""

    __slots__ = ("value",)

    def __init__(self, value: str):
        # ``Item.name`` is the literal "ValueElseIf" (class name), NOT
        # the condition expression -- matches rocisa's ctor.
        super().__init__(name="ValueElseIf")
        self.value: str = value

    def toString(self) -> str:
        # Raw ``.elseif`` + value + newline.
        return f".elseif {self.value}\n"

    def __deepcopy__(self, memo):
        clone = ValueElseIf(self.value)
        memo[id(self)] = clone
        return clone

    def __getstate__(self) -> str:
        return self.value

    def __setstate__(self, state: str) -> None:
        self.name = "ValueElseIf"
        self.parent = None
        self.value = state

    def __repr__(self) -> str:
        return f"ValueElseIf({self.value!r})"


class ValueEndif(Item):
    """``.endif [// <comment>]`` directive; mirror of
    ``rocisa::ValueEndif``.

    The comment is padding-aligned to column 50 to match how rocisa
    instruction lines align their trailing ``// ...`` notes, and is
    suppressed entirely when ``outputNoComment`` is set (see
    ``_format_endif_str`` for the exact rules).
    """

    __slots__ = ("comment",)

    def __init__(self, comment: str = ""):
        # Default ``comment=""`` matches the most common KernelWriter
        # call site (a bare ``ValueEndif()`` closing an .if block);
        # ``Item.name`` is the literal "ValueEndif" -- matches rocisa's
        # ctor.
        super().__init__(name="ValueEndif")
        self.comment: str = comment

    def toString(self) -> str:
        # ``.endif`` + optional ``// <comment>`` padded to column 50;
        # gated by ``outputNoComment``. See ``_format_endif_str`` for
        # the byte-level rules.
        return _format_endif_str(".endif", self.comment)

    def __deepcopy__(self, memo):
        clone = ValueEndif(self.comment)
        memo[id(self)] = clone
        return clone

    def __getstate__(self) -> str:
        # Pickle just the comment string (which is "" for the bare-
        # ``ValueEndif()`` common case).
        return self.comment

    def __setstate__(self, state: str) -> None:
        self.name = "ValueEndif"
        self.parent = None
        self.comment = state

    def __repr__(self) -> str:
        return f"ValueEndif({self.comment!r})"


# ---------------------------------------------------------------------------
# Label -- branch / loop target leaf.
# ---------------------------------------------------------------------------
#
# Mirror of rocisa's ``Label``. Emits an assembler label line of the
# form ``label_<name>:``, optionally preceded by ``.align <N>`` and
# followed by ``  /// <comment>`` (the comment uses ``///`` not ``//``
# so it survives later passes that strip ordinary ``//`` comments).
#
# KernelWriter creates Labels through ``LabelManager.getName(...)``
# (or ``getUniqueName*``) to guarantee uniqueness, then pairs each
# Label with one or more branch instructions (``s_branch``,
# ``s_cbranch_*``) that target it. The Label's ``getLabelName()``
# is the string those branches reference.
#
# Parity notes:
#   * ``Item.name`` is the EMPTY STRING (matches rocisa's ``Item("")``
#     ctor argument), NOT the label payload. The actual label text
#     comes from ``getLabelName()`` -> ``getFormatting(self.label)``.
#     This means ``findNamedItem("Label")`` does NOT match Labels --
#     KernelWriter relies on ``getLabelName()`` for the textual
#     identity instead.
#   * The ``label`` field carries an ``int | str`` payload (``rocisa::
#     Label`` uses ``std::variant<std::string, int>``). Python's union
#     types make this transparent -- we just store the value as-is
#     and dispatch in ``getFormatting`` via ``isinstance(label, int)``.
#   * ``toString`` is intentionally side-effecting: when ``HasVgprMSB``
#     is set, the active VGPR-MSB tracking is reset to ``-1``. The
#     reasoning is that emitting a label means "we're entering a new
#     basic block whose entry MSB state is not knowable at this point
#     in the rocisa-side analysis" -- the next instruction must
#     re-establish MSB explicitly. Consumers must NOT rely on the
#     pre-toString value of ``getVgprMsb()`` surviving across a
#     ``Label.toString()`` call when the cap is set.
#   * Comment formatting: ``"  /// <comment>"`` (three slashes, two
#     leading spaces). Suppressed entirely when ``outputNoComment``
#     is set, matching rocisa.
#   * ``alignment <= 1`` means "no .align prefix"; ``alignment > 1``
#     emits a ``.align <N>\\n`` line BEFORE the label. KernelWriter
#     uses ``alignment > 1`` for loop-entry labels (cache-line
#     alignment) and the default ``alignment=1`` for ordinary
#     branch targets.

class Label(Item):
    """``label_<id>:`` assembler label; mirror of ``rocisa::Label``.

    Carries a ``label`` payload (``int | str``) plus an optional
    ``comment`` and an ``alignment`` (default 1; ``> 1`` prepends
    ``.align <N>\\n`` to the emission). See the section header above
    for parity rules and the ``setVgprMsb(-1)`` side effect.
    """

    __slots__ = ("label", "comment", "alignment")

    def __init__(self, label, comment: str, alignment: int = 1):
        # ``Item.name`` is the EMPTY STRING (matches rocisa's
        # ``Item("")``) -- the label's textual identity comes from
        # ``getLabelName()``, not from Item.name.
        super().__init__(name="")
        self.label = label
        self.comment: str = comment
        self.alignment: int = alignment

    @staticmethod
    def getFormatting(label) -> str:  # noqa: N802 (matches rocisa public API)
        """Format an ``int | str`` payload into ``label_<text>``.

        rocisa ``Label::getFormatting`` is a static method visiting
        the ``std::variant<string, int>`` and returning ``"label_"``
        prefixed by the numeric or string value. Python's f-string
        formatting handles both branches uniformly, so a single
        format expression is enough -- the variant dispatch is
        implicit in ``__format__``.
        """
        return f"label_{label}"

    def getLabelName(self) -> str:  # noqa: N802 (matches rocisa public API)
        """Return the full ``label_<...>`` text for this Label.

        Equivalent to ``Label.getFormatting(self.label)``. Branch
        instructions reference the result of this call (NOT
        ``self.name`` or ``self.label`` directly).
        """
        return Label.getFormatting(self.label)

    def toString(self) -> str:
        body = self.getLabelName() + ":"
        if self.alignment > 1:
            body = f".align {self.alignment}\n" + body
        if self.comment and not _outputNoComment():
            body += "  /// " + self.comment
        body += "\n"
        # Side effect: emitting a label resets the in-process
        # VGPR-MSB tracker. See section header note for the
        # justification (entering a new basic block).
        if self.getAsmCaps().get("HasVgprMSB", 0):
            _set_vgpr_msb(-1)
        return body

    def __deepcopy__(self, memo):
        clone = Label(self.label, self.comment, self.alignment)
        # Item.name is "" by default but rocisa's copy ctor preserves
        # whatever the original Item had -- patch it explicitly here
        # so any caller-set ``label.name = ...`` round-trips.
        clone.name = self.name
        memo[id(self)] = clone
        return clone

    def __getstate__(self):
        # 4-tuple matches rocisa pickle order:
        # ``(name, label, comment, alignment)``.
        return (self.name, self.label, self.comment, self.alignment)

    def __setstate__(self, state) -> None:
        # Rebuild via the public ctor (so any ctor-level invariants
        # apply) then patch ``name`` -- mirrors rocisa's
        # ``new(&self) Label(label, comment, alignment); self.name = name``
        # placement-new pattern.
        name, label, comment, alignment = state
        self.__init__(label, comment, alignment)
        self.name = name

    def __repr__(self) -> str:
        return (
            f"Label({self.label!r}, comment={self.comment!r}, "
            f"alignment={self.alignment})"
        )


# ---------------------------------------------------------------------------
# Symbol-emission nodes -- ValueSet / RegSet.
# ---------------------------------------------------------------------------
#
# Mirror of rocisa's ``ValueSet`` (assembler ``.set`` directive) and
# ``RegSet`` (a ``ValueSet`` subclass that also tracks VGPR
# allocation in the rocIsa singleton for MSB-aware archs).
#
# These produce GNU assembler ``.set <name>, <payload>`` lines that
# KernelWriter sprinkles throughout a kernel to give names to integer
# constants and register aliases. RegSet adds the convention that
# ``.set vgprX, N`` ALSO informs the in-process index map of the
# binding so subsequent ``getVgprIdx()`` lookups see the latest
# allocation.
#
# Parity notes:
#   * C++ has 3 ValueSet ctors (int / uint32 / string ref) and 2
#     RegSet ctors (int / string value). Python collapses each into
#     a single ``__init__`` that dispatches on
#     ``isinstance(value, str)`` -- this matches what nanobind's
#     overload resolution does at runtime based on Python type.
#   * The ``uint32`` overload was a C++-side type-system concern only.
#     Python ints are arbitrary precision so both routes share the
#     int path; ``toString`` produces identical output either way.
#   * ``format`` is a raw int sentinel: ``-1`` = literal, ``0`` =
#     decimal-with-offset (default), ``1`` = hex-with-offset. Kept
#     as a plain int (not an IntEnum) to match rocisa's API surface
#     -- KernelWriter passes 0/1/-1 directly.
#   * Negative ``value + offset`` with ``format == 1`` mirrors
#     rocisa's ``std::hex`` over int64_t two's-complement -- see
#     ``_to_hex_parity``. Non-negative payloads are unaffected.
#   * ``RegSet.__init__`` AND ``RegSet.toString`` both call
#     ``setVgprIdx`` when ``regType == "v"`` on a HasVgprMSB arch.
#     ``toString`` is intentionally NOT pure here: KernelWriter
#     relies on each emitted ``.set vgprX, N`` line ALSO refreshing
#     the in-memory index map as the asm text is built.
#   * ``self.name[4:]`` strips the conventional ``"vgpr"`` prefix
#     before registering the index. rocisa does not validate that
#     the prefix is present; neither do we.

class ValueSet(Item):
    """``.set <name>, <value-or-ref>`` directive; mirror of
    ``rocisa::ValueSet``.

    Holds either an integer ``value`` OR a string ``ref`` (mutually
    exclusive); the unused field is ``None``. ``offset`` is added to
    the payload, and ``format`` controls the rendering:

        ``format == -1``  ->  literal ref / ``str(value)`` (no offset
                              arithmetic on the value path)
        ``format ==  0``  ->  decimal ``str(value + offset)`` (default)
        ``format ==  1``  ->  ``"0x" + hex(value + offset)``

    See ``_to_hex_parity`` for the negative-value byte-parity rules.
    """

    __slots__ = ("ref", "value", "offset", "format")

    def __init__(
        self,
        name: str,
        value,
        offset: int = 0,
        format: int = 0,  # noqa: A002  (matches rocisa's public arg name)
    ):
        # Single Python ctor dispatching on the runtime type of
        # ``value`` -- matches what nanobind's 3-overload resolution
        # does at runtime. ``isinstance(value, str)`` discriminates
        # the ref-payload path from the int-payload path; the C++
        # ``int`` vs ``uint32_t`` overloads collapse into a single
        # Python int route (arbitrary precision -- no truncation).
        super().__init__(name=name)
        if isinstance(value, str):
            self.ref: Optional[str] = value
            self.value: Optional[int] = None
        else:
            self.ref = None
            self.value = int(value)
        self.offset: int = offset
        self.format: int = format

    def toString(self) -> str:
        body = f".set {self.name}, "
        if self.ref is not None:
            # ref path: ``format == -1`` emits the ref alone; any other
            # format suffixes ``+<offset>``. (rocisa does NOT short-
            # circuit ``offset == 0`` to ``ref`` alone -- the literal
            # ``+0`` is preserved for byte parity with the assembler.)
            if self.format == -1:
                body += self.ref
            else:
                body += f"{self.ref}+{self.offset}"
        elif self.value is not None:
            # value path: ``format == -1`` emits ``str(value)`` with
            # NO offset arithmetic (mirrors C++ which calls
            # ``std::to_string(value)`` directly in that branch);
            # ``format == 0`` adds the offset; ``format == 1`` adds
            # the offset and emits hex.
            v = self.value if self.format == -1 else self.value + self.offset
            if self.format == 1:
                body += f"0x{_to_hex_parity(v)}"
            else:
                body += str(v)
        # ``ref is None and value is None`` is unreachable under the
        # ctor invariant (one of the two is always set); rocisa would
        # dereference an empty optional and crash here.
        return body + "\n"

    def __deepcopy__(self, memo):
        if self.ref is not None:
            clone = ValueSet(self.name, self.ref, self.offset, self.format)
        else:
            clone = ValueSet(self.name, self.value, self.offset, self.format)
        memo[id(self)] = clone
        return clone

    def __getstate__(self):
        # 5-tuple: ``(name, ref, value, offset, format)``.
        return (self.name, self.ref, self.value, self.offset, self.format)

    def __setstate__(self, state) -> None:
        # Restore slots directly (no side effect to mirror, matching
        # how ValueIf / ValueElseIf / ValueEndif round-trip).
        name, ref, value, offset, fmt = state
        self.name = name
        self.parent = None
        self.ref = ref
        self.value = value
        self.offset = offset
        self.format = fmt

    def __repr__(self) -> str:
        payload = self.ref if self.ref is not None else self.value
        return (
            f"ValueSet({self.name!r}, {payload!r}, "
            f"offset={self.offset}, format={self.format})"
        )


class RegSet(ValueSet):
    """``.set <name>, <value-or-ref>`` plus VGPR-index tracking;
    mirror of ``rocisa::RegSet``.

    Adds a ``regType`` field (``"v"`` for VGPR, ``"s"`` for SGPR,
    etc.) on top of ``ValueSet``. When ``regType == "v"`` AND the
    active arch has ``HasVgprMSB`` set, BOTH ``__init__`` and every
    call to ``toString`` invoke ``setVgprIdx`` on the rocIsa
    singleton so later ``getVgprIdx()`` lookups observe the latest
    allocation.

    The conventional ``"vgpr"`` prefix on ``name`` (e.g.
    ``"vgprFoo"``) is stripped before registering -- the key in the
    index map is ``name[4:]``.

    ``toString`` is intentionally side-effecting: KernelWriter relies
    on each emitted ``.set vgprX, N`` line refreshing the in-memory
    index map as the asm text is built, not just at construction
    time.
    """

    __slots__ = ("regType",)

    def __init__(
        self,
        regType: str,
        name: str,
        value,
        offset: int = 0,
    ):
        # No ``format`` parameter at the RegSet ctor surface (matches
        # rocisa -- the two C++ RegSet ctors only forward
        # ``(name, value, offset)`` to ValueSet, leaving ``format``
        # at its default 0). Callers can patch ``self.format``
        # post-construction; pickle round-trip preserves it.
        super().__init__(name=name, value=value, offset=offset)
        self.regType: str = regType
        if self._vgpr_msb_active():
            self._set_idx(value, offset)

    def toString(self) -> str:
        # Re-trigger the index update every time. Mirrors rocisa's
        # ``RegSet::toString`` which first re-runs ``setIdx`` then
        # delegates the actual string formatting to ``ValueSet``.
        if self._vgpr_msb_active():
            if self.ref is not None:
                self._set_idx(self.ref, self.offset)
            elif self.value is not None:
                self._set_idx(self.value, self.offset)
        return super().toString()

    def _vgpr_msb_active(self) -> bool:
        """Cap-gated guard for the ``setVgprIdx`` side effect.

        ``getAsmCaps()`` returns a dict; missing keys default to 0
        (matches ``std::map<string,int>::operator[]`` which value-
        initialises on access).
        """
        return (
            self.regType == "v"
            and bool(self.getAsmCaps().get("HasVgprMSB", 0))
        )

    def _set_idx(self, value, offset: int) -> None:
        # Two cases mirror the ``setIdx(int, int)`` and
        # ``setIdx(string, int)`` C++ overloads:
        #   * int  value -> store ``value + offset`` directly
        #   * str  value -> look ``value[4:]`` up in the current
        #                   vgprIdx map then add offset (the lookup
        #                   key has the ``"vgpr"`` prefix stripped)
        #
        # IMPORTANT: rocisa uses ``std::map<string,int>::operator[]``
        # for the lookup, which value-initialises missing keys to 0
        # rather than throwing. KernelWriter's ``macroAndSet`` pattern
        # ``RegSet("v", "vgprX", "vgprX_BASE", 0)`` deliberately
        # establishes an alias at offset 0 of a not-yet-registered
        # base name, so we MUST treat missing keys as 0 to match.
        if isinstance(value, str):
            current = self.getVgprIdx()
            idx = current.get(value[4:], 0) + offset
        else:
            idx = int(value) + offset
        _set_vgpr_idx(self.name[4:], idx)

    def __deepcopy__(self, memo):
        # Use the int- or string-payload ctor depending on which is
        # set, then patch ``format`` (RegSet's ctor does not take it).
        # Mirrors the C++ copy ctor's effect.
        if self.ref is not None:
            clone = RegSet(self.regType, self.name, self.ref, self.offset)
        else:
            clone = RegSet(self.regType, self.name, self.value, self.offset)
        clone.format = self.format
        memo[id(self)] = clone
        return clone

    def __getstate__(self):
        # 6-tuple: ``(regType, name, ref, value, offset, format)``.
        return (
            self.regType,
            self.name,
            self.ref,
            self.value,
            self.offset,
            self.format,
        )

    def __setstate__(self, state) -> None:
        # Restore via ``__init__`` (so the ``setVgprIdx`` side effect
        # fires for ``regType == "v"`` on HasVgprMSB archs, matching
        # the C++ ``new(&self) RegSet(...)`` placement-new), then
        # patch ``format`` -- RegSet's ctor does not accept it.
        regType, name, ref, value, offset, fmt = state
        payload = ref if ref is not None else value
        self.__init__(regType, name, payload, offset)
        self.format = fmt

    def __repr__(self) -> str:
        payload = self.ref if self.ref is not None else self.value
        return (
            f"RegSet({self.regType!r}, {self.name!r}, {payload!r}, "
            f"offset={self.offset}, format={self.format})"
        )


# ---------------------------------------------------------------------------
# Module -- real tree-shaped IR container.
# ---------------------------------------------------------------------------
#
# Methods are 1:1 with the nanobind binding in rocisa/src/code.cpp:157-212.
# Item-handling semantics (parent rebind on add/setItem, None tolerance on
# add, ``count`` recurses into sub-Modules but stops at leaves) match
# rocisa::Module behaviour so byte-for-byte downstream emission is preserved.
#
# Items inserted here can be *anything that quacks like a rocisa Item* --
# we do not type-check on ``Item`` so the dummy instruction shims still
# work during the bring-up phase. The only hard requirement at lowering
# time is that logical-IR leaves expose ``to_stinky_logical()`` (Step 3).
class Module(Item):
    """Tree-shaped IR container mirroring ``rocisa::Module``.

    The container is identity-based for ``replaceItem`` / ``removeItem``
    (matches rocisa, which compares ``shared_ptr<Item>`` equality).
    Children with a ``.parent`` attribute are reparented to this Module
    on insertion so KernelWriter's tree-walks ascend correctly.

    Item inheritance: ``Module(Item)`` -- ``name`` / ``parent`` come
    from Item's ``__slots__``; the seven capability proxies
    (``getAsmCaps`` / ``getArchCaps`` / ``getRegCaps`` / ``getAsmBugs``
    / ``getVgprIdx`` / ``getVgprMsb`` / ``kernel``) are inherited as
    methods that forward to module-level ``base.py`` accessors.

    Property vs method note: rocisa C++ binds these as ``def_prop_ro``
    (so ``module.asmCaps`` works in C++/Python); the Python adapter
    exposes them as methods (``module.getAsmCaps()``). KernelWriter
    never reads them off Module instances today (workspace-wide grep
    for ``\\.asmCaps`` / ``\\.kernel`` on Module comes back empty -- it
    only does so on Instruction subclasses), so this method-vs-property
    asymmetry is invisible in practice. Promote to ``@property`` here
    only if KernelWriter ever starts reading them off a Module.

    Pickling: ``__reduce__`` raises ``RuntimeError("Module is not
    picklable")``, matching rocisa code.cpp -- ``ParallelMap2`` workers
    are expected to round-trip the kernel **arguments** (and let each
    worker rebuild its own Module tree from scratch), not the IR.

    Left-path entry point::

        asm_module = code_module.to_stinky_asm([12, 5, 0])
        # Or with Tensile-style kernel label:
        # asm_module = code_module.to_stinky_asm([12, 5, 0], logical_name="my_kernel")
        print(asm_module.emitAssembly())

    See module-level docstring for the architectural picture.
    """

    __slots__ = ("itemList", "tempVgpr", "_isNoOpt", "isCallable", "callableName")

    def __init__(self, name: str = ""):
        super().__init__(name=name)
        self.itemList: List[Any] = []
        self.tempVgpr: Any = None
        self._isNoOpt: bool = False
        # Mirror native rocisa Module::isCallable (bool, default false) and
        # callableName (str, default ""): mark a module as a callable function
        # body (e.g. activation func-call target for s_swappc_b64) and record
        # its label. Metadata only; consumed by KernelWriter.
        self.isCallable: bool = False
        self.callableName: str = ""

    # ------------------------------------------------------------------ add
    def add(self, item: Any, pos: int = -1) -> Any:
        """Add ``item`` to ``itemList`` and return it for chaining.

        ``None`` is silently dropped, matching rocisa's ``if(item)``
        guard (KernelWriter frequently passes optional values directly).
        """
        if item is None:
            return item
        self._reparent(item)
        if pos == -1:
            self.itemList.append(item)
        else:
            self.itemList.insert(pos, item)
        return item

    def addItems(self, items: Iterable[Any]) -> None:
        for it in items:
            self.add(it)

    def addSpaceLine(self) -> None:
        self.add(TextBlock("\n"))

    def addComment(self, comment: str) -> None:
        self.add(TextBlock(_slash(comment)))

    def addCommentAlign(self, comment: str) -> None:
        self.add(TextBlock(_slash50(comment)))

    def addComment0(self, comment: str) -> None:
        self.add(TextBlock(_block(comment)))

    def addComment1(self, comment: str) -> None:
        self.add(TextBlock(_block_newline(comment)))

    def addComment2(self, comment: str) -> None:
        self.add(TextBlock(_block_3line(comment)))

    # ---------------------------------------------------------- accessors
    def items(self) -> List[Any]:
        return self.itemList

    def itemsSize(self) -> int:
        return len(self.itemList)

    def count(self) -> int:
        """Recursive leaf count (sub-Modules expand, everything else +=1)."""
        n = 0
        for it in self.itemList:
            if isinstance(it, Module):
                n += it.count()
            else:
                n += 1
        return n

    def countType(self, type_obj: type) -> int:
        """Recursive ``isinstance`` count -- mirror of
        ``rocisa::Module::countType`` (code.hpp:441-449).

        Sums ``isinstance(self, type_obj)`` (this Module) plus the
        ``countType`` of every child. Non-Item children that happen to
        live in ``itemList`` (raw dummies during bring-up) fall back
        to a manual ``isinstance`` check so they still contribute.
        """
        n = 1 if isinstance(self, type_obj) else 0
        for it in self.itemList:
            inner_count = getattr(it, "countType", None)
            if callable(inner_count):
                res = inner_count(type_obj)
                if isinstance(res, int):
                    n += res
                    continue
            # Fallback for items that don't expose countType (e.g. raw
            # Python objects KernelWriter might stash). Still respects
            # isinstance so type checks against ``object`` work.
            if isinstance(it, type_obj):
                n += 1
        return n

    def countExactType(self, type_obj: type) -> int:
        """Recursive ``type(...) is`` count -- mirror of
        ``rocisa::Module::countExactType`` (code.hpp:451-459).

        Uses identity comparison (``type(self) is type_obj``) rather
        than ``isinstance`` so subclasses do NOT count -- a
        ``StructuredModule`` is not counted as a ``Module`` here even
        though it inherits from it. Matches ``typeid(*this) ==
        targetType`` in C++.
        """
        n = 1 if type(self) is type_obj else 0
        for it in self.itemList:
            inner_count = getattr(it, "countExactType", None)
            if callable(inner_count):
                res = inner_count(type_obj)
                if isinstance(res, int):
                    n += res
                    continue
            if type(it) is type_obj:
                n += 1
        return n

    def getItem(self, index: int) -> Any:
        # rocisa throws ``std::runtime_error("index out of range")``; we
        # match the message exactly so callers comparing exception text
        # see the same string regardless of backend.
        if index >= len(self.itemList) or index < -len(self.itemList):
            raise RuntimeError("index out of range")
        return self.itemList[index]

    def setItem(self, index: int, item: Any) -> None:
        if index >= len(self.itemList) or index < -len(self.itemList):
            raise RuntimeError("index out of range")
        self._reparent(item)
        self.itemList[index] = item

    def setItems(self, items: Sequence[Any]) -> None:
        self.itemList = list(items)
        for it in self.itemList:
            self._reparent(it)

    # ------------------------------------------------------------ find / index
    def findNamedItem(self, name: str) -> Any:
        for it in self.itemList:
            if getattr(it, "name", None) == name:
                return it
        return None

    def findIndex(self, target: Any) -> int:
        for i, it in enumerate(self.itemList):
            if it is target:
                return i
        return -1

    def findIndexByType(self, type_obj: type) -> int:
        for i, it in enumerate(self.itemList):
            if isinstance(it, type_obj):
                return i
        return -1

    # ------------------------------------------------------------- mutations
    def replaceItem(self, src: Any, dst: Any) -> None:
        for i, it in enumerate(self.itemList):
            if it is src:
                self._reparent(dst)
                self.itemList[i] = dst
                return  # rocisa replaces only the first match

    def replaceItemByIndex(self, index: int, item: Any) -> None:
        # rocisa silently no-ops when ``index >= itemList.size()``.
        if index >= len(self.itemList) or index < -len(self.itemList):
            return
        self._reparent(item)
        self.itemList[index] = item

    def removeItem(self, item: Any) -> None:
        self.itemList = [it for it in self.itemList if it is not item]

    def removeItemByIndex(self, index: int) -> None:
        if not self.itemList:
            return
        # rocisa clamps over-range indices to the last element (not an error).
        if index >= len(self.itemList):
            index = len(self.itemList) - 1
        del self.itemList[index]

    def removeItemsByName(self, name: str) -> None:
        self.itemList = [
            it for it in self.itemList if getattr(it, "name", None) != name
        ]

    def popFirstItem(self) -> Any:
        if not self.itemList:
            return None
        return self.itemList.pop(0)

    def popFirstNItems(self, n: int) -> List[Any]:
        if n >= len(self.itemList):
            popped, self.itemList = self.itemList, []
            return popped
        popped = self.itemList[:n]
        self.itemList = self.itemList[n:]
        return popped

    # ---------------------------------------------------------- tree ops
    def appendModule(self, module: "Module") -> "Module":
        for it in module.items():
            self.add(it)
        return module

    def addModuleAsFlatItems(self, module: "Module") -> "Module":
        if module is None:
            return module
        for it in module.flatitems():
            self.add(it)
        return module

    def flatitems(self) -> List[Any]:
        out: List[Any] = []
        for it in self.itemList:
            if isinstance(it, Module):
                out.extend(it.flatitems())
            else:
                out.append(it)
        return out

    def setParent(self) -> None:
        for it in self.itemList:
            self._reparent(it)
            if isinstance(it, Module):
                it.setParent()

    def setNoOpt(self, b: bool) -> None:
        self._isNoOpt = bool(b)

    def isNoOpt(self) -> bool:
        return self._isNoOpt

    def setInlineAsmPrintMode(self, mode: bool) -> None:
        for it in self.itemList:
            if isinstance(it, Module):
                it.setInlineAsmPrintMode(mode)
            elif hasattr(it, "setInlineAsm"):
                it.setInlineAsm(mode)

    def addTempVgpr(self, vgpr: Any) -> None:
        self.tempVgpr = vgpr

    # --------------------------------------------------------------- render
    def toString(self) -> str:
        return "".join(str(it) for it in self.itemList)

    def __str__(self) -> str:
        return self.toString()

    def prettyPrint(self, indent: str = "") -> str:
        """Tree dump mirroring ``rocisa::Module::prettyPrint`` (code.hpp:418-427).

        Format (byte-for-byte):

            {indent}{ClassName} "{name}"\\n        <- this Module's header
            {indent}|--child.prettyPrint(indent+"|--")
            ...

        Children are concatenated **verbatim** (no separator) -- each child
        is expected to emit its own trailing ``\\n`` (Module's header line
        does, ``Instruction.toString`` does; ``Item::prettyPrint`` does
        NOT, matching rocisa C++). Dummy shims whose ``__getattr__`` makes
        ``prettyPrint`` return ``None`` fall back to a class-name line with
        an explicit ``\\n`` so the dump stays well-formed during bring-up.
        """
        out = f'{indent}{type(self).__name__} "{self.name}"\n'
        for it in self.itemList:
            pp = getattr(it, "prettyPrint", None)
            if callable(pp):
                res = pp(indent + "|--")
                if isinstance(res, str):
                    out += res
                    continue
                # dummy ``_noop`` returned None -- fall through to fallback
            out += f"{indent}|--{type(it).__name__}\n"
        return out

    # ------------------------------------------------------------- pickle
    def __deepcopy__(self, memo):
        clone = Module(self.name)
        memo[id(self)] = clone
        clone._isNoOpt = self._isNoOpt
        clone.isCallable = self.isCallable
        clone.callableName = self.callableName
        # rocisa clones tempVgpr via Container::clone(); here we deepcopy so
        # value-typed wrappers stay independent. Non-deepcopyable objects
        # (rare) fall through to a shallow copy to match nanobind tolerance.
        if self.tempVgpr is not None:
            try:
                clone.tempVgpr = _copy.deepcopy(self.tempVgpr, memo)
            except Exception:  # noqa: BLE001  (intentionally permissive)
                clone.tempVgpr = self.tempVgpr
        for it in self.itemList:
            new_it = _copy.deepcopy(it, memo)
            clone._reparent(new_it)
            clone.itemList.append(new_it)
        return clone

    def __reduce__(self):
        # rocisa raises "Module is not picklable"; mirror that exactly so the
        # ParallelMap2 worker harness sees the same failure mode.
        raise RuntimeError("Module is not picklable")

    # ----------------------------------------------------- logicalIR handoff
    def to_stinky_asm(
        self,
        arch: Sequence[int],
        *,
        logical_name: Optional[str] = None,
        options: Any = None,
    ):
        """Lower this Module tree to a ``stinkytofu.StinkyAsmModule``.

        Walks ``itemList`` in tree order and forwards every leaf that
        exposes ``to_stinky_logical()`` (the Step-3 instruction shims)
        into a fresh ``_stinkytofu.LogicalModule``. The logical module
        is then run through ``_stinkytofu.lower_logical_module(...)``
        which wires composite expansion + ToStinkyAsmPass and produces
        a ``StinkyAsmModule`` ready for ``emitAssembly()``.

        Non-logical items (``TextBlock``, ``Label``, raw asm fragments)
        are silently skipped because they have no logical-IR counterpart.
        Once Step 3+ lands more instruction families, this also serves
        as the natural place to surface "leaf X has no
        ``to_stinky_logical``" diagnostics.

        Args:
            arch: target ISA ``[major, minor, stepping]``, e.g.
                ``[12, 5, 0]`` for gfx1250.
            logical_name: optional override for the ``LogicalModule`` /
                asm module name. When omitted, uses ``self.name`` or
                ``"kernel"`` (matches prior behaviour and pairs with
                ``rocisa.toStinkyTofuModule(..., moduleName, ...)``).

        Returns:
            ``stinkytofu.StinkyAsmModule``.

        Raises:
            ImportError: when the ``stinkytofu`` Python binding
                (``_stinkytofu.so``) is not built / on PYTHONPATH.
        """
        import stinkytofu as _st  # noqa: WPS433  (optional runtime dep)

        lm_label = logical_name if logical_name is not None else (self.name or "kernel")
        lm = _st.LogicalModule(lm_label)
        self._populate_logical_module(lm)

        return _PostProcessModule(_st.lower_logical_module(lm, list(arch), options))

    def _populate_logical_module(self, lm: Any) -> None:
        """In-order walk adding instructions and .set directives to *lm*.

        Preserves source ordering: when a ``ValueSet`` appears between two
        instructions in the Module tree, ``lm.add_set_directive(symbol, value)``
        is called at that position so the lowered BasicBlock will contain the
        ``AsmDirective(SET)`` node interleaved with instructions — matching the
        native ``toStinkyTofuModule`` behaviour.

        Named sub-Modules emit begin_group/end_group markers so the C++
        lowering pipeline can reconstruct instruction-group ranges (used by
        ScopeAdaptor passes like ESM2, RegionClone, DAG scheduler).
        """
        for it in self.itemList:
            if isinstance(it, Module):
                if it.name:
                    lm.begin_group(it.name)
                it._populate_logical_module(lm)
                if it.name:
                    lm.end_group(it.name)
                continue
            if isinstance(it, ValueSet):
                text = it.toString().strip()  # ".set <sym>, <val>"
                prefix = ".set "
                if text.startswith(prefix):
                    rest = text[len(prefix):]
                    comma = rest.find(", ")
                    if comma != -1:
                        sym = rest[:comma]
                        val = rest[comma + 2:]
                        lm.add_set_directive(sym, val)
                continue
            if isinstance(it, Label):
                lm.add_label(it.getLabelName(), it.alignment, it.comment or "")
                continue
            if isinstance(it, TextBlock):
                lm.add_textblock(it.text)
                continue
            # Skip SDelayAlu instructions — the optimization pipeline handles
            # all hazard insertion (InsertWaitAluPass for ESM2, InsertDelayAluPass
            # for non-ESM2 regions).  The adaptor previously emitted these as
            # s_nop 0 placeholders that survived the pipeline unrecognized.
            if getattr(it, "instStr", "") == "s_delay_alu":
                continue
            handle = getattr(it, "to_stinky_logical", None)
            if not callable(handle):
                continue
            logical = handle()
            if logical is None:
                continue
            comment = getattr(it, "comment", None) or ""
            if isinstance(logical, list):
                for inst in logical:
                    if comment and not inst.comment:
                        inst.comment = comment
                    lm.add(inst)
            else:
                if comment and not logical.comment:
                    logical.comment = comment
                lm.add(logical)

    def _collect_logical_insts(self) -> List[Any]:
        """Legacy helper for tests: collect logical instructions in-order."""
        out: List[Any] = []
        for it in self.itemList:
            if isinstance(it, Module):
                out.extend(it._collect_logical_insts())
                continue
            handle = getattr(it, "to_stinky_logical", None)
            if not callable(handle):
                continue
            logical = handle()
            if logical is None:
                continue
            if isinstance(logical, list):
                out.extend(logical)
            else:
                out.append(logical)
        return out

    # ---------------------------------------------------------- internal
    def _reparent(self, item: Any) -> None:
        """Best-effort ``item.parent = self`` (frozen / __slots__ tolerant)."""
        if not hasattr(item, "parent"):
            return
        try:
            item.parent = self
        except (AttributeError, TypeError):
            # Some dummy shims forbid attribute writes; not fatal for
            # rocisa-shape parity here. KernelWriter's tree-walks only
            # rely on ``parent`` for known item types.
            pass


# ---------------------------------------------------------------------------
# StructuredModule -- 3-bucket Module subclass for SIA scheduling.
# ---------------------------------------------------------------------------
#
# Mirror of rocisa's ``StructuredModule``. A Module subclass that
# carves its body into three named sub-modules -- ``header``,
# ``middle``, ``footer`` -- and auto-adds them to its ``itemList``
# at construction. The result is an Item tree where:
#
#     sm = StructuredModule("globalReadA")  # itemList = [header, middle, footer]
#     sm.middle.add(SomeLoadInstruction(...))
#     str(sm)  # emits header + middle + footer (in that order)
#
# Why this exists:
#   The Tensile SIA (Scheduling Iterations Ahead) pass needs to
#   distinguish "load issue" (header), "decomposable middle body"
#   (the chunk it can reorder for latency hiding), and "guard /
#   bookkeeping tail" (footer). By giving each chunk its own named
#   sub-module, the scheduler can splice ``sm.middle.items()`` into
#   the pipeline at any cycle without touching ``sm.header`` /
#   ``sm.footer``. See KernelWriter.scheduleLatencyHiding callers.
#
#   The future logical-IR lowering (``to_stinky_asm``) flattens
#   StructuredModule transparently -- the structural distinction is
#   only needed on the Python side while SIA still runs there.
#
# Parity contract:
#   * Public surface matches rocisa's nanobind binding (code.cpp:
#     233-244): ctor ``(name="")``, read-write attrs ``header /
#     middle / footer``, ``__deepcopy__``, AND a ``__reduce__`` that
#     raises ``RuntimeError("StructuredModule is not picklable")``
#     -- note the message differs from Module's "Module is not
#     picklable" (the C++ binding has its own message).
#   * The 3 sub-modules are aliased into ``itemList`` AT
#     CONSTRUCTION: ``sm.header is sm.itemList[0]`` is True so
#     mutations to ``sm.header.add(...)`` show up in ``str(sm)``.
#   * Conscious divergence from rocisa C++ on deepcopy: rocisa's
#     C++ copy ctor (code.hpp:781-790) accidentally BREAKS the
#     aliasing post-clone (clones header/middle/footer a second
#     time, leaving them out of itemList). We deliberately PRESERVE
#     the aliasing using Python's deepcopy ``memo`` mechanism --
#     see ``__deepcopy__``'s long comment for full rationale.
#     Net effect: ``clone.header is clone.itemList[0]`` after
#     deepcopy in this adapter (False in rocisa). Original / clone
#     remain fully isolated either way.
#   * Names of the sub-modules are the literals ``"header" /
#     "middle" / "footer"``, matching the C++ ``make_shared<Module>
#     ("header")`` calls. ``findNamedItem("header")`` works.


class StructuredModule(Module):
    """``Module`` subclass with auto-added ``header / middle / footer``
    sub-modules. Mirror of ``rocisa::StructuredModule``.

    See the section header above for the SIA-scheduling rationale,
    the parity contract, and the deepcopy-aliasing quirk.
    """

    __slots__ = ("header", "middle", "footer")

    def __init__(self, name: str = ""):
        # ``Module.__init__`` populates ``itemList`` as empty plus
        # all the Item-inherited slots. THEN we mint the three
        # sub-modules and add them in order so itemList = [header,
        # middle, footer].
        super().__init__(name)
        self.header: Module = Module("header")
        self.middle: Module = Module("middle")
        self.footer: Module = Module("footer")
        # ``Module.add`` patches the child's ``.parent`` to ``self``,
        # so the 3 sub-modules end up reparented here.
        self.add(self.header)
        self.add(self.middle)
        self.add(self.footer)

    def __deepcopy__(self, memo):
        # ------------------------------------------------------------------
        # Conscious divergence from rocisa C++ copy ctor (code.hpp:781-790).
        # ------------------------------------------------------------------
        # The C++ copy ctor breaks the construction-time aliasing
        # between ``header / middle / footer`` attrs and
        # ``itemList[0..2]``: it first calls ``Module(other)`` (which
        # clones the entire itemList, including the 3 sub-modules),
        # then independently calls ``other.header->clone()`` AGAIN
        # in the field initializer -- producing a SECOND clone that
        # is NOT in itemList. After the C++ copy ctor:
        #
        #     clone.header is not clone.itemList[0]   # True (!)
        #
        # That is almost certainly an unintentional bug. The
        # construction-time aliasing is THE central invariant of
        # this class -- it's what makes ``sm.middle.add(x)``
        # propagate into ``str(sm)``. Breaking that invariant in the
        # copy ctor means a freshly-constructed instance and a
        # deepcopy'd instance behave DIFFERENTLY for the same API
        # call -- a violation of the "principle of least surprise"
        # that ``deepcopy`` typically obeys.
        #
        # We deliberately diverge from rocisa here for two reasons:
        #
        #   1. KernelWriter's PGR=2 path indirectly triggers this
        #      via ``deepcopy(perIterGlobalRead[0])`` -- when
        #      Components/SIA.py:564 has inserted a StructuredModule
        #      as a transitive child, Python's recursive deepcopy
        #      will reach ``__deepcopy__`` here. If we mirror the
        #      C++ bug bit-for-bit, any subsequent code that touches
        #      ``cloned_sm.middle.add(...)`` on those nested clones
        #      silently produces nothing -- a sleeper bug nobody
        #      would notice until output diffs appear.
        #
        #   2. Test suite isolation: ``copy.deepcopy(sm)`` should
        #      produce a clone that behaves like a fresh ctor result.
        #      Mirroring the C++ aliasing-break would force every
        #      consumer to add "did this come from deepcopy?"
        #      branches around any sub-module mutation -- not
        #      feasible.
        #
        # Implementation: we let Python's standard ``memo`` mechanism
        # preserve aliasing for free. Step 1 ``deepcopy(it, memo)``
        # on each itemList entry populates ``memo[id(it)] ->
        # cloned_it``. Step 2 then asks for ``deepcopy(self.header,
        # memo)`` with the SAME memo -- which short-circuits to the
        # already-cloned ``itemList[0]``. Result: the post-deepcopy
        # invariant ``clone.header is clone.itemList[0]`` holds, AND
        # original / clone remain fully isolated (mutating
        # ``original.header`` after the copy does NOT bleed into
        # ``clone.header`` -- ``memo`` only caches within a SINGLE
        # deepcopy operation, not across them).
        #
        # If rocisa upstream eventually fixes this (the natural fix
        # would be ``header = dynamic_pointer_cast<Module>(itemList[0])``
        # in the C++ copy ctor body), the two sides will converge
        # and this comment can be deleted.
        # ------------------------------------------------------------------
        clone = StructuredModule.__new__(StructuredModule)
        memo[id(self)] = clone
        # --- Step 1: Module(other) equivalent (clone itemList). ----------
        Module.__init__(clone, self.name)
        clone._isNoOpt = self._isNoOpt
        clone.isCallable = self.isCallable
        clone.callableName = self.callableName
        if self.tempVgpr is not None:
            try:
                clone.tempVgpr = _copy.deepcopy(self.tempVgpr, memo)
            except Exception:  # noqa: BLE001
                clone.tempVgpr = self.tempVgpr
        for it in self.itemList:
            new_it = _copy.deepcopy(it, memo)
            clone._reparent(new_it)
            clone.itemList.append(new_it)
        # --- Step 2: rebind attrs to the already-cloned itemList ---------
        # entries via the SAME memo -- this is the aliasing-preserving
        # step. See the long comment above for the rationale.
        clone.header = _copy.deepcopy(self.header, memo)
        clone.middle = _copy.deepcopy(self.middle, memo)
        clone.footer = _copy.deepcopy(self.footer, memo)
        return clone

    def __reduce__(self):
        # rocisa's StructuredModule binding (code.cpp:242-244) has its
        # OWN ``__reduce__`` that raises with a class-specific
        # message -- distinct from Module's "Module is not
        # picklable". We mirror the message text exactly so any
        # consumer that string-matches on the exception keeps
        # working.
        raise RuntimeError("StructuredModule is not picklable")


# ---------------------------------------------------------------------------
# Macro -- ``.macro <name> args ... .endm`` block.
# ---------------------------------------------------------------------------
#
# Mirror of ``rocisa::Macro``. KernelWriter constructs 4 macros (KWA:
# ``GLOBAL_OFFSET_*``, ``MAC_*``, ``MAC_*_OneIUI``;
# Components/CustomSchedule.py: ``MAINLOOP``) and adds CommonInstruction
# / Module / TextBlock children into each.
#
# Notable divergences from generic ``Module``:
#   * Macro is NOT a subclass of Module -- it holds its own ``itemList``
#     directly (rocisa C++ does the same). Different add() contract,
#     different toString format, no addComment1 / addComment2, no
#     findNamedItem etc.
#   * ``add()`` rejects unknown item types with RuntimeError -- only
#     Instruction / Module / TextBlock / ValueIf* are allowed (matches
#     the rocisa C++ dynamic_cast whitelist).
#   * ``addComment0`` emits a single-line ``/* ... */\n`` -- distinct
#     from Module.addComment0's 3-line banner (rocisa C++ explicitly
#     chose simpler formatting for macro bodies).
#   * ``toString()`` wraps children with ``.macro <header>`` / ``.endm``
#     and indents each child line 4 spaces. The ``s_set_vgpr_msb``
#     line gets a SECOND 4-space indent on the body line right after
#     the first newline (rocisa C++ hack to keep auto-inserted MSB
#     toggles visually aligned).
#
# Implementation note: the ``.macro NAME arg0, arg1, ...`` header line
# is rendered by an internal ``MacroInstruction`` (stored on
# ``self.macro``) -- same trick as rocisa C++.


class Macro(Item):
    """Mirror of ``rocisa::Macro``.

    Layout: ``.macro <name> <args>\\n  <child0>\\n  <child1>\\n.endm\\n``.
    Children are typed (Instruction / Module / TextBlock / ValueIf*);
    other item types raise RuntimeError at ``add()`` time.
    """

    __slots__ = ("itemList", "macro")

    # Whitelist mirrors the rocisa C++ dynamic_cast chain.
    # ``_Instruction`` covers CommonInstruction / MacroInstruction /
    # CompositeInstruction subclasses transitively.
    _ALLOWED_TYPES: tuple = ()  # populated lazily in __init__

    def __init__(self, name: str, args: List[Any]):
        super().__init__(name=name)
        self.itemList: List[Any] = []
        # The header MacroInstruction is what renders ``.macro NAME args``
        # via its own toString(). We never expose it via items() -- it's
        # an internal rendering helper, not a child.
        self.macro = _MacroInstruction(name, args)

    @classmethod
    def _allowed(cls) -> tuple:
        # Built once on first call; defers import-order dependencies.
        if not cls._ALLOWED_TYPES:
            cls._ALLOWED_TYPES = (
                _Instruction, Module, TextBlock,
                ValueIf, ValueEndif, ValueElseIf,
            )
        return cls._ALLOWED_TYPES

    def add(self, item: Any) -> Any:
        if not isinstance(item, self._allowed()):
            # Match rocisa's exception message verbatim so any string-
            # matching consumer still works.
            raise RuntimeError(
                "unknown item type for Macro.add: " + str(item)
            )
        item.parent = self
        self.itemList.append(item)
        return item

    def addT(self, cls, *args, **kwargs) -> Any:
        # rocisa template helper: construct + add.
        return self.add(cls(*args, **kwargs))

    def addComment0(self, comment: str) -> None:
        # NB: simpler than Module.addComment0 (single line vs 3-line
        # banner) -- matches rocisa C++ exactly.
        self.add(TextBlock("/* " + comment + " */\n"))

    def setItems(self, items: Sequence[Any]) -> None:
        # rocisa C++ does plain ``itemList = items`` -- no type check,
        # no reparent. Mirror exactly.
        self.itemList = list(items)

    def items(self) -> List[Any]:
        return self.itemList

    def prettyPrint(self, indent: str = "") -> str:
        out = f'{indent}{type(self).__name__} "{self.name}"\n'
        for it in self.itemList:
            pp = getattr(it, "prettyPrint", None)
            if callable(pp):
                res = pp(indent + "|--")
                if isinstance(res, str):
                    out += res
                    continue
            out += f"{indent}|--{type(it).__name__}\n"
        return out

    def toString(self) -> str:
        # Layout:
        #   .macro <header>          (header includes the trailing newline)
        #       <child0_line0>
        #       <child0_line1>       (continuation lines NOT re-indented)
        #       ...
        #   .endm
        # When a child's text contains ``s_set_vgpr_msb``, insert an extra
        # 4-space indent right after the FIRST newline of that child
        # (rocisa quirk to keep auto-inserted MSB toggles aligned).
        s = ".macro " + self.macro.toString()
        for it in self.itemList:
            tmp = it.toString() if hasattr(it, "toString") else str(it)
            if "s_set_vgpr_msb" in tmp:
                pos = tmp.find("\n")
                if pos != -1:
                    tmp = tmp[:pos + 1] + "    " + tmp[pos + 1:]
            s += "    " + tmp
        s += ".endm\n"
        return s

    def __str__(self) -> str:
        return self.toString()

    def __deepcopy__(self, memo):
        clone = Macro.__new__(Macro)
        memo[id(self)] = clone
        # Item-inherited slots first.
        clone.name = self.name
        clone.parent = None
        # Header MacroInstruction is cloned via its own __deepcopy__.
        clone.macro = _copy.deepcopy(self.macro, memo)
        clone.itemList = []
        for it in self.itemList:
            new_it = _copy.deepcopy(it, memo)
            new_it.parent = clone
            clone.itemList.append(new_it)
        return clone

    def __reduce__(self):
        # rocisa Macro binding has no pickle support; mirror by raising.
        raise RuntimeError("Macro is not picklable")


# ---------------------------------------------------------------------------
# Signature helpers (module-private) + public SignatureCodeMeta / SignatureBase
# ---------------------------------------------------------------------------
#
# ``_SignatureArgument`` and ``_SignatureKernelDescriptor`` mirror rocisa
# C++ internal structs; only ``SignatureCodeMeta`` and ``SignatureBase``
# are exported (nanobind binds those two, not the helpers).

_VALUE_TYPE_SIZE: dict = {
    "i8": 1, "i16": 2, "i32": 4, "i64": 8,
    "u8": 1, "u16": 2, "u32": 4, "u64": 8,
    "bf16": 2, "f16": 2, "f32": 4, "f32c": 8,
    "f64": 8, "f64c": 16, "pkf16": 4, "struct": 8,
}


def _sig_block(comment: str) -> str:
    """Single-line block comment -- mirror of rocisa ``block()``."""
    return f"/* {comment} */\n"


def _sig_block3line(comment: str) -> str:
    """Multi-line banner comment -- mirror of rocisa ``block3Line()``."""
    out = "\n/******************************************/\n"
    for line in comment.splitlines():
        out += f"/* {line:<38} */\n"
    out += "/******************************************/\n"
    return out


def _sig_kind_is_global_buffer(kind: Any) -> bool:
    return int(kind) == int(_SVK.SIG_GLOBALBUFFER)


def _sig_kind_is_value(kind: Any) -> bool:
    return int(kind) == int(_SVK.SIG_VALUE)


class _SignatureArgument(Item):
    """Internal kernarg descriptor leaf -- mirror of ``SignatureArgument``."""

    __slots__ = ("valueKind", "valueType", "offset", "size", "addrSpaceQual")

    def __init__(
        self,
        offset: int,
        name: str,
        valueKind: Any,
        valueType: str,
        addrSpaceQual: str = "",
    ) -> None:
        super().__init__(name=name)
        self.valueKind = valueKind
        self.valueType = valueType
        self.offset = int(offset)
        self.size = self._value_to_size(valueKind, valueType)
        self.addrSpaceQual = addrSpaceQual

    @staticmethod
    def _value_to_size(valueKind: Any, valueType: str) -> int:
        if _sig_kind_is_global_buffer(valueKind):
            return 8
        try:
            return _VALUE_TYPE_SIZE[valueType]
        except KeyError as exc:
            raise RuntimeError(f"Unknown value type: {valueType}") from exc

    def _value_kind_to_str(self) -> str:
        if _sig_kind_is_global_buffer(self.valueKind):
            return "global_buffer"
        if _sig_kind_is_value(self.valueKind):
            return "by_value"
        raise RuntimeError("Unknown value kind")

    def toString(self) -> str:
        indent = "        "
        out = f"{indent[2:]}- .name:            {self.name}\n"
        out += f"{indent}.size:            {self.size}\n"
        out += f"{indent}.offset:          {self.offset}\n"
        out += f"{indent}.value_kind:      {self._value_kind_to_str()}\n"
        out += f"{indent}.value_type:      {self.valueType}\n"
        if self.addrSpaceQual:
            out += f"{indent}.address_space:   {self.addrSpaceQual}\n"
        return out

    def __str__(self) -> str:
        return self.toString()


class _SignatureKernelDescriptor(Item):
    """Internal ``.amdhsa_kernel`` header -- mirror of ``SignatureKernelDescriptor``."""

    __slots__ = (
        "totalVgprs", "totalAgprs", "totalSgprs", "originalTotalVgprs",
        "accumOffset", "groupSegSize", "sgprWorkGroup", "vgprWorkItem",
        "numSgprPreload",
    )

    def __init__(
        self,
        name: str,
        groupSegSize: int,
        sgprWorkGroup: Sequence[int],
        vgprWorkItem: int,
        totalVgprs: int = 0,
        totalAgprs: int = 0,
        totalSgprs: int = 0,
        numSgprPreload: int = 0,
    ) -> None:
        super().__init__(name=name)
        self.groupSegSize = int(groupSegSize)
        self.sgprWorkGroup = tuple(int(x) for x in sgprWorkGroup)
        self.vgprWorkItem = int(vgprWorkItem)
        self.totalAgprs = int(totalAgprs)
        self.totalSgprs = int(totalSgprs)
        self.originalTotalVgprs = int(totalVgprs)
        self.numSgprPreload = int(numSgprPreload)
        self._apply_gpr_layout(int(totalVgprs), int(totalAgprs))

    def _apply_gpr_layout(self, total_vgprs: int, total_agprs: int) -> None:
        if self.getArchCaps()["ArchAccUnifiedRegs"]:
            self.accumOffset = ((total_vgprs + 7) // 8) * 8
            self.totalVgprs = self.accumOffset + total_agprs
        else:
            self.accumOffset = -1
            self.totalVgprs = total_vgprs

    def setGprs(self, totalVgprs: int, totalAgprs: int, totalSgprs: int) -> None:
        if self.getArchCaps()["ArchAccUnifiedRegs"]:
            self.accumOffset = ((totalVgprs + 7) // 8) * 8
            self.totalVgprs = self.accumOffset + totalAgprs
        else:
            self.accumOffset = -1
            self.totalVgprs = max(totalAgprs, totalVgprs)
        self.originalTotalVgprs = int(totalVgprs)
        self.totalAgprs = int(totalAgprs)
        self.totalSgprs = int(totalSgprs)

    def getNextFreeVgpr(self) -> int:
        return self.totalVgprs

    def getNextFreeSgpr(self) -> int:
        return self.totalSgprs

    def toString(self) -> str:
        kd_indent = "  "
        isa = self.kernel().isa
        if isa is None:
            raise RuntimeError("kernel ISA is not set")
        out = _sig_block3line("Begin Kernel")
        out += f'.amdgcn_target "amdgcn-amd-amdhsa--{_isa_to_gfx(isa)}"\n'
        out += ".text\n"
        out += f".protected {self.name}\n"
        out += f".globl {self.name}\n"
        out += ".p2align 8\n"
        out += f".type {self.name},@function\n"
        out += ".section .rodata,#alloc\n"
        out += ".p2align 6\n"
        out += f".amdhsa_kernel {self.name}\n"
        out += f"{kd_indent}.amdhsa_user_sgpr_kernarg_segment_ptr 1\n"
        if self.accumOffset != -1:
            out += (
                f"{kd_indent}.amdhsa_accum_offset {self.accumOffset}"
                " // accvgpr offset\n"
            )
        out += (
            f"{kd_indent}.amdhsa_next_free_vgpr {self.totalVgprs}"
            " // vgprs\n"
        )
        out += (
            f"{kd_indent}.amdhsa_next_free_sgpr {self.totalSgprs}"
            " // sgprs\n"
        )
        out += (
            f"{kd_indent}.amdhsa_group_segment_fixed_size {self.groupSegSize}"
            " // lds bytes\n"
        )
        if self.getArchCaps()["HasWave32"]:
            wavefront = self.kernel().wavefrontSize
            if wavefront == 32:
                out += (
                    f"{kd_indent}.amdhsa_wavefront_size32 1"
                    " // 32-thread wavefronts\n"
                )
            else:
                out += (
                    f"{kd_indent}.amdhsa_wavefront_size32 0"
                    " // 64-thread wavefronts\n"
                )
        out += f"{kd_indent}.amdhsa_private_segment_fixed_size 0\n"
        out += (
            f"{kd_indent}.amdhsa_system_sgpr_workgroup_id_x "
            f"{self.sgprWorkGroup[0]}\n"
        )
        out += (
            f"{kd_indent}.amdhsa_system_sgpr_workgroup_id_y "
            f"{self.sgprWorkGroup[1]}\n"
        )
        out += (
            f"{kd_indent}.amdhsa_system_sgpr_workgroup_id_z "
            f"{self.sgprWorkGroup[2]}\n"
        )
        out += (
            f"{kd_indent}.amdhsa_system_vgpr_workitem_id "
            f"{self.vgprWorkItem}\n"
        )
        out += f"{kd_indent}.amdhsa_float_denorm_mode_32 3\n"
        out += f"{kd_indent}.amdhsa_float_denorm_mode_16_64 3\n"
        if self.numSgprPreload:
            # kernArg ptr (2 sgprs) is preloaded in user sgpr, but not counted
            # in preload_length (rocisa code.hpp SignatureKernelDescriptor).
            out += (
                f"{kd_indent}.amdhsa_user_sgpr_count "
                f"{self.numSgprPreload + 2}\n"
            )
            out += (
                f"{kd_indent}.amdhsa_user_sgpr_kernarg_preload_length "
                f"{self.numSgprPreload}\n"
            )
            out += (
                f"{kd_indent}.amdhsa_user_sgpr_kernarg_preload_offset 0\n"
            )
        out += ".end_amdhsa_kernel\n"
        out += ".text\n"
        out += _sig_block(f"Num VGPR   ={self.originalTotalVgprs}")
        out += _sig_block(f"Num AccVGPR={self.totalAgprs}")
        out += _sig_block(f"Num SGPR   ={self.totalSgprs}")
        return out

    def prettyPrint(self, indent: str = "") -> str:
        return f"{indent}{type(self).__name__} "

    def __str__(self) -> str:
        return self.toString()


class SignatureCodeMeta(Item):
    """YAML ``.amdgpu_metadata`` trailer -- mirror of ``SignatureCodeMeta``."""

    __slots__ = (
        "kernArgsVersion", "groupSegSize", "flatWgSize", "codeObjectVersion",
        "totalVgprs", "totalSgprs", "offset", "argList",
    )

    def __init__(
        self,
        name: str,
        kernArgsVersion: int,
        groupSegSize: int,
        flatWgSize: int,
        codeObjectVersion: str,
        totalVgprs: int = 0,
        totalSgprs: int = 0,
    ) -> None:
        super().__init__(name=name)
        self.kernArgsVersion = int(kernArgsVersion)
        self.groupSegSize = int(groupSegSize)
        self.flatWgSize = int(flatWgSize)
        self.codeObjectVersion = str(codeObjectVersion)
        self.totalVgprs = int(totalVgprs)
        self.totalSgprs = int(totalSgprs)
        self.offset = 0
        self.argList: List[_SignatureArgument] = []

    def setGprs(self, totalVgprs: int, totalSgprs: int) -> None:
        self.totalVgprs = int(totalVgprs)
        self.totalSgprs = int(totalSgprs)

    def addArg(
        self,
        name: str,
        kind: Any,
        type: str,
        addrSpaceQual: Optional[str] = None,
    ) -> None:
        qual = addrSpaceQual or ""
        sa = _SignatureArgument(self.offset, name, kind, type, qual)
        self.argList.append(sa)
        self.offset += sa.size

    def toString(self) -> str:
        out = ".amdgpu_metadata\n"
        out += "---\n"
        out += "custom.config:\n"
        out += "  InternalSupportParams:\n"
        out += f"    KernArgsVersion: {self.kernArgsVersion}\n"
        out += "amdhsa.version:\n"
        out += "  - 1\n"
        if self.codeObjectVersion in ("4", "default"):
            out += "  - 1\n"
        elif self.codeObjectVersion == "5":
            out += "  - 2\n"
        out += "amdhsa.kernels:\n"
        out += f"  - .name: {self.name}\n"
        out += f"    .symbol: '{self.name}.kd'\n"
        out += "    .language:                   OpenCL C\n"
        out += "    .language_version:\n"
        out += "      - 2\n"
        out += "      - 0\n"
        out += "    .args:\n"
        for arg in self.argList:
            out += arg.toString()
        out += f"    .group_segment_fixed_size:   {self.groupSegSize}\n"
        out += "    .kernarg_segment_align:      8\n"
        kernarg_size = ((self.offset + 7) // 8) * 8
        out += f"    .kernarg_segment_size:       {kernarg_size}\n"
        out += f"    .max_flat_workgroup_size:    {self.flatWgSize}\n"
        out += "    .private_segment_fixed_size: 0\n"
        out += f"    .sgpr_count:                 {self.totalSgprs}\n"
        out += "    .sgpr_spill_count:           0\n"
        out += f"    .vgpr_count:                 {self.totalVgprs}\n"
        out += "    .vgpr_spill_count:           0\n"
        out += (
            f"    .wavefront_size:             {self.kernel().wavefrontSize}\n"
        )
        out += "...\n"
        out += ".end_amdgpu_metadata\n"
        out += f"{self.name}:\n"
        return out

    def prettyPrint(self, indent: str = "") -> str:
        return f"{indent}{type(self).__name__} "

    def __str__(self) -> str:
        return self.toString()

    def __deepcopy__(self, memo):
        raise RuntimeError("SignatureCodeMeta is not deepcopyable")

    def __reduce__(self):
        raise RuntimeError("SignatureCodeMeta is not picklable")


class SignatureBase(Item):
    """Full kernel signature -- mirror of ``SignatureBase``."""

    __slots__ = ("kernelDescriptor", "codeMeta", "descriptionTopic", "descriptionList")

    def __init__(
        self,
        kernelName: str,
        kernArgsVersion: int,
        codeObjectVersion: str,
        groupSegmentSize: int,
        sgprWorkGroup: Sequence[int],
        vgprWorkItem: int,
        flatWorkGroupSize: int,
        totalVgprs: int = 0,
        totalAgprs: int = 0,
        totalSgprs: int = 0,
        numSgprPreload: int = 0,
    ) -> None:
        super().__init__(name=kernelName)
        self.kernelDescriptor = _SignatureKernelDescriptor(
            kernelName,
            groupSegmentSize,
            sgprWorkGroup,
            vgprWorkItem,
            totalVgprs,
            totalAgprs,
            totalSgprs,
            numSgprPreload,
        )
        self.codeMeta = SignatureCodeMeta(
            kernelName,
            kernArgsVersion,
            groupSegmentSize,
            flatWorkGroupSize,
            codeObjectVersion,
            totalVgprs,
            totalSgprs,
        )
        self.descriptionTopic = TextBlock("")
        self.descriptionList: List[TextBlock] = []

    def setGprs(self, totalVgprs: int, totalAgprs: int, totalSgprs: int) -> None:
        self.kernelDescriptor.setGprs(totalVgprs, totalAgprs, totalSgprs)
        self.codeMeta.setGprs(totalVgprs, totalSgprs)

    @property
    def offset(self) -> int:
        return self.codeMeta.offset

    def addArg(
        self,
        name: str,
        kind: Any,
        type: str,
        addrSpaceQual: Optional[str] = None,
    ) -> None:
        self.codeMeta.addArg(name, kind, type, addrSpaceQual)

    def addDescriptionTopic(self, text: str) -> None:
        self.descriptionTopic = TextBlock(_sig_block3line(text))

    def addDescriptionBlock(self, text: str) -> None:
        self.descriptionList.append(TextBlock(_sig_block(text)))

    def addDescription(self, text: str) -> None:
        self.descriptionList.append(TextBlock(_slash(text)))

    def getNextFreeVgpr(self) -> int:
        return self.kernelDescriptor.getNextFreeVgpr()

    def getNextFreeSgpr(self) -> int:
        return self.kernelDescriptor.getNextFreeSgpr()

    def clearDescription(self) -> None:
        self.descriptionList.clear()

    def toString(self) -> str:
        out = self.kernelDescriptor.toString()
        topic = self.descriptionTopic.toString()
        if topic:
            out += topic
        for block in self.descriptionList:
            out += block.toString()
        out += self.codeMeta.toString()
        return out

    def prettyPrint(self, indent: str = "") -> str:
        return f"{indent}{type(self).__name__} "

    def __str__(self) -> str:
        return self.toString()

    def __deepcopy__(self, memo):
        raise RuntimeError("SignatureBase is not deepcopyable")

    def __reduce__(self):
        raise RuntimeError("SignatureBase is not picklable")


class KernelBody(Item):
    """Top-level kernel wrapper -- mirror of ``rocisa::KernelBody``.

    Holds an optional ``SignatureBase`` (via ``addSignature``) and a
    ``Module`` body (via ``addBody`` / the rw ``body`` attribute).
    ``toString()`` emits the ``Begin Kernel`` banner, then signature,
    then body; missing body raises ``RuntimeError`` (rocisa parity).
    """

    __slots__ = (
        "signature", "body", "totalVgprs", "totalAgprs", "totalSgprs",
    )

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.signature: Optional[SignatureBase] = None
        self.body: Optional[Module] = None
        self.totalVgprs: int = 0
        self.totalAgprs: int = 0
        self.totalSgprs: int = 0

    def addSignature(self, signature: SignatureBase) -> None:
        self.signature = signature

    def addBody(self, body: Module) -> None:
        self.body = body

    def setGprs(self, totalVgprs: int, totalAgprs: int, totalSgprs: int) -> None:
        self.totalVgprs = int(totalVgprs)
        self.totalAgprs = int(totalAgprs)
        self.totalSgprs = int(totalSgprs)
        if self.signature is not None:
            self.signature.setGprs(totalVgprs, totalAgprs, totalSgprs)

    def getNextFreeVgpr(self) -> int:
        if self.signature is not None:
            return self.signature.getNextFreeVgpr()
        return 0

    def getNextFreeSgpr(self) -> int:
        if self.signature is not None:
            return self.signature.getNextFreeSgpr()
        return 0

    def toString(self) -> str:
        out = TextBlock(_sig_block3line("Begin Kernel")).toString()
        if self.signature is not None:
            out += self.signature.toString()
        if self.body is not None:
            out += self.body.toString()
        else:
            raise RuntimeError("Kernel body is empty")
        return out

    def prettyPrint(self, indent: str = "") -> str:
        return f"{indent}{type(self).__name__} "

    def __str__(self) -> str:
        return self.toString()

    def __deepcopy__(self, memo):
        raise RuntimeError("KernelBody is not deepcopyable")

    def __reduce__(self):
        raise RuntimeError("KernelBody is not picklable")


# ---------------------------------------------------------------------------
# Dummy code-composition components (pending real implementation).
# ---------------------------------------------------------------------------
#
# Every dummy below inherits from the real ``Item`` base so:
#   * ``isinstance(dummy_instance, Item)`` is True -- ``Module.findIndex
#     ByType(Item)`` / KernelWriter type-walks see them as IR nodes.
#   * ``dummy.toString()`` / ``dummy.prettyPrint()`` /
#     ``dummy.countType()`` / ``dummy.countExactType()`` resolve to the
#     real Item methods (not the no-op ``__getattr__`` shim), so a
#     dummy dropped into a Module tree still produces sensible
#     prettyPrint / count output during bring-up.
#
# Inheritance map (mirror of code.hpp):
#   BitfieldUnion -- standalone polymorphic root in C++
#     (code.hpp:928, NOT a subclass of Item). Kept ``base=object``
#     so ``isinstance(bf, Item)`` correctly stays False; the C++ class
#     is the polymorphic root for the ``SrdUpperValue*`` family.
#
# Already real (above this block): TextBlock, Module, ValueIf,
# ValueElseIf, ValueEndif, ValueSet, RegSet, Label, StructuredModule,
# Macro, SignatureCodeMeta, SignatureBase, KernelBody.

BitfieldUnion = make_dummy_class(f"{_P}.BitfieldUnion")


# ---------------------------------------------------------------------------
# logicalIR-backed: gfx1250 SRD upper accessor.
#
# Soft-import so this package itself stays importable even when
# ``_stinkytofu.so`` hasn't been built yet (the rocisa dispatcher silently
# falls back to native bindings on any adapter import failure; we don't
# want that fallback triggered just because logicalIR is missing).
# ``SrdUpperValue`` re-raises a clear actionable error on first use.
# ---------------------------------------------------------------------------
try:
    from stinkytofu import SrdUpperValue as _stinky_srd_upper_value  # type: ignore[import-not-found]

    _STINKYTOFU_IMPORT_ERR: "ImportError | None" = None
except ImportError as _e:  # pragma: no cover - exercised only without a build
    _stinky_srd_upper_value = None
    _STINKYTOFU_IMPORT_ERR = _e


def SrdUpperValue(isa):  # noqa: N802 (matches rocisa public API)
    """Forwarder matching ``rocisa::SrdUpperValue(IsaVersion)`` for gfx1250.

    Accepts either a 3-tuple/list (``kernel["ISA"]``-style) or a struct
    with ``.major / .minor / .stepping`` (rocisa's ``IsaVersion``).
    Only ``(12, 5, *)`` is supported today; ISA dispatch inside stinkytofu
    is handled by ``createSrdUpperValue``.
    """
    if _stinky_srd_upper_value is None:
        raise ImportError(
            "rocisa_stinkytofu_adaptor.code.SrdUpperValue requires the "
            "stinkytofu Python binding (_stinkytofu.so). Build it via:\n"
            "  cmake --build <build_dir> --target stinkytofu_python\n"
            "and ensure <build_dir>/tensilelite/rocisa/stinkytofu is on PYTHONPATH.\n"
            f"  Underlying error: {_STINKYTOFU_IMPORT_ERR}"
        )
    if hasattr(isa, "major"):
        major = int(isa.major)
        minor = int(isa.minor)
        stepping = int(isa.stepping)
    else:
        major = int(isa[0])
        minor = int(isa[1])
        stepping = int(isa[2]) if len(isa) > 2 else 0
    if (major, minor) != (12, 5):
        raise NotImplementedError(
            f"rocisa_stinkytofu_adaptor.code.SrdUpperValue is gfx1250-only; "
            f"got ISA major={major}, minor={minor}."
        )
    return _stinky_srd_upper_value((major, minor, stepping))
