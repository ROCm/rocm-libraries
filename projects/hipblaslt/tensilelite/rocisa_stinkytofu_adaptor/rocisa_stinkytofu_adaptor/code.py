################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################
"""Shim for ``rocisa.code``.

What this file is:
    Mirrors ``rocisa/rocisa/src/code.cpp`` (``init_code``) — the
    code-composition components (``Module``, ``KernelBody``, ``Label``,
    ``RegSet``, ``ValueSet``, ...). These are the structural / control-
    flow / declarative ``Item`` subclasses that KernelWriter
    assembles into a kernel; ``Instruction`` subclasses (the
    operational leaves) live in ``instruction.py``.

What it does (real):
    - ``Module`` / ``TextBlock`` — real Python container nodes that
      mirror rocisa's tree API (add / items / itemsSize / count /
      flatitems / findIndex / replaceItem / popFirstItem / ...).
      ``Module.to_stinky_asm(arch)`` is the **left-path entry point**:
      walks the item tree, builds a ``_stinkytofu.LogicalModule`` from
      every leaf that exposes ``to_stinky_logical()``, then routes it
      through ``_stinkytofu.lower_logical_module(...)`` to return a
      ``StinkyAsmModule`` ready for ``emitAssembly()``.

      This is the rocisa-shaped entry point that pairs with
      ``stinkytofu::toStinkyTofuModule()`` (the right-path,
      rocisa::Module → asm IR, implemented in
      ``shared/stinkytofu/src/conversion/rocisa/ToStinkyTofuUtils.cpp``).
      Step 3 (instruction shims) plugs in by giving each instruction
      a ``to_stinky_logical()`` method.

    - ``SrdUpperValue(isa)`` — gfx1250-only wrapper backed by the
      stinkytofu C++ free functions ``getSrdUpperValue125X`` /
      ``getSrdUpperDesc125X`` (declared in
      ``shared/stinkytofu/include/stinkytofu/ir/asm/StinkySignature.hpp``
      next to ``SrdUpperValue125X``, implemented via
      ``SrdUpperValue125X::staticInit()``). Returns a small wrapper
      exposing rocisa's ``.getValue() / .desc() / .toString()`` API.

      This is the first end-to-end "vertical slice" through
      KernelWriter → rocisa_stinkytofu_adaptor → ``_stinkytofu.so``
      (nanobind) → ``libstinkytofu.so`` (C++). Use the same recipe for
      future shim entries that need to delegate to logicalIR.

      Other ISAs are intentionally not supported today — the rocisa →
      stinkytofu adapter is gfx1250-only.

    - ``ValueIf`` / ``ValueElseIf`` / ``ValueEndif`` — GNU assembler
      preprocessor-conditional leaves (``.if`` / ``.elseif`` /
      ``.endif``). KernelWriter uses these to gate macro / kernel-text
      sections at assemble time (CustomSchedule.py:448-508,
      KernelWriterAssembly.py:1827). ``ValueEndif``'s comment formatting
      matches rocisa byte-for-byte (width-50 padding then
      ``// comment``), gated by ``outputNoComment``.

Not yet done (dummy):
    - Container nodes: ``KernelBody``, ``Label``, ``Macro``,
      ``StructuredModule``, ``ValueSet``, ``RegSet``,
      ``BitfieldUnion``, ``SignatureCodeMeta``, ``SignatureBase``.

Future:
    When this shim grows beyond gfx1250, prefer adding sibling free
    functions in ``StinkySignature.hpp`` (``getSrdUpperValue12XX`` …)
    or surface a method on ``SignatureBase`` (already exported) rather
    than re-exporting the polymorphic ``BitfieldUnion`` base across
    DSO boundaries.

logicalIR correspondence:
    ``StinkyAsmModule`` is the closest analogue at the *module* level
    (different API: ``getName / emitAssembly / runOptimizationPipeline``).
    Sub-nodes have no 1:1 counterpart.
"""

from __future__ import annotations

import copy as _copy
from typing import Any, Iterable, List, Optional, Sequence

from ._dummy import make_dummy_class
from .base import Item, outputNoComment as _outputNoComment

_P = "rocisa.code"


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
    return f"{'':<50}// {comment}\n"


def _block(comment: str) -> str:
    """3-line block comment ``/****/ /* COMMENT */ /****/``."""
    bar = "/" + "*" * 42 + "/"
    return f"{bar}\n/* {comment:<40} */\n{bar}\n"


def _block_newline(comment: str) -> str:
    """Same as ``_block`` but with a leading blank line."""
    return "\n" + _block(comment)


def _block_3line(comment: str) -> str:
    """``_block`` followed by a trailing blank line."""
    return _block(comment) + "\n"


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
        print(asm_module.emitAssembly())

    See module-level docstring for the architectural picture.
    """

    __slots__ = ("itemList", "tempVgpr", "_isNoOpt")

    def __init__(self, name: str = ""):
        super().__init__(name=name)
        self.itemList: List[Any] = []
        self.tempVgpr: Any = None
        self._isNoOpt: bool = False

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
    def to_stinky_asm(self, arch: Sequence[int]):
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

        Returns:
            ``stinkytofu.StinkyAsmModule``.

        Raises:
            ImportError: when the ``stinkytofu`` Python binding
                (``_stinkytofu.so``) is not built / on PYTHONPATH.
        """
        import stinkytofu as _st  # noqa: WPS433  (optional runtime dep)

        lm = _st.LogicalModule(self.name or "kernel")
        for inst in self._collect_logical_insts():
            lm.add(inst)
        return _st.lower_logical_module(lm, list(arch))

    def _collect_logical_insts(self) -> List[Any]:
        """In-order walk of leaf instructions exposing ``to_stinky_logical``.

        Discrimination intentionally relies on the *return value* (must be
        non-None) rather than ``hasattr``: ``rocisa_stinkytofu_adaptor``'s
        dummy shims (``_dummy.make_dummy_class``) expose a fake
        ``__getattr__`` that makes every attribute name appear present,
        so ``hasattr(dummy, "to_stinky_logical")`` is True. The dummy
        ``_noop`` returns None, which lets us cheaply filter both kinds
        of "no logical mapping" cases (dummy class + Step-3 shim that
        deliberately opts out) at the same gate.
        """
        out: List[Any] = []
        for it in self.itemList:
            if isinstance(it, Module):
                out.extend(it._collect_logical_insts())
                continue
            handle = getattr(it, "to_stinky_logical", None)
            if not callable(handle):
                continue  # TextBlock / value object / non-instruction
            logical = handle()
            if logical is None:
                continue  # dummy shim or explicit opt-out
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
# Dummy code-composition components (pending real implementation).
# ---------------------------------------------------------------------------
#
# Every dummy below inherits from the real ``Item`` base so:
#   * ``isinstance(dummy_instance, Item)`` is True -- ``Module.findIndex
#     ByType(Item)`` / KernelWriter type-walks see them as IR nodes.
#   * ``dummy.toString()`` / ``dummy.prettyPrint()`` /
#     ``dummy.countType()`` / ``dummy.countExactType()`` resolve to the
#     real Item methods (not the no-op ``__getattr__`` shim), so a
#     dummy Label dropped into a Module tree still produces sensible
#     prettyPrint / count output during bring-up.
#
# Inheritance map (mirror of code.hpp):
#   Label / Macro / ValueSet / RegSet / SignatureCodeMeta
#     / SignatureBase / KernelBody   -- ``: Item`` in C++ ⇒
#     ``base=Item`` here.
#   StructuredModule -- ``: Module`` in C++ (code.hpp:469). Using
#     ``base=Module`` here means KernelWriter's
#     ``self.codes.globalReadA = StructuredModule()`` actually gets a
#     working ``.add() / .items() / .appendModule(...)`` surface
#     during bring-up instead of silent no-ops. The "structured"
#     part (separate header/body/footer modules) is the only thing
#     missing; a follow-up commit makes it real.
#   BitfieldUnion -- standalone polymorphic root in C++
#     (code.hpp:928, NOT a subclass of Item). Kept ``base=object``
#     so ``isinstance(bf, Item)`` correctly stays False; the C++ class
#     is the polymorphic root for the ``SrdUpperValue*`` family.
#
# Already real (above this block): TextBlock, Module, ValueIf,
# ValueElseIf, ValueEndif.

Label = make_dummy_class(f"{_P}.Label", base=Item)
Macro = make_dummy_class(f"{_P}.Macro", base=Item)
StructuredModule = make_dummy_class(f"{_P}.StructuredModule", base=Module)
ValueSet = make_dummy_class(f"{_P}.ValueSet", base=Item)
RegSet = make_dummy_class(f"{_P}.RegSet", base=Item)
BitfieldUnion = make_dummy_class(f"{_P}.BitfieldUnion")
SignatureCodeMeta = make_dummy_class(f"{_P}.SignatureCodeMeta", base=Item)
SignatureBase = make_dummy_class(f"{_P}.SignatureBase", base=Item)
KernelBody = make_dummy_class(f"{_P}.KernelBody", base=Item)


# ---------------------------------------------------------------------------
# logicalIR-backed: gfx1250 SRD upper accessor (the first end-to-end slice).
#
# Soft-import so this package itself stays importable even when
# ``_stinkytofu.so`` hasn't been built yet (the rocisa dispatcher silently
# falls back to native bindings on any adapter import failure; we don't
# want that fallback triggered just because logicalIR is missing).
# ``SrdUpperValue`` re-raises a clear actionable error on first use.
# ---------------------------------------------------------------------------
try:
    from stinkytofu import (  # type: ignore[import-not-found]
        getSrdUpperValue125X as _get_srd_upper_value_125x,
        getSrdUpperDesc125X as _get_srd_upper_desc_125x,
    )

    _STINKYTOFU_IMPORT_ERR: "ImportError | None" = None
except ImportError as _e:  # pragma: no cover - exercised only without a build
    _get_srd_upper_value_125x = None
    _get_srd_upper_desc_125x = None
    _STINKYTOFU_IMPORT_ERR = _e


class _Gfx1250SrdUpper:
    """rocisa-shaped wrapper around the two gfx1250 free functions.

    Tensile only reads ``.getValue() / .desc() / .toString()`` off
    ``SrdUpperValue(IsaVersion)`` (see ``KernelWriterAssembly.py:1497``);
    we expose exactly that surface and forward to logicalIR. Keeping
    the wrapper on the Python side lets logicalIR's public C++ ABI
    stay primitive-typed (no ``BitfieldUnion`` base crossing the DSO).
    """

    __slots__ = ()

    def getValue(self) -> int:  # noqa: N802 (matches rocisa public API)
        return int(_get_srd_upper_value_125x())

    def desc(self) -> str:
        return _get_srd_upper_desc_125x()

    def toString(self) -> str:  # noqa: N802 (matches rocisa public API)
        return f"0x{self.getValue():x}"


def SrdUpperValue(isa):  # noqa: N802 (matches rocisa public API)
    """Wrapper matching ``rocisa::SrdUpperValue(IsaVersion)`` for gfx1250.

    Accepts either a 3-tuple/list (``kernel["ISA"]``-style) or a struct
    with ``.major / .minor / .stepping`` (rocisa's ``IsaVersion``).
    Only ``(12, 5, *)`` is supported today; other ISAs raise
    ``NotImplementedError`` deliberately — extending coverage should add
    sibling free functions in ``StinkySignature.hpp`` rather than
    re-exporting the polymorphic ``BitfieldUnion`` base.
    """
    if _get_srd_upper_value_125x is None:
        raise ImportError(
            "rocisa_stinkytofu_adaptor.code.SrdUpperValue requires the "
            "stinkytofu Python binding (_stinkytofu.so). Build it via:\n"
            "  cmake --build <build_dir> --target stinkytofu_python\n"
            "and ensure <build_dir>/lib is on PYTHONPATH.\n"
            f"  Underlying error: {_STINKYTOFU_IMPORT_ERR}"
        )
    if hasattr(isa, "major"):
        major, minor = int(isa.major), int(isa.minor)
    else:
        major, minor = int(isa[0]), int(isa[1])
    if (major, minor) != (12, 5):
        raise NotImplementedError(
            f"rocisa_stinkytofu_adaptor.code.SrdUpperValue is gfx1250-only; "
            f"got ISA major={major}, minor={minor}. Extend coverage by "
            f"adding sibling free functions in "
            f"shared/stinkytofu/include/stinkytofu/ir/asm/StinkySignature.hpp."
        )
    return _Gfx1250SrdUpper()
