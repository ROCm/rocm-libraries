# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Standalone tests for ``rocisa_stinkytofu_adaptor.code``.

1:1 mirror of every export in ``code.py``: TextBlock, Module,
ValueIf/ElseIf/Endif, ValueSet, RegSet, Label, StructuredModule,
Macro, Signature*, KernelBody, BitfieldUnion, and ``to_stinky_asm``.

``Item`` inheritance shape for TextBlock / Module lives in
``test_base.py``. Cross-backend emit parity lives in
``test_emission_consistency.py``.

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_code.py

Or via the wrapper:

    ./test.sh test_code

The ``TestToStinkyAsm`` block is gated on a built ``stinkytofu``
binding and exercises the full left-path lowering pipeline (Module tree
-> ``LogicalModule`` -> ``StinkyAsmModule.emitAssembly``).
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
import unittest

# ---------------------------------------------------------------------------
# Self-contained sys.path bootstrap (mirrors test_container.py).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.normpath(os.path.join(_HERE, ".."))
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor.base import Item  # noqa: E402
from rocisa_stinkytofu_adaptor.enum import SignatureValueKind as SVK  # noqa: E402
from rocisa_stinkytofu_adaptor.code import (  # noqa: E402
    BitfieldUnion,
    KernelBody,
    Label,
    Macro,
    Module,
    RegSet,
    SignatureBase,
    SignatureCodeMeta,
    StructuredModule,
    TextBlock,
    ValueElseIf,
    ValueEndif,
    ValueIf,
    ValueSet,
)


# ---------------------------------------------------------------------------
# Shared fixtures -- isolate rocIsa singleton side effects across tests.
# ---------------------------------------------------------------------------


class _VgprIdxIsolation:
    """Snapshots / restores ``_vgpr_idx`` and ``HasVgprMSB`` for RegSet."""

    def setUp(self) -> None:
        from rocisa_stinkytofu_adaptor import base as _base  # noqa: WPS433
        self._base = _base
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        rocIsa.getInstance().init((12, 5, 0))
        self._caps = _base.getAsmCaps()
        self._saved_hasvgprmsb = self._caps.get("HasVgprMSB", 0)
        self._saved_vgpr_idx = dict(_base.getVgprIdx())

    def tearDown(self) -> None:
        if self._saved_hasvgprmsb == 0:
            self._caps.pop("HasVgprMSB", None)
        else:
            self._caps["HasVgprMSB"] = self._saved_hasvgprmsb
        live = self._base.getVgprIdx()
        live.clear()
        live.update(self._saved_vgpr_idx)


class _VgprMsbIsolation:
    """Snapshots / restores ``HasVgprMSB`` and ``_vgpr_msb`` for Label."""

    def setUp(self) -> None:
        from rocisa_stinkytofu_adaptor import base as _base  # noqa: WPS433
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        self._base = _base
        rocIsa.getInstance().init((12, 5, 0))
        self._caps = _base.getAsmCaps()
        self._saved_hasvgprmsb = self._caps.get("HasVgprMSB", 0)
        self._saved_vgpr_msb = _base.getVgprMsb()

    def tearDown(self) -> None:
        if self._saved_hasvgprmsb == 0:
            self._caps.pop("HasVgprMSB", None)
        else:
            self._caps["HasVgprMSB"] = self._saved_hasvgprmsb
        self._base.setVgprMsb(self._saved_vgpr_msb)


class _SignatureKernelSetup(unittest.TestCase):
    """Pump ISA + wavefront so signature emitters can read kernel()."""

    def setUp(self) -> None:
        from rocisa_stinkytofu_adaptor import base as _base  # noqa: WPS433
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        rocIsa.getInstance().init((12, 5, 0))
        _base.setKernel((12, 5, 0), 64)


# ===========================================================================
# TextBlock -- raw text leaf.
# ===========================================================================


class TestTextBlockConstruction(unittest.TestCase):
    def test_default_text_empty(self):
        tb = TextBlock()
        self.assertEqual(tb.text, "")
        # rocisa C++: ``TextBlock(text) : Item(text)`` -- default ``text=""``
        # produces ``name == ""`` because Item's ctor stores ``name = text``.
        # The common ``addComment / addSpaceLine`` path goes through this
        # branch via ``TextBlock(formatted_text)`` (non-empty), so ``name``
        # will normally equal the formatted string -- see
        # ``test_name_mirrors_text`` below.
        self.assertEqual(tb.name, "")
        self.assertIsNone(tb.parent)

    def test_text_arg_stored(self):
        tb = TextBlock("hello\n")
        self.assertEqual(tb.text, "hello\n")

    def test_name_mirrors_text(self):
        # rocisa parity (code.hpp:137-141): the Item(name) base ctor
        # stores ``name = text``, so a non-empty TextBlock has its text
        # as its name. KernelWriter does NOT rely on TextBlock names
        # for ``removeItemsByName`` (it targets Module / Label names),
        # but the adapter must still mirror the field exactly so any
        # downstream tooling that inspects ``tb.name`` sees the same
        # value across backends.
        tb = TextBlock("// foo\n")
        self.assertEqual(tb.name, "// foo\n")

    def test_name_mirrors_text_for_comment_textblocks(self):
        # Module.addComment("hello") creates TextBlock("// hello\n").
        # After P1 (name == text), that TextBlock now has its full
        # formatted text as its name. This is rocisa-faithful; documented
        # here so reviewers can sanity-check the diff.
        from rocisa_stinkytofu_adaptor.code import Module as _Module  # noqa: WPS433
        m = _Module()
        m.addComment("hello")
        self.assertEqual(m.itemList[0].name, "// hello\n")

    def test_str_and_toString_match(self):
        tb = TextBlock("// foo\n")
        self.assertEqual(str(tb), "// foo\n")
        self.assertEqual(tb.toString(), "// foo\n")

    def test_str_delegates_to_toString(self):
        # ``__str__`` must route through ``toString`` so that any future
        # gating on ``outputNoComment`` (or future production toggles)
        # automatically applies to ``str(tb)`` too -- mirrors rocisa
        # binding ``__str__ -> toString`` (code.cpp:142).
        class _TBSub(TextBlock):
            def toString(self):  # noqa: D401
                return "<<gated>>"

        self.assertEqual(str(_TBSub("hi")), "<<gated>>")

    def test_repr_shows_text(self):
        # Debug-only; exact format isn't pinned but ``text`` must appear.
        r = repr(TextBlock("abc"))
        self.assertIn("TextBlock", r)
        self.assertIn("abc", r)

    def test_deepcopy_independent(self):
        tb = TextBlock("a")
        c = copy.deepcopy(tb)
        c.text = "b"
        self.assertEqual(tb.text, "a")
        self.assertEqual(c.text, "b")

    def test_deepcopy_preserves_renamed_name(self):
        # rocisa __deepcopy__ (code.cpp:143-148) rebuilds via the ctor
        # AND patches ``name`` separately so renamed TextBlocks survive
        # the clone. Verify the adapter matches.
        tb = TextBlock("a")
        tb.name = "renamed"
        c = copy.deepcopy(tb)
        self.assertEqual(c.name, "renamed")
        self.assertEqual(c.text, "a")

    def test_prettyPrint_returns_string(self):
        # Defensive: Module.prettyPrint joins children's prettyPrint output;
        # a non-str return would explode the join.
        self.assertIsInstance(TextBlock("x").prettyPrint(), str)


# ===========================================================================
# TextBlock prettyPrint -- text content visibility + rocisa-shape parity.
# ===========================================================================


class TestTextBlockPrettyPrint(unittest.TestCase):
    """``TextBlock.prettyPrint`` inherits ``rocisa::Item::prettyPrint``::

        return indent + className + " " + toString();   // base.hpp:287-293

    The line carries the text content and has NO trailing newline.
    """

    def test_pretty_print_includes_text(self):
        # Regression for P3 -- the old impl dropped ``text`` and just
        # emitted "{indent}TextBlock\n", silently hiding contents in
        # any debug dump that walked a Module tree.
        self.assertEqual(TextBlock("abc").prettyPrint(), "TextBlock abc")

    def test_pretty_print_with_indent(self):
        self.assertEqual(TextBlock("x").prettyPrint("|--"), "|--TextBlock x")

    def test_pretty_print_no_trailing_newline(self):
        # Item::prettyPrint base does NOT append \n -- Module's prettyPrint
        # concatenates children verbatim and is responsible for the line
        # break (it adds \n only on its own header; child Modules add their
        # own header \n; leaf items either include \n in toString() or
        # don't, just like rocisa).
        self.assertFalse(TextBlock("x").prettyPrint().endswith("\n"))

    def test_pretty_print_reflects_outputNoComment_suppression(self):
        # Item::prettyPrint calls toString(); when ``outputNoComment`` is
        # set, TextBlock.toString() returns "" -- so prettyPrint of a
        # TextBlock degrades to ``"{indent}TextBlock "`` (trailing space
        # is what Item::prettyPrint emits; matches rocisa C++).
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        opts = rocIsa.getInstance().getOutputOptions()
        saved = opts.outputNoComment
        try:
            opts.outputNoComment = True
            self.assertEqual(TextBlock("anything").prettyPrint(), "TextBlock ")
        finally:
            opts.outputNoComment = saved


# ===========================================================================
# TextBlock.toString -- outputNoComment production-build suppression.
# ===========================================================================


class TestTextBlockOutputNoComment(unittest.TestCase):
    """``rocIsa.outputNoComment=True`` blanket-suppresses TextBlock text.

    Mirrors rocisa code.hpp:154-159 -- the flag suppresses EVERY TextBlock
    payload (comments AND inline-asm) so production builds emit no human-
    readable annotations.
    """

    def setUp(self):
        # Snapshot the flag so we can restore after each test, regardless
        # of pass/fail. Important because the singleton is process-wide.
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        self._opts = rocIsa.getInstance().getOutputOptions()
        self._saved = self._opts.outputNoComment

    def tearDown(self):
        self._opts.outputNoComment = self._saved

    def test_default_does_not_suppress(self):
        self._opts.outputNoComment = False
        self.assertEqual(TextBlock("// comment\n").toString(), "// comment\n")
        self.assertEqual(str(TextBlock("// comment\n")), "// comment\n")

    def test_outputNoComment_blanks_text(self):
        # Regression for P2 -- the old toString returned ``self.text``
        # unconditionally, so production-build kernels carried every
        # comment & inline-asm fragment in the emitted output.
        self._opts.outputNoComment = True
        self.assertEqual(TextBlock("// comment\n").toString(), "")
        self.assertEqual(str(TextBlock("// comment\n")), "")

    def test_outputNoComment_suppresses_inline_asm_too(self):
        # Not just comments: rocisa's gate is "blanket suppress" -- inline
        # asm fragments (which KernelWriter occasionally injects as
        # TextBlocks) get stripped just the same. We must match.
        self._opts.outputNoComment = True
        self.assertEqual(TextBlock("v_mov_b32 v0, v1").toString(), "")

    def test_outputNoComment_module_str_concats_empty(self):
        # Module.toString concatenates ``str(child)`` -- with the flag
        # on, every TextBlock collapses to "" so the whole Module string
        # contains only non-TextBlock items.
        from rocisa_stinkytofu_adaptor.code import Module as _Module  # noqa: WPS433
        m = _Module()
        m.add(TextBlock("// a\n"))
        m.add(TextBlock("// b\n"))
        self._opts.outputNoComment = True
        self.assertEqual(str(m), "")

    def test_outputNoComment_round_trip_via_setOutputOptions(self):
        # rocIsa.setOutputOptions(opts) is how Tensile ships the flag to
        # ParallelMap2 workers; verify the helper picks the new value up.
        from rocisa_stinkytofu_adaptor import rocIsa, OutputOptions  # noqa: WPS433
        rocIsa.getInstance().setOutputOptions(OutputOptions(outputNoComment=True))
        try:
            self.assertEqual(TextBlock("x").toString(), "")
        finally:
            rocIsa.getInstance().setOutputOptions(
                OutputOptions(outputNoComment=self._saved)
            )
            # Re-snapshot because we just swapped the *instance*.
            self._opts = rocIsa.getInstance().getOutputOptions()


# ===========================================================================
# TextBlock pickle -- (name, text) round-trip mirroring rocisa C++.
# ===========================================================================


class TestTextBlockPickle(unittest.TestCase):
    """rocisa code.cpp:149-154 -- ``(name, text)`` tuple round-trip."""

    def test_pickle_round_trip_default(self):
        tb = TextBlock("hello\n")
        # name defaults to text after P1, so both are "hello\n".
        clone = pickle.loads(pickle.dumps(tb))
        self.assertEqual(clone.name, "hello\n")
        self.assertEqual(clone.text, "hello\n")
        self.assertIsNone(clone.parent)
        self.assertIsNot(clone, tb)

    def test_pickle_round_trip_after_rename(self):
        # rocisa __setstate__ rebuilds via TextBlock(text), then patches
        # ``name``. Verify the two-step actually preserves a name that
        # was changed post-construction.
        tb = TextBlock("payload")
        tb.name = "label-X"
        clone = pickle.loads(pickle.dumps(tb))
        self.assertEqual(clone.name, "label-X")
        self.assertEqual(clone.text, "payload")

    def test_pickle_protocol_2_and_5(self):
        # Sanity: KernelWriter / ParallelMap2 don't pin a protocol, so
        # cover the common ones (2 = py3 default-ish, 5 = py3.8+ buffer
        # protocol used by multiprocessing fast path).
        tb = TextBlock("v_mov_b32 v0, v1")
        for proto in (2, pickle.HIGHEST_PROTOCOL):
            clone = pickle.loads(pickle.dumps(tb, protocol=proto))
            self.assertEqual(clone.text, "v_mov_b32 v0, v1")
            self.assertEqual(clone.name, "v_mov_b32 v0, v1")

    def test_getstate_returns_name_text_tuple(self):
        # Pin the wire format so any future shape change is intentional.
        tb = TextBlock("payload")
        tb.name = "lbl"
        self.assertEqual(tb.__getstate__(), ("lbl", "payload"))


# ===========================================================================
# Module construction & basic attributes.
# ===========================================================================


class TestModuleConstruction(unittest.TestCase):
    def test_default_name_empty(self):
        m = Module()
        self.assertEqual(m.name, "")
        self.assertEqual(m.itemList, [])
        self.assertIsNone(m.parent)
        self.assertIsNone(m.tempVgpr)
        self.assertFalse(m.isNoOpt())

    def test_named_ctor(self):
        m = Module("TopModule")
        self.assertEqual(m.name, "TopModule")

    def test_itemsSize_zero_on_construction(self):
        self.assertEqual(Module().itemsSize(), 0)
        self.assertEqual(Module().count(), 0)


# ===========================================================================
# add / addItems -- parent rebind, None tolerance, positional insert.
# ===========================================================================


class TestModuleAdd(unittest.TestCase):
    def test_add_returns_item(self):
        # rocisa returns the added item to enable one-liners
        # like ``foo = mod.add(SomeInstr(...))``.
        m = Module()
        tb = TextBlock("x")
        result = m.add(tb)
        self.assertIs(result, tb)
        self.assertEqual(m.itemList, [tb])

    def test_add_None_silently_ignored(self):
        # rocisa's ``if(item)`` guard; KernelWriter passes optional values
        # directly to add() and expects None to drop on the floor.
        m = Module()
        result = m.add(None)
        self.assertIsNone(result)
        self.assertEqual(m.itemList, [])

    def test_add_sets_parent(self):
        m = Module()
        tb = TextBlock("a")
        self.assertIsNone(tb.parent)
        m.add(tb)
        self.assertIs(tb.parent, m)

    def test_add_to_end_by_default(self):
        m = Module()
        a, b, c = TextBlock("a"), TextBlock("b"), TextBlock("c")
        m.add(a)
        m.add(b)
        m.add(c)
        self.assertEqual(m.itemList, [a, b, c])

    def test_add_at_index(self):
        # ``pos`` mirrors rocisa's ``itemList.insert(begin() + pos, item)``.
        m = Module()
        a, b, c = TextBlock("a"), TextBlock("b"), TextBlock("c")
        m.add(a)
        m.add(c)
        m.add(b, pos=1)
        self.assertEqual(m.itemList, [a, b, c])

    def test_add_at_pos_zero(self):
        m = Module()
        a, b = TextBlock("a"), TextBlock("b")
        m.add(a)
        m.add(b, pos=0)
        self.assertEqual(m.itemList, [b, a])

    def test_addItems_extends(self):
        m = Module()
        items = [TextBlock("a"), TextBlock("b"), TextBlock("c")]
        m.addItems(items)
        self.assertEqual(m.itemList, items)
        # All children reparented.
        for it in items:
            self.assertIs(it.parent, m)

    def test_addItems_iterable_supported(self):
        # ``addItems`` takes any iterable, not just list (rocisa parity).
        m = Module()
        m.addItems(iter([TextBlock("a"), TextBlock("b")]))
        self.assertEqual(m.itemsSize(), 2)

    def test_add_item_without_parent_attr_tolerated(self):
        # Some leaf shims do not expose ``parent`` (e.g. immutable value
        # objects). add() must not raise on them.
        class _NoParent:
            __slots__ = ()

            def __str__(self):
                return ""

        np = _NoParent()
        m = Module()
        m.add(np)  # must not raise
        self.assertEqual(m.itemList, [np])


# ===========================================================================
# Comment / spacing helpers -- TextBlock content is what matters.
# ===========================================================================


class TestModuleCommentHelpers(unittest.TestCase):
    def test_addSpaceLine_appends_newline_textblock(self):
        m = Module()
        m.addSpaceLine()
        self.assertEqual(m.itemsSize(), 1)
        self.assertIsInstance(m.itemList[0], TextBlock)
        self.assertEqual(str(m), "\n")

    def test_addComment_single_slash_format(self):
        # `// comment\n` is the bare minimum KernelWriter assumes.
        m = Module()
        m.addComment("hello")
        self.assertEqual(str(m), "// hello\n")

    def test_addCommentAlign_pads_to_col_50(self):
        # Aligned comments must end with the same `// comment\n` payload;
        # the padding leading is what makes them align in instruction-rich
        # blocks. We assert the suffix, not the exact column count, so the
        # format can be retuned without breaking the test.
        m = Module()
        m.addCommentAlign("aligned")
        text = str(m)
        self.assertTrue(text.endswith("// aligned\n"), text)
        self.assertIn("                                                  ", text)

    def test_addComment0_block_format(self):
        # addComment0 produces a single-line block comment matching native
        # format.hpp::block() — ``/* COMMENT */\n``.
        m = Module()
        m.addComment0("section")
        text = str(m)
        self.assertEqual(text, "/* section */\n")

    def test_addComment1_includes_leading_blank_line(self):
        m = Module()
        m.addComment1("with blank")
        text = str(m)
        self.assertTrue(text.startswith("\n"), text)
        self.assertIn("with blank", text)

    def test_addComment2_includes_trailing_blank_line(self):
        m = Module()
        m.addComment2("trailing")
        text = str(m)
        # _block_3line produces: \n + bar + \n + content + bar + \n
        self.assertIn("trailing", text)
        self.assertIn("/******", text)
        self.assertTrue(text.endswith("/\n"), text)


# ===========================================================================
# Accessors -- items / itemsSize / count.
# ===========================================================================


class TestModuleAccessors(unittest.TestCase):
    def test_items_returns_list_alias(self):
        # rocisa returns ``const vector<...>&``; in Python we expose the
        # underlying list (callers must NOT mutate it directly -- but the
        # alias behaviour itself is rocisa-compat).
        m = Module()
        m.add(TextBlock("a"))
        self.assertIs(m.items(), m.itemList)

    def test_itemsSize_grows_with_add(self):
        m = Module()
        for i in range(5):
            m.add(TextBlock(str(i)))
        self.assertEqual(m.itemsSize(), 5)

    def test_count_flat(self):
        # Flat module: count == itemsSize for non-Module children.
        m = Module()
        m.add(TextBlock("a"))
        m.add(TextBlock("b"))
        self.assertEqual(m.count(), 2)

    def test_count_recurses_into_submodules(self):
        # Sub-Module contributes its own recursive ``count()`` (not 1).
        outer = Module("outer")
        inner = Module("inner")
        inner.add(TextBlock("a"))
        inner.add(TextBlock("b"))
        outer.add(inner)
        outer.add(TextBlock("c"))
        self.assertEqual(outer.count(), 3)  # 2 from inner + 1 leaf
        self.assertEqual(outer.itemsSize(), 2)  # children of outer only

    def test_count_empty_submodule(self):
        outer = Module()
        outer.add(Module())  # empty inner
        outer.add(TextBlock("x"))
        self.assertEqual(outer.count(), 1)

    def test_count_deeply_nested(self):
        # 4-deep tree of nested Modules, one leaf at the bottom.
        leaf = TextBlock("L")
        root = Module()
        cur = root
        for _ in range(4):
            inner = Module()
            cur.add(inner)
            cur = inner
        cur.add(leaf)
        self.assertEqual(root.count(), 1)


# ===========================================================================
# getItem / setItem / setItems -- bounds + parent rebind.
# ===========================================================================


class TestModuleGetSetItem(unittest.TestCase):
    def test_getItem_returns_child(self):
        m = Module()
        a, b = TextBlock("a"), TextBlock("b")
        m.add(a)
        m.add(b)
        self.assertIs(m.getItem(0), a)
        self.assertIs(m.getItem(1), b)

    def test_getItem_out_of_range_raises_runtime_error(self):
        # rocisa throws std::runtime_error("index out of range") -> we
        # raise RuntimeError with the same message so any caller's
        # exception-text match keeps working.
        m = Module()
        m.add(TextBlock("a"))
        with self.assertRaises(RuntimeError) as cm:
            m.getItem(5)
        self.assertEqual(str(cm.exception), "index out of range")

    def test_setItem_replaces_and_reparents(self):
        m = Module()
        a, b = TextBlock("a"), TextBlock("b")
        m.add(a)
        m.setItem(0, b)
        self.assertIs(m.getItem(0), b)
        self.assertIs(b.parent, m)

    def test_setItem_out_of_range_raises(self):
        m = Module()
        m.add(TextBlock("a"))
        with self.assertRaises(RuntimeError):
            m.setItem(5, TextBlock("z"))

    def test_setItems_replaces_entire_list(self):
        m = Module()
        m.add(TextBlock("old"))
        new = [TextBlock("a"), TextBlock("b"), TextBlock("c")]
        m.setItems(new)
        self.assertEqual(m.itemList, new)
        for it in new:
            self.assertIs(it.parent, m)

    def test_setItems_copies_input(self):
        # Mutating the source list after setItems must not bleed in.
        m = Module()
        src = [TextBlock("a")]
        m.setItems(src)
        src.append(TextBlock("b"))
        self.assertEqual(m.itemsSize(), 1)


# ===========================================================================
# Find APIs.
# ===========================================================================


class TestModuleFind(unittest.TestCase):
    def test_findNamedItem_returns_matching(self):
        m = Module()
        named = Module("target")
        other = Module("other")
        m.add(other)
        m.add(named)
        self.assertIs(m.findNamedItem("target"), named)

    def test_findNamedItem_returns_None_when_absent(self):
        m = Module()
        m.add(Module("foo"))
        self.assertIsNone(m.findNamedItem("missing"))

    def test_findNamedItem_first_match_wins(self):
        m = Module()
        m.add(Module("dup"))
        second = Module("dup")
        m.add(second)
        # rocisa uses ``std::find_if`` which returns the first match.
        self.assertIsNot(m.findNamedItem("dup"), second)

    def test_findIndex_identity_match(self):
        # rocisa's ``std::find`` on ``shared_ptr<Item>`` -> identity.
        m = Module()
        a, b, c = TextBlock("a"), TextBlock("b"), TextBlock("c")
        m.add(a)
        m.add(b)
        m.add(c)
        self.assertEqual(m.findIndex(b), 1)

    def test_findIndex_missing_returns_minus_one(self):
        m = Module()
        m.add(TextBlock("a"))
        self.assertEqual(m.findIndex(TextBlock("a")), -1)

    def test_findIndexByType_returns_first_match(self):
        m = Module()
        m.add(TextBlock("t"))
        m.add(Module("sub"))
        m.add(TextBlock("u"))
        self.assertEqual(m.findIndexByType(Module), 1)
        self.assertEqual(m.findIndexByType(TextBlock), 0)

    def test_findIndexByType_missing_returns_minus_one(self):
        m = Module()
        m.add(TextBlock("t"))
        self.assertEqual(m.findIndexByType(Module), -1)


# ===========================================================================
# Mutations -- replaceItem / removeItem / popFirstItem.
# ===========================================================================


class TestModuleReplace(unittest.TestCase):
    def test_replaceItem_swaps_first_identity_match(self):
        m = Module()
        a, b, replacement = TextBlock("a"), TextBlock("b"), TextBlock("R")
        m.add(a)
        m.add(b)
        m.replaceItem(a, replacement)
        self.assertEqual(m.itemList, [replacement, b])
        self.assertIs(replacement.parent, m)

    def test_replaceItem_no_match_no_op(self):
        m = Module()
        m.add(TextBlock("a"))
        m.replaceItem(TextBlock("missing"), TextBlock("R"))
        self.assertEqual(m.itemsSize(), 1)

    def test_replaceItem_first_match_only(self):
        # rocisa's loop breaks on the first match.
        m = Module()
        a1, a2 = TextBlock("a"), TextBlock("a")
        m.add(a1)
        m.add(a2)
        replacement = TextBlock("R")
        m.replaceItem(a1, replacement)
        self.assertIs(m.itemList[0], replacement)
        self.assertIs(m.itemList[1], a2)

    def test_replaceItemByIndex(self):
        m = Module()
        m.add(TextBlock("a"))
        m.add(TextBlock("b"))
        r = TextBlock("R")
        m.replaceItemByIndex(1, r)
        self.assertEqual(m.itemList[1].text, "R")
        self.assertIs(r.parent, m)

    def test_replaceItemByIndex_out_of_range_silent_noop(self):
        # rocisa: ``if(index >= itemList.size()) return;``.
        m = Module()
        m.add(TextBlock("a"))
        m.replaceItemByIndex(99, TextBlock("R"))
        self.assertEqual(m.itemsSize(), 1)
        self.assertEqual(m.itemList[0].text, "a")


class TestModuleRemove(unittest.TestCase):
    def test_removeItem_identity_match(self):
        m = Module()
        a, b, c = TextBlock("a"), TextBlock("b"), TextBlock("c")
        m.addItems([a, b, c])
        m.removeItem(b)
        self.assertEqual(m.itemList, [a, c])

    def test_removeItem_missing_is_noop(self):
        m = Module()
        m.add(TextBlock("a"))
        m.removeItem(TextBlock("a"))  # equal text but different identity
        self.assertEqual(m.itemsSize(), 1)

    def test_removeItem_removes_all_identity_matches(self):
        m = Module()
        a = TextBlock("a")
        # Same identity twice -- ``[it for it in ... if it is not item]``
        # drops every copy. Matches rocisa's std::remove semantics for
        # shared_ptr identity.
        m.add(a)
        m.add(TextBlock("b"))
        m.itemList.append(a)  # alias the same identity in twice
        m.removeItem(a)
        self.assertEqual(len(m.itemList), 1)
        self.assertEqual(m.itemList[0].text, "b")

    def test_removeItemByIndex(self):
        m = Module()
        a, b, c = TextBlock("a"), TextBlock("b"), TextBlock("c")
        m.addItems([a, b, c])
        m.removeItemByIndex(1)
        self.assertEqual(m.itemList, [a, c])

    def test_removeItemByIndex_clamps_overrange_to_last(self):
        # rocisa: ``if(index >= size) index = size - 1`` then erase.
        m = Module()
        a, b = TextBlock("a"), TextBlock("b")
        m.addItems([a, b])
        m.removeItemByIndex(99)
        self.assertEqual(m.itemList, [a])

    def test_removeItemByIndex_empty_is_noop(self):
        m = Module()
        m.removeItemByIndex(0)  # must not raise
        self.assertEqual(m.itemsSize(), 0)

    def test_removeItemsByName(self):
        m = Module()
        a, b, c = Module("foo"), Module("bar"), Module("foo")
        m.addItems([a, b, c])
        m.removeItemsByName("foo")
        self.assertEqual(m.itemList, [b])


class TestModulePop(unittest.TestCase):
    def test_popFirstItem(self):
        m = Module()
        a, b = TextBlock("a"), TextBlock("b")
        m.addItems([a, b])
        self.assertIs(m.popFirstItem(), a)
        self.assertEqual(m.itemList, [b])

    def test_popFirstItem_empty_returns_None(self):
        # rocisa returns ``nullptr``; Python equivalent is None.
        self.assertIsNone(Module().popFirstItem())

    def test_popFirstNItems_partial(self):
        m = Module()
        items = [TextBlock(str(i)) for i in range(5)]
        m.addItems(items)
        popped = m.popFirstNItems(2)
        self.assertEqual(popped, items[:2])
        self.assertEqual(m.itemList, items[2:])

    def test_popFirstNItems_whole_list(self):
        # ``n >= size`` drains the list (rocisa: ``std::move``).
        m = Module()
        items = [TextBlock("a"), TextBlock("b")]
        m.addItems(items)
        popped = m.popFirstNItems(5)
        self.assertEqual(popped, items)
        self.assertEqual(m.itemList, [])

    def test_popFirstNItems_zero(self):
        m = Module()
        items = [TextBlock("a"), TextBlock("b")]
        m.addItems(items)
        popped = m.popFirstNItems(0)
        self.assertEqual(popped, [])
        self.assertEqual(m.itemList, items)


# ===========================================================================
# Tree ops -- appendModule / addModuleAsFlatItems / flatitems / setParent.
# ===========================================================================


class TestModuleTreeOps(unittest.TestCase):
    def test_appendModule_copies_children(self):
        # Each child of ``module`` is added to ``self`` (parent gets
        # rewritten to ``self``). The donor module is returned.
        target = Module("target")
        donor = Module("donor")
        a, b = TextBlock("a"), TextBlock("b")
        donor.addItems([a, b])
        result = target.appendModule(donor)
        self.assertIs(result, donor)
        self.assertEqual(target.itemList, [a, b])
        self.assertIs(a.parent, target)
        self.assertIs(b.parent, target)

    def test_addModuleAsFlatItems_flattens_subtree(self):
        # Flattens transitively before adding -- nested Modules are
        # gone in the target's itemList.
        target = Module()
        donor = Module()
        sub = Module()
        leaf_a, leaf_b = TextBlock("a"), TextBlock("b")
        sub.add(leaf_a)
        donor.add(sub)
        donor.add(leaf_b)
        target.addModuleAsFlatItems(donor)
        self.assertEqual(target.itemList, [leaf_a, leaf_b])

    def test_flatitems_flattens_all_levels(self):
        m = Module()
        leaf_a = TextBlock("a")
        leaf_b = TextBlock("b")
        leaf_c = TextBlock("c")
        inner1 = Module()
        inner1.add(leaf_a)
        inner2 = Module()
        inner2.add(leaf_b)
        m.add(inner1)
        m.add(inner2)
        m.add(leaf_c)
        self.assertEqual(m.flatitems(), [leaf_a, leaf_b, leaf_c])

    def test_flatitems_empty_module(self):
        self.assertEqual(Module().flatitems(), [])

    def test_setParent_recurses(self):
        # setParent is called after deep-loaded trees to make sure every
        # node points at its lexical parent. We simulate a broken parent
        # chain and verify setParent fixes it top-down.
        outer = Module()
        inner = Module()
        leaf = TextBlock("L")
        # Bypass add() to avoid auto-reparent.
        outer.itemList.append(inner)
        inner.itemList.append(leaf)
        outer.setParent()
        self.assertIs(inner.parent, outer)
        self.assertIs(leaf.parent, inner)


class TestModuleSetters(unittest.TestCase):
    def test_setNoOpt_isNoOpt_roundtrip(self):
        m = Module()
        self.assertFalse(m.isNoOpt())
        m.setNoOpt(True)
        self.assertTrue(m.isNoOpt())
        m.setNoOpt(False)
        self.assertFalse(m.isNoOpt())

    def test_addTempVgpr_stores(self):
        m = Module()
        sentinel = object()
        m.addTempVgpr(sentinel)
        self.assertIs(m.tempVgpr, sentinel)

    def test_setInlineAsmPrintMode_recurses_modules_and_calls_instructions(self):
        # rocisa: walks itemList; sub-Module -> recurse, Instruction ->
        # setInlineAsm(mode). We verify both branches with a fake Instr.
        class _FakeInstr:
            def __init__(self):
                self.parent = None
                self.mode = None

            def setInlineAsm(self, m):
                self.mode = m

            def __str__(self):
                return ""

        i_outer = _FakeInstr()
        i_inner = _FakeInstr()
        inner = Module()
        inner.add(i_inner)
        outer = Module()
        outer.add(inner)
        outer.add(i_outer)
        outer.setInlineAsmPrintMode(True)
        self.assertEqual(i_outer.mode, True)
        self.assertEqual(i_inner.mode, True)


# ===========================================================================
# Rendering -- toString / str / prettyPrint.
# ===========================================================================


class TestModuleToString(unittest.TestCase):
    def test_empty_module_empty_string(self):
        self.assertEqual(str(Module()), "")
        self.assertEqual(Module().toString(), "")

    def test_concatenates_children_str(self):
        m = Module()
        m.add(TextBlock("alpha "))
        m.add(TextBlock("beta\n"))
        self.assertEqual(str(m), "alpha beta\n")

    def test_recursive_render(self):
        outer = Module()
        outer.add(TextBlock("# outer-start\n"))
        inner = Module()
        inner.add(TextBlock("# inner\n"))
        outer.add(inner)
        outer.add(TextBlock("# outer-end\n"))
        self.assertEqual(str(outer), "# outer-start\n# inner\n# outer-end\n")

    def test_str_delegates_to_toString(self):
        m = Module()
        m.add(TextBlock("ABC"))
        self.assertEqual(str(m), m.toString())


class TestModulePrettyPrint(unittest.TestCase):
    def test_empty_module(self):
        out = Module("Foo").prettyPrint()
        self.assertIn('Module "Foo"', out)

    def test_empty_module_exact_format(self):
        # rocisa code.hpp:418-427 -- ``{indent}{ClassName} "{name}"\n`` and
        # nothing else when the module is empty. Pin the exact bytes so
        # any future format drift surfaces here, not in a downstream diff.
        self.assertEqual(Module("Foo").prettyPrint(), 'Module "Foo"\n')

    def test_nested_structure(self):
        outer = Module("Outer")
        inner = Module("Inner")
        inner.add(TextBlock("x"))
        outer.add(inner)
        out = outer.prettyPrint()
        self.assertIn('Module "Outer"', out)
        self.assertIn('Module "Inner"', out)
        self.assertIn("TextBlock", out)

    def test_includes_textblock_payload(self):
        # Regression for P3: the old TextBlock.prettyPrint dropped its
        # text, so debugging a tree gave you no idea what the comments
        # said. After the fix, each TextBlock line contains its content.
        outer = Module("Outer")
        outer.add(TextBlock("INTERESTING-PAYLOAD"))
        out = outer.prettyPrint()
        self.assertIn("INTERESTING-PAYLOAD", out)

    def test_nested_exact_concat_format(self):
        # Byte-for-byte format check against rocisa C++:
        #   - Module header line ends with `"\n`
        #   - Each child's prettyPrint(indent + "|--") is concatenated
        #     verbatim (no extra separators / no rstrip / no joins)
        #   - Sub-Module brings its own trailing newline (its header)
        #   - Leaf TextBlock has NO trailing newline (Item base)
        outer = Module("Outer")
        inner = Module("Inner")
        inner.add(TextBlock("leaf"))
        outer.add(inner)
        expected = (
            'Module "Outer"\n'
            '|--Module "Inner"\n'
            '|--|--TextBlock leaf'
        )
        self.assertEqual(outer.prettyPrint(), expected)

    def test_deep_nesting_indent_doubles(self):
        # Each level adds "|--" to the indent. A 3-level nest has 3 |--.
        root = Module("L0")
        l1 = Module("L1")
        l2 = Module("L2")
        l2.add(TextBlock("x"))
        l1.add(l2)
        root.add(l1)
        expected = (
            'Module "L0"\n'
            '|--Module "L1"\n'
            '|--|--Module "L2"\n'
            '|--|--|--TextBlock x'
        )
        self.assertEqual(root.prettyPrint(), expected)

    def test_starting_indent_param_propagates(self):
        # ``prettyPrint("> ")`` ships the indent through to children too.
        m = Module("Root")
        m.add(TextBlock("x"))
        self.assertEqual(m.prettyPrint("> "), '> Module "Root"\n> |--TextBlock x')

    def test_multiple_siblings_concatenated_in_order(self):
        # rocisa: ``for(const auto& i : itemList) ostream += i->prettyPrint(...)``.
        # Two siblings emit two consecutive child lines, in itemList order.
        m = Module("M")
        m.add(TextBlock("first"))
        m.add(TextBlock("second"))
        # No newline between the two TextBlock lines because Item::prettyPrint
        # has no trailing \n -- they fuse exactly as in rocisa.
        self.assertEqual(
            m.prettyPrint(),
            'Module "M"\n|--TextBlock first|--TextBlock second',
        )

    def test_indent_propagates(self):
        # Inner items should be indented more than the outer header.
        outer = Module("Outer")
        outer.add(TextBlock("x"))
        out = outer.prettyPrint()
        lines = [ln for ln in out.split("\n") if ln]
        self.assertTrue(any("|--" in ln for ln in lines))

    def test_dummy_child_without_prettyPrint_falls_back(self):
        # If a child returns a non-string from prettyPrint (e.g. dummy
        # shim's noop __getattr__), Module emits a class-name fallback
        # line so the tree dump stays usable during bring-up.
        class _NoPretty:
            def __str__(self):
                return ""

            def prettyPrint(self, indent=""):
                return None  # simulate dummy ``_noop``

        m = Module("M")
        m.add(_NoPretty())
        out = m.prettyPrint()
        self.assertIn("_NoPretty", out)
        self.assertTrue(out.endswith("\n"))


# ===========================================================================
# Copy / pickle semantics.
# ===========================================================================


class TestModuleDeepCopy(unittest.TestCase):
    def test_deepcopy_returns_independent_module(self):
        m = Module("orig")
        m.add(TextBlock("a"))
        m.add(TextBlock("b"))
        c = copy.deepcopy(m)
        self.assertEqual(c.name, "orig")
        self.assertEqual(len(c.itemList), 2)
        self.assertIsNot(c, m)
        # Children are deep-copied -- mutating the clone's TextBlock
        # must not bleed back.
        c.itemList[0].text = "MUTATED"
        self.assertEqual(m.itemList[0].text, "a")

    def test_deepcopy_reparents_children_to_clone(self):
        m = Module()
        tb = TextBlock("x")
        m.add(tb)
        c = copy.deepcopy(m)
        # Clone's child points at the clone (not the original).
        self.assertIs(c.itemList[0].parent, c)
        # Original is unchanged.
        self.assertIs(m.itemList[0].parent, m)

    def test_deepcopy_preserves_noopt_flag(self):
        m = Module()
        m.setNoOpt(True)
        c = copy.deepcopy(m)
        self.assertTrue(c.isNoOpt())

    def test_deepcopy_handles_nested_modules(self):
        outer = Module("outer")
        inner = Module("inner")
        inner.add(TextBlock("leaf"))
        outer.add(inner)
        c = copy.deepcopy(outer)
        self.assertEqual(str(c), str(outer))
        # Reparent walks the whole tree.
        self.assertIs(c.itemList[0].parent, c)
        self.assertIs(c.itemList[0].itemList[0].parent, c.itemList[0])


class TestModulePickleRejected(unittest.TestCase):
    def test_pickle_raises_runtime_error(self):
        # rocisa explicitly raises ``Module is not picklable`` to keep
        # ParallelMap2 workers from silently shipping malformed IR.
        m = Module()
        m.add(TextBlock("a"))
        with self.assertRaises(RuntimeError) as cm:
            pickle.dumps(m)
        self.assertIn("not picklable", str(cm.exception))



# ===========================================================================
# logicalIR handoff -- _populate_logical_module walk semantics.
# ===========================================================================
#
# Module-side collector tests (fake leaves, traversal, dummy skip, ValueSet).
# Real ``VMovB32`` pickup when stinkytofu is built lives in
# ``test_instruction.TestCollectLogicalIntegration``. End-to-end lowering
# is in ``TestToStinkyAsm`` below and ``test_emission_consistency``.


class _FakeLogicalInst:
    """Looks like a Step-3 instruction shim to ``_populate_logical_module``."""

    def __init__(self, payload):
        self.parent = None
        self.payload = payload
        self.lowered = False

    def to_stinky_logical(self):
        self.lowered = True
        return self.payload

    def __str__(self):
        return ""


class _MockLogicalModule:
    """Records add() and add_set_directive() calls for assertion."""

    def __init__(self):
        self.items = []  # list of ("inst", payload) or ("set", sym, val)

    def add(self, inst):
        self.items.append(("inst", inst))

    def add_set_directive(self, symbol, value):
        self.items.append(("set", symbol, value))

    def add_textblock(self, text):
        self.items.append(("textblock", text))

    def add_label(self, name, alignment, comment):
        self.items.append(("label", name, alignment, comment))


class TestPopulateLogicalModule(unittest.TestCase):
    def _payloads(self, m):
        """Helper: run _populate_logical_module and return recorded items."""
        mock = _MockLogicalModule()
        m._populate_logical_module(mock)
        return mock.items

    def test_empty_module(self):
        self.assertEqual(self._payloads(Module()), [])

    def test_emits_textblock(self):
        m = Module()
        m.add(TextBlock("ignored"))
        self.assertEqual(self._payloads(m), [("textblock", "ignored")])

    def test_picks_up_logical_leaf(self):
        m = Module()
        fake = _FakeLogicalInst(payload="P1")
        m.add(fake)
        self.assertEqual(self._payloads(m), [("inst", "P1")])
        self.assertTrue(fake.lowered)

    def test_in_order_traversal_flat(self):
        m = Module()
        a, b, c = (_FakeLogicalInst("A"), _FakeLogicalInst("B"), _FakeLogicalInst("C"))
        m.add(a)
        m.add(b)
        m.add(c)
        self.assertEqual(self._payloads(m), [("inst", "A"), ("inst", "B"), ("inst", "C")])

    def test_in_order_traversal_with_nested_modules(self):
        outer = Module()
        outer.add(_FakeLogicalInst("0"))
        inner = Module()
        inner.add(_FakeLogicalInst("1"))
        inner.add(_FakeLogicalInst("2"))
        outer.add(inner)
        outer.add(_FakeLogicalInst("3"))
        self.assertEqual(
            self._payloads(outer),
            [("inst", "0"), ("inst", "1"), ("inst", "2"), ("inst", "3")],
        )

    def test_skips_dummy_instruction_classes(self):
        from rocisa_stinkytofu_adaptor.instruction import SAddPCI64_SIMM  # noqa: WPS433
        m = Module()
        m.add(SAddPCI64_SIMM())
        self.assertEqual(self._payloads(m), [])

    def test_textblocks_and_logical_mixed(self):
        m = Module()
        m.add(TextBlock("// header\n"))
        m.add(_FakeLogicalInst("INST"))
        m.add(TextBlock("// footer\n"))
        items = self._payloads(m)
        self.assertEqual(items[0], ("textblock", "// header\n"))
        self.assertEqual(items[1], ("inst", "INST"))
        self.assertEqual(items[2], ("textblock", "// footer\n"))

    def test_valueset_emitted_as_set_directive(self):
        m = Module()
        m.add(ValueSet("vgprBase", 516))
        m.add(_FakeLogicalInst("A"))
        m.add(ValueSet("vgprFoo", "vgprBase", offset=0))
        m.add(_FakeLogicalInst("B"))
        m.add(ValueSet("vgprBase", "UNDEF", format=-1))
        items = self._payloads(m)
        self.assertEqual(items[0], ("set", "vgprBase", "516"))
        self.assertEqual(items[1], ("inst", "A"))
        self.assertEqual(items[2], ("set", "vgprFoo", "vgprBase+0"))
        self.assertEqual(items[3], ("inst", "B"))
        self.assertEqual(items[4], ("set", "vgprBase", "UNDEF"))


# ===========================================================================
# to_stinky_asm -- end-to-end binding call (gated on built stinkytofu).
# ===========================================================================


try:
    import stinkytofu as _stinky  # noqa: F401
    _STINKY_OK = (
        hasattr(_stinky, "LogicalModule")
        and hasattr(_stinky, "lower_logical_module")
        and hasattr(_stinky, "VMovB32")
    )
except ImportError:
    _STINKY_OK = False


@unittest.skipUnless(_STINKY_OK, "stinkytofu binding not built / missing left-path symbols")
class TestToStinkyAsm(unittest.TestCase):
    """Run the full left-path lowering on a single-VMovB32 toy module.

    Uses a thin wrapper class that fabricates the ``_stinkytofu.VMovB32``
    on demand so this test does not depend on Step 3 (the
    ``rocisa.instruction.VMovB32`` shim) landing first.
    """

    def _make_fake_vmovb32(self):
        # Build a logical-IR VMovB32 the way Step 3 will, encapsulated
        # in a shim that exposes ``to_stinky_logical()``.
        import stinkytofu as _st

        dst = _st.vgpr(0, 1)
        src = _st.vgpr(1, 1)

        class _ShimVMovB32:
            def __init__(self):
                self.parent = None

            def to_stinky_logical(self):
                # ``VMovB32`` ctor signature in the bindings is
                # (dest, src0, comment=""); see
                # ``shared/stinkytofu/python_module/src/python_bindings.cpp``.
                return _st.VMovB32(dst, src, "smoke")

            def __str__(self):
                return ""

        return _ShimVMovB32()

    def test_empty_module_lowers_to_empty_asm(self):
        # Empty LogicalModule -> StinkyAsmModule with no real instructions.
        m = Module("kEmpty")
        asm = m.to_stinky_asm([12, 5, 0])
        text = asm.emitAssembly()
        # Empty kernel still emits a header / directives; the only hard
        # invariant is that no v_mov_b32 leaks in.
        self.assertNotIn("v_mov_b32", text)

    def test_single_vmovb32_lowers_to_assembly(self):
        m = Module("kSingle")
        m.add(self._make_fake_vmovb32())
        asm = m.to_stinky_asm([12, 5, 0])
        text = asm.emitAssembly()
        # Byte-parity is for the right-path tests; here we just assert
        # the lowering pipeline ran end-to-end and emitted the expected
        # mnemonic.
        self.assertIn("v_mov_b32", text)

    def test_nested_modules_are_flattened_for_lowering(self):
        outer = Module("kNested")
        inner = Module("kInner")
        inner.add(self._make_fake_vmovb32())
        outer.add(inner)
        outer.add(self._make_fake_vmovb32())
        asm = outer.to_stinky_asm([12, 5, 0])
        text = asm.emitAssembly()
        # Two leaves were added, so two v_mov_b32 lines should emerge.
        self.assertEqual(text.count("v_mov_b32"), 2)

    def test_textblock_items_appear_in_output(self):
        # TextBlock items are emitted via add_textblock and appear in
        # the final assembly output as standalone comments.
        m = Module("kWithComments")
        m.add(TextBlock("// a header comment\n"))
        m.add(self._make_fake_vmovb32())
        m.add(TextBlock("// a footer comment\n"))
        asm = m.to_stinky_asm([12, 5, 0])
        text = asm.emitAssembly()
        self.assertIn("v_mov_b32", text)
        self.assertIn("a header comment", text)
        self.assertIn("a footer comment", text)

    def test_arch_accepts_sequence_not_just_list(self):
        # Tuples / arrays are common in KernelWriter (kernel["ISA"] is
        # often a tuple). Accept anything sequence-like.
        m = Module()
        m.add(self._make_fake_vmovb32())
        asm = m.to_stinky_asm((12, 5, 0))
        self.assertIn("v_mov_b32", asm.emitAssembly())


# ===========================================================================
# KernelWriter-shaped integration -- the actual usage patterns from
# tensilelite KernelWriter that the adapter has to support 1:1.
# ===========================================================================


class TestKernelWriterModuleUsage(unittest.TestCase):
    """Pin the Module/TextBlock interactions KernelWriter relies on.

    KernelWriter constructs a kernel as a Module containing named child
    Modules (one per logical phase: SetupVgpr, LoopHeader, ...), interleaved
    with comment / spacing TextBlocks. It then uses ``findNamedItem`` to
    splice extra code into a specific phase, and ``removeItemsByName`` to
    drop a section entirely. Cover the combinations that matter.
    """

    def _make_kernel_skeleton(self):
        """Build a realistic mini-Module: comments + named sub-Modules."""
        kernel = Module("MyKernel")
        kernel.addComment0("Kernel preamble")
        kernel.add(Module("SetupVgpr"))
        kernel.addComment("BetaCheck setup")
        kernel.add(Module("BetaCheck"))
        kernel.addSpaceLine()
        kernel.add(Module("LoopBody"))
        kernel.addComment("Cleanup")
        kernel.add(Module("Cleanup"))
        return kernel

    def test_findNamedItem_skips_comment_textblocks(self):
        # After P1 TextBlock.name == text, so a TextBlock from addComment
        # has name == "// BetaCheck setup\n" -- which must NOT collide
        # with the named ``Module("BetaCheck")`` lookup.
        kernel = self._make_kernel_skeleton()
        beta = kernel.findNamedItem("BetaCheck")
        self.assertIsNotNone(beta)
        self.assertIsInstance(beta, Module)
        self.assertEqual(beta.name, "BetaCheck")

    def test_findNamedItem_returns_None_for_partial_match(self):
        # ``BetaCheck setup`` is a comment's name (well, the formatted
        # ``// BetaCheck setup\n`` is). ``findNamedItem`` is exact-match
        # so the substring ``"BetaCheck"`` of the comment does NOT
        # short-circuit the real ``Module("BetaCheck")`` -- otherwise
        # KernelWriter splicing logic would target the wrong node.
        kernel = self._make_kernel_skeleton()
        # The comment's exact name (the formatted text), not just substring:
        comment_name = "// BetaCheck setup\n"
        found = kernel.findNamedItem(comment_name)
        # This DOES match the TextBlock by exact name -- demonstrating
        # the rocisa-faithful exact-equality semantic. KernelWriter
        # never calls findNamedItem with such a string, so no harm done.
        self.assertIsNotNone(found)
        self.assertNotIsInstance(found, Module)

    def test_removeItemsByName_targets_module_not_textblock(self):
        # KernelWriter calls e.g. ``module.removeItemsByName("LoopBody")``;
        # only the Module with that exact name is removed, comments stay.
        kernel = self._make_kernel_skeleton()
        before = kernel.itemsSize()
        kernel.removeItemsByName("LoopBody")
        self.assertEqual(kernel.itemsSize(), before - 1)
        # Comments untouched:
        self.assertTrue(
            any(
                isinstance(it, TextBlock) and "Cleanup" in it.text
                for it in kernel.itemList
            )
        )

    def test_findNamedItem_returns_first_among_modules_only(self):
        # A subtle KernelWriter assumption: ``findNamedItem`` is identity-
        # free, name-based, first-match. We covered the first-match case
        # in TestModuleFind; here pin the cross-Module-and-TextBlock
        # variant for a more realistic kernel skeleton.
        kernel = self._make_kernel_skeleton()
        setup = kernel.findNamedItem("SetupVgpr")
        self.assertEqual(setup.name, "SetupVgpr")

    def test_emitted_asm_contains_comments_by_default(self):
        # Default OutputOptions(outputNoComment=False): every comment makes
        # it into the rendered string. This is the development-build path.
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        opts = rocIsa.getInstance().getOutputOptions()
        saved = opts.outputNoComment
        try:
            opts.outputNoComment = False
            kernel = self._make_kernel_skeleton()
            text = str(kernel)
            self.assertIn("// BetaCheck setup", text)
            self.assertIn("// Cleanup", text)
        finally:
            opts.outputNoComment = saved

    def test_emitted_asm_strips_comments_when_outputNoComment(self):
        # Production build path (P2 regression). With the flag on, every
        # TextBlock in the kernel -- including header / divider / spacing
        # / per-section comments -- collapses to "". Named sub-Modules
        # still render normally (they have no text payload).
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        opts = rocIsa.getInstance().getOutputOptions()
        saved = opts.outputNoComment
        try:
            opts.outputNoComment = True
            kernel = self._make_kernel_skeleton()
            text = str(kernel)
            self.assertNotIn("BetaCheck setup", text)
            self.assertNotIn("Cleanup", text)
            self.assertNotIn("//", text)
            self.assertNotIn("/*", text)
        finally:
            opts.outputNoComment = saved

    def test_replaceItem_swap_preserves_outer_parent(self):
        # KernelWriter occasionally swaps a phase Module wholesale
        # (e.g. choosing between two LoopBody implementations). The
        # replacement must inherit the outer Module as its parent, or
        # any subsequent tree walk that ascends would break.
        kernel = Module("kernel")
        old = Module("LoopBody")
        new = Module("LoopBody")
        kernel.add(old)
        kernel.replaceItem(old, new)
        self.assertIs(new.parent, kernel)
        self.assertIs(kernel.findNamedItem("LoopBody"), new)


# ===========================================================================
# Import-time absence of stinkytofu does NOT break Module.
# ===========================================================================


class TestStinkytofuOptional(unittest.TestCase):
    def test_module_works_without_stinkytofu(self):
        # to_stinky_asm imports stinkytofu *lazily*; constructing /
        # editing Modules must work even when the binding is missing
        # (matches the SrdUpperValue soft-import pattern elsewhere
        # in code.py).
        m = Module()
        m.add(TextBlock("x"))
        m.add(Module())  # noqa: WPS441 -- intentional nest
        # str / count / items still work without ever touching stinkytofu.
        self.assertEqual(str(m), "x")
        self.assertEqual(m.itemsSize(), 2)


# ===========================================================================
# Item hierarchy -- BitfieldUnion exception + Module recursion.
# ===========================================================================
#
# ``TextBlock`` / ``Module`` isinstance-Item shape is in ``test_base``.
# Here: ``BitfieldUnion`` stays outside Item (code.hpp:928); Module's
# recursive ``countType`` / ``countExactType`` (code.hpp:441-459).


class TestDummyClassesInheritItem(unittest.TestCase):
    """``BitfieldUnion`` is the only code export outside the Item tree."""

    def test_bitfieldunion_is_not_item(self):
        # Intentionally NOT in the Item hierarchy -- mirror of
        # rocisa C++ where ``BitfieldUnion`` (code.hpp:928) is its
        # own standalone polymorphic root for the SrdUpperValue
        # family. Counting it as an Item would incorrectly inflate
        # ``Module.countType(Item)`` for any tree that contains
        # a BitfieldUnion sibling.
        self.assertNotIsInstance(BitfieldUnion(), Item)


class TestKernelBodyItemDefaults(unittest.TestCase):
    """``KernelBody`` inherits ``Item.countType`` and raises on empty
    ``toString`` when no body is attached (rocisa parity)."""

    def test_toString_raises_when_body_missing(self):
        kb = KernelBody("kb")
        with self.assertRaises(RuntimeError):
            kb.toString()

    def test_str_raises_when_body_missing(self):
        kb = KernelBody("kb")
        with self.assertRaises(RuntimeError):
            str(kb)

    def test_countType_is_one_for_itself(self):
        kb = KernelBody("kb")
        self.assertEqual(kb.countType(KernelBody), 1)
        self.assertEqual(kb.countType(Item), 1)
        self.assertEqual(kb.countType(Module), 0)


class TestModuleCountTypeRecursion(unittest.TestCase):
    """``Module.countType`` / ``countExactType`` override Item's default
    to recurse through ``itemList``. Mirror of rocisa C++ code.hpp:
    441-459."""

    def test_countType_recurses_through_children(self):
        # ``Module(Item)`` so countType(Item) on a module with two
        # TextBlocks (also Items) yields 1 (self) + 2 (children) = 3.
        m = Module("k")
        m.add(TextBlock("a"))
        m.add(TextBlock("b"))
        self.assertEqual(m.countType(Item), 3)

    def test_countType_recurses_into_submodules(self):
        # Two-level tree: 1 (outer Module) + 1 (TextBlock) +
        # 1 (inner Module) + 1 (TextBlock inside inner) = 4 Items.
        outer = Module("o")
        outer.add(TextBlock("a"))
        inner = Module("i")
        inner.add(TextBlock("b"))
        outer.add(inner)
        self.assertEqual(outer.countType(Item), 4)

    def test_countType_TextBlock_only(self):
        # TextBlock is the target, not Item -- Module counts as 0,
        # TextBlocks count as 1 each.
        m = Module()
        m.add(TextBlock("a"))
        m.add(TextBlock("b"))
        m.add(TextBlock("c"))
        self.assertEqual(m.countType(TextBlock), 3)

    def test_countType_includes_dummy_descendants(self):
        # Real and dummy Items both inherit from Item and MUST be
        # visible to ``countType(Item)`` walks (the very reason for
        # ``make_dummy_class(..., base=Item)`` for the still-dummy
        # nodes, and for ``class Label(Item)`` for the real ones).
        m = Module()
        m.add(Label(0, ""))
        m.add(KernelBody("kb"))
        m.add(TextBlock("x"))
        # 1 (Module) + 1 (Label) + 1 (KernelBody) + 1 (TextBlock) = 4.
        self.assertEqual(m.countType(Item), 4)

    def test_countExactType_is_strict_identity(self):
        # ``type(self) is Module`` is True for Module but False for
        # any subclass. We don't have a real Module subclass to
        # contrast with yet (StructuredModule is dummy + base=Module),
        # so verify with TextBlock as the comparison.
        m = Module()
        m.add(TextBlock("a"))
        m.add(TextBlock("b"))
        # countExactType(Module): only ``m`` itself counts (1).
        self.assertEqual(m.countExactType(Module), 1)
        # countExactType(TextBlock): only the two leaves count (2).
        self.assertEqual(m.countExactType(TextBlock), 2)

    def test_countExactType_subclass_does_not_match_module(self):
        # StructuredModule is a Module subclass; countExactType(Module)
        # on a StructuredModule itself does NOT count it (the exact
        # type check is ``typeid(*this) == targetType``, so the
        # derived StructuredModule doesn't match Module). But the
        # 3 auto-added sub-modules (header / middle / footer) ARE
        # exact Modules, so countExactType recurses into itemList
        # and counts them = 3.
        sm = StructuredModule()
        self.assertEqual(sm.countExactType(Module), 3)
        # ... StructuredModule itself matches its own exact type.
        # The 3 sub-modules are Modules (not StructuredModules) so
        # they DON'T count here -- just self.
        self.assertEqual(sm.countExactType(StructuredModule), 1)
        # ... and isinstance-based countType matches BOTH self
        # (StructuredModule is-a Module) AND the 3 sub-modules,
        # giving 1 + 3 = 4.
        self.assertEqual(sm.countType(Module), 4)


class TestStructuredModuleConstruction(unittest.TestCase):
    """``StructuredModule`` is a real ``Module`` subclass that
    auto-adds three named sub-modules ``(header, middle, footer)`` to
    ``itemList`` at construction. Mirror of ``rocisa::StructuredModule``
    (code.hpp:763-791)."""

    def test_inherits_module(self):
        sm = StructuredModule()
        self.assertIsInstance(sm, Module)
        # ``isinstance(sm, Item)`` follows from Module; see
        # ``test_base.TestItemInheritanceShape``.

    def test_default_name_empty(self):
        # rocisa ctor default: ``StructuredModule(name="")``.
        self.assertEqual(StructuredModule().name, "")

    def test_custom_name_set(self):
        # KernelWriter call sites pass things like
        # ``StructuredModule("globalReadDoA_0")`` -- the name must
        # propagate to ``Item.name``.
        self.assertEqual(StructuredModule("globalReadA").name, "globalReadA")

    def test_three_submodules_present(self):
        sm = StructuredModule()
        # All three are real Module instances (NOT None, NOT dummy).
        self.assertIsInstance(sm.header, Module)
        self.assertIsInstance(sm.middle, Module)
        self.assertIsInstance(sm.footer, Module)

    def test_submodule_names_match_rocisa(self):
        # Names are the literal strings rocisa uses
        # (``make_shared<Module>("header")`` / etc.) so that
        # ``findIndexByName("middle")`` works against either side.
        sm = StructuredModule()
        self.assertEqual(sm.header.name, "header")
        self.assertEqual(sm.middle.name, "middle")
        self.assertEqual(sm.footer.name, "footer")

    def test_itemList_has_three_entries(self):
        # itemList starts at 3 (NOT 0) -- this is a behaviour change
        # vs. the dummy version, where ``StructuredModule()`` was an
        # empty Module. Any caller that previously asserted
        # ``itemsSize() == 0`` post-construction must be updated.
        self.assertEqual(StructuredModule().itemsSize(), 3)

    def test_submodules_aliased_into_itemList(self):
        # ``add(header)`` / ``add(middle)`` / ``add(footer)`` push the
        # SAME shared_ptr (Python: same object) into itemList. This
        # aliasing is what makes ``sm.middle.add(...)`` show up in
        # ``str(sm)`` -- toString iterates itemList.
        sm = StructuredModule()
        self.assertIs(sm.itemList[0], sm.header)
        self.assertIs(sm.itemList[1], sm.middle)
        self.assertIs(sm.itemList[2], sm.footer)

    def test_submodule_parent_set_to_owner(self):
        # ``Module.add`` reparents the child; the 3 sub-modules must
        # have their ``.parent`` set to the owning StructuredModule.
        sm = StructuredModule()
        self.assertIs(sm.header.parent, sm)
        self.assertIs(sm.middle.parent, sm)
        self.assertIs(sm.footer.parent, sm)


class TestStructuredModuleEmission(unittest.TestCase):
    """``StructuredModule.toString`` is inherited from ``Module`` and
    just concatenates ``str(item)`` over itemList -- so the emission
    is the concatenation of header + middle + footer (in order)."""

    def test_empty_emits_empty_string(self):
        # All three sub-modules are empty Modules; each emits "" so
        # the StructuredModule emits "" overall.
        self.assertEqual(str(StructuredModule()), "")

    def test_middle_mutation_visible_in_str(self):
        # The whole point of the aliasing -- mutating sm.middle.add
        # MUST round-trip through ``str(sm)`` because itemList[1]
        # IS sm.middle.
        sm = StructuredModule("globalReadA")
        sm.middle.add(TextBlock("load_a_0\n"))
        sm.middle.add(TextBlock("load_a_1\n"))
        self.assertEqual(str(sm), "load_a_0\nload_a_1\n")

    def test_header_middle_footer_concatenated_in_order(self):
        # Emission order is fixed: header content first, then
        # middle, then footer. Any reordering would silently break
        # SIA-scheduled kernels.
        sm = StructuredModule()
        sm.header.add(TextBlock("H\n"))
        sm.middle.add(TextBlock("M\n"))
        sm.footer.add(TextBlock("F\n"))
        self.assertEqual(str(sm), "H\nM\nF\n")

    def test_external_add_appends_after_footer(self):
        # ``StructuredModule`` is still a Module -- raw ``.add(...)``
        # appends to itemList AFTER the 3 sub-modules. The new
        # entry's content shows up AFTER footer in the emission.
        sm = StructuredModule()
        sm.middle.add(TextBlock("M\n"))
        sm.add(TextBlock("X\n"))  # itemList[3]
        self.assertEqual(sm.itemsSize(), 4)
        self.assertEqual(str(sm), "M\nX\n")


class TestStructuredModuleAttributeSwap(unittest.TestCase):
    """The ``header / middle / footer`` attributes are read-write
    (``def_rw`` in rocisa). A caller may swap one out for a freshly
    built Module -- emission must reflect the swap, but ONLY if the
    caller also patches itemList (the aliasing is positional)."""

    def test_assigning_new_module_does_not_auto_resync_itemList(self):
        # ``sm.middle = Module("replacement")`` breaks the aliasing
        # with itemList[1] -- a caller that wants the new middle to
        # show up in ``str(sm)`` MUST also patch ``sm.itemList[1]``.
        # We pin this behavior so anyone tempted to add magic
        # syncing later thinks twice (rocisa doesn't do it either).
        sm = StructuredModule()
        replacement = Module("replacement")
        sm.middle = replacement
        # Aliasing broken: itemList[1] still points to the original.
        self.assertIsNot(sm.itemList[1], replacement)
        # Emission still reflects the OLD middle (empty).
        replacement.add(TextBlock("X\n"))
        self.assertEqual(str(sm), "")


class TestStructuredModuleFind(unittest.TestCase):
    """``findIndexByName`` / ``findIndexByType`` inherited from Module
    must locate the 3 sub-modules at their fixed positions."""

    def test_findNamedItem_locates_sub_modules(self):
        sm = StructuredModule()
        # ``findNamedItem`` returns the matching Item (the aliased
        # sub-module), not an index. Mirror of rocisa code.hpp:216.
        self.assertIs(sm.findNamedItem("header"), sm.header)
        self.assertIs(sm.findNamedItem("middle"), sm.middle)
        self.assertIs(sm.findNamedItem("footer"), sm.footer)

    def test_findIndexByType_Module_returns_zero(self):
        # ``findIndexByType(Module)`` returns the FIRST itemList
        # entry that ``isinstance(..., Module)`` -- which is
        # ``sm.header`` at position 0.
        self.assertEqual(StructuredModule().findIndexByType(Module), 0)

    def test_findIndexByType_TextBlock_skips_sub_modules(self):
        # Sub-modules are Modules, not TextBlocks; the first
        # TextBlock-typed child added afterwards lives at position 3.
        sm = StructuredModule()
        sm.add(TextBlock("x"))
        self.assertEqual(sm.findIndexByType(TextBlock), 3)


class TestStructuredModuleDeepCopy(unittest.TestCase):
    """``copy.deepcopy(sm)`` returns an isolated clone that PRESERVES
    the construction-time aliasing between ``header / middle /
    footer`` and ``itemList[0..2]``.

    This is a CONSCIOUS DIVERGENCE from rocisa's C++ copy ctor
    (code.hpp:781-790), which accidentally re-clones the 3 sub-
    modules a second time and ends up with attrs that point to
    objects NOT in itemList. See ``StructuredModule.__deepcopy__``'s
    long docstring for the full rationale -- in short, breaking the
    aliasing on deepcopy makes ``cloned_sm.middle.add(...)``
    silently no-op (the mutation lands on a Module nobody reaches
    via toString), which is a sleeper bug. We use Python's standard
    ``memo`` mechanism to keep attrs aliased to itemList entries
    while still fully isolating clone from original."""

    def test_deepcopy_returns_distinct_instance(self):
        original = StructuredModule("foo")
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertIsInstance(clone, StructuredModule)
        self.assertEqual(clone.name, "foo")

    def test_deepcopy_clones_sub_modules(self):
        original = StructuredModule()
        clone = copy.deepcopy(original)
        # Each sub-module attr on the clone is a different object
        # from the original's same-named attr.
        self.assertIsNot(clone.header, original.header)
        self.assertIsNot(clone.middle, original.middle)
        self.assertIsNot(clone.footer, original.footer)
        # But their identity-content matches.
        self.assertEqual(clone.header.name, "header")
        self.assertEqual(clone.middle.name, "middle")
        self.assertEqual(clone.footer.name, "footer")

    def test_deepcopy_preserves_aliasing_with_itemList(self):
        # The construction-time aliasing invariant (``sm.header is
        # sm.itemList[0]``) MUST survive deepcopy -- this is the
        # whole point of our divergence from rocisa C++ here.
        # Achieved via Python's standard deepcopy ``memo`` cache:
        # Step 1 deepcopies itemList entries (registering them in
        # memo); Step 2 deepcopies ``self.header / middle / footer``
        # with the SAME memo, hitting the cache and rebinding to
        # the already-cloned itemList entries.
        original = StructuredModule()
        original.middle.add(TextBlock("M\n"))
        clone = copy.deepcopy(original)
        self.assertIs(clone.header, clone.itemList[0])
        self.assertIs(clone.middle, clone.itemList[1])
        self.assertIs(clone.footer, clone.itemList[2])

    def test_deepcopy_emission_matches_original(self):
        # toString iterates itemList. With aliasing preserved (see
        # test_deepcopy_preserves_aliasing_with_itemList), the clone's
        # ``itemList[0..2]`` ARE ``clone.header/middle/footer`` --
        # so emission round-trips trivially.
        original = StructuredModule()
        original.header.add(TextBlock("H\n"))
        original.middle.add(TextBlock("M\n"))
        original.footer.add(TextBlock("F\n"))
        clone = copy.deepcopy(original)
        self.assertEqual(str(clone), str(original))
        self.assertEqual(str(clone), "H\nM\nF\n")

    def test_deepcopy_mutation_isolation_via_itemList(self):
        # Mutating original.header propagates into original's
        # itemList[0] (aliased), but the clone's itemList[0] is a
        # fresh deepcopy and must be untouched.
        original = StructuredModule()
        clone = copy.deepcopy(original)
        original.header.add(TextBlock("new_in_original\n"))
        self.assertEqual(str(original), "new_in_original\n")
        self.assertEqual(str(clone), "")

    def test_deepcopy_mutation_via_clone_middle_does_show_up(self):
        # Direct consequence of preserved aliasing: mutating
        # ``clone.middle.add(...)`` MUST appear in ``str(clone)``
        # because clone.middle IS clone.itemList[1] (toString
        # iterates itemList). This is the "principle of least
        # surprise" behavior that motivated diverging from rocisa
        # C++ here -- a fresh-ctor instance and a deepcopy'd
        # instance behave identically for the same API call.
        clone = copy.deepcopy(StructuredModule())
        clone.middle.add(TextBlock("yes\n"))
        self.assertEqual(str(clone), "yes\n")


class TestStructuredModulePickleRejected(unittest.TestCase):
    """``StructuredModule`` is explicitly NOT picklable -- rocisa's
    nanobind binding installs a ``__reduce__`` that raises with a
    class-specific message (distinct from Module's own
    ``"Module is not picklable"``)."""

    def test_pickle_raises_runtime_error(self):
        # ``pickle.dumps`` triggers ``__reduce__`` -- must raise.
        with self.assertRaises(RuntimeError) as cm:
            pickle.dumps(StructuredModule())
        self.assertEqual(str(cm.exception), "StructuredModule is not picklable")

    def test_pickle_message_distinct_from_module(self):
        # Verify the message is the class-specific text, NOT the
        # parent Module's ("Module is not picklable"). Any consumer
        # that string-matches on the exception text relies on this
        # distinction.
        sm_msg = ""
        try:
            pickle.dumps(StructuredModule())
        except RuntimeError as e:
            sm_msg = str(e)
        m_msg = ""
        try:
            pickle.dumps(Module())
        except RuntimeError as e:
            m_msg = str(e)
        self.assertNotEqual(sm_msg, m_msg)
        self.assertIn("StructuredModule", sm_msg)
        self.assertIn("Module is not picklable", m_msg)


class TestStructuredModuleCountType(unittest.TestCase):
    """``countType`` inherited from Module recurses through itemList,
    so a fresh StructuredModule counts as 4 Modules (self + 3 sub-
    modules) and 4 Items."""

    def test_countType_Module_recurses_through_sub_modules(self):
        sm = StructuredModule()
        # 1 (sm itself) + 3 (header/middle/footer) = 4.
        self.assertEqual(sm.countType(Module), 4)

    def test_countType_Item_includes_nested_content(self):
        sm = StructuredModule()
        sm.middle.add(TextBlock("x"))
        # 1 (sm) + 3 (3 sub-modules) + 1 (TextBlock in middle) = 5.
        self.assertEqual(sm.countType(Item), 5)


# ===========================================================================
# Preprocessor conditional blocks -- ValueIf / ValueElseIf / ValueEndif.
# ===========================================================================
#
# Mirror of rocisa's ``ValueIf`` / ``ValueElseIf`` / ``ValueEndif``.
# KernelWriter uses these to gate macro / kernel-text sections at
# assemble time -- byte-for-byte parity matters because the GNU
# assembler is strict about ``.if`` / ``.elseif`` / ``.endif``
# placement, and the ``.endif`` comment alignment is the only
# difference between a clean diff and a noisy one when comparing
# adapter output to the rocisa baseline.


class TestValueIfConstruction(unittest.TestCase):
    """``ValueIf`` ctor + toString format + Item integration."""

    def test_construction_positional(self):
        vi = ValueIf("foo == 1")
        self.assertEqual(vi.value, "foo == 1")

    def test_construction_keyword(self):
        # KernelWriterAssembly.py:1855 -- ``ValueIf(value="0")``.
        vi = ValueIf(value="0")
        self.assertEqual(vi.value, "0")

    def test_name_is_class_name_not_value(self):
        # ``Item.name`` is the literal class name, NOT the condition
        # expression -- KernelWriter's ``findNamedItem`` searches by
        # name; this parity guarantees those searches match between
        # the two backends.
        self.assertEqual(ValueIf("foo == 1").name, "ValueIf")

    def test_parent_starts_none(self):
        self.assertIsNone(ValueIf("x").parent)

    def test_isinstance_item(self):
        # The whole point of inheriting Item in Commit Y: standard
        # type-walks see ValueIf as a code-composition node.
        self.assertIsInstance(ValueIf("x"), Item)

    def test_toString_format(self):
        # ``".if " + value + "\\n"``. The trailing newline matters
        # because Module.toString concatenates child toString()
        # outputs verbatim.
        self.assertEqual(ValueIf("a == b").toString(), ".if a == b\n")

    def test_toString_empty_value(self):
        # C++ doesn't reject empty value; produces ".if \n".
        # KernelWriter never does this in practice but parity is
        # cheap so we keep it.
        self.assertEqual(ValueIf("").toString(), ".if \n")

    def test_str_delegates_to_toString(self):
        # Inherited Item.__str__ -> self.toString().
        self.assertEqual(str(ValueIf("k > 0")), ".if k > 0\n")


class TestValueElseIfConstruction(unittest.TestCase):
    """``ValueElseIf`` -- mirror of ``ValueIf`` with ``.elseif`` prefix."""

    def test_construction(self):
        vei = ValueElseIf("y == 2")
        self.assertEqual(vei.value, "y == 2")
        self.assertEqual(vei.name, "ValueElseIf")
        self.assertIsNone(vei.parent)

    def test_isinstance_item(self):
        self.assertIsInstance(ValueElseIf("x"), Item)

    def test_toString_format(self):
        # ``".elseif " + value + "\\n"``.
        self.assertEqual(
            ValueElseIf("\\useGR == 0").toString(),
            ".elseif \\useGR == 0\n",
        )

    def test_str_delegates_to_toString(self):
        self.assertEqual(str(ValueElseIf("a")), ".elseif a\n")


class TestValueEndifConstruction(unittest.TestCase):
    """``ValueEndif`` -- ``.endif`` with optional trailing comment."""

    def test_construction_default_comment(self):
        # Default ``comment=""`` matches the most common KernelWriter
        # call site (a bare ``ValueEndif()`` closing an .if block).
        ve = ValueEndif()
        self.assertEqual(ve.comment, "")
        self.assertEqual(ve.name, "ValueEndif")

    def test_construction_positional_comment(self):
        # KernelWriterAssembly.py:1827 -- ``ValueEndif("overflowed
        # resources")``.
        ve = ValueEndif("overflowed resources")
        self.assertEqual(ve.comment, "overflowed resources")

    def test_construction_keyword_comment(self):
        # CustomSchedule.py:493 -- ``ValueEndif(comment="EndIf %s"
        # % macroGuard)``.
        ve = ValueEndif(comment="EndIf foo")
        self.assertEqual(ve.comment, "EndIf foo")

    def test_isinstance_item(self):
        self.assertIsInstance(ValueEndif(), Item)


class TestValueEndifToStringFormatting(unittest.TestCase):
    """ValueEndif's ``toString`` mirrors rocisa's ``formatStr``
    byte-for-byte. The padding-to-column-50 behaviour is the only
    non-trivial bit in this batch; we pin it explicitly because
    production-build diffs against the rocisa baseline would
    otherwise show as spurious whitespace changes."""

    def test_empty_comment_no_padding(self):
        # rocisa formatStr: empty comment -> ".endif\n" with no
        # padding (avoids trailing-whitespace lines).
        self.assertEqual(ValueEndif().toString(), ".endif\n")
        self.assertEqual(ValueEndif("").toString(), ".endif\n")

    def test_nonempty_comment_padded_to_column_50(self):
        # ``.endif`` is 6 chars, so 44 spaces are appended to reach
        # column 50, then ``" // closing\n"``. Total line length:
        # 6 + 44 + 4 + 7 + 1 = 62 chars.
        out = ValueEndif("closing").toString()
        expected = ".endif" + " " * 44 + " // closing\n"
        self.assertEqual(out, expected)
        self.assertEqual(len(out), 62)
        # The ``//`` must land at exactly column 51 (0-indexed),
        # the same column rocisa instruction lines target.
        self.assertEqual(out.index("//"), 51)

    def test_long_instr_no_negative_padding(self):
        # The ``max(0, 50 - len)`` guard in _format_endif_str
        # protects against the unlikely future case where the
        # instruction string itself exceeds width 50. We exercise
        # it via the private helper directly since ValueEndif's
        # instr is always ``.endif`` (6 chars).
        from rocisa_stinkytofu_adaptor.code import _format_endif_str
        out = _format_endif_str("X" * 55, "tail")
        # No padding (negative clamped to 0), so the comment is
        # appended immediately after the long instr.
        self.assertEqual(out, "X" * 55 + " // tail\n")

    def test_outputNoComment_suppresses_comment(self):
        # When the rocIsa output-options flag is set, ValueEndif
        # drops the comment AND the padding -- matches rocisa's
        # ``formatStr`` ``noComment=True`` branch (falls through to
        # ``formattedStr + "\n"``).
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        opts = rocIsa.getInstance().getOutputOptions()
        saved = opts.outputNoComment
        try:
            opts.outputNoComment = True
            self.assertEqual(
                ValueEndif("would be suppressed").toString(),
                ".endif\n",
            )
        finally:
            opts.outputNoComment = saved


class TestValueConditionalPickle(unittest.TestCase):
    """Pickle round-trip preserves the single string field on each of
    the three classes. Mirrors rocisa's pickle hooks which serialise
    just the value/comment string."""

    def test_valueif_pickle_round_trip(self):
        original = ValueIf("count > 0")
        restored = pickle.loads(pickle.dumps(original))
        self.assertIsInstance(restored, ValueIf)
        self.assertEqual(restored.value, "count > 0")
        self.assertEqual(restored.name, "ValueIf")
        self.assertIsNone(restored.parent)
        self.assertEqual(restored.toString(), original.toString())

    def test_valueelseif_pickle_round_trip(self):
        original = ValueElseIf("y == 2")
        restored = pickle.loads(pickle.dumps(original))
        self.assertIsInstance(restored, ValueElseIf)
        self.assertEqual(restored.value, "y == 2")
        self.assertEqual(restored.toString(), original.toString())

    def test_valueendif_pickle_round_trip_with_comment(self):
        original = ValueEndif("EndIf guard")
        restored = pickle.loads(pickle.dumps(original))
        self.assertIsInstance(restored, ValueEndif)
        self.assertEqual(restored.comment, "EndIf guard")
        self.assertEqual(restored.toString(), original.toString())

    def test_valueendif_pickle_round_trip_default(self):
        # The bare ``ValueEndif()`` case picks up the default "".
        restored = pickle.loads(pickle.dumps(ValueEndif()))
        self.assertEqual(restored.comment, "")


class TestValueConditionalDeepCopy(unittest.TestCase):
    """deepcopy yields a fresh instance with the same string payload
    and no shared mutable state, matching rocisa's copy ctor."""

    def test_valueif_deepcopy(self):
        original = ValueIf("x")
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertIsInstance(clone, ValueIf)
        self.assertEqual(clone.value, "x")

    def test_valueelseif_deepcopy(self):
        original = ValueElseIf("y")
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.value, "y")

    def test_valueendif_deepcopy(self):
        original = ValueEndif("c")
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.comment, "c")


class TestValueConditionalModuleIntegration(unittest.TestCase):
    """A full ``.if`` / ``.elseif`` / ``.endif`` block built inside a
    Module reproduces the CustomSchedule.py:448-466 pattern. The
    emitted string must concatenate the three children verbatim
    (each child supplies its own trailing newline)."""

    def _build_if_elseif_endif_module(self) -> Module:
        m = Module("conditional")
        m.add(ValueIf("\\useGR == 1"))
        m.add(ValueElseIf("\\useGR == 0"))
        m.add(ValueEndif("EndIf useGR"))
        return m

    def test_module_toString_concatenates_block(self):
        out = str(self._build_if_elseif_endif_module())
        expected = (
            ".if \\useGR == 1\n"
            ".elseif \\useGR == 0\n"
            ".endif" + " " * 44 + " // EndIf useGR\n"
        )
        self.assertEqual(out, expected)

    def test_reparented_on_add(self):
        # Item.parent must be set to the containing Module on add()
        # -- the same parent-rebind invariant exercised for TextBlock
        # and sub-Modules elsewhere in this file.
        m = Module()
        vi = ValueIf("x")
        m.add(vi)
        self.assertIs(vi.parent, m)

    def test_countType_recurses_into_conditionals(self):
        m = self._build_if_elseif_endif_module()
        # 3 children, none of them Modules; countType(Item) on m:
        # 1 (Module) + 3 (children) = 4.
        self.assertEqual(m.countType(Item), 4)
        # Targeting individual conditional types:
        self.assertEqual(m.countType(ValueIf), 1)
        self.assertEqual(m.countType(ValueElseIf), 1)
        self.assertEqual(m.countType(ValueEndif), 1)

    def test_deepcopy_module_with_conditionals_preserves_block(self):
        # Cloning a Module containing ValueIf/ElseIf/Endif must
        # round-trip the emitted block exactly -- ParallelMap2-style
        # workers rely on this if they ever decide to deepcopy a
        # Module subtree (rare but legal).
        m = self._build_if_elseif_endif_module()
        clone = copy.deepcopy(m)
        self.assertIsNot(clone, m)
        self.assertEqual(str(clone), str(m))


# ===========================================================================
# Symbol-emission leaves -- ValueSet / RegSet.
# ===========================================================================
#
# Mirror of rocisa's ``ValueSet`` (assembler ``.set`` directive) and
# ``RegSet`` (a ValueSet subclass that also tracks VGPR allocation
# in the rocIsa singleton for MSB-aware archs).
# matters because KernelWriter sprinkles ``.set`` lines throughout
# every kernel and any divergence (whitespace, decimal-vs-hex,
# +offset literal preservation) shows up as a noisy diff against the
# rocisa baseline.


class TestValueSetCtorIntPath(unittest.TestCase):
    """Single Python ``__init__`` dispatches to the int-payload branch
    when ``value`` is not a string. Mirrors the ``int`` and
    ``uint32_t`` C++ ctors which both store the integer in ``value``
    and leave ``ref`` unset."""

    def test_minimal_construction(self):
        vs = ValueSet("foo", 42)
        self.assertEqual(vs.name, "foo")
        self.assertEqual(vs.value, 42)
        self.assertIsNone(vs.ref)
        self.assertEqual(vs.offset, 0)
        self.assertEqual(vs.format, 0)
        self.assertIsNone(vs.parent)

    def test_with_offset_and_format(self):
        vs = ValueSet("bar", 7, 3, 1)
        self.assertEqual(vs.value, 7)
        self.assertEqual(vs.offset, 3)
        self.assertEqual(vs.format, 1)

    def test_keyword_args_match_rocisa_names(self):
        vs = ValueSet(name="baz", value=5, offset=1, format=-1)
        self.assertEqual(vs.name, "baz")
        self.assertEqual(vs.value, 5)
        self.assertEqual(vs.offset, 1)
        self.assertEqual(vs.format, -1)


class TestValueSetCtorRefPath(unittest.TestCase):
    """``isinstance(value, str)`` discriminator routes string payloads
    into the ref-payload branch (mirror of the third C++ ctor)."""

    def test_string_value_stored_as_ref(self):
        vs = ValueSet("alias", "other_sym")
        self.assertEqual(vs.ref, "other_sym")
        self.assertIsNone(vs.value)

    def test_string_value_with_offset_and_format(self):
        vs = ValueSet("alias", "other_sym", 4, 0)
        self.assertEqual(vs.ref, "other_sym")
        self.assertIsNone(vs.value)
        self.assertEqual(vs.offset, 4)
        self.assertEqual(vs.format, 0)


class TestValueSetToStringValuePath(unittest.TestCase):
    """``.set <name>, <integer-payload>`` rendering for the three
    ``format`` codes. Byte-for-byte match required."""

    def test_format_default_zero_emits_value_plus_offset(self):
        # ``format == 0`` -> decimal ``str(value + offset)``.
        self.assertEqual(ValueSet("a", 5).toString(), ".set a, 5\n")
        self.assertEqual(ValueSet("a", 5, 3).toString(), ".set a, 8\n")

    def test_format_minus_one_emits_value_alone(self):
        # ``format == -1`` -> raw ``str(value)``; offset is NOT applied
        # on the value path (mirror of C++ which calls
        # ``std::to_string(value.value())`` directly in that branch).
        self.assertEqual(
            ValueSet("a", 100, 5, -1).toString(),
            ".set a, 100\n",
        )

    def test_format_one_emits_hex_with_offset(self):
        # ``format == 1`` -> ``"0x" + hex(value + offset)``, lowercase,
        # no padding.
        self.assertEqual(
            ValueSet("a", 0xFF, 0, 1).toString(),
            ".set a, 0xff\n",
        )
        self.assertEqual(
            ValueSet("a", 0x10, 0x20, 1).toString(),
            ".set a, 0x30\n",
        )
        self.assertEqual(
            ValueSet("a", 0, 0, 1).toString(),
            ".set a, 0x0\n",
        )

    def test_format_one_negative_uses_64bit_two_complement(self):
        # rocisa's ``std::hex`` over int64_t prints two's-complement
        # bits for negatives. ``-1`` -> ``0xffffffffffffffff``.
        self.assertEqual(
            ValueSet("a", -1, 0, 1).toString(),
            ".set a, 0xffffffffffffffff\n",
        )
        # ``-2 + 1 = -1`` after offset arithmetic.
        self.assertEqual(
            ValueSet("a", -2, 1, 1).toString(),
            ".set a, 0xffffffffffffffff\n",
        )


class TestValueSetToStringRefPath(unittest.TestCase):
    """``.set <name>, <ref-payload>`` rendering. ``format == -1`` emits
    the ref alone; any other format suffixes ``+<offset>`` (including
    the literal ``+0`` -- not short-circuited)."""

    def test_format_default_appends_plus_offset(self):
        self.assertEqual(
            ValueSet("a", "other", 7).toString(),
            ".set a, other+7\n",
        )

    def test_format_zero_offset_preserves_plus_zero_literal(self):
        # rocisa does NOT short-circuit ``offset == 0`` to ``ref``
        # alone -- the literal ``+0`` is preserved for byte parity.
        self.assertEqual(
            ValueSet("a", "other", 0).toString(),
            ".set a, other+0\n",
        )

    def test_format_minus_one_emits_ref_alone(self):
        self.assertEqual(
            ValueSet("a", "other", 7, -1).toString(),
            ".set a, other\n",
        )

    def test_format_one_on_ref_path_still_appends_offset(self):
        # ``format != -1`` -> ``ref + "+" + str(offset)``. The hex
        # branch is value-only; for ref+format==1 rocisa just does
        # the same plain ``+offset`` decimal output.
        self.assertEqual(
            ValueSet("a", "other", 3, 1).toString(),
            ".set a, other+3\n",
        )


class TestValueSetInheritance(unittest.TestCase):
    """``ValueSet`` is an ``Item`` subclass so Module type-walks /
    cap proxies / Item defaults all apply."""

    def test_isinstance_item(self):
        self.assertIsInstance(ValueSet("a", 1), Item)

    def test_str_goes_through_item_toString(self):
        # ``str(vs)`` must equal ``vs.toString()`` (via Item.__str__).
        vs = ValueSet("a", 5)
        self.assertEqual(str(vs), ".set a, 5\n")


class TestValueSetPickle(unittest.TestCase):
    """5-tuple round-trip: ``(name, ref, value, offset, format)``.
    Both ref and value branches must survive the round-trip with
    identical ``toString`` output."""

    def test_int_payload_round_trip(self):
        original = ValueSet("foo", 42, 3, 1)
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(restored.name, "foo")
        self.assertEqual(restored.value, 42)
        self.assertIsNone(restored.ref)
        self.assertEqual(restored.offset, 3)
        self.assertEqual(restored.format, 1)
        self.assertEqual(restored.toString(), original.toString())

    def test_ref_payload_round_trip(self):
        original = ValueSet("alias", "other", 5, 0)
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(restored.ref, "other")
        self.assertIsNone(restored.value)
        self.assertEqual(restored.offset, 5)
        self.assertEqual(restored.toString(), original.toString())


class TestValueSetDeepCopy(unittest.TestCase):
    def test_int_payload_independent(self):
        original = ValueSet("foo", 42, 3, 1)
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.value, 42)
        self.assertEqual(clone.toString(), original.toString())

    def test_ref_payload_independent(self):
        original = ValueSet("foo", "other", 3)
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.ref, "other")
        self.assertEqual(clone.toString(), original.toString())


class TestValueSetModuleIntegration(unittest.TestCase):
    """Module operations (add, str, countType) treat ValueSet leaves
    correctly -- parent rebind, recursive counting, concatenation."""

    def test_reparented_on_add(self):
        m = Module()
        vs = ValueSet("a", 1)
        m.add(vs)
        self.assertIs(vs.parent, m)

    def test_module_str_concatenates_toString(self):
        m = Module()
        m.add(ValueSet("a", 1))
        m.add(ValueSet("b", "other", 2))
        self.assertEqual(str(m), ".set a, 1\n.set b, other+2\n")

    def test_module_countType_finds_valueset(self):
        m = Module()
        m.add(ValueSet("a", 1))
        m.add(ValueSet("b", 2))
        self.assertEqual(m.countType(ValueSet), 2)
        # 1 (Module) + 2 (children) = 3 Items.
        self.assertEqual(m.countType(Item), 3)


# ===========================================================================
# RegSet -- ValueSet subclass with VGPR-index side effect.
# ===========================================================================


class TestRegSetCtor(_VgprIdxIsolation, unittest.TestCase):
    """RegSet ctor accepts ``(regType, name, int_or_str, offset=0)``
    and stores ``regType`` on top of ValueSet's fields."""

    def test_int_payload(self):
        self._caps["HasVgprMSB"] = 0  # disable side effect
        rs = RegSet("s", "sgprFoo", 5)
        self.assertEqual(rs.regType, "s")
        self.assertEqual(rs.name, "sgprFoo")
        self.assertEqual(rs.value, 5)
        self.assertIsNone(rs.ref)
        self.assertEqual(rs.offset, 0)
        self.assertEqual(rs.format, 0)

    def test_string_payload(self):
        self._caps["HasVgprMSB"] = 0
        rs = RegSet("s", "sgprFoo", "sgprOther", 3)
        self.assertEqual(rs.ref, "sgprOther")
        self.assertIsNone(rs.value)
        self.assertEqual(rs.offset, 3)

    def test_isinstance_valueset_and_item(self):
        self._caps["HasVgprMSB"] = 0
        rs = RegSet("s", "sgprFoo", 5)
        self.assertIsInstance(rs, ValueSet)
        self.assertIsInstance(rs, Item)


class TestRegSetVgprIdxSideEffect(_VgprIdxIsolation, unittest.TestCase):
    """When ``regType == "v"`` AND ``HasVgprMSB == 1``, both ``__init__``
    and ``toString`` MUST refresh ``getVgprIdx()`` with the latest
    binding (stripping the ``"vgpr"`` prefix from the name)."""

    def test_ctor_sets_index_int_payload(self):
        self._caps["HasVgprMSB"] = 1
        RegSet("v", "vgprFoo", 5, 2)
        self.assertEqual(self._base.getVgprIdx()["Foo"], 7)

    def test_ctor_sets_index_string_payload(self):
        self._caps["HasVgprMSB"] = 1
        self._base.getVgprIdx()["Existing"] = 10
        # ``RegSet("v", "vgprBar", "vgprExisting", 3)`` -> looks up
        # ``Existing`` (= 10) and registers ``Bar`` = 13.
        RegSet("v", "vgprBar", "vgprExisting", 3)
        self.assertEqual(self._base.getVgprIdx()["Bar"], 13)

    def test_ctor_string_payload_missing_key_uses_zero(self):
        # Mirror of ``std::map<string,int>::operator[]`` which value-
        # initialises missing keys to 0 instead of throwing.
        # KernelWriter's ``macroAndSet`` deliberately establishes
        # ``RegSet("v", "vgprG2LA", "vgprG2LA_BASE", 0)`` BEFORE
        # ``vgprG2LA_BASE`` has been registered -- rocisa silently
        # treats the missing key as 0 so the alias resolves to 0+0=0.
        # If we raise here, KernelWriter crashes mid-kernel.
        self._caps["HasVgprMSB"] = 1
        self.assertNotIn("G2LA_BASE", self._base.getVgprIdx())
        RegSet("v", "vgprG2LA", "vgprG2LA_BASE", 0)
        self.assertEqual(self._base.getVgprIdx()["G2LA"], 0)
        # Lookup of a missing key MUST NOT auto-insert into the live
        # map (rocisa looks up in a copy of the map, so the singleton
        # is unaffected).
        self.assertNotIn("G2LA_BASE", self._base.getVgprIdx())

    def test_ctor_string_payload_missing_key_with_offset(self):
        # Same parity rule but with a non-zero offset -- missing key
        # contributes 0 to the sum, offset survives.
        self._caps["HasVgprMSB"] = 1
        RegSet("v", "vgprTarget", "vgprMissing", 7)
        self.assertEqual(self._base.getVgprIdx()["Target"], 7)

    def test_toString_re_triggers_setIdx(self):
        # Construct RegSet, mutate the in-memory map, call toString,
        # and verify the map was refreshed back to the RegSet's view
        # of the world.
        self._caps["HasVgprMSB"] = 1
        rs = RegSet("v", "vgprFoo", 5, 2)
        self.assertEqual(self._base.getVgprIdx()["Foo"], 7)
        # External mutation -- simulate a stale snapshot.
        self._base.getVgprIdx()["Foo"] = 999
        # toString must re-set to 7 (mirror of C++ which calls setIdx
        # at the top of toString).
        rs.toString()
        self.assertEqual(self._base.getVgprIdx()["Foo"], 7)

    def test_toString_output_does_not_include_regType(self):
        # The side-effecting toString delegates string formatting to
        # ValueSet.toString -- regType must NOT appear in the output.
        self._caps["HasVgprMSB"] = 1
        rs = RegSet("v", "vgprFoo", 5)
        self.assertEqual(rs.toString(), ".set vgprFoo, 5\n")


class TestRegSetNoSideEffectWhenDisabled(_VgprIdxIsolation, unittest.TestCase):
    """The side effect is gated on BOTH ``regType == "v"`` AND
    ``HasVgprMSB``; missing either skips the index update."""

    def test_sgpr_does_not_set_index(self):
        self._caps["HasVgprMSB"] = 1
        before = dict(self._base.getVgprIdx())
        rs = RegSet("s", "sgprFoo", 5)
        rs.toString()
        self.assertEqual(self._base.getVgprIdx(), before)

    def test_vgpr_without_HasVgprMSB_does_not_set_index(self):
        self._caps["HasVgprMSB"] = 0
        before = dict(self._base.getVgprIdx())
        rs = RegSet("v", "vgprFoo", 5)
        rs.toString()
        self.assertEqual(self._base.getVgprIdx(), before)

    def test_missing_HasVgprMSB_key_is_safe(self):
        # Missing key must behave the same as ``0`` (matches C++
        # ``std::map[]`` value-initialisation).
        self._caps.pop("HasVgprMSB", None)
        before = dict(self._base.getVgprIdx())
        rs = RegSet("v", "vgprFoo", 5)
        rs.toString()
        self.assertEqual(self._base.getVgprIdx(), before)


class TestRegSetPickle(_VgprIdxIsolation, unittest.TestCase):
    """6-tuple round-trip ``(regType, name, ref, value, offset, format)``.
    ``format`` is preserved even though the ctor does not accept it
    (mirror of C++ which does ``self.format = std::get<5>(t)`` after
    placement-new)."""

    def test_int_payload_round_trip_preserves_format(self):
        self._caps["HasVgprMSB"] = 0
        original = RegSet("s", "sgprFoo", 5, 2)
        original.format = 1  # mutate post-ctor
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(restored.regType, "s")
        self.assertEqual(restored.name, "sgprFoo")
        self.assertEqual(restored.value, 5)
        self.assertIsNone(restored.ref)
        self.assertEqual(restored.offset, 2)
        self.assertEqual(restored.format, 1)
        self.assertEqual(restored.toString(), original.toString())

    def test_ref_payload_round_trip(self):
        self._caps["HasVgprMSB"] = 0
        original = RegSet("s", "sgprFoo", "sgprOther", 3)
        restored = pickle.loads(pickle.dumps(original))
        self.assertEqual(restored.regType, "s")
        self.assertEqual(restored.ref, "sgprOther")
        self.assertIsNone(restored.value)
        self.assertEqual(restored.toString(), original.toString())

    def test_round_trip_re_fires_setIdx_on_HasVgprMSB(self):
        # ``__setstate__`` MUST re-fire the side effect (matches C++
        # which restores via placement-new of the RegSet ctor).
        self._caps["HasVgprMSB"] = 1
        original = RegSet("v", "vgprFoo", 5, 2)
        # Clear the map so we can observe the restore re-populating it.
        self._base.getVgprIdx().clear()
        pickle.loads(pickle.dumps(original))
        self.assertEqual(self._base.getVgprIdx()["Foo"], 7)


class TestRegSetDeepCopy(_VgprIdxIsolation, unittest.TestCase):
    def test_int_payload_independent_with_format_preserved(self):
        self._caps["HasVgprMSB"] = 0
        original = RegSet("s", "sgprFoo", 5, 2)
        original.format = 1
        clone = copy.deepcopy(original)
        self.assertIsNot(clone, original)
        self.assertEqual(clone.format, 1)
        self.assertEqual(clone.toString(), original.toString())

    def test_ref_payload_independent(self):
        self._caps["HasVgprMSB"] = 0
        original = RegSet("v", "vgprFoo", "vgprOther", 1)
        clone = copy.deepcopy(original)
        self.assertEqual(clone.regType, "v")
        self.assertEqual(clone.ref, "vgprOther")


class TestRegSetModuleIntegration(_VgprIdxIsolation, unittest.TestCase):
    """Mix RegSet leaves into a Module tree and verify str / countType /
    parent rebind behave like any other Item subclass."""

    def test_reparented_on_add(self):
        self._caps["HasVgprMSB"] = 0
        m = Module()
        rs = RegSet("s", "sgprFoo", 5)
        m.add(rs)
        self.assertIs(rs.parent, m)

    def test_module_str_concatenates_regset_lines(self):
        self._caps["HasVgprMSB"] = 0
        m = Module()
        m.add(RegSet("v", "vgprA", 0))
        m.add(RegSet("v", "vgprB", "vgprA", 1))
        self.assertEqual(
            str(m),
            ".set vgprA, 0\n.set vgprB, vgprA+1\n",
        )

    def test_countType_distinguishes_regset_from_valueset(self):
        # ``countType`` matches by isinstance; RegSet IS a ValueSet so
        # counting ValueSet must include RegSets too. Counting RegSet
        # alone must NOT include the plain ValueSet.
        self._caps["HasVgprMSB"] = 0
        m = Module()
        m.add(ValueSet("a", 1))
        m.add(RegSet("s", "sgprB", 2))
        m.add(RegSet("v", "vgprC", 3))
        self.assertEqual(m.countType(ValueSet), 3)
        self.assertEqual(m.countType(RegSet), 2)
        # ``countExactType`` must distinguish by exact class -- only
        # the plain ValueSet matches when targeting ValueSet exactly.
        self.assertEqual(m.countExactType(ValueSet), 1)
        self.assertEqual(m.countExactType(RegSet), 2)


# ===========================================================================
# Label -- branch / loop target leaf (real implementation).
# ===========================================================================
#
# Mirror of ``rocisa::Label``. Tests cover the int / str payload
# variants, the static ``getFormatting`` helper, the ``getLabelName``
# accessor, ``toString`` formatting (alignment prefix, comment
# suffix, ``outputNoComment`` gating), the ``HasVgprMSB`` side
# effect (resets ``setVgprMsb(-1)`` after emission), and the
# pickle / deepcopy / Module-integration round-trips.


class TestLabelConstruction(unittest.TestCase):
    """``Label(label, comment, alignment=1)`` stores the three fields
    verbatim and leaves ``Item.name`` empty (rocisa's ``Item("")``)."""

    def test_int_label(self):
        lbl = Label(5, "")
        self.assertEqual(lbl.label, 5)
        self.assertEqual(lbl.comment, "")
        self.assertEqual(lbl.alignment, 1)
        # Item.name is the EMPTY STRING -- the textual identity comes
        # from getLabelName(), not from Item.name. Critical for
        # rocisa's ``findNamedItem`` semantics: a Label is NEVER
        # findable by its label payload through findNamedItem.
        self.assertEqual(lbl.name, "")

    def test_string_label(self):
        lbl = Label("foo", "bar")
        self.assertEqual(lbl.label, "foo")
        self.assertEqual(lbl.comment, "bar")
        self.assertEqual(lbl.alignment, 1)

    def test_explicit_alignment(self):
        lbl = Label(5, "x", 8)
        self.assertEqual(lbl.alignment, 8)

    def test_keyword_args(self):
        # rocisa's nanobind binding (code.cpp:108-116) names the args
        # ``label`` / ``comment`` / ``alignment`` -- KernelWriter
        # (KernelWriterAssembly.py:2139) uses keyword form, so we
        # must accept it.
        lbl = Label(label="foo", comment="bar", alignment=4)
        self.assertEqual(lbl.label, "foo")
        self.assertEqual(lbl.comment, "bar")
        self.assertEqual(lbl.alignment, 4)

    def test_is_item_subclass(self):
        self.assertIsInstance(Label(0, ""), Item)


class TestLabelGetFormatting(unittest.TestCase):
    """Static ``Label.getFormatting`` formats an ``int | str`` payload
    into ``label_<text>``. Mirror of code.hpp:87-103."""

    def test_int_payload(self):
        self.assertEqual(Label.getFormatting(5), "label_5")

    def test_string_payload(self):
        self.assertEqual(Label.getFormatting("foo"), "label_foo")

    def test_negative_int_payload(self):
        # Negative int must NOT lose its sign -- the format string
        # uses the default ``__format__`` which preserves it.
        self.assertEqual(Label.getFormatting(-1), "label_-1")


class TestLabelGetLabelName(unittest.TestCase):
    """``Label.getLabelName`` is the public accessor branch
    instructions reference -- forwards to ``getFormatting(self.label)``."""

    def test_int_payload(self):
        self.assertEqual(Label(5, "").getLabelName(), "label_5")

    def test_string_payload(self):
        self.assertEqual(Label("foo", "").getLabelName(), "label_foo")


class TestLabelToString(_VgprMsbIsolation, unittest.TestCase):
    """``Label.toString`` produces ``[.align <N>\\n]label_<x>:[  /// <c>]\\n``.

    Mirror of code.hpp:110-127. Caps init is required because the
    method reads ``getAsmCaps()["HasVgprMSB"]``; we suppress the
    side effect for these formatting-focused tests by forcing the
    cap to 0.
    """

    def setUp(self) -> None:
        super().setUp()
        # Disable the MSB side effect for pure formatting tests --
        # it's covered separately in TestLabelMsbSideEffect.
        self._caps["HasVgprMSB"] = 0

    def test_basic_int(self):
        self.assertEqual(Label(5, "").toString(), "label_5:\n")

    def test_basic_string(self):
        self.assertEqual(Label("L", "").toString(), "label_L:\n")

    def test_with_comment(self):
        # Two-space indent + ``///`` + single space + comment text,
        # no padding (unlike ValueEndif which width-pads to col 50).
        self.assertEqual(Label(5, "hi").toString(), "label_5:  /// hi\n")

    def test_alignment_one_emits_no_prefix(self):
        # ``alignment == 1`` is the default and is treated as "no
        # prefix" -- the ``.align 1\n`` line is suppressed because
        # alignment-of-1 is a no-op assembler directive.
        self.assertEqual(Label(5, "", 1).toString(), "label_5:\n")

    def test_alignment_only(self):
        # ``alignment > 1`` prepends ``.align <N>\n`` to the label
        # line. KernelWriter uses this for cache-line-aligned loop
        # entries.
        self.assertEqual(Label(5, "", 8).toString(), ".align 8\nlabel_5:\n")

    def test_alignment_and_comment(self):
        self.assertEqual(
            Label(5, "hi", 8).toString(),
            ".align 8\nlabel_5:  /// hi\n",
        )

    def test_outputNoComment_suppresses_comment(self):
        # ``rocIsa.outputNoComment=True`` blanket-suppresses the
        # ``  /// <comment>`` suffix while keeping the label line
        # itself. Production builds rely on this to scrub
        # human-readable annotations from the emitted asm.
        from rocisa_stinkytofu_adaptor import rocIsa  # noqa: WPS433
        opts = rocIsa.getInstance().getOutputOptions()
        saved = opts.outputNoComment
        try:
            opts.outputNoComment = True
            self.assertEqual(Label(5, "hi").toString(), "label_5:\n")
            # Alignment is NOT a comment, must still be emitted.
            self.assertEqual(
                Label(5, "hi", 8).toString(),
                ".align 8\nlabel_5:\n",
            )
        finally:
            opts.outputNoComment = saved

    def test_str_dunder_equals_toString(self):
        # ``Module.toString`` concatenates ``str(it)``; ``__str__``
        # must therefore go through Item.__str__ -> toString to match
        # rocisa's binding (code.cpp:122).
        lbl = Label(5, "hi")
        self.assertEqual(str(lbl), lbl.toString())


class TestLabelMsbSideEffect(_VgprMsbIsolation, unittest.TestCase):
    """``Label.toString`` resets ``setVgprMsb(-1)`` on a HasVgprMSB
    arch -- mirror of code.hpp:122-125. The semantic is "emitting a
    label means we're entering a new basic block whose entry MSB is
    not knowable", so callers must NOT rely on the pre-toString MSB
    surviving the call."""

    def test_msb_resets_when_HasVgprMSB(self):
        self._caps["HasVgprMSB"] = 1
        self._base.setVgprMsb(7)
        Label(5, "").toString()
        self.assertEqual(self._base.getVgprMsb(), -1)

    def test_msb_unchanged_when_no_HasVgprMSB(self):
        # Cap absent / zero ⇒ no side effect. The pre-toString MSB
        # value MUST round-trip unchanged.
        self._caps["HasVgprMSB"] = 0
        self._base.setVgprMsb(7)
        Label(5, "").toString()
        self.assertEqual(self._base.getVgprMsb(), 7)

    def test_msb_unchanged_before_toString(self):
        # Reading ``label.label`` / ``label.comment`` / etc. must NOT
        # trigger the side effect -- only ``toString`` does.
        self._caps["HasVgprMSB"] = 1
        self._base.setVgprMsb(7)
        lbl = Label(5, "")
        # Touch attributes; verify MSB still 7.
        _ = (lbl.label, lbl.comment, lbl.alignment, lbl.getLabelName())
        self.assertEqual(self._base.getVgprMsb(), 7)


class TestLabelDeepCopy(unittest.TestCase):
    """``copy.deepcopy(label)`` produces an isolated clone with every
    field preserved. Mirror of code.cpp:123-127."""

    def test_deepcopy_preserves_fields(self):
        original = Label("foo", "bar", 8)
        clone = copy.deepcopy(original)
        self.assertEqual(clone.label, "foo")
        self.assertEqual(clone.comment, "bar")
        self.assertEqual(clone.alignment, 8)

    def test_deepcopy_independent(self):
        # Mutating the clone must NOT bleed back into the original.
        original = Label(5, "hi", 4)
        clone = copy.deepcopy(original)
        clone.label = 99
        clone.comment = "changed"
        clone.alignment = 16
        self.assertEqual(original.label, 5)
        self.assertEqual(original.comment, "hi")
        self.assertEqual(original.alignment, 4)

    def test_deepcopy_preserves_patched_name(self):
        # Item.name defaults to "" but a caller may patch it post-
        # construction; deepcopy must round-trip the patched value.
        original = Label(5, "hi")
        original.name = "custom_name"
        clone = copy.deepcopy(original)
        self.assertEqual(clone.name, "custom_name")


class TestLabelPickle(unittest.TestCase):
    """Pickle round-trip uses the 4-tuple ``(name, label, comment,
    alignment)`` shape from rocisa's ``__getstate__`` /
    ``__setstate__`` (code.cpp:128-138)."""

    def test_pickle_int_payload(self):
        original = Label(5, "hi", 4)
        clone = pickle.loads(pickle.dumps(original))
        self.assertEqual(clone.label, 5)
        self.assertEqual(clone.comment, "hi")
        self.assertEqual(clone.alignment, 4)
        self.assertEqual(clone.name, "")

    def test_pickle_string_payload(self):
        original = Label("foo", "bar", 1)
        clone = pickle.loads(pickle.dumps(original))
        self.assertEqual(clone.label, "foo")
        self.assertEqual(clone.comment, "bar")
        self.assertEqual(clone.alignment, 1)

    def test_pickle_preserves_patched_name(self):
        # ``__setstate__`` calls ``__init__`` (which resets name="")
        # then patches name -- mirrors rocisa's placement-new +
        # post-patch pattern. The tuple's name field must round-trip.
        original = Label(5, "hi")
        original.name = "custom"
        clone = pickle.loads(pickle.dumps(original))
        self.assertEqual(clone.name, "custom")


class TestLabelModuleIntegration(_VgprMsbIsolation, unittest.TestCase):
    """Real Label inside a real Module -- emission, parent linkage,
    and findIndexByType all behave like any other Item."""

    def setUp(self) -> None:
        super().setUp()
        self._caps["HasVgprMSB"] = 0  # disable side effect

    def test_module_str_concatenates_label_lines(self):
        m = Module()
        m.add(Label("loop_top", ""))
        m.add(TextBlock("v_mov_b32 v0, 0\n"))
        m.add(Label(2, "exit", 8))
        self.assertEqual(
            str(m),
            "label_loop_top:\nv_mov_b32 v0, 0\n.align 8\nlabel_2:  /// exit\n",
        )

    def test_label_parent_set_after_module_add(self):
        # Module.add patches child.parent. Critical for Item walks
        # that rely on upward navigation (e.g. tree-prefixed
        # prettyPrint).
        m = Module()
        lbl = Label(5, "")
        m.add(lbl)
        self.assertIs(lbl.parent, m)

    def test_findIndexByType_finds_label(self):
        # ``findIndexByType(Label)`` must locate the real Label and
        # NOT confuse it with siblings of a different concrete type.
        m = Module()
        m.add(TextBlock("a"))
        m.add(Label("L", ""))
        m.add(TextBlock("b"))
        self.assertEqual(m.findIndexByType(Label), 1)


# ===========================================================================
# Macro -- ``.macro <name> args ... .endm`` block.
# ===========================================================================
#
# KernelWriter constructs 4 macros (KWA ``GLOBAL_OFFSET_*``, ``MAC_*``,
# ``MAC_*_OneIUI``; Components/CustomSchedule.py ``MAINLOOP``). Each
# is filled with CommonInstruction / Module / TextBlock children, then
# attached to a parent Module via ``module.add(macro)``.


class TestMacroConstruction(unittest.TestCase):
    def test_basic_construction(self):
        mc = Macro("GLOBAL_OFFSET_A", ["vgprAddr:req", "vgprTmp:req"])
        self.assertEqual(mc.name, "GLOBAL_OFFSET_A")
        self.assertEqual(mc.itemList, [])
        # Internal header object type; field parity is in
        # ``test_instruction.TestMacroInstruction*``.
        from rocisa_stinkytofu_adaptor.instruction import MacroInstruction
        self.assertIsInstance(mc.macro, MacroInstruction)

    def test_empty_args(self):
        # KWA:1774 pattern: ``Macro("MAC_...", [])``.
        mc = Macro("MAC_4x4_X0", [])
        self.assertEqual(mc.macro.args, [])

    def test_inherits_Item(self):
        mc = Macro("X", [])
        self.assertIsInstance(mc, Item)

    def test_is_NOT_Module_subclass(self):
        # rocisa::Macro inherits from Item directly, NOT from Module.
        # KernelWriter type-walks rely on this distinction so e.g.
        # ``Macro`` doesn't get picked up by ``findIndexByType(Module)``.
        mc = Macro("X", [])
        self.assertNotIsInstance(mc, Module)


class TestMacroAdd(unittest.TestCase):
    def test_add_TextBlock(self):
        mc = Macro("X", [])
        tb = TextBlock("inst\n")
        ret = mc.add(tb)
        self.assertIs(ret, tb)
        self.assertEqual(mc.itemList, [tb])
        self.assertIs(tb.parent, mc)

    def test_add_Module(self):
        mc = Macro("X", [])
        inner = Module("inner")
        mc.add(inner)
        self.assertIs(inner.parent, mc)

    def test_add_ValueIf_ValueElseIf_ValueEndif(self):
        # Whitelist includes the value-conditional family.
        mc = Macro("X", [])
        for it in [ValueIf("1"), ValueElseIf("2"), ValueEndif()]:
            mc.add(it)
        self.assertEqual(len(mc.itemList), 3)

    def test_add_Instruction_subclass(self):
        # CommonInstruction / MacroInstruction subclasses must be
        # accepted -- this is the common case (KernelWriter feeds
        # macros V* / S* instructions).
        from rocisa_stinkytofu_adaptor.instruction import (
            MacroInstruction, VMovB32,
        )
        from rocisa_stinkytofu_adaptor.container import vgpr
        mc = Macro("X", [])
        mc.add(VMovB32(vgpr(0), vgpr(1)))
        mc.add(MacroInstruction("Y", [1]))
        self.assertEqual(len(mc.itemList), 2)

    def test_add_unknown_type_raises(self):
        # rocisa throws "unknown item type for Macro.add: ...".
        mc = Macro("X", [])
        with self.assertRaises(RuntimeError) as ctx:
            mc.add("just a string")
        self.assertIn("unknown item type for Macro.add", str(ctx.exception))

    def test_add_int_raises(self):
        mc = Macro("X", [])
        with self.assertRaises(RuntimeError):
            mc.add(42)


class TestMacroAddComment0(unittest.TestCase):
    """Single-line ``/* ... */\\n``, distinct from Module's 3-line banner."""

    def test_addComment0_single_line(self):
        mc = Macro("X", [])
        mc.addComment0("hello")
        self.assertEqual(len(mc.itemList), 1)
        tb = mc.itemList[0]
        self.assertIsInstance(tb, TextBlock)
        self.assertEqual(tb.text, "/* hello */\n")

    def test_addComment0_same_as_Module_addComment0(self):
        # Both Module and Macro addComment0 now produce single-line
        # block comments matching native format.hpp::block().
        mc = Macro("X", [])
        mc.addComment0("hi")
        mod = Module()
        mod.addComment0("hi")
        self.assertEqual(mc.itemList[0].text, mod.itemList[0].text)


class TestMacroSetItems(unittest.TestCase):
    def test_setItems_replaces_list(self):
        mc = Macro("X", [])
        mc.add(TextBlock("a"))
        new_items = [TextBlock("b"), TextBlock("c")]
        mc.setItems(new_items)
        self.assertEqual(mc.itemList, new_items)

    def test_setItems_does_NOT_reparent(self):
        # rocisa C++ does plain assignment; no type check and no parent
        # update. Mirror exactly so SetItems is a fast no-frills swap
        # (callers that care must reparent manually).
        mc = Macro("X", [])
        tb = TextBlock("a")
        mc.setItems([tb])
        self.assertIsNone(tb.parent)  # NOT reparented


class TestMacroToString(unittest.TestCase):
    """Byte-parity with ``rocisa::Macro::toString``."""

    def test_empty_macro(self):
        mc = Macro("X", [])
        self.assertEqual(str(mc), ".macro X\n.endm\n")

    def test_macro_with_args_only(self):
        mc = Macro("GLOBAL_OFFSET_A", ["vgprAddr:req", "vgprTmp:req"])
        self.assertEqual(
            str(mc),
            ".macro GLOBAL_OFFSET_A vgprAddr:req, vgprTmp:req\n.endm\n",
        )

    def test_macro_with_text_children_indented(self):
        # Each child line is prefixed with 4 spaces.
        mc = Macro("X", [])
        mc.add(TextBlock("v_add v0, v1, v2\n"))
        mc.add(TextBlock("v_mul v3, v4, v5\n"))
        expected = (
            ".macro X\n"
            "    v_add v0, v1, v2\n"
            "    v_mul v3, v4, v5\n"
            ".endm\n"
        )
        self.assertEqual(str(mc), expected)

    def test_s_set_vgpr_msb_quirk_extra_indent(self):
        # rocisa C++ hack: any child whose toString contains
        # ``s_set_vgpr_msb`` gets an EXTRA 4-space indent inserted
        # right after the first newline (so subsequent body lines
        # stay visually aligned with auto-inserted MSB toggles).
        mc = Macro("foo", [])
        mc.add(TextBlock("s_set_vgpr_msb 1\nactual_inst v0, v1\n"))
        expected = (
            ".macro foo\n"
            "    s_set_vgpr_msb 1\n"
            "    actual_inst v0, v1\n"
            ".endm\n"
        )
        self.assertEqual(str(mc), expected)

    def test_addComment0_renders_inside_macro_body(self):
        mc = Macro("X", [])
        mc.addComment0("note")
        expected = ".macro X\n    /* note */\n.endm\n"
        self.assertEqual(str(mc), expected)


class TestMacroPrettyPrint(unittest.TestCase):
    def test_prettyPrint_header_and_children(self):
        mc = Macro("MyMacro", [])
        mc.add(TextBlock("inst\n"))
        out = mc.prettyPrint()
        # Header: `Macro "MyMacro"\n`, then child via `|--` prefix.
        self.assertIn('Macro "MyMacro"', out)
        self.assertIn("|--", out)


class TestMacroDeepCopy(unittest.TestCase):
    def test_deepcopy_independent(self):
        mc = Macro("X", ["a:req"])
        mc.add(TextBlock("inst\n"))
        c = copy.deepcopy(mc)
        # Mutating clone must not affect original.
        c.add(TextBlock("extra\n"))
        self.assertEqual(len(mc.itemList), 1)
        self.assertEqual(len(c.itemList), 2)

    def test_deepcopy_clones_header_macro(self):
        # The internal ``self.macro`` (a MacroInstruction) must be deep-
        # cloned -- mutating clone's macro args must not bleed in.
        mc = Macro("X", [1, 2])
        c = copy.deepcopy(mc)
        c.macro.args.append(99)
        self.assertEqual(mc.macro.args, [1, 2])
        self.assertEqual(c.macro.args, [1, 2, 99])

    def test_deepcopy_reparents_children(self):
        # Cloned children's parent must point to the clone, not the
        # original (matches rocisa C++ parent-fixup).
        mc = Macro("X", [])
        mc.add(TextBlock("inst\n"))
        c = copy.deepcopy(mc)
        self.assertIs(c.itemList[0].parent, c)
        self.assertIsNot(c.itemList[0].parent, mc)

    def test_deepcopy_preserves_subclass(self):
        mc = Macro("X", [])
        c = copy.deepcopy(mc)
        self.assertIs(type(c), Macro)


class TestMacroPickleRejected(unittest.TestCase):
    def test_pickle_raises(self):
        import pickle
        mc = Macro("X", [])
        with self.assertRaises(RuntimeError):
            pickle.dumps(mc)


class TestMacroModuleIntegration(unittest.TestCase):
    """KernelWriter pattern: ``module.add(macro)`` -- the Macro itself
    is an Item, so a parent Module can carry it like any other child."""

    def test_macro_added_to_module(self):
        mod = Module("kernel")
        mc = Macro("GLOBAL_OFFSET_A", ["vgprAddr:req"])
        mc.add(TextBlock("v_add v0, v1, v2\n"))
        mod.add(mc)
        # Module parented the macro; rendering nests the .macro block
        # inside the module's flat join.
        self.assertIs(mc.parent, mod)
        s = str(mod)
        self.assertIn(".macro GLOBAL_OFFSET_A", s)
        self.assertIn(".endm", s)

    def test_macro_survives_module_deepcopy(self):
        # When a parent Module is deepcopied, the nested Macro must
        # come along (with its own internal header MacroInstruction).
        mod = Module()
        mod.add(Macro("X", [1]))
        c = copy.deepcopy(mod)
        self.assertIsInstance(c.itemList[0], Macro)
        self.assertEqual(c.itemList[0].macro.args, [1])


# ===========================================================================
# SignatureCodeMeta / SignatureBase
# ===========================================================================


def _make_signature_code_meta() -> SignatureCodeMeta:
    return SignatureCodeMeta(
        "k",
        kernArgsVersion=1,
        groupSegSize=256,
        flatWgSize=64,
        codeObjectVersion="4",
    )


def _make_signature_base(**kwargs) -> SignatureBase:
    defaults = dict(
        kernelName="k",
        kernArgsVersion=1,
        codeObjectVersion="4",
        groupSegmentSize=256,
        sgprWorkGroup=(1, 1, 0),
        vgprWorkItem=0,
        flatWorkGroupSize=64,
    )
    defaults.update(kwargs)
    return SignatureBase(**defaults)


class TestSignatureCodeMetaConstruction(unittest.TestCase):
    def test_is_item_subclass(self):
        meta = _make_signature_code_meta()
        self.assertIsInstance(meta, Item)

    def test_ctor_fields(self):
        meta = _make_signature_code_meta()
        self.assertEqual(meta.name, "k")
        self.assertEqual(meta.kernArgsVersion, 1)
        self.assertEqual(meta.groupSegSize, 256)
        self.assertEqual(meta.flatWgSize, 64)
        self.assertEqual(meta.codeObjectVersion, "4")
        self.assertEqual(meta.offset, 0)
        self.assertEqual(meta.argList, [])


class TestSignatureCodeMetaAddArg(_SignatureKernelSetup, unittest.TestCase):
    def test_add_arg_accumulates_offset(self):
        meta = _make_signature_code_meta()
        meta.addArg("alpha", SVK.SIG_VALUE, "f32")
        meta.addArg("D", SVK.SIG_GLOBALBUFFER, "f32", "generic")
        self.assertEqual(len(meta.argList), 2)
        self.assertEqual(meta.argList[0].offset, 0)
        self.assertEqual(meta.argList[0].size, 4)
        self.assertEqual(meta.argList[1].offset, 4)
        self.assertEqual(meta.argList[1].size, 8)
        self.assertEqual(meta.offset, 12)

    def test_unknown_value_type_raises(self):
        meta = _make_signature_code_meta()
        with self.assertRaises(RuntimeError):
            meta.addArg("x", SVK.SIG_VALUE, "not_a_type")


class TestSignatureCodeMetaSetGprs(_SignatureKernelSetup, unittest.TestCase):
    def test_set_gprs_updates_counts(self):
        meta = _make_signature_code_meta()
        meta.setGprs(40, 20)
        s = meta.toString()
        self.assertIn(".vgpr_count:                 40", s)
        self.assertIn(".sgpr_count:                 20", s)


class TestSignatureCodeMetaToString(_SignatureKernelSetup, unittest.TestCase):
    def test_metadata_header_and_kernarg_align(self):
        meta = _make_signature_code_meta()
        meta.addArg("numWG", SVK.SIG_VALUE, "u32")
        s = meta.toString()
        self.assertTrue(s.startswith(".amdgpu_metadata\n"))
        self.assertIn("KernArgsVersion: 1", s)
        self.assertIn("amdhsa.version:\n  - 1\n  - 1\n", s)
        self.assertIn(".kernarg_segment_size:       8\n", s)
        self.assertIn(".wavefront_size:             64\n", s)
        self.assertTrue(s.endswith("k:\n"))

    def test_code_object_version_five(self):
        meta = SignatureCodeMeta("k", 1, 0, 64, "5")
        s = meta.toString()
        self.assertIn("amdhsa.version:\n  - 1\n  - 2\n", s)


class TestSignatureCodeMetaCopyRejected(unittest.TestCase):
    def test_deepcopy_raises(self):
        meta = _make_signature_code_meta()
        with self.assertRaises(RuntimeError):
            copy.deepcopy(meta)

    def test_pickle_raises(self):
        meta = _make_signature_code_meta()
        with self.assertRaises(RuntimeError):
            pickle.dumps(meta)


class TestSignatureBaseConstruction(unittest.TestCase):
    def test_is_item_subclass(self):
        sig = _make_signature_base()
        self.assertIsInstance(sig, Item)

    def test_composes_descriptor_and_meta(self):
        sig = _make_signature_base(kernelName="my_k")
        self.assertEqual(sig.name, "my_k")
        self.assertEqual(sig.kernelDescriptor.name, "my_k")
        self.assertEqual(sig.codeMeta.name, "my_k")


class TestSignatureBaseSetGprs(_SignatureKernelSetup, unittest.TestCase):
    def test_set_gprs_syncs_both_children(self):
        sig = _make_signature_base()
        sig.setGprs(32, 4, 16)
        self.assertEqual(sig.getNextFreeVgpr(), 32)
        self.assertEqual(sig.getNextFreeSgpr(), 16)
        s = sig.toString()
        self.assertIn(".amdhsa_next_free_vgpr 32 // vgprs", s)
        self.assertIn(".vgpr_count:                 32", s)


class TestSignatureBaseAddArg(_SignatureKernelSetup, unittest.TestCase):
    def test_add_arg_delegates_to_code_meta(self):
        sig = _make_signature_base()
        sig.addArg("A", SVK.SIG_GLOBALBUFFER, "f32", "generic")
        self.assertEqual(len(sig.codeMeta.argList), 1)
        self.assertIn("- .name:            A", sig.toString())


class TestSignatureBaseDescriptions(_SignatureKernelSetup, unittest.TestCase):
    def test_description_helpers_emit_in_order(self):
        sig = _make_signature_base()
        sig.addDescriptionTopic("Optimizations and Config:")
        sig.addDescriptionBlock("ThreadTile= 8 x 8")
        s = sig.toString()
        self.assertIn("/* Optimizations and Config:", s)
        self.assertIn("/* ThreadTile= 8 x 8 */", s)

        sig.addDescription("tail note")
        sig.clearDescription()
        sig.addDescriptionBlock("after clear")
        s2 = sig.toString()
        self.assertNotIn("tail note", s2)
        self.assertNotIn("ThreadTile= 8 x 8", s2)
        self.assertIn("/* after clear */", s2)
        self.assertIn("Optimizations and Config:", s2)


class TestSignatureBaseToString(_SignatureKernelSetup, unittest.TestCase):
    def test_smoke_matches_rocisa_test_shape(self):
        sig = SignatureBase(
            kernelName="123",
            kernArgsVersion=1,
            codeObjectVersion="4",
            groupSegmentSize=256,
            sgprWorkGroup=(1, 1, 100),
            vgprWorkItem=1,
            flatWorkGroupSize=256,
            numSgprPreload=16,
        )
        s = str(sig)
        self.assertIn('.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"', s)
        self.assertIn(".protected 123", s)
        self.assertIn(".amdhsa_user_sgpr_count 18\n", s)
        self.assertIn(".amdhsa_user_sgpr_kernarg_preload_length 16\n", s)
        self.assertIn(".amdhsa_user_sgpr_kernarg_preload_offset 0\n", s)
        self.assertIn(".amdgpu_metadata", s)
        self.assertIn("123:\n", s)

    def test_num_sgpr_preload_zero_omits_preload_lines(self):
        sig = _make_signature_base(numSgprPreload=0)
        s = str(sig)
        self.assertNotIn(".amdhsa_user_sgpr_kernarg_preload_length", s)
        self.assertNotIn(".amdhsa_user_sgpr_count", s)


class TestSignatureBaseCopyRejected(unittest.TestCase):
    def test_deepcopy_raises(self):
        sig = _make_signature_base()
        with self.assertRaises(RuntimeError):
            copy.deepcopy(sig)

    def test_pickle_raises(self):
        sig = _make_signature_base()
        with self.assertRaises(RuntimeError):
            pickle.dumps(sig)


# ===========================================================================
# KernelBody
# ===========================================================================


def _make_kernel_body(**kwargs) -> KernelBody:
    name = kwargs.pop("name", "kernelBody")
    return KernelBody(name)


class TestKernelBodyConstruction(unittest.TestCase):
    def test_is_item_subclass(self):
        kb = _make_kernel_body()
        self.assertIsInstance(kb, Item)

    def test_ctor_sets_name_and_defaults(self):
        kb = KernelBody("kernelBody")
        self.assertEqual(kb.name, "kernelBody")
        self.assertIsNone(kb.signature)
        self.assertIsNone(kb.body)
        self.assertEqual(kb.totalVgprs, 0)
        self.assertEqual(kb.totalAgprs, 0)
        self.assertEqual(kb.totalSgprs, 0)


class TestKernelBodyAddSignatureAndBody(_SignatureKernelSetup, unittest.TestCase):
    def test_add_signature_and_body(self):
        kb = _make_kernel_body()
        sig = _make_signature_base()
        body = Module("body")
        body.add(TextBlock("s_nop 0\n"))
        kb.addSignature(sig)
        kb.addBody(body)
        self.assertIs(kb.signature, sig)
        self.assertIs(kb.body, body)

    def test_body_rw_attribute(self):
        kb = _make_kernel_body()
        body = Module("body")
        kb.body = body
        self.assertIs(kb.body, body)


class TestKernelBodySetGprs(_SignatureKernelSetup, unittest.TestCase):
    def test_set_gprs_updates_fields_and_signature(self):
        kb = _make_kernel_body()
        kb.addSignature(_make_signature_base())
        kb.setGprs(48, 4, 20)
        self.assertEqual(kb.totalVgprs, 48)
        self.assertEqual(kb.totalAgprs, 4)
        self.assertEqual(kb.totalSgprs, 20)
        self.assertEqual(kb.getNextFreeVgpr(), 48)
        self.assertEqual(kb.getNextFreeSgpr(), 20)

    def test_get_next_free_without_signature_returns_zero(self):
        kb = _make_kernel_body()
        self.assertEqual(kb.getNextFreeVgpr(), 0)
        self.assertEqual(kb.getNextFreeSgpr(), 0)


class TestKernelBodyToString(_SignatureKernelSetup, unittest.TestCase):
    def test_emits_banner_signature_and_body(self):
        kb = _make_kernel_body()
        kb.addSignature(_make_signature_base(kernelName="my_k"))
        body = Module("body")
        body.add(TextBlock("s_nop 0\n"))
        kb.addBody(body)
        s = str(kb)
        self.assertIn("Begin Kernel", s)
        self.assertIn('.amdgcn_target "amdgcn-amd-amdhsa--gfx1250"', s)
        self.assertIn("s_nop 0", s)

    def test_signature_only_still_requires_body(self):
        kb = _make_kernel_body()
        kb.addSignature(_make_signature_base())
        with self.assertRaises(RuntimeError):
            kb.toString()


class TestKernelBodyCheckResourcesPattern(_SignatureKernelSetup, unittest.TestCase):
    def test_body_add_after_set_gprs(self):
        """Mirror ``KernelWriterAssembly.checkResources`` overflow patch."""
        kb = _make_kernel_body()
        kb.addSignature(_make_signature_base())
        kb.addBody(Module("body"))
        kb.setGprs(totalVgprs=32, totalAgprs=0, totalSgprs=16)
        kb.body.add(TextBlock("/* overflow patch */\n"))
        s = str(kb)
        self.assertIn("/* overflow patch */", s)
        self.assertIn(".amdhsa_next_free_vgpr 32 // vgprs", s)


class TestKernelBodyCopyRejected(unittest.TestCase):
    def test_deepcopy_raises(self):
        kb = _make_kernel_body()
        with self.assertRaises(RuntimeError):
            copy.deepcopy(kb)

    def test_pickle_raises(self):
        kb = _make_kernel_body()
        with self.assertRaises(RuntimeError):
            pickle.dumps(kb)


class TestKernelBodyModuleIntegration(_SignatureKernelSetup, unittest.TestCase):
    def test_kernel_body_counted_in_module_tree(self):
        outer = Module("outer")
        kb = _make_kernel_body()
        kb.addSignature(_make_signature_base())
        kb.addBody(Module("inner"))
        outer.add(kb)
        self.assertEqual(outer.countType(Item), 2)
        self.assertEqual(outer.countType(KernelBody), 1)


if __name__ == "__main__":
    unittest.main()
