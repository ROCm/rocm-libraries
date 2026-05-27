################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
"""Standalone tests for ``rocisa_stinkytofu_adaptor.code`` (Module / TextBlock).

Run from any working directory:

    python3 projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_code.py

Or with pytest:

    pytest projects/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor/tests/test_code.py

These tests pin behaviour that the rest of the adapter (and KernelWriter)
depend on -- the IR-tree semantics ``add`` / ``replaceItem`` /
``popFirstNItems`` / ``flatitems`` / parent rebinding must match
``rocisa::Module`` byte-for-byte so an adapter swap does not silently
re-order, drop, or re-parent items in the emitted asm.

The ``to_stinky_asm`` test block is gated on a built ``stinkytofu``
binding and exercises the full left-path lowering pipeline (Module tree
-> ``LogicalModule`` -> ``StinkyAsmModule.emitAssembly``) using a fake
instruction that fabricates a single ``_stinkytofu.VMovB32`` -- decoupling
the Module-layer tests from Step-3 (instruction shims) progress.
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

from rocisa_stinkytofu_adaptor.code import (  # noqa: E402
    Module,
    TextBlock,
)


# ===========================================================================
# TextBlock -- raw text leaf.
# ===========================================================================


class TestTextBlockConstruction(unittest.TestCase):
    def test_default_text_empty(self):
        tb = TextBlock()
        self.assertEqual(tb.text, "")
        # name == "" matches rocisa's ``Item("")`` ctor; KernelWriter's
        # removeItemsByName("foo") relies on TextBlocks NOT picking up
        # "TextBlock" as a name (else mass-delete by class name would
        # silently nuke all comments).
        self.assertEqual(tb.name, "")
        self.assertIsNone(tb.parent)

    def test_text_arg_stored(self):
        tb = TextBlock("hello\n")
        self.assertEqual(tb.text, "hello\n")

    def test_str_and_toString_match(self):
        tb = TextBlock("// foo\n")
        self.assertEqual(str(tb), "// foo\n")
        self.assertEqual(tb.toString(), "// foo\n")

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

    def test_prettyPrint_returns_string(self):
        # Defensive: Module.prettyPrint joins children's prettyPrint output;
        # a non-str return would explode the join.
        self.assertIsInstance(TextBlock("x").prettyPrint(), str)


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
        # Block comments must be a multi-line banner so KernelWriter
        # section dividers stay visually distinct.
        m = Module()
        m.addComment0("section")
        text = str(m)
        self.assertIn("/*", text)
        self.assertIn("section", text)
        self.assertGreaterEqual(text.count("\n"), 3)

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
        self.assertTrue(text.endswith("\n\n"), text)
        self.assertIn("trailing", text)


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

    def test_nested_structure(self):
        outer = Module("Outer")
        inner = Module("Inner")
        inner.add(TextBlock("x"))
        outer.add(inner)
        out = outer.prettyPrint()
        self.assertIn('Module "Outer"', out)
        self.assertIn('Module "Inner"', out)
        self.assertIn("TextBlock", out)

    def test_indent_propagates(self):
        # Inner items should be indented more than the outer header.
        outer = Module("Outer")
        outer.add(TextBlock("x"))
        out = outer.prettyPrint()
        lines = [ln for ln in out.split("\n") if ln]
        self.assertTrue(any("|--" in ln for ln in lines))


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
# logicalIR handoff -- _collect_logical_insts walk semantics.
# ===========================================================================
#
# Decoupled from stinkytofu: a fake instruction with ``to_stinky_logical()``
# is enough to assert the walk picks it up. The TestToStinkyAsm block
# below covers the end-to-end binding call.


class _FakeLogicalInst:
    """Looks like a Step-3 instruction shim to ``_collect_logical_insts``."""

    def __init__(self, payload):
        self.parent = None
        self.payload = payload
        self.lowered = False

    def to_stinky_logical(self):
        # Returns whatever the binding consumes. The collector does not
        # inspect the value -- only forwards it -- so we can use a sentinel.
        self.lowered = True
        return self.payload

    def __str__(self):
        return ""


class TestCollectLogicalInsts(unittest.TestCase):
    def test_empty_module(self):
        self.assertEqual(Module()._collect_logical_insts(), [])

    def test_skips_textblock(self):
        m = Module()
        m.add(TextBlock("ignored"))
        self.assertEqual(m._collect_logical_insts(), [])

    def test_picks_up_logical_leaf(self):
        m = Module()
        fake = _FakeLogicalInst(payload="P1")
        m.add(fake)
        self.assertEqual(m._collect_logical_insts(), ["P1"])
        self.assertTrue(fake.lowered)

    def test_in_order_traversal_flat(self):
        m = Module()
        a, b, c = (_FakeLogicalInst("A"), _FakeLogicalInst("B"), _FakeLogicalInst("C"))
        m.add(a)
        m.add(b)
        m.add(c)
        self.assertEqual(m._collect_logical_insts(), ["A", "B", "C"])

    def test_in_order_traversal_with_nested_modules(self):
        # Depth-first, in-order: inner items appear at the point their
        # parent Module appears in the outer list.
        outer = Module()
        outer.add(_FakeLogicalInst("0"))
        inner = Module()
        inner.add(_FakeLogicalInst("1"))
        inner.add(_FakeLogicalInst("2"))
        outer.add(inner)
        outer.add(_FakeLogicalInst("3"))
        self.assertEqual(outer._collect_logical_insts(), ["0", "1", "2", "3"])

    def test_skips_dummy_instruction_classes(self):
        # As of Step 3 most instruction shims (e.g. ``SBarrier``) remain
        # dummies whose ``__getattr__`` returns a no-op that yields
        # ``None``. The collector must skip them rather than smuggling
        # ``None`` into the logical IR module. We pick ``SBarrier`` here
        # specifically because ``VMovB32`` was promoted to a real shim
        # in Step 3; using a still-dummy class keeps this regression
        # guard meaningful as more shims light up.
        from rocisa_stinkytofu_adaptor.instruction import SBarrier  # noqa: WPS433
        m = Module()
        m.add(SBarrier())
        self.assertEqual(m._collect_logical_insts(), [])

    def test_textblocks_and_logical_mixed(self):
        m = Module()
        m.add(TextBlock("// header\n"))
        m.add(_FakeLogicalInst("INST"))
        m.add(TextBlock("// footer\n"))
        # Comments are silently dropped; KernelWriter's text is meant for
        # the rocisa right-path, not logicalIR left-path.
        self.assertEqual(m._collect_logical_insts(), ["INST"])


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

    def test_textblock_items_are_silently_ignored(self):
        # Comments don't have a logical counterpart; they must not break
        # the pipeline (or smuggle themselves into the emitted asm).
        m = Module("kWithComments")
        m.add(TextBlock("// a header comment\n"))
        m.add(self._make_fake_vmovb32())
        m.add(TextBlock("// a footer comment\n"))
        asm = m.to_stinky_asm([12, 5, 0])
        text = asm.emitAssembly()
        self.assertIn("v_mov_b32", text)
        self.assertNotIn("a header comment", text)
        self.assertNotIn("a footer comment", text)

    def test_arch_accepts_sequence_not_just_list(self):
        # Tuples / arrays are common in KernelWriter (kernel["ISA"] is
        # often a tuple). Accept anything sequence-like.
        m = Module()
        m.add(self._make_fake_vmovb32())
        asm = m.to_stinky_asm((12, 5, 0))
        self.assertIn("v_mov_b32", asm.emitAssembly())


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


if __name__ == "__main__":
    unittest.main()
