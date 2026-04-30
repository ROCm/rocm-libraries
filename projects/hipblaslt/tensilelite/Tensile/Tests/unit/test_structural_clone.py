################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for structural_clone.

structural_clone is the over-strong-deepcopy alternative used at the
default-side capture call site: it recursively clones Module wrappers
(so SIA3's destructive popFirstItem doesn't corrupt the source) while
preserving leaf instruction identity (so id-based categorization
survives across the call boundary).
"""

from rocisa.code import Module, TextBlock, Label, ValueIf, ValueEndif

from Tensile.Components.ScheduleCapture import structural_clone


def test_leaves_shared():
    """Leaf instruction id() is preserved across the clone."""
    m = Module("leaves")
    children = [TextBlock(f"// l{i}") for i in range(4)]
    for c in children:
        m.add(c)
    cloned = structural_clone(m)
    for orig, got in zip(children, cloned.items()):
        assert got is orig
        assert id(got) == id(orig)


def test_modules_distinct():
    """Every Module in the cloned tree is a distinct Python object."""
    inner = Module("inner")
    inner.add(TextBlock("// x"))
    outer = Module("outer")
    outer.add(inner)

    cloned = structural_clone(outer)
    assert cloned is not outer
    assert cloned.items()[0] is not inner


def test_pop_isolation():
    """popFirstItem on the clone leaves the original's child list intact."""
    m = Module("pop")
    a = TextBlock("// a")
    b = TextBlock("// b")
    m.add(a); m.add(b)

    cloned = structural_clone(m)
    cloned.popFirstItem()
    assert m.itemsSize() == 2
    assert cloned.itemsSize() == 1


def test_named_modules_preserved():
    """findNamedItem still works on the clone."""
    m = Module("outer")
    inner = Module("named_inner")
    inner.add(TextBlock("// inside"))
    m.add(inner)

    cloned = structural_clone(m)
    found = cloned.findNamedItem("named_inner")
    assert found is not None
    assert found.name == "named_inner"


def test_handles_non_module_non_instruction_children():
    """Non-Module children (Label, ValueIf, TextBlock) are shared by id()."""
    m = Module("mixed")
    lbl = Label("L_test", "")
    vi = ValueIf("flag")
    vei = ValueEndif()
    tb = TextBlock("// t")
    for c in (lbl, vi, vei, tb):
        m.add(c)

    cloned = structural_clone(m)
    for orig, got in zip((lbl, vi, vei, tb), cloned.items()):
        assert got is orig


def test_deeply_nested():
    """3-level nesting: every Module distinct, every leaf shared."""
    leaf = TextBlock("// deep")
    l3 = Module("l3"); l3.add(leaf)
    l2 = Module("l2"); l2.add(l3)
    l1 = Module("l1"); l1.add(l2)

    c1 = structural_clone(l1)
    c2 = c1.items()[0]
    c3 = c2.items()[0]
    c_leaf = c3.items()[0]

    assert c1 is not l1
    assert c2 is not l2
    assert c3 is not l3
    assert c_leaf is leaf


def test_unnamed_module_round_trip():
    """An unnamed Module clones to an unnamed clone (truthiness fix)."""
    m = Module("")
    m.add(TextBlock("// x"))
    cloned = structural_clone(m)
    assert cloned.name == ""


def test_setnoopt_preserved():
    """isNoOpt state is carried across the clone (Step 0 audit item)."""
    m_default = Module("default")
    m_noopt = Module("noopt")
    m_noopt.setNoOpt(True)

    c_default = structural_clone(m_default)
    c_noopt = structural_clone(m_noopt)

    assert c_default.isNoOpt() is False
    assert c_noopt.isNoOpt() is True


def test_label_sharing_safe():
    """Both old and new contain the same Label python object.

    Confirms Label-sharing is intentional. The Label-correctness invariant
    (capture-side iterCode is discarded, never emitted) is verified
    separately at the call site, not here.
    """
    lbl = Label("L_shared", "comment")
    m = Module("with_label"); m.add(lbl)
    cloned = structural_clone(m)
    assert cloned.items()[0] is lbl


def test_valueif_sharing_safe():
    """ValueIf/ValueEndif are pure markers and safely shared."""
    vi = ValueIf("guard")
    ve = ValueEndif()
    m = Module("guarded"); m.add(vi); m.add(ve)
    cloned = structural_clone(m)
    assert cloned.items()[0] is vi
    assert cloned.items()[1] is ve


def test_non_module_input_returned_as_is():
    """Calling structural_clone on a leaf returns it unchanged (recursion base)."""
    leaf = TextBlock("// solo")
    assert structural_clone(leaf) is leaf
