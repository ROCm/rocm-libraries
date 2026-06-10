#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Smoke parity test for the C++ rocisa module-builder foundation.

The C++ ``ModuleBuilder`` (``tensile_writer.subtile.module_builder``) constructs
genuine ``rocisa`` ``Module`` / instruction objects by driving the rocisa Python
API. This test asserts that what the C++ builder produces renders to *exactly*
the same assembly string as the equivalent objects built directly with rocisa in
Python — i.e. the construction the Subtile emit path performs today
(``InstructionEmitter.emit_sync`` / ``emit_wait_lr`` and a ``DSLoadB32`` leaf).

This is the foundation slice's acceptance check: "C++-built rocisa output equals
the existing Python module output." No production path is switched to the builder
here; later slices move the emit loops onto it.

Runs only when both rocisa and the compiled extension are importable (both are
hard dependencies of the subtile emit path); otherwise it skips. Pure-string
test (rocisa pinned to gfx950); no GPU runtime / hip dependency.

PR creation for this slice is human-only.
"""

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# Both the ISA layer (rocisa) and the compiled builder layer must exist.
pytest.importorskip("rocisa")
pytest.importorskip("tensile_writer.subtile.module_builder")

from tensile_writer.subtile.module_builder import ModuleBuilder


def _init_rocisa_gfx950():
    """Pin rocisa to gfx950 (wave64) for deterministic string emission.

    Mirrors the helper in test_subtileEmitMfmaRocisa.py so this pure-string test
    has no GPU-runtime (hip) import dependency.
    """
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    _init_rocisa_gfx950()


@pytest.fixture(scope="module")
def mb():
    # Instantiating ModuleBuilder triggers the rocisa import inside C++.
    return ModuleBuilder()


def test_builder_constructs_real_rocisa_module(mb):
    """The builder returns a genuine rocisa Code.Module instance."""
    from rocisa.code import Module
    cpp_mod = mb.module("smoke")
    assert isinstance(cpp_mod, Module), type(cpp_mod)


def test_barrier_matches_python(mb):
    """ModuleBuilder.barrier() == InstructionEmitter.emit_sync() construction."""
    from rocisa.instruction import SBarrier
    cpp = str(mb.barrier())
    py = str(SBarrier(comment="Barrier"))
    assert cpp == py, f"barrier mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
    assert cpp.strip(), "barrier emitted empty asm"


def test_wait_lr_matches_python(mb):
    """ModuleBuilder.wait_lr() == InstructionEmitter.emit_wait_lr() construction."""
    from rocisa.instruction import SWaitCnt
    cpp = str(mb.wait_lr())
    py = str(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                      comment="Wait for LR to complete"))
    assert cpp == py, f"wait_lr mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"


def test_single_item_module_matches_python(mb):
    """A C++-built Module wrapping a barrier renders identically to a Python
    Module built the same way (the single-instruction emit-leaf shape)."""
    from rocisa.code import Module
    from rocisa.instruction import SBarrier

    cpp_mod = mb.single_item_module(mb.barrier(), "leaf")

    py_mod = Module("leaf")
    py_mod.add(SBarrier(comment="Barrier"))

    assert str(cpp_mod) == str(py_mod)


def test_generic_instruction_hook_and_containers_match_python(mb):
    """The generic instruction(...) hook plus vgpr/sgpr/ds_modifiers build a
    DSLoadB32 identical to the rocisa Python construction used in
    InstructionEmitter.emit_lr (scale ds_read leaf)."""
    from rocisa.container import vgpr, DSModifiers
    from rocisa.instruction import DSLoadB32

    cpp = str(mb.instruction(
        "DSLoadB32",
        dst=mb.vgpr(5),
        src=mb.vgpr(9),
        ds=mb.ds_modifiers(offset=64),
        comment="scale ds_read leaf"))

    py = str(DSLoadB32(
        dst=vgpr(5),
        src=vgpr(9),
        ds=DSModifiers(offset=64),
        comment="scale ds_read leaf"))

    assert cpp == py, f"DSLoadB32 mismatch:\nC++ : {cpp!r}\nPy  : {py!r}"
    assert "ds_load" in cpp.lower() or "ds_read" in cpp.lower()


def test_ds_modifiers_dual_address_matches_python(mb):
    """ds_modifiers(offset, na=2) builds a dual-address DSModifiers identical to
    the rocisa Python construction the subtile LR/GR emit paths need."""
    from rocisa.container import DSModifiers

    assert str(mb.ds_modifiers(offset=64)) == str(DSModifiers(na=1, offset=64))
    assert str(mb.ds_modifiers(offset=64, na=2)) == str(DSModifiers(na=2, offset=64))


def test_flatitems_parity(mb):
    """flatitems() on a C++-built nested module yields the same rendered
    sequence as the equivalent Python-built module."""
    from rocisa.code import Module
    from rocisa.instruction import SBarrier, SWaitCnt

    inner = mb.module("inner")
    mb.add(inner, mb.wait_lr())
    outer = mb.module("outer")
    mb.add_comment(outer, "outer comment")
    mb.add(outer, inner)
    mb.add(outer, mb.barrier())

    py_inner = Module("inner")
    py_inner.add(SWaitCnt(dscnt=0, vlcnt=-1, vscnt=-1,
                          comment="Wait for LR to complete"))
    py_outer = Module("outer")
    py_outer.addComment("outer comment")
    py_outer.add(py_inner)
    py_outer.add(SBarrier(comment="Barrier"))

    cpp_flat = [str(i) for i in mb.flatitems(outer)]
    py_flat = [str(i) for i in py_outer.flatitems()]
    assert cpp_flat == py_flat
    assert str(outer) == str(py_outer)
