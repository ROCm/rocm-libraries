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
"""Three-path emission consistency tests.

A rocisa-shape ``Module`` built from KernelWriter-style code can be turned
into gfx1250 assembly text via three distinct production code paths -- this
module verifies they all produce byte-identical kernel body for the same
input.

The three paths (drawn from the IR architecture diagram):

    (1) default rocisa  ->  ``str(module)``                      [right path]
        KernelWriter --(rocisa native C++)--> toString --> asm

    (2) default rocisa  ->  ``toStinkyTofuModule`` -> ``emitAssembly``
        KernelWriter --(rocisa native C++)--> stinkytofu asm IR --> emitAssembly --> asm

    (3) stinkytofu adapter -> ``module.to_stinky_asm`` -> ``emitAssembly``  [left path]
        KernelWriter --(rocisa_stinkytofu_adaptor)--> stinkytofu python_binding
                     --> logical IR --> lowering pass --> stinkytofu asm IR
                     --> emitAssembly --> asm

Why all three should match exactly:
    - (1) ↔ (3) is the actual Phase 3 acceptance criterion ("the new
      stinkytofu logical-IR pipeline is observationally equivalent to the
      native rocisa right path for any single instruction we promote").
    - (1) ↔ (2) catches regressions in the rocisa->stinky asm-IR bridge.
    - (2) ↔ (3) catches regressions on either side of the asm-IR layer.

Adding coverage for a new instruction shim:
    Subclass ``_ThreePathEqualityCase`` and set ``BUILD_MODULE_SNIPPET`` to
    a snippet that constructs a variable named ``module`` (using only
    ``rocisa.*`` imports -- they resolve to the right backend in each
    subprocess). Three equality tests are generated automatically.

Running:
    Use the ``test.sh`` wrapper in this directory -- it auto-detects the
    built ``stinkytofu`` / ``rocisa`` binding directory and exports
    ``PYTHONPATH`` for both the parent runner and the per-path subprocesses:

        ./test.sh test_emission_consistency        # this file
        ./test.sh test_emission_consistency -v
        ./test.sh                                  # discover all tests

    If you'd rather invoke ``python3`` directly, set ``PYTHONPATH`` to
    the build's ``tensilelite/rocisa`` directory first, e.g.::

        PYTHONPATH=/.../tensilelite/<build>/tensilelite/rocisa \\
            python3 test_emission_consistency.py -v

Notes:
    - Each test method spawns 2 or 3 fresh Python subprocesses (one per
      path). They inherit ``PYTHONPATH`` from the parent runner. Backend
      selection is via the ``ROCISA_BACKEND`` env var only.
    - We always call ``rocIsa.getInstance().init(arch, "")`` then
      ``setKernel(arch, 64)`` in the preamble. ``init`` alone registers ISA
      metadata (caps); ``setKernel`` installs the per-thread ``KernelInfo``
      whose ``isaVersion`` ``ReadWriteInstruction::typeConvert()`` uses for
      gfx11+ mnemonic suffixes (e.g. ``s_load_b64`` vs legacy ``s_load_dwordx2``).
      Path (2) still needs caps for ``getAsmCaps()``; paths (1) and (3) need
      the active ISA for byte-identical ``toString`` / lowering.
    - ``s_set_vgpr_msb`` (gfx1250 VGPR-MSB workaround) is *not* triggered
      under the current gfx1250 caps snapshot for VGPR indices < 256.
      Once we promote tests that touch VGPRs >= 256 OR enable the
      ``HasVgprMSB`` cap, the adapter shim's ``CommonInstruction.__str__``
      will need to grow ``setMsb`` to keep path (1) == (3); the left-path
      side is already covered by ``InsertVgprMsbPass`` in stinkytofu.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import unittest


# ===========================================================================
# Environment probes
# ===========================================================================
#
# We gate path (2) and path (3) on the stinkytofu binding being importable
# in the current process. Path (1) only needs native rocisa (i.e.
# _rocisa.so), which is always built alongside the adapter anyway.
#
# When invoked via the ``test.sh`` wrapper in this directory, PYTHONPATH
# is auto-set to a built ``tensilelite/<build>/tensilelite/rocisa`` and
# the gate passes. Otherwise, the user must set PYTHONPATH themselves --
# all cases will skip otherwise.

try:
    import stinkytofu as _stinky  # noqa: F401
    _STINKY_OK = True
except ImportError:
    _STINKY_OK = False


# ===========================================================================
# Subprocess runner
# ===========================================================================


# Sentinels framing the asm payload in each subprocess's stdout. Anything
# stinkytofu / rocisa prints during import or init (e.g. the
# ``IntrinsicRegistry: Loaded N intrinsics`` banner) lands outside the
# sentinels and gets dropped by the extractor.
_BEGIN = "<<<EMIT_BEGIN_E5A9E2>>>"
_END = "<<<EMIT_END_E5A9E2>>>"


def _run_in_subproc(script: str, *, backend, timeout: float = 30) -> str:
    """Run @p script in a fresh Python process and return the emitted asm.

    @p backend == None   -> default rocisa (no ROCISA_BACKEND env var)
    @p backend == "stinkytofu"  -> our adapter (via env var)

    The script is wrapped so that the asm payload is written between
    sentinels; banner prints from third-party imports are stripped.
    PYTHONPATH and other env vars are inherited from the parent process so
    that ``import rocisa`` / ``import stinkytofu`` resolve to the built
    .so files that the parent test runner could already see.
    """
    env = os.environ.copy()
    env.pop("ROCISA_BACKEND", None)
    if backend is not None:
        env["ROCISA_BACKEND"] = backend
    # PYTHONPATH is inherited from the parent runner (the test.sh wrapper
    # in this directory sets it; manual ``python3`` invocations need to
    # set it themselves).
    proc = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"emission subprocess (backend={backend!r}) failed "
            f"with exit {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- script ---\n{script}"
        )
    start = proc.stdout.find(_BEGIN)
    end = proc.stdout.find(_END)
    if start < 0 or end < 0 or end < start:
        raise AssertionError(
            f"emission subprocess (backend={backend!r}) returned 0 but "
            f"sentinels were not found in stdout.\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}\n"
            f"--- script ---\n{script}"
        )
    return proc.stdout[start + len(_BEGIN):end]


# Shared preamble: ``init`` loads caps for ``arch_tuple``; ``setKernel`` binds
# that ISA to this thread so ``Item::kernel().isaVersion`` (used by native
# ``toString`` type suffixes) matches ``arch_tuple`` — same as real KernelWriter
# flows (see ``KernelWriter`` / unit tests calling ``setKernel`` after ``init``).
_INIT_PREAMBLE = textwrap.dedent("""\
    import rocisa
    _ri = rocisa.rocIsa.getInstance()
    _ri.init({arch_tuple}, "", False)
    _ri.setKernel({arch_tuple}, 64)
""")


# ===========================================================================
# Path emitters
# ===========================================================================
#
# Each takes a snippet that constructs a variable named ``module`` and
# returns the asm text emitted by that specific code path. Snippets must
# import via ``rocisa.*`` (not ``rocisa_stinkytofu_adaptor.*``) so the
# backend dispatch in ``rocisa/__init__.py`` can swap implementations.


def _strip_kernel_descriptor(asm: str, kernel_name: str) -> str:
    """Strip path-2's kernel header / metadata wrapper.

    ``toStinkyTofuModule`` produces a full kernel object: ``.amdgcn_target``
    directive, ``.amdhsa_kernel`` block, ``.amdgpu_metadata`` YAML, then
    the kernel symbol label, then the instruction body. We extract only
    the instruction body to compare against paths (1) and (3), which emit
    just the body.

    Strategy: split on the kernel symbol label (``\\n<kernel_name>:\\n``).
    The last occurrence is the body marker; everything before it is
    descriptor / metadata.
    """
    marker = "\n" + kernel_name + ":\n"
    idx = asm.rfind(marker)
    if idx < 0:
        raise AssertionError(
            f"path-2 output is missing kernel-symbol marker {marker!r}; "
            f"got:\n{asm[:500]}..."
        )
    return asm[idx + len(marker):]


_EMIT_TAIL = textwrap.dedent(f"""\

    import sys
    sys.stdout.write({_BEGIN!r})
    sys.stdout.write(_payload)
    sys.stdout.write({_END!r})
    sys.stdout.flush()
""")


def emit_path1_rocisa_tostring(build_snippet: str, *,
                               arch_tuple=(12, 5, 0)) -> str:
    """Path 1 -- default rocisa native + ``str(module)``."""
    script = (
        _INIT_PREAMBLE.format(arch_tuple=arch_tuple)
        + build_snippet
        + "\n_payload = str(module)\n"
        + _EMIT_TAIL
    )
    return _run_in_subproc(script, backend=None)


def emit_path2_rocisa_stinkyasm(build_snippet: str, *,
                                arch_tuple=(12, 5, 0),
                                kernel_name: str = "k") -> str:
    """Path 2 -- default rocisa native + ``toStinkyTofuModule`` + ``emitAssembly``.

    Wraps the module in a minimal kernel signature, calls the C++ bridge
    that converts rocisa Module to stinkytofu asm IR, emits assembly,
    then strips the kernel descriptor wrapper so only the body is
    compared.
    """
    script = (
        _INIT_PREAMBLE.format(arch_tuple=arch_tuple)
        + build_snippet
        + textwrap.dedent(f"""\

            import rocisa
            from rocisa.code import SignatureBase
            sig = SignatureBase(
                kernelName="{kernel_name}",
                kernArgsVersion=1,
                codeObjectVersion="4",
                groupSegmentSize=0,
                sgprWorkGroup=(1, 1, 0),
                vgprWorkItem=0,
                flatWorkGroupSize=64,
                numSgprPreload=0,
            )
            module.setParent()
            st = rocisa.toStinkyTofuModule(
                module, {arch_tuple}, "{kernel_name}",
                signature=sig, options={{"OptLevel": 0}},
            )
            _payload = st.emitAssembly()
        """)
        + _EMIT_TAIL
    )
    raw = _run_in_subproc(script, backend=None)
    return _strip_kernel_descriptor(raw, kernel_name)


def emit_path3_adapter_logical(build_snippet: str, *,
                               arch_tuple=(12, 5, 0)) -> str:
    """Path 3 -- stinkytofu adapter + logical IR pipeline + ``emitAssembly``.

    ``ROCISA_BACKEND=stinkytofu`` swaps ``rocisa.*`` for our adapter, so
    the same build snippet now constructs adapter ``Module`` /
    ``VMovB32`` / ``vgpr`` objects. ``to_stinky_asm(list(arch))`` runs
    the C++ ``CompositeInstructionLoweringPass`` + ``ToStinkyAsmPass``
    via ``lower_logical_module``.
    """
    script = (
        _INIT_PREAMBLE.format(arch_tuple=arch_tuple)
        + build_snippet
        + textwrap.dedent(f"""\

            _asm_mod = module.to_stinky_asm(list({arch_tuple}))
            _payload = _asm_mod.emitAssembly()
        """)
        + _EMIT_TAIL
    )
    return _run_in_subproc(script, backend="stinkytofu")


# ===========================================================================
# Mixin: subclass + set BUILD_MODULE_SNIPPET = automatic 3 equality tests
# ===========================================================================


class _ThreePathEqualityCase:
    """Mixin generating three byte-equality tests from a single snippet.

    Subclasses must set ``BUILD_MODULE_SNIPPET`` to a Python snippet that:
      * Imports only via ``rocisa.*`` (so backend dispatch can rewire it).
      * Builds a top-level variable named ``module``.
      * Does NOT print or terminate the process.

    Optional overrides:
      * ``ARCH_TUPLE``: target arch (default gfx1250 = ``(12, 5, 0)``).
      * ``KERNEL_NAME``: kernel name used for path-2 SignatureBase
        (default ``"k"``).
    """

    BUILD_MODULE_SNIPPET: str = ""  # override
    ARCH_TUPLE: tuple = (12, 5, 0)
    KERNEL_NAME: str = "k"

    @unittest.skipUnless(_STINKY_OK,
                         "path-3 needs the stinkytofu Python binding")
    def test_path1_equals_path3(self):
        """Native ``toString`` == adapter logical-IR pipeline emit.

        This is the Phase-3 acceptance criterion: the new stinkytofu
        left-path is observationally equivalent to the rocisa right-path.
        """
        a = emit_path1_rocisa_tostring(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE)
        b = emit_path3_adapter_logical(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE)
        self.assertEqual(
            a, b,
            f"\n[path-1 toString  ] {a!r}"
            f"\n[path-3 adapter   ] {b!r}",
        )

    @unittest.skipUnless(_STINKY_OK,
                         "path-2 needs stinkytofu compiled into rocisa")
    def test_path1_equals_path2(self):
        """Native ``toString`` == native ``toStinkyTofuModule`` body.

        Catches regressions in the rocisa->stinkytofu asm-IR bridge.
        """
        a = emit_path1_rocisa_tostring(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE)
        b = emit_path2_rocisa_stinkyasm(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE,
            kernel_name=self.KERNEL_NAME)
        self.assertEqual(
            a, b,
            f"\n[path-1 toString  ] {a!r}"
            f"\n[path-2 stinky-asm] {b!r}",
        )

    @unittest.skipUnless(_STINKY_OK,
                         "paths 2 and 3 need the stinkytofu binding")
    def test_path2_equals_path3(self):
        """Native ``toStinkyTofuModule`` body == adapter logical-IR emit.

        Both ultimately hit ``emitAssembly`` on a stinkytofu asm-IR; this
        verifies the rocisa->asm bridge and the adapter->logical->asm
        pipeline land in the same asm-IR state for the same input.
        """
        a = emit_path2_rocisa_stinkyasm(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE,
            kernel_name=self.KERNEL_NAME)
        b = emit_path3_adapter_logical(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE)
        self.assertEqual(
            a, b,
            f"\n[path-2 stinky-asm] {a!r}"
            f"\n[path-3 adapter   ] {b!r}",
        )


# ===========================================================================
# VMovB32 scenarios
# ===========================================================================
#
# Each class below is one input scenario. Three equality tests are
# auto-generated. Add a new instruction = add a new class.


class TestVMovB32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mov_b32 v0, v1`` -- the most basic VGPR-to-VGPR move."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMovB32(dst=vgpr(0), src=vgpr(1), comment="probe"))
    """)


class TestVMovB32_VgprToVgprNoComment(unittest.TestCase,
                                       _ThreePathEqualityCase):
    """Empty-comment branch in ``formatStr`` (no padding, no '//')."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMovB32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVMovB32_HexImmediate(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mov_b32 v0, 0x0`` -- string-immediate src (KernelWriter's
    typical ``hex(N)`` pattern)."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMovB32(dst=vgpr(0), src="0x0", comment="init zero"))
    """)


class TestVMovB32_IntImmediate(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mov_b32 v0, 42`` -- int-immediate src."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMovB32(dst=vgpr(0), src=42, comment="int imm"))
    """)


class TestVMovB32_MultipleSequential(unittest.TestCase,
                                      _ThreePathEqualityCase):
    """Three sequential ``v_mov_b32`` -- verifies Module child order
    survives all three paths."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        for i in range(3):
            module.add(VMovB32(dst=vgpr(i), src=vgpr(i + 10),
                               comment=f"move {i}"))
    """)


class TestVMovB32_NestedModules(unittest.TestCase, _ThreePathEqualityCase):
    """Inner Module inside outer Module -- depth-first traversal must
    preserve emit order across all three paths."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        inner = Module("sub")
        inner.add(VMovB32(dst=vgpr(0), src=vgpr(1), comment="inner"))
        module.add(inner)
        module.add(VMovB32(dst=vgpr(2), src=vgpr(3), comment="outer"))
    """)


class TestVMovB32_NamedRegister(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mov_b32 vgprValuA, vgprValuB`` -- symbolic-name VGPR via
    ``vgpr("Name")``. Verifies the RegName round-trip survives the
    rocisa-shape -> logical-IR -> asm-IR pipeline."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMovB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMovB32(dst=vgpr("ValuA"), src=vgpr("ValuB"),
                           comment="symbolic"))
    """)


# ===========================================================================
# SMovB32 / SMovB64 scenarios (Phase A scalar moves)
# ===========================================================================


class TestSMovB32_SgprToSgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``s_mov_b32 s0, s1`` -- basic SGPR-to-SGPR move."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMovB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMovB32(dst=sgpr(0), src=sgpr(1), comment="probe"))
    """)


class TestSMovB32_SgprToSgprNoComment(unittest.TestCase, _ThreePathEqualityCase):
    """Empty-comment branch for ``s_mov_b32`` (no ``//`` suffix)."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMovB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMovB32(dst=sgpr(0), src=sgpr(1)))
    """)


class TestSMovB32_HexImmediate(unittest.TestCase, _ThreePathEqualityCase):
    """``s_mov_b32 s0, 0x0`` -- string-immediate src."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMovB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMovB32(dst=sgpr(0), src="0x0", comment="init zero"))
    """)


class TestSMovB64_PairToPair(unittest.TestCase, _ThreePathEqualityCase):
    """``s_mov_b64 s[0:1], s[4:5]`` -- 64-bit pair operands."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMovB64
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMovB64(dst=sgpr(0, 2), src=sgpr(4, 2), comment="wide"))
    """)


class TestSLoadB32_BaseImmOffset(unittest.TestCase, _ThreePathEqualityCase):
    """``s_load_b32 s0, s[2:3], 0`` -- minimal SMEM load (no modifiers)."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SLoadB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SLoadB32(dst=sgpr(0), base=sgpr(2, 2), soffset=0,
                            comment="smem"))
    """)


class TestSLoadB64_WideDst(unittest.TestCase, _ThreePathEqualityCase):
    """``s_load_b64 s[0:1], s[4:5], 0``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SLoadB64
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SLoadB64(dst=sgpr(0, 2), base=sgpr(4, 2), soffset=0,
                            comment="wide load"))
    """)


if __name__ == "__main__":
    unittest.main()
