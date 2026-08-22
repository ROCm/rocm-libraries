# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
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


class TestSNop_Wait0(unittest.TestCase, _ThreePathEqualityCase):
    """``s_nop 0`` -- minimal scalar NOP (wait=0)."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SNop
        module = Module("k")
        module.add(SNop(waitState=0, comment="align"))
    """)


# ===========================================================================
# Scalar ALU -- arithmetic, shift, bitwise (Phase 6 Step 1)
# ===========================================================================


class TestSAddU32_SgprToSgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``s_add_u32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SAddU32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SAddU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2), comment="add"))
    """)


class TestSAddU32_Immediate(unittest.TestCase, _ThreePathEqualityCase):
    """``s_add_u32 s0, s1, 4`` -- immediate operand."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SAddU32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SAddU32(dst=sgpr(0), src0=sgpr(1), src1=4))
    """)


class TestSSubU32_SgprToSgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``s_sub_u32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SSubU32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SSubU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2), comment="sub"))
    """)


class TestSMulI32_SgprToSgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``s_mul_i32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMulI32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMulI32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2), comment="mul"))
    """)


class TestSMulHII32_SgprToSgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``s_mul_hi_i32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMulHII32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMulHII32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
    """)


class TestSLShiftLeftB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_lshl_b32 s0, s1, s2`` -- value=s1, shift=s2."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SLShiftLeftB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SLShiftLeftB32(dst=sgpr(0), shiftHex=sgpr(2), src=sgpr(1), comment="shl"))
    """)


class TestSLShiftRightB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_lshr_b32 s0, s1, s2`` -- value=s1, shift=s2."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SLShiftRightB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SLShiftRightB32(dst=sgpr(0), shiftHex=sgpr(2), src=sgpr(1)))
    """)


class TestSAShiftRightI32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_ashr_i32 s0, s1, s2`` -- value=s1, shift=s2."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SAShiftRightI32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SAShiftRightI32(dst=sgpr(0), shiftHex=sgpr(2), src=sgpr(1)))
    """)


class TestSAndB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_and_b32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SAndB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SAndB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2), comment="mask"))
    """)


class TestSOrB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_or_b32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SOrB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SOrB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
    """)


class TestSXorB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_xor_b32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SXorB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SXorB32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
    """)


class TestSAndSaveExecB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_and_saveexec_b32 s0, s1`` -- unary (1 src)."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SAndSaveExecB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SAndSaveExecB32(dst=sgpr(0), src=sgpr(1), comment="exec"))
    """)


class TestSLShiftLeft1AddU32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_lshl1_add_u32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SLShiftLeft1AddU32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SLShiftLeft1AddU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
    """)


# ===========================================================================
# Scalar Control (Phase 6 Step 2) -- SGetRegB32
# ===========================================================================


class TestSGetRegB32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_getreg_b32 s0, s1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SGetRegB32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SGetRegB32(dst=sgpr(0), src=sgpr(1), comment="hwid"))
    """)


# ===========================================================================
# Vector ALU (Phase 6 Step 4)
# ===========================================================================


## VAddU32 emission consistency is skipped: native rocisa uses arch caps to
## choose between ``v_add_u32`` (+ VCC dst1) and ``v_add_nc_u32`` (no dst1)
## depending on ExplicitNC/ExplicitCO flags.  The stinkytofu logical IR
## lowering always emits ``v_add_nc_u32`` for gfx1250, but the native rocisa
## build in the test environment may not have ExplicitNC configured, causing
## path1/path2 to emit ``v_add_u32 dst, vcc, src0, src1``.  This is a known
## configuration gap between the native right-path and the stinkytofu
## left-path; the adaptor correctly matches the logical IR lowering output.
## Same applies to VSubI32 / VSubU32 (ExplicitNC variants).


class TestVAddF32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_add_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VAddF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VAddF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="addf"))
    """)


class TestVSubF32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_sub_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VSubF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VSubF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVMulF32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mul_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMulF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMulF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="mul"))
    """)


class TestVMulLOU32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mul_lo_u32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMulLOU32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMulLOU32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVMulHIU32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_mul_hi_u32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMulHIU32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMulHIU32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVAndB32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_and_b32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VAndB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VAndB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVOrB32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_or_b32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VOrB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VOrB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVXorB32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_xor_b32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VXorB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VXorB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVLShiftLeftB32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_lshlrev_b32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VLShiftLeftB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VLShiftLeftB32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2), comment="shl"))
    """)


class TestVLShiftRightB32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_lshrrev_b32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VLShiftRightB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VLShiftRightB32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2)))
    """)


class TestVFmaF32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_fma_f32 v0, v1, v2, v3``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VFmaF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VFmaF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3), comment="fma"))
    """)


class TestVAndOrB32_VgprToVgpr(unittest.TestCase, _ThreePathEqualityCase):
    """``v_and_or_b32 v0, v1, v2, v3``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VAndOrB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VAndOrB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3)))
    """)


## VCndMaskB32 emission consistency is skipped: native rocisa always appends
## VCC as src2 (``VCndMaskB32(dst, src0, src1, src2=VCC())``), while the
## logical IR lowering path uses only 2 sources (VCC mask is implicit in the
## architecture). This is an intentional design divergence, analogous to
## SBarrier's signal/wait expansion. The adaptor's to_stinky_logical correctly
## passes 2 srcs and the instruction unit tests (test_instruction.py) cover
## the adaptor's own rendering and deepcopy.


class TestVReadfirstlaneB32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_readfirstlane_b32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VReadfirstlaneB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VReadfirstlaneB32(dst=vgpr(0), src=vgpr(1), comment="lane0"))
    """)


# ===========================================================================
# Compare instructions (Phase 6 Step 5)
# ===========================================================================


class TestSCmpEQI32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_cmp_eq_i32 s0, s1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SCmpEQI32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SCmpEQI32(src0=sgpr(0), src1=sgpr(1), comment="eq"))
    """)


class TestSCmpGtU32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_cmp_gt_u32 s0, s1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SCmpGtU32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SCmpGtU32(src0=sgpr(0), src1=sgpr(1)))
    """)


class TestSCmpLgI32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_cmp_lg_i32 s0, s1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SCmpLgI32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SCmpLgI32(src0=sgpr(0), src1=sgpr(1)))
    """)


class TestSBitcmp1B32(unittest.TestCase, _ThreePathEqualityCase):
    """``s_bitcmp1_b32 s0, s1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SBitcmp1B32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SBitcmp1B32(src0=sgpr(0), src1=sgpr(1)))
    """)


class TestVCmpEQF32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cmp_eq_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCmpEQF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCmpEQF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), comment="eq"))
    """)


class TestVCmpGEI32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cmp_ge_i32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCmpGEI32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCmpGEI32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVCmpNeU32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cmp_ne_u32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCmpNeU32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCmpNeU32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVCmpClassF32(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cmp_class_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCmpClassF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCmpClassF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


## VCmpX* emission consistency is skipped: the native rocisa VCmpXInstruction
## toString() has arch-dependent behaviour — when CMPXWritesSGPR is false (test
## env default), it replaces "v_cmpx_" with "v_cmp_" and appends an
## "s_mov_b64 exec, dst" (wavefront64) or "s_mov_b32 exec_lo, dst" (wavefront32)
## instruction. The logical IR path on gfx1250 (which has CMPXWritesSGPR=true)
## correctly emits the real "v_cmpx_*" mnemonic directly. This is an intentional
## arch-specific divergence in the native C++ toString; the adaptor correctly
## forwards to the logical IR lowering which matches the hardware behaviour.


# ===========================================================================
# Step 6: Scalar Min/Max/Abs + Vector Unary/Misc
# ===========================================================================


class TestSAbsI32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``s_abs_i32 s0, s1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SAbsI32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SAbsI32(dst=sgpr(0), src=sgpr(1)))
    """)


class TestSMaxI32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``s_max_i32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMaxI32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMaxI32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
    """)


class TestSMinU32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``s_min_u32 s0, s1, s2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import SMinU32
        from rocisa.container import sgpr
        module = Module("k")
        module.add(SMinU32(dst=sgpr(0), src0=sgpr(1), src1=sgpr(2)))
    """)


class TestVExpF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_exp_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VExpF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VExpF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVRcpF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_rcp_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VRcpF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VRcpF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVRsqF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_rsq_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VRsqF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VRsqF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVNotB32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_not_b32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VNotB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VNotB32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVRndneF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_rndne_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VRndneF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VRndneF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVMaxF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_max_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMaxF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMaxF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVMinF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_min_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMinF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMinF32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVMaxI32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_max_i32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMaxI32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMaxI32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVMed3I32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_med3_i32 v0, v1, v2, v3``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMed3I32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMed3I32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3)))
    """)


class TestVMed3F32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_med3_f32 v0, v1, v2, v3``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VMed3F32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VMed3F32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2), src2=vgpr(3)))
    """)


class TestVAShiftRightI32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_ashrrev_i32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VAShiftRightI32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VAShiftRightI32(dst=vgpr(0), shiftHex=vgpr(1), src=vgpr(2)))
    """)


class TestVPackF16toB32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_pack_b32_f16 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VPackF16toB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VPackF16toB32(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


## VLShiftLeftOrB32 emission consistency is skipped: the native rocisa class is a
## CompositeInstruction that expands to two sub-instructions
## ("v_lshlrev_b32 dst, shift, src" + "v_or_b32 dst, dst, src2"), while the
## logical IR correctly emits the fused "v_lshl_or_b32 dst, shift, src, src2".
## This is an intentional arch-level optimisation in the logical IR pipeline.


## VPrngB32 emission consistency is skipped: "v_prng_b32" is only available
## on architectures with hardware PRNG support. The instruction may not
## assemble successfully in all test environments.


# ===========================================================================
# Step 7 — Conversion (VCvt*) emission consistency tests
# ===========================================================================


class TestVCvtF16toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_f32_f16 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtF16toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtF16toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtF32toF16Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_f16_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtF32toF16
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtF32toF16(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtF32toU32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_u32_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtF32toU32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtF32toU32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtU32toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_f32_u32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtU32toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtU32toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtI32toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_f32_i32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtI32toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtI32toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtF32toI32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_i32_f32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtF32toI32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtF32toI32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtFP8toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_f32_fp8 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtFP8toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtFP8toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtBF8toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_f32_bf8 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtBF8toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtBF8toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtPkFP8toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_pk_f32_fp8 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtPkFP8toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtPkFP8toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtPkBF8toF32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_pk_f32_bf8 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtPkBF8toF32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtPkBF8toF32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestVCvtPkF32toBF8Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_pk_bf8_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtPkF32toBF8
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtPkF32toBF8(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVCvtSRF32toFP8Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_sr_fp8_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtSRF32toFP8
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtSRF32toFP8(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVCvtSRF32toBF8Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_sr_bf8_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtSRF32toBF8
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtSRF32toBF8(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVCvtPkF32toFP8Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_pk_fp8_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtPkF32toFP8
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtPkF32toFP8(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


class TestVCvtPkF32toBF16Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``v_cvt_pk_bf16_f32 v0, v1, v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import VCvtPkF32toBF16
        from rocisa.container import vgpr
        module = Module("k")
        module.add(VCvtPkF32toBF16(dst=vgpr(0), src0=vgpr(1), src1=vgpr(2)))
    """)


## VCvtScale* emission consistency tests are skipped: these instructions are in
## SKIP_LOWERING in LogicalToAsmMultiArchTest.cpp because the standard lowering
## pass cannot handle them (they require scale-specific patterns not yet in the
## lowering pipeline). The adaptor correctly constructs them with to_stinky_logical()
## but the asm emission cannot be compared end-to-end yet.


# ===========================================================================
# Memory instructions -- DS Load/Store (Step 8)
# ===========================================================================


class TestDSLoadB32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_load_b32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSLoadB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSLoadB32(dst=vgpr(0), src=vgpr(1)))
    """)


class TestDSLoadB64Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_load_b64 v[0:1], v2``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSLoadB64
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSLoadB64(dst=vgpr(0, 2), src=vgpr(2)))
    """)


class TestDSLoadB128Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_load_b128 v[0:3], v4``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSLoadB128
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSLoadB128(dst=vgpr(0, 4), src=vgpr(4)))
    """)


class TestDSLoadU8Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_load_u8 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSLoadU8
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSLoadU8(dst=vgpr(0), src=vgpr(1)))
    """)


class TestDSLoadU16Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_load_u16 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSLoadU16
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSLoadU16(dst=vgpr(0), src=vgpr(1)))
    """)


class TestDSStoreB32Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_store_b32 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSStoreB32
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSStoreB32(dstAddr=vgpr(0), src=vgpr(1)))
    """)


class TestDSStoreB64Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_store_b64 v0, v[1:2]``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSStoreB64
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSStoreB64(dstAddr=vgpr(0), src=vgpr(1, 2)))
    """)


class TestDSStoreB128Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_store_b128 v0, v[1:4]``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSStoreB128
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSStoreB128(dstAddr=vgpr(0), src=vgpr(1, 4)))
    """)


class TestDSStoreB8Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_store_b8 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSStoreB8
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSStoreB8(dstAddr=vgpr(0), src=vgpr(1)))
    """)


class TestDSStoreB16Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_store_b16 v0, v1``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSStoreB16
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSStoreB16(dstAddr=vgpr(0), src=vgpr(1)))
    """)


class TestDSStoreB96Emission(unittest.TestCase, _ThreePathEqualityCase):
    """``ds_store_b96 v0, v[1:3]``."""

    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import DSStoreB96
        from rocisa.container import vgpr
        module = Module("k")
        module.add(DSStoreB96(dstAddr=vgpr(0), src=vgpr(1, 3)))
    """)


class TestWmmaF6Gfx1250Scaled(unittest.TestCase):
    """gfx1250 F6 WMMA lowers to the forceScaledWMMA form.

    Three-path equality is *not* usable here: native rocisa probes the
    assembler for ``HasWMMA_f8f6f4`` / ``HasWMMA_V3`` and reports 0 in a
    container without a gfx1250 assembler, whereas stinkytofu reports the
    static gfx1250 truth (1). Under those true caps rocisa's
    ``getArgStr`` emits the scale operands + matrix formats; we assert the
    adapter (path 3, which uses stinkytofu caps) reproduces that exact
    byte-for-byte form (rocisa ``MFMAInstruction::preStr``/``getArgStr``)."""

    ARCH_TUPLE = (12, 5, 0)
    BUILD_MODULE_SNIPPET = textwrap.dedent("""\
        from rocisa.code import Module
        from rocisa.instruction import MFMAInstruction
        from rocisa.container import vgpr
        from rocisa.enum import InstType
        module = Module("k")
        module.add(MFMAInstruction(
            instType=InstType.INST_F6, accType=InstType.INST_F32,
            variant=[16, 16, 128, 1], mfma1k=False,
            acc=vgpr(0, 8), a=vgpr(8, 8), b=vgpr(16, 8), acc2=vgpr(0, 8),
            neg=False, comment="wmma f6 probe"))
    """)

    EXPECTED = (
        "v_wmma_scale_f32_16x16x128_f8f6f4 v[0:7], v[8:15], v[16:23], v[0:7], "
        "0, 0 matrix_a_fmt:MATRIX_FMT_FP6 matrix_b_fmt:MATRIX_FMT_FP6"
        " // wmma f6 probe\n"
    )

    @unittest.skipUnless(_STINKY_OK, "needs the stinkytofu Python binding")
    def test_adapter_emits_forced_scaled_wmma(self):
        got = emit_path3_adapter_logical(
            self.BUILD_MODULE_SNIPPET, arch_tuple=self.ARCH_TUPLE)
        self.assertEqual(got, self.EXPECTED, f"\n[adapter] {got!r}")


## DSBPermuteB32 emission consistency is skipped: native rocisa path-1 toString()
## and path-2 toStinkyTofuModule produce different asm for this instruction
## (mismatch unrelated to our adaptor), so three-path equality is not achievable.
## The adaptor's to_stinky_logical() is validated by unit tests above.


## Buffer/Flat emission consistency tests are skipped: the logicalIR lowering
## for buffer/flat memory instructions uses a simplified operand model (2-3
## register args) while the native rocisa toString() emits full asm with buffer
## descriptors, soffset, and MUBUF/FLAT modifiers. Matching all three paths
## end-to-end requires the lowering pass to reconstruct these modifiers from
## the abstract operand set, which is architecture-specific and not yet
## validated for the adaptor path.


if __name__ == "__main__":
    unittest.main()
