# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""StaggerU global-read increment must be widened unsigned (CPU-only).

``GlobalReadIncs<tc>+unrollIdx`` is one SGPR holding ``stride * DepthU * bpeGR``,
a byte count that is non-negative by construction. Two sites in
``calculateStagger`` widen it to 64 bits before it reaches the global-read SRD:

  1. the stagger byte offset, ``StaggerUIter * GlobalReadIncs``
  2. the unroll-loop span, ``LoopCounter * GlobalReadIncs``, which becomes ``WrapU``

Widening either of them *signed* is wrong. Once ``stride * DepthU * bpeGR``
reaches ``2^31`` the register's bit 31 is set, a signed widen sign-extends it to a
negative 64-bit value, and ``incrementSrd`` then walks the global-read base
*backwards* by roughly 2 GB. On gfx1201 that is ROCM-31230: an fp32 NN GEMM at
``lda = 67108864, DepthU = 8`` page-faults, and at larger ``lda`` it returns a
silently wrong result instead, because the displaced base still lands in mapped
memory.

The defect needs a non-zero ``StaggerUIter`` to reach the multiply, which is why
it is kernel-specific rather than shape-specific: the loop-count clamp in
``calculateStagger`` zeroes the stagger for some ``StaggerUStrideShift`` values,
and ``StaggerUMapping`` chooses which workgroup dimension feeds the mask, so a
problem with a single workgroup in that dimension also zeroes it.

Asserted on fully generated assembly rather than on the Python source, because
the signedness is only observable in the emitted instruction: ``SMulInt64to32``
selects ``s_mul_hi_i32`` when its ``sign`` argument is true and ``s_mul_hi_u32``
when it is false, and both spellings carry the same comment through to the
listing.

CPU-only: no GPU required. gfx1201 is the arch ROCM-31230 was reported on.
"""

import os
import re

import pytest

import codegen_harness as _ch
import config_harness as _cfgh

pytestmark = pytest.mark.unit

_ARCH = "gfx1201"

# Reuse the existing designed gfx1201 config rather than adding a fixture. It
# already emits kernels whose prologue reaches both multiplies, which is all this
# test needs; the defect is in stagger codegen and is independent of data type.
_CONFIG = os.path.join(
    os.path.dirname(__file__), "data", "test_data", "_designed", _ARCH, "rich_wmma.yaml"
)

_LIMIT = 2

# The two comments calculateStagger attaches to the widening multiplies. Matching
# on the comment rather than the register keeps this readable if register
# allocation shifts.
_STAGGER_COMMENT = "stagger byte offset"
_WRAPU_COMMENT = "Number of bytes accessed by the unroll loop"

_SIGNED_HI = re.compile(r"^\s*s_mul_hi_i32\b")
_UNSIGNED_HI = re.compile(r"^\s*s_mul_hi_u32\b")

# calculateStagger opens each tensor's block with this comment. incrementSrd emits
# "gra SRD += inc(upper)" from nine call sites, most of them the ordinary
# global-read increment that every kernel emits, so an unanchored search for that
# text says nothing about the stagger path. Everything below is scoped to a block.
_STAGGER_BLOCK_OPEN = re.compile(
    r"^\s*/\* addr \+= \(StaggerUIter\) \* GlobalReadIncs(?P<tc>\w+?)\+\d+ \*/\s*$"
)


def _stagger_blocks(src):
    """Yield ``(tc, body_lines)`` for each calculateStagger block in ``src``.

    A block runs from its opening comment to the next comment line, which is
    where the generator starts emitting something else.
    """
    lines = src.splitlines()
    for i, line in enumerate(lines):
        match = _STAGGER_BLOCK_OPEN.match(line)
        if not match:
            continue
        body = []
        for following in lines[i + 1 :]:
            if following.strip().startswith("/*"):
                break
            body.append(following)
        yield match.group("tc"), body


def _emit_asm(config_path, arch, limit):
    """Emit kernel assembly for ``config_path``, returning [(base, src)]."""
    from Tensile.Common.Types import DebugConfig
    from Tensile.KernelWriterAssembly import KernelWriterAssembly
    from Tensile.SolutionStructs.Naming import getKernelFileBase
    from Tensile.TensileCreateLibrary.Run import (
        generateKernelObjectsFromSolutions,
        processKernelSource,
    )

    assembler, isaInfoMap = _cfgh._toolchain_for(arch)
    out = []
    with _cfgh._isolated_globals_with_isa(isaInfoMap):
        solutions = _cfgh._solutions_from_config_unguarded(
            config_path, assembler, isaInfoMap, limit_solutions=limit
        )
        kernels = generateKernelObjectsFromSolutions(solutions)
        kernels = sorted(kernels, key=lambda k: getKernelFileBase(False, k))[:limit]
        kwa = KernelWriterAssembly(assembler, DebugConfig())
        for kernel in kernels:
            ri = _ch._init_rocisa_for(kernel)
            base = _ch._prepare_kernel(kernel, False)
            res = processKernelSource(
                kwa, ri.getData(), ri.getOutputOptions(), False, kernel
            )
            assert res.err == 0, f"{base} failed to emit: err={res.err}"
            src = res.src
            if isinstance(src, (bytes, bytearray)):
                src = src.decode(errors="replace")
            out.append((base, src or ""))
    return out


@pytest.fixture(scope="module")
def emitted():
    asm = _emit_asm(_CONFIG, _ARCH, _LIMIT)
    assert asm, "no kernels emitted"
    return asm


def _widening_lines(src, comment):
    """The high-word multiply lines carrying ``comment``."""
    return [
        line
        for line in src.splitlines()
        if comment in line and ("s_mul_hi_i32" in line or "s_mul_hi_u32" in line)
    ]


@pytest.mark.parametrize(
    "comment",
    [_STAGGER_COMMENT, _WRAPU_COMMENT],
    ids=["stagger_byte_offset", "wrapu_unroll_span"],
)
def test_globalreadincs_widened_unsigned(emitted, comment):
    """Neither widening of GlobalReadIncs may sign-extend it."""
    checked = 0
    for base, src in emitted:
        if "sgprStaggerUIter" not in src:
            # StaggerU compiled out for this variant; nothing to widen.
            continue
        lines = _widening_lines(src, comment)
        assert lines, (
            f"{base}: found no high-word multiply commented {comment!r}. Either the "
            "comment text changed or the widening was removed; re-point this test at "
            "the current site rather than deleting it."
        )
        for line in lines:
            assert not _SIGNED_HI.match(line), (
                f"{base}: GlobalReadIncs widened with a signed high multiply, which "
                f"sign-extends it once stride*DepthU*bpeGR reaches 2^31 (ROCM-31230):"
                f"\n    {line.strip()}"
            )
            assert _UNSIGNED_HI.match(line), (
                f"{base}: unexpected widening instruction:\n    {line.strip()}"
            )
        checked += 1
    assert checked, (
        "no emitted kernel reached the stagger path, so this test proved nothing. "
        "Pick a config whose solutions keep StaggerU enabled."
    )


def test_stagger_offset_feeds_the_srd_increment(emitted):
    """Pin that the widened stagger offset is what moves the global-read SRD.

    Without this, the tests above could keep passing while the stagger offset
    stopped reaching the SRD, which would make its signedness irrelevant and hide
    a real behavior change.

    The check ties the two registers the stagger multiply writes to the two the
    SRD increment reads, inside one block. Merely finding an SRD increment is not
    enough: the ordinary global-read increment emits the identical instruction and
    comment in every kernel, so that would hold even with the stagger increment
    deleted.
    """
    checked = 0
    for base, src in emitted:
        if "sgprStaggerUIter" not in src:
            continue
        for tc, body in _stagger_blocks(src):
            low = high = None
            for line in body:
                # The sparse arm reuses the phrase for its own tensor; it has its
                # own increment and is not what this test pins.
                if "stagger byte offset" not in line or "of metadata" in line:
                    continue
                hi_match = re.match(r"\s*s_mul_hi_u32\s+([^,]+),", line)
                if hi_match:
                    high = hi_match.group(1)
                low_match = re.match(r"\s*s_mul_i32\s+([^,]+),", line)
                if low_match:
                    low = low_match.group(1)
            assert low and high, (
                f"{base}: the {tc} stagger block no longer computes a 64-bit "
                "stagger byte offset"
            )

            block = "\n".join(body)
            for reg, half, mnemonic in ((low, "0", "s_add_u32"), (high, "1", "s_addc_u32")):
                assert re.search(
                    rf"{mnemonic}\s+s\[sgprSrd{tc}\+{half}\],\s*"
                    rf"s\[sgprSrd{tc}\+{half}\],\s*{re.escape(reg)}\b",
                    block,
                ), (
                    f"{base}: the {tc} stagger offset in {reg} is no longer added "
                    f"into sgprSrd{tc}+{half}, so its signedness cannot affect the "
                    "global-read base"
                )
            checked += 1

    assert checked, (
        "no emitted kernel reached the stagger path, so this test proved nothing. "
        "Pick a config whose solutions keep StaggerU enabled."
    )
