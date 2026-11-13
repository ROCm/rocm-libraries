# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""WMMA operand VGPR bank ISA characterization (gfx1151, gfx1201).

Directly inspects the emitted assembly (CPU-only, no GPU) and checks the VGPR
bank (bank = vgpr_index % 4) of each ``v_wmma`` operand (A, B, C):

* gfx1151 (WMMA V1): A, B and C are on three distinct banks. Same-bank operands
  serialize the WMMA read on gfx11 and cost +2 cycles per instruction; the
  register allocator places A on a bank distinct from C (B is already forced
  odd).
* gfx1201 (WMMA V2): A stays on C's bank. Pins the non-gfx11 placement so the
  gfx11-only rule is not widened by accident.

Configs: data/test_data/_designed/<arch>/wmma_bank_distinct.yaml
  BBS (bf16-in / bf16-out / f32-compute), TN, MatrixInstruction [16,16,16,1]
  MIWaveTile 6x2 (MT96x128x64).

The emitted operands are symbolic (``v[vgprValuA_X0_I0+8+0:...]``) resolved via
``.set`` directives (e.g. ``.set vgprValuA_X0_I0_BASE, vgprBase+0`` and
``.set vgprBase, 102``), so we build the full ``.set`` symbol table, resolve
each operand's start register to a concrete index, and take ``% 4``.
"""

import os
import re

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCHS = ["gfx1151", "gfx1201"]


def _config(arch):
    return os.path.join(
        os.path.dirname(__file__),
        "data",
        "test_data",
        "_designed",
        arch,
        "wmma_bank_distinct.yaml",
    )


_NUM_BANKS = 4  # gfx11/gfx12 VGPR file: bank = vgpr_index % 4

_SET_RE = re.compile(r"^\s*\.set\s+(\w+)\s*,\s*(.+?)\s*$")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
# First register index of each ``v[<expr>:<expr>]`` operand.
_VREG_RE = re.compile(r"\bv\[([^:\]]+):[^\]]*\]")


def _build_symbol_table(src):
    """Map every ``.set NAME, EXPR`` to its (first) EXPR string."""
    table = {}
    for line in src.splitlines():
        m = _SET_RE.match(line)
        if m:
            table.setdefault(m.group(1), m.group(2))
    return table


def _resolve(expr, table, cache, stack=()):
    """Evaluate an assembler expression of ints and ``.set`` symbols to an int."""

    def repl(m):
        return str(_resolve_name(m.group(0), table, cache, stack))

    numeric = _IDENT_RE.sub(repl, expr)
    # numeric now contains only digits and + - * ( ) whitespace.
    return int(eval(numeric, {"__builtins__": {}}, {}))  # noqa: S307 (test-only, sanitized)


def _resolve_name(name, table, cache, stack):
    if name.isdigit():
        return int(name)
    if name in cache:
        return cache[name]
    if name not in table:
        raise KeyError(f"symbol {name!r} has no .set definition")
    if name in stack:
        raise ValueError(f"cyclic .set definition through {name!r}")
    val = _resolve(table[name], table, cache, stack + (name,))
    cache[name] = val
    return val


def _operand_role(expr):
    """Classify a WMMA operand expression as 'A', 'B', or 'C' by its Valu base."""
    if "ValuA" in expr:
        return "A"
    if "ValuB" in expr:
        return "B"
    if "ValuC" in expr:
        return "C"
    return None


def _wmma_operand_banks(src):
    """Yield (line, {role: bank}) for every v_wmma instruction in ``src``."""
    table = _build_symbol_table(src)
    cache = {}
    for line in src.splitlines():
        if "v_wmma" not in line.lower():
            continue
        # Drop trailing "// ..." comment so we only parse real operands.
        code = line.split("//", 1)[0]
        banks = {}
        for expr in _VREG_RE.findall(code):
            role = _operand_role(expr)
            if role is None:
                continue
            idx = _resolve(expr, table, cache)
            banks[role] = idx % _NUM_BANKS
        yield line.strip(), banks


@pytest.mark.parametrize("arch", _ARCHS)
def test_emits_assembly(arch):
    """WMMA config emits real assembly, all err==0."""
    results = emit_kernels_from_config(_config(arch), limit=8, arch=arch)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )


def _all_wmma_banks(arch):
    """Return [(kernel, line, {role: bank})] for every v_wmma emitted for ``arch``."""
    results = emit_kernels_from_config(_config(arch), limit=8, arch=arch)
    assert len(results) >= 1

    out = []
    for base, src, _err in results:
        for line, banks in _wmma_operand_banks(src):
            # Each v_wmma must expose all three operands (A, B, C).
            assert set(banks) == {"A", "B", "C"}, (
                f"{base}: could not resolve all A/B/C operands for: {line} "
                f"(got roles {sorted(banks)})"
            )
            out.append((base, line, banks))

    assert out, "Expected at least one v_wmma instruction across kernels"
    return out


def test_gfx1151_wmma_operands_on_distinct_banks():
    """gfx1151: every v_wmma reads A, B and C from three distinct VGPR banks."""
    for base, line, banks in _all_wmma_banks("gfx1151"):
        assert len(set(banks.values())) == 3, (
            f"{base}: v_wmma operands share a VGPR bank "
            f"(A={banks['A']}, B={banks['B']}, C={banks['C']}) for: {line}"
        )


def test_gfx1201_wmma_a_on_c_bank():
    """gfx1201: A keeps C's VGPR bank; the gfx11 distinct-bank rule must not apply."""
    for base, line, banks in _all_wmma_banks("gfx1201"):
        assert banks["A"] == banks["C"], (
            f"{base}: expected A on C's VGPR bank "
            f"(A={banks['A']}, B={banks['B']}, C={banks['C']}) for: {line}"
        )
