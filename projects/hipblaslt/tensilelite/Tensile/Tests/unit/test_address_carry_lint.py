# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The address-carry lint must report a 64-bit address update that drops the carry, and stay
quiet on correct carry chains and on registers that are only reused for other arithmetic."""

import functools
import os

import pytest

from Tensile.Utilities.address_carry_lint import lint

pytestmark = pytest.mark.unit

_DESIGNED = os.path.join(
    os.path.dirname(__file__), "characterization", "_codegen", "data", "test_data", "_designed"
)

# Generated kernels from designed configs that cover the address arithmetic most at risk: batch
# and stride address setup, Stream-K and GSU workspace addressing, and fp8 and MX scale loads. The
# gfx1250 configs run TDM kernels, whose descriptor addresses the lint does not see; it checks
# their other address updates. gfx90a, gfx942 and gfx950 have no native 64-bit add, so every
# address update is a 32-bit carry chain there.
_GENERATED = [
    ("gfx90a", "rich_gemm.yaml"),
    ("gfx942", "asmaddr_initstrides.yaml"),
    ("gfx942", "s00_loadbatchedaddress_non_stridedba.yaml"),
    ("gfx942", "streamk.yaml"),
    ("gfx942", "streamk_fixup_tree.yaml"),
    ("gfx942", "gsu.yaml"),
    ("gfx942", "fp8_gr_conv.yaml"),
    ("gfx950", "mx_fp8_scale_swizzle.yaml"),
    ("gfx950", "mx_bias_act_gsu.yaml"),
    ("gfx950", "subtile.yaml"),
    ("gfx1250", "streamk.yaml"),
    ("gfx1250", "streamk_tdm_prefetchgl2.yaml"),
]


@functools.lru_cache(maxsize=None)
def _assembler_supports(arch):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    isa = gfxToIsa(arch)
    return bool(makeIsaInfoMap([isa], validateToolchain("amdclang++"))[isa].asmCaps["SupportedISA"])


@pytest.mark.parametrize("arch,config", _GENERATED, ids=lambda x: str(x).replace(".yaml", ""))
def test_generated_kernels_carry_every_address_update(arch, config):
    from config_harness import emit_kernels_from_config

    if not _assembler_supports(arch):
        pytest.skip(f"amdclang++ in this environment does not support {arch}")

    results = emit_kernels_from_config(
        os.path.join(_DESIGNED, arch, config), limit=4, arch=arch, canonical=False
    )
    assert results, f"{arch}/{config} emitted no kernels"
    for base, src, err in results:
        assert err == 0, f"{base} failed to emit"
        findings = lint(src)
        assert not findings, f"{base}:\n" + "\n".join(str(f) for f in findings)


def _reasons(asm):
    return [f.reason for f in lint(asm)]


def test_scalar_carry_chain_on_a_descriptor_base_is_clean():
    asm = """
    s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA]
    s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0
    buffer_load_dwordx4 v[0:3], v4, s[sgprSrdA:sgprSrdA+3], 0 offen
    """
    assert _reasons(asm) == []


def test_scalar_carry_that_never_reaches_the_high_dword_is_reported():
    # The ROCM-32046 shape: the low dword is advanced, the carry is dropped.
    asm = """
    s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA]
    s_mov_b32 s[sgprTmp], 0
    buffer_load_dwordx4 v[0:3], v4, s[sgprSrdA:sgprSrdA+3], 0 offen
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1
    assert "s[sgprSrdA+0]" in reasons[0] and "s[sgprSrdA+1]" in reasons[0]


def test_add_without_carry_on_an_address_low_dword_is_reported():
    asm = """
    s_add_i32 s8, s8, 64
    s_load_dwordx2 s[10:11], s[8:9], 0x0
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1 and "s_add_i32" in reasons[0] and "s9" in reasons[0]


def test_vector_global_address_needs_a_carry_chain():
    good = """
    v_add_co_u32 v4, vcc, v4, v6
    v_addc_co_u32 v5, vcc, v5, 0, vcc
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    bad = """
    v_add_u32 v4, v4, v6
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    assert _reasons(good) == []
    reasons = _reasons(bad)
    assert len(reasons) == 1 and "v4" in reasons[0]


def test_a_register_reused_for_other_arithmetic_is_not_reported():
    # s20 is an address elsewhere, but this sum is overwritten before any address use.
    asm = """
    s_load_dwordx2 s[30:31], s[20:21], 0x0
    s_sub_u32 s20, s2, s20
    s_add_u32 s20, s20, s3
    s_mov_b64 s[20:21], s[0:1]
    s_load_dwordx2 s[32:33], s[20:21], 0x8
    """
    assert _reasons(asm) == []


@pytest.mark.parametrize(
    "op",
    ["v_sub_u32", "v_subrev_u32", "v_sub_i32", "v_subrev_nc_u32", "v_add_nc_i32", "v_sub_nc_i32"],
)
def test_vector_subtract_without_carry_on_an_address_is_reported(op):
    asm = f"""
    {op} v4, v4, v6
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1 and op in reasons[0] and "v4" in reasons[0]


# A returning atomic is dst, vaddr, data[, saddr]; one that does not return is vaddr, data[, saddr].
@pytest.mark.parametrize("marker", ["glc", "sc0", "th:TH_ATOMIC_RETURN"])
def test_a_returning_atomic_address_is_its_second_operand(marker):
    bad = f"""
    v_add_u32 v2, v2, v6
    global_atomic_add_u64 v[0:1], v[2:3], v[4:5], off {marker}
    """
    reasons = _reasons(bad)
    assert len(reasons) == 1 and "v2" in reasons[0]
    data_only = f"""
    v_add_u32 v4, v4, v6
    v_add_co_u32 v2, vcc, v2, v6
    v_addc_co_u32 v3, vcc, v3, 0, vcc
    global_atomic_add_u64 v[0:1], v[2:3], v[4:5], off {marker}
    """
    assert _reasons(data_only) == []


def test_a_non_returning_atomic_address_is_its_first_operand():
    asm = """
    v_add_u32 v4, v4, v6
    global_atomic_add_u64 v[2:3], v[4:5], off
    """
    assert _reasons(asm) == []
    asm = """
    v_add_u32 v2, v2, v6
    global_atomic_add_u64 v[2:3], v[4:5], off
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1 and "v2" in reasons[0]


def test_data_registers_of_a_buffer_load_are_not_addresses():
    asm = """
    buffer_load_dwordx4 v[34:37], v5, s[40:43], 0 offen
    v_sub_u32 v34, s12, v34
    buffer_load_dwordx4 v[34:37], v5, s[40:43], 0 offen
    """
    assert _reasons(asm) == []


def test_a_long_branch_offset_is_not_an_address():
    # The gfx950 long-branch sequence: s64 holds a constant branch offset, and the next use of
    # s[64:65] as an address is in another block, past the jump.
    asm = """
    s_getpc_b64 s[62:63]
    s_add_i32 s64, 0x424f4, 4
    s_add_u32 s62, s62, s64
    s_addc_u32 s63, s63, 0
    s_setpc_b64 s[62:63]
    s_cmp_eq_u64 s[64:65], 0
    s_load_dword s8, s[64:65], 0x0
    """
    assert _reasons(asm) == []


def test_an_address_update_before_a_jump_is_still_reported():
    asm = """
    s_add_i32 s8, s8, 64
    s_load_dwordx2 s[10:11], s[8:9], 0x0
    s_branch label_next
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1 and "s8" in reasons[0]


def test_a_scalar_carry_moved_far_by_the_scheduler_is_clean():
    # The gfx950 fp8 kernels: the carry-in comes 40 vector instructions after the carry-out, and
    # nothing in between writes SCC.
    filler = "\n".join("    v_perm_b32 v27, v55, v54, s78" for _ in range(40))
    asm = f"""
    s_add_u32 s68, s68, s76
{filler}
    s_addc_u32 s69, s69, 0
    buffer_load_dwordx4 v8, s[68:71], 0 offen lds
    """
    assert _reasons(asm) == []


def test_a_scalar_carry_lost_to_an_scc_write_is_reported():
    asm = """
    s_add_u32 s68, s68, s76
    s_cmp_eq_u32 s73, 0
    s_addc_u32 s69, s69, 0
    buffer_load_dwordx4 v8, s[68:71], 0 offen lds
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1 and "before SCC changes" in reasons[0]


def test_findings_keep_their_line_numbers_across_kernels():
    asm = "\n".join(
        [
            ".amdgpu_hsa_kernel first",
            "    s_load_dwordx2 s[10:11], s[4:5], 0x0",
            ".amdgpu_hsa_kernel second",
            "    s_nop 0",
            "    s_add_i32 s8, s8, 64",
            "    s_load_dwordx2 s[10:11], s[8:9], 0x0",
        ]
    )
    findings = lint(asm)
    assert [f.line for f in findings] == [5]


def test_tab_separated_instructions_are_parsed():
    asm = "\ts_add_i32\ts8, s8, 64\n\ts_load_dwordx2\ts[10:11], s[8:9], 0x0\n"
    reasons = _reasons(asm)
    assert len(reasons) == 1 and "s8" in reasons[0]


def test_a_vector_carry_overwritten_before_the_carry_in_is_reported():
    good = """
    v_add_co_u32 v4, vcc, v4, v6
    v_add_co_u32 v10, s[20:21], v10, v12
    v_addc_co_u32 v5, vcc, v5, 0, vcc
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    bad = """
    v_add_co_u32 v4, vcc, v4, v6
    v_add_co_u32 v10, vcc, v10, v12
    v_addc_co_u32 v5, vcc, v5, 0, vcc
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    assert _reasons(good) == []
    reasons = _reasons(bad)
    assert len(reasons) == 1 and "v4" in reasons[0]


# A carry-in add also sets a carry out, so on a low dword its carry must reach the high dword.
@pytest.mark.parametrize(
    "op,carry",
    [("v_add_co_ci_u32", "vcc_lo"), ("v_sub_co_ci_u32", "vcc_lo"), ("v_addc_co_u32", "vcc")],
)
def test_a_carry_in_add_on_a_low_dword_must_carry_on(op, carry):
    good = f"""
    {op} v4, {carry}, v4, v6, {carry}
    v_add_co_ci_u32 v5, {carry}, v5, 0, {carry}
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    bad = f"""
    {op} v4, {carry}, v4, v6, {carry}
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    assert _reasons(good) == []
    reasons = _reasons(bad)
    assert len(reasons) == 1 and op in reasons[0] and "v4" in reasons[0]


@pytest.mark.parametrize(
    "carry,partial",
    [
        ("s[20:21]", "s_mov_b32 s20, 0"),
        ("s[20:21]", "s_mov_b32 s21, 0"),
        ("vcc", "s_mov_b32 vcc_lo, 0"),
    ],
)
def test_a_partial_write_to_the_carry_register_is_reported(carry, partial):
    good = f"""
    v_add_co_u32 v4, {carry}, v4, v6
    s_mov_b32 s30, 0
    v_addc_co_u32 v5, {carry}, v5, 0, {carry}
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    bad = f"""
    v_add_co_u32 v4, {carry}, v4, v6
    {partial}
    v_addc_co_u32 v5, {carry}, v5, 0, {carry}
    global_load_dwordx2 v[0:1], v[4:5], off
    """
    assert _reasons(good) == []
    reasons = _reasons(bad)
    assert len(reasons) == 1 and "v4" in reasons[0]


_LOOP = """
label_LoopBeginL:
    buffer_load_dwordx4 v[0:3], v4, s[sgprSrdA:sgprSrdA+3], 0 offen
    s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA]
    {carry}
    s_sub_u32 s[sgprLoopCounterL], s[sgprLoopCounterL], 1
    s_cmp_eq_u32 s[sgprLoopCounterL], 0
    s_cbranch_scc1 label_LoopEndL
    s_branch label_LoopBeginL
label_LoopEndL:
    s_endpgm
"""


# A main-loop increment is next used as an address only at the top of the loop, through the
# back-edge.
def test_a_dropped_carry_reached_through_a_loop_back_edge_is_reported():
    assert _reasons(_LOOP.format(carry="s_addc_u32 s[sgprSrdA+1], s[sgprSrdA+1], 0")) == []
    reasons = _reasons(_LOOP.format(carry=""))
    assert len(reasons) == 1 and "sgprSrdA" in reasons[0]


def test_a_dropped_carry_used_long_after_is_reported():
    mfmas = "\n".join(["    v_mfma_f32_32x32x8_f16 v[0:15], v[16:17], v[18:19], v[0:15]"] * 450)
    asm = f"""
    s_add_u32 s[sgprSrdA+0], s[sgprSrdA+0], s[sgprGlobalReadIncsA]
{mfmas}
    buffer_load_dwordx4 v[0:3], v4, s[sgprSrdA:sgprSrdA+3], 0 offen
    """
    reasons = _reasons(asm)
    assert len(reasons) == 1 and "sgprSrdA" in reasons[0]


# The updated value reaches the descriptor through 32-bit copies.
def test_a_dropped_carry_copied_into_an_address_is_reported():
    asm = """
    s_add_u32 s[sgprAddrA+0], s[sgprAddrA+0], s[sgprInc]
    {carry}
    s_mov_b32 s[sgprSrdA+0], s[sgprAddrA+0]
    s_mov_b32 s[sgprSrdA+1], s[sgprAddrA+1]
    buffer_load_dwordx4 v[0:3], v4, s[sgprSrdA:sgprSrdA+3], 0 offen
    """
    assert _reasons(asm.format(carry="s_addc_u32 s[sgprAddrA+1], s[sgprAddrA+1], 0")) == []
    reasons = _reasons(asm.format(carry=""))
    assert len(reasons) == 1 and "sgprAddrA" in reasons[0]


# gfx12 names the scalar adds s_add_co_u32, s_add_co_ci_u32 and s_add_co_i32.
def test_gfx12_scalar_add_names_are_recognized():
    good = """
    s_add_co_u32 s8, s8, s12
    s_add_co_ci_u32 s9, s9, 0
    s_load_b64 s[10:11], s[8:9], 0x0
    """
    assert _reasons(good) == []
    for op in ("s_add_co_u32", "s_add_co_i32"):
        reasons = _reasons(f"""
    {op} s8, s8, s12
    s_load_b64 s[10:11], s[8:9], 0x0
    """)
        assert len(reasons) == 1 and op in reasons[0] and "s8" in reasons[0]


def test_registers_are_judged_within_their_own_kernel():
    asm = """
    .amdgpu_hsa_kernel first
    s_add_i32 s4, s4, 1
    s_endpgm
    .amdgpu_hsa_kernel second
    s_load_dwordx2 s[10:11], s[4:5], 0x0
    """
    assert _reasons(asm) == []
