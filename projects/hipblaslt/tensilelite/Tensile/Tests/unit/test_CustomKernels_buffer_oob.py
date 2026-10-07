# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Hand-written custom kernels must bound their stores like generated kernels.

A generated kernel sets the length field (num_records) of its D, C and
workspace buffer descriptors to BufferOOB, and moves out-of-bounds lanes to a
store offset of BufferOOB so the hardware discards them. The host predicate
BufferStoreOffsetLimitCheck admits shapes up to the same value. A custom kernel
that keeps an older, smaller value silently drops every store past it for
shapes the host still admits (ROCM-31258). Custom kernels come in two forms:
readable assembly that sets BufferOOB by name, and pre-assembled kernels
stored as raw `.long` instruction words.
"""

import functools
import importlib.util
import pathlib
import re

import pytest

from Tensile.resources import custom_kernel_names, custom_kernel_text

pytestmark = pytest.mark.unit

# BufferOOB before ae0ca64a9c8. Raw-encoded kernels carry it as a literal.
STALE_BUFFER_OOB = 0x80000000
# Buffer descriptor dword 3 (bits 127:96) for the gfx9 kernels in this tree.
SRD_127_96 = 0x00020000

REACHABLE_READABLE_KERNEL = (
    "Custom_Cijk_Ailk_Bljk_S_MX_B_BIAS_HA_S_SAV_NTD_SK3_UserArgs_"
    "MT256x256x32_MI16x16x1_shortname0_gfx950"
)
REACHABLE_ENCODED_KERNEL = (
    "CustomGSUs_Cijk_Ailk_Bljk_HSS_BH_Bias_AS_SAV_MT64x16x64_MI16x16x1_Freesize_2stage_gfx942"
)

# Kernels from other generators, which bound their stores without BufferOOB.
# A kernel in which the scan recognizes no store bound must be listed here.
NO_BUFFER_OOB_KERNELS = {
    "RRGEMM_TN_half_half_half_half_float_WGTS64x64x64_WGS128x2_WGMXCC0_LABufferToLDS_"
    "LBBufferToLDS_SD1_LSABufferToVGPR_LSBBufferToVGPR_UNROLL0x0_SwizzleScale00_"
    "SwizzleTileSize0x0X0x0_PF4x1m1_M_98fd6c3c16c08d0e": "rocRoller",
    "_ZN5aiter24bf16gemm_bf16_tn_256x256E": "aiter",
    "_ZN5aiter42f4gemm_bf16_per1x32Fp4_BpreShuffle_256x256E": "aiter",
    "_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_256_BLOCK_SIZE_N_256_BLOCK_SIZE_K_64_GROUP_SIZE_M_8_"
    "num_warps_4_num_stages_2_waves_per_eu_0_matrix_instr_nonkdim_16_cache_modifier_NONE_"
    "NUM_KSPLIT_1": "Triton, global_store only",
    "c_ck_gemm_basic_hip_amdgcn_amd_amdhsa_gfx942": "Composable Kernel",
    "wave_bf16_gemm_256x256x64": "Wave, global_store only",
}

_SET_BUFFER_OOB = re.compile(r"^\s*\.set\s+BufferOOB\s*,\s*([^\s/]+)", re.M)
_STORE_SRD_NUM_RECORDS = re.compile(
    r"^\s*s_mov_b32\s+s\[sgprSrd(D|C|WS)\+2\]\s*,\s*([^\s/]+)", re.M
)
_LONG_DIRECTIVE = re.compile(r"^\s*\.long\s+([^/;\n]+)", re.M)
_INTEGER = re.compile(r"(0[xX][0-9a-fA-F]+|\d+)")

# s_mov_b32 sdst, <32-bit literal>: SOP1, op 0, ssrc0 = 0xff.
_SOP1_MOV_LITERAL_MASK, _SOP1_MOV_LITERAL = 0xFF80FFFF, 0xBE8000FF
# v_mov_b32_e32 vdst, <32-bit literal>: VOP1, op 1, src0 = 0xff.
_VOP1_MOV_LITERAL_MASK, _VOP1_MOV_LITERAL = 0xFE01FFFF, 0x7E0002FF


@functools.lru_cache(maxsize=None)
def generator_buffer_oob():
    spec = importlib.util.find_spec("Tensile.KernelWriterAssembly")
    source = pathlib.Path(spec.origin).read_text(encoding="utf-8")
    match = re.search(r'ValueSet\(\s*"BufferOOB"\s*,\s*(0x[0-9a-fA-F]+|\d+)', source)
    assert match, "KernelWriterAssembly.py no longer emits BufferOOB through ValueSet"
    return int(match.group(1), 0)


def _encoded_words(text):
    """Integer `.long` words in order; symbolic data words become None."""
    words = []
    for line in _LONG_DIRECTIVE.findall(text):
        for word in line.split(","):
            word = word.strip()
            words.append(int(word, 0) if _INTEGER.fullmatch(word) else None)
    return words


def _readable_findings(text, expected):
    checked, violations = [], []
    for value in _SET_BUFFER_OOB.findall(text):
        checked.append(".set BufferOOB")
        if int(value, 0) != expected:
            violations.append(f".set BufferOOB, {value}")
    for srd, operand in _STORE_SRD_NUM_RECORDS.findall(text):
        checked.append(f"Srd{srd} num_records")
        if operand != "BufferOOB" and int(operand, 0) != expected:
            violations.append(f"s_mov_b32 s[sgprSrd{srd}+2], {operand}")
    return checked, violations


def _encoded_findings(text, expected):
    """Decode moves of 32-bit literals in a raw-encoded kernel.

    A descriptor length is `s_mov_b32 s[4n+2], literal` next to a move of
    SRD_127_96 into s[4n+3]. An out-of-bounds store offset is
    `v_mov_b32 vN, STALE_BUFFER_OOB`.
    """
    words = _encoded_words(text)
    pairs = [
        (i, words[i], words[i + 1])
        for i in range(len(words) - 1)
        if words[i] is not None and words[i + 1] is not None
    ]
    smov = {
        i: ((word >> 16) & 0x7F, literal)
        for i, word, literal in pairs
        if word & _SOP1_MOV_LITERAL_MASK == _SOP1_MOV_LITERAL
    }
    checked, violations = [], []
    for i, (sdst, literal) in smov.items():
        if sdst % 4 != 2:
            continue
        neighbors = (smov.get(j) for j in range(i - 6, i + 7))
        if (sdst + 1, SRD_127_96) not in neighbors:
            continue
        checked.append(f"s{sdst - 2} num_records")
        if literal != expected:
            violations.append(f"word {i}: s_mov_b32 s{sdst}, {literal:#x}")
    for i, word, literal in pairs:
        if word & _VOP1_MOV_LITERAL_MASK == _VOP1_MOV_LITERAL:
            checked.append("v_mov_b32 literal")
            if literal == STALE_BUFFER_OOB:
                vdst = (word >> 17) & 0xFF
                violations.append(f"word {i}: v_mov_b32 v{vdst}, {STALE_BUFFER_OOB:#x}")
    return checked, violations


def store_bound_findings(text, expected):
    readable = _readable_findings(text, expected)
    encoded = _encoded_findings(text, expected)
    return readable[0] + encoded[0], readable[1] + encoded[1]


@functools.lru_cache(maxsize=None)
def _kernel_text(name):
    return custom_kernel_text(name)


def test_generator_buffer_oob_is_parsed():
    assert generator_buffer_oob() == 0xFFFFF000


@pytest.mark.parametrize("name", custom_kernel_names())
def test_custom_kernel_store_bounds_match_generator(name):
    _, violations = store_bound_findings(_kernel_text(name), generator_buffer_oob())
    assert not violations, (
        f"{name} bounds stores with a value other than BufferOOB = "
        f"{generator_buffer_oob():#x}, so the host can admit shapes whose "
        f"stores this kernel drops:\n  " + "\n  ".join(violations)
    )


def test_every_kernel_without_a_recognized_store_bound_is_listed():
    unrecognized = set()
    for name in custom_kernel_names():
        checked, _ = store_bound_findings(_kernel_text(name), generator_buffer_oob())
        if not any("BufferOOB" in kind or "num_records" in kind for kind in checked):
            unrecognized.add(name)
    listed = set(NO_BUFFER_OOB_KERNELS)
    assert unrecognized == listed, (
        f"no store bound recognized, not listed: {sorted(unrecognized - listed)}; "
        f"listed, but a store bound is recognized or the kernel is gone: "
        f"{sorted(listed - unrecognized)}"
    )


@pytest.mark.parametrize(
    "name, kinds",
    [
        (REACHABLE_READABLE_KERNEL, {".set BufferOOB", "SrdD num_records", "SrdC num_records"}),
        (REACHABLE_ENCODED_KERNEL, {"s12 num_records", "v_mov_b32 literal"}),
    ],
)
def test_scan_recognizes_store_bounds_in_known_kernels(name, kinds):
    checked, _ = store_bound_findings(_kernel_text(name), generator_buffer_oob())
    assert kinds <= set(checked)


@pytest.mark.parametrize(
    "text, violation",
    [
        (".set BufferOOB, 0x80000000\n", ".set BufferOOB, 0x80000000"),
        (
            "s_mov_b32 s[sgprSrdD+2], 0x80000000 // num_records\n",
            "s_mov_b32 s[sgprSrdD+2], 0x80000000",
        ),
        (
            ".long 0xBE8E00FF, 0x80000000\n.long 0xBE8F00FF, 0x00020000\n",
            "word 0: s_mov_b32 s14, 0x80000000",
        ),
        (".long 0x7E1602FF, 0x80000000\n", "word 0: v_mov_b32 v11, 0x80000000"),
    ],
)
def test_scan_reports_stale_store_bounds(text, violation):
    assert store_bound_findings(text, 0xFFFFF000)[1] == [violation]


@pytest.mark.parametrize(
    "text",
    [
        ".set BufferOOB, 0xfffff000\ns_mov_b32 s[sgprSrdD+2], BufferOOB\n",
        ".long 0xBE8E00FF, 0xFFFFF000\n.long 0xBE8F00FF, 0x00020000\n",
        ".long 0x7E1602FF, 0xFFFFF000\n",
        # A literal move into a register that is not a descriptor length.
        ".long 0xBE8D00FF, 0x80000000\n.long 0xBE8E00FF, 0x00020000\n",
    ],
)
def test_scan_accepts_current_store_bounds(text):
    assert store_bound_findings(text, 0xFFFFF000)[1] == []
