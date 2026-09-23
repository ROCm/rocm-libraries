# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Hoist Q27B MT64x256 offsets and adapt packed FP16 to sequential int4.

The existing lifts extract nibble pairs (0,4), (1,5), (2,6), (3,7) from a
sequential unsigned_bias8 dword. Subtraction of the integer zero point is
exact in FP16; multiplication by the FP16 scale rounds once. Four byte
permutations restore consecutive element pairs for the existing LDS writes.
The permutations reuse the four lift temporaries, adding no VGPRs.
"""

from pathlib import Path
import re


NAME = "Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8X_UserArgs_MT64x256x64_MI16x16x1_gfx1151"


def hoist_zero_point_offsets(source):
    """Retain lane-dependent byte offsets across K iterations in v189/v190.

    The original nibble offsets remain live for zero-point selection. Only
    the scalar buffer descriptor advances along K. These two extra VGPRs
    stay within the existing gfx1151 allocation block (192 registers).
    """
    marker = "// Zero-point byte offsets hoisted into v189/v190."
    if marker in source:
        return source
    replacements = {
        ".amdhsa_next_free_vgpr 189": ".amdhsa_next_free_vgpr 191",
        "/* Num VGPR   =189 */": "/* Num VGPR   =191 */",
        ".vgpr_count:                 189": ".vgpr_count:                 191",
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(f"Unexpected register metadata: {old}")
        source = source.replace(old, new)
    source = source.replace(".amdgcn_target", marker + "\n.amdgcn_target", 1)
    for index in range(2):
        offset = f"v[vgprGlobalReadOffsetScaleZeroA+{index}]"
        saved = f"v{189 + index}"
        # The first prefetch initializes the saved offset before every use.
        old = f"v_lshrrev_b32 v0, 1, {offset}"
        if source.count(old) != 1:
            raise ValueError(f"Unexpected prefetch offset calculation: {old}")
        source = source.replace(old, f"v_lshrrev_b32 {saved}, 1, {offset}")
        old = f"v_lshrrev_b32 v179, 1, {offset}\n"
        if source.count(old) != 1:
            raise ValueError(f"Unexpected loop offset calculation: {old}")
        source = source.replace(old, "")
        for temporary in ("v0", "v179"):
            old = f"buffer_load_d16_u8 v[vgprG2LScaleZeroA+{index}], {temporary},"
            if source.count(old) != 1:
                raise ValueError(f"Unexpected zero-point load: {old}")
            source = source.replace(old, f"buffer_load_d16_u8 v[vgprG2LScaleZeroA+{index}], {saved},")
    return source


def convert(source):
    source = source.replace("ZPU8X", "ZPU8").replace("UnsignedBias8ExLlama", "UnsignedBias8")
    source = source.replace(
        "// Generated with W4A16 generator revision 108edb6c5f0.",
        "// Packed FP16 dequantization adapted from the ExLlama kernel (generator 108edb6c5f0).\n"
        "// Sequential U8 nibbles lift as (0,4),(1,5),(2,6),(3,7); byte permutations\n"
        "// restore (0,1),(2,3),(4,5),(6,7) after the exact subtract and FP16 multiply.",
    )

    def permute(match):
        rows = [re.fullmatch(
            r"v_pk_mul_f16 (v\[[^]]+\]), (v\d+), (v\d+)", line
        ).groups() for line in match[0].splitlines()]
        # Keep scaled pairs in their scratch registers until all four outputs
        # have consumed them. V_PERM selects bytes 0..3 from src1, 4..7 from src0.
        result = [f"v_pk_mul_f16 {temp}, {temp}, {scale}\n" for _, temp, scale in rows]
        for i, (a, b, selector) in enumerate([
            (0, 1, "0x05040100"), (2, 3, "0x05040100"),
            (0, 1, "0x07060302"), (2, 3, "0x07060302"),
        ]):
            result.append(
                f"v_perm_b32 {rows[i][0]}, {rows[b][1]}, {rows[a][1]}, {selector}\n"
            )
        return "".join(result)

    source, count = re.subn(r"(?:v_pk_mul_f16 v\[vgprG2LA[^\n]+\n){4}", permute, source)
    if count != 4:
        raise ValueError(f"Expected four dequantization blocks; found {count}. Review source kernel.")
    return source


def main():
    directory = Path(__file__).resolve().parent.parent
    source = directory / (NAME + ".s")
    destination = directory / (NAME.replace("ZPU8X", "ZPU8") + ".s")
    hoisted = hoist_zero_point_offsets(source.read_text())
    source.write_text(hoisted)
    destination.write_text(convert(hoisted))


if __name__ == "__main__":
    main()
