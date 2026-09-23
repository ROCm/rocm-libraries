# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adapt the Q27B MT64x256 ExLlama packed-FP16 kernel to sequential int4.

The existing lifts extract nibble pairs (0,4), (1,5), (2,6), (3,7) from a
sequential unsigned_bias8 dword. Subtraction of the integer zero point is
exact in FP16; multiplication by the FP16 scale rounds once. Four byte
permutations restore consecutive element pairs for the existing LDS writes.
The permutations reuse the four lift temporaries, adding no VGPRs.
"""

from pathlib import Path
import re


NAME = "Custom_Cijk_Alik_Bljk_I4H_HHS_BH_SABB32ZPU8X_UserArgs_MT64x256x64_MI16x16x1_gfx1151"


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
    destination.write_text(convert(source.read_text()))


if __name__ == "__main__":
    main()
