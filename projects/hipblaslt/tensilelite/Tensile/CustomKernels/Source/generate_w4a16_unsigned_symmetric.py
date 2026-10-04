# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Derive unsigned symmetric matrix kernels and equality logic from asymmetric ones."""
from pathlib import Path
import re


def symmetric_matrix(text):
    text = text.replace(".amdgcn_target", "// Derived by Source/generate_w4a16_unsigned_symmetric.py; implicit zero point 8.\n.amdgcn_target", 1)
    text = text.replace("SABBGZPU8", "SABBGU8")
    # Keep the universal argument layout, but never dereference the zero buffer.
    text = re.sub(r"^buffer_load[^\n]*ScaleZeroA[^\n]*\n", "", text, flags=re.M)
    pattern = (
        r"^v_cmp_class_f16[^\n]*\n"
        r"v_cndmask_b32[^\n]*\n"
        r"v_and_b32[^\n]*GlobalReadOffsetScaleZeroA[^\n]*\n"
        r"v_lshlrev_b32[^\n]*\n"
        r"v_bfe_u32[^\n]*\n"
        r"v_lshl_or_b32[^\n]*\n"
        r"v_lshl_or_b32 (v\d+),[^\n]*0xe400e400\n"
        r"v_lshl_or_b32 (v\d+),[^\n]*0xd400d400\n"
    )
    text, count = re.subn(pattern, r"v_mov_b32 \1, 0xe408e408\nv_mov_b32 \2, 0xd480d480\n", text, flags=re.M)
    assert count, "Missing packed dequantization blocks"
    text = re.sub(r"v_pk_fma_f16 ([^\n]+), 0\n", r"v_pk_mul_f16 \1\n", text)
    assert not re.search(r"^v_.*G2LScaleZeroA", text, re.M)
    # The pipelined waits now target scale loads instead of the removed ZP loads.
    # Each has the same number of later scale/B loads as the original ZP/B loads.
    return text


def main():
    kernels = Path(__file__).resolve().parent.parent
    for source in kernels.glob("RuntimeGroup_*I4H*SABBGZPU8*.s"):
        target = source.with_name(source.name.replace("SABBGZPU8", "SABBGU8"))
        target.write_text(symmetric_matrix(source.read_text()))
    logic = kernels.parents[2] / "library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151/Equality"
    source = logic / "gfx1151_Cijk_Alik_Bljk_I4H_HHS_BH_SABBGZPU8_Q27B.yaml"
    text = source.read_text().replace("SABBGZPU8", "SABBGU8").replace("SABB32ZPU8", "SABB32U8")
    text = text.replace("# Q27B projections: group-32 unsigned_bias8, FP16, measured with padded A rows.",
                        "# Unsigned symmetric FP16, runtime G32/G64/G128.\n"
                        "# Shape selections and historical scores copied from unsigned asymmetric logic.")
    text = text.replace("ScaleZeroPointA: true", "ScaleZeroPointA: false")
    text = re.sub(r"(RuntimeGroup_Decode\w*)_UnsignedBias8", r"\1_Symmetric_UnsignedBias8", text)
    source.with_name(source.name.replace("SABBGZPU8", "SABBGU8")).write_text(text)


if __name__ == "__main__":
    main()
