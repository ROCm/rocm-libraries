#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Optimize generated gfx1151 W4A16 dequantization, including prologues and tails.

Run on fresh generator output, before applying this pass:
    uv run python optimize_w4a16_prefill.py generated.s optimized.s
"""

import argparse
from pathlib import Path
import re


def optimize_bf16(text):
    # Pack the two rounded high halves directly, retaining the FP32 RNE path.
    pattern = re.compile(
        r"^v_lshrrev_b32 (v\d+), 16, \1[^\n]*\n"
        r"((?:[^\n]*\n){4})"
        r"v_lshrrev_b32 (v\d+), 16, \3[^\n]*\n"
        r"v_pack_b32_f16 (v\[vgprG2LA[^\]]+\]), \1, \3[^\n]*",
        re.M,
    )

    def replace(m):
        assert "v_cmp_u_f32" in m[2] and "v_cndmask_b32" in m[2]
        return m[2] + f"v_perm_b32 {m[4]}, {m[3]}, {m[1]}, 0x07060302"

    result, count = pattern.subn(replace, text)
    assert count, "No BF16 rounding/packing sequences found"
    return result, {"bf16_pairs": count}


def optimize_fp16(text):
    sgpr = int(re.search(r"\.amdhsa_next_free_sgpr (\d+)", text)[1])
    vgpr = int(re.search(r"\.amdhsa_next_free_vgpr (\d+)", text)[1])
    starts = list(
        re.finditer(
            r"^v_(?:cvt_f32_f16|and_b32) (v\d+), (?:0xffff, )?"
            r"(v\[vgprG2LScaleA\+\d+\])[^\n]*\n",
            text,
            re.M,
        )
    )
    assert starts, "Expected unoptimized generated FP16 input"
    edits = []
    for match in starts:
        end = text.index("\nds_store_", match.end())
        block = text[match.start() : end]
        raw = re.search(r"v_mov_b32 (v\d+), (v\[vgprG2LA[^\]]+\])", block)
        assert raw, block
        base = int(raw[1][1:]) - 2
        tmp, high, source, scale, low = (f"v{base + i}" for i in range(5))
        pairs = [f"v{base + 5 + i}" for i in range(4)]
        vgpr = max(vgpr, base + 9)
        assert vgpr <= 256
        dests = re.findall(
            r"^v_(?:pack_b32_f16|perm_b32) (v\[vgprG2LA[^\]]+\])", block, re.M
        )
        assert len(dests) == 4
        zp = re.search(
            r"v_bfe_([ui])32 v\d+, (v\[vgprG2LScaleZeroA[^\]]+\]), ([^,\n]+), 0x4",
            block,
        )
        signed = bool(re.search(r"v_bfe_i32 v\d+, " + re.escape(raw[1]) + r",", block))
        assert not (zp and zp[1] == "i"), (
            "FP16 signed asymmetric not supported by this input family"
        )
        out = [f"v_pack_b32_f16 {scale}, {match[2]}, {match[2]}"]
        if zp:
            # Match the original FP32 cancellation for NaN/Inf scales.
            out += [
                f"v_cmp_class_f16 s{sgpr + 2}, {match[2]}, 0x207",
                f"v_cndmask_b32 {scale}, {scale}, 0x7e007e00, s{sgpr + 2}",
            ]
            shift = zp[3]
            if shift.startswith("v"):
                offset = re.search(r"v\[vgprGlobalReadOffsetScaleZeroA\+\d+\]", block)[
                    0
                ]
                out += [
                    f"v_and_b32 {tmp}, 1, {offset}",
                    f"v_lshlrev_b32 {tmp}, 2, {tmp}",
                ]
                shift = tmp
            out += [
                f"v_bfe_u32 {tmp}, {zp[2]}, {shift}, 0x4",
                f"v_lshl_or_b32 {tmp}, {tmp}, 16, {tmp}",
                f"v_lshl_or_b32 {low}, {tmp}, 0, 0xe400e400",
                f"v_lshl_or_b32 {high}, {tmp}, 4, 0xd400d400",
            ]
            biases = [low, high]
        else:
            assert signed
            biases = ["0xe408e408", "0xd480d480"]
        out.append(f"v_mov_b32 {source}, {raw[2]}")
        if signed:
            out.append(f"v_xor_b32 {source}, 0x88888888, {source}")
        out += [
            f"v_and_or_b32 {pairs[0]}, {source}, s{sgpr}, 0x64006400",
            f"v_and_or_b32 {pairs[1]}, {source}, s{sgpr + 1}, 0x54005400",
            f"v_lshrrev_b32 {source}, 8, {source}",
            f"v_and_or_b32 {pairs[2]}, {source}, s{sgpr}, 0x64006400",
            f"v_and_or_b32 {pairs[3]}, {source}, s{sgpr + 1}, 0x54005400",
        ]
        for i, reg in enumerate(pairs):
            out.append(f"v_pk_add_f16 {reg}, {reg}, {biases[i % 2]}")
        for reg in pairs:
            # Symmetric multiply preserves -0; asymmetric FMA preserves +0.
            out.append(
                f"v_pk_fma_f16 {reg}, {reg}, {scale}, 0"
                if zp
                else f"v_pk_mul_f16 {reg}, {reg}, {scale}"
            )
        for i, dest in enumerate(dests):
            lo = 2 * (i % 2)
            perm = "0x05040100" if i < 2 else "0x07060302"
            out.append(f"v_perm_b32 {dest}, {pairs[lo + 1]}, {pairs[lo]}, {perm}")
        edits.append((match.start(), end, "\n".join(out)))
    for start, end, replacement in reversed(edits):
        text = text[:start] + replacement + text[end:]
    name = re.search(r"^\.globl (\S+)", text, re.M)[1]
    entry = name + ":\n"
    assert text.count(entry) == 1
    text = text.replace(
        entry,
        entry + f"s_mov_b32 s{sgpr}, 0x000f000f\ns_mov_b32 s{sgpr + 1}, 0x00f000f0\n",
    )
    text = re.sub(r"(\.amdhsa_next_free_sgpr )\d+", rf"\g<1>{sgpr + 3}", text)
    text = re.sub(r"(\.sgpr_count:\s*)\d+", rf"\g<1>{sgpr + 3}", text)
    text = re.sub(r"(\.amdhsa_next_free_vgpr )\d+", rf"\g<1>{vgpr}", text)
    text = re.sub(r"(\.vgpr_count:\s*)\d+", rf"\g<1>{vgpr}", text)
    text = re.sub(r"(/\* Num VGPR\s*=)\d+", rf"\g<1>{vgpr}", text)
    text = re.sub(r"(/\* Num SGPR\s*=)\d+", rf"\g<1>{sgpr + 3}", text)
    return text, {"fp16_dwords": len(edits), "vgprs": vgpr, "sgprs": sgpr + 3}


def convert(text):
    if "v_wmma_f32_16x16x16_bf16" in text:
        return optimize_bf16(text)
    assert "v_wmma_f32_16x16x16_f16" in text
    return optimize_fp16(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    text, stats = convert(args.source.read_text())
    args.output.write_text(text)
    print(stats)


if __name__ == "__main__":
    main()
