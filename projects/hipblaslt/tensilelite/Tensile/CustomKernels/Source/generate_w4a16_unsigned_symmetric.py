# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Derive unsigned symmetric matrix kernels and equality logic from asymmetric ones."""
from pathlib import Path
import re

import yaml


class LogicDumper(yaml.SafeDumper):
    """The streaming logic loader requires aliases to be expanded."""

    def ignore_aliases(self, data):
        return True


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
    # Zero-point addressing is dead after removing the data loads.
    text = re.sub(r"^[sv]_[^\n]*ScaleZeroA[^\n]*\n", "", text, flags=re.M)
    # Reuse the former scale-classification SGPR for constant packed biases.
    bias_sgpr = int(re.search(r"\.amdhsa_next_free_sgpr (\d+)", text)[1]) - 1
    assert not re.search(rf"\bs{bias_sgpr}\b", text)
    for i, bias in enumerate(("0xe408e408", "0xd480d480")):
        registers = set(re.findall(rf"^v_mov_b32 (v\d+), {bias}$", text, re.M))
        assert len(registers) == 1
        reg = registers.pop()
        text = re.sub(rf"^v_mov_b32 {reg}, {bias}\n", "", text, flags=re.M)
        text = re.sub(rf"^(v_pk_add_f16 [^\n]*), {reg}$",
                      rf"\1, s{bias_sgpr + i}", text, flags=re.M)
    name = re.search(r"^\.globl (\S+)", text, re.M)[1]
    text = text.replace(name + ":\n", name + ":\n"
                        + f"s_mov_b32 s{bias_sgpr}, 0xe408e408\n"
                        + f"s_mov_b32 s{bias_sgpr + 1}, 0xd480d480\n", 1)
    text = re.sub(r"(\.amdhsa_next_free_sgpr )\d+", rf"\g<1>{bias_sgpr + 2}", text)
    text = re.sub(r"(\.sgpr_count:\s*)\d+", rf"\g<1>{bias_sgpr + 2}", text)
    text = re.sub(r"(/\* Num SGPR\s*=)\d+", rf"\g<1>{bias_sgpr + 2}", text)
    # Read the packed load directly until the high-byte shift needs a temporary.
    def unpack_source(match):
        masks = match[3].replace(
            f", {match[1]},", f", {match[2]},")
        return masks + f"v_lshrrev_b32 {match[1]}, 8, {match[2]}\n"

    text, copies = re.subn(
        r"^v_mov_b32 (v\d+), (v\[vgprG2LA[^\]]+\])\n"
        r"((?:v_and_or_b32[^\n]*\n){2})v_lshrrev_b32 \1, 8, \1\n",
        unpack_source, text, flags=re.M)
    assert copies == count
    # The pipelined waits now target scale loads instead of the removed ZP loads.
    # Each has the same number of later scale/B loads as the original ZP/B loads.
    return text


def small_n_logic(text):
    logic = yaml.safe_load(text)
    original = {s["SolutionIndex"]: s for s in logic["Solutions"]}
    next_index = max(original) + 1
    replacements = {}
    for index, solution in original.items():
        name = solution["CustomKernelName"]
        if "Decode" not in name:
            continue
        new_name = name.replace("_UnsignedBias8", "_N4_UnsignedBias8")
        small_n = {k: v.replace(name, new_name) if isinstance(v, str) else v
                   for k, v in solution.items()}
        small_n["SolutionIndex"] = next_index
        logic["Solutions"].append(small_n)
        replacements[index] = next_index
        next_index += 1
    for shape, selection in logic["ExactLogic"]:
        m, n, batch, k = shape
        if n in (2, 3, 4):
            decode = next(row[1][0] for row in logic["ExactLogic"]
                          if row[0] == [m, 1, batch, k])
            selection[0] = replacements[decode]
    header = "\n".join(text.splitlines()[:4]) + "\n"
    return header + yaml.dump(logic, Dumper=LogicDumper, sort_keys=False)


def preserve_exact_logic(text, previous):
    """Retain measured symmetric selections when regenerating kernel metadata."""
    logic = yaml.safe_load(text)
    old = yaml.safe_load(previous)
    indices = {s["CustomKernelName"]: s["SolutionIndex"] for s in logic["Solutions"]}
    old_solutions = {s["SolutionIndex"]: s for s in old["Solutions"]}
    entries = {tuple(shape): selection for shape, selection in logic["ExactLogic"]}
    for shape, selection in old["ExactLogic"]:
        solution = old_solutions[selection[0]]
        name = solution["CustomKernelName"]
        if name not in indices:
            index = max(indices.values(), default=-1) + 1
            solution = dict(solution, SolutionIndex=index)
            logic["Solutions"].append(solution)
            indices[name] = index
        entries[tuple(shape)] = [indices[name], *selection[1:]]
    logic["ExactLogic"] = [[list(shape), selection] for shape, selection in entries.items()]
    return "\n".join(text.splitlines()[:4]) + "\n" + yaml.dump(logic, Dumper=LogicDumper, sort_keys=False)


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
                        "# Exact selections retain measured symmetric tuning.")
    text = text.replace("ScaleZeroPointA: true", "ScaleZeroPointA: false")
    text = re.sub(r"(RuntimeGroup_Decode\w*)_UnsignedBias8", r"\1_Symmetric_UnsignedBias8", text)
    target = source.with_name(source.name.replace("SABBGZPU8", "SABBGU8"))
    text = small_n_logic(text)
    if target.exists():
        text = preserve_exact_logic(text, target.read_text())
    target.write_text(text)


if __name__ == "__main__":
    main()
