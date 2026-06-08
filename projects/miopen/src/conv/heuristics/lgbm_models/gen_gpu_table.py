#!/usr/bin/env python3
"""
Generate lgbm_gpu_table.cpp from ~/AutoResearchAllLGBM/GPUInfo/*.json.

The emitted table has one entry per training-time spec_id (9 total), ordered
to match model_meta.json categorical_vocab.spec_id so the array index
doubles as the spec_id categorical code passed to the rank/appl predictors.

Run this whenever GPUInfo/*.json or model_meta.json change. The script
deliberately produces only a .cpp file (no header) - the C++ struct
definition lives in lgbm_gpu_features.hpp and is hand-maintained.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_GPU_INFO = Path(os.path.expanduser("~/AutoResearchAllLGBM/GPUInfo"))
DEFAULT_META     = Path(os.path.expanduser("~/AutoResearchAllLGBM/deploy/model_meta.json"))
OUT_FILE         = HERE.parent / "lgbm_gpu_table.cpp"

# Must match GpuFeatures struct field order in lgbm_gpu_features.hpp.
NUMERIC_FIELDS = [
    "cu_count", "wave_size", "simds_per_cu", "max_waves_per_cu",
    "lds_size_per_cu_kb", "lds_size_per_workgroup_kb",
    "l1_cache_kb_per_cu", "l2_cache_total_kb", "l3_infinity_cache_kb",
    "vgpr_per_simd", "sgpr_per_simd", "boost_clock_mhz",
    "xcd_count", "shader_engines", "cacheline_size_bytes", "vram_bytes",
    "peak_tflops_fp64", "peak_tflops_fp32", "peak_tflops_fp16",
    "peak_tflops_bf16", "peak_tflops_fp8", "peak_tflops_fp4",
    "peak_tflops_int8",
    "mfma_shape_count", "dtype_support_count",
]
PEAK_DTYPES = ["fp64", "fp32", "fp16", "bf16", "fp8", "fp4", "int8"]
# Categorical fields are stored as their integer vocab codes (or -1 for missing).
CAT_FIELDS = [
    "gfx_id", "arch_family", "matrix_core_gen",
    "winograd_support", "asm_implicit_gemm_support",
    "spec_id",
]


def cat_code(vocab: list[str], value) -> int:
    """Map a categorical value to its index in the model's vocabulary.

    Booleans get stringified to "true"/"false". Missing/None values map to
    the empty-string slot if the vocab has one, else -1 (Treelite missing
    sentinel)."""
    if isinstance(value, bool):
        value = "true" if value else "false"
    s = "" if value is None else str(value)
    if s in vocab:
        return vocab.index(s)
    if "" in vocab:
        return vocab.index("")
    return -1


def num_literal(v) -> str:
    """Emit a NaN-safe double literal."""
    if v is None:
        return "std::numeric_limits<double>::quiet_NaN()"
    return f"{float(v)!r}"


def build_entry(spec_id: str, spec: dict, vocab: dict[str, list[str]], indent: str) -> str:
    nums: list[str] = []
    for k in NUMERIC_FIELDS:
        if k.startswith("peak_tflops_"):
            d = k[len("peak_tflops_"):]
            peak = spec.get("peak_tflops") or {}
            nums.append(num_literal(peak.get(d)))
        elif k == "mfma_shape_count":
            nums.append(num_literal(len(spec.get("mfma_shapes") or [])))
        elif k == "dtype_support_count":
            nums.append(num_literal(len(spec.get("dtypes") or [])))
        else:
            nums.append(num_literal(spec.get(k)))

    cats: list[str] = []
    for k in CAT_FIELDS:
        v = spec_id if k == "spec_id" else spec.get(k)
        cats.append(str(cat_code(vocab[k], v)))

    fields = ", ".join(nums + cats)
    return f"{indent}/* {spec_id:24s} */ {{{fields}}}"


# When cu_count + vram cannot distinguish two SKUs, prefer this spec_id at
# runtime. Currently gfx950 mi350x and mi355x are byte-identical; default to
# mi355x because its abstain thresholds are more conservative.
PREFERRED_FALLBACK = {
    "gfx950": "gfx950-mi355x",
}


def disambig_table(specs_by_gfx: dict[str, list[tuple[str, dict, int]]], indent: str) -> str:
    """Emit a C++ switch-like function body for SKU disambiguation."""
    lines: list[str] = []
    I1 = indent
    I2 = indent + "    "
    I3 = indent + "        "
    for gfx in sorted(specs_by_gfx):
        entries = specs_by_gfx[gfx]
        lines.append(f"{I1}if(gfx_id == \"{gfx}\")")
        lines.append(f"{I1}{{")
        if len(entries) == 1:
            sid, _spec, idx = entries[0]
            lines.append(f"{I2}return {idx}; // {sid}")
        else:
            from itertools import groupby
            entries_sorted = sorted(entries, key=lambda e: (e[1].get("cu_count") or 0,
                                                            e[1].get("vram_bytes") or 0))
            for cu, group in groupby(entries_sorted, key=lambda e: e[1].get("cu_count")):
                group_list = list(group)
                lines.append(f"{I2}if(cu_count == {cu})")
                lines.append(f"{I2}{{")
                if len(group_list) == 1:
                    sid, _spec, idx = group_list[0]
                    lines.append(f"{I3}return {idx}; // {sid}")
                else:
                    group_list.sort(key=lambda e: e[1].get("vram_bytes") or 0)
                    # Identify groups of indistinguishable entries (same vram).
                    # For those, fall through to a preferred default if one is
                    # configured, else emit them in declared order.
                    distinct_vrams = sorted({sp.get("vram_bytes") or 0 for _, sp, _ in group_list})
                    if len(distinct_vrams) == 1:
                        preferred = PREFERRED_FALLBACK.get(gfx)
                        chosen = next((t for t in group_list if t[0] == preferred), group_list[-1])
                        sid, _, idx = chosen
                        lines.append(f"{I3}return {idx}; // {sid} (preferred among {[t[0] for t in group_list]})")
                    else:
                        for i, (sid, sp, idx) in enumerate(group_list):
                            vram = sp.get("vram_bytes") or 0
                            if i < len(group_list) - 1:
                                next_vram = group_list[i + 1][1].get("vram_bytes") or 0
                                cutoff = (vram + next_vram) // 2
                                lines.append(f"{I3}if(vram_bytes < {cutoff}ULL) return {idx}; // {sid}")
                            else:
                                lines.append(f"{I3}return {idx}; // {sid}")
                lines.append(f"{I2}}}")
            lines.append(f"{I2}return -1;")
        lines.append(f"{I1}}}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-info", default=str(DEFAULT_GPU_INFO))
    parser.add_argument("--meta",     default=str(DEFAULT_META))
    parser.add_argument("--out",      default=str(OUT_FILE))
    args = parser.parse_args()

    gpu_info_dir = Path(args.gpu_info).expanduser()
    meta = json.loads(Path(args.meta).expanduser().read_text())
    vocab = meta["rank"]["categorical_vocab"]
    spec_id_vocab = vocab["spec_id"]

    # Load only the spec_ids that appear in the model's spec_id vocab. SKUs
    # like gfx1100w / gfx1030v are training-time unknown -> abstain at
    # runtime, no table entry needed.
    specs: dict[str, dict] = {}
    for sid in spec_id_vocab:
        path = gpu_info_dir / f"{sid}.json"
        if not path.exists():
            sys.exit(f"missing GPUInfo for vocab spec_id: {path}")
        specs[sid] = json.loads(path.read_text())

    rows = [build_entry(sid, specs[sid], vocab, "    ") for sid in spec_id_vocab]

    specs_by_gfx: dict[str, list[tuple[str, dict, int]]] = {}
    for idx, sid in enumerate(spec_id_vocab):
        gfx = specs[sid].get("gfx_id")
        if gfx:
            specs_by_gfx.setdefault(gfx, []).append((sid, specs[sid], idx))

    spec_name_array = ",\n".join(f"    \"{sid}\"" for sid in spec_id_vocab)
    rows_str = ",\n".join(rows)
    disambig_str = disambig_table(specs_by_gfx, "    ")

    body = (
        "// GENERATED by gen_gpu_table.py from ~/AutoResearchAllLGBM/GPUInfo/*.json.\n"
        "// Do not edit by hand. Regenerate when GPUInfo or model_meta.json change.\n"
        "\n"
        "#include <miopen/conv/heuristics/lgbm_gpu_features.hpp>\n"
        "\n"
        "#include <array>\n"
        "#include <limits>\n"
        "#include <string_view>\n"
        "\n"
        "namespace miopen {\n"
        "namespace ai {\n"
        "namespace lgbm {\n"
        "\n"
        "// Order matches model_meta.json rank.categorical_vocab.spec_id, so the\n"
        "// array index doubles as the spec_id categorical code.\n"
        "const std::array<GpuFeatures, kNumSpecIds> kGpuTable = {{\n"
        f"{rows_str}\n"
        "}};\n"
        "\n"
        "const std::array<std::string_view, kNumSpecIds> kSpecIdNames = {{\n"
        f"{spec_name_array}\n"
        "}};\n"
        "\n"
        "int ResolveSpecId(std::string_view gfx_id, std::size_t cu_count, std::size_t vram_bytes)\n"
        "{\n"
        f"{disambig_str}\n"
        "    return -1;\n"
        "}\n"
        "\n"
        "} // namespace lgbm\n"
        "} // namespace ai\n"
        "} // namespace miopen\n"
    )

    Path(args.out).write_text(body)
    print(f"wrote {args.out} ({len(spec_id_vocab)} specs)")


if __name__ == "__main__":
    main()
