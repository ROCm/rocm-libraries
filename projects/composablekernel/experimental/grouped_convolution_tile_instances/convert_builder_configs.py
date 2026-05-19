#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Converts CK Builder .conf config files to JSON format for the CK Dispatcher codegen.
#
# CK Builder instances are parameterized with Seq() thread block cluster lengths
# and k0/k1 decompositions that control thread-to-data mappings at a level of
# detail the dispatcher codegen does not model.  Multiple Builder instances that
# differ only in these parameters produce identical dispatcher configurations
# (same tile/warp/vector sizes, pipeline, scheduler, specialization).  The
# converter therefore deduplicates the output so each unique dispatcher config
# appears exactly once in the JSON.
#
# Two categories of Builder instances are skipped because of hardware or
# architecture limitations (the Builder's generate_instances.py also skips them):
#   1. Irregular vector sizes (odd values other than 1) — AMD GPUs only have
#      vector load instructions for widths 1, 2, 4, 8, 16
#   2. Multi-warp per continuous tile dimension
#      (tile_m > warp_size * vec_a, or tile_n > warp_size * vec_b) — the
#      codegen assumes single-warp coverage per tile dimension for data loading
#
# Usage example:
#   python3 convert_builder_configs.py convert \
#     --input configs/backward_weight/profiler/nhwgc_bf16.conf \
#     --output ../../dispatcher/codegen/configs/grouped_conv/backward_weight/profiler/nhwgc_bf16.json \
#     --variant bwd_weight --layout nhwgc --datatype bf16 --ndim 2
#
# Or convert all configs at once:
#   python3 convert_builder_configs.py convert-all

import argparse
import json
import sys
from pathlib import Path

# generate_instances.py lives is the authoritative source for parsing CK Builder .conf files.
# Import from it directly such that this converter doesn't duplicates the logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_instances import (  # noqa: E402
    ConvInstanceTemplateParams,
    parse_fwd_instances,
    parse_bwd_weight_instances,
    parse_bwd_data_instances,
)


def map_pipeline_version(version_str):
    """Map CK Builder pipeline version to dispatcher pipeline string."""
    mapping = {
        "V1": "compv1",
        "V2": "mem",
        "V3": "compv3",
        "V4": "compv4",
        "V5": "compv5",
        "V6": "compv6",
        "ASYNC_V1": "basic_async_v1",
        "ASYNC_V4": "mem",
    }
    return mapping.get(version_str, version_str.lower())


def map_scheduler(scheduler_str):
    """Map CK Builder scheduler to dispatcher scheduler string."""
    if "Intrawave" in scheduler_str:
        return "intrawave"
    elif "Interwave" in scheduler_str:
        return "interwave"
    return scheduler_str.lower()


def map_specialization(spec_str):
    """Map CK Builder specialization to dispatcher specialization string."""
    mapping = {
        "Default": "default",
        "OddC": "default",
        "Filter1x1Pad0": "filter1x1_pad0",
        "Filter1x1Stride1Pad0": "filter1x1_stride1_pad0",
        "Filter3x3": "filter3x3",
    }
    return mapping.get(spec_str, spec_str.lower())


def conv_params_to_dict(p: ConvInstanceTemplateParams) -> dict:
    """Convert a ConvInstanceTemplateParams (CK Builder) to a Dispatcher JSON dict."""
    return {
        "id":                p.id,
        "tile_m":            p.tile_size[0],
        "tile_n":            p.tile_size[1],
        "tile_k":            p.tile_size[2],
        "warp_m":            p.warps[0],
        "warp_n":            p.warps[1],
        "warp_k":            p.warps[2],
        "warp_tile_m":       p.warp_tile[0],
        "warp_tile_n":       p.warp_tile[1],
        "warp_tile_k":       p.warp_tile[2],
        "vector_size_a":     p.scalar_per_vector[0],
        "vector_size_b":     p.scalar_per_vector[1],
        "vector_size_c":     p.scalar_per_vector[2],
        "pipeline":          map_pipeline_version(p.pipeline_version),
        "scheduler":         map_scheduler(p.scheduler),
        "epilogue":          "cshuffle",
        "double_smem_buffer": p.double_smem_buffer,
        "num_groups_to_merge": p.num_groups_to_merge,
        "num_wave_groups":   p.num_wave_groups,
        "specialization":    map_specialization(p.specialization),
        "two_stage":         p.is_two_stage_instance,
        "explicit_gemm":     p.explicit_gemm,
        "split_image":       p.split_image,
    }


def convert_config_file(input_path, variant, layout, datatype, ndim):
    """Convert a single CK Builder .conf file to JSON format."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    # problem_name is used only for dtype detection (fp32/fp16/bf16 substring match)
    problem_name = f"grouped_convolution_{variant}_tile_{layout}_{datatype}"

    if variant == "bwd_weight":
        raw = parse_bwd_weight_instances(lines, problem_name)
    elif variant == "forward":
        raw = parse_fwd_instances(lines, problem_name)
    elif variant == "bwd_data":
        raw = parse_bwd_data_instances(lines, problem_name)
    else:
        raise RuntimeError(f"Variant '{variant}' conversion not yet implemented.")

    instances = [conv_params_to_dict(p) for p in raw]

    # Deduplicate: Builder instances that differ only in Seq() thread block
    # cluster lengths or k0/k1 decomposition produce identical dispatcher
    # configs because the converter discards these parameters.
    seen = set()
    unique_instances = []
    for inst in instances:
        key = tuple(sorted((k, str(v)) for k, v in inst.items() if k != "id"))
        if key not in seen:
            seen.add(key)
            unique_instances.append(inst)
    if len(unique_instances) < len(instances):
        print(f"  Deduplicated: {len(instances)} -> {len(unique_instances)} "
              f"({len(instances) - len(unique_instances)} duplicates removed)")
    instances = unique_instances

    output = {
        "variant": variant,
        "ndim_spatial": ndim,
        "layout": layout,
        "datatype": datatype,
        "instances": instances,
    }

    print(f"Converted {len(instances)} instances from {input_path}")
    return output

def convert_all(builder_configs_dir, output_dir):
    """Convert all config files for all variants."""
    builder_dir = Path(builder_configs_dir)
    out_dir = Path(output_dir)

    configs = [
        ("nhwgc_fp32", "nhwgc", "fp32", 2),
        ("nhwgc_fp16", "nhwgc", "fp16", 2),
        ("nhwgc_bf16", "nhwgc", "bf16", 2),
        ("ndhwgc_fp32", "ndhwgc", "fp32", 3),
        ("ndhwgc_fp16", "ndhwgc", "fp16", 3),
        ("ndhwgc_bf16", "ndhwgc", "bf16", 3),
    ]

    variants = [
        ("backward_weight", "bwd_weight"),
        ("forward", "forward"),
        ("backward_data", "bwd_data"),
    ]

    for variant_dir, variant_name in variants:
        for prefix in ["tests", "profiler"]:
            for config_name, layout, datatype, ndim in configs:
                input_path = builder_dir / variant_dir / prefix / f"{config_name}.conf"
                if not input_path.exists():
                    print(f"Skipping {input_path} (not found)")
                    continue

                output_path = out_dir / variant_dir / prefix / f"{config_name}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                result = convert_config_file(input_path, variant_name, layout, datatype, ndim)

                with open(output_path, "w") as f:
                    json.dump(result, f, indent=2)
                print(f"  -> {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert CK Builder .conf config files to JSON for CK Dispatcher codegen."
    )
    subparsers = parser.add_subparsers(dest="command")

    # Single file conversion
    single = subparsers.add_parser("convert", help="Convert a single config file")
    single.add_argument("--input", required=True, help="Input .conf file path")
    single.add_argument("--output", required=True, help="Output .json file path")
    single.add_argument("--variant", required=True, choices=["bwd_weight", "forward", "bwd_data"])
    single.add_argument("--layout", required=True, choices=["nhwgc", "ndhwgc"])
    single.add_argument("--datatype", required=True, choices=["fp32", "fp16", "bf16"])
    single.add_argument("--ndim", required=True, type=int, choices=[2, 3])

    # Batch conversion
    batch = subparsers.add_parser("convert-all", help="Convert all backward_weight configs")
    batch.add_argument(
        "--builder-configs-dir",
        default=str(Path(__file__).resolve().parent / "configs"),
        help="Path to CK Builder configs directory",
    )
    batch.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent.parent.parent / "dispatcher/codegen/configs/grouped_conv"),
        help="Output directory for JSON configs",
    )

    args = parser.parse_args()

    if args.command == "convert":
        result = convert_config_file(args.input, args.variant, args.layout, args.datatype, args.ndim)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {output_path}")
    elif args.command == "convert-all":
        convert_all(args.builder_configs_dir, args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
