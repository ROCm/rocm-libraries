# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import argparse
import importlib.util
import os


def _import_gemm_kernel_builder():
    """Import the shared GEMM kernel builder."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder


GemmKernelBuilder = _import_gemm_kernel_builder()


def normalize_gpu_targets(raw_targets):
    if raw_targets is None:
        return []

    targets = []
    for raw_target in raw_targets:
        for part in str(raw_target).replace(";", ",").split(","):
            normalized = part.strip()
            if normalized:
                targets.append(normalized)

    ordered = []
    for target in targets:
        if target not in ordered:
            ordered.append(target)
    return ordered


class FlatmmKernelBuilder(GemmKernelBuilder):
    SUPPORTED_GPU_TARGETS = ["gfx90a", "gfx942", "gfx950"]
    DEFAULT_VECTOR_LOAD_SIZE_BYTES = 16
    DEFAULT_SMEM_CAPACITY_BYTES = 65536
    GFX950_SMEM_CAPACITY_BYTES = 163840

    def __init__(self, working_path, gpu_target, datatype, layout, config_json):
        self.gpu_targets = normalize_gpu_targets(gpu_target)
        super().__init__(
            "flatmm",
            working_path,
            ",".join(self.gpu_targets),
            datatype,
            layout,
            config_json,
        )

    @staticmethod
    def _with_target_archs(tile_config, target_archs):
        config = dict(tile_config)
        config["target_archs"] = list(target_archs)
        return config

    @staticmethod
    def _get_wave_size(gpu_target):
        if gpu_target.startswith("gfx9"):
            return 64
        return 32

    @staticmethod
    def _get_element_size_bytes(datatype):
        element_sizes = {
            "fp16": 2,
            "bf16": 2,
            "fp8": 1,
            "bf8": 1,
            "int8": 1,
            "fp32": 4,
            "fp64": 8,
        }
        return element_sizes.get(datatype)

    @staticmethod
    def _get_smem_capacity_bytes(gpu_target):
        if gpu_target == "gfx950":
            return FlatmmKernelBuilder.GFX950_SMEM_CAPACITY_BYTES
        return FlatmmKernelBuilder.DEFAULT_SMEM_CAPACITY_BYTES

    def _get_k_per_thread(self, warp_tile_m, warp_tile_k, gpu_target):
        return (warp_tile_m * warp_tile_k) // self._get_wave_size(gpu_target)

    def _is_valid_flatmm_vector_load(self, warp_tile_m, warp_tile_k, gpu_target):
        element_size = self._get_element_size_bytes(self.datatype)
        if element_size is None:
            return True

        dsread_bytes_per_wave = (
            warp_tile_m
            * warp_tile_k
            * element_size
            // self._get_wave_size(gpu_target)
        )
        return (
            dsread_bytes_per_wave % self.DEFAULT_VECTOR_LOAD_SIZE_BYTES == 0
        )

    def _get_flatmm_pipeline_smem_bytes(
        self, tile_m, tile_k, warp_tile_m, warp_tile_n, warp_tile_k, gpu_target
    ):
        element_size = self._get_element_size_bytes(self.datatype)
        if element_size is None:
            return 0

        vector_load_elements = self.DEFAULT_VECTOR_LOAD_SIZE_BYTES // element_size
        k_per_thread = self._get_k_per_thread(warp_tile_m, warp_tile_k, gpu_target)
        k_pack = min(k_per_thread, vector_load_elements)

        if warp_tile_m == 16 and warp_tile_n == 16:
            smem_elements = tile_m * tile_k
        else:
            smem_elements = (tile_m + 1) * tile_k - k_pack

        return smem_elements * element_size

    def _is_valid_flatmm_lds_capacity(
        self,
        tile_m,
        tile_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        gpu_target,
    ):
        flatmm_pipeline_smem = self._get_flatmm_pipeline_smem_bytes(
            tile_m, tile_k, warp_tile_m, warp_tile_n, warp_tile_k, gpu_target
        )
        total_smem = 2 * flatmm_pipeline_smem
        return total_smem <= self._get_smem_capacity_bytes(gpu_target)

    def _validate_tile_config(
        self,
        tile_m,
        tile_n,
        tile_k,
        warp_m,
        warp_n,
        warp_k,
        warp_tile_m,
        warp_tile_n,
        warp_tile_k,
        pipeline,
    ):
        if not super()._validate_tile_config(
            tile_m,
            tile_n,
            tile_k,
            warp_m,
            warp_n,
            warp_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            pipeline,
        ):
            return False

        gpu_target = self.gpu_targets[0] if self.gpu_targets else self.gpu_target
        if not self._is_valid_flatmm_vector_load(
            warp_tile_m, warp_tile_k, gpu_target
        ):
            return False

        return self._is_valid_flatmm_lds_capacity(
            tile_m,
            tile_k,
            warp_tile_m,
            warp_tile_n,
            warp_tile_k,
            gpu_target,
        )

def parse_tile_config(tile_config_str):
    tile_dims, warp_dims, warp_tile_dims = tile_config_str.split("_")
    tile_m, tile_n, tile_k = [int(value) for value in tile_dims.split("x")]
    warp_m, warp_n, warp_k = [int(value) for value in warp_dims.split("x")]
    warp_tile_m, warp_tile_n, warp_tile_k = [
        int(value) for value in warp_tile_dims.split("x")
    ]
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": tile_k,
        "warp_m": warp_m,
        "warp_n": warp_n,
        "warp_k": warp_k,
        "warp_tile_m": warp_tile_m,
        "warp_tile_n": warp_tile_n,
        "warp_tile_k": warp_tile_k,
    }


def parse_trait_combo(trait_combo_str):
    parts = trait_combo_str.split("_")
    if len(parts) != 7:
        raise ValueError(f"Unexpected trait combo: {trait_combo_str}")

    return (
        parts[0],
        parts[1],
        parts[2],
        parts[3] == "True",
        parts[4] == "True",
        parts[5] == "True",
        parts[6] == "True",
    )


def main():
    parser = argparse.ArgumentParser(description="FLATMM kernel instance builder")
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--gpu_target",
        nargs="+",
        required=True,
        help="GPU target architecture(s)",
    )
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp16", "bf16", "fp8", "bf8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", required=True, help="Configuration JSON file")
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations",
    )
    parser.add_argument(
        "--gen_single",
        action="store_true",
        help="Generate a single kernel file",
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument("--tile_config", help="Tile configuration string for single generation")
    parser.add_argument("--trait_combo", help="Trait combination string for single generation")

    args = parser.parse_args()

    builder = FlatmmKernelBuilder(
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
        # Generate a single kernel file input validation
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        tile_config = parse_tile_config(args.tile_config)
        trait_combo = parse_trait_combo(args.trait_combo)

        # expected_name = builder._format_kernel_name(trait_combo, tile_config)
        # if args.kernel_name != expected_name:
        #     raise ValueError(
        #         f"Kernel name mismatch: expected {expected_name}, got {args.kernel_name}"
        #     )

        builder._generate_kernel_instance(
            tile_config,
            trait_combo,
        )
    else:
        parser.error("Must specify one of: --list_kernels or --gen_single")


if __name__ == "__main__":
    main()
