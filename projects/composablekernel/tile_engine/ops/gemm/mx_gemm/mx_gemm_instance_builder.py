# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
import json
import argparse
import importlib.util
import multiprocessing
import concurrent.futures


def _import_gemm_kernel_builder():
    """Load the parent builder and reuse its architecture validation table."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(
        "gemm_instance_builder",
        os.path.join(parent_dir, "gemm_instance_builder.py"),
    )
    gemm_builder_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gemm_builder_module)

    return gemm_builder_module.GemmKernelBuilder, gemm_builder_module._validation_utils


GemmKernelBuilder, _validation_utils = _import_gemm_kernel_builder()
GEMM_MX_PIPELINES_BY_ARCH = _validation_utils.GEMM_MX_PIPELINES_BY_ARCH
mx_pipeline_extension = _validation_utils.mx_pipeline_extension
#: The pipelines that take a section of "pipeline_tuning_config".
TUNED_PIPELINES = tuple(_validation_utils.load_mx_pipeline_extensions().EXTENSIONS)


def pipeline_tuning_config(config):
    """The sweep's "pipeline_tuning_config": {pipeline: {"key": {"values": [...]}}}, one
    section per pipeline in TUNED_PIPELINES."""
    if "tuning_config" in config:
        raise ValueError(
            '"tuning_config" is now "pipeline_tuning_config": '
            f"{{<pipeline>: {{...}}}} for one of {list(TUNED_PIPELINES)}"
        )
    sections = config.get("pipeline_tuning_config") or {}
    if not isinstance(sections, dict):
        raise ValueError('"pipeline_tuning_config" must map pipelines to sections')
    untuned = sorted(set(sections) - set(TUNED_PIPELINES))
    if untuned:
        raise ValueError(
            f"pipeline_tuning_config: {untuned} take no tuning (tuned: {list(TUNED_PIPELINES)})"
        )
    return sections


class MxGemmKernelBuilder(GemmKernelBuilder):
    def __init__(
        self,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json=None,
        max_instances=None,
        seed=None,
        tier=None,
        manifest_path=None,
    ):
        super().__init__(
            kernel_name_prefix,
            working_path,
            gpu_target,
            datatype,
            layout,
            config_json,
            max_instances=max_instances,
            seed=seed,
            tier=tier,
            manifest_path=manifest_path,
        )
        self.pipeline_tuning_config = pipeline_tuning_config(
            getattr(self, "config", None) or {}
        )

    def _generate_trait_combinations(self):
        arch = self.gpu_target.split(":", 1)[0]
        expected = GEMM_MX_PIPELINES_BY_ARCH.get(arch, ())
        return [
            combo
            for combo in super()._generate_trait_combinations()
            if combo[0] in expected
            and not (arch == "gfx1250" and (combo[5] or combo[6]))  # pad_k, persistent
        ]

    # A pipeline with an extension (mx_pipeline_extensions.py) carries a tuning besides its tile
    # and traits, from its optional "pipeline_tuning_config" section (the product of its values;
    # the defaults when absent), and its own rules, name suffix and C++ instance. Other pipelines
    # take no tuning.

    def _pipeline_reject_reason(self, extension, tile_config, tuning):
        return _validation_utils.mx_pipeline_reject_reason(
            extension,
            tile_config,
            self.datatype,
            self.datatype,
            self._c_datatype(),
            self.gpu_target,
            tuning,
        )

    def pipeline_name_suffix(self, pipeline, tile_config, tuning):
        return mx_pipeline_extension(pipeline).name_suffix(
            tile_config, tuning, self.datatype, self.datatype, self._c_datatype()
        )

    def _get_sampled_kernel_list(self):
        """The parent's enumeration with every entry of an extension's pipeline expanded over
        its valid tunings (key "tuning"; None for other pipelines), then sampled. The
        dispatcher's expand_mx_sweep calls it."""
        max_instances, self.max_instances = self.max_instances, None
        try:
            kernels = super()._get_sampled_kernel_list()
        finally:
            self.max_instances = max_instances
        tunings = {}
        expanded = []
        for kernel in kernels:
            pipeline = kernel["trait_combo"][0]
            extension = mx_pipeline_extension(pipeline)
            if extension is None:
                expanded.append({**kernel, "tuning": None})
                continue
            if pipeline not in tunings:
                tunings[pipeline] = extension.expand_tuning(
                    self.pipeline_tuning_config.get(pipeline)
                )
            for tuning in tunings[pipeline]:
                if self._pipeline_reject_reason(
                    extension, kernel["tile_config"], tuning
                ):
                    continue
                suffix = self.pipeline_name_suffix(
                    pipeline, kernel["tile_config"], tuning
                )
                expanded.append(
                    {**kernel, "name": kernel["name"] + suffix, "tuning": tuning}
                )
        return self._apply_sampling(expanded)

    def _generate_kernel_instance(
        self, tile_config, trait_combo, validate=False, tuning=None
    ):
        pipeline = self._normalize_trait_combo(trait_combo)[0]
        extension = mx_pipeline_extension(pipeline)
        if extension is not None:
            return self._generate_extension_instance(
                extension, tile_config, trait_combo, tuning
            )
        if tuning:
            raise ValueError(
                f"{pipeline} takes no tuning (tuned: {list(TUNED_PIPELINES)})"
            )
        return super()._generate_kernel_instance(tile_config, trait_combo, validate)

    def _generate_extension_instance(self, extension, tile_config, trait_combo, tuning):
        tuning = extension.normalize_tuning(tuning)
        pipeline, epilogue, scheduler, pad_m, pad_n, pad_k, persistent = (
            self._normalize_trait_combo(trait_combo)
        )
        if not _validation_utils.is_trait_combination_valid(
            pipeline,
            epilogue,
            scheduler,
            persistent,
            self.kernel_name_prefix,
            self.layout,
            pad_m=pad_m,
            pad_n=pad_n,
            pad_k=pad_k,
            gpu_target=self.gpu_target,
        ):
            raise ValueError(
                f"{pipeline} needs an epilogue of {extension.epilogues}, "
                "the intrawave scheduler, no padding and no persistence"
            )
        reason = self._pipeline_reject_reason(extension, tile_config, tuning)
        if reason:
            raise ValueError(f"tile {tile_config} with tuning {tuning}: {reason}")
        kernel_name = self._format_kernel_name(
            trait_combo, tile_config
        ) + self.pipeline_name_suffix(pipeline, tile_config, tuning)
        instance_code = extension.instance_code(self, kernel_name, tile_config, tuning)

        simplified_name = kernel_name[len(self.kernel_name_prefix) + 1 :]
        header_file = (
            self.working_path
            / f"{self.kernel_name_prefix}_single_{simplified_name}.hpp"
        )
        with open(header_file, "w") as f:
            f.write(instance_code)
        print(f"Generated {header_file}")
        return kernel_name, instance_code

    def _generate_all_individual(self, num_workers=None):
        """Generate individual kernel files for separate compilation with parallel processing"""
        if num_workers is None:
            num_workers = min(
                multiprocessing.cpu_count(), 8
            )  # Limit to avoid memory issues

        tile_configs = self._get_tile_configs()
        trait_combos = self._generate_trait_combinations()

        # Prepare work items for parallel processing
        work_items = []
        for tile_config in tile_configs:
            for trait_combo in trait_combos:
                if not self._validate_tile_config(
                    tile_config["tile_m"],
                    tile_config["tile_n"],
                    tile_config["tile_k"],
                    tile_config["warp_m"],
                    tile_config["warp_n"],
                    tile_config["warp_k"],
                    tile_config["warp_tile_m"],
                    tile_config["warp_tile_n"],
                    tile_config["warp_tile_k"],
                    trait_combo[0],
                ):
                    continue
                work_items.append(
                    (
                        tile_config,
                        trait_combo,
                        self.kernel_name_prefix,
                        self.working_path,
                        self.gpu_target,
                        self.datatype,
                        self.layout,
                        self.config_json,
                    )
                )

        # Apply RFC-compliant sampling (Sobol + LHS + maximin)
        if self.max_instances is not None and len(work_items) > self.max_instances:
            kernel_dicts = [
                {"tile_config": item[0], "trait_combo": item[1], "_work_item": item}
                for item in work_items
            ]
            sampled = self._apply_sampling(kernel_dicts)
            work_items = [k["_work_item"] for k in sampled]

        print(
            f"Generating {len(work_items)} individual kernel files using {num_workers} workers..."
        )
        print(f"  Tile configs: {len(tile_configs)}")
        print(f"  Trait combinations: {len(trait_combos)}")
        print(f"  Total kernels: {len(work_items)}")

        # Show first few work items for debugging
        if work_items:
            print("  First work item example:")
            tile_config, trait_combo = work_items[0][:2]
            print(f"    Tile config: {tile_config}")
            print(f"    Trait combo: {trait_combo[:3]}")  # Show first 3 traits

        # Process work items in parallel
        kernel_list = []
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Submit all work items
            print(f"  Submitting {len(work_items)} tasks to executor...")
            future_to_item = {
                executor.submit(_generate_single_kernel_individual, item): item
                for item in work_items
            }
            print("  All tasks submitted, waiting for completion...")

            # Collect results with progress reporting
            for future in concurrent.futures.as_completed(future_to_item):
                completed += 1
                if completed % 100 == 0 or completed == len(work_items):
                    print(
                        f"  Progress: {completed}/{len(work_items)} kernels generated"
                    )
                try:
                    result = future.result()
                    if result:
                        kernel_list.append(result)
                except Exception as exc:
                    item = future_to_item[future]
                    print(f"Kernel generation failed for {item}: {exc}")

        # Sort kernel list for consistent ordering
        kernel_list.sort(key=lambda x: x[0])  # Sort by kernel name

        # Generate CMake include file for individual targets
        self._generate_cmake_individual_targets(kernel_list)

        print(
            f"Generated {len(kernel_list)} individual kernel files in {self.working_path}"
        )


def _generate_single_kernel_individual(work_item):
    """Worker function to generate a single individual kernel file"""
    (
        tile_config,
        trait_combo,
        kernel_name_prefix,
        working_path,
        gpu_target,
        datatype,
        layout,
        config_json,
    ) = work_item

    # Create a temporary builder instance for this worker
    builder = MxGemmKernelBuilder(
        kernel_name_prefix, working_path, gpu_target, datatype, layout, config_json
    )

    try:
        kernel_name, instance_code = builder._generate_kernel_instance(
            tile_config, trait_combo
        )

        # Create simplified filename without the "mx_gemm_" prefix
        # Remove "mx_gemm_" from the beginning of kernel_name for the filename
        simplified_name = kernel_name
        if simplified_name.startswith("mx_gemm_"):
            simplified_name = simplified_name[
                len(kernel_name_prefix) + 1 :
            ]  # Remove "mx_gemm" prefix

        # Write individual header file
        header_file = working_path / f"mx_gemm_single_{simplified_name}.hpp"
        with open(header_file, "w") as f:
            f.write(instance_code)

        return (kernel_name, trait_combo, tile_config)
    except Exception as e:
        print(f"Error generating individual kernel: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="MX GEMM kernel instance builder with parallel support"
    )
    parser.add_argument("--working_path", required=True, help="Working directory path")
    parser.add_argument(
        "--gpu_target",
        required=True,
        help="GPU target architecture",
    )
    parser.add_argument(
        "--datatype",
        required=True,
        choices=["fp4", "fp8"],
        help="Data type",
    )
    parser.add_argument(
        "--layout",
        required=True,
        choices=["rcr"],
        help="Matrix layout",
    )
    parser.add_argument("--config_json", help="Configuration JSON file")
    parser.add_argument(
        "--num_workers", type=int, help="Number of parallel workers (default: auto)"
    )
    parser.add_argument(
        "--gen_all_individual",
        action="store_true",
        help="Generate individual kernel files",
    )
    parser.add_argument(
        "--gen_single", action="store_true", help="Generate a single kernel file"
    )
    parser.add_argument("--kernel_name", help="Kernel name for single generation")
    parser.add_argument(
        "--tile_config", help="Tile configuration string for single generation"
    )
    parser.add_argument(
        "--trait_combo", help="Trait combination string for single generation"
    )
    parser.add_argument(
        "--list_kernels",
        action="store_true",
        help="List kernel configurations without generating files",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap on number of kernel instances per (dtype, layout) combo",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for deterministic sampling; if omitted, derived from today's date",
    )
    parser.add_argument(
        "--tier",
        default=None,
        help="Sampling tier (daily/weekly)",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Directory for chosen_instances.json",
    )

    args = parser.parse_args()

    kernel_name_prefix = "mx_gemm"
    if args.config_json:
        with open(args.config_json) as f:
            pipelines = (
                json.load(f)
                .get("trait_config", {})
                .get("pipeline", {})
                .get("values", [])
            )
        for pipeline in pipelines:
            extension = mx_pipeline_extension(pipeline)
            if extension is not None and extension.dispatcher_only:
                sys.exit(
                    f"{pipeline} runs through the dispatcher (mx_gemm_full_benchmark.py)"
                )
    builder = MxGemmKernelBuilder(
        kernel_name_prefix,
        args.working_path,
        args.gpu_target,
        args.datatype,
        args.layout,
        args.config_json,
        max_instances=args.max_instances,
        seed=args.seed,
        tier=args.tier,
        manifest_path=args.manifest_path,
    )

    if args.list_kernels:
        builder._list_kernels()
    elif args.gen_single:
        # Generate a single kernel file input validation
        if not args.kernel_name or not args.tile_config or not args.trait_combo:
            parser.error(
                "--gen_single requires --kernel_name, --tile_config, and --trait_combo"
            )

        # Parse tile config
        tile_parts = args.tile_config.split("_")
        tile_dims = tile_parts[0].split("x")
        warp_dims = tile_parts[1].split("x")
        warp_tile_dims = tile_parts[2].split("x")

        tile_config = {
            "tile_m": int(tile_dims[0]),
            "tile_n": int(tile_dims[1]),
            "tile_k": int(tile_dims[2]),
            "warp_m": int(warp_dims[0]),
            "warp_n": int(warp_dims[1]),
            "warp_k": int(warp_dims[2]),
            "warp_tile_m": int(warp_tile_dims[0]),
            "warp_tile_n": int(warp_tile_dims[1]),
            "warp_tile_k": int(warp_tile_dims[2]),
        }

        # Parse trait combo from the right so pipeline names may contain underscores.
        trait_parts = args.trait_combo.split("_")
        trait_combo = (
            "_".join(trait_parts[:-6]),  # pipeline
            trait_parts[-6],  # epilogue
            trait_parts[-5],  # scheduler
            trait_parts[-4] == "True",  # pad_m
            trait_parts[-3] == "True",  # pad_n
            trait_parts[-2] == "True",  # pad_k
            trait_parts[-1] == "True",  # persistent
        )

        # Generate the kernel
        builder._generate_kernel_instance(
            tile_config,
            trait_combo,
        )
    elif args.gen_all_individual:
        # Generate all individual kernel files
        builder._generate_all_individual(args.num_workers)
    else:
        parser.error(
            "Must specify one of: --list_kernels, --gen_all_individual, or --gen_single"
        )


if __name__ == "__main__":
    main()
