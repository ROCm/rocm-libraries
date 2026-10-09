#!/usr/bin/env python3

import argparse
import datetime
import glob
import json
import logging
import os
import pathlib

import typing_extensions as ty
import jinja2

log = logging.getLogger("confgen.generate")


def make_config(type_data: dict[str, ty.Any]):
    import_keys = ["arch_name", "tune_params_keys", "tune_params"]
    out = {k: type_data[k] for k in import_keys if k in type_data}
    return out


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--input",
        "-i",
        required=True,
        help="Directory or file glob-pattern that points to kernel tuner results",
    )
    cli.add_argument(
        "--output", "-o", required=True, help="The output directory for config headerse"
    )
    cli.add_argument(
        "--existing",
        "-e",
        help="The directory with current config headers to merge new configs with.",
    )
    cli.add_argument(
        "--target-arch",
        "-t",
        action="append",
        required=True,
        help="The target arch <gfx> that these configs are for.",
    )

    args = cli.parse_args()

    gfx_target = []
    if args.target_arch:
        gfx_target = args.target_arch

    # Read '--input' argument and suffix with '.json' if it's not '.json'.
    input_glob = args.input
    if not input_glob.endswith(".json"):
        input_glob = f"{input_glob}/*.json"

    # Find all files.
    file_paths = glob.glob(input_glob)

    algs: dict[str, dict[frozenset, dict[frozenset, dict[str, ty.Any]]]] = {}
    for file_path in file_paths:
        log.debug(f"Parsing: {file_path}")
        with open(file_path, "r") as f:
            try:
                data = json.load(f)
            except json.decoder.JSONDecodeError:
                log.warning(f"Skipping due to JSONDecodeError: {file_path}")
                continue

            # Filter/skip data entries that have no "time" column
            data_filter = [c for c in data["cache"].values() if "time" in c]
            filter_dif = len(data["cache"]) - len(data_filter)
            if filter_dif > 0:
                log.warning(f"Skipping failed compilations: {filter_dif}")

            # Filter/skip data entries that has an acrchitecture thats not targeted with -t
            arch = data["arch_name"]
            if arch not in gfx_target:
                log.warning(
                    f"Skipping {file_path}: arch {arch} not in --target-arch {gfx_target}"
                )
                continue

            # Find best config
            min_configs = min(data_filter, key=lambda c: c["time"])
            # Drop unrelated entries
            config_ignored_names = [
                "time",
                "times",
                "compile_time",
                "verification_time",
                "framework_time",
                "benchmark_time",
                "strategy_time",
                "timestamp",
            ]
            min_configs = {
                k: min_configs[k] for k in min_configs if k not in config_ignored_names
            }

            # Ensure algorithm entry exists
            alg: str = data["algo_name"]

            config = make_config(data)
            config["tune_params"] = min_configs
            algs.setdefault(alg, {})[arch] = config

    if not algs:
        log.error(f"No tuning reslults found for targets {gfx_target} in {input_glob}")
        quit(1)

    found_archs = {arch for targets in algs.values() for arch in targets}
    for target in gfx_target:
        if target not in found_archs:
            log.warning(f"No tuning results found for --target-arch {target}")

    log.info(f"Parsed {len(algs)} different algorithm(s)!")

    script_dir = pathlib.Path(__file__).parent.absolute()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(script_dir / "templates"),
        autoescape=jinja2.select_autoescape(),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Find existing configs
    existing_dir = None
    if args.existing:
        existing_dir = pathlib.Path(args.existing)
        if not existing_dir.exists():
            log.error(f"Could not find directory: {existing_dir}")
            quit(1)

    # Merge newly generated configs to existing configs
    if existing_dir:
        import parse as config_parser

        for algo_name, targets in algs.items():
            config_file = existing_dir / f"{algo_name}_config.hpp"
            log.debug(f"Parsing: {config_file}")
            with open(config_file, "r") as file:
                existing_data = config_parser.parse_lines(file.readlines())

                merged_targets = {
                    target_hash: make_config(target)
                    for target_hash, target in existing_data.items()
                }

                for target_hash, config in targets.items():
                    merged_targets[target_hash] = config
            algs[algo_name] = merged_targets

    # Output generated configs
    output_dir = pathlib.Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    for algo_name, configs in algs.items():

        template_name = f"{algo_name}.h.jinja2"
        template = env.get_template(template_name)
        rendered = template.render(
            {
                "year": datetime.datetime.now().year,
                "targets": configs,
                "algo_name": algo_name,
            }
        )

        with open(output_dir / f"{algo_name}_config.hpp", "w") as file:
            file.write(rendered)
            file.write(f"\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
