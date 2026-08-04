#!/usr/bin/env python3

import argparse
import datetime
import glob
import json
import logging
import os
import pathlib
import re

import typing_extensions as ty
import jinja2

log = logging.getLogger("confgen.generate")


def annotate_type(type_id, type_name):
    """Creates annotations for types to simplify jinja templating."""
    is_floating = False
    size = 0
    size_min = None

    # Floating point types
    if type_id == "rocprim::half":
        is_floating = True
        size = 2
        # Overwrite size_min as there is no 1-byte floating point type
        size_min = 0
    elif type_id == "float":
        is_floating = True
        size = 4
    elif type_id == "double":
        is_floating = True
        size = 8
    # Integer types
    elif type_id == "int8_t":
        size = 1
    elif type_id == "short":
        size = 2
    elif type_id == "int":
        size = 4
    elif type_id == "int64_t":
        size = 8
    elif type_id == "rocprim::int128_t":
        size = 16

    # Derive size_min from size if not set
    if size_min is None:
        size_min = size // 2

    return {
        "name": type_name,
        "type": type_id,
        "size": size,
        "size_min": size_min,
        "is_floating": is_floating,
    }


def make_config(
    type_data: dict[str, ty.Any], config_data: dict[str, ty.Any] | None = None
):
    """Constructs a dictionary that contains a config that can be used in jinja templating."""
    if config_data is None:
        config_data = type_data

    # Select the following type names
    # TODO: derive this from alg name maybe?
    type_names = ["key_type", "value_type"]
    # Only select relevant types
    type_info = {k: type_data[k] for k in type_names if k in type_data}
    # Annotate with extra info for jinja
    type_details = {k: annotate_type(type_info[k], k) for k in type_info}
    type_hash = frozenset(type_info.items())
    return (type_hash, {"type": type_details, "config": type_info | config_data})


def derive_gen(arch: str) -> str:
    """Derives the hardware generation from a given architecture."""
    arch_mapping = {
        "gfx803": "gcn3",
        "gfx90[06]": "gcn5",
        "gfx908": "cdna1",
        "gfx90a": "cdna2",
        "gfx94.": "cdna3",
        "gfx95.": "cdna4",
        "gfx101.": "rdna1",
        "gfx103.": "rdna2",
        "gfx11..": "rdna3",
        "gfx12..": "rdna4",
    }
    for pattern, gen in arch_mapping.items():
        if re.match(pattern, arch):
            return gen
    log.warning(f"Could not derive hardware generation from {arch}!")
    return "unknown"


def derive_target(data: dict[str, ty.Any], mapping: dict[str, str]) -> dict[str, str]:
    """Derives the target tuple from a data-dictionary"""
    arch = data["arch_name"]
    gpu = "generic"
    if "gpu" in data:
        gpu = data["gpu"]
    if arch in mapping:
        gpu = mapping[arch]

    if gpu == "generic":
        log.warning(
            f"No gpu name specified and no matching gfx-architecture mapping was specified!"
        )

    return {"gen": derive_gen(arch), "arch": arch, "gpu": gpu, "rep": "amdgcn"}


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
        "--map-gfx",
        "-g",
        action="append",
        nargs=2,
        help="The mapping from gfx-id to gpu name.",
    )

    args = cli.parse_args()

    gfx_mapping = {}
    if args.map_gfx:
        gfx_mapping = dict(args.map_gfx)

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

            # Record the target: gen, arch, gpu, rep
            target_info = derive_target(data, gfx_mapping)
            # Create hashable key
            target_hash = frozenset(target_info.items())

            # Find best config
            config_data = min(data["cache"].values(), key=lambda c: c["time"])
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
            config_data = {
                k: config_data[k] for k in config_data if k not in config_ignored_names
            }

            # Ensure algorithm entry exists
            alg: str = data["algo_name"]
            if alg not in algs:
                algs[alg] = {}

            # Ensure target entry exists
            if target_hash not in algs[alg]:
                algs[alg][target_hash] = {}

            type_hash, config = make_config(data, config_data)

            # Add config entry
            algs[alg][target_hash][type_hash] = config
    log.info(f"Parsed {len(algs)} different algorithm(s)!")

    script_dir = pathlib.Path(__file__).parent.absolute()
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(script_dir / "templates"),
        autoescape=jinja2.select_autoescape(),
    )

    # Find existing configs
    existing_dir = None
    if args.existing:
        existing_dir = pathlib.Path(args.existing)
        if not existing_dir.exists():
            log.error(f"Could not find directory: {existing_dir}")
            quit()

    # Merge newly generated configs to existing configs
    if existing_dir:
        import parse as config_parser

        for alg_name, targets in algs.items():
            config_file = existing_dir / f"{alg_name}.hpp"
            log.debug(f"Parsing: {config_file}")
            with open(config_file, "r") as file:
                existing_data = config_parser.parse_lines(file.readlines())
                merged_targets = {
                    target_hash: dict(make_config(entry) for entry in target)
                    for target_hash, target in existing_data.items()
                }

                for target_hash, target in targets.items():
                    if target_hash not in merged_targets:
                        merged_targets[target_hash] = {}

                    for type_hash, entry in target.items():
                        merged_targets[target_hash][type_hash] = entry
            algs[alg_name] = merged_targets

    # Output generated configs
    output_dir = pathlib.Path(args.output)
    os.makedirs(output_dir, exist_ok=True)
    for alg_name, targets in algs.items():

        default_target = min(
            (dict(t) for t in targets.keys() if dict(t)["gen"] != "gcn5"),
            key=lambda t: int(re.sub("gfx", "", t["arch"]), base=16),
        )

        template_name = f"{alg_name}.h.jinja2"
        template = env.get_template(template_name)
        rendered = template.render(
            {
                "year": datetime.datetime.now().year,
                "targets": targets,
                "alg_name": alg_name,
                "alg_short_name": re.sub(r"^device_", "", alg_name),
                "unknown_target": {
                    "gen": "unknown",
                    "arch": "unknown",
                    "gpu": "generic",
                    "rep": "amdgcn",
                },
                "default_target": default_target,
            }
        )

        with open(output_dir / f"{alg_name}.hpp", "w") as file:
            file.write(rendered)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
