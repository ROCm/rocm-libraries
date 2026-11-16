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

MERGE_SORT_BLOCK_MERGE_MAX_MERGEPATH_ITEMS = 1024


def round_up(value, alignment):
    """Rounds value up to a multiple of alignment."""
    return -(-value // alignment) * alignment


def annotate_type(type_id, type_name):
    """Creates annotations for types to simplify jinja templating."""
    is_floating = False
    is_arithmetic = True
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
    elif type_id in ("int8_t", "char"):
        size = 1
    elif type_id == "short":
        size = 2
    elif type_id == "int":
        size = 4
    elif type_id == "int64_t":
        size = 8
    elif type_id == "rocprim::int128_t":
        size = 16
    # Aggregate types: 'custom_type<T,U>' is 'struct { T x; U y; }'
    elif custom_type_match := re.fullmatch(r"custom_type<(.+),(.+)>", type_id):
        is_arithmetic = False
        members = [
            annotate_type(m.strip(), type_name) for m in custom_type_match.groups()
        ]
        if all(m["size"] > 0 for m in members):
            # Scalar members are aligned to their size
            offset = 0
            for member in members:
                offset = round_up(offset, member["size"]) + member["size"]
            size = round_up(offset, max(m["size"] for m in members))
        else:
            log.warning(f"Could not derive the size of {type_id}!")

    # Types without a known size (e.g. 'empty_type', channel counts) are not arithmetic
    if size == 0:
        is_arithmetic = False

    # Derive size_min from size if not set
    if size_min is None:
        size_min = size // 2

    return {
        "name": type_name,
        "type": type_id,
        "size": size,
        "size_min": size_min,
        "is_floating": is_floating,
        "is_arithmetic": is_arithmetic,
        "check_arithmetic": False,
    }


def make_config(
    type_data: dict[str, ty.Any], config_data: dict[str, ty.Any] | None = None
):
    """Constructs a dictionary that contains a config that can be used in jinja templating."""
    if config_data is None:
        config_data = type_data

    # Select the following type names
    # TODO: derive this from alg name maybe?
    type_names = [
        "key_type",
        "value_type",
        "data_type",
        "flag_type",
        "input_type",
        "output_type",
        "channels",
        "active_channels",
    ]
    # Only select relevant types
    type_info = {k: type_data[k] for k in type_names if k in type_data}

    # Annotate with extra info for jinja
    type_details = {k: annotate_type(type_info[k], k) for k in type_info}
    type_hash = frozenset(type_info.items())
    return (type_hash, {"type": type_details, "config": type_info | config_data})


def annotate_fp_check(targets):
    """Marks types that need a floating-point guard.

    A type only gets the guard on targets whose tuning data contains both
    floating-point and non-floating-point types for it; otherwise the types of
    the missing kind would match no config at all.
    """
    for configs in targets.values():
        types_by_name: dict[str, list[dict[str, ty.Any]]] = {}
        for c in configs.values():
            for name, t in c["type"].items():
                types_by_name.setdefault(name, []).append(t)
        for types in types_by_name.values():
            is_floating = {
                t["is_floating"]
                for t in types
                if t["type"] not in ("empty_type", "rocprim::empty_type")
            }
            for t in types:
                t["check_fp"] = len(is_floating) > 1


def guard_signature(type_details: dict[str, dict[str, ty.Any]]) -> tuple:
    """Returns what the generated guard checks for the types of a config."""

    def type_signature(t: dict[str, ty.Any]) -> tuple:
        if t["type"] in ("empty_type", "rocprim::empty_type"):
            return ("empty",)
        if t["size"] == 0:
            return ("type", t["type"])
        is_arithmetic = t["is_arithmetic"] if t["check_arithmetic"] else None
        is_floating = t["is_floating"] if t.get("check_fp") else None
        return (is_arithmetic, is_floating, t["size_min"], t["size"])

    return tuple(sorted((name, type_signature(t)) for name, t in type_details.items()))


def resolve_shadowed_configs(targets, new_type_hashes):
    """Makes sure every config of a target has a distinct guard.

    The generated picker returns the first config whose guard matches, so a
    config with the same guard as another one could never be selected.
    Configs whose guards collide get an arithmetic guard on the types where
    they differ, e.g. 'rocprim::int128_t' and a 16-byte 'custom_type'. Configs
    that still collide are dropped; newly tuned configs take precedence over
    existing ones.
    """
    for target_hash, configs in targets.items():
        groups: dict[tuple, list[dict[str, ty.Any]]] = {}
        for config in configs.values():
            groups.setdefault(guard_signature(config["type"]), []).append(config)
        for group in groups.values():
            for name in group[0]["type"]:
                types = [config["type"][name] for config in group]
                if len({t["is_arithmetic"] for t in types}) > 1:
                    for t in types:
                        t["check_arithmetic"] = True

        new = new_type_hashes.get(target_hash, set())
        kept = {}
        dropped = set()
        # Visit newly tuned configs first, so they win over existing ones.
        for type_hash in sorted(configs, key=lambda h: h not in new):
            signature = guard_signature(configs[type_hash]["type"])
            if signature in kept:
                log.warning(
                    f"Dropping config {dict(type_hash)} for {dict(target_hash)}: "
                    f"same guard as {dict(kept[signature])}"
                )
                dropped.add(type_hash)
            else:
                kept[signature] = type_hash
        targets[target_hash] = {h: c for h, c in configs.items() if h not in dropped}


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
    cli.add_argument(
        "--default-tuning",
        choices=["base", "fastest"],
        default="base",
        help="Config for unknown targets: 'base' keeps the algorithm's default "
        "config (*_config_params_base), 'fastest' aliases to the tuned configs "
        "of the lowest tuned architecture.",
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

            # Filter/skip data entries that have no "time" column
            data_filter = [c for c in data["cache"].values() if "time" in c]
            filter_dif = len(data["cache"]) - len(data_filter)
            if filter_dif > 0:
                log.warning(f"Skipping failed compilations: {filter_dif}")

            if data["algo_name"] == "device_merge_sort_block_merge":
                temp = [
                    c
                    for c in data_filter
                    if int(c["block_size_x"]) * int(c["ipt"])
                    <= MERGE_SORT_BLOCK_MERGE_MAX_MERGEPATH_ITEMS
                ]
                if len(temp) < len(data_filter):
                    log.info(
                        f"Dropped {len(data_filter) - len(temp)} constraint-violating configs: {file_path}"
                    )
                data_filter = temp

            if not data_filter:
                log.warning(f"No valid configs left, skipping: {file_path}")
                continue

            # Find best config
            config_data = min(data_filter, key=lambda c: c["time"])
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

    # Remember which configs were newly tuned, they take precedence over existing ones
    new_type_hashes = {
        alg_name: {target_hash: set(target) for target_hash, target in targets.items()}
        for alg_name, targets in algs.items()
    }

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
        annotate_fp_check(targets)
        resolve_shadowed_configs(targets, new_type_hashes[alg_name])

        default_target = None
        if args.default_tuning == "fastest":
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
