#!/usr/bin/env python3

import json
import logging
import argparse
import enum
import re

import typing_extensions as ty

log = logging.getLogger("confgen.parse")


class parse_state(enum.Enum):
    NONE = (0,)
    TARGET = (1,)
    CONFIG = (2,)


def parse_lines(lines: list[str]):
    COMMENT_PREFIX = "^ *// *"
    TARGET_TAG = "^TARGET: *"
    CONFIG_TAG = "^CONFIG: *"

    target_string = ""
    config_string = ""
    currently_parsing = parse_state.NONE

    num_configs = 0

    configs: dict[frozenset, list[dict[str, ty.Any]]] = {}
    current_target = None
    for line in lines:
        is_comment = re.match(COMMENT_PREFIX, line)
        line = re.sub(COMMENT_PREFIX, "", line)

        # Check if the line we're parsing is a tag we need to process.
        if re.match(TARGET_TAG, line):
            currently_parsing = parse_state.TARGET
            line = re.sub(TARGET_TAG, "", line)

        if re.match(CONFIG_TAG, line):
            currently_parsing = parse_state.CONFIG
            line = re.sub(CONFIG_TAG, "", line)

        if is_comment:
            # If we're a comment, and we're parsing something, then we
            # append it to the state's string.
            if currently_parsing == parse_state.TARGET:
                target_string += line.strip()
            if currently_parsing == parse_state.CONFIG:
                config_string += line.strip()
        else:
            # If it's not a comment, we must've exitted any parsing state.
            if currently_parsing == parse_state.TARGET:
                # Parse as JSON and use as immutable key.
                try:
                    current_target = frozenset(
                        json.loads(re.sub("'", '"', target_string)).items()
                    )
                except json.JSONDecodeError as e:
                    log.error(f"Error parsing target from: {target_string}")
                    raise e
                configs[current_target] = []

                # Reset parser state.
                currently_parsing = parse_state.NONE
                target_string = ""

            if currently_parsing == parse_state.CONFIG:
                # Parse as JSON and append.
                current_config = json.loads(re.sub("'", '"', config_string))
                try:
                    current_config = json.loads(re.sub("'", '"', config_string))
                except json.JSONDecodeError as e:
                    log.error(f"Error parsing config from: {config_string}")
                    raise e
                if current_target is not None:
                    configs[current_target].append(current_config)
                    num_configs += 1

                # Reset parser state.
                currently_parsing = parse_state.NONE
                config_string = ""
        pass
    log.debug(f"Parsed {num_configs} configs.")

    # Skip fallback case. Much easier to delete it from the
    # dict instead of detecting it during parsing.
    fallback_target = frozenset(
        {("rep", "amdgcn"), ("gpu", "generic"), ("gen", "unknown"), ("arch", "unknown")}
    )
    if fallback_target in configs:
        del configs[fallback_target]

    return configs


def parse_fallback_target(header: str) -> frozenset | None:
    """Returns the target that the generic (unknown) target of a config header forwards
    to, or None if it returns the algorithm's base config instead."""
    start = header.find("// TARGET: {'gen': 'unknown'")
    if start < 0:
        return None
    # Only look at the return statement of the generic target's config picker.
    return_start = header.find("return", start)
    if return_start < 0:
        return None
    return_end = header.find(";", return_start)
    picker = header[return_start : return_end if return_end >= 0 else len(header)]
    pattern = r"comp_target<\s*gen::(\w+),\s*target_arch::(\w+),\s*gpu::(\w+),\s*rep::(\w+)\s*>"
    for gen, arch, gpu, rep in re.findall(pattern, picker):
        if arch != "unknown":
            return frozenset({("gen", gen), ("arch", arch), ("gpu", gpu), ("rep", rep)})
    return None


def select_target(targets, target: dict[str, str]) -> frozenset | None:
    """Returns the target among 'targets' (keys of parse_lines()) that
    most_common_config() in config_types.hpp picks for 'target': an exact GPU match
    first, then the arch, then the generation. Returns None if none matches, in which
    case the library uses the generic target."""
    best_score = 0
    best = None
    for candidate in targets:
        c = dict(candidate)
        if c["gpu"] != "generic" and c["gpu"] == target.get("gpu"):
            score = 3
        elif c["arch"] != "unknown" and c["arch"] == target.get("arch"):
            score = 2
        elif c["gen"] != "unknown" and c["gen"] == target.get("gen"):
            score = 1
        else:
            score = 0
        # Like the library, strictly better wins and on ties the later entry wins.
        if score > 0 and score >= best_score:
            best_score = score
            best = candidate
    return best


def parse_gpu_names(config_types: str) -> list[tuple[str, str]]:
    """Returns the (device name substring, gpu enum) pairs of target_gpu_names in
    config_types.hpp, which the library uses to detect the GPU."""
    return re.findall(
        r'make_tuple<std::string_view, gpu>\("([^"]+)",\s*gpu::(\w+)\)', config_types
    )


def gpu_from_device_name(name: str, gpu_names: list[tuple[str, str]]) -> str:
    """Returns the gpu enum for a device name, like get_target_gpu_from_name()."""
    for substring, gpu in gpu_names:
        if substring in name:
            return gpu
    return "generic"


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--input", "-i", required=True)

    args = cli.parse_args()

    with open(args.input) as file:
        lines = file.readlines()
        configs = parse_lines(lines)
    print(configs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
