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
