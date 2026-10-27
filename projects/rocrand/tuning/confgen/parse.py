#!/usr/bin/env python3


def parse_lines(lines: list[str]):

    config = {}

    INSIDE_BLOCK = False
    INSIDE_GRID = False

    for line in lines:
        if "ROCRAND_RNG_PSEUDO" in line and "generator_config_selector" in line:
            engine_name = line.split("<")[-1].split(",")[0].split("_")[-1].lower()

        if "get_threads" in line:
            INSIDE_BLOCK = True
        if "get_blocks" in line:
            INSIDE_BLOCK = False
            INSIDE_GRID = True

        if "target_arch::" in line:
            arch = line.split()[1].split(":")[-2]
            if arch not in config:
                config[arch] = {"arch_name": arch, "engine": engine_name}
            if INSIDE_BLOCK:
                block_size = int(line.split()[-1][:-1])

                config[arch]["tune_param_keys"] = ["block_size_x"]
                config[arch]["tune_params"] = {"block_size_x": block_size}

            elif INSIDE_GRID:
                grid_size = int(line.split()[-1][:-1])

                config[arch]["tune_param_keys"].append("grid_size")
                config[arch]["tune_params"]["grid_size"] = grid_size

    return config
