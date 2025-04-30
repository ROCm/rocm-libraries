#!/usr/bin/env python3

import json5 as json #json5 supports comments in json files
import os
import argparse
from jinja2 import Environment, PackageLoader

env = Environment(loader=PackageLoader("generate_config_defaults"))

def load_default_configs_json(in_dir: str) -> dict:
    """
    Reads the config_defaults.json object for rocRAND's generators,
    converts their name to the corresponding rocrand_rng_type and returns them as dict.
    """

    with open(os.path.join(in_dir, "config_defaults.json")) as f:
        json_data = json.load(f)

    def generator_type(generator_name: str) -> str:
        if 'SOBOL' in generator_name:
            return 'QUASI'
        return 'PSEUDO'
    
    def get_rocrand_rng_type(generator_name: str) -> str:
        return "ROCRAND_RNG_" + generator_type(generator_name) + "_" + generator_name

    default_configs = {}
    for generator_name, configs in json_data.items():
        generator_rocrand_enum = get_rocrand_rng_type(generator_name.upper())
        default_configs[generator_rocrand_enum] = configs

    return default_configs

def generate_config_file(out_dir: str, default_configs: dict) -> None:
    """
    Generates config_defaults.hpp from the config_defaults.json file.
    """

    algorithm_template = env.get_template("config_defaults_template")

    with open(os.path.join(out_dir,  f"config_defaults.hpp"), "w") as text_file:
        text_file.write(algorithm_template.render(configs=default_configs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--in-dir',
        default=os.path.dirname(os.path.realpath(__file__)),
        help="Path to the directory that contains 'config_defaults.json',"
             "usually located in 'scripts/config-tuning/'")
    parser.add_argument('--out-dir', default=".")
    args = parser.parse_args()

    default_configs = load_default_configs_json(args.in_dir)
    generate_config_file(args.out_dir, default_configs)
