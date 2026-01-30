#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import importlib.util
import re
import sys
import os
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager
import re
import traceback
from utils import Parser

from tuning_scripts.base_tuner import TunerArgs


@contextmanager
def working_directory(path: Path):
    """Context manager for changing the current working directory."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def import_module_from_file(file_path: str, module_name: str) -> Optional[object]:
    """Import a module from a file path."""
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            print(f"Failed to load spec for {file_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error importing {file_path}: {e}")
        return None


def get_available_algorithms() -> List[str]:
    """Return list of supported tuning algorithms."""
    return [
        "device_merge"
        # Add new algorithms here
    ]


def filter_algorithms(available_algos: List[str], pattern: str) -> List[str]:
    """Filter algorithms based on regex pattern."""
    try:
        regex = re.compile(pattern)
        return [algo for algo in available_algos if regex.search(algo)]
    except re.error as e:
        print(f"Invalid regex pattern: {e}")
        sys.exit(1)


def run_tuning(
    algorithms: List[str],
    size: int = None,
    output_dir: str = None,
    include_default_config: bool = None,
    max_fevals: int = None,
    strategy: str = None,
    simulation_mode: bool = False,
    arch_name: str = None,
    seed: int = None,
) -> None:
    """Run tuning for specified algorithms."""
    current_dir = Path(__file__).parent
    tuning_dir = current_dir / "tuning_scripts"

    with working_directory(tuning_dir):
        for algo in algorithms:
            print(f"\nRunning tuning for {algo}...")
            tuning_file = tuning_dir / f"tuning_{algo}.py"

            if not tuning_file.exists():
                print(f"Error: Tuning file not found for {algo}")
                continue

            module_name = f"tuning_{algo}"
            tuning_module = import_module_from_file(str(tuning_file), module_name)

            if tuning_module is None:
                print(f"Error: Failed to import tuning module for {algo}")
                continue

            print(f"Working directory: {Path.cwd()}")
            try:
                if hasattr(tuning_module, "Tuner"):
                    args: TunerArgs = tuning_module.Tuner._get_default_args()
                    print(f"s: {strategy}")
                    args.update_with_kwargs(
                        algo_full_name=algo,
                        size=size,
                        output_dir=output_dir,
                        include_default_config=include_default_config,
                        max_fevals=max_fevals,
                        strategy=strategy,
                        simulation_mode=simulation_mode,
                        arch_name=arch_name,
                        seed=seed,
                    )
                    tuner = tuning_module.Tuner(args)
                    tuner.tune_all()
                else:
                    print(f"Warning: No tuner class found in {algo} module")
            except Exception:
                # Guarantees that earlier print statements get printed first
                sys.stdout.flush()

                traceback.print_exc()


class RunTuningParser(Parser):
    @classmethod
    def get_run_tuning_parser(cls):
        parser = cls.get_parser()
        parser.description = "Run tuning for algorithms matching the specified regex"
        parser.add_argument(
            "--algo-regex",
            type=str,
            default=".*",
            help='Regex pattern to match algorithm names (default: ".*" matches all)',
        )
        parser.add_argument(
            "--list",
            action="store_true",
            default=False,
            help="List available algorithms",
        )
        return parser


def main():
    available_algos = get_available_algorithms()

    parser = RunTuningParser.get_run_tuning_parser()
    args = parser.parse_args()

    if args.list:
        print("Available algorithms:")
        for algo in available_algos:
            print(f"  - {algo}")
        return

    args.size = int(args.size) if args.size else None

    matched_algos = filter_algorithms(available_algos, args.algo_regex)

    if not matched_algos:
        print(f"No algorithms matched the algo_regex: {args.algo_regex}")
        print("Available algorithms:")
        for algo in available_algos:
            print(f"  - {algo}")
        sys.exit(1)

    print(f"Running tuning for algorithms matching '{args.algo_regex}':")
    for algo in matched_algos:
        print(f"  - {algo}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_tuning(
        matched_algos,
        args.size,
        args.output_dir,
        args.include_default_config,
        args.max_fevals,
        args.strategy,
        args.simulation_mode,
        args.arch_name,
        args.seed,
    )


if __name__ == "__main__":
    main()
