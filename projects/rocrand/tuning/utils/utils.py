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

import argparse
from typing import Optional
import os

BASE_DIR = f"{os.path.dirname(os.path.abspath(__file__))}/.."

class Parser:
    @staticmethod
    def get_parser(
        default_bytes: Optional[int] = None,
        default_max_feval: int = 100,
        default_output_dir: str = "../output",
        default_strategy: str = "dual_annealing",
    ):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--size",
            type=int,
            default=default_bytes,
            help=f"Size in bytes (default: {default_bytes})",
        )
        parser.add_argument(
            "--max-fevals",
            type=int,
            default=default_max_feval,
            help=f"Maximum number of unique valid function evaluations (default: {default_max_feval})",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=default_output_dir,
            help=f"Output directory for JSON files (default: {default_output_dir})",
        )
        parser.add_argument(
            "--exclude-default-config",
            action="store_true",
            default=False,
            help="Exclude default configs of previous tuning in the new results (default: off)",
        )
        parser.add_argument(
            "--strategy",
            type=str,
            default=default_strategy,
            help=f"Strategy Kernel Tuner will use (default: {default_strategy})",
        )
        parser.add_argument(
            "--simulation-mode",
            action="store_true",
            default=False,
            help="Run strategy in simulation mode (you need the cache files of the full searchspace)",
        )
        parser.add_argument(
            "--arch-name",
            required=False,
            help="Specify arch, needed when running in simulation",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=-1,
            help=f"The initial seed to generate the input data with",
        )
        return parser
