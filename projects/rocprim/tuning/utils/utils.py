#!/usr/bin/env python3

# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

from hip import hip
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional


def hip_check(call_result):
    err = call_result[0]
    result = call_result[1:]
    if len(result) == 1:
        result = result[0]
    if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
        raise RuntimeError(str(err))
    return result


class Parser:
    @staticmethod
    def get_parser(
        default_bytes: Optional[int] = None,
        default_iterations: int = 32,
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
            "--iterations",
            type=int,
            default=default_iterations,
            help=f"Number of iterations (default: {default_iterations})",
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
            "--include-default-config",
            action="store_true",
            default=True,
            help="Include default configs of previous tuning in the new results",
        )
        parser.add_argument(
            "--no-include-default-config",
            action="store_false",
            dest="include-default-config",
            help="Do not include default configs",
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
            "--no-simulation-mode",
            action="store_false",
            dest="simulation-mode",
            help="Do not run strategy in simulation mode",
        )
        parser.add_argument(
            "--arch-name",
            type=str,
            required=False,
            help="Specify arch, needed when running in simulation",
        )
        parser.add_argument(
            "--save-metadata",
            action="store_true",
            default=False,
            help="Save Kernel Tuner metadata",
        )
        parser.add_argument(
            "--no-save-metadata",
            action="store_false",
            dest="save-metadata",
            help="Do not save Kernel Tuner metadata",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=-1,
            help=f"The initial seed to generate the input data with",
        )
        return parser


@dataclass
class TypeInfo:
    cpp_type: str
    size: int
    numpy_type: np.dtype


TYPE_CONFIGS = {
    "int64_t": TypeInfo("int64_t", 8, np.int64),
    "int": TypeInfo("int", 4, np.int32),
    "short": TypeInfo("short", 2, np.int16),
    "int8_t": TypeInfo("int8_t", 1, np.int8),
    "double": TypeInfo("double", 8, np.float64),
    "float": TypeInfo("float", 4, np.float32),
    "rocprim::half": TypeInfo("rocprim::half", 2, np.float16),
    "rocprim::int128_t": TypeInfo(
        "rocprim::int128_t", 16, np.dtype([("low", np.int64), ("high", np.int64)])
    ),
    "rocprim::uint128_t": TypeInfo(
        "rocprim::uint128_t", 16, np.dtype([("low", np.uint64), ("high", np.uint64)])
    ),
}
