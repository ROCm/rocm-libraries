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

from typing import Optional, OrderedDict, Callable, Dict, Any, List
import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
IPT = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]


class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        # Match benchmark_device_adjacent_difference.cpp (2 GiB).
        return TunerArgs(algo_full_name="device_adjacent_difference", size=2 * 1024**3)

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_compiler_options(self) -> List[str]:
        # benchmark_device_adjacent_difference is built at -O1 to work around
        # ROCm/llvm-project#2616 (see benchmark/CMakeLists.txt), so tune at -O1 too.
        # Remove this override once the benchmark is back at -O3.
        return [
            "-O1" if option == "-O3" else option
            for option in super()._get_compiler_options()
        ]

    def _get_tune_params(
        self, key_type: str, value_type: Optional[str] = None
    ) -> OrderedDict:
        params = OrderedDict()
        params["block_size_x"] = BLOCK_SIZES
        params["ipt"] = IPT
        return params

    def _get_restrictions(self, types: Dict[str, Any]) -> Callable[[dict], bool]:

        element_size = TYPE_CONFIGS[types["value_type"]].size
        # Unknown targets build the configs of every target, so they must fit in 64 KiB
        # even though gfx950 has more.
        MAX_SHARED_MEM = 65536
        LDS_BANKS = 32

        def validate(params):
            block_size = params["block_size_x"]
            ipt = params["ipt"]
            items_per_block = block_size * ipt

            # The largest member of the kernel's shared memory union is the transpose
            # buffer of block_load/block_store, which block_exchange pads for
            # power-of-two items per thread.
            padding = (
                items_per_block // LDS_BANKS if ipt >= 2 and ipt & (ipt - 1) == 0 else 0
            )
            return (items_per_block + padding) * element_size <= MAX_SHARED_MEM

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""
        for val_type in COMMON_KEY_TYPES:
            self.tune_type({"value_type": val_type})


if __name__ == "__main__":
    Tuner.cli()
