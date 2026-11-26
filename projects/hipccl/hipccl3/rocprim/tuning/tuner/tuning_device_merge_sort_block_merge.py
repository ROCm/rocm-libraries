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

from typing import OrderedDict, Callable, Dict, Any
from math import log2
import sys
import os
import warnings

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES, COMMON_VALUE_TYPES
import confgen.generate
import confgen.parse

# Used when no block_sort config is found for a type.
DEFAULT_SORTED_TILE = 1024

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [128, 256, 512, 1024]
IPT = [2**i for i in range(17)]


class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name="device_merge_sort_block_merge")

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)
        (self.block_sort_configs, self.block_sort_fallback) = self._read_config_header(
            "device_merge_sort_block_sort"
        )

    def _get_sorted_tile(self, types: Dict[str, Any]) -> int:
        """Returns the largest items per block the merge tiles can have: those of the
        block_sort config the library uses for these types on the tuned arch (the
        benchmark pre-sorts with it), and of radix sort's merge sort, which uses the
        same block_merge config (see confgen.generate.get_radix_merge_sort_tile)."""
        radix_tile = confgen.generate.get_radix_merge_sort_tile(
            TYPE_CONFIGS[types["key_type"]].size, TYPE_CONFIGS[types["value_type"]].size
        )
        config = confgen.parse.find_config(
            self.block_sort_configs,
            self.block_sort_fallback,
            self._get_target_info(),
            types,
        )
        if config is None:
            warnings.warn(
                f"No block_sort config found for '{types}', limiting merge tiles to "
                f"{DEFAULT_SORTED_TILE} items"
            )
            return min(DEFAULT_SORTED_TILE, radix_tile)
        return min(int(config["block_size_x"]) * int(config["ipt"]), radix_tile)

    def _get_tune_params(self, types: Dict[str, Any]) -> OrderedDict:
        params = OrderedDict()

        # These are required to match legacy tuner and to ensure migrate.py works correctly
        params["odd_even_block_size"] = [256]
        params["odd_even_ipt"] = [1]
        params["odd_even_size_limit"] = [(1 << 17) + 70000]
        params["partition_block_size"] = [128]
        params["partition_ipt"] = [1]

        params["block_size_x"] = BLOCK_SIZES
        params["ipt"] = IPT

        return params

    def _get_restrictions(self, types: Dict[str, Any]) -> Callable[[dict], bool]:
        key_size = TYPE_CONFIGS[types["key_type"]].size
        val_size = TYPE_CONFIGS[types["value_type"]].size

        max_shared_memory = 65536
        max_size_per_element = key_size + val_size
        sorted_tile = self._get_sorted_tile(types)

        def validate(params):
            block_size = params["block_size_x"]
            ipt = params["ipt"]
            max_ipt = max_shared_memory // (block_size * max_size_per_element)

            max_ipt_exponent = log2(max_ipt) - 1

            # The merge tiles must fit in the sorted tile of block_sort, otherwise the
            # library returns hipErrorAssert (device_merge_sort.hpp).
            odd_even_tile = params["odd_even_block_size"] * params["odd_even_ipt"]
            return (
                ipt <= (2**max_ipt_exponent)
                and block_size * ipt <= sorted_tile
                and odd_even_tile <= sorted_tile
            )

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""

        VALUE_TYPES = COMMON_VALUE_TYPES + ["rocprim::empty_type"]

        for key_type in COMMON_KEY_TYPES:
            for value_type in VALUE_TYPES:
                self.tune_type({"key_type": key_type, "value_type": value_type})


if __name__ == "__main__":
    Tuner.cli()
