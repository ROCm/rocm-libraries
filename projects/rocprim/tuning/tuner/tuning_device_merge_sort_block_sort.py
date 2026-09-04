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

from typing import Optional, OrderedDict, Callable
from math import log2
import sys
import os

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES, COMMON_VALUE_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [256, 512, 1024]
IPT = [2 ** i for i in range(17)]


class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_merge_sort_block_sort')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        return params

    def _get_restrictions(
        self, key_type: str, val_type: Optional[str] = None
    ) -> Callable[[dict], bool]:
        key_size = TYPE_CONFIGS[key_type].size
        val_size = 1 if val_type == "rocprim::empty_type"  else TYPE_CONFIGS[val_type].size

        max_shared_memory = 65536
        # legacy tuner: std::max(sizeof(Key) + sizeof(unsigned int), sizeof(Value))
        max_size_per_element = max(key_size + 4, val_size)
        def validate(params):
            block_size = params['block_size_x']
            ipt = params['ipt']
            max_ipt = max_shared_memory // (block_size * max_size_per_element)

            max_ipt_exponent = log2(max_ipt) - 1
            min_ipt_exponent = log2(1024 // block_size)

            return (2 ** min_ipt_exponent) <= ipt <= (2 ** max_ipt_exponent)


        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""

        VALUE_TYPES = COMMON_VALUE_TYPES + ["rocprim::empty_type"]

        for key_type in COMMON_KEY_TYPES:
            for value_type in VALUE_TYPES:
                self.tune_type(key_type, value_type)


if __name__ == "__main__":
    Tuner.cli()