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
        return TunerArgs(algo_full_name='device_adjacent_difference')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, types: Dict[str, Any]) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        return params

    def _get_restrictions(self, types: Dict[str, Any]) -> Callable[[dict], bool]:
        element_size = TYPE_CONFIGS[types["value_type"]].size

        # based on legacy tuning 
        MAX_SHARED_MEM = 65536

        def validate(params):
            block_size = params['block_size_x']
            ipt = params['ipt']

            max_ipt = (MAX_SHARED_MEM // (block_size * element_size * 2 )) + element_size

            return ipt < max_ipt

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""
        for value_type in COMMON_KEY_TYPES:
            self.tune_type({"value_type": value_type})


if __name__ == "__main__":
    Tuner.cli()