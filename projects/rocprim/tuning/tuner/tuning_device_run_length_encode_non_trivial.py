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
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [64, 128, 256, 512, 1024]
IPT = [2 ** i for i in range(3, 7)]
BLOCK_LOAD_FUNCS = ["::rocprim::block_load_method::block_load_vectorize", "::rocprim::block_load_method::block_load_warp_transpose"]

class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_run_length_encode')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        params['block_load_func'] = BLOCK_LOAD_FUNCS
        return params

    def _get_value_type_name(self):
        return ""

    def _get_restrictions(
        self, key_type: str, val_type: Optional[str] = None
    ) -> Callable[[dict], bool]:
        # using OffsetCountPairT = ::rocprim::tuple<unsigned int, unsigned int>; sizeof(OffsetCountPairT)
        offset_size = 8
        key_size = TYPE_CONFIGS[key_type].size
        max_shared_memory = 65536
        #static constexpr unsigned int min_items_per_thread_exponent = 3u;
        min_ipt_exp = 3

        max_size_per_element = max(key_size, offset_size)

        # See device_non_trivial_runs_benchmark_generator in /benchmark/benchmark_device_run_length_encode_non_trivial_runs.hpp
        def validate(params):
            bs = params['block_size_x']
            ipt  = params['ipt'] 
            block_load_func = params['block_load_func']

            max_ipt = max_shared_memory // (bs * max_size_per_element)
            max_ipt_exp = max(log2(max_ipt) // 1, min_ipt_exp) - 1

            is_load_warp_transpose = block_load_func == "::rocprim::block_load_method::block_load_warp_transpose"
            is_warp_load_supp = is_load_warp_transpose and bs == 64

            if not min_ipt_exp <= log2(ipt) <= max_ipt_exp:
                return False

            return not is_load_warp_transpose or is_warp_load_supp

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""

        for key_type in COMMON_KEY_TYPES:
                self.tune_type(key_type)


if __name__ == "__main__":
    Tuner.cli()