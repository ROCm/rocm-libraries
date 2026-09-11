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
import sys
import os
import subprocess

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS, BASE_DIR
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES, COMMON_VALUE_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [128, 256, 512, 1024]
IPT = [1, 4, 6, 8, 12, 16, 18, 22]
RADIX_BITS = [4, 5, 6, 7, 8]
ALGOS = ['block_radix_rank_algorithm::basic', 'block_radix_rank_algorithm::match']

"""
Class to compile and run a tiny probe to check if the params passed in is valid for device radix sort onesweep
"""
class CheckParam:
    def __init__(self):

        self.rocm_include = f"{BASE_DIR}/../rocprim/include"

        self.cache = {}
        self.v_params = {}
        self.code = r"""
        #include <rocprim/device/detail/device_radix_sort.hpp>

        #ifndef TUNING_SHARED_MEMORY_MAX
        #define TUNING_SHARED_MEMORY_MAX 65536u
        #endif

        using sharedmem_storage = typename rocprim::detail::onesweep_iteration_helper<
            PROBE_Key, PROBE_Value, size_t, PROBE_BlockSize, PROBE_ItemsPerThread, PROBE_RadixBits, false,
            PROBE_RadixRankAlgorithm,  rocprim::arch::wavefront::target::size32, rocprim::identity_decomposer,
            rocprim::detail::block_id_wrapper<>>::storage_type;

        static_assert(sizeof(sharedmem_storage) < TUNING_SHARED_MEMORY_MAX, "exceeds LDS");
        """

    def find_max_params(self):
        for bs in BLOCK_SIZES:
            for rb in RADIX_BITS:
                for algo in ALGOS:
                    l, r, out = 0, len(IPT) - 1, -1
                    while l <= r:
                        m = (l + r) // 2
                        if self.check_valid('rocprim::int128_t', 'rocprim::int128_t', bs, IPT[m], rb, f'rocprim::{algo}'):
                            out = m
                            l = m + 1
                        else:
                            r = m - 1

                    if out != -1:
                        self.v_params[(bs, rb, algo)] = IPT[out] 

    def check_valid(self, key, value, bs, ipt, rb, algo) -> bool:
        param = (key, value, bs, ipt, rb, algo)
        if param in self.cache:
            return self.cache[param]

        src = f'{BASE_DIR}/tuner/probe.cpp'
        with open(src, 'w') as f:
            f.write(self.code)

        cmd = [
            'hipcc', '-fsyntax-only',
            f'-I{self.rocm_include}',
            f'-DPROBE_Key={key}', f'-DPROBE_Value={value}',
            f'-DPROBE_BlockSize={bs}', f'-DPROBE_ItemsPerThread={ipt}',
            f'-DPROBE_RadixBits={rb}', f'-DPROBE_RadixRankAlgorithm={algo}',
            src,
        ]
        compiled = subprocess.run(cmd, capture_output=True, text=True)
        os.remove(src)

        valid = compiled.returncode == 0
        if not valid and "exceeds LDS" not in compiled.stderr:
            print(f"[probe] unexpected failure for {param}:\n{compiled.stderr[:500]}")
        self.cache[param] = valid
        return valid

class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_radix_sort_onesweep')

    def __init__(self, args: TunerArgs) -> None:
        self.param_checker = CheckParam()
        self.param_checker.find_max_params()
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        params['sort_block_size_x'] = BLOCK_SIZES
        params['sort_ipt'] = IPT
        params['radix_bits'] = RADIX_BITS
        params['algo'] = ALGOS

        return params

    def _get_restrictions(
        self, key_type: str, val_type: Optional[str] = None
    ) -> Callable[[dict], bool]:
        def validate(params):
            bs, rb, algo = params['block_size_x'], params['radix_bits'], params['algo']
            if bs != params['sort_block_size_x'] or  params['ipt'] != params['sort_ipt']:
                return False
            if (bs, rb, algo) in self.param_checker.v_params:
                return params['ipt'] <= self.param_checker.v_params[(bs, rb, algo)]
            else:
                return False

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""

        VALUE_TYPES = COMMON_VALUE_TYPES + ["rocprim::empty_type"]

        for key_type in COMMON_KEY_TYPES:
            for value_type in VALUE_TYPES:
                self.tune_type(key_type, value_type)


if __name__ == "__main__":
    Tuner.cli()
    