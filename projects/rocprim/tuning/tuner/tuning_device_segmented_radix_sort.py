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

sys.path.append(f"{os.path.dirname(__file__)}/../")

from utils import TYPE_CONFIGS, BASE_DIR
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES, COMMON_VALUE_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
RADIX_BITS = [8]
BLOCK_SIZES = [256]
IPT = [4, 8, 16]
WARP_SMALL_LWS = [8]
WARP_SMALL_IPT = [4]
WARP_SMALL_BS = [256]
WARP_PARTITION = [64]
WARP_MEDIUM_LWS = [16]
WARP_MEDIUM_IPT = [8]
WARP_MEDIUM_BS = [256]

class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_segmented_radix_sort')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['radix_bits'] = RADIX_BITS
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        params['warp_small_lws'] = WARP_SMALL_LWS
        params['warp_small_ipt'] = WARP_SMALL_IPT
        params['warp_small_bs'] = WARP_SMALL_BS
        params['warp_partition'] = WARP_PARTITION
        params['warp_medium_lws'] = WARP_MEDIUM_LWS
        params['warp_medium_ipt'] = WARP_MEDIUM_IPT
        params['warp_medium_bs'] = WARP_MEDIUM_BS
        params['warp_partitioning_allowed'] = [1]

        return params

    def _get_restrictions(
        self, key_type: str, val_type: Optional[str] = None
    ) -> Callable[[dict], bool]:

        key_size = TYPE_CONFIGS[key_type].size
        TUNING_SHARED_MAX = 65536

        def validate(params):
            bs = params['block_size_x']
            ipt = params['ipt']

            if not val_type:
                return key_size * bs * ipt < TUNING_SHARED_MAX 
            else:
                val_size = TYPE_CONFIGS[val_type].size
                return (key_size + val_size) * bs * ipt <= TUNING_SHARED_MAX

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""

        VALUE_TYPES = COMMON_VALUE_TYPES + [None]

        for key_type in COMMON_KEY_TYPES:
            for value_type in VALUE_TYPES:
                self.tune_type(key_type, value_type)


if __name__ == "__main__":
    Tuner.cli()
    