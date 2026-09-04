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

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""

BLOCK_SIZES = [64, 128, 256, 512, 1024]
IPT = [1, 2, 4, 8, 16]
THRESHOLD = [4, 8, 12, 16]
COUNT_FUNC = ['count_equal_to', 'count_is_percent_of_size']
COUNT = [1, 6, 10, 14, 25, 50, 100]

class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_adjacent_difference')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        params['threshold'] = THRESHOLD
        params['count_func'] = COUNT_FUNC
        params['count'] = COUNT
        return params

    def _get_key_type(self) -> str:
        return "data_type"

    def _get_value_type(self):
        return ""

    def _get_restrictions(
        self, value_type: str, _: Optional[str] = None
    ) -> Callable[[dict], bool]:
        def validate(params):
            count = params['count']
            count_func = params['count_func']

            return (count >= 50 and count_func == 'count_is_percent_of_size') or (count <= 25 and count_func == 'count_equal_to')

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""
        for data_type in COMMON_KEY_TYPES:
            self.tune_type(data_type)


if __name__ == "__main__":
    Tuner.cli()