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

from tuner.base_tuner import BaseTuner, TunerArgs, COMMON_KEY_TYPES, COMMON_VALUE_TYPES

"""
Inclusive range for params tuning, edit these to adjust tuning grid range.
"""
BLOCK_SIZES = [64, 128, 256]
IPT = [1, 2, 4, 8, 16]

class Tuner(BaseTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
        return TunerArgs(algo_full_name='device_binary_search')

    def __init__(self, args: TunerArgs) -> None:
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        params = OrderedDict()
        params['block_size_x'] = BLOCK_SIZES
        params['ipt'] = IPT
        return params

    def _get_key_type(self) -> str:
        return "value_type"

    def _get_value_type(self):
        return "output_type"

    def _get_restrictions(
        self, value_type: str, output_type: Optional[str] = None
    ) -> Callable[[dict], bool]:
        # There litterally was no restrictions in legacy tuner or benchmark
        def validate(params):
            return True

        return validate

    def tune_all(self) -> None:
        """Tune for all value type combinations"""
        for val_type in COMMON_KEY_TYPES:
            for out_type in COMMON_VALUE_TYPES:
                self.tune_type(val_type, out_type)


if __name__ == "__main__":
    Tuner.cli()