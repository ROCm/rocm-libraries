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

from typing import List, Optional, OrderedDict
import sys

sys.path.append("../")

from utils import TYPE_CONFIGS
from tuning_scripts.base_tuner import BaseTuner, TunerArgs

KEY_TYPES = ["rocprim::int128_t", "int64_t", "int", "short", "int8_t", "double", "float", "rocprim::half"]
VALUE_TYPES = ["rocprim::int128_t", "int64_t", "int", "short", "int8_t"]

class Tuner(BaseTuner):
    def _get_default_args() -> TunerArgs:
        return TunerArgs(
            algo_full_name="device_merge"
        )

    def __init__(self, args: TunerArgs):
        super().__init__(args)

    def _get_tune_params(self) -> OrderedDict:
        """Returns tuning parameters and their possible values as an OrderedDict.
        Each parameter maps to a list of valid values to explore during tuning."""
        params = OrderedDict()
        params["block_size_x"] = [64 * i for i in range(1, 17)]
        params["ipt"] = [1, 2] + [4 * i for i in range(1, 65)]
        return params

    def _get_restrictions(
        self, key_type: str, value_type: Optional[str] = None
    ) -> List[str]:
        """Constraints for what parameter combinations are valid during tuning"""
        size = self.bytes_size // TYPE_CONFIGS[key_type].size
        element_size = TYPE_CONFIGS[key_type].size
        if value_type:
            element_size += TYPE_CONFIGS[value_type].size

        def validate(params):
            block_size = params['block_size_x']
            ipt = params['ipt']

            # Total size constraint
            if block_size * ipt > size:
                return False

            # Memory size constraint
            if block_size * ipt * element_size > 65536:
                return False

            # Block size constraint
            if block_size > 1024:
                return False

            # Items per thread constraint - high ipts don't perform well
            if ipt >= block_size:
                return False

            # High ipts on gfx1030 cause HSA_STATUS_ERROR_INVALID_ISA
            if params.get('arch_name') == 'gfx1030' and ipt > 28:
                return False

            return True

        return validate

    def tune_all(self) -> None:
        """Tune for all key type and value type combinations"""
        for key_type in KEY_TYPES:
            self.tune_type(key_type)
            for value_type in VALUE_TYPES:
                self.tune_type(key_type, value_type)

if __name__ == "__main__":
    Tuner.cli()
