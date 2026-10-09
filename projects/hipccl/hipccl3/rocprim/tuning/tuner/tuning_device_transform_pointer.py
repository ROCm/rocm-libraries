#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

import sys
from typing import Any, Callable, Dict, OrderedDict

sys.path.append("../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import TunerArgs
from tuner.tuning_device_transform import TransformTuner

class Tuner(TransformTuner):
    @classmethod
    def _get_default_args(cls) -> TunerArgs:
            return TunerArgs(algo_full_name="device_transform_pointer")

    def _get_tune_params(self, types: Dict[str, Any]) -> OrderedDict:
        """Returns tuning parameters and their possible values as an OrderedDict.
        Each parameter maps to a list of valid values to explore during tuning."""
        params = OrderedDict()
        element_size = TYPE_CONFIGS[types["value_type"]].size
        max_items = min(64 // element_size, 32)
        params["block_size_x"] = list(range(64, 1025, 64))
        params["ipt"] = list(range(1, max_items + 1, 1))
        params["load_type"] = ["::rocprim::load_default", "::rocprim::load_nontemporal"] # cache_load_modifier
        return params

if __name__ == "__main__":
    Tuner.cli()
