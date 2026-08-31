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

from typing import List, Optional, OrderedDict, Callable
import sys

sys.path.append("../")

from utils import TYPE_CONFIGS
from tuner.base_tuner import BaseTuner, TunerArgs


class TunerSelect(BaseTuner):
    def __init__(self, args: TunerArgs):
        super().__init__(args)

    def _get_tune_params(self, key_type: str, value_type: Optional[str] = None) -> OrderedDict:
        """Returns tuning parameters and their possible values as an OrderedDict.
        Each parameter maps to a list of valid values to explore during tuning."""
        params = OrderedDict()
        element_size = TYPE_CONFIGS[key_type].size
        max_items = min(64 // element_size, 32)
        params["block_size_x"] = list(range(128, 513, 64))
        params["ipt"] = list(range(4, max_items + 1, 1))
        return params

    def _get_restrictions(
        self, key_type: str, value_type: Optional[str] = None
    ) -> Callable[[dict], bool]:
        """Constraints for what parameter combinations are valid during tuning"""

        # No constraints needed al handled in the parameters.
        def validate(params):
            return True

        return validate

    def _get_key_type(self) -> str:
        return "data_type"

