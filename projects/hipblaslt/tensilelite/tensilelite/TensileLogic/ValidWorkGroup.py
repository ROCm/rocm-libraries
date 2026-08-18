################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

"""
ValidWorkGroup
---
Dimensions of the workgroup which will operate on a tile and share lds
Example: ( wg0 x wg1 x LocalSplitU )
"""

from pathlib import Path

from Tensile.Common import elineno
from Tensile.SolutionStructs.Validators.WorkGroup import validateWorkGroup


def _validateWorkGroup(solution: dict, filepath: Path):
    try:
        validateWorkGroup(solution)
        assert solution["Valid"], f"Solution was rejected: {elineno()}"
        return True
    except AssertionError as e:
        print(
            f"Error: Validation failed: {e} (file: {filepath}, index: {solution['SolutionIndex']})"
        )
        return False
