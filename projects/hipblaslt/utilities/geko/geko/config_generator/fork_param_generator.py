################################################################################
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from typing import Any, Dict, List, Optional, Tuple

from geko.config_generator.fork_params.post_processor import BasePostProcessor
from geko.config_generator.utils import count_kernels
from geko.config_generator.shared_utils import ForkParameter


def generate_fork_params(
    mi_designer,
    opt_params,
    config: Dict[str, Any],
    size: Tuple[int, int, int, int],
    post_processor: Optional[BasePostProcessor] = None,
) -> Tuple[Dict[str, ForkParameter], int, int]:
    """Generate fork parameters for a single size.

    *size* is ``(M, N, B, K)`` (same convention as :class:`~geko.config_generator.shared_utils.SizeContext`).

    Combines MIDesigner + OptimizationParams, applies post-processing,
    assembles groups.
    Returns (fork_params, num_mis, nkernels).
    """
    mi_groups = mi_designer.generate_for_size(size)
    fork_params, opt_groups = opt_params.generate_for_size(size)

    if post_processor is not None:
        fork_params, mi_groups = post_processor.apply(fork_params, mi_groups, size)

    all_groups = [mi_groups] + opt_groups
    fork_params["Groups"] = ForkParameter(name="Groups", values=all_groups, active=True)

    nkernels = count_kernels(fork_params)

    return fork_params, len(mi_groups), nkernels
