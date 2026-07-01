# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

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
