# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Dict

from geko.config_generator.shared_utils import ForkParameter


def count_kernels(params: Dict[str, ForkParameter]) -> int:
    """Compute total kernel count from a merged param dict.

    The count is the product of the number of values for every active
    non-Groups parameter, times the product of the number of entries in
    each group dimension.
    """
    n = 1
    for name, fp in params.items():
        if name == "Groups":
            for grp_dim in fp.values:
                if grp_dim:
                    n *= len(grp_dim)
        elif fp.active and isinstance(fp.values, list):
            n *= len(fp.values)
    return n
