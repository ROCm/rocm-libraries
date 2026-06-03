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

from typing import Any, Dict, List, Tuple

from geko.config_generator.constants import LIST_OF_MT_MAX_SIZE
from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.fork_params.post_processor import BasePostProcessor, post_process
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)


class GFX942PostProcessor(BasePostProcessor):
    """GFX942 heuristic post-processor.

    Only augment_mi_arch_vgpr applies for GFX942.
    adjust_depth_u and adjust_work_group_mapping are GFX950-only
    (legacy code guards them with CUs==256).
    """

    @post_process
    def augment_mi_arch_vgpr(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Add MIArchVgpr=0 to MI entries with large macro tiles."""
        dt = self._gt.data_type
        threshold = LIST_OF_MT_MAX_SIZE[dt] // 3
        for entry in mi_groups:
            mi = entry["MatrixInstruction"].values
            MT0, MT1, *_ = MIDesign.calculate_mfma_parameters(mi)
            if MT0 * MT1 >= threshold:
                entry["MIArchVgpr"] = self._make_param("MIArchVgpr", [0])
        return fork_params, mi_groups


class GFX942GAPostProcessor(BasePostProcessor):
    """GFX942 GA post-processor.

    No CMS support on GFX942.
    """

    @post_process
    def augment_mi_arch_vgpr(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Add MIArchVgpr=0 to MI entries with large macro tiles."""
        dt = self._gt.data_type
        threshold = LIST_OF_MT_MAX_SIZE[dt] // 3
        for entry in mi_groups:
            mi = entry["MatrixInstruction"].values
            MT0, MT1, *_ = MIDesign.calculate_mfma_parameters(mi)
            if MT0 * MT1 >= threshold:
                entry["MIArchVgpr"] = self._make_param("MIArchVgpr", [0])
        return fork_params, mi_groups
