# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Tuple

from geko.config_generator.constants import get_list_of_mt_max_size
from geko.config_generator.mi_designer import MFMA, MIDesign
from geko.config_generator.fork_params.post_processor import BasePostProcessor, mark_post_process
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

    @mark_post_process
    def augment_mi_arch_vgpr(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Add MIArchVgpr=False to MI entries with large macro tiles."""
        dt = self._gt.data_type
        threshold = get_list_of_mt_max_size(self.config.get("search_space"))[dt] // 3
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            if mfma_params.MT0 * mfma_params.MT1 >= threshold:
                entry["MIArchVgpr"] = self._make_param("MIArchVgpr", [False])
        return fork_params, mi_groups


class GFX942GAPostProcessor(BasePostProcessor):
    """GFX942 generic search-space post-processor.

    No CMS support on GFX942.
    """

    @mark_post_process
    def augment_mi_arch_vgpr(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Add MIArchVgpr=False to MI entries with large macro tiles."""
        dt = self._gt.data_type
        threshold = get_list_of_mt_max_size(self.config.get("search_space"))[dt] // 3
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            if mfma_params.MT0 * mfma_params.MT1 >= threshold:
                entry["MIArchVgpr"] = self._make_param("MIArchVgpr", [False])
        return fork_params, mi_groups
