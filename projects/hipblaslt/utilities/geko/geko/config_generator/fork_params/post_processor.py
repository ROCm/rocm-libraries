# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Any, Dict, List, Tuple

from geko.config_generator.mi_designer import MFMA, MIDesign
from geko.config_generator.fork_params.optimization_param import BaseParamBuilder
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)


def mark_post_process(fn):
    """Mark a method as a post-processing step."""
    fn._is_post_process = True
    return fn


class BasePostProcessor(BaseParamBuilder):
    """Base class for post-processing MI groups and fork params.

    Runs after MIDesigner + OptimizationParams have produced their
    outputs.  Heuristic-only (generic params don't depend on MI properties).

    Decorate methods with @mark_post_process.  Each receives
    (fork_params, mi_groups) and returns the modified pair.
    Discovery uses vars(type(self)) — same pattern as OptimizationParams.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._post_process_methods: List[str] = []
        for name, method in vars(type(self)).items():
            if getattr(method, "_is_post_process", False):
                self._post_process_methods.append(name)

    def apply(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        size: Tuple[int, int, int, int],
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Run MT_DU adjustments (if active), then all @mark_post_process methods.

        *size* is ``(M, N, B, K)``.
        """
        M, N, B, K = size
        ctx = SizeContext(M=M, N=N, B=B, K=K)
        mt_du = self.config.get("MT_DU")
        if mt_du is not None:
            fork_params, mi_groups = self._apply_mt_du(fork_params, mi_groups, mt_du)
        for method_name in self._post_process_methods:
            fork_params, mi_groups = getattr(self, method_name)(fork_params, mi_groups, ctx)
        return fork_params, mi_groups

    # -----------------------------------------------------------------
    # Macrotile / Origami tuning (MT_DU) — HW-agnostic
    # -----------------------------------------------------------------

    def _apply_mt_du(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        mt_du: List,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Macrotile / Origami tuning overrides. HW-agnostic.

        Overrides select params with fixed values and filters MI groups
        to only keep entries matching the specified macro tile (MT0, MT1).
        """
        fixed_MT0, fixed_MT1, fixed_DU = mt_du[0], mt_du[1], mt_du[2]

        overrides = {
            "DepthU": [fixed_DU],
            "WorkGroupMapping": [0],
            "WorkGroupMappingXCC": [-1],
            "NonTemporalA": [0],
            "NonTemporalB": [0],
            "NonTemporalC": [0],
            "NonTemporalD": [0],
            "StreamKXCCMapping": [0],
        }
        for name, values in overrides.items():
            if name in fork_params:
                fork_params[name].values = values
            else:
                fork_params[name] = self._make_param(name, values)

        mi_groups = [
            entry for entry in mi_groups
            if _mi_matches_mt(entry, fixed_MT0, fixed_MT1)
        ]

        return fork_params, mi_groups


def _mi_matches_mt(entry: Dict[str, ForkParameter], fixed_MT0: int, fixed_MT1: int) -> bool:
    """Check if an MI group entry's macro tile matches the fixed MT."""
    mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
    return mfma_params.MT0 == fixed_MT0 and mfma_params.MT1 == fixed_MT1
