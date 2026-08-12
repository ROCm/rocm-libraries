# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from typing import Any, Callable, Dict, List, Optional, Tuple

from geko.config_generator.constants import LIST_OF_MT_MAX_SIZE
from geko.config_generator.mi_designer import MFMA, MIDesign
from geko.config_generator.fork_params.post_processor import BasePostProcessor, mark_post_process
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)
import logging

logger = logging.getLogger("GEKO")


class GFX950PostProcessor(BasePostProcessor):
    """GFX950 heuristic post-processor.

    Adjusts fork params and MI groups based on MI-derived properties.
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
        threshold = LIST_OF_MT_MAX_SIZE[dt] // 3
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            if mfma_params.MT0 * mfma_params.MT1 >= threshold:
                entry["MIArchVgpr"] = self._make_param("MIArchVgpr", [False])
        return fork_params, mi_groups

    @mark_post_process
    def adjust_prefetch_global_read(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """usePGR1: if any MI has MT0<64 and MT1<64, add 1 to PGR."""
        use_pgr1 = False
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            use_pgr1 = use_pgr1 or (mfma_params.MT0 < 64 and mfma_params.MT1 < 64)
        if use_pgr1 and "PrefetchGlobalRead" in fork_params:
            pgr = fork_params["PrefetchGlobalRead"]
            if 1 not in pgr.values:
                pgr.values.insert(0, 1)
        return fork_params, mi_groups

    @mark_post_process
    def adjust_depth_u(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """useLargeDepthU: shift DepthU range if small MT + large K."""
        K = ctx.K
        use_large = False
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            use_large = use_large or (K > 1024 and mfma_params.MT0 * mfma_params.MT1 < 64 * 64)
        if use_large and "DepthU" in fork_params:
            du = fork_params["DepthU"]
            du.values.append(2 * du.values[-1])
            du.values.pop(0)
        return fork_params, mi_groups

    @mark_post_process
    def adjust_work_group_mapping(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """useWGM1: replace 16 with 1 if any MI has MT0<=32 and MT1<=32."""
        use_wgm1 = False
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            use_wgm1 = use_wgm1 or (mfma_params.MT0 <= 32 and mfma_params.MT1 <= 32)
        if use_wgm1 and "WorkGroupMapping" in fork_params:
            wgm = fork_params["WorkGroupMapping"]
            if 16 in wgm.values:
                wgm.values[wgm.values.index(16)] = 1
        return fork_params, mi_groups

    @mark_post_process
    def merge_cms_groups(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Load CMS kernels and prepend to MI groups.
        When CMS is disabled, set UseCustomMainLoopSchedule=0
        NOTE - Legacy code doesn't support CMS for non-GA workflows"""
        if not self.config.get("CMS", False):
            fork_params["UseCustomMainLoopSchedule"] = self._make_param(
                "UseCustomMainLoopSchedule", [0])
            return fork_params, mi_groups

        dt = self._gt.data_type
        transA = self._gt.transA
        transB = self._gt.transB
        MT_DU = self.config.get("MT_DU")

        cms_groups = load_CMS_groups(dt, transA, transB, self._make_param, MT_DU)
        mi_groups = cms_groups + mi_groups

        return fork_params, mi_groups


class GFX950GAPostProcessor(BasePostProcessor):
    """GFX950 GA post-processor.

    Augments MI groups with MIArchVgpr and merges CMS groups.
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
        threshold = LIST_OF_MT_MAX_SIZE[dt] // 3
        for entry in mi_groups:
            mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(entry["MatrixInstruction"].values))
            if mfma_params.MT0 * mfma_params.MT1 >= threshold:
                entry["MIArchVgpr"] = self._make_param("MIArchVgpr", [False])
        return fork_params, mi_groups

    @mark_post_process
    def merge_cms_groups(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """Load CMS kernels and prepend to MI groups.
        When CMS is disabled, set UseCustomMainLoopSchedule=0."""
        if not self.config.get("CMS", False):
            fork_params["UseCustomMainLoopSchedule"] = self._make_param(
                "UseCustomMainLoopSchedule", [0])
            return fork_params, mi_groups

        dt = self._gt.data_type
        transA = self._gt.transA
        transB = self._gt.transB
        MT_DU = self.config.get("MT_DU")

        cms_groups = load_CMS_groups(dt, transA, transB, self._make_param, MT_DU)
        mi_groups = cms_groups + mi_groups

        return fork_params, mi_groups


# =====================================================================
# Helpers — CMS group loading
# =====================================================================

_DTYPE_TO_CMS_QUERY: Dict[str, str] = {
    'b':  '16bit',
    'h':  '16bit',
    'x':  'TF32',
    'f8': '8bit',
}


def _reconstruct_matrix_instruction(d: Dict[str, Any]) -> List[int]:
    """Build the 9-element MatrixInstruction list from query_cms_kernels output."""
    mi, miwg, mt0, mt1 = (d.pop(k) for k in ('MatrixInstruction', 'MIWaveGroup', 'MacroTile0', 'MacroTile1'))
    miBlockM = 1
    matrixInstM = miBlockM * mi[0]
    wt0 = mt0 // (matrixInstM * miwg[0])
    matrixInstN = mi[1] // (miBlockM * mi[3])
    wt1 = mt1 // (matrixInstN * miwg[1])
    return mi + [1, wt0, wt1] + miwg


def load_CMS_groups(
    datatype: str,
    transA: str,
    transB: str,
    make_param: Callable[..., ForkParameter],
    MT_DU: Optional[List] = None,
) -> GroupDimension:
    """Load CMS groups from Tensile's kernel registry, returned as GroupDimension.

    *make_param* must be a bound ``_make_param`` from a ``BaseParamBuilder``
    subclass: it attaches default Tensile metadata comments for ordinary
    parameters. For ``MatrixInstruction``, that default is replaced here with
    a comment derived from ``MIDesign.calculate_mfma_parameters`` (MT, TT, WG,
    MIBlockM), matching the loop below.
    """
    try:
        from Tensile.Components.CustomSchedule import query_cms_kernels
        from Tensile.Common.ValidParameters import validParameters
    except ImportError:
        import sys
        import os
        # Walk up directory tree looking for tensilelite as a sibling directory
        # This works regardless of where the file is moved within the project structure
        current_file = os.path.abspath(__file__)
        current_dir = os.path.dirname(current_file)
        tensilelite_path = None
        
        # Walk up until we find a parent containing 'tensilelite' directory
        while current_dir != os.path.dirname(current_dir):  # not at filesystem root
            candidate_path = os.path.join(current_dir, "tensilelite")
            if os.path.isdir(candidate_path):
                tensilelite_path = candidate_path
                break
            current_dir = os.path.dirname(current_dir)
        
        if tensilelite_path:
            sys.path.insert(0, tensilelite_path)
            from Tensile.Components.CustomSchedule import query_cms_kernels
            from Tensile.Common.ValidParameters import validParameters
        else:
            raise ImportError(
                "Tensile not found. Could not locate tensilelite directory "
                "in parent directories of this file."
            )
    

    cms_dtype = _DTYPE_TO_CMS_QUERY.get(datatype.lower())
    if cms_dtype is None:
        return []

    layout = ('T' if transA == 'T' else 'N') + ('T' if transB == 'T' else 'N')
    kernels = query_cms_kernels(dtype=cms_dtype, layout=layout)

    groups: GroupDimension = []
    for d in kernels:
        if MT_DU is not None and ([d['MacroTile0'], d['MacroTile1'], d['DepthU']] != MT_DU):
            continue
        entry: Dict[str, ForkParameter] = {}
        mi_values = _reconstruct_matrix_instruction(d)
        mfma_params = MIDesign.calculate_mfma_parameters(MFMA.from_list(mi_values))
        mi_comment = (
            f"CMS — MT {mfma_params.MT0}x{mfma_params.MT1} - TT {mfma_params.TT0}x{mfma_params.TT1} "
            f"- WG {mfma_params.WG0}x{mfma_params.WG1} - MIBlockM {mfma_params.MIBlockM}"
        )
        
        entry['MatrixInstruction'] = make_param(
            'MatrixInstruction', 
            mi_values, 
            comment=mi_comment,
            metadata={
                "MT": (mfma_params.MT0, mfma_params.MT1),
                "wave": (mi_values[7], mi_values[8]),
                "LSU": 1,
                "GSU": 1, # TODO For now using GSU=1 for CMS kernels
                }
            )

        for key in d.keys():
            if key not in validParameters.keys():
                continue
            if type(d[key]) != type(validParameters[key][0]):
                try:
                    d[key] = type(validParameters[key][0])(d[key])
                except ValueError as e:   
                    logger.warning(f"Failed to convert {key}={d[key]} "
                                   f"to {type(validParameters[key][0])}: {e}")
            entry[key] = make_param(key, [d[key]])
        entry['UseCustomMainLoopSchedule'] = make_param(
            'UseCustomMainLoopSchedule', [1])
        groups.append(entry)

    return groups
