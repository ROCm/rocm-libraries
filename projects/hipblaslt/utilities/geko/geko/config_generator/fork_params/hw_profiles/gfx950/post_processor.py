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

from typing import Any, Callable, Dict, List, Optional, Tuple

from geko.config_generator.constants import LIST_OF_MT_MAX_SIZE
from geko.config_generator.mi_designer import MIDesign
from geko.config_generator.fork_params.post_processor import BasePostProcessor, post_process
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)


class GFX950PostProcessor(BasePostProcessor):
    """GFX950 heuristic post-processor.

    Adjusts fork params and MI groups based on MI-derived properties.
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

    @post_process
    def adjust_prefetch_global_read(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """usePGR1: if any MI has MT0<64 and MT1<64, add 1 to PGR."""
        use_pgr1 = False
        for entry in mi_groups:
            mi = entry["MatrixInstruction"].values
            MT0, MT1, *_ = MIDesign.calculate_mfma_parameters(mi)
            use_pgr1 = use_pgr1 or (MT0 < 64 and MT1 < 64)
        if use_pgr1 and "PrefetchGlobalRead" in fork_params:
            pgr = fork_params["PrefetchGlobalRead"]
            if 1 not in pgr.values:
                pgr.values.insert(0, 1)
        return fork_params, mi_groups

    @post_process
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
            mi = entry["MatrixInstruction"].values
            MT0, MT1, *_ = MIDesign.calculate_mfma_parameters(mi)
            use_large = use_large or (K > 1024 and MT0 * MT1 < 64 * 64)
        if use_large and "DepthU" in fork_params:
            du = fork_params["DepthU"]
            du.values.append(2 * du.values[-1])
            du.values.pop(0)
        return fork_params, mi_groups

    @post_process
    def adjust_work_group_mapping(
        self,
        fork_params: Dict[str, ForkParameter],
        mi_groups: GroupDimension,
        ctx: SizeContext,
    ) -> Tuple[Dict[str, ForkParameter], GroupDimension]:
        """useWGM1: replace 16 with 1 if any MI has MT0<=32 and MT1<=32."""
        use_wgm1 = False
        for entry in mi_groups:
            mi = entry["MatrixInstruction"].values
            MT0, MT1, *_ = MIDesign.calculate_mfma_parameters(mi)
            use_wgm1 = use_wgm1 or (MT0 <= 32 and MT1 <= 32)
        if use_wgm1 and "WorkGroupMapping" in fork_params:
            wgm = fork_params["WorkGroupMapping"]
            if 16 in wgm.values:
                wgm.values[wgm.values.index(16)] = 1
        return fork_params, mi_groups

    @post_process
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

    @post_process
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

        MT0, MT1, TT0, TT1, WG0, WG1, MIBlockM = (
            MIDesign.calculate_mfma_parameters(mi_values)
        )
        mi_comment = (
            f"CMS — MT {MT0}x{MT1} - TT {TT0}x{TT1} "
            f"- WG {WG0}x{WG1} - MIBlockM {MIBlockM}"
        )
        
        entry['MatrixInstruction'] = make_param(
            'MatrixInstruction', 
            mi_values, 
            comment=mi_comment,
            metadata={
                "MT": (MT0, MT1),
                "wave": (mi_values[7], mi_values[8]),
                "LSU": 1,
                "GSU": 1, # TODO For now using GSU=1 for CMS kernels
                }
            )

        for key in d.keys():
            if key not in validParameters.keys():
                continue
            entry[key] = make_param(key, [int(d[key])])
        entry['UseCustomMainLoopSchedule'] = make_param(
            'UseCustomMainLoopSchedule', [1])
        groups.append(entry)

    return groups
