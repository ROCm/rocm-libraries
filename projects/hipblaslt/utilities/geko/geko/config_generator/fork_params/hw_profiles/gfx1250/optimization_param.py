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

"""gfx1250 (MI45X) optimization-parameter profiles.

gfx1250 is a CDNA4 WMMA architecture (ISA 12,5,0, Wave32).

GFX1250Params (heuristic):
    Parameter values derived from the golden reference tensilelite YAML configs
    (bbs_nn, bbs_nt, bbs_tn, bbs_tn_large, bbs_tn_maf, bbs_tn_batch4096,
     bss_nt, f8_bf16out_tn, f8_tn_maf, f8bf8_bf16out_tn, f8bf8_f32out_tn).
    Non-StreamK (SK=0) by default.

GFX1250GAParams (generic):
    Broader exploratory ranges for GA/Ductile search on gfx1250.

Key differences vs gfx950 (MFMA) profiles:
  * TDMInst (tensor_load_to_lds) — dominant data-movement lever on gfx1250.
  * ScheduleIterAlg=[4] (StinkyTofu) in heuristic.
  * WavefrontSize=[32] (Wave32 only).
  * No CMS/UseCustomMainLoopSchedule (gfx950 MFMA-only).
  * No DirectToLds (absent on gfx12).
  * No MIArchVgpr (auto-forced True under WMMA).
  * PrefetchGlobalRead capped at [1,2] (PGR>=3 needs DirectToLds).
  * Misc gfx1250 specific optimizations - ClusterDim, LDSSegmentInterleave 
"""

from typing import List, Optional

from geko.config_generator.constants import dataSize
from geko.config_generator.fork_params.optimization_param import (
    BaseOptimizationParams,
    param,
    group,
)
from geko.config_generator.shared_utils import (
    ForkParameter,
    GroupDimension,
    SizeContext,
)

_SCALED_LOWP = ("F8", "F8B8", "B8F8")


class GFX1250Params(BaseOptimizationParams):
    """gfx1250 heuristic profile matching golden reference configs.

    Derived from the users/minsukim/mi45x_readiness branch and the golden
    reference tensilelite YAML configs for gfx1250 (bbs_nn, bbs_nt, bbs_tn,
    bbs_tn_large, bbs_tn_maf, bbs_tn_batch4096, bss_nt, f8_bf16out_tn,
    f8_tn_maf, f8bf8_bf16out_tn, f8bf8_f32out_tn).

    Parameter values are conditioned on data type and layout (transpose)
    to reproduce the exact search spaces from the reference YAMLs.
    StreamK=3 mode is activated via config["StreamK"] = True.
    """

    def _is_tn(self) -> bool:
        return self._gt.transA == "T" and self._gt.transB == "N"

    def _is_nn(self) -> bool:
        return self._gt.transA == "N" and self._gt.transB == "N"

    def _is_nt(self) -> bool:
        return self._gt.transA == "N" and self._gt.transB == "T"

    def _sk3(self) -> bool:
        return self.config.get("StreamK", False)

    # =================================================================
    # Tiling / unroll
    # =================================================================

    @param
    def depth_u(self, ctx: SizeContext) -> ForkParameter:
        dt = self._gt.data_type
        if dt in ("H", "B", "X", "X1"):
            # if self._is_nt():
            #     return self._make_param("DepthU", [64, 128])
            # # bbs_tn_batch4096 uses [128,256,512,1024] for high-batch;
            # # standard TN/NN use [128]. Include the wider range so the
            # # heuristic covers both variants.
            # if self._is_tn() and ctx.B > 1:
            #     return self._make_param("DepthU", [128, 256, 512, 1024])
            # return self._make_param("DepthU", [128])
            return self._make_param("DepthU", [64, 128, 256])
        if dt in _SCALED_LOWP:
            return self._make_param("DepthU", [256])
        if dt == "F4":
            return self._make_param("DepthU", [256, 512])
        if dt == "I8":
            return self._make_param("DepthU", [64, 128, 256])
        if dt == "S":
            return self._make_param("DepthU", [16, 32, 64, 128])
        return self._make_param("DepthU", [64, 128, 256])

    @param
    def force_disable_shadow_init(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ForceDisableShadowInit", [True], active=False)

    @param
    def preload_kern_args(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PreloadKernArgs", [True])

    @param
    def compact_loop_store(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("CompactLoopStore", [True])
    
    @param
    def tdm_plus_lds_buff(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMPlusLdsBuf", [-1, 0 , 1])

    # =================================================================
    # gfx1250-exclusive: TDM / cluster / schedule
    # =================================================================

    @param
    def tdm_inst(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMInst", [3])

    @param
    def tdm_iterate_mode(self, ctx: SizeContext) -> Optional[ForkParameter]:
        # if self._is_tn():
        return self._make_param("TDMIterateMode", [0]) # 0, 1, 2, 3
        # return None

    @param
    def init_c_iter_wmma(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("InitCIterWmma", [-1], active=False)

    @param
    def lds_segment_interleave(self, ctx: SizeContext) -> Optional[ForkParameter]:
        # if self._gt.data_type == "F4":
        #     return None
        return self._make_param("LDSSegmentInterleave", [0, 1])

    @param
    def schedule_iter_alg(self, ctx: SizeContext) -> ForkParameter:
        # if self._sk3():
        #     return self._make_param("ScheduleIterAlg", [0, 4]) # 3
        # return self._make_param("ScheduleIterAlg", [4])
        return self._make_param("ScheduleIterAlg", [0, 4])

    @param # not benificial for A0 nodes.
    def cluster_dim(self, ctx: SizeContext) -> ForkParameter:
        if self._sk3():
            return self._make_param("ClusterDim", [[1, 1]], active=False)
        dt = self._gt.data_type
        if self._is_tn():
            if dt in _SCALED_LOWP or ctx.M * ctx.N >= 3072 * 3072:
                return self._make_param("ClusterDim",
                                        [[1, 1], [2, 2], [2, 4], [4, 2], [4, 4]], active=False)
            return self._make_param("ClusterDim", [[1, 1]], active=False)
        if self._is_nn():
            return self._make_param("ClusterDim", [[1, 1], [2, 4], [4, 2], [4, 4]], active=False)
        return self._make_param("ClusterDim", [[1, 1], [2, 2], [2, 4], [4, 2], [4, 4]], active=False)

    @param
    def lds_tr_inst(self, ctx: SizeContext) -> ForkParameter:
        if self._is_tn():
            return self._make_param("LDSTrInst", [False])
        return self._make_param("LDSTrInst", [True])

    # =================================================================
    # Prefetch / scheduling
    # =================================================================

    @param
    def prefetch_global_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchGlobalRead", [2])

    @param
    def prefetch_local_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchLocalRead", [1])

    @param # not benificial for A0 nodes.
    def prefetch_gl2(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._is_tn() and ctx.M * ctx.N >= 3072 * 3072:
            return self._make_param("PrefetchGL2", [0, 1, 2], active=False)
        return None

    @param
    def half_plr(self, ctx: SizeContext) -> ForkParameter:
        # dt = self._gt.data_type
        # if dt in _SCALED_LOWP and self._is_tn() and ctx.M * ctx.N >= 3072 * 3072:
        #     return self._make_param("HalfPLR", [1, 3])
        # if self._is_nn():
        #     return self._make_param("HalfPLR", [0, 1, 3])
        return self._make_param("HalfPLR", [0, 1, 2, 3])

    # =================================================================
    # Global / local read vectorization
    # =================================================================

    @param
    def global_read_vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        if self._is_nn():
            return self._make_param("GlobalReadVectorWidthA", [4], active=False)
        return self._make_param("GlobalReadVectorWidthA", [-1], active=False)

    @param
    def global_read_vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        if self._is_nt() and self._gt.data_type in ("H", "B", "X", "X1"):
            return self._make_param("GlobalReadVectorWidthB", [4], active=False)
        return self._make_param("GlobalReadVectorWidthB", [-1], active=False)

    @param
    def local_read_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LocalReadVectorWidth", [-1], active=False)

    @param
    def vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        dt = self._gt.data_type
        if self._is_tn() and dt == "F4":
            return self._make_param("VectorWidthA", [8], active=False)
        dsz = dataSize.get(dt, 2)
        if self._is_tn() and dsz <= 2:
            return self._make_param("VectorWidthA", [-1, 8], active=False)
        return self._make_param("VectorWidthA", [-1], active=False)

    @param
    def vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("VectorWidthB", [-1], active=False)

    @param
    def store_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreVectorWidth", [-1], active=False)

    # =================================================================
    # K-decomposition
    # =================================================================

    @param
    def global_split_u(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return None
        dt = self._gt.data_type
        dest = self._gt.dest_data_type
        same_type = (dt == dest)
        if self._is_tn() and same_type and dt in ("H", "B"):
            return self._make_param("GlobalSplitU", [1, 2, 4, 8, 16, 32], active=False)
        if self._is_nt() and same_type and dt in ("H", "B"):
            return self._make_param("GlobalSplitU", [1, 2, 4, 8, 16], active=False)
        if self._is_nt():
            return self._make_param("GlobalSplitU", [1, 2, 3, 4], active=False)
        return self._make_param("GlobalSplitU", [1], active=False)

    @param
    def global_split_u_algorithm(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return None
        dt = self._gt.data_type
        dest = self._gt.dest_data_type
        same_type = (dt == dest)
        if (self._is_tn() or self._is_nt()) and same_type and dt in ("H", "B"):
            return self._make_param("GlobalSplitUAlgorithm", ["MultipleBufferSingleKernel"], active=False)
        return self._make_param("GlobalSplitUAlgorithm", ["MultipleBuffer"], active=False)

    # =================================================================
    # Workgroup / addressing
    # =================================================================

    @param
    def wavefront_size(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WavefrontSize", [32], active=False)

    @param
    def work_group_mapping(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WorkGroupMapping", [1], active=False)

    @param
    def use_sgpr_for_gro(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("UseSgprForGRO", [0], active=False)

    # @param # suggested for a specific f4 case. 
    # def source_swap(self, ctx: SizeContext) -> ForkParameter:
    #     return self._make_param("SourceSwap", [True], active=False)

    @param
    def expand_pointer_swap(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ExpandPointerSwap", [False], active=False)

    @param
    def schedule_global_read(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return None
        return self._make_param("ScheduleGlobalRead", [1], active=False)

    @param
    def schedule_local_write(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return None
        return self._make_param("ScheduleLocalWrite", [1], active=False)

    @param
    def use_plr_pack(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type == "X":
            return self._make_param("UsePLRPack", [0, 1], active=False)
        return None

    # =================================================================
    # LDS / stagger / store
    # =================================================================

    @param
    def one_lds_buffer(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("1LDSBuffer", [0])

    @param
    def transpose_lds(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TransposeLDS", [-1], active=False)

    @param
    def lds_pad_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsPadA", [-1])

    @param
    def lds_pad_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsPadB", [-1])

    @param
    def lds_block_size_per_pad_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsBlockSizePerPadA", [-1])

    @param
    def lds_block_size_per_pad_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsBlockSizePerPadB", [-1])

    @param
    def lds_pad_mxsa(self, ctx: SizeContext) -> Optional[ForkParameter]:
        # bbs_tn_maf emits these even for bf16; fp8/fp4 always need them.
        if self._gt.data_type in (*_SCALED_LOWP, "F4") or self._is_tn():
            return self._make_param("LdsPadMXSA", [-1])
        return None

    @param
    def lds_pad_mxsb(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4") or self._is_tn():
            return self._make_param("LdsPadMXSB", [-1])
        return None

    @param
    def lds_block_size_per_pad_mxsa(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4") or self._is_tn():
            return self._make_param("LdsBlockSizePerPadMXSA", [-1])
        return None

    @param
    def lds_block_size_per_pad_mxsb(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4") or self._is_tn():
            return self._make_param("LdsBlockSizePerPadMXSB", [-1])
        return None

    @param
    def stagger_u(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StaggerU", [0], active=False)

    @param
    def store_remap_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreRemapVectorWidth", [0], active=False)

    @param
    def stream_k(self, ctx: SizeContext) -> ForkParameter:
        if self._sk3():
            return self._make_param("StreamK", [3])
        return self._make_param("StreamK", [0])

    # =================================================================
    # StreamK=3-only params (emitted when config StreamK is True)
    # =================================================================

    @param
    def prefetch_across_persistent(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("PrefetchAcrossPersistent", [0, 1], active=False)
        return None

    @param
    def stream_k_force_dp_only(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("StreamKForceDPOnly", [0, 1], active=False)
        return None

    @param
    def use_subtile_impl(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("UseSubtileImpl", [False], active=False)
        return None

class GFX1250GAParams(BaseOptimizationParams):
    """gfx1250 generic (GA/Ductile) search-space profile.

    Derived from the users/minsukim/mi45x_readiness branch (explore_all mode).

    Broad exploratory ranges for all WMMA-valid parameters on gfx1250.
    Uses a correlated @group for StreamK x TDMInst x ClusterDim x SIA x
    UseSubtileImpl x DepthU x LDSTrInst to avoid invalid combinations
    (ClusterDim != [1,1] requires TDMInst=3; NN+TDM needs LDSTrInst=True;
    SK+cluster miscomputes; subtile needs TDM=3 + DepthU>=128).
    """

    def _is_tn(self) -> bool:
        return self._gt.transA == "T" and self._gt.transB == "N"

    def _depthu_values(self) -> List[int]:
        dt = self._gt.data_type
        if dt in ("H", "B", "X", "X1"):
            return [32, 64, 128, 256]
        if dt in _SCALED_LOWP:
            return [64, 128, 256, 512]
        if dt == "F4":
            return [128, 256, 512]
        if dt == "I8":
            return [64, 128, 256]
        if dt == "S":
            return [16, 32, 64, 128]
        return [64, 128, 256]

    def _compute_grvw(self) -> List[int]:
        dsz = dataSize[self._gt.data_type]
        min_grvw = max(1, int(4 / dsz))
        max_grvw = min(16, int(16 / dsz))
        pow2 = [v for v in (1, 2, 4, 8, 16) if min_grvw <= v <= max_grvw]
        return [-1] + pow2

    _CLUSTER_TILE_PROXY = 128
    _CLUSTER_PER_DIM_CAP = 4
    _CLUSTER_MAX_SIZE = 16

    def _analytic_cluster_dims(self, ctx: SizeContext) -> List[List[int]]:
        """Shape-aware ClusterDim candidates from an arithmetic-intensity model.

        Derived from users/minsukim/mi45x_readiness. Selects cluster dims that
        fit the tile grid (Cm <= ceil(M/T), Cn <= ceil(N/T)), biased toward
        square super-tiles (minimizes traffic). Drops size<=2 clusters (too
        little reuse for the barrier cost) and caps at 7 candidates.
        """
        T = self._CLUSTER_TILE_PROXY
        Mg = max(1, (ctx.M + T - 1) // T)
        Ng = max(1, (ctx.N + T - 1) // T)
        cands = []
        for cm in (1, 2, 4):
            for cn in (1, 2, 4):
                s = cm * cn
                if s <= 2:
                    continue
                if s > self._CLUSTER_MAX_SIZE:
                    continue
                if cm > self._CLUSTER_PER_DIM_CAP or cn > self._CLUSTER_PER_DIM_CAP:
                    continue
                if cm > Mg or cn > Ng:
                    continue
                cands.append((cm, cn, s))

        def _score(c):
            cm, cn, s = c
            balance = abs(cm - cn)
            exact = (0 if Mg % cm == 0 else 1) + (0 if Ng % cn == 0 else 1)
            return (balance, exact, -s)

        cands.sort(key=_score)
        dims: List[List[int]] = [[1, 1]]
        for cm, cn, _s in cands:
            if [cm, cn] not in dims:
                dims.append([cm, cn])
        return dims[:7]

    # =================================================================
    # Correlated group: StreamK x TDMInst x ClusterDim x SIA x
    #                   UseSubtileImpl x DepthU x LDSTrInst
    # =================================================================

    @group
    def explore_features_group(self, ctx: SizeContext) -> GroupDimension:
        """Generate valid bundles of correlated execution-shape parameters.

        Constraints enforced:
          - ClusterDim != [1,1] only when TDMInst=3 (cluster barrier codegen)
          - SK=3 forces ClusterDim=[1,1] (SK+cluster miscomputes)
          - SK=3 bundles omit explicit SIA (Tensile default), except SIA=4
            which works with SK=3+TDM=3
          - LDSTrInst correlated with layout+TDM (_ldstr)
          - UseSubtileImpl disabled (wave32 codegen broken)
        """
        p = self._make_param
        du_all = self._depthu_values()
        dt = self._gt.data_type
        tn = self._is_tn()
        tdm_opts = [0, 3] if (tn and dt not in _SCALED_LOWP) else [0]
        all_cd = self._analytic_cluster_dims(ctx)
        cluster_cds = [c for c in all_cd if c != [1, 1]]

        def _ldstr(tdmi: int) -> bool:
            if dt == "F4":
                return True
            if tn:
                return False
            return tdmi != 0

        dims: list = []
        for d in du_all:
            # SK=0, non-subtile: TDMInst x ClusterDim x SIA
            for tdmi in tdm_opts:
                ldstr = _ldstr(tdmi)
                for sia in [0, 4]: #[1, 3]
                    dims.append({
                        "StreamK": p("StreamK", [0]),
                        "TDMInst": p("TDMInst", [tdmi]),
                        "ClusterDim": p("ClusterDim", [[1, 1]]),
                        "ScheduleIterAlg": p("ScheduleIterAlg", [sia]),
                        "UseSubtileImpl": p("UseSubtileImpl", [False]),
                        "LDSTrInst": p("LDSTrInst", [ldstr]),
                        "DepthU": p("DepthU", [d]),
                        "GlobalSplitUAlgorithm": p("GlobalSplitUAlgorithm",
                                                    ["MultipleBuffer", "MultipleBufferSingleKernel"]),
                    })
                if tdmi != 0:
                    for cd in cluster_cds:
                        for sia in [0, 4]: #[1, 3]
                            dims.append({
                                "StreamK": p("StreamK", [0]),
                                "TDMInst": p("TDMInst", [tdmi]),
                                "ClusterDim": p("ClusterDim", [cd]),
                                "ScheduleIterAlg": p("ScheduleIterAlg", [sia]),
                                "UseSubtileImpl": p("UseSubtileImpl", [False]),
                                "LDSTrInst": p("LDSTrInst", [ldstr]),
                                "DepthU": p("DepthU", [d]),
                                "GlobalSplitUAlgorithm": p("GlobalSplitUAlgorithm",
                                                            ["MultipleBuffer", "MultipleBufferSingleKernel"]),
                            })
            # SK=3: ClusterDim=[1,1] forced, no explicit SIA except SIA=4+TDM=3
            for tdmi in tdm_opts:
                dims.append({
                    "StreamK": p("StreamK", [3]),
                    "TDMInst": p("TDMInst", [tdmi]),
                    "ClusterDim": p("ClusterDim", [[1, 1]]),
                    "UseSubtileImpl": p("UseSubtileImpl", [False]),
                    "LDSTrInst": p("LDSTrInst", [_ldstr(tdmi)]),
                    "DepthU": p("DepthU", [d]),
                })
                if tdmi == 3:
                    dims.append({
                        "StreamK": p("StreamK", [3]),
                        "TDMInst": p("TDMInst", [3]),
                        "ClusterDim": p("ClusterDim", [[1, 1]]),
                        "ScheduleIterAlg": p("ScheduleIterAlg", [4]),
                        "UseSubtileImpl": p("UseSubtileImpl", [False]),
                        "LDSTrInst": p("LDSTrInst", [_ldstr(3)]),
                        "DepthU": p("DepthU", [d]),
                    })
        return dims

    # =================================================================
    # Flat params (not owned by the group)
    # =================================================================

    @param
    def force_disable_shadow_init(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ForceDisableShadowInit", [True])

    @param
    def preload_kern_args(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PreloadKernArgs", [True])

    @param
    def compact_loop_store(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("CompactLoopStore", [True, False])

    @param
    def tdm_split(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMSplit", [False, True])

    @param
    def tdm_iterate_mode(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMIterateMode", [0]) # -1, 0, 1, 2, 3

    @param
    def init_c_iter_wmma(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("InitCIterWmma", [-1, 0, 1])

    @param
    def lds_segment_interleave(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type == "F4":
            return None
        return self._make_param("LDSSegmentInterleave", [-1, 0, 1])
    
    @param
    def tdm_plus_lds_buff(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMPlusLdsBuf", [-1, 0 , 1])

    # =================================================================
    # Prefetch / scheduling
    # =================================================================

    @param
    def prefetch_global_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchGlobalRead", [1, 2])

    @param
    def prefetch_local_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchLocalRead", [0, 1])

    @param
    def prefetch_gl2(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchGL2", [0, 1, 2])

    @param
    def half_plr(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("HalfPLR", [0, 1, 2, 3])

    @param
    def prefetch_across_persistent(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchAcrossPersistent", [0, 1])

    @param
    def stream_k_force_dp_only(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StreamKForceDPOnly", [0, 1])

    @param
    def schedule_global_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ScheduleGlobalRead", [0, 1])

    @param
    def schedule_local_write(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ScheduleLocalWrite", [0, 1])

    # =================================================================
    # Global / local read vectorization
    # =================================================================

    @param
    def global_read_vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalReadVectorWidthA", self._compute_grvw())

    @param
    def global_read_vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalReadVectorWidthB", self._compute_grvw())

    @param
    def local_read_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LocalReadVectorWidth", [-1])

    @param
    def vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        dt = self._gt.data_type
        if dt == "F4":
            return self._make_param("VectorWidthA", [8])
        dsz = dataSize[dt]
        if dsz <= 2:
            return self._make_param("VectorWidthA", [-1, 8])
        return self._make_param("VectorWidthA", [-1])

    @param
    def vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("VectorWidthB", [-1])

    @param
    def store_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreVectorWidth", [-1])

    # =================================================================
    # K-decomposition
    # =================================================================

    @param
    def adaptive_gemm(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("AdaptiveGemm", [0, 1])

    @param
    def stream_k_xcc_mapping(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StreamKXCCMapping", [0, 4, 8])

    # =================================================================
    # Workgroup / addressing
    # =================================================================

    @param
    def wavefront_size(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WavefrontSize", [32])

    @param
    def work_group_mapping(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WorkGroupMapping", [1, 2, 4, 8, 16, 32, -1])

    @param
    def work_group_mapping_xcc(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WorkGroupMappingXCC", [1, 2, 4, 8])

    @param
    def use_sgpr_for_gro(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("UseSgprForGRO", [-1])

    @param
    def source_swap(self, ctx: SizeContext) -> ForkParameter:
        if self._gt.data_type == "F4":
            return self._make_param("SourceSwap", [True])
        return self._make_param("SourceSwap", [False, True])

    @param
    def expand_pointer_swap(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ExpandPointerSwap", [False, True])

    @param
    def use_plr_pack(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type == "X":
            return self._make_param("UsePLRPack", [0, 1])
        return None

    # =================================================================
    # LDS / stagger / store
    # =================================================================

    @param
    def one_lds_buffer(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("1LDSBuffer", [0, 1])

    @param
    def transpose_lds(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TransposeLDS", [-1])

    @param
    def lds_pad_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsPadA", [-1])

    @param
    def lds_pad_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsPadB", [-1])

    @param
    def lds_block_size_per_pad_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsBlockSizePerPadA", [-1])

    @param
    def lds_block_size_per_pad_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LdsBlockSizePerPadB", [-1])

    @param
    def lds_pad_mxsa(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4"):
            return self._make_param("LdsPadMXSA", [-1])
        return None

    @param
    def lds_pad_mxsb(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4"):
            return self._make_param("LdsPadMXSB", [-1])
        return None

    @param
    def lds_block_size_per_pad_mxsa(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4"):
            return self._make_param("LdsBlockSizePerPadMXSA", [-1])
        return None

    @param
    def lds_block_size_per_pad_mxsb(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type in (*_SCALED_LOWP, "F4"):
            return self._make_param("LdsBlockSizePerPadMXSB", [-1])
        return None

    @param
    def stagger_u(self, ctx: SizeContext) -> ForkParameter:
        if ctx.M * ctx.N < 2000 * 2000:
            return self._make_param("StaggerU", [0])
        return self._make_param("StaggerU", [0, 8, 16, 32])

    @param
    def store_remap_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreRemapVectorWidth", [-1])

    # =================================================================
    # Non-temporal / cache hints
    # =================================================================

    @param
    def non_temporal_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalA", [0, 4])

    @param
    def non_temporal_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalB", [0, 4])

    @param
    def non_temporal_c(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalC", [0, 4])

    @param
    def non_temporal_d(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalD", [0, 4])

    # =================================================================
    # Store / epilogue
    # =================================================================

    @param
    def store_priority_opt(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StorePriorityOpt", [False, True])

    @param
    def store_sync_opt(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreSyncOpt", [0, 1, 4])

    @param
    def num_elements_per_batch_store(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NumElementsPerBatchStore", [0, 2, 4, 8, 10, 12, 14, 16])

    @param
    def direct_to_vgpr_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("DirectToVgprA", [False, True])

    @param
    def direct_to_vgpr_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("DirectToVgprB", [False, True])

    @param
    def cluster_local_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ClusterLocalRead", [0, 1])

    @param
    def wave_separate_global_read_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WaveSeparateGlobalReadA", [0, 1])

    @param
    def wave_separate_global_read_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WaveSeparateGlobalReadB", [0, 1])

    @param
    def extra_mi_latency_left(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ExtraMiLatencyLeft", [-1, 0])

    @param
    def unroll_loop_swap_global_read_order(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("UnrollLoopSwapGlobalReadOrder", [0, 1])

    @param
    def schedule_gr_over_barrier(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ScheduleGROverBarrier", [0, 1])

    @param
    def stagger_u_stride(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StaggerUStride", [-1, 256, 512])
