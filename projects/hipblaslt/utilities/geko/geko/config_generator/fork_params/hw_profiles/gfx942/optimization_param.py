# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import math
from typing import Any, Dict, List, Optional, Tuple

from geko.config_generator.constants import dataSize, depthURange, LIST_OF_MIN_DIM
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


class GFX942Params(BaseOptimizationParams):
    """GFX942 heuristic profile (CUs = 304).

    Size-dependent logic for all parameters.
    MI groups are external (from MIDesigner) — not handled here.
    """

    # =================================================================
    # GFX942-specific params
    # =================================================================

    @param
    def depth_u(self, ctx: SizeContext) -> ForkParameter:
        dt = self._gt.data_type
        CUs = self.config["CUs"]
        area = ctx.M * ctx.N
        idx = 0
        if area > 256 * 128 * CUs:
            idx = 3
        elif area > 128 * 128 * CUs:
            idx = 2
        elif area > 32 * 32 * CUs:
            idx = 1
        return self._make_param("DepthU", list(depthURange[dt][idx]))

    @param
    def direct_to_vgpr_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("DirectToVgprA", self._dtv_values("A", ctx))

    @param
    def direct_to_vgpr_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("DirectToVgprB", self._dtv_values("B", ctx))

    def _dtv_values(self, a_or_b: str, ctx: SizeContext) -> List[int]:
        transA, transB = self._gt.transA, self._gt.transB
        dsz = dataSize[self._gt.data_type]
        if transA == "N" and transB == "T" and dsz < 4:
            if a_or_b == "B" and ctx.M >= ctx.N and ctx.N < 512:
                return [0]
            if a_or_b == "A" and ctx.M < ctx.N and ctx.M < 512:
                return [0]
            return [0, 1]
        tlu = (a_or_b == "A" and transA == "N") or (a_or_b == "B" and transB == "T")
        if tlu:
            return [0, 1]
        return [0]

    @param
    def global_read_vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param(
            "GlobalReadVectorWidthA",
            self._grvw_values("A", self._gt.transA, ctx.M, ctx),
        )

    @param
    def global_read_vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param(
            "GlobalReadVectorWidthB",
            self._grvw_values("B", self._gt.transB, ctx.N, ctx),
        )

    def _grvw_values(self, a_or_b: str, trans: str, mn_dim: int, ctx: SizeContext) -> List[int]:
        dt = self._gt.data_type
        dsz = dataSize[dt]
        is_grid = self.config.get("SIZE_OPTION", 0) == 1
        max_glvw = 16 // dsz
        tlu = (a_or_b == "A" and trans == "N") or (a_or_b == "B" and trans == "T")

        if tlu:
            min_dim = LIST_OF_MIN_DIM[dt]
            if mn_dim <= min_dim:
                if not is_grid:
                    min_glvw = 1
                    while True:
                        if mn_dim > min_glvw:
                            min_glvw //= 2
                            break
                        elif mn_dim == min_glvw:
                            break
                        min_glvw *= 2
                    return [min_glvw]
                return [max_glvw]
            return [max_glvw]

        if dsz <= 4:
            return [4 // dsz, 16 // dsz]
        if dsz <= 8:
            return [1, 2]
        return [1]

    @param
    def wave_separate_global_read_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param(
            "WaveSeparateGlobalReadA",
            self._wsgr_values("A", self._gt.transA, ctx.M, ctx.K),
        )

    @param
    def wave_separate_global_read_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param(
            "WaveSeparateGlobalReadB",
            self._wsgr_values("B", self._gt.transB, ctx.N, ctx.K),
        )

    def _wsgr_values(self, a_or_b: str, trans: str, mn_dim: int, k_dim: int) -> List[int]:
        dsz = dataSize[self._gt.data_type]
        th = 65536 // dsz
        tlu = (a_or_b == "A" and trans == "N") or (a_or_b == "B" and trans == "T")
        if tlu:
            if mn_dim >= th:
                return [0, 2]
        else:
            if k_dim >= th:
                return [0, 2]
        return [0]

    @param
    def non_temporal(self, ctx: SizeContext) -> ForkParameter:
        """Only one of NonTemporalA / NonTemporalB is used."""
        small = min(ctx.M, ctx.N)
        large = max(ctx.M, ctx.N)
        if small <= 256 and large >= 4000:
            vals = [0, 4]
        else:
            vals = [0]
        name = "NonTemporalB" if ctx.N > ctx.M else "NonTemporalA"
        return self._make_param(name, vals)

    @param
    def work_group_mapping(self, ctx: SizeContext) -> ForkParameter:
        dt = self._gt.data_type
        CUs = self.config["CUs"]
        wgm_unit = self.config.get("WGMUnit", self.config.get("XCC", 8))
        dsz = dataSize[dt]

        wgmunit = wgm_unit if ctx.M <= ctx.N else -wgm_unit
        min_dim = min(ctx.M, ctx.N)
        max_dim = max(ctx.M, ctx.N)
        maxunit = int(math.sqrt(CUs / wgm_unit))
        mt = 256 if dsz <= 4 else 128

        if min_dim > 256 and max_dim >= mt * wgm_unit * 2 * maxunit:
            return self._make_param("WorkGroupMapping", [wgmunit * maxunit])
        if min_dim > 256 and max_dim >= mt * wgm_unit * maxunit and maxunit > 4:
            return self._make_param("WorkGroupMapping", [wgmunit * maxunit])
        if min_dim > 256 and max_dim >= mt * wgm_unit * 4:
            return self._make_param("WorkGroupMapping", [wgmunit * 2, wgmunit * 4])
        if min_dim > 256 and max_dim >= mt * wgm_unit * 2:
            return self._make_param("WorkGroupMapping", [1, wgmunit * 2])
        return self._make_param("WorkGroupMapping", [1])

    @param
    def vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        dsz = dataSize[self._gt.data_type]
        if dsz == 8:
            vals = [1, 2]
        elif dsz <= 4:
            vals = [1, 2, 4]
        else:
            vals = [1]
        return self._make_param("VectorWidthA", vals)

    @param
    def vector_width_b(self, ctx: SizeContext) -> Optional[ForkParameter]:
        dt = self._gt.data_type
        transB = self._gt.transB
        dsz = dataSize[dt]
        if transB != "T":
            return None
        if dt == "C":
            return self._make_param("VectorWidthB", [1])
        if dsz == 8:
            return self._make_param("VectorWidthB", [1, 2])
        if dsz <= 4:
            return self._make_param("VectorWidthB", [1, 2, 4])
        return self._make_param("VectorWidthB", [1])

    @param
    def mi_arch_vgpr(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("MIArchVgpr", [True])

    @param
    def cluster_local_read(self, ctx: SizeContext) -> ForkParameter:
        dsz = dataSize[self._gt.data_type]
        transA, transB = self._gt.transA, self._gt.transB
        if dsz < 4 and not (transA == "T" and transB == "N"):
            return self._make_param("ClusterLocalRead", [1])
        return self._make_param("ClusterLocalRead", [0])

    @param
    def prefetch_local_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchLocalRead", [1])

    @param
    def source_swap(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("SourceSwap", [True])

    @param
    def work_group_mapping_xcc(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WorkGroupMappingXCC", [1])

    @param
    def global_split_u_algorithm(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalSplitUAlgorithm", ["MultipleBuffer"])

    @param
    def local_read_vector_width(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LocalReadVectorWidth", [-1])

    @param
    def store_priority_opt(self, ctx: SizeContext) -> ForkParameter:
        if ctx.M * ctx.N >= 2000 * 2000:
            return self._make_param("StorePriorityOpt", [True])
        return self._make_param("StorePriorityOpt", [False])

    @param
    def store_sync_opt(self, ctx: SizeContext) -> ForkParameter:
        if ctx.M * ctx.N >= 2000 * 2000:
            return self._make_param("StoreSyncOpt", [4])
        return self._make_param("StoreSyncOpt", [0])

    @param
    def stagger_u(self, ctx: SizeContext) -> ForkParameter:
        transA, transB = self._gt.transA, self._gt.transB
        dsz = dataSize[self._gt.data_type]
        if ctx.M * ctx.N < 2000 * 2000:
            return self._make_param("StaggerU", [0])
        if transA == "N" and transB == "T" and dsz <= 4:
            return self._make_param("StaggerU", [32])
        if transA == "N" and transB == "T" and dsz >= 8:
            return self._make_param("StaggerU", [0, 32])
        if transA == "T" and transB == "T" and dsz >= 8:
            return self._make_param("StaggerU", [0, 8])
        return self._make_param("StaggerU", [8])

    @param
    def num_elements_per_batch_store(self, ctx: SizeContext) -> ForkParameter:
        dsz = dataSize[self._gt.data_type]
        if dsz <= 2:
            return self._make_param("NumElementsPerBatchStore", [16])
        return self._make_param("NumElementsPerBatchStore", [32 // dsz])

    @param
    def one_lds_buffer(self, ctx: SizeContext) -> ForkParameter:
        if ctx.M * ctx.N < 256 * 128 * 16 * 16:
            return self._make_param("1LDSBuffer", [0, 1])
        return self._make_param("1LDSBuffer", [1])

    @param
    def stream_k(self, ctx: SizeContext) -> ForkParameter:
        if self.config.get("StreamK", False):
            return self._make_param("StreamK", [3])
        return self._make_param("StreamK", [0])

    @param
    def stream_k_xcc_mapping(self, ctx: SizeContext) -> ForkParameter:
        if self.config.get("StreamK", False):
            return self._make_param("StreamKXCCMapping", [0, 8])
        return self._make_param("StreamKXCCMapping", [0])

    @param
    def use_sgpr_for_gro(self, ctx: SizeContext) -> ForkParameter:
        transA, transB = self._gt.transA, self._gt.transB
        if self.config.get("StreamK", False) and transA == "T" and transB == "N":
            return self._make_param("UseSgprForGRO", [0, 1])
        return self._make_param("UseSgprForGRO", [-1])

    @param
    def use_custom_main_loop_schedule(self, ctx: SizeContext) -> ForkParameter:
        if self.config.get("CMS", False):
            return self._make_param("UseCustomMainLoopSchedule", [-1])
        return self._make_param("UseCustomMainLoopSchedule", [0])

    @param
    def prefetch_global_read(self, ctx: SizeContext) -> ForkParameter:
        transA, transB = self._gt.transA, self._gt.transB
        pgr = [2]
        if transA == "T" and transB == "N" and ctx.K in (8192, 16384):
            pgr = [1, 2]
        return self._make_param("PrefetchGlobalRead", pgr)

    # =================================================================
    # Groups
    # =================================================================

    @group
    def ntcd_group(self, ctx: SizeContext) -> GroupDimension:
        area = ctx.M * ctx.N
        if area >= 2000 * 2000:
            ntcds = [4]
        elif area >= 1000 * 1000 or max(ctx.M, ctx.N) >= 10_000:
            ntcds = [0, 3]
        else:
            ntcds = [0]
        return [
            {
                "NonTemporalC": self._make_param("NonTemporalC", [v]),
                "NonTemporalD": self._make_param("NonTemporalD", [v]),
            }
            for v in ntcds
        ]

    @group
    def clr_ldstri_group(self, ctx: SizeContext) -> GroupDimension:
        dsz = dataSize[self._gt.data_type]
        transA, transB = self._gt.transA, self._gt.transB
        if transA == "T" and transB == "N":
            combos = [(0, False)]
        elif dsz == 2:
            combos = [(0, True), (1, False)]
        elif dsz in (1, 4):
            combos = [(1, False)]
        else:
            raise ValueError(f"CLR/LDSTrI logic not defined for dataSize={dsz}")
        return [
            {
                "ClusterLocalRead": self._make_param("ClusterLocalRead", [clr]),
                "LDSTrInst": self._make_param("LDSTrInst", [ldstri]),
            }
            for clr, ldstri in combos
        ]


class GFX942GAParams(BaseOptimizationParams):
    """GFX942 generic search-space profile.

    Broad exploratory ranges for all parameters.
    Inherits directly from BaseOptimizationParams — generic defines its own
    complete parameter set, independent of heuristic.
    """

    # =================================================================
    # Independent params — static broad ranges
    # =================================================================

    @param
    def depth_u(self, ctx: SizeContext) -> ForkParameter:
        dt = self._gt.data_type
        if dt in ("D", "Z"):
            return self._make_param("DepthU", [8, 16, 32, 64, 128, 256])
        if dt == "C":
            return self._make_param("DepthU", [16, 32, 64, 128, 256, 512])
        return self._make_param("DepthU", [32, 64, 128, 256, 512, 1024])

    @param
    def one_lds_buffer(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("1LDSBuffer", [0, 1])

    @param
    def wave_separate_global_read_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WaveSeparateGlobalReadA", [0, 2])

    @param
    def wave_separate_global_read_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WaveSeparateGlobalReadB", [0, 2])

    @param
    def num_elements_per_batch_store(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NumElementsPerBatchStore", [0, 2, 4, 8, 10, 12, 14, 16])

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

    @param
    def prefetch_global_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchGlobalRead", [1, 2, 3, 4])

    @param
    def prefetch_local_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchLocalRead", [1])

    @param
    def source_swap(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("SourceSwap", [False, True])

    @param
    def stagger_u(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StaggerU", [0, 8, 16])

    @param
    def stagger_u_stride(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StaggerUStride", [64, 128, 256, 512])

    @param
    def store_priority_opt(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StorePriorityOpt", [False, True])

    @param
    def store_sync_opt(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreSyncOpt", [0, 1, 4])

    @param
    def work_group_mapping(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param(
            "WorkGroupMapping",
            [-48, -32, -24, -16, -8, -6, -4, -2, -1, 0, 2, 4, 6, 8, 16, 24, 32, 48],
        )

    @param
    def work_group_mapping_xcc(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WorkGroupMappingXCC", [1, 2, 4, 8, 16])

    @param
    def mi_arch_vgpr(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("MIArchVgpr", [False, True])

    @param
    def global_split_u_algorithm(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalSplitUAlgorithm", ["MultipleBuffer"])

    @param
    def transpose_lds(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TransposeLDS", [-1, 0, 1, 2])

    @param
    def adaptive_gemm(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("AdaptiveGemm", [0, 1])

    @param
    def tailloop_in_nll(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TailloopInNll", [False, True])

    @param
    def extra_mi_latency_left(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ExtraMiLatencyLeft", [-1, 0])

    @param
    def schedule_gr_over_barrier(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ScheduleGROverBarrier", [0, 1])

    @param
    def dtl_plus_lds_buf(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("DtlPlusLdsBuf", [1])

    @param
    def unroll_loop_swap_global_read_order(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("UnrollLoopSwapGlobalReadOrder", [0, 1])

    # =================================================================
    # Computed params — depend on config
    # =================================================================

    @param
    def global_read_vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalReadVectorWidthA", self._compute_grvw())

    @param
    def global_read_vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalReadVectorWidthB", self._compute_grvw())

    def _compute_grvw(self) -> List[int]:
        valid = (1, 2, 3, 4, 6, 8, 16)
        dsz = dataSize[self._gt.data_type]
        min_grvw = max(1, int(4 / dsz))
        max_grvw = min(max(valid), int(16 / dsz))
        return [-1, -2] + list(valid[valid.index(min_grvw):valid.index(max_grvw) + 1])

    @param
    def direct_to_vgpr_a(self, ctx: SizeContext) -> Optional[ForkParameter]:
        """GFX942 only: explore DirectToVgprA when transA is not transposed."""
        if self._gt.transA != "T":
            return self._make_param("DirectToVgprA", [False, True])
        return None

    @param
    def stream_k(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self.config.get("StreamK", False):
            return self._make_param("StreamK", [3])
        return None

    @param
    def stream_k_xcc_mapping(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self.config.get("StreamK", False):
            return self._make_param("StreamKXCCMapping", [0, 4, 8])
        return None

    @param
    def use_plr_pack(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type == "X":
            return self._make_param("UsePLRPack", [0, 1])
        return None

    @param
    def cluster_local_read(self, ctx: SizeContext) -> Optional[ForkParameter]:
        """Independent param only for TN layout; otherwise handled by
        clr_ldstri_group."""
        if self._gt.transA == "T" and self._gt.transB == "N":
            return self._make_param("ClusterLocalRead", [0, 1])
        return None

    # =================================================================
    # Active groups
    # =================================================================

    @group
    def dtl_usfgro_group(self, ctx: SizeContext) -> GroupDimension:
        return [
            {
                "DirectToLds": self._make_param("DirectToLds", [0]),
                "UseSgprForGRO": self._make_param("UseSgprForGRO", [0]),
            },
            {
                "DirectToLds": self._make_param("DirectToLds", [0]),
                "UseSgprForGRO": self._make_param("UseSgprForGRO", [1]),
            },
            {
                "DirectToLds": self._make_param("DirectToLds", [1]),
                "UseSgprForGRO": self._make_param("UseSgprForGRO", [0]),
            },
        ]

    @group
    def clr_ldstri_group(self, ctx: SizeContext) -> Optional[GroupDimension]:
        """CLR + LDSTrInst group for non-TN layouts.
        For TN, cluster_local_read() handles it as an independent param."""
        if self._gt.transA == "T" and self._gt.transB == "N":
            return None
        return [
            {
                "ClusterLocalRead": self._make_param("ClusterLocalRead", [0]),
                "LDSTrInst": self._make_param("LDSTrInst", [True]),
            },
            {
                "ClusterLocalRead": self._make_param("ClusterLocalRead", [1]),
                "LDSTrInst": self._make_param("LDSTrInst", [False]),
            },
        ]

    # =================================================================
    # Inactive groups — no @group decorator, not called by
    # generate_for_size. Available for future activation.
    # =================================================================

    def tailloop_stagger_group(self, ctx: SizeContext) -> GroupDimension:
        """TailloopInNll + StaggerU combinations."""
        return [
            {
                "TailloopInNll": self._make_param("TailloopInNll", [True]),
                "StaggerU": self._make_param("StaggerU", [0]),
            },
            {
                "TailloopInNll": self._make_param("TailloopInNll", [False]),
                "StaggerU": self._make_param("StaggerU", [8]),
            },
            {
                "TailloopInNll": self._make_param("TailloopInNll", [False]),
                "StaggerU": self._make_param("StaggerU", [16]),
            },
        ]

    def extra_latency_dtv_group(self, ctx: SizeContext) -> GroupDimension:
        """ExtraLatencyForLR + DirectToVgprA/B combinations."""
        p = self._make_param

        def _entry(el: int, dtva: bool, dtvb: bool) -> Dict[str, ForkParameter]:
            return {
                "ExtraLatencyForLR": p("ExtraLatencyForLR", [el]),
                "DirectToVgprA": p("DirectToVgprA", [dtva]),
                "DirectToVgprB": p("DirectToVgprB", [dtvb]),
            }

        return [
            _entry(0, True, False),
            _entry(0, False, True),
            _entry(-20, True, False),
            _entry(-20, False, True),
            _entry(-40, True, False),
            _entry(-40, False, True),
            _entry(0, False, False),
        ]
