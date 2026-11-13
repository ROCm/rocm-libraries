# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx1250 (MI45X) optimization-parameter profiles.

gfx1250 is a CDNA4 WMMA architecture (ISA 12,5,0, Wave32).

GFX1250Params (heuristic):
    Non-persistent (TileProcessingStrategy None) by default.

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

from typing import Optional

from geko.config_generator.constants import HARDWARE_MAP, dataSize, mx_format
from geko.config_generator.fork_params.hw_profiles.gfx1250.cluster_dim import (
    hardware_cluster_dims,
    wgps_per_shader_engine,
)
from geko.config_generator.fork_params.optimization_param import (
    BaseOptimizationParams,
    group,
    param,
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
    Persistent StreamK (on the StaticGrid assignment) is activated via
    config["StreamK"] = True.
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
    def tile_processing_strategy(self, ctx: SizeContext) -> ForkParameter:
        if self._sk3():
            return self._make_param("TileProcessingStrategy", ["StreamK"])
        return self._make_param("TileProcessingStrategy", ["None"])

    @param
    def work_assignment(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("WorkAssignment", ["StaticGrid"])
        return None

    # =================================================================
    # StreamK-only params (emitted when config StreamK is True)
    # =================================================================

    @param
    def prefetch_across_persistent(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("PrefetchAcrossPersistent", [0, 1], active=False)
        return None

    @param
    def use_subtile_impl(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("UseSubtileImpl", [False], active=False)
        return None


class GFX1250GAOrigamiPolicy:
    """Persistent / origami decisions for gfx1250 generic mode.

    Shared by GFX1250GAParams and GFX1250GAPostProcessor: the profile emits the
    WGM / StaggerU axes and the post-processor's MT_DU overrides pin them, so
    both have to agree on when the runtime picks them.
    """

    def _sk3(self) -> bool:
        """Whether the input config's ``StreamK: true`` puts persistent kernels in the space."""
        return bool(self.config.get("StreamK", False))

    def _origami_picks_wgm(self) -> bool:
        """Whether the runtime, not this search, decides WGM and StaggerU.

        ContractionSolution takes the Origami path under exactly

            isPersistent() && skgrid != 0
            && workGroupMapping == 0 && workGroupMappingXCC == -1

        and then calls ``origami::select_workgroup_mapping`` and
        ``origami::select_staggerU`` per problem shape. Both sentinels have to
        be present or neither override fires, and StaggerU is replaced wholesale
        (the else branch is ``defaultStaggerU = sizeMapping.staggerU``), so a
        tuned value would simply be discarded at run time.

        That is what an OOB library wants: it is selected for shapes it was
        never tuned at, and these axes are shape-dependent. An Equality library
        only runs on its tuned shape, so it should keep searching them.

        Requires a persistent strategy, since ``WorkGroupMappingXCC: -1`` is
        rejected without it ("Auto WGMXCC requires persistent execution").
        """
        is_oob = str(self.config.get("LIBRARY_TYPE", "OOB")).lower() != "equality"
        return is_oob and self._sk3()


class GFX1250GAParams(GFX1250GAOrigamiPolicy, BaseOptimizationParams):
    """gfx1250 generic (GA / Ductile) search space.

    Two tiers, in priority order:

    1. **Pinned by the gfx1250 recommended configuration.** TDM on both tensors
       with its companions (PLR 1, PGR 2, 1LDSBuffer 0), SIA 4 (StinkyTofu),
       LDS transpose on, kernel-arg preload on, cluster launch and GL2 prefetch
       off on A0, and the LdsPad* / LdsBlockSizePerPad* family left at -1 so
       codegen derives padding and block size. These are requirements of the
       codegen path, not tuning knobs.
    2. **Searched axes**: everything gfx950 generic sweeps that gfx1250 supports,
       plus the gfx1250-exclusive knobs.

    Dropped from the gfx950 generic set, with reason (gfx1250 asmCaps:
    ``HasDirectToLds=0``, ``HasMFMA=0``, ``HasWMMA=1``):

    * ``DirectToLds`` / ``DtlPlusLdsBuf`` -- ``isDirectToLdsDoable`` rejects with
      "DirectToLds not supported on ISA" whenever ``HasDirectToLds`` is unset.
      gfx950's ``dtl_usfgro_group`` therefore collapses to its DTL=0 entries and
      ``UseSgprForGRO`` becomes an independent param.
    * ``MIArchVgpr`` -- force-set True whenever ``EnableMatrixInstruction and
      HasWMMA``, so sweeping it spends a dimension on a value codegen overwrites.
    * ``PrefetchGlobalRead`` 3 and 4 -- "PrefetchGlobalRead>=3 Supports only
      DirectToLdsA and DirectToLdsB".
    """

    # -----------------------------------------------------------------
    # Helpers (this class does not inherit the heuristic profile)
    # -----------------------------------------------------------------

    def _is_tn(self) -> bool:
        return self._gt.transA == "T" and self._gt.transB == "N"

    def _mx_block(self) -> int:
        """MX block-scaling block size (0 when not MX), from the same
        ``mx_format`` as ``ConfigSectionGenerator`` so the two cannot drift."""
        mx = mx_format(self.config["GemmProblem"], self.config["ARCH"])
        return mx[0] if mx else 0

    # =================================================================
    # 1. Pinned: the gfx1250 recommended configuration
    # =================================================================

    @param
    def wavefront_size(self, ctx: SizeContext) -> ForkParameter:
        # gfx1250 is Wave32 only; stated rather than left to -1.
        return self._make_param("WavefrontSize", [32])

    @param
    def tdm_inst(self, ctx: SizeContext) -> ForkParameter:
        """Pinned [3] -- TDM on both A and B.

        1 and 2 are rejected outright ("Currently TDMA and TDMB must be enabled
        simultaneously"). MX scales additionally require the TDM transport on this
        arch: with TDMInst=0 the scale format resolves to NoSwizzle, and
        "MXScaleFormat=NoSwizzle is not supported on gfx1250".
        """
        return self._make_param("TDMInst", [3])

    @param
    def prefetch_local_read(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchLocalRead", [1])

    @param
    def prefetch_global_read(self, ctx: SizeContext) -> ForkParameter:
        # PGR >= 3 needs DirectToLds, absent on gfx1250.
        return self._make_param("PrefetchGlobalRead", [2])

    @param
    def prefetch_global_read_a(self, ctx: SizeContext) -> ForkParameter:
        # -1: Tensile takes the largest (A, B) pair up to PrefetchGlobalRead that
        # fits in LDS. The A and B keys must be set together or not at all.
        return self._make_param("PrefetchGlobalReadA", [-1])

    @param
    def prefetch_global_read_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PrefetchGlobalReadB", [-1])

    @param
    def expand_pointer_swap(self, ctx: SizeContext) -> ForkParameter:
        # Tensile defaults this on and forces it off only at PGR >= 2, while
        # LDSTrInst with 1LDSBuffer 0 rejects it. A (2, 1) / (1, 2) prefetch pair
        # runs at scalar PGR 1, so without this pin every such pair is rejected.
        return self._make_param("ExpandPointerSwap", [False])

    @param
    def one_lds_buffer(self, ctx: SizeContext) -> ForkParameter:
        # Must be 0 here: wave-separated TDM rejects the (1LDSBuffer=1, PGR=2)
        # pair with "TDM requires at least 2 LDS buffers for PGR2".
        return self._make_param("1LDSBuffer", [0])

    @param
    def tdm_iterate_mode(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMIterateMode", [0, 1, 2, 3])

    @param
    def tdm_split(self, ctx: SizeContext) -> ForkParameter:
        # Also rejected unconditionally by this tensilelite ("TDMSplit is
        # currently disabled"), so [False] is the only value that can build.
        return self._make_param("TDMSplit", [False])

    @param
    def compact_loop_store(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("CompactLoopStore", [False])

    @param
    def preload_kern_args(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("PreloadKernArgs", [True])

    @param
    def lds_tr_inst(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LDSTrInst", [True])

    @param
    def schedule_iter_alg(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ScheduleIterAlg", [4])

    def _is_strict(self) -> bool:
        """True on the A0 part, i.e. any ``gfx1250-strict*`` ARCH key.

        A0 is its own compiler target, gfx1250-strict, and the plain gfx1250
        name means the shipping part. Tensile builds A0 kernels but still
        accepts features A0 cannot use (it drops TDM multicast there instead of
        rejecting a cluster), so the profile has to leave them out itself.
        """
        return str(self.config.get("ARCH", "")).startswith("gfx1250-strict")

    @param
    def cluster_dim(self, ctx: SizeContext) -> ForkParameter:
        """Every cluster shape the device can place whole; [[1, 1]] on A0.

        A0 has no TDM multicast, which is what clusters are for, so there a
        cluster would only add synchronization. Elsewhere the post-processor
        narrows these shapes per MI to the ones worth tuning on its tile grid
        (see ``cluster_dim.py``). The shapes follow the physical topology of the
        ARCH key, not a ``CUs`` override, which only budgets work.
        """
        if self._is_strict():
            return self._make_param("ClusterDim", [[1, 1]])
        hw = HARDWARE_MAP[self.config["ARCH"]]
        shapes = hardware_cluster_dims(wgps_per_shader_engine(hw["CUs"], hw["XCC"]))
        return self._make_param("ClusterDim", [list(shape) for shape in shapes])

    @param
    def prefetch_gl2(self, ctx: SizeContext) -> ForkParameter:
        """[0] on A0; [0, 1, 2] on every other part, with or without StreamK.

        Tensile emits both GSU branches of the prefetch and picks one off the
        runtime GSU, so PrefetchGL2 > 0 is valid with GlobalSplitU -1.
        """
        if self._is_strict():
            return self._make_param("PrefetchGL2", [0])
        return self._make_param("PrefetchGL2", [0, 1, 2])

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
        return self._make_param("LdsPadMXSA", [-1]) if self._mx_block() else None

    @param
    def lds_pad_mxsb(self, ctx: SizeContext) -> Optional[ForkParameter]:
        return self._make_param("LdsPadMXSB", [-1]) if self._mx_block() else None

    @param
    def lds_block_size_per_pad_mxsa(self, ctx: SizeContext) -> Optional[ForkParameter]:
        return self._make_param("LdsBlockSizePerPadMXSA", [-1]) if self._mx_block() else None

    @param
    def lds_block_size_per_pad_mxsb(self, ctx: SizeContext) -> Optional[ForkParameter]:
        return self._make_param("LdsBlockSizePerPadMXSB", [-1]) if self._mx_block() else None

    # =================================================================
    # 2. Persistent strategy and GlobalSplitU
    # =================================================================
    # SAFETY NOTE: the old StreamK=3 with TDMInst=3 and UseSubtileImpl=1
    # is this campaign's one CONFIRMED hard-wedge (~22% fault rate, reboot-only
    # recovery). UseSubtileImpl is never emitted for gfx1250 and Tensile
    # defaults it False, so the confirmed triple cannot form -- a mitigation,
    # not a clearance. StreamK: true puts persistent kernels in the space.

    @param
    def global_split_u(self, ctx: SizeContext) -> ForkParameter:
        """-1 lets Tensile choose. With StreamK: false, MIDesign also attaches
        a per-MI GlobalSplitU that would override this one for those MIs
        (Tensile lets a group entry win over a flat parameter);
        _GFX1250DropMIGroupGSU in post_processor.py strips it."""
        return self._make_param("GlobalSplitU", [-1])

    @param
    def tile_processing_strategy(self, ctx: SizeContext) -> Optional[ForkParameter]:
        """None (non-persistent) unless the geko input config sets ``StreamK: true``.

        With it, :meth:`execution_policy` carries the strategy together with its
        persistent-only options.
        """
        if self._sk3():
            return None
        return self._make_param("TileProcessingStrategy", ["None"])

    @param
    def work_assignment(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._sk3():
            return self._make_param("WorkAssignment", ["StaticGrid"])
        return None

    def _xcc_remap_values(self) -> list:
        """Off, plus the counts that split this ARCH's XCCs into equal groups.

        PersistentXCCMapping groups persistent workgroups across XCCs, and Tensile
        accepts 2 through 8. Only divisors of the XCC count partition evenly (2, 4
        and 8 on an 8-XCC key, 3 on a 3-XCC DPX partition); the others are legal
        but leave a ragged last group, so they are not spent on.
        """
        xcc = HARDWARE_MAP[self.config["ARCH"]]["XCC"]
        return [0] + [n for n in range(2, 9) if xcc % n == 0]

    @group
    def execution_policy(self, ctx: SizeContext) -> Optional[GroupDimension]:
        """Persistent strategies with their options, when the input config sets ``StreamK: true``.

        StreamK and DataParallel each pair with PrefetchAcrossPersistent [0, 1]
        and the XCC remap. Both options exist only for a persistent kernel: Tensile
        raises UnsupportedExecutionPolicy for an explicit PrefetchAcrossPersistent
        with TileProcessingStrategy None, which aborts the run, and zeroes the XCC
        remap there. So an Equality library keeps the non-persistent kernel as an
        entry of its own. An OOB library drops it, since its origami sentinel
        ``WorkGroupMappingXCC: -1`` needs a persistent kernel, and keeps only XCC
        remap 0 ("Cannot use auto WGMXCC with SKXCC").
        """
        if not self._sk3():
            return None
        origami = self._origami_picks_wgm()
        entries = [] if origami else [
            {"TileProcessingStrategy": self._make_param("TileProcessingStrategy", ["None"])}
        ]
        for strategy in ("StreamK", "DataParallel"):
            for prefetch in (0, 1):
                for remap in [0] if origami else self._xcc_remap_values():
                    entries.append({
                        "TileProcessingStrategy": self._make_param("TileProcessingStrategy", [strategy]),
                        "PrefetchAcrossPersistent": self._make_param("PrefetchAcrossPersistent", [prefetch]),
                        "PersistentXCCMapping": self._make_param("PersistentXCCMapping", [remap]),
                    })
        return entries

    @param
    def global_split_u_algorithm(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalSplitUAlgorithm", ["MultipleBuffer"])

    @param
    def work_group_mapping_xcc(self, ctx: SizeContext) -> ForkParameter:
        """Explicit values, except when the runtime picks them.

        ``-1`` (auto) is rejected without a persistent kernel ("Auto WGMXCC
        requires persistent execution"), so it is only offered when StreamK is in
        play and the library is OOB. Otherwise gfx950's explicit generic list
        carries over.
        """
        if self._origami_picks_wgm():
            return self._make_param("WorkGroupMappingXCC", [-1])
        return self._make_param("WorkGroupMappingXCC", [1, 2, 4, 8, 16, 32])

    # =================================================================
    # 3a. Searched: gfx950 generic set, carried over
    # =================================================================

    @param
    def depth_u(self, ctx: SizeContext) -> ForkParameter:
        """gfx950's generic ladder, kept to multiples of this dtype's MI K.

        gfx950 sweeps [32, 64, 128, 256, 512, 1024] for everything. On gfx1250 the
        8-bit and 4-bit types run the K=128 WMMA, so values below 128 are not
        expressible; the bf16/fp16 K=32 WMMA takes 64 to 512.
        """
        if self._gt.data_type in ("F8", "F8B8", "B8F8", "F4"):
            return self._make_param("DepthU", [128, 256, 512])
        return self._make_param("DepthU", [64, 128, 256, 512])

    @param
    def wave_separate_global_read_a(self, ctx: SizeContext) -> ForkParameter:
        # Only applies when TDMInst != 3, and this profile pins TDMInst=3, so
        # every value produces the same kernel. Default (0).
        return self._make_param("WaveSeparateGlobalReadA", [0])

    @param
    def wave_separate_global_read_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("WaveSeparateGlobalReadB", [0])

    @param
    def num_elements_per_batch_store(self, ctx: SizeContext) -> ForkParameter:
        """0 (auto) is enough -- measured neutral on MI450."""
        return self._make_param("NumElementsPerBatchStore", [0])

    @param
    def source_swap(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("SourceSwap", [False, True])

    @param
    def stagger_u(self, ctx: SizeContext) -> ForkParameter:
        """Full valid range; the old [0, 8, 16] sampled three of seven.

        Pinned when the runtime picks it: origami::select_staggerU replaces the
        tuned value outright, so searching seven of them would spend the budget
        on a number that never reaches the GPU. 0 matches what gfx950's Origami
        libraries ship (19289 of 20077 kernels).
        """
        if self._origami_picks_wgm():
            return self._make_param("StaggerU", [0])
        return self._make_param("StaggerU", [0, 2, 4, 8, 16, 32, 64])

    @param
    def stagger_u_stride(self, ctx: SizeContext) -> ForkParameter:
        """Full valid range, including -1 (auto) and 0 (off). Tensile rejects
        strides invalid for the chosen DepthU; that is ordinary attrition.

        Pinned to auto alongside StaggerU: with no stagger the stride cannot
        change the kernel, and -1 is legal for every DepthU.
        """
        if self._origami_picks_wgm():
            return self._make_param("StaggerUStride", [-1])
        return self._make_param(
            "StaggerUStride", [-1, 0, 16, 32, 64, 128, 256, 512, 1024, 2048]
        )

    @param
    def store_priority_opt(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StorePriorityOpt", [False])

    @param
    def store_sync_opt(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("StoreSyncOpt", [0])

    @param
    def work_group_mapping(self, ctx: SizeContext) -> ForkParameter:
        if self._origami_picks_wgm():
            # 0 is the auto sentinel; paired with WorkGroupMappingXCC -1 it
            # hands the choice to origami::select_workgroup_mapping per shape.
            #
            # This is not the same 0 the ladder below excludes. There it was
            # offered without WorkGroupMappingXCC -1, which is the pairing that
            # makes it meaningful, so it scored no valid kernels. Paired, it
            # builds: measured 8/8 solutions on MT128x128 DU64, reaching 95.5 /
            # 99.8 / 101.2 % of the per-shape-tuned ladder on 2048x2048x512,
            # 1024x4096x512 and 4096x1024x512 -- and the tuned side had the
            # advantage of matching each shape, which an OOB library cannot do.
            return self._make_param("WorkGroupMapping", [0])
        # gfx950's ladder minus 0 and 32, which scored 0 valid in 54 and 71
        # sampled points (0 for want of the WGMXCC -1 pairing above).
        # Remapping has little to do on a single-tile-row shape, but the
        # surviving values all produce kernels.
        return self._make_param(
            "WorkGroupMapping",
            [-48, -32, -24, -16, -8, -6, -4, -2, -1, 2, 4, 6, 8, 16, 24, 48],
        )

    @param
    def global_read_vector_width_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalReadVectorWidthA", [-1])

    @param
    def global_read_vector_width_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("GlobalReadVectorWidthB", [-1])

    @param
    def cluster_local_read(self, ctx: SizeContext) -> Optional[ForkParameter]:
        """Independent param for TN (gfx950 pairs it with LDSTrInst otherwise)."""
        if self._is_tn():
            return self._make_param("ClusterLocalRead", [0, 1])
        return None

    @param
    def use_sgpr_for_gro(self, ctx: SizeContext) -> ForkParameter:
        """Independent, not grouped with DirectToLds.

        gfx950 correlates the two in ``dtl_usfgro_group`` because DTL=1 forces
        UseSgprForGRO=0. gfx1250 has no DirectToLds at all, so that group would
        degenerate to its DTL=0 entries.
        """
        # TDMInst=3 makes this inert; default (-1 = auto).
        return self._make_param("UseSgprForGRO", [-1])

    # NonTemporal*: pinned to 0.
    #
    # NOT because gfx1250 ignores the field -- it does not. decodeNonTemporal
    # (Components/NonTemporal.py) switches on the assembler-probed HasTHModifier
    # cap: on gfx950 NonTemporal is a 3-bit glc/slc/nt mask, while on gfx1250
    # glc/slc/nt are forced off and the SAME integer is re-purposed as the cache
    # SCOPE selector, (value & 0x3) -> SCOPE_CU / SE / DEV / SYS. Only bit 0x4
    # is discarded, which is what the client warning refers to.
    #
    # The pin is still right for the pair it replaced: gfx950's [0, 4] is
    # degenerate here, since 4 & 0x3 == 0 & 0x3 == 0 -- both select SCOPE_CU and
    # generate a byte-identical kernel. What it does cost is that scope stays
    # frozen at SCOPE_CU; NonTemporal in {1, 2, 3} (SCOPE_SE / DEV / SYS) has
    # never been measured on gfx1250. Worth an isolated sweep before opening it
    # as a GA axis.
    #
    # Non-temporal itself is not lost -- it moved to TemporalHint*=1 (TH_NT).
    @param
    def non_temporal_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalA", [0])

    @param
    def non_temporal_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalB", [0])

    @param
    def non_temporal_c(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalC", [0])

    @param
    def non_temporal_d(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonTemporalD", [0])

    @param
    def transpose_lds(self, ctx: SizeContext) -> ForkParameter:
        """Pinned [-1]; gfx950 sweeps [-1, 0, 1, 2].

        0 means "coalesced LDS dimension is the tile dimension", i.e. NOT
        unroll-major. That kills two things this profile needs at once:
        ``LDSSegmentInterleave=1`` requires ``UnrollMajorLDSA/B`` ("not
        unrollMajor"), and fp4 local writes reject with "invalid local write
        block width ... This typically means TransposeLDS". Measured 0 valid in
        311 sampled points, against 68-86 for each of the other three.
        """
        # -1 (auto) is enough: Tensile derives the unroll-major layout this
        # profile needs, and the explicit values only re-state it.
        return self._make_param("TransposeLDS", [-1])

    @param
    def adaptive_gemm(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("AdaptiveGemm", [0, 1])

    @param
    def tailloop_in_nll(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TailloopInNll", [False, True])

    @param
    def extra_mi_latency_left(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ExtraMiLatencyLeft", [-1])

    @param
    def schedule_gr_over_barrier(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("ScheduleGROverBarrier", [-1])

    @param
    def unroll_loop_swap_global_read_order(self, ctx: SizeContext) -> ForkParameter:
        """Pinned [0]. gfx950 sweeps [0, 1], but on this geometry 1 rejects with
        "G2LA/B vgpr has bubble inside. Cannot use UnrollLoopSwapGlobalReadOrder=1"
        -- measured 0 valid in 586 sampled points, against 49 in 614 for 0."""
        return self._make_param("UnrollLoopSwapGlobalReadOrder", [0])

    @param
    def use_plr_pack(self, ctx: SizeContext) -> Optional[ForkParameter]:
        if self._gt.data_type == "X":
            return self._make_param("UsePLRPack", [0, 1])
        return None

    # =================================================================
    # 3b. gfx1250-exclusive knobs
    # =================================================================

    @param
    def half_plr(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("HalfPLR", [0, 1, 2, 3])

    @param
    def lds_segment_interleave(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("LDSSegmentInterleave", [0, 1])

    @param
    def tdm_plus_lds_buff(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TDMPlusLdsBuf", [-1])

    @param
    def init_c_iter_wmma(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("InitCIterWmma", [-1])

    @param
    def sw_instruction_prefetch(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("SwInstructionPrefetch", [-1])

    @param
    def non_volatile(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("NonVolatile", [-1])

    # TemporalHint: 0 or 1 (TH_NT). Tensile accepts 0-7, but rejects 7 on any
    # load tensor ("TemporalHint=7 is reserved for loads").
    @param
    def temporal_hint_a(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TemporalHintA", [0, 1])

    @param
    def temporal_hint_b(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("TemporalHintB", [0, 1])

    @param
    def temporal_hint_d(self, ctx: SizeContext) -> ForkParameter:
        """The D store -- the one cache-hint tensor certainly on a live path.

        D is written once per output element and never re-read, so TH_NT is the
        hint most likely to pay.
        """
        return self._make_param("TemporalHintD", [0, 1])

    @param
    def mbsk_prefetch_method(self, ctx: SizeContext) -> ForkParameter:
        return self._make_param("MbskPrefetchMethod", [-1])

