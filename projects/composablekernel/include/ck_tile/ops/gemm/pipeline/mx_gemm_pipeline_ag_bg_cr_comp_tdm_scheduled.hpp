// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

namespace ck_tile {

namespace mx_tdm_scheduled {

// The GL2 prefetch's cache policy: th:TH_LOAD_NT | scope:SCOPE_SE (the enum has it on gfx12 only).
#if defined(__gfx12__)
inline constexpr auto kGl2PrefetchCoherence = amd_buffer_coherence_enum::SE_NT;
#else
inline constexpr auto kGl2PrefetchCoherence = amd_buffer_coherence_enum::coherence_default;
#endif

/// Keeps LLVM from moving LDS reads or WMMAs across this point; ALU work may cross.
CK_TILE_DEVICE void fence()
{
    sched_fence<LLVMSchedGroupMask::VALU | LLVMSchedGroupMask::SALU | LLVMSchedGroupMask::TRANS>();
}

/// The MxTdmScheduledGemmTraits values rocKE derives from `spec`'s shape, as host values.
struct ShapeTraits
{
    bool edge_m;
    bool edge_n;
    bool wide_a;
    bool wide_b;
    bool wide_d;
    index_t odd_tiles;
    index_t tail_steps;
    index_t store_width;
};

CK_TILE_HOST_DEVICE constexpr ShapeTraits shape_traits(const MxTdmScheduledSpec& s)
{
    return {s.edge_m(),
            s.edge_n(),
            s.wide_a(),
            s.wide_b(),
            s.wide_d(),
            static_cast<index_t>(s.K / s.tile_k % 2),
            s.tail_steps(),
            s.store_width()};
}
} // namespace mx_tdm_scheduled

/**
 * @brief MX GEMM main loop for gfx1250 after rocKE's mx_gemm (instances/gfx1250/mx_gemm.py at
 *        commit b32989439dc): TDM loads of A, B and their K-major scales into two LDS buffers,
 *        scaled WMMAs scheduled explicitly, optional cluster multicast and GL2 prefetch.
 *
 * The roles follow rocKE's, in Run: TdmLoader (descriptors built once, each tile patches only
 * its LDS address, global address and remaining sizes; even waves load one B component, odd
 * waves one A component), Gl2Prefetcher, LdsReader (16 B reads), TileCompute (local_prefetch,
 * half-B and the cluster signal) and Pipeline (PrefetchGlobalRead 2: tile k refills its own
 * buffer with tile k+2 once the cluster has read it; tile k+1 is waited for at tile k's last K
 * step).
 *
 * rocKE's M, N and K are compile-time; here they are run time and the Problem's traits
 * (MxTdmScheduledGemmTraits) carry what rocKE
 * derives from them (edges, wide offsets, a K tail). The loop runs tile pairs, one per buffer; an
 * odd tile count runs its last full tile after the loop (rocKE peels it before the loop; the
 * buffers the tiles use are the same) and the K tail tile picks its K step count at run time.
 */
template <typename Problem_, typename Policy_ = MxGemmPipelineAgBgCrCompTDMScheduledPolicy<>>
struct MxGemmPipelineAgBgCrCompTDMScheduled
{
    using Problem   = remove_cvref_t<Problem_>;
    using Policy    = remove_cvref_t<Policy_>;
    using Traits    = typename Problem::Traits;
    using Tuning    = typename Policy::TuningType;
    using ADataType = typename Problem::ADataType;
    using BDataType = typename Problem::BDataType;
    using ODataType = typename Problem::ODataType;

    using AsDataType     = remove_cvref_t<typename Problem::AsDataTypeTuple>;
    using BsDataType     = remove_cvref_t<typename Problem::BsDataTypeTuple>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using AsLayout       = remove_cvref_t<typename Problem::AsLayoutTuple>;
    using BsLayout       = remove_cvref_t<typename Problem::BsLayoutTuple>;
    using ALayout        = remove_cvref_t<typename Problem::ALayout>;
    using BLayout        = remove_cvref_t<typename Problem::BLayout>;
    using CLayout        = remove_cvref_t<typename Problem::CLayout>;
    using AElementWise   = remove_cvref_t<typename Problem::AElementWise>;
    using BElementWise   = remove_cvref_t<typename Problem::BElementWise>;

    static_assert(mx_tdm_scheduled::InvalidTuning<Policy::template GetReason<Problem>()>::value,
                  "comp_tdm_scheduled: the Problem and tuning break a rule of rocKE's validate()");

    static constexpr index_t kBlockSize     = Problem::kBlockSize;
    static constexpr index_t BlockSize      = Problem::kBlockSize;
    static constexpr bool kPadM             = Problem::kPadM;
    static constexpr bool kPadN             = Problem::kPadN;
    static constexpr bool kPadK             = Problem::kPadK;
    static constexpr bool Preshuffle        = false;
    static constexpr bool LargeTensors      = Problem::LargeTensors;
    static constexpr index_t ScaleBlockSize = Problem::ScaleBlockSize;
    static constexpr index_t MXdlPackEff    = 1;
    static constexpr index_t NXdlPackEff    = 1;
    static constexpr index_t KXdlPackEff    = 4;
    /// One MX GEMM of single A, B and C tensors with K-major scales: the multi-D, grouped and
    /// batched kernels reject this pipeline.
    static constexpr bool kPlainGemmOnly = true;

    // TDM reads A and B, so the vector sizes are 1 and the launch check of the vector lengths is
    // skipped, as for the other TDM pipelines.
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeA()
    {
        return 1;
    }
    template <bool IsWave32Host = false>
    static constexpr index_t GetVectorSizeB()
    {
        return 1;
    }
    static constexpr bool skipCheckValidLaunchParams = true;
    /// The scales are K-major, [ceil(K / 128)][MN] int32 packs of four E8M0 scales: the kernel's
    /// scale windows are (MN, K / 128) views of that layout.
    static constexpr bool kKMajorScales = true;
    /// Runs k_batch == 1 only.
    static constexpr bool kSupportsSplitK = false;
    /// kPadM / kPadN (EdgeM / EdgeN) are false only when the tiles divide M / N.
    static constexpr bool kExactPadMN = true;
    /// The cluster multicast takes grid x as N tiles and grid y as M tiles
    /// (GemmTile2DNMajorPartitioner).
    static constexpr bool kGridXWalksN = true;

    /// The workgroup's threads: kBlockSize of the device pass (4 waves of 32).
    static constexpr index_t kThreads         = Policy::template GetBlockSize<Problem>();
    static constexpr index_t kTileM           = BlockGemmShape::kM;
    static constexpr index_t kTileN           = BlockGemmShape::kN;
    static constexpr index_t kTileK           = BlockGemmShape::kK;
    static constexpr index_t kWarpM           = BlockGemmShape::BlockWarps::at(number<0>{});
    static constexpr index_t kWarpN           = BlockGemmShape::BlockWarps::at(number<1>{});
    static constexpr index_t kRm              = Policy::template GetWaveRepeatM<Problem>();
    static constexpr index_t kRn              = Policy::template GetWaveRepeatN<Problem>();
    static constexpr index_t kNumAcc          = kRm * kRn;
    static constexpr index_t kKSteps          = Policy::template GetKSteps<Problem>();
    static constexpr index_t kAtomM           = BlockGemmShape::WarpTile::at(number<0>{});
    static constexpr index_t kAtomN           = BlockGemmShape::WarpTile::at(number<1>{});
    static constexpr bool kMma32              = kAtomM == 32;
    static constexpr index_t kCFragLen        = Policy::template GetCFragLen<Problem>();
    static constexpr index_t kKPerByte        = Policy::template GetKPerByte<Problem>();
    static constexpr index_t kStepBytes       = Policy::template GetStepBytes<Problem>();
    static constexpr index_t kChunks          = Policy::template GetChunksPerLane<Problem>();
    static constexpr index_t kRowWords        = kChunks * 4;
    static constexpr bool kLocalPrefetch      = Policy::template GetLocalPrefetch<Problem>();
    static constexpr bool kHalfB              = Policy::template GetHalfB<Problem>();
    static constexpr bool kHalfBLate          = Policy::template GetHalfBLate<Problem>();
    static constexpr index_t kHalf            = kRn / 2;
    static constexpr bool kHasCluster         = Policy::HasCluster();
    static constexpr index_t kClusterX        = Tuning::kClusterDimX;
    static constexpr index_t kClusterY        = Tuning::kClusterDimY;
    static constexpr index_t kClusterZ        = Tuning::kClusterDimZ;
    static constexpr index_t kSignalWmmas     = Policy::template GetSignalWmmas<Problem>();
    static constexpr index_t kGl2Distance     = Tuning::kGl2PrefetchDistance;
    static constexpr index_t kReadsPerStep    = Policy::template GetReadsPerStep<Problem>();
    static constexpr index_t kLdsRowBytes     = Policy::template GetLdsRowBytes<Problem>();
    static constexpr index_t kLdsBufferStride = Policy::template GetLdsBufferStride<Problem>();
    static constexpr index_t kWait0           = Policy::template GetDscntWait<Problem>(0);
    static constexpr index_t kWait1           = Policy::template GetDscntWait<Problem>(1);
    static constexpr index_t kLatePending     = Policy::template GetLatePendingReads<Problem>();
    using AFrag                               = int32x16_t;
    using BFrag                               = std::conditional_t<kMma32, int32x8_t, int32x16_t>;
    using RowFrag = std::conditional_t<kRowWords == 8, int32x8_t, int32x16_t>;
    using CVec    = std::conditional_t<kMma32, fp32x16_t, fp32x8_t>;
    /// The accumulators: one CVec per M and N repeat, index rm * kRn + rn.
    using CBlockTile = decltype(make_static_distributed_tensor<float>(
        Policy::template MakeCBlockTileDistribution<Problem>()));
    using AccRegs    = thread_buffer<CVec, kNumAcc>;
    static_assert(CBlockTile::get_thread_buffer_size() == kNumAcc * kCFragLen,
                  "comp_tdm_scheduled: the C block tile holds the C fragments of every repeat");

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return Policy::template GetSmemSize<Problem>();
    }

    CK_TILE_HOST static std::string GetName()
    {
        return std::string("pipeline_AgBgCrCompTDMScheduled") + MxTdmScheduledTuningName<Tuning>();
    }

    /// The name of an MxGemmKernel of this pipeline: data types, tile, tuning and traits.
    [[nodiscard]] CK_TILE_HOST static std::string GetKernelName()
    {
        constexpr MxTdmScheduledSpec spec = Policy::template GetSpec<Problem>();
        std::string name =
            std::string("mx_gemm_tdm_scheduled_mx") + mx_tdm_scheduled_dtype_name(spec.a_dtype);
        if(spec.b_dtype != spec.a_dtype)
            name += mx_tdm_scheduled_dtype_name(spec.b_dtype);
        name += "_t" + std::to_string(spec.tile_m) + "x" + std::to_string(spec.tile_n) + "x" +
                std::to_string(spec.tile_k) + mx_tdm_scheduled_tuning_name(spec);
        if(Traits::kEdgeM || Traits::kEdgeN)
            name += "_edge";
        if(Traits::kWideA || Traits::kWideB || Traits::kWideD)
            name += "_wide";
        if(Traits::kOddTiles >= 0)
            name += Traits::kOddTiles ? "_odd" : "_even";
        if(Traits::kHasTail)
            name += "_tail" + std::to_string(Traits::kTailSteps);
        return name + "_w" + std::to_string(Traits::kStoreWidth);
    }

    /// nullptr if this instance runs an M x N x K problem, else why not: rocKE's validate() of the
    /// full spec, or a shape path the traits lack.
    CK_TILE_HOST static const char* GetUnsupportedReason(index_t M, index_t N, index_t K)
    {
        const MxTdmScheduledSpec spec     = Policy::template GetSpec<Problem>(M, N, K);
        const MxTdmScheduledReason reason = mx_tdm_scheduled_validate(spec);
        if(reason != MxTdmScheduledReason::Ok)
            return mx_tdm_scheduled_reason_message(reason);
        const mx_tdm_scheduled::ShapeTraits need = mx_tdm_scheduled::shape_traits(spec);
        if(need.edge_m && !Traits::kEdgeM)
            return "M is not a multiple of tile_m: needs an EdgeM instance";
        if(need.edge_n && !Traits::kEdgeN)
            return "N is not a multiple of tile_n: needs an EdgeN instance";
        if(need.wide_a && !Traits::kWideA)
            return "A's offsets pass 2^31: needs a WideA instance";
        if(need.wide_b && !Traits::kWideB)
            return "B's offsets pass 2^31: needs a WideB instance";
        if(need.wide_d && !Traits::kWideD)
            return "D's offsets pass 2^31: needs a WideD instance";
        if(Traits::kOddTiles >= 0 && need.odd_tiles != Traits::kOddTiles)
            return "the tile count K / tile_k does not have the instance's OddTiles parity";
        if(need.tail_steps != Traits::kTailSteps)
            return "the K tail (K mod tile_k, in 128-wide K steps) is not the instance's TailSteps";
        if(N % Traits::kStoreWidth != 0)
            return "the store width does not divide N";
        return nullptr;
    }

    /// Whether A's or B's fragments are read through a tile window (see
    /// MxGemmPipelineAgBgCrCompTDMScheduledPolicy::IsLdsReadLinear); otherwise RowFragment reads
    /// them.
    static constexpr bool kTileReadA = Policy::template IsLdsReadLinear<Problem, true>();
    static constexpr bool kTileReadB = Policy::template IsLdsReadLinear<Problem, false>();

    /// Wave 0's window on A's or B's fragment of one K step at repeat 0 of buffer 0. The wave
    /// stays out of the window's per-lane coordinate so that its offset remains scalar.
    template <bool IsA>
    CK_TILE_DEVICE static auto MakeLdsWindow(CK_TILE_LDS_ADDR uint8_t* smem)
    {
        constexpr index_t rows = Policy::template GetFragmentHalves<Problem, IsA>() * 16;
        const auto view        = make_tensor_view<address_space_enum::lds>(
            c_style_pointer_cast<int32_t*>(smem),
            Policy::template MakeLdsBlockDescriptor<Problem, IsA>());
        return make_tile_window(view,
                                make_tuple(number<rows>{}, number<kStepBytes / 4>{}),
                                make_multi_index(0, 0),
                                Policy::template MakeLdsReadDistribution<Problem, IsA>());
    }
    using ALdsWindow = decltype(MakeLdsWindow<true>(nullptr));
    using BLdsWindow = decltype(MakeLdsWindow<false>(nullptr));

    /// Whether A's or B's scales are read through a tile window (see
    /// MxGemmPipelineAgBgCrCompTDMScheduledPolicy::IsLdsScaleReadLinear); otherwise ScaleWords
    /// reads them.
    static constexpr bool kTileReadAScale = Policy::template IsLdsScaleReadLinear<Problem, true>();
    static constexpr bool kTileReadBScale = Policy::template IsLdsScaleReadLinear<Problem, false>();

    /// A wave's window on A's or B's scales of K step 0 in buffer 0.
    template <bool IsA>
    CK_TILE_DEVICE static auto MakeLdsScaleWindow(CK_TILE_LDS_ADDR uint8_t* smem, index_t wave)
    {
        constexpr index_t atom    = IsA ? kAtomM : kAtomN;
        constexpr index_t repeats = IsA ? kRm : kRn;
        constexpr index_t lanes   = Policy::template GetScaleLanes<Problem, IsA>();
        const auto view           = make_tensor_view<address_space_enum::lds>(
            c_style_pointer_cast<int32_t*>(smem),
            Policy::template MakeLdsScaleDescriptor<Problem, IsA>());
        return make_tile_window(view,
                                make_tuple(number<1>{}, number<lanes>{}, number<repeats>{}),
                                make_multi_index(0, wave * atom, 0),
                                Policy::template MakeLdsScaleReadDistribution<Problem, IsA>());
    }
    using AScaleWindow = decltype(MakeLdsScaleWindow<true>(nullptr, 0));
    using BScaleWindow = decltype(MakeLdsScaleWindow<false>(nullptr, 0));

    struct Head
    {
        AFrag a[kRm];
        int32_t as[kRm];
        int32_t bs[kRn];
    };
    /// The B fragments read ahead: all of them, or two halves with half-B.
    struct BRegs
    {
        BFrag b[kRn];
    };
    struct Carry
    {
        Head head;
        BRegs b;
    };

    /// A fragment as an operand of the WMMA traits: the fragment itself, or the 8 data words of
    /// an fp4 one padded to 16 (the traits pad it again).
    template <typename Operand, typename Frag>
    CK_TILE_DEVICE static Operand MmaOperand(const Frag& f)
    {
        if constexpr(sizeof(Operand) == sizeof(Frag))
            return bit_cast<Operand>(f);
        else
            return bit_cast<Operand>(__builtin_shufflevector(f, f, 0, 1, 2, 3, 4, 5, 6, 7));
    }

    CK_TILE_DEVICE static void Mma(CVec& c, const AFrag& a, const BFrag& b, int32_t as, int32_t bs)
    {
#if defined(__gfx125__)
        using WarpGemm    = WarpGemmDispatcher<ADataType,
                                               BDataType,
                                               float,
                                               kAtomM,
                                               kAtomN,
                                               mx_tdm_scheduled::kAtomK,
                                               false>;
        using AVec        = typename WarpGemm::WarpGemmAttribute::AVecType;
        using BVec        = typename WarpGemm::WarpGemmAttribute::BVecType;
        constexpr auto I0 = number<0>{};
        typename WarpGemm::AWarpTensor a_warp;
        typename WarpGemm::BWarpTensor b_warp;
        typename WarpGemm::CWarpTensor c_warp;
        a_warp.get_thread_buffer().template set_as<AVec>(I0, MmaOperand<AVec>(a));
        b_warp.get_thread_buffer().template set_as<BVec>(I0, MmaOperand<BVec>(b));
        c_warp.get_thread_buffer().template set_as<CVec>(I0, c);
        WarpGemm{}(c_warp, a_warp, b_warp, as, bs);
        c = c_warp.get_thread_buffer().template get_as<CVec>()[I0];
#else
        ignore = c;
        ignore = a;
        ignore = b;
        ignore = as;
        ignore = bs;
#endif
    }

    struct Run
    {
        // The operands as bytes and the problem, from the DRAM windows' views.
        const uint8_t* a_data;
        const uint8_t* b_data;
        const uint8_t* a_scales;
        const uint8_t* b_scales;
        index_t M;
        index_t N;
        index_t K;
        // The calling thread's place: lane, wave, the tile's first row and column (the A and B
        // windows' origins) and the cluster (0 without one).
        index_t tid;
        index_t lane;
        index_t lane_col;
        index_t lane_half;
        index_t wave;
        index_t wave_m;
        index_t wave_n;
        index_t block_m;
        index_t block_n;
        index_t cluster_wx = 0;
        index_t cluster_wy = 0;
        CK_TILE_LDS_ADDR uint8_t* smem;
        uint32_t smem_addr;
        index_t packed_k;
        index_t k_atoms;
        uint32_t a_mask = 0;
        uint32_t b_mask = 0;
        TDMGroups data{};
        TDMGroups scale{};
        // Group 0 of each load is rebuilt from these: patching the hoisted descriptor's own
        // words keeps it live, and every load then copies it into a fresh register quad.
        uint32_t data_lds     = 0;
        uint32_t scale_lds    = 0;
        uint64_t data_global  = 0;
        uint64_t scale_global = 0;
        index_t scale_stride  = 0;
        // Scale row bytes for the GL2 prefetch: as M * 4 / N * 4, LLVM factors the 4 out of
        // row * stride + offset and the address takes an extra VALU op.
        index_t a_scale_stride;
        index_t b_scale_stride;
        ALdsWindow a_window;
        BLdsWindow b_window;
        AScaleWindow as_window;
        BScaleWindow bs_window;

        template <typename ADram, typename BDram, typename AScaleDram, typename BScaleDram>
        CK_TILE_DEVICE Run(const ADram& a_dram,
                           const BDram& b_dram,
                           const AScaleDram& a_scale_dram,
                           const BScaleDram& b_scale_dram,
                           CK_TILE_LDS_ADDR uint8_t* smem_)
            : a_data(Bytes(a_dram)),
              b_data(Bytes(b_dram)),
              a_scales(Bytes(a_scale_dram)),
              b_scales(Bytes(b_scale_dram)),
              M(Lengths(a_dram)[number<0>{}]),
              N(Lengths(b_dram)[number<0>{}]),
              K(Lengths(a_dram)[number<1>{}]),
              tid(get_thread_id()),
              lane(integer_modulo_nonnegative<mx_tdm_scheduled::kWave>(tid)),
              lane_col(integer_modulo_nonnegative<16>(lane)),
              lane_half(integer_divide_nonnegative<16>(lane)),
              // Wave-uniform, so the TDM descriptors built from it stay scalar.
              wave(get_warp_id()),
              wave_m(integer_divide_nonnegative<kWarpN>(wave)),
              wave_n(integer_modulo_nonnegative<kWarpN>(wave)),
              block_m(a_dram.get_window_origin()[number<0>{}]),
              block_n(b_dram.get_window_origin()[number<0>{}]),
              smem(smem_),
              smem_addr(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_))),
              packed_k(integer_divide_nonnegative<kKPerByte>(K)),
              // The last K step is partial (and its scales zero-padded) only with a K tail.
              k_atoms(integer_divide_nonnegative<mx_tdm_scheduled::kAtomK>(
                  Traits::kHasTail ? K + mx_tdm_scheduled::kAtomK - 1 : K)),
              a_scale_stride(opaque_scalar(M * 4)),
              b_scale_stride(opaque_scalar(N * 4)),
              a_window(MakeLdsWindow<true>(smem_)),
              b_window(MakeLdsWindow<false>(smem_)),
              as_window(MakeLdsScaleWindow<true>(smem_, wave_m)),
              bs_window(MakeLdsScaleWindow<false>(smem_, wave_n))
        {
            if constexpr(kHasCluster)
            {
                cluster_wx = static_cast<index_t>(get_cluster_workgroup_id_x());
                cluster_wy = static_cast<index_t>(get_cluster_workgroup_id_y());
            }
            b_mask = Policy::template GetTDMWorkgroupMask<false>(cluster_wx, cluster_wy);
            a_mask = Policy::template GetTDMWorkgroupMask<true>(cluster_wx, cluster_wy);
        }

        template <typename Window>
        CK_TILE_DEVICE static const uint8_t* Bytes(const Window& window)
        {
            return reinterpret_cast<const uint8_t*>(
                window.get_bottom_tensor_view().get_buffer_view().p_data_);
        }
        template <typename Window>
        CK_TILE_DEVICE static auto Lengths(const Window& window)
        {
            return detail::tdm_real_lengths(
                window.get_bottom_tensor_view().get_tensor_descriptor());
        }

        /// RowMaps: WMMA positions to tile rows. A lane's M repeats are adjacent A rows, so its A
        /// scales of a K step are contiguous; its N repeats are adjacent D columns, so one
        /// accumulator slot is one vector store.
        CK_TILE_DEVICE index_t MRow(index_t rm, index_t m_local) const
        {
            return wave_m * (kRm * kAtomM) + m_local * kRm + rm;
        }
        CK_TILE_DEVICE index_t NRow(index_t rn, index_t n_local) const
        {
            return wave_n * (kRn * kAtomN) + n_local * kRn + rn;
        }

        CK_TILE_DEVICE static constexpr index_t BufferBase(index_t parity)
        {
            return parity * kLdsBufferStride;
        }

        CK_TILE_DEVICE index_t ClampK(index_t k) const { return k < K ? k : K - kTileK; }

        // ---------------------------------------------------------------- TdmLoader

        /// Rows from row0 to the end of the operand; in a partial last tile a component can start
        /// past the end and then gets 0.
        template <bool IsA>
        CK_TILE_DEVICE index_t RowsLeft(index_t row0) const
        {
            const index_t extent = IsA ? M : N;
            const index_t left   = extent - row0;
            constexpr bool edge  = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            if constexpr(!edge)
                return left;
            else
                return row0 < extent ? left : 0;
        }

        /// The groups of a rank-2 byte load through CK's descriptor: dim0 x dim1 tensor of
        /// row_stride-byte rows (dim-1 stride 1, as CK Tile programs rank 2) into a box0 x box1
        /// tile at LDS byte address `lds`. `lds` goes in as an integer, not an LDS pointer: a
        /// flat cast of one would add an aperture and null check.
        CK_TILE_DEVICE static TDMGroups Groups(const void* global,
                                               uint32_t lds,
                                               uint32_t dim0,
                                               uint32_t dim1,
                                               uint16_t box0,
                                               uint16_t box1,
                                               uint32_t row_stride,
                                               const TDMConfig& config)
        {
            const uint32_t dims[2]    = {dim0, dim1};
            const uint64_t strides[2] = {row_stride, 1};
            const uint16_t box[2]     = {box0, box1};
            return TDMDescriptor<uint8_t, 2>(global,
                                             reinterpret_cast<void*>(static_cast<uintptr_t>(lds)),
                                             dims,
                                             strides,
                                             box,
                                             config)
                .getGroups();
        }

        /// The TDMConfig of A's or B's data or scales: its LDS padding and cluster multicast mask.
        template <bool IsA, bool IsScale>
        CK_TILE_DEVICE TDMConfig Config() const
        {
            const uint32_t workgroup_mask = IsA ? a_mask : b_mask;
            // A cluster has at most 16 workgroups; without this the 16-bit field costs an s_and.
            __builtin_assume(workgroup_mask <= 0xFFFFu);
            constexpr bool pad_enable = Policy::template GetLdsPadEnable<Problem, IsA, IsScale>();
            constexpr TDMLdsPaddingConfig padding =
                Policy::template GetLdsPaddingConfig<Problem, IsA, IsScale>();
            TDMConfig config{};
            config.pad_enable     = pad_enable;
            config.pad_config     = padding;
            config.workgroup_mask = static_cast<uint16_t>(workgroup_mask);
            return config;
        }

        template <bool IsA>
        CK_TILE_DEVICE TDMGroups DataDescriptor(index_t component) const
        {
            constexpr MxTdmScheduledOperand L = Policy::template GetLdsOperand<Problem, IsA>();
            constexpr bool edge               = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            constexpr bool wide               = IsA ? Traits::kWideA : Traits::kWideB;
            constexpr bool iterate            = L.iterate_rows > 0;
            constexpr index_t tile_rows       = iterate ? L.iterate_rows : L.rows;
            const index_t block               = IsA ? block_m : block_n;
            const uint8_t* base               = IsA ? a_data : b_data;
            const index_t row0                = block + component * L.rows;
            const uint8_t* global =
                wide ? base + static_cast<long_index_t>(row0) * packed_k : base + row0 * packed_k;
            const index_t rows_left = RowsLeft<IsA>(row0);

            TDMConfig config = Config<IsA, false>();
            if constexpr(iterate)
            {
                constexpr index_t count = L.rows / tile_rows;
                index_t iterations      = count;
                if constexpr(edge)
                {
                    // Each iteration moves tile_rows rows whatever tensor_dim1 says, so walk
                    // ceil(rows left / tile_rows), at least one: with no rows left, tensor_dim1 0
                    // keeps that one off memory.
                    const index_t held = max(min(rows_left, L.rows), 1);
                    iterations = integer_divide_nonnegative<tile_rows>(held + tile_rows - 1);
                }
                config.iterate_enable = true;
                config.iter_config    = {static_cast<uint32_t>(packed_k * tile_rows),
                                         static_cast<uint32_t>(L.iterate_stride),
                                         static_cast<uint16_t>(iterations - 1)};
            }
            return Groups(global,
                          smem_addr + static_cast<uint32_t>(L.base + component * L.stride),
                          static_cast<uint32_t>(packed_k),
                          static_cast<uint32_t>(rows_left),
                          kLdsRowBytes,
                          tile_rows,
                          static_cast<uint32_t>(packed_k),
                          config);
        }

        /// K-major scales [K / 128][MN][4]: one row per K step, 4 B per MN row.
        template <bool IsA>
        CK_TILE_DEVICE TDMGroups ScaleDescriptor(index_t component) const
        {
            constexpr MxTdmScheduledOperand L =
                Policy::template GetLdsOperand<Problem, IsA, true>();
            constexpr bool edge      = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            const index_t block      = IsA ? block_m : block_n;
            const index_t extent     = IsA ? M : N;
            const uint8_t* base      = IsA ? a_scales : b_scales;
            const index_t row_stride = extent * 4;
            const index_t mn0        = block + component * L.rows;
            const index_t col0       = mn0 * 4;
            const index_t bytes_left = edge ? RowsLeft<IsA>(mn0) * 4 : row_stride - col0;

            return Groups(base + col0,
                          smem_addr + static_cast<uint32_t>(L.base + component * L.stride),
                          static_cast<uint32_t>(bytes_left),
                          static_cast<uint32_t>(k_atoms),
                          L.rows * 4,
                          kKSteps,
                          static_cast<uint32_t>(row_stride),
                          Config<IsA, true>());
        }

        /// Both descriptors for tile 0, buffer 0, selected by wave parity; every word uniform.
        CK_TILE_DEVICE void Hoist()
        {
            const bool even         = integer_modulo_nonnegative<2>(wave) == 0;
            const index_t component = integer_divide_nonnegative<2>(wave);
            const auto select       = [&](const TDMGroups& b, const TDMGroups& a) {
                return amd_wave_read_first_lane(even ? b : a);
            };
            data      = select(DataDescriptor<false>(component), DataDescriptor<true>(component));
            scale     = select(ScaleDescriptor<false>(component), ScaleDescriptor<true>(component));
            data_lds  = data.g0.lds_addr;
            scale_lds = scale.g0.lds_addr;
            data_global  = data.g0.getGlobalAddr();
            scale_global = scale.g0.getGlobalAddr();
            scale_stride = even ? N * 4 : M * 4;
        }

        /// This wave's data and scale loads of the tile at k0 into buffer parity.
        CK_TILE_DEVICE void Issue(index_t k0, index_t parity) const
        {
            const index_t k_byte      = integer_divide_nonnegative<kKPerByte>(k0);
            const index_t remaining_k = packed_k - k_byte;
            TDMGroups d               = data;
            d.g0 = TDM_GROUP0{data_lds + static_cast<uint32_t>(BufferBase(parity)),
                              data_global + static_cast<uint32_t>(k_byte),
                              0,
                              0};
            d.g1.tensorDim0(static_cast<uint32_t>(remaining_k));
            amd_tdm_load(d.g0, d.g1, d.g2);
            const index_t k_step     = integer_divide_nonnegative<mx_tdm_scheduled::kAtomK>(k0);
            const index_t steps_left = k_atoms - k_step;
            TDMGroups s              = scale;
            s.g0 = TDM_GROUP0{scale_lds + static_cast<uint32_t>(BufferBase(parity)),
                              scale_global + static_cast<uint32_t>(k_step * scale_stride),
                              0,
                              0};
            s.g1.tensorDim1(static_cast<uint32_t>(steps_left));
            amd_tdm_load(s.g0, s.g1, s.g2);
        }

        // ---------------------------------------------------------------- Gl2Prefetcher

        /// `value`, or limit - 1 past a partial tile's end.
        template <bool Edge>
        CK_TILE_DEVICE static index_t LastInside(index_t value, index_t limit)
        {
            if constexpr(Edge)
                return value < limit ? value : limit - 1;
            else
                return value;
        }

        /// PrefetchGL2: the cluster's A/B and scale lines of the tile at k_ahead, split over
        /// every thread of the cluster, one 256 B line each. AheadParity is the parity of the
        /// prefetched tile's index: with 128 B rows one line holds an even tile and the next one,
        /// so the data lines are only prefetched for even tiles. Branch-free: any branch in the
        /// loop body makes LLVM drain the LDS read pipeline. Past the end of K the last tile is
        /// prefetched again.
        template <index_t AheadParity>
        CK_TILE_DEVICE void Prefetch(index_t k_ahead) const
        {
            if constexpr(kGl2Distance != 0)
            {
                constexpr index_t line = mx_tdm_scheduled::kGl2LineBytes;
                k_ahead                = k_ahead < K ? k_ahead : K - kTileK;
                const index_t k_byte   = integer_divide_nonnegative<kKPerByte>(k_ahead);
                constexpr index_t data_lines =
                    kLdsRowBytes / line == 0 && AheadParity == 0 ? 1 : kLdsRowBytes / line;
                const index_t cluster_m0 = kHasCluster ? block_m - cluster_wy * kTileM : block_m;
                const index_t cluster_n0 = kHasCluster ? block_n - cluster_wx * kTileN : block_n;
                constexpr index_t b_scale_row     = kClusterX * kTileN * 4;
                constexpr index_t a_scale_row     = kClusterY * kTileM * 4;
                constexpr index_t b_lines_per_row = b_scale_row / line;
                constexpr index_t a_lines_per_row = a_scale_row / line;
                // Categories in order: A data, B data, B scales, A scales. Scales are prefetched
                // only for an operand whose cluster scale row is whole lines.
                constexpr index_t counts[4] = {
                    data_lines ? kClusterY * kTileM * data_lines : 0,
                    data_lines ? kClusterX * kTileN * data_lines : 0,
                    b_scale_row % line == 0 ? kKSteps * b_lines_per_row : 0,
                    a_scale_row % line == 0 ? kKSteps * a_lines_per_row : 0};
                constexpr index_t starts[4] = {
                    0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2]};
                constexpr index_t total = starts[3] + counts[3];
                constexpr index_t pool  = kClusterX * kClusterY * kThreads;
                const index_t thread =
                    kHasCluster ? (cluster_wx + cluster_wy * kClusterX) * kThreads + tid : tid;
                // A partial last tile has fewer K steps than the k_steps rows prefetched for it;
                // a partial M/N tile has lines past the end.
                const index_t k_step =
                    integer_divide_nonnegative<mx_tdm_scheduled::kAtomK>(k_ahead);

                const auto address = [&](auto cat, index_t index) -> const uint8_t* {
                    constexpr index_t kCat = decltype(cat)::value;
                    if constexpr(kCat < 2)
                    {
                        constexpr bool is_a = kCat == 0;
                        constexpr bool edge = is_a ? Traits::kEdgeM : Traits::kEdgeN;
                        constexpr bool wide = is_a ? Traits::kWideA : Traits::kWideB;
                        // Dependent on kCat: a tile without data lines (data_lines 0) never
                        // instantiates the divide.
                        constexpr index_t lines = is_a ? data_lines : data_lines;
                        const index_t row0      = is_a ? cluster_m0 : cluster_n0;
                        const index_t row       = LastInside<edge>(
                            row0 + integer_divide_nonnegative<lines>(index), is_a ? M : N);
                        const index_t offset =
                            k_byte + integer_modulo_nonnegative<lines>(index) * line;
                        const uint8_t* base = is_a ? a_data : b_data;
                        if constexpr(wide)
                            return base + (static_cast<uint64_t>(static_cast<uint32_t>(row)) *
                                               static_cast<uint32_t>(packed_k) +
                                           static_cast<uint32_t>(offset));
                        else
                            return base + static_cast<uint32_t>(row * packed_k + offset);
                    }
                    else
                    {
                        constexpr bool is_a = kCat == 3;
                        constexpr bool clamp =
                            (is_a ? Traits::kEdgeM : Traits::kEdgeN) || Traits::kHasTail;
                        constexpr index_t lines_per_row = is_a ? a_lines_per_row : b_lines_per_row;
                        const index_t row_bytes         = is_a ? a_scale_stride : b_scale_stride;
                        const index_t mn0               = is_a ? cluster_m0 : cluster_n0;
                        const index_t row =
                            k_step + integer_divide_nonnegative<lines_per_row>(index);
                        const index_t offset =
                            mn0 * 4 + integer_modulo_nonnegative<lines_per_row>(index) * line;
                        const index_t element =
                            LastInside<clamp>(row * row_bytes + offset, k_atoms * row_bytes);
                        return (is_a ? a_scales : b_scales) + static_cast<uint32_t>(element);
                    }
                };

                static_for<0, (total + pool - 1) / pool, 1>{}([&](auto round) {
                    constexpr index_t round_start = decltype(round)::value * pool;
                    constexpr index_t remaining   = total - round_start;
                    const index_t index =
                        remaining >= pool
                            ? thread + round_start
                            : integer_modulo_nonnegative<remaining>(thread) + round_start;
                    // Ranges are visited last to first, so each lane ends up with the first
                    // range whose end lies above its index.
                    const uint8_t* pointer = nullptr;
                    bool first             = true;
                    static_for<0, 4, 1>{}([&](auto rev) {
                        constexpr index_t cat   = 3 - decltype(rev)::value;
                        constexpr index_t start = starts[cat];
                        constexpr index_t count = counts[cat];
                        if constexpr(count > 0 && start < round_start + pool &&
                                     start + count > round_start)
                        {
                            const index_t local = start ? index - start : index;
                            const uint8_t* here = address(number<cat>{}, local);
                            pointer =
                                first ? here : select_address(index < start + count, here, pointer);
                            first = false;
                        }
                    });
                    amd_global_prefetch<mx_tdm_scheduled::kGl2PrefetchCoherence>(pointer);
                });
            }
            else
            {
                ignore = k_ahead;
            }
        }

        // ---------------------------------------------------------------- LdsReader

        CK_TILE_DEVICE int32x4_t Load4(index_t offset) const
        {
            return *reinterpret_cast<const CK_TILE_LDS_ADDR int32x4_t*>(smem + offset);
        }

        CK_TILE_DEVICE index_t LaneK(index_t step) const
        {
            return step * kStepBytes + lane_half * mx_tdm_scheduled::kChunkBytes;
        }

        CK_TILE_DEVICE RowFrag RowFragment(index_t buffer,
                                           bool is_a,
                                           index_t row,
                                           index_t lane_k) const
        {
            constexpr MxTdmScheduledOperand a_lds = Policy::template GetLdsOperand<Problem, true>();
            constexpr MxTdmScheduledOperand b_lds =
                Policy::template GetLdsOperand<Problem, false>();
            const uint32_t urow = static_cast<uint32_t>(row);
            const index_t base =
                buffer +
                static_cast<index_t>(
                    is_a ? mx_tdm_scheduled_row_offset<uint32_t>(a_lds, kLdsRowBytes, urow)
                         : mx_tdm_scheduled_row_offset<uint32_t>(b_lds, kLdsRowBytes, urow)) +
                lane_k;
            constexpr index_t o0 = Policy::template GetChunkOffset<Problem>(0);
            constexpr index_t o1 = Policy::template GetChunkOffset<Problem>(1);
            const int32x4_t c0   = Load4(base + o0);
            const int32x4_t c1   = Load4(base + o1);
            if constexpr(kChunks == 2)
            {
                return __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            }
            else
            {
                constexpr index_t o2 = Policy::template GetChunkOffset<Problem>(2);
                constexpr index_t o3 = Policy::template GetChunkOffset<Problem>(3);
                const int32x4_t c2   = Load4(base + o2);
                const int32x4_t c3   = Load4(base + o3);
                const int32x8_t lo   = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
                const int32x8_t hi   = __builtin_shufflevector(c2, c3, 0, 1, 2, 3, 4, 5, 6, 7);
                return __builtin_shufflevector(
                    lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            }
        }

        /// One 8-word row fragment zero-padded to the 16 words of the 16x16x128 operand.
        CK_TILE_DEVICE static int32x16_t Pad16(const int32x8_t& x)
        {
            return __builtin_shufflevector(
                x, int32x8_t{}, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        }
        CK_TILE_DEVICE static int32x16_t Pad16(const int32x16_t& x) { return x; }

        CK_TILE_DEVICE AFrag AFragment(index_t buffer, index_t rm, index_t step) const
        {
            if constexpr(kTileReadA)
            {
                return Pad16(LdsFragment<true>(buffer, rm, step));
            }
            else if constexpr(kMma32)
            {
                const index_t lane_k = LaneK(step);
                const int32x8_t r0   = RowFragment(buffer, true, MRow(rm, lane_col), lane_k);
                const int32x8_t r1   = RowFragment(buffer, true, MRow(rm, lane_col + 16), lane_k);
                return __builtin_shufflevector(
                    r0, r1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            }
            else
            {
                return Pad16(RowFragment(buffer, true, MRow(rm, lane_col), LaneK(step)));
            }
        }

        /// A's or B's fragment of repeat r at K step `step` through its tile window: the wave,
        /// buffer, repeat and step are a word offset from the window's.
        template <bool IsA>
        CK_TILE_DEVICE auto LdsFragment(index_t buffer, index_t r, index_t step) const
        {
            constexpr index_t atom        = IsA ? kAtomM : kAtomN;
            constexpr index_t repeats     = IsA ? kRm : kRn;
            constexpr auto repeat_offsets = [] {
                array<index_t, repeats> offsets{};
                for(index_t i = 0; i < repeats; ++i)
                    offsets[i] = Policy::template GetLdsRowOffset<Problem, IsA>(i * atom);
                return offsets;
            }();
            constexpr index_t base = Policy::template GetLdsOperand<Problem, IsA>().base / 4;
            constexpr index_t wave_words =
                Policy::template GetLdsRowOffset<Problem, IsA>(repeats * atom);
            const index_t wave_index = IsA ? wave_m : wave_n;
            const index_t offset = buffer / 4 + base + wave_index * wave_words + repeat_offsets[r] +
                                   step * (kStepBytes / 4);
            const auto& window = [&]() -> const auto& {
                if constexpr(IsA)
                    return a_window;
                else
                    return b_window;
            }();
            const auto tile = window.load_with_offset(offset, number<-1>{}, bool_constant<false>{});
            using Vec       = ext_vector_t<int32_t, decltype(tile)::get_thread_buffer_size()>;
            return tile.get_thread_buffer().template get_as<Vec>()[number<0>{}];
        }

        CK_TILE_DEVICE BFrag BFragment(index_t buffer, index_t rn, index_t step) const
        {
            const RowFrag f = [&] {
                if constexpr(kTileReadB)
                    return LdsFragment<false>(buffer, rn, step);
                else
                    return RowFragment(buffer, false, NRow(rn, lane_col), LaneK(step));
            }();
            if constexpr(kMma32)
                return f;
            else
                return Pad16(f);
        }

        /// Per-lane E8M0 words of one K step and repeat count: 16 B reads of four repeats, or one
        /// read of all of them.
        template <index_t Count>
        CK_TILE_DEVICE void ScaleWords(int32_t (&out)[Count], index_t base) const
        {
            if constexpr(Count >= 4)
            {
                static_for<0, Count / 4, 1>{}([&](auto chunk) {
                    constexpr index_t i = decltype(chunk)::value;
                    const int32x4_t w   = Load4(base + i * 16);
                    out[i * 4 + 0]      = w[0];
                    out[i * 4 + 1]      = w[1];
                    out[i * 4 + 2]      = w[2];
                    out[i * 4 + 3]      = w[3];
                });
            }
            else if constexpr(Count == 2)
            {
                const int32x2_t w =
                    *reinterpret_cast<const CK_TILE_LDS_ADDR int32x2_t*>(smem + base);
                out[0] = w[0];
                out[1] = w[1];
            }
            else
            {
                out[0] = *reinterpret_cast<const CK_TILE_LDS_ADDR int32_t*>(smem + base);
            }
        }

        /// A's or B's per-lane E8M0 words of K step `step` through the scale window.
        template <bool IsA, index_t Count>
        CK_TILE_DEVICE void LdsScaleWords(int32_t (&out)[Count], index_t buffer, index_t step) const
        {
            constexpr MxTdmScheduledOperand op =
                Policy::template GetLdsOperand<Problem, IsA, true>();
            constexpr index_t step_words =
                static_cast<index_t>(mx_tdm_scheduled_scale_offset<uint32_t>(op, 1u, 0u) -
                                     op.base) /
                4;
            const auto& window = [&]() -> const auto& {
                if constexpr(IsA)
                    return as_window;
                else
                    return bs_window;
            }();
            const auto tile = window.load_with_offset(
                buffer / 4 + op.base / 4 + step * step_words, number<-1>{}, bool_constant<false>{});
            static_for<0, Count, 1>{}([&](auto i) { out[i] = tile.get_thread_buffer()[i]; });
        }

        /// A K step's scales and A rows.
        CK_TILE_DEVICE Head ReadHead(index_t buffer, index_t step) const
        {
            Head h;
            if constexpr(kTileReadAScale)
            {
                LdsScaleWords<true>(h.as, buffer, step);
            }
            else
            {
                constexpr MxTdmScheduledOperand as_lds =
                    Policy::template GetLdsOperand<Problem, true, true>();
                const index_t a_lane = kMma32 ? lane : lane_col;
                ScaleWords<kRm>(h.as,
                                buffer +
                                    static_cast<index_t>(mx_tdm_scheduled_scale_offset<uint32_t>(
                                        as_lds,
                                        static_cast<uint32_t>(step),
                                        static_cast<uint32_t>(MRow(0, a_lane)))));
            }
            if constexpr(kTileReadBScale)
            {
                LdsScaleWords<false>(h.bs, buffer, step);
            }
            else
            {
                constexpr MxTdmScheduledOperand bs_lds =
                    Policy::template GetLdsOperand<Problem, false, true>();
                ScaleWords<kRn>(h.bs,
                                buffer +
                                    static_cast<index_t>(mx_tdm_scheduled_scale_offset<uint32_t>(
                                        bs_lds,
                                        static_cast<uint32_t>(step),
                                        static_cast<uint32_t>(NRow(0, lane_col)))));
            }
            static_for<0, kRm, 1>{}([&](auto rm) { h.a[rm] = AFragment(buffer, rm, step); });
            return h;
        }

        /// B fragments First .. First + Count - 1 of a K step into b[First ..].
        template <index_t First, index_t Count>
        CK_TILE_DEVICE void ReadB(BRegs& regs, index_t buffer, index_t step) const
        {
            static_for<First, First + Count, 1>{}(
                [&](auto rn) { regs.b[rn] = BFragment(buffer, rn, step); });
        }

        // ---------------------------------------------------------------- Pipeline hooks

        template <index_t StaticParity>
        CK_TILE_DEVICE void PrefetchAhead(index_t k0) const
        {
            if constexpr(kGl2Distance != 0)
                Prefetch<(StaticParity + kGl2Distance) % 2>(k0 + kGl2Distance * kTileK);
        }

        CK_TILE_DEVICE void Signal() const
        {
            if constexpr(kHasCluster)
            {
                mx_tdm_scheduled::fence();
                cluster_barrier_signal_wave0(wave);
                mx_tdm_scheduled::fence();
            }
        }

        template <index_t StaticParity>
        CK_TILE_DEVICE void Refill(index_t k0) const
        {
            PrefetchAhead<StaticParity>(k0);
            mx_tdm_scheduled::fence();
            if constexpr(kLocalPrefetch)
            {
                // Tile k+1 is read next, from this point: it is the only TDM load in flight, and
                // the barrier below publishes it.
                s_wait_tensorcnt<0>();
                s_barrier();
                mx_tdm_scheduled::fence();
            }
        }

        CK_TILE_DEVICE void Late(index_t k0, index_t parity) const
        {
            mx_tdm_scheduled::fence();
            if constexpr(kHasCluster)
                cluster_barrier_wait();
            // No wave drained its reads of this buffer before the signal; the workgroup barrier
            // publishes them here.
            s_wait_dscnt_pinned<kLatePending>();
            s_barrier();
            Issue(ClampK(k0 + 2 * kTileK), parity);
            mx_tdm_scheduled::fence();
        }

        CK_TILE_DEVICE static void WaitNextTile()
        {
            s_wait_tensorcnt<2>();
            s_barrier();
            compiler_memory_fence();
        }

        // ---------------------------------------------------------------- TileCompute

        /// A one-K-step tile has no second-to-last K step to signal in: it signals just before
        /// its cluster wait.
        template <index_t Steps>
        CK_TILE_DEVICE void SignalBeforeLate() const
        {
            if constexpr(Steps == 1)
                Signal();
        }

        /// The WMMAs of B fragments [First, First + Count) with signal at Signal WMMAs when
        /// SignalHere: half-B order (j outer, rm inner), else rm outer.
        template <bool BOuter, index_t First, index_t Count, bool SignalHere>
        CK_TILE_DEVICE void Wmmas(AccRegs& acc, const Head& h, const BRegs& b) const
        {
            constexpr index_t total = kRm * Count;
            static_for<0, total, 1>{}([&](auto i) {
                constexpr index_t issued = decltype(i)::value;
                constexpr index_t rm     = BOuter ? issued % kRm : issued / Count;
                constexpr index_t rn     = First + (BOuter ? issued / kRm : issued % Count);
                if constexpr(SignalHere && issued == kSignalWmmas)
                    Signal();
                Mma(acc[rm * kRn + rn], h.a[rm], b.b[rn], h.as[rm], h.bs[rn]);
            });
            if constexpr(SignalHere && total == kSignalWmmas)
                Signal();
        }

        /// local_prefetch 1, half-B ahead: all reads of K step s + 1 before the first wait but
        /// B's second half, which follows the first half's WMMAs.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void TileHalfB(AccRegs& acc,
                                      Carry& carry,
                                      index_t k0,
                                      index_t parity,
                                      index_t buffer,
                                      index_t next_buffer) const
        {
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step = decltype(s)::value;
                constexpr bool last    = step + 1 == Steps;
                const Head cur         = carry.head;
                const BRegs cur_b      = carry.b;
                if constexpr(!last)
                {
                    carry.head = ReadHead(buffer, step + 1);
                    ReadB<0, kHalf>(carry.b, buffer, step + 1);
                }
                else
                {
                    Refill<StaticParity>(k0);
                    carry.head = ReadHead(next_buffer, 0);
                    ReadB<0, kHalf>(carry.b, next_buffer, 0);
                }
                constexpr bool signal = kHasCluster && step + 2 == Steps;
                s_wait_dscnt_pinned<kReadsPerStep>();
                Wmmas<true, 0, kHalf, false>(acc, cur, cur_b);
                if constexpr(!last)
                    ReadB<kHalf, kHalf>(carry.b, buffer, step + 1);
                else
                {
                    ReadB<kHalf, kHalf>(carry.b, next_buffer, 0);
                    SignalBeforeLate<Steps>();
                    Late(k0, parity);
                }
                s_wait_dscnt_pinned<kReadsPerStep>();
                Wmmas<true, kHalf, kHalf, signal>(acc, cur, cur_b);
            });
        }

        /// local_prefetch 1, half-B late: wait, scales and A, half 0's WMMAs; wait, B half 0,
        /// half 1's WMMAs; B half 1.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void TileHalfBLate(AccRegs& acc,
                                          Carry& carry,
                                          index_t k0,
                                          index_t parity,
                                          index_t buffer,
                                          index_t next_buffer) const
        {
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step     = decltype(s)::value;
                constexpr bool last        = step + 1 == Steps;
                const Head cur             = carry.head;
                const BRegs cur_b          = carry.b;
                const index_t src_buf      = last ? next_buffer : buffer;
                constexpr index_t src_step = last ? 0 : step + 1;
                constexpr bool signal      = kHasCluster && step + 2 == Steps;
                s_wait_dscnt_pinned<kWait0>();
                if constexpr(last)
                    Refill<StaticParity>(k0);
                carry.head = ReadHead(src_buf, src_step);
                Wmmas<true, 0, kHalf, false>(acc, cur, cur_b);
                if constexpr(last)
                {
                    SignalBeforeLate<Steps>();
                    Late(k0, parity);
                }
                s_wait_dscnt_pinned<kWait1>();
                ReadB<0, kHalf>(carry.b, src_buf, src_step);
                Wmmas<true, kHalf, kHalf, signal>(acc, cur, cur_b);
                ReadB<kHalf, kHalf>(carry.b, src_buf, src_step);
            });
        }

        /// local_prefetch 1, B whole.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void TileFull(AccRegs& acc,
                                     Carry& carry,
                                     index_t k0,
                                     index_t parity,
                                     index_t buffer,
                                     index_t next_buffer) const
        {
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step = decltype(s)::value;
                constexpr bool last    = step + 1 == Steps;
                const Head cur         = carry.head;
                const BRegs cur_b      = carry.b;
                if constexpr(!last)
                {
                    carry.head = ReadHead(buffer, step + 1);
                    ReadB<0, kRn>(carry.b, buffer, step + 1);
                }
                else
                {
                    Refill<StaticParity>(k0);
                    carry.head = ReadHead(next_buffer, 0);
                    ReadB<0, kRn>(carry.b, next_buffer, 0);
                    SignalBeforeLate<Steps>();
                    Late(k0, parity);
                }
                constexpr bool signal = kHasCluster && step + 2 == Steps;
                s_wait_dscnt_pinned<kReadsPerStep>();
                Wmmas<false, 0, kRn, signal>(acc, cur, cur_b);
            });
        }

        /// local_prefetch 0: each K step reads, drains and computes. The tile's last read is the
        /// last K step's: its refill (GL2 prefetch, the signal) follows the drain, the cluster
        /// wait and TDM come cluster_signal_wmmas WMMAs later (at once without a cluster).
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void
        TileDirect(AccRegs& acc, index_t k0, index_t parity, index_t buffer) const
        {
            constexpr index_t late_at = kHasCluster ? kSignalWmmas : 0;
            static_for<0, Steps, 1>{}([&](auto s) {
                constexpr index_t step = decltype(s)::value;
                constexpr bool last    = step + 1 == Steps;
                const Head h           = ReadHead(buffer, step);
                BRegs b;
                ReadB<0, kRn>(b, buffer, step);
                s_wait_dscnt_pinned<0>();
                if constexpr(last)
                {
                    Refill<StaticParity>(k0);
                    Signal();
                }
                static_for<0, kNumAcc, 1>{}([&](auto i) {
                    constexpr index_t issued = decltype(i)::value;
                    constexpr index_t rm     = issued / kRn;
                    constexpr index_t rn     = issued % kRn;
                    if constexpr(last && issued == late_at)
                        Late(k0, parity);
                    Mma(acc[issued], h.a[rm], b.b[rn], h.as[rm], h.bs[rn]);
                });
                if constexpr(last && kNumAcc == late_at)
                    Late(k0, parity);
            });
        }

        /// One K tile at k0 in buffer parity: StaticParity is its parity when known at compile
        /// time (the GL2 short-row choice), Steps its K step count.
        template <index_t Steps, index_t StaticParity>
        CK_TILE_DEVICE void Tile(AccRegs& acc, Carry& carry, index_t k0, index_t parity) const
        {
            const index_t buffer      = BufferBase(parity);
            const index_t next_buffer = BufferBase(1 - parity);
            if constexpr(!kLocalPrefetch)
            {
                WaitNextTile();
                TileDirect<Steps, StaticParity>(acc, k0, parity, buffer);
            }
            else if constexpr(kHalfBLate)
                TileHalfBLate<Steps, StaticParity>(acc, carry, k0, parity, buffer, next_buffer);
            else if constexpr(kHalfB)
                TileHalfB<Steps, StaticParity>(acc, carry, k0, parity, buffer, next_buffer);
            else
                TileFull<Steps, StaticParity>(acc, carry, k0, parity, buffer, next_buffer);
        }

        CK_TILE_DEVICE void Main(AccRegs& acc)
        {
            Hoist();
            if constexpr(kHasCluster)
                cluster_barrier();
            else
                s_barrier();
            Issue(0, 0);
            // With one tile, buffer 1 gets tile 0 again and is never read.
            Issue(ClampK(kTileK), 1);
            static_for<2, (kGl2Distance > 2 ? kGl2Distance : 2), 1>{}(
                [&](auto ahead) { Prefetch<decltype(ahead)::value % 2>(ahead * kTileK); });
            if constexpr(kLocalPrefetch)
                WaitNextTile();
            static_for<0, kNumAcc, 1>{}([&](auto i) { acc[i] = CVec{}; });
            Carry carry{};
            if constexpr(kLocalPrefetch)
            {
                carry.head = ReadHead(BufferBase(0), 0);
                ReadB<0, kRn>(carry.b, BufferBase(0), 0);
            }
            const index_t full_tiles = integer_divide_nonnegative<kTileK>(K);
            constexpr index_t kOdd   = Traits::kOddTiles;
            // The loop runs tile pairs, one tile per buffer, so buffer offsets fold. A known odd
            // tile count peels tile 0 before the loop, as rocKE; an unknown one runs the odd
            // tile after the loop.
            index_t start = 0;
            if constexpr(kOdd == 1)
            {
                Tile<kKSteps, 0>(acc, carry, 0, 0);
                start = kTileK;
            }
            constexpr index_t kFirst = kOdd == 1 ? 1 : 0;
            const index_t loop_end   = kOdd < 0
                                           ? integer_divide_nonnegative<2>(full_tiles) * (2 * kTileK)
                                           : (Traits::kHasTail ? full_tiles * kTileK : K);
            for(index_t k0 = start; k0 < loop_end; k0 += 2 * kTileK)
            {
                Tile<kKSteps, kFirst>(acc, carry, k0, kFirst);
                Tile<kKSteps, 1 - kFirst>(acc, carry, k0 + kTileK, 1 - kFirst);
            }
            if constexpr(kOdd < 0)
            {
                if(integer_modulo_nonnegative<2>(full_tiles))
                    Tile<kKSteps, 0>(acc, carry, loop_end, 0);
            }
            if constexpr(Traits::kHasTail)
            {
                // The K tail is one more tile: its TDM loads are clipped at K and only its first
                // tail_steps K steps are read. With the tile parity unknown at compile time the
                // GL2 prefetch takes it as even, which only changes which lines are hinted.
                static_assert(Traits::kTailSteps < kKSteps,
                              "comp_tdm_scheduled: tail steps < k_steps");
                constexpr index_t kTailParity = kOdd < 0 ? 0 : kOdd;
                Tile<Traits::kTailSteps, kTailParity>(
                    acc,
                    carry,
                    full_tiles * kTileK,
                    kOdd < 0 ? integer_modulo_nonnegative<2>(full_tiles) : kTailParity);
            }
            // No workgroup may exit while a TDM load into its LDS is in flight: with multicast,
            // peers' loads write here too.
            s_wait_tensorcnt<0>();
            if constexpr(kHasCluster)
                cluster_barrier();
        }
    };

    /// The workgroup's tile of the M x N x K problem: A and B are the (M, K) and (N, K) windows
    /// of their tiles, the scales (M, K / 128) and (N, K / 128) windows of K-major packs. The TDM
    /// loads read the operands from the windows' buffers and clip at the unpadded extents; the
    /// tile origin is the A and B windows'. num_loop is unused: the K tile count and the K tail
    /// come from K and the shape traits. `p_smem` is GetSmemSize() bytes of LDS.
    template <typename AsDramBlockWindowTmp,
              typename AElementFunction,
              typename BsDramBlockWindowTmp,
              typename BElementFunction,
              typename ScaleADramBlockWindowTmp,
              typename ScaleBDramBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const AsDramBlockWindowTmp& a_dram_block_window_tmp,
                                   const AElementFunction&,
                                   const BsDramBlockWindowTmp& b_dram_block_window_tmp,
                                   const BElementFunction&,
                                   const ScaleADramBlockWindowTmp& scale_a_window,
                                   const ScaleBDramBlockWindowTmp& scale_b_window,
                                   [[maybe_unused]] index_t num_loop,
                                   void* p_smem) const
    {
        static_assert(std::is_same_v<AElementFunction, element_wise::PassThrough> &&
                          std::is_same_v<BElementFunction, element_wise::PassThrough>,
                      "comp_tdm_scheduled reads A and B as stored");
        static_assert(std::tuple_size_v<AsDramBlockWindowTmp> == 1 &&
                          std::tuple_size_v<BsDramBlockWindowTmp> == 1 &&
                          std::tuple_size_v<ScaleADramBlockWindowTmp> == 1 &&
                          std::tuple_size_v<ScaleBDramBlockWindowTmp> == 1,
                      "comp_tdm_scheduled runs single A and B tensors");
        CBlockTile c_block_tile;
#if defined(__gfx125__)
        Run run(a_dram_block_window_tmp[number<0>{}],
                b_dram_block_window_tmp[number<0>{}],
                scale_a_window[number<0>{}],
                scale_b_window[number<0>{}],
                c_style_pointer_cast<CK_TILE_LDS_ADDR uint8_t*>(p_smem));
        run.Main(c_block_tile.get_thread_buffer().template get_as<CVec>());
#else
        ignore = a_dram_block_window_tmp;
        ignore = b_dram_block_window_tmp;
        ignore = scale_a_window;
        ignore = scale_b_window;
        ignore = p_smem;
#endif
        return c_block_tile;
    }
};

} // namespace ck_tile
