// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_tdm_v4_policy.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp"

namespace ck_tile {

namespace mx_tdm_v4 {

/// The operands of one run: A, B and their K-major scales as bytes, and the problem.
struct Operands
{
    const uint8_t* a;
    const uint8_t* b;
    const uint8_t* a_scale;
    const uint8_t* b_scale;
    index_t M;
    index_t N;
    index_t K;
};

/// The calling thread's place: lane, wave, the tile's first row and column, and the cluster.
struct Coords
{
    index_t tid;
    index_t lane;
    index_t lane_col;
    index_t lane_half;
    index_t wave;
    index_t wave_m;
    index_t wave_n;
    index_t block_m;
    index_t block_n;
    index_t cluster_wx;
    index_t cluster_wy;
};

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

/// The descriptor groups of a rank-2 TDM tile load; groups 3 and 4 are zero.
// g1 first: its int32x8_t is 32-byte aligned, and after g0 it would add 16 bytes of padding
// that amd_wave_read_first_lane reads as well.
struct TdmDesc
{
    TDM_GROUP1 g1;
    TDM_GROUP0 g0;
    TDM_GROUP2 g2;
};
static_assert(sizeof(TdmDesc) == 64, "TdmDesc must hold its 16 SGPRs without padding");

/// Group 1 of a rank-2 load of 1-byte elements: tensor_dim1 rows of tensor_dim0 bytes,
/// row_stride bytes apart, in a tile_dim0 x tile_dim1 box. The dim-1 stride is 1, as CK Tile
/// programs rank 2. Padding is the zero config without padding.
template <bool Iterate, bool Pad, TDMLdsPaddingConfig Padding = TDMLdsPaddingConfig{}>
CK_TILE_DEVICE TDM_GROUP1 tdm_group1(uint32_t workgroup_mask,
                                     uint32_t tensor_dim0,
                                     uint32_t tensor_dim1,
                                     uint16_t tile_dim0,
                                     uint16_t tile_dim1,
                                     uint32_t row_stride)
{
    // A cluster has at most 16 workgroups; without this the 16-bit field costs an s_and.
    __builtin_assume(workgroup_mask <= 0xFFFFu);
    TDM_GROUP1 g1;
    g1.workgroup_mask = workgroup_mask;
    g1.iterate_enable = Iterate;
    g1.pad_enable     = Pad;
    g1.pad_interval   = Padding.pad_interval;
    g1.pad_amount     = Padding.pad_amount;
    g1.tensorDim0(tensor_dim0);
    g1.tensorDim1(tensor_dim1);
    g1.tileDim(0, tile_dim0);
    g1.tileDim(1, tile_dim1);
    g1.tensorDim0Stride(row_stride);
    g1.tensorDim1Stride(1);
    return g1;
}
} // namespace mx_tdm_v4

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
 * (MxTdmV4GemmTraits) carry what rocKE
 * derives from them (edges, wide offsets, a K tail). The loop runs tile pairs, one per buffer; an
 * odd tile count runs its last full tile after the loop (rocKE peels it before the loop; the
 * buffers the tiles use are the same) and the K tail tile picks its K step count at run time.
 */
template <typename Problem_, typename Policy_ = GemmPipelineAgBgCrCompTDMV4Policy<>>
struct GemmPipelineAgBgCrCompTDMV4
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

    static constexpr MxTdmV4Spec kSpec = Policy::template GetSpec<Problem>();
    static_assert(mx_tdm_v4::InvalidTuning<Policy::template GetReason<Problem>()>::value,
                  "comp_tdm_v4: the Problem and tuning break a rule of rocKE's validate()");
    static constexpr MxTdmV4LdsLayout kLayout = Policy::template GetLdsLayout<Problem>();

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

    /// The workgroup's threads: kBlockSize of the device pass (4 waves of 32).
    static constexpr index_t kThreads     = kSpec.block_size();
    static constexpr index_t kTileM       = kSpec.tile_m;
    static constexpr index_t kTileN       = kSpec.tile_n;
    static constexpr index_t kTileK       = kSpec.tile_k;
    static constexpr index_t kWarpM       = kSpec.warp_m;
    static constexpr index_t kWarpN       = kSpec.warp_n;
    static constexpr index_t kRm          = kSpec.wave_repeat_m();
    static constexpr index_t kRn          = kSpec.wave_repeat_n();
    static constexpr index_t kNumAcc      = kRm * kRn;
    static constexpr index_t kKSteps      = kSpec.k_steps();
    static constexpr index_t kAtomM       = kSpec.atom_m();
    static constexpr index_t kAtomN       = kSpec.atom_n();
    static constexpr bool kMma32          = kSpec.mma == MxTdmV4Mma::M32N16;
    static constexpr index_t kCFragLen    = kSpec.c_frag_len();
    static constexpr index_t kKPerByte    = kSpec.k_per_byte();
    static constexpr index_t kStepBytes   = kSpec.step_bytes();
    static constexpr index_t kChunks      = kSpec.chunks_per_lane();
    static constexpr index_t kRowWords    = kChunks * 4;
    static constexpr bool kLocalPrefetch  = kSpec.local_prefetch == 1;
    static constexpr bool kHalfB          = kSpec.half_b();
    static constexpr bool kHalfBLate      = kSpec.half_b_late();
    static constexpr index_t kHalf        = kRn / 2;
    static constexpr bool kHasCluster     = kSpec.has_cluster();
    static constexpr index_t kClusterX    = kSpec.cluster_x;
    static constexpr index_t kClusterY    = kSpec.cluster_y;
    static constexpr index_t kSignalWmmas = kSpec.signal_wmmas();
    static constexpr index_t kGl2Distance = kSpec.gl2_distance;
    static constexpr index_t kReadsPerStep =
        mx_tdm_v4::min_i(kSpec.reads_per_step(), mx_tdm_v4::kDscntMax);
    static constexpr index_t kWait0       = Policy::template GetDscntWait<Problem>(0);
    static constexpr index_t kWait1       = Policy::template GetDscntWait<Problem>(1);
    static constexpr index_t kLatePending = Policy::template GetLatePendingReads<Problem>();
    using AFrag                           = int32x16_t;
    using BFrag                           = std::conditional_t<kMma32, int32x8_t, int32x16_t>;
    using RowFrag = std::conditional_t<kRowWords == 8, int32x8_t, int32x16_t>;
    using CVec    = std::conditional_t<kMma32, fp32x16_t, fp32x8_t>;

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return kLayout.lds_bytes; }

    CK_TILE_HOST static std::string GetName()
    {
        return std::string("pipeline_AgBgCrCompTDMV4") + MxTdmV4TuningName<Tuning>();
    }

    /// This thread's coordinates; the cluster ids are 0 without a cluster.
    CK_TILE_DEVICE static mx_tdm_v4::Coords MakeCoords()
    {
        mx_tdm_v4::Coords c{};
        c.tid       = static_cast<index_t>(threadIdx.x);
        c.lane      = integer_modulo_nonnegative<mx_tdm_v4::kWave>(c.tid);
        c.lane_col  = integer_modulo_nonnegative<16>(c.lane);
        c.lane_half = integer_divide_nonnegative<16>(c.lane);
        // Wave-uniform, so the TDM descriptors built from it stay scalar.
        c.wave    = amd_wave_read_first_lane(integer_divide_nonnegative<mx_tdm_v4::kWave>(c.tid));
        c.wave_n  = integer_modulo_nonnegative<kWarpN>(c.wave);
        c.wave_m  = integer_divide_nonnegative<kWarpN>(c.wave);
        c.block_m = static_cast<index_t>(blockIdx.y) * kTileM;
        c.block_n = static_cast<index_t>(blockIdx.x) * kTileN;
        if constexpr(kHasCluster)
        {
            c.cluster_wx = static_cast<index_t>(get_cluster_workgroup_id_x());
            c.cluster_wy = static_cast<index_t>(get_cluster_workgroup_id_y());
        }
        return c;
    }

    /// RowMaps: WMMA positions to tile rows. A lane's M repeats are adjacent A rows, so its A
    /// scales of a K step are contiguous; its N repeats are adjacent D columns, so one
    /// accumulator slot is one vector store.
    CK_TILE_DEVICE static index_t MRow(const mx_tdm_v4::Coords& c, index_t rm, index_t m_local)
    {
        return c.wave_m * (kRm * kAtomM) + m_local * kRm + rm;
    }
    CK_TILE_DEVICE static index_t NRow(const mx_tdm_v4::Coords& c, index_t rn, index_t n_local)
    {
        return c.wave_n * (kRn * kAtomN) + n_local * kRn + rn;
    }

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
        using Op = WmmaTraits<gfx125_t,
                              typename Problem::ADataType,
                              typename Problem::BDataType,
                              float,
                              kAtomM,
                              kAtomN,
                              128>;
        c = bit_cast<CVec>(Op::template wmma_intrinsic<>(MmaOperand<typename Op::AVecType>(a),
                                                         as,
                                                         MmaOperand<typename Op::BVecType>(b),
                                                         bs,
                                                         bit_cast<typename Op::CVecType>(c)));
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
        const mx_tdm_v4::Operands& p;
        const mx_tdm_v4::Coords& c;
        CK_TILE_LDS_ADDR uint8_t* smem;
        uint32_t smem_addr;
        index_t packed_k;
        index_t k_atoms;
        uint32_t a_mask = 0;
        uint32_t b_mask = 0;
        mx_tdm_v4::TdmDesc data{};
        mx_tdm_v4::TdmDesc scale{};
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

        CK_TILE_DEVICE Run(const mx_tdm_v4::Operands& p_,
                           const mx_tdm_v4::Coords& c_,
                           CK_TILE_LDS_ADDR uint8_t* smem_)
            : p(p_),
              c(c_),
              smem(smem_),
              smem_addr(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem_))),
              packed_k(integer_divide_nonnegative<kKPerByte>(p_.K)),
              // The last K step is partial (and its scales zero-padded) only with a K tail.
              k_atoms(integer_divide_nonnegative<mx_tdm_v4::kAtomK>(
                  Traits::kHasTail ? p_.K + mx_tdm_v4::kAtomK - 1 : p_.K)),
              a_scale_stride(opaque_scalar(p_.M * 4)),
              b_scale_stride(opaque_scalar(p_.N * 4))
        {
            if constexpr(kHasCluster)
            {
                // Grid x walks N tiles and y walks M tiles: workgroups with the same cluster x
                // share B, those with the same y share A.
                b_mask = cluster_mask_same_x(kClusterX, kClusterY, c.cluster_wx);
                a_mask = cluster_mask_same_y(kClusterX, c.cluster_wy);
            }
        }

        CK_TILE_DEVICE static constexpr index_t BufferBase(index_t parity)
        {
            return parity * kLayout.buffer_stride;
        }

        CK_TILE_DEVICE index_t ClampK(index_t k) const { return k < p.K ? k : p.K - kTileK; }

        // ---------------------------------------------------------------- TdmLoader

        /// Rows from row0 to the end of the operand; in a partial last tile a component can start
        /// past the end and then gets 0.
        template <bool IsA>
        CK_TILE_DEVICE index_t RowsLeft(index_t row0) const
        {
            const index_t extent = IsA ? p.M : p.N;
            const index_t left   = extent - row0;
            constexpr bool edge  = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            if constexpr(!edge)
                return left;
            else
                return row0 < extent ? left : 0;
        }

        template <bool IsA>
        CK_TILE_DEVICE mx_tdm_v4::TdmDesc DataDescriptor(index_t component) const
        {
            constexpr MxTdmV4Operand L  = IsA ? kLayout.a : kLayout.b;
            constexpr bool edge         = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            constexpr bool wide         = IsA ? Traits::kWideA : Traits::kWideB;
            constexpr bool iterate      = L.iterate_rows > 0;
            constexpr bool pad          = !iterate && L.pad_bytes > 0;
            constexpr index_t tile_rows = iterate ? L.iterate_rows : L.rows;
            const index_t block         = IsA ? c.block_m : c.block_n;
            const uint8_t* base         = IsA ? p.a : p.b;
            const index_t row0          = block + component * L.rows;
            const uint8_t* global =
                wide ? base + static_cast<long_index_t>(row0) * packed_k : base + row0 * packed_k;
            const index_t rows_left = RowsLeft<IsA>(row0);

            mx_tdm_v4::TdmDesc d{};
            d.g0 = TDM_GROUP0{smem_addr + static_cast<uint32_t>(L.base + component * L.stride),
                              reinterpret_cast<uintptr_t>(global),
                              0,
                              0};
            constexpr TDMLdsPaddingConfig padding =
                pad ? make_tdm_lds_padding_config(L.block_bytes, L.pad_bytes)
                    : TDMLdsPaddingConfig{};
            d.g1 = mx_tdm_v4::tdm_group1<iterate, pad, padding>(IsA ? a_mask : b_mask,
                                                                static_cast<uint32_t>(packed_k),
                                                                static_cast<uint32_t>(rows_left),
                                                                kLayout.row_bytes,
                                                                tile_rows,
                                                                static_cast<uint32_t>(packed_k));
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
                d.g2.mode0.lds_addr_increment = static_cast<uint32_t>(L.iterate_stride);
                d.g2.mode0.globalAddrIncrement(static_cast<uint32_t>(packed_k * tile_rows));
                d.g2.mode0.iterate_count = static_cast<uint32_t>(iterations - 1);
            }
            return d;
        }

        /// K-major scales [K / 128][MN][4]: one row per K step, 4 B per MN row.
        template <bool IsA>
        CK_TILE_DEVICE mx_tdm_v4::TdmDesc ScaleDescriptor(index_t component) const
        {
            constexpr MxTdmV4Operand L = IsA ? kLayout.a_scale : kLayout.b_scale;
            constexpr bool edge        = IsA ? Traits::kEdgeM : Traits::kEdgeN;
            constexpr bool pad         = L.pad_bytes > 0;
            const index_t block        = IsA ? c.block_m : c.block_n;
            const index_t extent       = IsA ? p.M : p.N;
            const uint8_t* base        = IsA ? p.a_scale : p.b_scale;
            const index_t row_stride   = extent * 4;
            const index_t mn0          = block + component * L.rows;
            const index_t col0         = mn0 * 4;
            const index_t bytes_left   = edge ? RowsLeft<IsA>(mn0) * 4 : row_stride - col0;

            mx_tdm_v4::TdmDesc d{};
            d.g0 = TDM_GROUP0{smem_addr + static_cast<uint32_t>(L.base + component * L.stride),
                              reinterpret_cast<uintptr_t>(base + col0),
                              0,
                              0};
            constexpr TDMLdsPaddingConfig padding =
                pad ? make_tdm_lds_padding_config(L.block_bytes, L.pad_bytes)
                    : TDMLdsPaddingConfig{};
            d.g1 = mx_tdm_v4::tdm_group1<false, pad, padding>(IsA ? a_mask : b_mask,
                                                              static_cast<uint32_t>(bytes_left),
                                                              static_cast<uint32_t>(k_atoms),
                                                              L.rows * 4,
                                                              kLayout.k_steps,
                                                              static_cast<uint32_t>(row_stride));
            return d;
        }

        /// Both descriptors for tile 0, buffer 0, selected by wave parity; every word uniform.
        CK_TILE_DEVICE void Hoist()
        {
            const bool even         = integer_modulo_nonnegative<2>(c.wave) == 0;
            const index_t component = integer_divide_nonnegative<2>(c.wave);
            const auto select = [&](const mx_tdm_v4::TdmDesc& b, const mx_tdm_v4::TdmDesc& a) {
                return amd_wave_read_first_lane(even ? b : a);
            };
            data      = select(DataDescriptor<false>(component), DataDescriptor<true>(component));
            scale     = select(ScaleDescriptor<false>(component), ScaleDescriptor<true>(component));
            data_lds  = data.g0.lds_addr;
            scale_lds = scale.g0.lds_addr;
            data_global  = data.g0.getGlobalAddr();
            scale_global = scale.g0.getGlobalAddr();
            scale_stride = even ? p.N * 4 : p.M * 4;
        }

        /// This wave's data and scale loads of the tile at k0 into buffer parity.
        CK_TILE_DEVICE void Issue(index_t k0, index_t parity) const
        {
            const index_t k_byte      = integer_divide_nonnegative<kKPerByte>(k0);
            const index_t remaining_k = packed_k - k_byte;
            mx_tdm_v4::TdmDesc d      = data;
            d.g0 = TDM_GROUP0{data_lds + static_cast<uint32_t>(BufferBase(parity)),
                              data_global + static_cast<uint32_t>(k_byte),
                              0,
                              0};
            d.g1.tensorDim0(static_cast<uint32_t>(remaining_k));
            amd_tdm_load(d.g0, d.g1, d.g2);
            const index_t k_step     = integer_divide_nonnegative<mx_tdm_v4::kAtomK>(k0);
            const index_t steps_left = k_atoms - k_step;
            mx_tdm_v4::TdmDesc s     = scale;
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
                constexpr index_t line       = mx_tdm_v4::kGl2LineBytes;
                k_ahead                      = k_ahead < p.K ? k_ahead : p.K - kTileK;
                const index_t k_byte         = integer_divide_nonnegative<kKPerByte>(k_ahead);
                constexpr index_t data_lines = kLayout.row_bytes / line == 0 && AheadParity == 0
                                                   ? 1
                                                   : kLayout.row_bytes / line;
                const index_t cluster_m0 =
                    kHasCluster ? c.block_m - c.cluster_wy * kTileM : c.block_m;
                const index_t cluster_n0 =
                    kHasCluster ? c.block_n - c.cluster_wx * kTileN : c.block_n;
                constexpr index_t b_scale_row     = kClusterX * kTileN * 4;
                constexpr index_t a_scale_row     = kClusterY * kTileM * 4;
                constexpr index_t b_lines_per_row = b_scale_row / line;
                constexpr index_t a_lines_per_row = a_scale_row / line;
                // Categories in order: A data, B data, B scales, A scales. Scales are prefetched
                // only for an operand whose cluster scale row is whole lines.
                constexpr index_t counts[4] = {
                    data_lines ? kClusterY * kTileM * data_lines : 0,
                    data_lines ? kClusterX * kTileN * data_lines : 0,
                    b_scale_row % line == 0 ? kLayout.k_steps * b_lines_per_row : 0,
                    a_scale_row % line == 0 ? kLayout.k_steps * a_lines_per_row : 0};
                constexpr index_t starts[4] = {
                    0, counts[0], counts[0] + counts[1], counts[0] + counts[1] + counts[2]};
                constexpr index_t total = starts[3] + counts[3];
                constexpr index_t pool  = kClusterX * kClusterY * kThreads;
                const index_t thread =
                    kHasCluster ? (c.cluster_wx + c.cluster_wy * kClusterX) * kThreads + c.tid
                                : c.tid;
                // A partial last tile has fewer K steps than the k_steps rows prefetched for it;
                // a partial M/N tile has lines past the end.
                const index_t k_step = integer_divide_nonnegative<mx_tdm_v4::kAtomK>(k_ahead);

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
                            row0 + integer_divide_nonnegative<lines>(index), is_a ? p.M : p.N);
                        const index_t offset =
                            k_byte + integer_modulo_nonnegative<lines>(index) * line;
                        const uint8_t* base = is_a ? p.a : p.b;
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
                        return (is_a ? p.a_scale : p.b_scale) + static_cast<uint32_t>(element);
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
                    amd_global_prefetch<mx_tdm_v4::kGl2PrefetchCoherence>(pointer);
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
            return step * kStepBytes + c.lane_half * mx_tdm_v4::kChunkBytes;
        }

        CK_TILE_DEVICE RowFrag RowFragment(index_t buffer,
                                           bool is_a,
                                           index_t row,
                                           index_t lane_k) const
        {
            const uint32_t urow = static_cast<uint32_t>(row);
            const index_t base =
                buffer +
                static_cast<index_t>(
                    is_a ? mx_tdm_v4_row_offset<uint32_t>(kLayout.a, kLayout.row_bytes, urow)
                         : mx_tdm_v4_row_offset<uint32_t>(kLayout.b, kLayout.row_bytes, urow)) +
                lane_k;
            const int32x4_t c0 = Load4(base + kSpec.chunk_offset(0));
            const int32x4_t c1 = Load4(base + kSpec.chunk_offset(1));
            if constexpr(kChunks == 2)
            {
                return __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
            }
            else
            {
                const int32x4_t c2 = Load4(base + kSpec.chunk_offset(2));
                const int32x4_t c3 = Load4(base + kSpec.chunk_offset(3));
                const int32x8_t lo = __builtin_shufflevector(c0, c1, 0, 1, 2, 3, 4, 5, 6, 7);
                const int32x8_t hi = __builtin_shufflevector(c2, c3, 0, 1, 2, 3, 4, 5, 6, 7);
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
            const index_t lane_k = LaneK(step);
            if constexpr(kMma32)
            {
                const int32x8_t r0 = RowFragment(buffer, true, MRow(c, rm, c.lane_col), lane_k);
                const int32x8_t r1 =
                    RowFragment(buffer, true, MRow(c, rm, c.lane_col + 16), lane_k);
                return __builtin_shufflevector(
                    r0, r1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
            }
            else
            {
                return Pad16(RowFragment(buffer, true, MRow(c, rm, c.lane_col), lane_k));
            }
        }

        CK_TILE_DEVICE BFrag BFragment(index_t buffer, index_t rn, index_t step) const
        {
            const RowFrag f = RowFragment(buffer, false, NRow(c, rn, c.lane_col), LaneK(step));
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

        /// A K step's scales and A rows.
        CK_TILE_DEVICE Head ReadHead(index_t buffer, index_t step) const
        {
            Head h;
            const index_t a_lane = kMma32 ? c.lane : c.lane_col;
            const uint32_t ustep = static_cast<uint32_t>(step);
            ScaleWords<kRm>(
                h.as,
                buffer + static_cast<index_t>(mx_tdm_v4_scale_offset<uint32_t>(
                             kLayout.a_scale, ustep, static_cast<uint32_t>(MRow(c, 0, a_lane)))));
            ScaleWords<kRn>(
                h.bs,
                buffer +
                    static_cast<index_t>(mx_tdm_v4_scale_offset<uint32_t>(
                        kLayout.b_scale, ustep, static_cast<uint32_t>(NRow(c, 0, c.lane_col)))));
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
                mx_tdm_v4::fence();
                cluster_barrier_signal_wave0(c.wave);
                mx_tdm_v4::fence();
            }
        }

        template <index_t StaticParity>
        CK_TILE_DEVICE void Refill(index_t k0) const
        {
            PrefetchAhead<StaticParity>(k0);
            mx_tdm_v4::fence();
            if constexpr(kLocalPrefetch)
            {
                // Tile k+1 is read next, from this point: it is the only TDM load in flight, and
                // the barrier below publishes it.
                s_wait_tensorcnt<0>();
                s_barrier();
                mx_tdm_v4::fence();
            }
        }

        CK_TILE_DEVICE void Late(index_t k0, index_t parity) const
        {
            mx_tdm_v4::fence();
            if constexpr(kHasCluster)
                cluster_barrier_wait();
            // No wave drained its reads of this buffer before the signal; the workgroup barrier
            // publishes them here.
            s_wait_dscnt_pinned<kLatePending>();
            s_barrier();
            Issue(ClampK(k0 + 2 * kTileK), parity);
            mx_tdm_v4::fence();
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
        CK_TILE_DEVICE void Wmmas(CVec (&acc)[kNumAcc], const Head& h, const BRegs& b) const
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
        CK_TILE_DEVICE void TileHalfB(CVec (&acc)[kNumAcc],
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
        CK_TILE_DEVICE void TileHalfBLate(CVec (&acc)[kNumAcc],
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
        CK_TILE_DEVICE void TileFull(CVec (&acc)[kNumAcc],
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
        TileDirect(CVec (&acc)[kNumAcc], index_t k0, index_t parity, index_t buffer) const
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
        CK_TILE_DEVICE void
        Tile(CVec (&acc)[kNumAcc], Carry& carry, index_t k0, index_t parity) const
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

        CK_TILE_DEVICE void Main(CVec (&acc)[kNumAcc])
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
            const index_t full_tiles = integer_divide_nonnegative<kTileK>(p.K);
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
                                           : (Traits::kHasTail ? full_tiles * kTileK : p.K);
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
                static_assert(Traits::kTailSteps < kKSteps, "comp_tdm_v4: tail steps < k_steps");
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

    /// Runs the K loop of the workgroup's tile into acc (one CVec per M and N repeat, index
    /// rm * kRn + rn). `smem` is GetSmemSize() bytes of LDS.
    CK_TILE_DEVICE void operator()(const mx_tdm_v4::Operands& p,
                                   const mx_tdm_v4::Coords& c,
                                   CK_TILE_LDS_ADDR uint8_t* smem,
                                   CVec (&acc)[kNumAcc]) const
    {
        Run run(p, c, smem);
        run.Main(acc);
    }
};

} // namespace ck_tile
