// rocwmma_16chain.hpp — Multi-chain SWMMAC XDL pipeline
// Layer 3: Orchestrates N independent accumulator chains to fill the
// 16-deep XDL pipeline.
//
// Two pipeline variants:
//   ChainPipeline<N>     — original sync API (blockIdx.x dispatch)
//   StaggeredPipeline<N,T> — wave-level staggered + B pre-load (K6 v2)
//
// Architecture:
//   Chain 0 issues at T=0, Chain 1 at T=1, ... Chain 15 at T=15
//   After pipeline fill (T=16), 1 result per cycle emerges.
//   16 chains × 8 accum regs = 128 accum VGPRs
//
// Key discoveries:
//   2026-05-15: ChainPipeline uses blockIdx.x → all waves issue SWMMAC
//     in lockstep → VGPR read port contention → IPC collapses to 0.115.
//     StaggeredPipeline: atomicAdd work-claim → natural wave staggering
//     → IPC recovers to 0.621 (5.4× over sync).
//
//   2026-05-18: RDNA4 HWXDL silent drop discovered — thread-level atomicAdd
//     causes EXEC divergence → XDL pipeline suppresses VGPR write-back.
//     Fixed via wave-level __builtin_amdgcn_readfirstlane broadcast:
//     lane 0 claims, all 32 lanes receive unified task_idx → EXEC=0xFFFFFFFF.
//     See: /data/rtl-sdr/swmmac/active/silent_drop/DISCOVERY.md
//
// Usage:
//   #include <rocwmma/rocwmma_16chain.hpp>
//
//   // Sync (original, ~675 TOPs)
//   rocwmma::ChainPipeline<16> pipe;
//   pipe.zero();
//   for (int i = 0; i < LOOPS; i++) pipe.step(A, B, 0);
//   pipe.store(C);
//
//   // Staggered v2 — wave-level readfirstlane (EXEC-secure, ~4190 TOPs)
//   // Lane 0 claims 32 tiles at once via atomicAdd(counter, 32),
//   // broadcasts base to all lanes. Each lane gets base+threadIdx.x
//   // independent tile → EXEC=0xFFFFFFFF, 32 tiles per wave.
//   __global__ void kernel(..., int* counter) {
//       int wb = 0;
//       if (threadIdx.x == 0) wb = atomicAdd(counter, 32);
//       wb = __builtin_amdgcn_readfirstlane(wb);
//       if (wb >= TOTAL) return;
//       int w = wb + threadIdx.x;
//       rocwmma::StaggeredPipeline<16, 1> sp;
//       sp.zero();
//       sp.load_B(B_ptr, w);
//       sp.run(A_ptr, w, LOOPS);
//       sp.store(C + w * 16 * 8);
//   }
//       sp.store(C_ptr + w * 16 * 8);
//   }
//   // Launch with 2× TOTAL blocks for optimal stagger
#pragma once

#include "rocwmma_swmmac.hpp"

#if ROCWMMA_HAS_SWMMAC

namespace rocwmma {

// ============================================================================
// ChainPipeline<NCHAINS, Backend> — N independent SWMMAC accumulator chains
//
// DEPRECATED: Use StaggeredPipeline for production (5.4× throughput).
// ChainPipeline uses blockIdx.x sync dispatch → all waves lockstep →
// VGPR read port contention. Retained for educational/reference use.
//
// @tparam NCHAINS  16 (peak throughput) or 14 (dual-wave compatible)
// @tparam Backend  SwmmacI4 (INT4, default) or SwmmacI8 (INT8)
// ============================================================================
template <uint32_t NCHAINS = 16, typename Backend = SwmmacI4>
struct ChainPipeline {
    static_assert(NCHAINS == 14 || NCHAINS == 16,
        "ChainPipeline supports 14 or 16 chains");

    // ---- Types derived from Backend ----
    using ARegsT     = typename Backend::ARegsT;
    using BRegsT     = typename Backend::BRegsT;
    using CRegsT     = typename Backend::CRegsT;
    using AccumT     = typename VecTraits<CRegsT>::DataT;     // int32_t or float32_t
    using APtrT      = typename VecTraits<ARegsT>::DataT;     // element type for A ptr
    using BPtrT      = typename VecTraits<BRegsT>::DataT;     // element type for B ptr

    // ---- Derive ops from backend Block dimensions ----
    static constexpr uint32_t BLOCK_M       = Backend::BlockM;
    static constexpr uint32_t BLOCK_N       = Backend::BlockN;
    static constexpr uint32_t BLOCK_K       = Backend::BlockK;
    static constexpr double   OPS_PER_SW    = BLOCK_M * BLOCK_N * BLOCK_K * 2.0;
    static constexpr double   OPS_PER_CYCLE = OPS_PER_SW / 16.0;

    // ---- VGPR budget (measured on LLVM 23 / gfx1200 / -O3) ----
    static constexpr uint32_t VGPR_PER_CHAIN = 8;
    static constexpr uint32_t VGPR_OVERHEAD   = 7u;
    static constexpr uint32_t TOTAL_VGPR      = NCHAINS * VGPR_PER_CHAIN + VGPR_OVERHEAD;
    static constexpr uint32_t WAVES_PER_SIMD  = (TOTAL_VGPR <= 128) ? 2u : 1u;

    alignas(32) AccumT acc[NCHAINS][8];

    // ---- Accumulator management ----
    __device__ void zero() {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[c][i] = AccumT{0};
    }

    __device__ void load(AccumT const* C) {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[c][i] = C[c * 8 + i];
    }

    __device__ void store(AccumT* C) const {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) C[c * 8 + i] = acc[c][i];
    }

    // ---- Single chain dispatch to backend ----
    __device__ __attribute__((always_inline)) static inline CRegsT
    swmmac_chain(ARegsT const& a, BRegsT const& b,
                 CRegsT const& c, int32_t idx)
    {
        return Backend::exec(a, b, c, idx);
    }

    // ---- N-chain step — fills XDL pipeline ----
    __device__ void step(
        APtrT const* __restrict__ A,
        BPtrT const* __restrict__ B,
        int32_t sparse_idx = 0)
    {
        ARegsT const& a = *reinterpret_cast<ARegsT const*>(A);
        BRegsT const& b = *reinterpret_cast<BRegsT const*>(B);

        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c) {
            CRegsT& accum = *reinterpret_cast<CRegsT*>(acc[c]);
            accum = swmmac_chain(a, b, accum, sparse_idx);
        }
    }

    // ---- Sparse step convenience — uses 2:4 structured sparsity ----
    __device__ void step_sparse(
        APtrT const* __restrict__ A,
        BPtrT const* __restrict__ B)
    {
        step(A, B, static_cast<int32_t>(SparseSel::SPARSE));
    }

    // ---- Dual-buffer step ----
    __device__ void step_dual(
        APtrT const* __restrict__ A0, BPtrT const* __restrict__ B0,
        APtrT const* __restrict__ A1, BPtrT const* __restrict__ B1,
        int32_t sparse_idx = 0)
    {
        constexpr uint32_t HALF = NCHAINS / 2;
        ARegsT const& a0 = *reinterpret_cast<ARegsT const*>(A0);
        BRegsT const& b0 = *reinterpret_cast<BRegsT const*>(B0);
        ARegsT const& a1 = *reinterpret_cast<ARegsT const*>(A1);
        BRegsT const& b1 = *reinterpret_cast<BRegsT const*>(B1);

        #pragma unroll
        for (uint32_t c = 0; c < HALF; ++c) {
            CRegsT& accum = *reinterpret_cast<CRegsT*>(acc[c]);
            accum = swmmac_chain(a0, b0, accum, sparse_idx);
        }
        #pragma unroll
        for (uint32_t c = HALF; c < NCHAINS; ++c) {
            CRegsT& accum = *reinterpret_cast<CRegsT*>(acc[c]);
            accum = swmmac_chain(a1, b1, accum, sparse_idx);
        }
    }

    // ---- Theoretical peak (naive, 1 SWMMAC/cycle/SIMD) ----
    // NOTE: This is the IDEAL limit. Real throughput is ~74% of this
    // due to VGPR read bandwidth + wave slot constraints.
    // Use StaggeredPipeline::predicted_tops() for realistic estimates.
    __host__ __device__ static constexpr double theoretical_tops(
        double   gpu_clock_ghz = 3.15,
        uint32_t num_cus       = 32)
    {
        return OPS_PER_CYCLE * 2.0 * num_cus * gpu_clock_ghz / 1000.0;
    }
};

// ============================================================================
// StaggeredPipeline<NCHAINS, TILES, Backend> — atomic wave-staggered pipeline
//
// K6 pattern: pre-load B tiles into VGPR → wave-staggered SWMMAC execution.
// Eliminates wave lockstep (28K VGPR reads/cycle contention) by using
// atomicAdd work-claim at kernel entry. IPC: 0.115→0.621 (5.4×).
//
// @tparam NCHAINS  16 (peak throughput) or 14 (dual-wave compatible)
// @tparam TILES    1-8 B tiles to pre-load per wave (1 = peak throughput)
// @tparam Backend  SwmmacI4 (default, INT4) or SwmmacI8 (INT8)
//
// Kernel template:
//   __global__ void my_kernel(int32_t* C, int32_t const* A,
//                              int32_t const* B, int L, int* counter) {
//       int w = atomicAdd(counter, 1);
//       if (w >= TOTAL_WAVES) return;
//
//       StaggeredPipeline<16, 1> sp;
//       sp.zero();
//       sp.load_B(B, w);
//       sp.run(A, w, L);
//       sp.store(C + w * 16 * 8);
//   }
//   // Launch: <<<TOTAL_WAVES * 2, 32>>> for optimal 2× oversubscription
// ============================================================================
template <uint32_t NCHAINS = 16, uint32_t TILES = 1, typename Backend = SwmmacI4>
struct StaggeredPipeline {
    static_assert(NCHAINS == 14 || NCHAINS == 16,
        "StaggeredPipeline supports 14 or 16 chains");

    using ARegsT     = typename Backend::ARegsT;
    using BRegsT     = typename Backend::BRegsT;
    using CRegsT     = typename Backend::CRegsT;
    using AccumT     = typename VecTraits<CRegsT>::DataT;
    using APtrT      = typename VecTraits<ARegsT>::DataT;
    using BPtrT      = typename VecTraits<BRegsT>::DataT;

    static constexpr uint32_t BLOCK_M       = Backend::BlockM;
    static constexpr uint32_t BLOCK_N       = Backend::BlockN;
    static constexpr uint32_t BLOCK_K       = Backend::BlockK;
    static constexpr double   OPS_PER_SW    = BLOCK_M * BLOCK_N * BLOCK_K * 2.0;
    static constexpr uint32_t VGPR_PER_CHAIN = 8;
    static constexpr uint32_t VGPR_OVERHEAD   = 7u;
    static constexpr uint32_t B_LOAD_VGPR     = TILES * 4u;
    static constexpr uint32_t TOTAL_VGPR      = NCHAINS * VGPR_PER_CHAIN + VGPR_OVERHEAD + B_LOAD_VGPR;
    static constexpr uint32_t WAVES_PER_SIMD  = (TOTAL_VGPR <= 128) ? 2u : 1u;

    // Accumulator chains
    alignas(32) AccumT acc[NCHAINS][8];

    // B tile pre-load buffer (loaded once, reused for all loop iterations)
    alignas(32) BPtrT bt[TILES][4];

    // ---- Accumulator management (force inline to match raw K6) ----
    __device__ __forceinline__ void zero() {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[c][i] = AccumT{0};
    }

    __device__ __forceinline__ void load(AccumT const* C) {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) acc[c][i] = C[c * 8 + i];
    }

    __device__ __forceinline__ void store(AccumT* C) const {
        #pragma unroll
        for (uint32_t c = 0; c < NCHAINS; ++c)
            #pragma unroll
            for (int i = 0; i < 8; ++i) C[c * 8 + i] = acc[c][i];
    }

    // ---- Pre-load B tiles into VGPR (call once before run) ----
    // The B tiles are reused across all loop iterations, eliminating
    // redundant global memory loads from the hot path.
    __device__ __forceinline__ void load_B(BPtrT const* __restrict__ B, uint32_t worker_id) {
        #pragma unroll
        for (uint32_t t = 0; t < TILES; ++t)
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                bt[t][j] = B[(worker_id * TILES + t) * 4 + j];
    }

    // ---- Single chain dispatch to backend ----
    __device__ __attribute__((always_inline)) static inline CRegsT
    swmmac_chain(ARegsT const& a, BRegsT const& b,
                 CRegsT const& c, int32_t idx)
    {
        return Backend::exec(a, b, c, idx);
    }

    // ---- Run SWMMAC: tiles × loops × chains (pre-loaded B) ----
    // After load_B(), call this to execute the full SWMMAC pipeline.
    __device__ __forceinline__ void run(
        APtrT const* __restrict__ A,
        uint32_t worker_id,
        int loops,
        int32_t sparse_idx = 0)
    {
        ARegsT const& a = *reinterpret_cast<ARegsT const*>(A + worker_id * 2);

        #pragma unroll
        for (uint32_t t = 0; t < TILES; ++t) {
            BRegsT const& b = *reinterpret_cast<BRegsT const*>(bt[t]);
            for (int i = 0; i < loops; ++i) {
                #pragma unroll
                for (uint32_t c = 0; c < NCHAINS; ++c) {
                    CRegsT& accum = *reinterpret_cast<CRegsT*>(acc[c]);
                    accum = swmmac_chain(a, b, accum, sparse_idx);
                }
            }
        }
    }

        // ---- 26-cycle model: predicted throughput (HW-validated 2026-05-16) ----
        // Constants from gfx1200 measurement + LLVM SISchedule.td + rocprofv3:
        //   SWMMAC exec:      16 cyc    VGPR overhead:  ~10 cyc
        //   Effective pipe:   26 cyc    Wave slots/SIMD: 16
        //   Sync IPC:         0.615     L2 wrap boost:   +21%
        __host__ __device__ static double predicted_tops(
            uint32_t total_waves=1024, uint32_t loops=160,
            double mhz=2780.0, uint32_t cus=32, bool l2_wrap=true)
        {
            uint32_t simd = cus*2;
            double waves_per_simd = (double)total_waves / (double)simd;
            double ipc = waves_per_simd / 26.0;  // 26-cycle effective pipeline
            if(ipc > 1.0) ipc = 1.0;
            if(l2_wrap) ipc *= 1.21;
            if(ipc > 1.0) ipc = 1.0;
            double theory = OPS_PER_SW * (double)simd * mhz / 1e6;
            return theory * ipc;
        }
        __host__ __device__ static constexpr double model_eff_pipeline(){ return 26.0; }
        __host__ __device__ static constexpr double model_l2_boost()     { return 1.21; }
    };

    // ============================================================================
    // Convenience aliases
    // ============================================================================

// INT4 (default backend)
using ChainPipelineInt4 = ChainPipeline<16, SwmmacI4>;
using Chain16           = ChainPipeline<16, SwmmacI4>;
using Chain14           = ChainPipeline<14, SwmmacI4>;

// INT8
template <uint32_t NC = 16>
using ChainPipelineInt8 = ChainPipeline<NC, SwmmacI8>;
using Chain16Int8       = ChainPipeline<16, SwmmacI8>;

// FP8/BF8 — same A/B layout as INT, f32 accum
using ChainFp8Fp8 = ChainPipeline<16, SwmmacFp8Fp8>;
using ChainFp8Bf8 = ChainPipeline<16, SwmmacFp8Bf8>;
using ChainBf8Fp8 = ChainPipeline<16, SwmmacBf8Fp8>;
using ChainBf8Bf8 = ChainPipeline<16, SwmmacBf8Bf8>;

// FP16/BF16 — wider A/B register layout
using ChainFp16 = ChainPipeline<16, SwmmacFp16>;
using ChainBf16 = ChainPipeline<16, SwmmacBf16>;

// MXFP4 (INT4 HW + UE8M0 scale, float accum path)
using ChainPipelineMXFP4 = ChainPipeline<16, MXFp4>;
using Chain16MXFP4       = ChainPipeline<16, MXFp4>;

// ============================================================================
// Production usage: L2-persistent wrap-counter + wave-level readfirstlane
//
//   // Host: allocate counter ONCE, never reset
//   hipMalloc(&cnt, 4); hipMemset(cnt, 0, 4);
//   int base = 0;
//   constexpr int PER_LAUNCH = TT; // 1 wave × TT blocks (readfirstlane)
//
//   // Kernel: wave-level claim — lane 0 claims 32 tiles via atomicAdd(cnt,32)
//   // All 32 lanes get base via readfirstlane, then w = wb + threadIdx.x
//   // → each lane has independent tile, EXEC=0xFFFFFFFF, silent-drop-free
//   __global__ void kernel(..., int* cnt, int base) {
//       int wb = 0;
//       if (threadIdx.x == 0) wb = atomicAdd(cnt, 32) - base;
//       wb = __builtin_amdgcn_readfirstlane(wb);
//       if (wb >= TT) return;
//       int w = wb + threadIdx.x;
//       rocwmma::StaggeredPipeline<16, 1> sp;
//       sp.zero(); sp.load_B(B, w); sp.run(A, w, loops);
//       sp.store(C + w * 16 * 8);
//   }
//
//   // Launch loop: advance base, never memset cnt
//   constexpr int PER_LAUNCH = TT * 32; // counter advances by 32 per wave × TT waves
//   for (int batch = 0; batch < num_batches; ++batch) {
//       kernel<<<TT, 32>>>(..., cnt, base);
//       hipDeviceSynchronize();
//       base += PER_LAUNCH;  // counter accumulates, stays in L2
//   }
//
// Eliminates hipMemset (L2 eviction). Wave-level atomicAdd(cnt, 32) claims
// 32 tiles per atomic — 32× fewer atomics than thread-level, reducing L2
// contention while maintaining exact same workload. EXEC=0xFFFFFFFF guaranteed.
// Measured 4190 TOPs (71.9% theory) + guaranteed ZERO silent drop.
// ============================================================================
// StaggeredPipeline aliases (K6 pattern — atomic wave-staggered)
// ============================================================================

// INT4 (default, 1 tile)
template <uint32_t NC = 16, uint32_t TL = 1>
using StagPipeI4 = StaggeredPipeline<NC, TL, SwmmacI4>;
using Stag16      = StaggeredPipeline<16, 1, SwmmacI4>;
using Stag14      = StaggeredPipeline<14, 1, SwmmacI4>;

// INT8
template <uint32_t NC = 16, uint32_t TL = 1>
using StagPipeI8 = StaggeredPipeline<NC, TL, SwmmacI8>;
using Stag16I8   = StaggeredPipeline<16, 1, SwmmacI8>;

// FP8/BF8
using StagFp8Fp8 = StaggeredPipeline<16, 1, SwmmacFp8Fp8>;
using StagFp8Bf8 = StaggeredPipeline<16, 1, SwmmacFp8Bf8>;
using StagBf8Fp8 = StaggeredPipeline<16, 1, SwmmacBf8Fp8>;
using StagBf8Bf8 = StaggeredPipeline<16, 1, SwmmacBf8Bf8>;

// FP16/BF16
using StagFp16 = StaggeredPipeline<16, 1, SwmmacFp16>;
using StagBf16 = StaggeredPipeline<16, 1, SwmmacBf16>;

// MXFP4 (INT4 HW + UE8M0 scale, float accum path)
template <uint32_t NC = 16, uint32_t TL = 1>
using StagPipeMXFP4 = StaggeredPipeline<NC, TL, MXFp4>;
using Stag16MXFP4   = StaggeredPipeline<16, 1, MXFp4>;

} // namespace rocwmma

#endif // ROCWMMA_HAS_SWMMAC
