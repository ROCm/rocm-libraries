// rocwmma_fragment_swmmac.hpp — rocWMMA fragment API → SWMMAC bridge
//
// Bridges rocWMMA's fragment<> + load_matrix_sync + store_matrix_sync
// with the SWMMAC backend. Enables drop-in use of SWMMAC in rocWMMA code.
//
// Usage:
//   #include <rocwmma/rocwmma_fragment_swmmac.hpp>
//
//   fragment<accumulator, 16, 16, 64, int32_t> d;
//   fragment<matrix_a, 16, 16, 64, int32_t> a;
//   fragment<matrix_b, 16, 16, 64, int32_t> b;
//
//   load_matrix_sync(a, A_ptr, lda);   // standard rocWMMA load
//   load_matrix_sync(b, B_ptr, ldb);
//   fill_fragment(d, 0);
//
//   rocwmma::swmmac_mma(d, a, b, d);   // SWMMAC compute
//
//   store_matrix_sync(C_ptr, d, ldc);  // standard rocWMMA store
//
// Design: INT4 values are stored as nibble-packed i32 in fragments.
// Each i32 holds 8 INT4 values — exactly the SWMMAC hardware format.
// The fragment system treats int32_t as the base type (PackRatio=1).
#pragma once

#include "rocwmma.hpp"
#include "rocwmma_16chain.hpp"

namespace rocwmma {

// ============================================================================
// Fragment storage extraction helpers
//
// rocWMMA fragments store data in packed VGPR vectors.
// For int32_t data type (PackRatio=1), the packed storage IS the register.
// We extract the raw i32 array from the fragment's internal storage.
// ============================================================================

// Extract raw i32 pointer from a fragment's packed storage
template <typename FragT>
__device__ inline auto frag_data(FragT& f) -> decltype(&(*f)[0]) {
    return &(*f)[0];
}
template <typename FragT>
__device__ inline auto frag_data(FragT const& f) -> decltype(&(*f)[0]) {
    return &(*f)[0];
}

// ============================================================================
// swmmac_mma — SWMMAC matrix multiply-accumulate for fragments
//
// Computes: D = A × B + C
//   A: fragment<matrix_a, 16, 16, 64, int32_t>  (2 i32 = 16 INT4)
//   B: fragment<matrix_b, 16, 16, 64, int32_t>  (4 i32 = 32 INT4)
//   C: fragment<accumulator, 16, 16, 64, int32_t> (8 i32, input)
//   D: fragment<accumulator, 16, 16, 64, int32_t> (8 i32, output)
//
// For 16-chain XDL pipeline: use swmmac_mma_16chain() instead.
// ============================================================================

template <typename FragA, typename FragB, typename FragC, typename FragD>
__device__ inline void swmmac_mma(
    FragD& d, FragA const& a, FragB const& b, FragC const& c)
{
    static_assert(FragA::num_elements >= 2, "A fragment needs ≥2 i32 for INT4 SWMMAC");
    static_assert(FragB::num_elements >= 4, "B fragment needs ≥4 i32 for INT4 SWMMAC");
    static_assert(FragC::num_elements >= 8, "C fragment needs ≥8 i32 for INT4 SWMMAC");

    auto da = frag_data(a);
    auto db = frag_data(b);
    auto dc = frag_data(c);
    auto dd = frag_data(d);

    // Single SWMMAC: D = A×B + C
    SwmmacARegsT const& ra = *reinterpret_cast<SwmmacARegsT const*>(da);
    SwmmacBRegsT const& rb = *reinterpret_cast<SwmmacBRegsT const*>(db);
    SwmmacAccumT const& rc = *reinterpret_cast<SwmmacAccumT const*>(dc);
    SwmmacAccumT&       rd = *reinterpret_cast<SwmmacAccumT*>(dd);

    rd = SwmmacI4::exec(ra, rb, rc, 0);
}

// ============================================================================
// swmmac_mma_sparse — SWMMAC with 2:4 structured sparsity
// ============================================================================
template <typename FragA, typename FragB, typename FragC, typename FragD>
__device__ inline void swmmac_mma_sparse(
    FragD& d, FragA const& a, FragB const& b, FragC const& c, int32_t sparse_idx)
{
    auto da = frag_data(a);
    auto db = frag_data(b);
    auto dc = frag_data(c);
    auto dd = frag_data(d);

    SwmmacARegsT const& ra = *reinterpret_cast<SwmmacARegsT const*>(da);
    SwmmacBRegsT const& rb = *reinterpret_cast<SwmmacBRegsT const*>(db);
    SwmmacAccumT const& rc = *reinterpret_cast<SwmmacAccumT const*>(dc);
    SwmmacAccumT&       rd = *reinterpret_cast<SwmmacAccumT*>(dd);

    rd = SwmmacI4::exec(ra, rb, rc, sparse_idx);
}

// ============================================================================
// swmmac_mma_16chain — 16-chain XDL pipeline via fragment API
//
// All 16 accumulators stored in a single fragment:
//   fragment<accumulator, 16, 16*16, 64, int32_t> d;  // 16×8 = 128 i32
// ============================================================================
template <typename FragA, typename FragB, typename FragAccum>
__device__ inline void swmmac_mma_16chain(
    FragAccum& accum, FragA const& a, FragB const& b,
    int loops, int32_t sparse_idx = 0)
{
    auto da = frag_data(a);
    auto db = frag_data(b);
    auto dacc = frag_data(accum);

    ChainPipeline<16> pipe;
    pipe.zero();
    for (int i = 0; i < loops; ++i)
        pipe.step(da, db, sparse_idx);
    pipe.store(dacc);
}

// ============================================================================
// swmmac_mma_int8 — INT8 SWMMAC via fragment API
// ============================================================================
template <typename FragA, typename FragB, typename FragC, typename FragD>
__device__ inline void swmmac_mma_int8(
    FragD& d, FragA const& a, FragB const& b, FragC const& c)
{
    auto da = frag_data(a);
    auto db = frag_data(b);
    auto dc = frag_data(c);
    auto dd = frag_data(d);

    SwmmacARegsT const& ra = *reinterpret_cast<SwmmacARegsT const*>(da);
    SwmmacBRegsT const& rb = *reinterpret_cast<SwmmacBRegsT const*>(db);
    SwmmacAccumT const& rc = *reinterpret_cast<SwmmacAccumT const*>(dc);
    SwmmacAccumT&       rd = *reinterpret_cast<SwmmacAccumT*>(dd);

    rd = SwmmacI8::exec(ra, rb, rc, 0);
}

// ============================================================================
// atomic_mma_16chain — K6 wave-staggered 16-chain via fragment API
//
// Uses StaggeredPipeline for atomic wave staggering (5.4× over sync).
// Caller MUST provide a global atomic counter and launch 2× blocks.
//
// Kernel template (wave-level readfirstlane, EXEC-secure, 32 tiles/claim):
//   __global__ void kernel(int32_t* C, int32_t const* A,
//                           int32_t const* B, int L, int* counter) {
//       int wb = 0;
//       if (threadIdx.x == 0) wb = atomicAdd(counter, 32);
//       wb = __builtin_amdgcn_readfirstlane(wb);
//       if (wb >= TOTAL_WAVES) return;
//       int w = wb + threadIdx.x;
//
//       fragment<accumulator, 16, 16*16, 64, int32_t> d;
//       fragment<matrix_a, 16, 16, 64, int32_t> a;
//       fragment<matrix_b, 16, 16, 64, int32_t> b;
//       load_matrix_sync(a, A + w * 2, 16);
//       load_matrix_sync(b, B + w * 4, 16);
//
//       rocwmma::atomic_mma_16chain(d, a, b, L, w, B);
//       store_matrix_sync(C + w * 16 * 8, d, 16 * 16);
//   }
// ============================================================================
template <typename FragA, typename FragB, typename FragAccum>
__device__ inline void atomic_mma_16chain(
    FragAccum& accum, FragA const& a, FragB const& b,
    int loops, uint32_t worker_id,
    typename VecTraits<SwmmacBRegsT>::DataT const* B_ptr,
    int32_t sparse_idx = 0)
{
    auto da = frag_data(a);
    auto dacc = frag_data(accum);

    StaggeredPipeline<16, 1, SwmmacI4> sp;
    sp.zero();
    sp.load_B(B_ptr, worker_id);
    sp.run(da, worker_id, loops, sparse_idx);
    sp.store(dacc);
}

// ============================================================================
// Convenience: auto-dispatch mma for the best available backend
//
// gfx12 (SWMMAC): auto_mma → swmmac_mma (single SWMMAC, backward compatible)
//   For MAX throughput: use StaggeredPipeline with wrap-counter pattern.
//   See rocwmma_16chain.hpp for the production-optimized API.
// gfx11 (WMMA):   auto_mma → mma_sync (rocWMMA native)
//
// Note: auto_mma is the simplest API (drop-in replacement for mma_sync).
// For production at >4000 TOPs, use StaggeredPipeline directly.
// ============================================================================
#if ROCWMMA_HAS_SWMMAC
template <typename FragD, typename FragA, typename FragB, typename FragC>
__device__ inline void auto_mma(FragD& d, FragA const& a, FragB const& b, FragC const& c) {
    swmmac_mma(d, a, b, c);
}
#else
template <typename FragD, typename FragA, typename FragB, typename FragC>
__device__ inline void auto_mma(FragD& d, FragA const& a, FragB const& b, FragC const& c) {
    mma_sync(d, a, b, c);  // fallback to rocWMMA native
}
#endif

} // namespace rocwmma
