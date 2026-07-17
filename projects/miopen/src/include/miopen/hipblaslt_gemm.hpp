// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#ifndef GUARD_MIOPEN_HIPBLASLT_GEMM_HPP_
#define GUARD_MIOPEN_HIPBLASLT_GEMM_HPP_

#include <miopen/config.h>

#if MIOPEN_USE_HIPBLASLT

#include <miopen/export_internals.h>

#include <cstddef>
#include <memory>

namespace miopen {

// Identifies a hipBLASLt GEMM problem for the purpose of reusing the matrix
// layout objects, matmul descriptor, preference object and algorithm chosen by
// hipblasLtMatmulAlgoGetHeuristic across repeated calls with the same shape.
//
// The key intentionally has no hipBLASLt type dependencies so this header can
// be included anywhere miopen::Handle is visible without pulling in
// hipblaslt.h. The full descriptor cache is PIMPL'd for the same reason.
struct hipblaslt_gemm_cache_key
{
    bool transA;
    bool transB;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
    int batch_count;
    long long strideA;
    long long strideB;
    long long strideC;
    bool skip_batches;
    int type_AB; // hipDataType cast to int for trivial equality
    int type_C;

    bool operator==(const hipblaslt_gemm_cache_key&) const = default;
};

struct hipblaslt_gemm_cache_key_hash
{
    MIOPEN_INTERNALS_EXPORT std::size_t
    operator()(const hipblaslt_gemm_cache_key& k) const noexcept;
};

// Owns the hipBLASLt objects associated with a single cached shape. Defined in
// hipblaslt_gemm.cpp because its destructor calls hipBLASLt destroy functions.
struct hipblaslt_gemm_cache_entry;

// Per-Handle cache of hipBLASLt descriptor objects and the heuristic-selected
// algorithm. The original miopen_hipblasLt_gemm implementation recreated four
// matrix layouts, the matmul descriptor and the preference object and ran a
// heuristic lookup on every CallGemm, costing ~7 us/call on small GEMMs (see
// the MIOpen-RNN LSTM hot path). Caching by shape brings the steady-state
// path down to a single hipblasLtMatmul launch.
//
// The cache is not bounded; population is the number of distinct GEMM shapes
// used by the model, which is small (O(100) for typical architectures).
// KernelCache follows the same unbounded policy.
class hipblaslt_gemm_cache
{
public:
    MIOPEN_INTERNALS_EXPORT hipblaslt_gemm_cache();
    MIOPEN_INTERNALS_EXPORT ~hipblaslt_gemm_cache();

    hipblaslt_gemm_cache(const hipblaslt_gemm_cache&)            = delete;
    hipblaslt_gemm_cache& operator=(const hipblaslt_gemm_cache&) = delete;
    hipblaslt_gemm_cache(hipblaslt_gemm_cache&&)                 = delete;
    hipblaslt_gemm_cache& operator=(hipblaslt_gemm_cache&&)      = delete;

    // Returns the entry for `key` if present, otherwise nullptr.
    MIOPEN_INTERNALS_EXPORT hipblaslt_gemm_cache_entry*
    find(const hipblaslt_gemm_cache_key& key) const;

    // Inserts `entry` under `key` if no entry exists yet, and returns the
    // resident entry (which may be the just-inserted one or one inserted
    // concurrently between the caller's find and this insert).
    MIOPEN_INTERNALS_EXPORT hipblaslt_gemm_cache_entry*
    insert(const hipblaslt_gemm_cache_key& key, std::unique_ptr<hipblaslt_gemm_cache_entry> entry);

private:
    struct impl;
    std::unique_ptr<impl> pimpl_;
};

} // namespace miopen

#endif // MIOPEN_USE_HIPBLASLT
#endif // GUARD_MIOPEN_HIPBLASLT_GEMM_HPP_
