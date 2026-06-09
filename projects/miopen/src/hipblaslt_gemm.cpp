// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <miopen/hipblaslt_gemm.hpp>

#if MIOPEN_USE_HIPBLASLT

#include <miopen/hipblaslt_gemm_impl.hpp>

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace miopen {

std::size_t
hipblaslt_gemm_cache_key_hash::operator()(const hipblaslt_gemm_cache_key& k) const noexcept
{
    // boost::hash_combine pattern using the 64-bit golden-ratio constant.
    std::size_t h = 0;
    auto mix      = [&](std::size_t v) {
        h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    };
    mix(static_cast<std::size_t>(k.transA) | (static_cast<std::size_t>(k.transB) << 1) |
        (static_cast<std::size_t>(k.skip_batches) << 2));
    mix(static_cast<std::size_t>(k.m));
    mix(static_cast<std::size_t>(k.n));
    mix(static_cast<std::size_t>(k.k));
    mix(static_cast<std::size_t>(k.lda));
    mix(static_cast<std::size_t>(k.ldb));
    mix(static_cast<std::size_t>(k.ldc));
    mix(static_cast<std::size_t>(k.batch_count));
    mix(static_cast<std::size_t>(k.strideA));
    mix(static_cast<std::size_t>(k.strideB));
    mix(static_cast<std::size_t>(k.strideC));
    mix(static_cast<std::size_t>(k.type_AB));
    mix(static_cast<std::size_t>(k.type_C));
    return h;
}

struct hipblaslt_gemm_cache::impl
{
    using map_type = std::unordered_map<hipblaslt_gemm_cache_key,
                                        std::unique_ptr<hipblaslt_gemm_cache_entry>,
                                        hipblaslt_gemm_cache_key_hash>;

    mutable std::shared_mutex mutex;
    map_type entries;
};

hipblaslt_gemm_cache::hipblaslt_gemm_cache() : pimpl_(std::make_unique<impl>()) {}

hipblaslt_gemm_cache::~hipblaslt_gemm_cache() = default;

hipblaslt_gemm_cache_entry*
hipblaslt_gemm_cache::find(const hipblaslt_gemm_cache_key& key) const
{
    std::shared_lock<std::shared_mutex> guard{pimpl_->mutex};
    auto it = pimpl_->entries.find(key);
    return it == pimpl_->entries.end() ? nullptr : it->second.get();
}

hipblaslt_gemm_cache_entry*
hipblaslt_gemm_cache::insert(const hipblaslt_gemm_cache_key& key,
                             std::unique_ptr<hipblaslt_gemm_cache_entry> entry)
{
    std::unique_lock<std::shared_mutex> guard{pimpl_->mutex};
    auto [it, inserted] = pimpl_->entries.emplace(key, std::move(entry));
    return it->second.get();
}

} // namespace miopen

#endif // MIOPEN_USE_HIPBLASLT
