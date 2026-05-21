// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "HipModule.hpp"
#include "KernelArtifact.hpp"

namespace ck_dsl_provider {

/// Opaque cache key. Per plan §3.4 the key is a deterministic hash
/// over ``(op_kind_string, dtype_tuple, shape_tuple, stride_tuple,
/// layout_string, dsl_version_string)``. ``GraphSignature`` (I-7)
/// owns the derivation; ``JitCache`` is intentionally agnostic so a
/// future on-disk cache (M3) can reuse the same key without inheriting
/// any knowledge of how it was computed.
using SignatureHash = std::uint64_t;

/// In-memory JIT cache mapping ``SignatureHash`` → loaded ``HipModule``.
///
/// One instance per plugin handle (per plan §3.4). Cache lookups
/// happen on the plan-builder hot path before the adapter+bridge
/// compile path is invoked. M1 has no eviction policy -- modules live
/// until the handle (and therefore the cache) is destroyed. Disk
/// cache + LRU + version-based invalidation are M3 work.
///
/// Concurrency: a single mutex guards the map. ``getOrLoad`` holds
/// the mutex for the duration of the loader call so a second thread
/// that misses on the same key waits for the first compile rather
/// than racing it (no duplicate compile, no double-load of the same
/// HSACO). For M1 the provider is effectively single-threaded per
/// handle, so this lock is uncontended in practice; if M2+ surfaces
/// real concurrent compiles, the obvious next step is a per-key
/// std::shared_future so distinct keys can compile in parallel.
class JitCache {
   public:
    /// Loader signature. Returning a ``KernelArtifact`` lets the cache
    /// own the HSACO bytes inside the HipModule (and lets a future
    /// disk-cache layer round-trip them) without forcing the caller
    /// to construct the module itself.
    using Loader = std::function<KernelArtifact()>;

    JitCache() = default;
    ~JitCache() = default;

    JitCache(const JitCache&) = delete;
    JitCache& operator=(const JitCache&) = delete;
    JitCache(JitCache&&) = delete;
    JitCache& operator=(JitCache&&) = delete;

    /// Return a cached HipModule for ``key`` on hit; on miss, invoke
    /// ``loader``, construct a HipModule from the resulting artifact,
    /// insert it, and return it. The loader is invoked at most once
    /// per key for the lifetime of the cache.
    ///
    /// The returned ``shared_ptr`` aliases the cache's own ownership,
    /// so callers can safely hold the module past further cache
    /// operations -- the underlying ``hipModule_t`` stays alive until
    /// every shared_ptr is dropped AND the cache entry is gone. For
    /// M1 the cache entry never goes away before the cache itself, so
    /// in practice "alive while the handle is alive" is the lifetime
    /// to reason about.
    std::shared_ptr<HipModule> getOrLoad(SignatureHash key, const Loader& loader);

    /// Test-only: does the cache already hold an entry for ``key``?
    /// Acquires the mutex; safe to call from any thread. Not part of
    /// the production hot path -- callers should use ``getOrLoad``
    /// which avoids the double lookup.
    bool contains(SignatureHash key) const;

    /// Number of cached modules. Test-only; production code has no
    /// reason to read this and a non-stale value would require holding
    /// the lock across the read AND any subsequent decision, which
    /// callers cannot do through this interface.
    std::size_t size() const;

   private:
    mutable std::mutex _mutex;
    std::unordered_map<SignatureHash, std::shared_ptr<HipModule>> _modules;
};

}  // namespace ck_dsl_provider
