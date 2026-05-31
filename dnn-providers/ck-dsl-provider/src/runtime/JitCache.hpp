// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "HipModule.hpp"
#include "KernelArtifact.hpp"

namespace ck_dsl_provider {

/// Opaque cache key. The key is a deterministic hash over the op-kind
/// string, every codegen-relevant field of the built
/// ``ConvImplicitGemmSpec`` (the ConvProblem shape/stride/pad/dilation
/// plus all tiling/pipeline/epilogue knobs), and the DSL version
/// string. ``GraphSignature::computeForSpec`` owns the derivation and
/// is the authoritative list of folded -- and intentionally omitted --
/// inputs; ``JitCache`` is intentionally agnostic so a future on-disk
/// cache can reuse the same key without inheriting any knowledge of how
/// it was computed.
///
/// **Precondition for an on-disk cache.** The current key is only
/// sufficient for the in-memory, single-process, FP16-only path. The
/// target arch IS folded (so the key is multi-GPU safe), but several
/// codegen inputs are still constant today and therefore not folded --
/// dtype, toolchain version, and physical tensor layout (see
/// ``GraphSignature``). A persisted cache outlives the process/build
/// that produced it, so it MUST extend the key with those inputs before
/// reusing entries, or it will hand back a module built for a different
/// dtype/toolchain/layout.
using SignatureHash = std::uint64_t;

/// In-memory JIT cache mapping ``SignatureHash`` → loaded ``HipModule``.
///
/// One instance per process, owned by ``CkDslContainer``. Cache lookups
/// happen on the plan-builder path before the adapter+bridge compile
/// path is invoked. No eviction policy today -- modules live until the
/// process exits. Disk cache + LRU + version-based invalidation are
/// future work.
///
/// **Concurrency model.** Each key is associated with a
/// ``std::shared_future<std::shared_ptr<HipModule>>``. ``getOrLoad``
/// takes the map mutex only long enough to look up or install the
/// future, then drops the mutex before invoking the loader. The thread
/// that installs the future runs the loader and fulfils the promise;
/// every other thread that finds the future already installed simply
/// waits on it. Consequences:
///
///   * Two distinct keys can compile in parallel -- a long compile of
///     key K1 does not block a fast-path lookup or compile of key K2.
///   * Each key is compiled at most once even under concurrent misses
///     (the second arrival finds the future and waits rather than
///     re-running the loader).
///   * Cache-hit lookups never wait on an in-flight compile of a
///     different key.
///
/// **Lock-ordering note for callers.** Loaders typically acquire the
/// GIL inside the closure. Because the loader runs OUTSIDE the cache
/// mutex, the only ordering constraint is whatever the loader itself
/// imposes (GIL → loader-local locks). Callers that hold other locks
/// when calling ``getOrLoad`` must keep this in mind: the mutex held
/// across ``getOrLoad`` will also be held across the loader's wait.
///
/// **Loader failure semantics.** If the loader throws,
/// ``set_exception`` is propagated to every waiting thread (each of
/// them re-throws when it calls ``future.get()``) and the entry is
/// removed from the map so a subsequent ``getOrLoad`` will re-run the
/// loader rather than re-throwing the same stale failure.
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
    /// insert it, and return it. The loader is invoked at most once per
    /// key even under concurrent misses on the same key.
    ///
    /// The returned ``shared_ptr`` is owned jointly by the cache and
    /// every caller; the underlying ``hipModule_t`` stays alive until
    /// every shared_ptr is dropped AND the cache entry is gone.
    std::shared_ptr<HipModule> getOrLoad(SignatureHash key, const Loader& loader);

    /// Test-only: does the cache already hold a (fulfilled or
    /// in-flight) entry for ``key``? Acquires the mutex; safe to call
    /// from any thread. Not part of the production hot path.
    bool contains(SignatureHash key) const;

    /// Number of cached entries. Test-only; production code has no
    /// reason to read this and a non-stale value would require holding
    /// the lock across the read AND any subsequent decision, which
    /// callers cannot do through this interface.
    std::size_t size() const;

   private:
    using SharedModule = std::shared_ptr<HipModule>;
    using SharedFuture = std::shared_future<SharedModule>;

    mutable std::mutex _mutex;
    std::unordered_map<SignatureHash, SharedFuture> _entries;
};

}  // namespace ck_dsl_provider
