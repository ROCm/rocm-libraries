// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// engine_freshness.hpp -- defense-in-depth stale/mixed-bundle guard.
//
// A shipped kernel bundle records the engine build-id it was produced with
// (manifest field `engine_build_id`). This provider links a specific build of
// the ck_dsl engine, which reports its own build-id via ckc_build_id(). If the
// two disagree, the bundle's prebuilt HSACOs / .ll were produced by a different
// engine than the one this provider would JIT/dispatch with -- a stale or mixed
// build that can silently produce wrong kernels. We FAIL LOUD on mismatch.
//
// This header is intentionally SEPARATE from artifact_store.hpp (which stays
// link-clean, header-only, and usable by host-only tests that do NOT link the
// engine archive). Only translation units that already link the engine include
// this header and call check_bundle_engine_freshness().
//
// Override: set CK_DSL_ALLOW_ENGINE_MISMATCH=1 to downgrade the hard error to a
// warning (default is to error).
#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

#include "ck_dsl_runtime/artifact_store.hpp"

// The engine build-id stamp. extern "C" from libckc_core.a; only TUs that link
// the engine archive may include this header.
extern "C" const char* ckc_build_id(void);

namespace ck_dsl {

// Compare every stamped manifest in `store` against the linked engine's
// build-id. On the first mismatch, throw std::runtime_error with a clear
// message (unless CK_DSL_ALLOW_ENGINE_MISMATCH=1, in which case it returns the
// offending manifest id without throwing so the caller can log a warning).
// Manifests with an empty engine_build_id (produced before the stamp existed)
// are skipped -- the stamp is additive and must not break legacy bundles.
//
// Returns the id of the first mismatching manifest (empty string == all match
// or all unstamped). Throws on mismatch unless the override env is set.
inline std::string check_bundle_engine_freshness(const ArtifactStore& store) {
    const std::string engine_id = ckc_build_id();
    const bool allow = [] {
        const char* e = std::getenv("CK_DSL_ALLOW_ENGINE_MISMATCH");
        return e != nullptr && std::string(e) == "1";
    }();
    for (const auto& [id, entry] : store.entries()) {
        const std::string& mid = entry.manifest.engine_build_id;
        if (mid.empty()) continue;  // legacy/unstamped bundle: skip
        if (mid != engine_id) {
            const std::string msg =
                "ck-dsl-provider: stale/mixed build -- kernel bundle '" + id +
                "' was produced by engine build-id " + mid +
                " but this provider is linked against engine build-id " + engine_id +
                ". Rebuild the bundle and the provider from the same engine source, or set "
                "CK_DSL_ALLOW_ENGINE_MISMATCH=1 to override (at your own risk).";
            if (allow) return id;  // caller logs a warning; do not throw
            throw std::runtime_error(msg);
        }
    }
    return std::string();
}

}  // namespace ck_dsl
