// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// src/core/rocke_build_id.cpp -- defines the engine freshness / provenance stamp.
//
// The two values are injected at configure time by the build system as the
// compile definitions ROCKE_BUILD_ID (a content hash of the engine sources) and
// ROCKE_ENGINE_VERSION (a human-readable version/date). If a build forgets to
// define them, this TU falls back to "unknown" so callers never see an empty
// string.
//
// NOTE ON THE FILE NAME: named with the ``rocke_`` prefix (not ``build_id.*``)
// because the repository .gitignore ignores ``build*`` -- a ``build_id.*`` file
// is silently dropped from a fresh clone, which breaks the engine build.
//
// HARD INVARIANT: this translation unit must NOT be referenced by any lowering
// or emission code path. It carries no IR-building logic; it only returns two
// string literals. Keeping it off the emission path guarantees the emitted
// LLVM-IR is byte-identical regardless of the build-id, preserving the .ll
// byte-identity contract. The build-id is an artifact stamp, never IR content.

extern "C" {
#include "rocke/rocke_build_id.h"
}

// Stringize the configure-time -D values. ROCKE_BUILD_ID / ROCKE_ENGINE_VERSION are
// passed as compile definitions; the indirection through ROCKE_STR forces macro
// expansion before stringization.
#ifndef ROCKE_BUILD_ID
#define ROCKE_BUILD_ID unknown
#endif
#ifndef ROCKE_ENGINE_VERSION
#define ROCKE_ENGINE_VERSION unknown
#endif

#define ROCKE_STR2(x) #x
#define ROCKE_STR(x) ROCKE_STR2(x)

extern "C" const char* rocke_build_id(void)
{
    return ROCKE_STR(ROCKE_BUILD_ID);
}

extern "C" const char* rocke_engine_version(void)
{
    return ROCKE_STR(ROCKE_ENGINE_VERSION);
}
