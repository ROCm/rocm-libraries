// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// src/core/ckc_build_id.cpp -- defines the engine freshness / provenance stamp.
//
// The two values are injected at configure time by the build system as the
// compile definitions CKC_BUILD_ID (a content hash of the engine sources) and
// CKC_ENGINE_VERSION (a human-readable version/date). If a build forgets to
// define them, this TU falls back to "unknown" so callers never see an empty
// string.
//
// NOTE ON THE FILE NAME: named with the ``ckc_`` prefix (not ``build_id.*``)
// because the repository .gitignore ignores ``build*`` -- a ``build_id.*`` file
// is silently dropped from a fresh clone, which breaks the engine build.
//
// HARD INVARIANT: this translation unit must NOT be referenced by any lowering
// or emission code path. It carries no IR-building logic; it only returns two
// string literals. Keeping it off the emission path guarantees the emitted
// LLVM-IR is byte-identical regardless of the build-id, preserving the .ll
// byte-identity contract. The build-id is an artifact stamp, never IR content.

extern "C" {
#include "ckc/ckc_build_id.h"
}

// Stringize the configure-time -D values. CKC_BUILD_ID / CKC_ENGINE_VERSION are
// passed as compile definitions; the indirection through CKC_STR forces macro
// expansion before stringization.
#ifndef CKC_BUILD_ID
#define CKC_BUILD_ID unknown
#endif
#ifndef CKC_ENGINE_VERSION
#define CKC_ENGINE_VERSION unknown
#endif

#define CKC_STR2(x) #x
#define CKC_STR(x) CKC_STR2(x)

extern "C" const char* ckc_build_id(void)
{
    return CKC_STR(CKC_BUILD_ID);
}

extern "C" const char* ckc_engine_version(void)
{
    return CKC_STR(CKC_ENGINE_VERSION);
}
