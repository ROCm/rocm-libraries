// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// build_id.cpp -- defines the engine freshness / provenance accessors declared
// in ckc/build_id.h. The CKC_BUILD_ID and CKC_ENGINE_VERSION values are injected
// as compile definitions on THIS translation unit only (see CMakeLists.txt), so
// no emission TU ever observes them and the emitted LLVM-IR stays byte-identical
// across builds. This file is intentionally NOT referenced by any lowering path.
#include "ckc/build_id.h"

// Fallbacks so the TU still compiles if built outside the engine CMake (the
// canonical build always supplies both via -D compile definitions).
#ifndef CKC_BUILD_ID
#define CKC_BUILD_ID unknown
#endif
#ifndef CKC_ENGINE_VERSION
#define CKC_ENGINE_VERSION 0.0.0 + unknown
#endif

// Two-step stringize so the bare token injected by CMake (e.g. an unquoted hex
// digest or a 1.0.0+YYYYMMDD version) becomes a string literal.
#define CKC_STRINGIZE_IMPL(x) #x
#define CKC_STRINGIZE(x) CKC_STRINGIZE_IMPL(x)

extern "C" const char* ckc_build_id(void)
{
    return CKC_STRINGIZE(CKC_BUILD_ID);
}

extern "C" const char* ckc_engine_version(void)
{
    return CKC_STRINGIZE(CKC_ENGINE_VERSION);
}
