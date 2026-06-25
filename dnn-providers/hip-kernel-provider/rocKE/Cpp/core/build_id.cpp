// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// build_id.cpp -- defines the engine freshness / provenance accessors declared
// in rocke/build_id.h. The ROCKE_BUILD_ID and ROCKE_ENGINE_VERSION values are injected
// as compile definitions on THIS translation unit only (see CMakeLists.txt), so
// no emission TU ever observes them and the emitted LLVM-IR stays byte-identical
// across builds. This file is intentionally NOT referenced by any lowering path.
#include "rocke/build_id.h"

// Fallbacks so the TU still compiles if built outside the engine CMake (the
// canonical build always supplies both via -D compile definitions).
#ifndef ROCKE_BUILD_ID
#define ROCKE_BUILD_ID unknown
#endif
#ifndef ROCKE_ENGINE_VERSION
#define ROCKE_ENGINE_VERSION 0.0.0 + unknown
#endif

// Two-step stringize so the bare token injected by CMake (e.g. an unquoted hex
// digest or a 1.0.0+YYYYMMDD version) becomes a string literal.
#define ROCKE_STRINGIZE_IMPL(x) #x
#define ROCKE_STRINGIZE(x) ROCKE_STRINGIZE_IMPL(x)

extern "C" const char* rocke_build_id(void)
{
    return ROCKE_STRINGIZE(ROCKE_BUILD_ID);
}

extern "C" const char* rocke_engine_version(void)
{
    return ROCKE_STRINGIZE(ROCKE_ENGINE_VERSION);
}
