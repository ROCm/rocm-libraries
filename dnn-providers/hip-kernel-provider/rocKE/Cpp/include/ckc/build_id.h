// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// ckc/build_id.h -- engine freshness / provenance stamp.
//
// Declares the two artifact-stamp accessors exposed by the engine archive.
// Their values come from compile definitions injected ONLY into the
// build_id.cpp translation unit (see CMakeLists.txt + cmake/ckc_build_id.cmake),
// so they are deliberately off every lowering/emission path and never affect
// the emitted LLVM-IR. Consumers (harnesses, the pybind language binding, the
// hipDNN provider) read them purely as an artifact stamp.
#ifndef CKC_BUILD_ID_H
#define CKC_BUILD_ID_H

#ifdef __cplusplus
extern "C" {
#endif

// Deterministic, git-independent content hash of the engine source tree
// (16 hex chars). Stable across rebuilds of identical sources.
const char *ckc_build_id(void);

// Human-readable engine version, e.g. "1.0.0+20260620".
const char *ckc_engine_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // CKC_BUILD_ID_H
