/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/rocke_build_id.h -- engine freshness / provenance stamp.
 *
 * Exposes a stable content hash of the engine sources (computed at CMake
 * configure time) and a human-readable engine version. Consumers use these to
 * detect a stale or mixed-build artifact: a shared library, a static archive,
 * and a directory of prebuilt emitters that were NOT built from the same source
 * snapshot will report different build-ids, letting callers fail loud instead of
 * silently producing spurious mismatches.
 *
 * NOTE ON THE FILE NAME: this pair is named with the ``rocke_`` prefix (mirroring
 * cmake/rocke_build_id.cmake) rather than ``build_id.*`` deliberately -- the
 * repository .gitignore ignores ``build*``, which would silently drop a
 * ``build_id.*`` file from a fresh clone and break the engine build.
 *
 * CRITICAL: these stamps live on ARTIFACTS only. The translation unit that
 * defines them (src/core/rocke_build_id.cpp) is intentionally never referenced by
 * any lowering / emission path, so the emitted LLVM-IR is unaffected and the .ll
 * byte-identity contract holds. Do not call these from any code that
 * participates in IR construction or text emission.
 */
#ifndef ROCKE_BUILD_ID_H
#define ROCKE_BUILD_ID_H

#ifdef __cplusplus
extern "C" {
#endif

/* Stable content hash (lowercase hex) of the engine source tree, computed at
 * configure time. Changes if and only if a tracked source byte changes. Never
 * empty; returns "unknown" if the build did not define a build-id. */
const char* rocke_build_id(void);

/* Human-readable engine version (semantic version or date stamp). */
const char* rocke_engine_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_BUILD_ID_H */
