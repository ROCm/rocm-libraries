/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/abi.h -- the two compatibility contracts between a rocke engine and the
 * things that talk to it.
 *
 * These are NOT the same as rocke/rocke_build_id.h. Those stamps are provenance:
 * a content hash and a date, which change on every commit and every build. They
 * answer "were these artifacts built together". Comparing them for
 * compatibility would force a lockstep upgrade of the entire stack on any source
 * change, which is why they cannot serve as the contract and why these exist.
 *
 * There are two numbers here because two different things can be mismatched,
 * for different reasons, with different blast radii. Conflating them would mean
 * that adding a recipe instruction invalidates every hipDNN binary, and that
 * changing a struct invalidates every bundle on disk. Neither is true.
 *
 *   ROCKE_ABI_VERSION   BINARY. Does this header match this .so? Governs struct
 *                       layouts, enum values and function signatures -- the
 *                       things a C++ caller and the ctypes bindings in
 *                       src/online.py hard-code at their own build time. A
 *                       mismatch here is memory-unsafe, not merely wrong, so
 *                       callers should check it once at load and refuse.
 *
 *   ROCKE_RECIPE_ABI    WIRE. Can this engine correctly read this CBOR artifact?
 *                       Governs the recipe/bundle/guard content vocabulary. A
 *                       bundle is a persisted artifact that outlives the engine
 *                       that wrote it and may be read by an engine older or
 *                       newer than the writer, so this is a data-format question
 *                       and is answered per artifact, not per process.
 *
 * -------------------------------------------------------------------- wire
 * The wire check is deliberately NOT "does the artifact's version equal mine".
 * Each artifact declares the OLDEST reader that can read it correctly:
 *
 *     "abi": {"min_reader": 1, "writer": 1, "engine": "...", "build_id": "..."}
 *
 * and a reader refuses exactly when `min_reader > ROCKE_RECIPE_ABI`. The
 * generator derives min_reader from what the artifact actually USES (see
 * python/rocke/portable_ir/src/abi.py), so a recipe built by a newer generator
 * that happens to use only old constructs still declares 1 and every engine
 * reads it. An equality check, or a plain monotonic format version, would
 * instead reject every newer bundle regardless of content -- turning a
 * generator upgrade into a fleet-wide flag day for artifacts that did not
 * change in any way that matters.
 *
 * `writer`, `engine` and `build_id` in that block are provenance for debugging
 * a bad artifact. They are never compared. Only `min_reader` decides anything.
 *
 * A MISSING abi block means level 1: every recipe recorded before this existed
 * stays readable, which is the same additive rule guards follow.
 *
 * ------------------------------------------------------------- when to bump
 * Bump ROCKE_ABI_VERSION when a C caller compiled against the old header would
 * be WRONG about memory: a struct gains/loses/reorders a field, an enumerator is
 * inserted rather than appended, a function's signature changes. Appending a new
 * function, or a new enumerator at the end, does not require a bump.
 *
 * Bump ROCKE_RECIPE_ABI when the engine learns a construct that an older engine
 * would MISREAD rather than reject -- and register the construct in abi.py so
 * that artifacts using it declare the higher min_reader. Note the asymmetry
 * worth understanding before reaching for a bump: the VM already fails loudly on
 * an unknown instruction op, an unknown opcode and an unknown intexpr node, so
 * adding one of those is self-policing and an old engine refuses it on its own.
 * The bump matters for changes an old engine would silently accept and get
 * wrong -- a changed default, a reinterpreted field, a relaxed invariant.
 *
 * LIMIT worth stating: this gate covers what the VM dispatches on. Attribute
 * VALUES are passed through to the IR builder uninterpreted, so their meaning is
 * the lowerer's contract, not the VM's, and a lowerer that silently ignores an
 * attribute it does not know is not something this version can catch.
 */
#ifndef ROCKE_ABI_H
#define ROCKE_ABI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Binary ABI of the shared library: struct layout, enum values, signatures. */
#define ROCKE_ABI_VERSION 1

/* Wire ABI: the CBOR recipe/bundle/guard vocabulary this engine understands. */
#define ROCKE_RECIPE_ABI 1

/* The values the LOADED library was compiled with. A caller compares these
 * against the ROCKE_* macros it was itself compiled with; they differ exactly
 * when the header and the binary come from different builds.
 *
 * rocke_abi_version() is the one that matters for a ctypes or dlopen caller,
 * because every argument it passes is described by the old header. Check it
 * before the first real call. */
int rocke_abi_version(void);
int rocke_recipe_abi_level(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_ABI_H */
