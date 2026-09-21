/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/arena.h -- bump / region allocator.
 *
 * Every IR node (rocke_value_t, rocke_op_t, rocke_region_t, rocke_type_t, ...) is
 * owned by an arena. This mirrors the Python implementation's reliance on the
 * garbage collector: in Python an IRBuilder + its KernelDef keep the whole
 * graph alive for the lifetime of the build; here, a single arena does the
 * same job. There is no per-node free -- the whole arena is reset/destroyed at
 * once when codegen for a kernel is finished.
 *
 * The allocator is a chain of fixed-size blocks; large requests get their own
 * dedicated block. All allocations are pointer-aligned (max_align_t).
 */
#ifndef ROCKE_ARENA_H
#define ROCKE_ARENA_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_arena_block rocke_arena_block_t;

typedef struct rocke_arena
{
    rocke_arena_block_t* head; /* current block (most recent)               */
    size_t block_size; /* default block payload size, in bytes  */
    size_t total_bytes; /* total bytes requested by the user    */
    size_t total_alloc; /* total bytes mapped from the OS        */
} rocke_arena_t;

/* Initialise an arena with the given default block size (0 => 64 KiB).
 * Returns 0 on success, -1 on allocation failure. */
int rocke_arena_init(rocke_arena_t* a, size_t block_size);

/* Allocate `size` bytes, pointer-aligned. Returns NULL on OOM. Memory is
 * NOT zeroed. */
void* rocke_arena_alloc(rocke_arena_t* a, size_t size);

/* Allocate `size` bytes, zero-initialised. Returns NULL on OOM. */
void* rocke_arena_calloc(rocke_arena_t* a, size_t size);

/* Duplicate a NUL-terminated string into the arena. Returns NULL on OOM or if
 * `s` is NULL. */
char* rocke_arena_strdup(rocke_arena_t* a, const char* s);

/* printf into a freshly arena-allocated string. Returns NULL on OOM. */
char* rocke_arena_printf(rocke_arena_t* a, const char* fmt, ...);

/* Free every block. The arena is left zeroed and may be re-initialised. */
void rocke_arena_destroy(rocke_arena_t* a);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_ARENA_H */
