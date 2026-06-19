/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/vec.h -- generic, arena-backed dynamic arrays.
 *
 * The Python IR uses Python lists everywhere: op.operands, op.results,
 * op.regions, region.ops, kernel.params. Those are all variable-length and
 * grow as the builder appends. This header provides a header-only typed dynamic
 * array whose backing storage is owned by a ckc_arena_t (so it shares the
 * IR-graph lifetime and is never individually freed).
 *
 * Usage:
 *   typedef CKC_VEC(ckc_value_t *) ckc_value_vec_t;
 *   ckc_value_vec_t ops; ckc_vec_init(&ops);
 *   ckc_vec_push(arena, &ops, val);          // returns int (0 ok, -1 OOM)
 *   for (size_t i = 0; i < ops.len; i++) use(ops.data[i]);
 *
 * Growth doubles capacity and re-copies into a fresh arena block (the old block
 * is abandoned to the arena -- acceptable, the arena is bulk-freed). Typical IR
 * lists are tiny (<= 8 elements), so this is rarely hit.
 */
#ifndef CKC_VEC_H
#define CKC_VEC_H

#include <string.h>

#include "ckc/arena.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Declare an anonymous-struct vector type holding elements of type T. */
#define CKC_VEC(T)  \
    struct          \
    {               \
        T* data;    \
        size_t len; \
        size_t cap; \
    }

/* Zero-initialise a vector. */
#define ckc_vec_init(v)   \
    do                    \
    {                     \
        (v)->data = NULL; \
        (v)->len  = 0;    \
        (v)->cap  = 0;    \
    } while(0)

/* Ensure capacity for at least `n` total elements. Sets `ok` (an int lvalue) to
 * 0 on success or -1 on OOM. Internal helper; prefer ckc_vec_push. */
#define ckc_vec_reserve(arena, v, n, ok)                                        \
    do                                                                          \
    {                                                                           \
        (ok) = 0;                                                               \
        if((size_t)(n) > (v)->cap)                                              \
        {                                                                       \
            size_t _nc = (v)->cap ? (v)->cap : 4;                               \
            while(_nc < (size_t)(n))                                            \
            {                                                                   \
                _nc *= 2;                                                       \
            }                                                                   \
            void* _p = ckc_arena_alloc((arena), _nc * sizeof(*(v)->data));      \
            if(!_p)                                                             \
            {                                                                   \
                (ok) = -1;                                                      \
            }                                                                   \
            else                                                                \
            {                                                                   \
                if((v)->data && (v)->len)                                       \
                {                                                               \
                    memcpy(_p, (v)->data, (v)->len * sizeof(*(v)->data));       \
                }                                                               \
                /* WS3 C++ build: the arena returns void*; cast back to the     \
                 * element type. __typeof__ works in both C and C++ (g++/clang) \
                 * and keeps the assignment behaviour identical to C99.         \
                 */                                                             \
                (v)->data = (__typeof__((v)->data))_p;                          \
                (v)->cap  = _nc;                                                \
            }                                                                   \
        }                                                                       \
    } while(0)

/* Append `val`. Evaluates to an int statement-expression-like result via the
 * `ok` out param. Use ckc_vec_push for the common int-returning form. */
#define ckc_vec_push_ok(arena, v, val, ok)                 \
    do                                                     \
    {                                                      \
        ckc_vec_reserve((arena), (v), (v)->len + 1, (ok)); \
        if((ok) == 0)                                      \
        {                                                  \
            (v)->data[(v)->len++] = (val);                 \
        }                                                  \
    } while(0)

/* Convenience push that declares its own status variable. Sets the int lvalue
 * `rc` to 0 on success / -1 on OOM. */
#define ckc_vec_push(arena, v, val, rc) ckc_vec_push_ok((arena), (v), (val), (rc))

#ifdef __cplusplus
}
#endif

#endif /* CKC_VEC_H */
