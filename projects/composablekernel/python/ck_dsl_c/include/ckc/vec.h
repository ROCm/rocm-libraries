/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * ckc/vec.h -- generic, arena-backed dynamic arrays.
 *
 * The Python IR uses Python lists everywhere: op.operands, op.results,
 * op.regions, region.ops, kernel.params. Those are all variable-length and
 * grow as the builder appends. This header provides a typed dynamic array whose
 * backing storage is owned by a ckc_arena_t (so it shares the IR-graph lifetime
 * and is never individually freed).
 *
 * Usage:
 *   typedef CKC_VEC(ckc_value_t *) ckc_value_vec_t;  // == ckc_vec_t<ckc_value_t*>
 *   ckc_value_vec_t ops; ckc_vec_init(&ops);
 *   int rc; ckc_vec_push(arena, &ops, val, rc);     // rc: 0 ok, -1 OOM
 *   for (size_t i = 0; i < ops.len; i++) use(ops.data[i]);
 *
 * Growth doubles capacity and re-copies into a fresh arena block (the old block
 * is abandoned to the arena -- acceptable, the arena is bulk-freed). Typical IR
 * lists are tiny (<= 8 elements), so this is rarely hit. The vector is a
 * trivial, standard-layout aggregate ({data,len,cap}): it can be zero-init'd by
 * memset / arena calloc exactly like the structs that embed it, which is why
 * the lowerer state (allocated with ckc_arena_calloc / memset) stays correct.
 * Growth order is deterministic, so the emitted IR is reproducible.
 */
#ifndef CKC_VEC_H
#define CKC_VEC_H

#include <string.h>

#include "ckc/arena.h"

#ifdef __cplusplus

/* A typed, arena-backed dynamic array. Trivial + standard-layout so embedding
 * structs can be zero-initialised by memset / ckc_arena_calloc unchanged. */
template <class T>
struct ckc_vec_t
{
    T* data;
    size_t len;
    size_t cap;
};

/* Spell a vector type. Kept as a macro so existing `CKC_VEC(T)` declarations --
 * including nested forms like CKC_VEC(CKC_VEC(const char*) *) -- compile as-is.
 * Unlike the former anonymous-struct macro, equal element types now name the
 * same type, which removes the void* laundering the old distinct-struct form
 * required (existing casts still compile). */
#define CKC_VEC(T) ckc_vec_t<T>

/* Zero-initialise a vector. */
template <class T>
inline void ckc_vec_init(ckc_vec_t<T>* v)
{
    v->data = nullptr;
    v->len  = 0;
    v->cap  = 0;
}

/* Ensure capacity for at least `n` total elements. Returns 0 on success, -1 on
 * OOM. Growth schedule (cap?cap:4, doubling) is identical to the original. */
template <class T>
inline int ckc_vec_reserve_fn(ckc_arena_t* arena, ckc_vec_t<T>* v, size_t n)
{
    if(n <= v->cap)
    {
        return 0;
    }
    size_t nc = v->cap ? v->cap : 4;
    while(nc < n)
    {
        nc *= 2;
    }
    void* p = ckc_arena_alloc(arena, nc * sizeof(*v->data));
    if(!p)
    {
        return -1;
    }
    if(v->data && v->len)
    {
        memcpy(p, v->data, v->len * sizeof(*v->data));
    }
    v->data = static_cast<T*>(p);
    v->cap  = nc;
    return 0;
}

/* Append `val`. Returns 0 on success, -1 on OOM. */
template <class T, class U>
inline int ckc_vec_push_fn(ckc_arena_t* arena, ckc_vec_t<T>* v, U val)
{
    if(ckc_vec_reserve_fn(arena, v, v->len + 1) != 0)
    {
        return -1;
    }
    v->data[v->len++] = static_cast<T>(val);
    return 0;
}

/* Reserve helper, sets the int lvalue `ok` to 0 / -1 (legacy spelling). */
#define ckc_vec_reserve(arena, v, n, ok) ((ok) = ckc_vec_reserve_fn((arena), (v), (size_t)(n)))

/* Push helper, sets the int lvalue `rc` to 0 / -1 (legacy spelling). */
#define ckc_vec_push(arena, v, val, rc) ((rc) = ckc_vec_push_fn((arena), (v), (val)))

#else /* !__cplusplus -- original C99 macro forms (no C consumers today) */

#define CKC_VEC(T)  \
    struct          \
    {               \
        T* data;    \
        size_t len; \
        size_t cap; \
    }

#define ckc_vec_init(v)   \
    do                    \
    {                     \
        (v)->data = NULL; \
        (v)->len  = 0;    \
        (v)->cap  = 0;    \
    } while(0)

#define ckc_vec_reserve(arena, v, n, ok)                                   \
    do                                                                     \
    {                                                                      \
        (ok) = 0;                                                          \
        if((size_t)(n) > (v)->cap)                                         \
        {                                                                  \
            size_t _nc = (v)->cap ? (v)->cap : 4;                          \
            while(_nc < (size_t)(n))                                       \
            {                                                              \
                _nc *= 2;                                                  \
            }                                                              \
            void* _p = ckc_arena_alloc((arena), _nc * sizeof(*(v)->data)); \
            if(!_p)                                                        \
            {                                                              \
                (ok) = -1;                                                 \
            }                                                              \
            else                                                           \
            {                                                              \
                if((v)->data && (v)->len)                                  \
                {                                                          \
                    memcpy(_p, (v)->data, (v)->len * sizeof(*(v)->data));  \
                }                                                          \
                (v)->data = (__typeof__((v)->data))_p;                     \
                (v)->cap  = _nc;                                           \
            }                                                              \
        }                                                                  \
    } while(0)

#define ckc_vec_push_ok(arena, v, val, ok)                 \
    do                                                     \
    {                                                      \
        ckc_vec_reserve((arena), (v), (v)->len + 1, (ok)); \
        if((ok) == 0)                                      \
        {                                                  \
            (v)->data[(v)->len++] = (val);                 \
        }                                                  \
    } while(0)

#define ckc_vec_push(arena, v, val, rc) ckc_vec_push_ok((arena), (v), (val), (rc))

#endif /* __cplusplus */

#endif /* CKC_VEC_H */
