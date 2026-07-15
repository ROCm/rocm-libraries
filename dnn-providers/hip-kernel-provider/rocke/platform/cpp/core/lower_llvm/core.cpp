// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_core.c -- BUCKET 0 (the SPINE) of the C99 port of
 * rocke.core.lower_llvm.
 *
 * This file owns the lowerer state plumbing that every other bucket calls:
 *   - the public entry points (rocke_lower_kernel_to_llvm[_ex]),
 *   - the rocke_ll_dispatch table + rocke_ll_set_handler,
 *   - the op / region walkers (rocke_ll_lower_op / rocke_ll_lower_region),
 *   - the _Block / CFG model (current/new_block/block_at/emit/...),
 *   - the smem pre-pass + smem-global lookup,
 *   - finalize (module assembly),
 *   - flavor helpers + the ISA backend resolver,
 *   - and ALL the rocke_ll_* operand / type / constant / fresh-name / need
 *     utility helpers the per-op buckets consume.
 *
 * Faithful translation of rocke.core.lower_llvm (_Lowerer + module helpers).
 * Every spine helper below is fully ported from its Python counterpart; no
 * stub bodies remain in this file.
 */
#include "rocke/lower_llvm_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/error.hpp" /* ckc::Error boundary translation */

#include <exception>
#include <new>

/* ====================================================================== */
/* Flavor helpers (Python LLVM_FLAVOR_LLVM20 / LLVM_FLAVOR_LLVM22)        */
/* ====================================================================== */

const char* rocke_llvm_flavor_name(rocke_llvm_flavor_t flavor)
{
    switch(flavor)
    {
    case ROCKE_LLVM_FLAVOR_LLVM20:
        return "llvm20";
    case ROCKE_LLVM_FLAVOR_LLVM22:
        return "llvm22";
    case ROCKE_LLVM_FLAVOR_AUTO:
    default:
        return "";
    }
}

rocke_llvm_flavor_t rocke_llvm_flavor_from_name(const char* name)
{
    if(name)
    {
        if(strcmp(name, "llvm20") == 0)
            return ROCKE_LLVM_FLAVOR_LLVM20;
        if(strcmp(name, "llvm22") == 0)
            return ROCKE_LLVM_FLAVOR_LLVM22;
    }
    return ROCKE_LLVM_FLAVOR_AUTO;
}

/* The lowerer's private symbols live in namespace ckc; the public entry points
 * (rocke_llvm_flavor_name/from_name above, rocke_lower_kernel_to_llvm[_ex] below)
 * stay at global scope under their extern "C" header declarations. */
namespace ckc
{

/* Python _resolve_llvm_flavor: $ROCKE_LLVM_FLAVOR, then /opt/rocm version,
 * then default LLVM22. (torch.version.hip step is not portable.) */
static rocke_llvm_flavor_t ll_resolve_flavor(void)
{
    const char* env = getenv("ROCKE_LLVM_FLAVOR");
    if(env)
    {
        rocke_llvm_flavor_t f = rocke_llvm_flavor_from_name(env);
        if(f != ROCKE_LLVM_FLAVOR_AUTO)
        {
            return f;
        }
    }
    /* Best-effort: read /opt/rocm/.info/version "M.m...". >= 7.2 => LLVM22. */
    FILE* fp = fopen("/opt/rocm/.info/version", "r");
    if(fp)
    {
        int major = 0, minor = 0;
        if(fscanf(fp, "%d.%d", &major, &minor) == 2)
        {
            fclose(fp);
            if(major > 7 || (major == 7 && minor >= 2))
            {
                return ROCKE_LLVM_FLAVOR_LLVM22;
            }
            return ROCKE_LLVM_FLAVOR_LLVM20;
        }
        fclose(fp);
    }
    return ROCKE_LLVM_FLAVOR_LLVM22;
}

/* ====================================================================== */
/* ISA backend (lower_llvm_internal.h's own backend struct)               */
/* ====================================================================== */

/* The two CDNA backends we port (gfx942 / gfx950). The waitcnt encoders are
 * defined in the control bucket. Static storage: returned by pointer. */
static const rocke_isa_backend_t LL_BACKEND_GFX950 = {"gfx950",
                                                      NULL,
                                                      NULL,
                                                      ROCKE_LL_BUFFER_RSRC_WORD3_CDNA,
                                                      rocke_ll_encode_waitcnt_gfx9_10,
                                                      ROCKE_LL_ISA_CDNA};
static const rocke_isa_backend_t LL_BACKEND_GFX942 = {"gfx942",
                                                      NULL,
                                                      NULL,
                                                      ROCKE_LL_BUFFER_RSRC_WORD3_CDNA,
                                                      rocke_ll_encode_waitcnt_gfx9_10,
                                                      ROCKE_LL_ISA_CDNA};
static const rocke_isa_backend_t LL_BACKEND_GFX908 = {"gfx908",
                                                      NULL,
                                                      NULL,
                                                      ROCKE_LL_BUFFER_RSRC_WORD3_CDNA,
                                                      rocke_ll_encode_waitcnt_gfx9_10,
                                                      ROCKE_LL_ISA_CDNA};
static const rocke_isa_backend_t LL_BACKEND_GFX90A = {"gfx90a",
                                                      NULL,
                                                      NULL,
                                                      ROCKE_LL_BUFFER_RSRC_WORD3_CDNA,
                                                      rocke_ll_encode_waitcnt_gfx9_10,
                                                      ROCKE_LL_ISA_CDNA};
/* RDNA backends (Python Gfx11RdnaBackend / Gfx12RdnaBackend): same
 * datalayout/triple as CDNA on the ROCm releases we target, but the RDNA buffer
 * SRD word3 and the contiguous gfx11 s_waitcnt layout. gfx12 differs from gfx11
 * only in WMMA fragment width, which the op_id ("wmma_gfx12_*") encodes. */
static const rocke_isa_backend_t LL_BACKEND_GFX1151 = {"gfx1151",
                                                       NULL,
                                                       NULL,
                                                       ROCKE_LL_BUFFER_RSRC_WORD3_RDNA,
                                                       rocke_ll_encode_waitcnt_gfx11,
                                                       ROCKE_LL_ISA_RDNA};
static const rocke_isa_backend_t LL_BACKEND_GFX1201 = {"gfx1201",
                                                       NULL,
                                                       NULL,
                                                       ROCKE_LL_BUFFER_RSRC_WORD3_RDNA,
                                                       rocke_ll_encode_waitcnt_gfx11,
                                                       ROCKE_LL_ISA_RDNA};
static const rocke_isa_backend_t LL_BACKEND_GFX11_GENERIC = {"gfx11-generic",
                                                             NULL,
                                                             NULL,
                                                             ROCKE_LL_BUFFER_RSRC_WORD3_RDNA,
                                                             rocke_ll_encode_waitcnt_gfx11,
                                                             ROCKE_LL_ISA_RDNA};

/* Mutable copies so datalayout/triple (extern consts resolved at runtime) can
 * be patched in. backend_for fills them from ROCKE_LL_DATALAYOUT/TRIPLE. */
static rocke_isa_backend_t LL_BACKEND_RESOLVED;

const rocke_isa_backend_t* rocke_ll_backend_for(const char* arch, rocke_status_t* st)
{
    const rocke_isa_backend_t* base = NULL;
    if(arch == NULL || strcmp(arch, "gfx950") == 0)
    {
        base = &LL_BACKEND_GFX950;
    }
    else if(strcmp(arch, "gfx942") == 0)
    {
        base = &LL_BACKEND_GFX942;
    }
    else if(strcmp(arch, "gfx908") == 0)
    {
        base = &LL_BACKEND_GFX908;
    }
    else if(strcmp(arch, "gfx90a") == 0)
    {
        base = &LL_BACKEND_GFX90A;
    }
    else if(strcmp(arch, "gfx1151") == 0)
    {
        base = &LL_BACKEND_GFX1151;
    }
    else if(strcmp(arch, "gfx1201") == 0)
    {
        base = &LL_BACKEND_GFX1201;
    }
    else if(strcmp(arch, "gfx11-generic") == 0)
    {
        base = &LL_BACKEND_GFX11_GENERIC;
    }
    else
    {
        if(st)
        {
            *st = ROCKE_ERR_KEY;
        }
        return NULL;
    }
    LL_BACKEND_RESOLVED = *base;
    LL_BACKEND_RESOLVED.datalayout = ROCKE_LL_DATALAYOUT;
    LL_BACKEND_RESOLVED.triple = ROCKE_LL_TRIPLE;
    if(st)
    {
        *st = ROCKE_OK;
    }
    return &LL_BACKEND_RESOLVED;
}

/* ====================================================================== */
/* Error model                                                            */
/* ====================================================================== */

[[noreturn]] void rocke_ll_fail(rocke_lower_t* L, rocke_status_t st, const char* fmt, ...)
{
    /* Format the reason once (bounded exactly like the legacy sink), then raise.
     * This [[noreturn]]s via ckc::raise_status. The thrown exception is caught at
     * the lowerer boundary and translated back into the status code + caller
     * `err` buffer, so the extern "C" ABI is unchanged. */
    (void)L; /* the lowerer no longer carries a sticky error; we raise instead */
    char buf[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    buf[sizeof buf - 1] = '\0';
    ckc::raise_status(st, buf);
}

bool rocke_ll_live(const rocke_lower_t* L)
{
    return L != NULL;
}

/* ====================================================================== */
/* Block / CFG model (Python _Block)                                      */
/* ====================================================================== */

rocke_ll_block_t* rocke_ll_current(rocke_lower_t* L)
{
    if(!L || L->blocks.len == 0)
    {
        return NULL;
    }
    return L->blocks.data[L->blocks.len - 1];
}

static rocke_ll_block_t* ll_make_block(rocke_lower_t* L, const char* label)
{
    rocke_ll_block_t* blk = (rocke_ll_block_t*)rocke_arena_calloc(&L->arena, sizeof(*blk));
    if(!blk)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "block alloc");
    }
    blk->label = rocke_arena_strdup(&L->arena, label ? label : "");
    rocke_vec_init(&blk->lines);
    blk->terminated = false;
    int rc;
    rocke_vec_push(&L->arena, &L->blocks, blk, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "blocks push");
    }
    return blk;
}

rocke_ll_block_t* rocke_ll_new_block(rocke_lower_t* L, const char* base)
{
    if(!L)
    {
        return NULL;
    }
    L->block_counter += 1;
    char* label = rocke_arena_printf(&L->arena, "%s.%d", base ? base : "", L->block_counter);
    if(!label)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "new_block label");
    }
    return ll_make_block(L, label);
}

rocke_ll_block_t* rocke_ll_block_at(rocke_lower_t* L, int idx)
{
    if(!L || idx < 0 || (size_t)idx >= L->blocks.len)
    {
        return NULL;
    }
    return L->blocks.data[idx];
}

int rocke_ll_block_count(const rocke_lower_t* L)
{
    return L ? (int)L->blocks.len : 0;
}

void rocke_ll_block_emit(rocke_lower_t* L, rocke_ll_block_t* blk, const char* line)
{
    if(!L || !blk)
    {
        return;
    }
    if(blk->terminated)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "block %s already terminated", blk->label);
    }
    char* copy = rocke_arena_strdup(&L->arena, line ? line : "");
    if(!copy)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "emit strdup");
    }
    int rc;
    rocke_vec_push(&L->arena, &blk->lines, copy, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "emit push");
    }
}

void rocke_ll_block_emitf(rocke_lower_t* L, rocke_ll_block_t* blk, const char* fmt, ...)
{
    if(!L || !blk)
    {
        return;
    }
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if(n >= 0 && (size_t)n < sizeof buf)
    {
        rocke_ll_block_emit(L, blk, buf);
        return;
    }
    /* Long line: format into the arena. */
    va_start(ap, fmt);
    char big[8192];
    vsnprintf(big, sizeof big, fmt, ap);
    va_end(ap);
    rocke_ll_block_emit(L, blk, big);
}

void rocke_ll_emit(rocke_lower_t* L, const char* line)
{
    rocke_ll_block_emit(L, rocke_ll_current(L), line);
}

void rocke_ll_emitf(rocke_lower_t* L, const char* fmt, ...)
{
    rocke_ll_block_t* blk = rocke_ll_current(L);
    if(!L || !blk)
    {
        return;
    }
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if(n >= 0 && (size_t)n < sizeof buf)
    {
        rocke_ll_block_emit(L, blk, buf);
        return;
    }
    va_start(ap, fmt);
    char big[8192];
    vsnprintf(big, sizeof big, fmt, ap);
    va_end(ap);
    rocke_ll_block_emit(L, blk, big);
}

/* ====================================================================== */
/* Fresh names (Python _fresh)                                            */
/* ====================================================================== */

const char* rocke_ll_fresh(rocke_lower_t* L, const char* hint)
{
    if(!L)
    {
        return "";
    }
    L->tmp_counter += 1;
    char* s = rocke_arena_printf(&L->arena, "%%%s.%d", hint ? hint : "t", L->tmp_counter);
    if(!s)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "fresh");
    }
    return s;
}

/* ====================================================================== */
/* Intrinsic need-tracking (Python _need / self._decls)                   */
/* ====================================================================== */

/* Resolve a decl key to its declaration text: dyn_decls override, then the
 * flavor override table (LLVM22), then the base table. NULL if unknown. */
static const char* ll_resolve_decl(rocke_lower_t* L, const char* key)
{
    if(!L || !key)
    {
        return NULL;
    }
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        if(strcmp(L->dyn_decls.data[i].key, key) == 0)
        {
            return L->dyn_decls.data[i].decl;
        }
    }
    if(L->flavor == ROCKE_LLVM_FLAVOR_LLVM22)
    {
        for(int i = 0; i < ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES_COUNT; i++)
        {
            if(strcmp(ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[i].key, key) == 0)
            {
                return ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[i].decl;
            }
        }
    }
    for(int i = 0; i < ROCKE_LL_INTRINSIC_DECLS_COUNT; i++)
    {
        if(strcmp(ROCKE_LL_INTRINSIC_DECLS[i].key, key) == 0)
        {
            return ROCKE_LL_INTRINSIC_DECLS[i].decl;
        }
    }
    return NULL;
}

static bool ll_need_has(const rocke_lower_t* L, const char* key)
{
    for(size_t i = 0; i < L->needs.len; i++)
    {
        if(strcmp(L->needs.data[i].key, key) == 0)
        {
            return true;
        }
    }
    return false;
}

void rocke_ll_need(rocke_lower_t* L, const char* key)
{
    if(!L || !key)
    {
        return;
    }
    if(ll_need_has(L, key))
    {
        return;
    }
    rocke_ll_need_t rec;
    rec.key = rocke_arena_strdup(&L->arena, key);
    rec.decl = ll_resolve_decl(L, key); /* may be NULL; finalize tolerates */
    int rc;
    rocke_vec_push(&L->arena, &L->needs, rec, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "need push");
    }
}

void rocke_ll_need_dynamic(rocke_lower_t* L, const char* key, const char* decl)
{
    if(!L || !key)
    {
        return;
    }
    /* Register/replace the dynamic decl text (Python self._decls[key] = decl). */
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        if(strcmp(L->dyn_decls.data[i].key, key) == 0)
        {
            L->dyn_decls.data[i].decl = rocke_arena_strdup(&L->arena, decl ? decl : "");
            rocke_ll_need(L, key);
            return;
        }
    }
    rocke_ll_decl_t d;
    d.key = rocke_arena_strdup(&L->arena, key);
    d.decl = rocke_arena_strdup(&L->arena, decl ? decl : "");
    int rc;
    rocke_vec_push(&L->arena, &L->dyn_decls, d, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "dyn_decl push");
    }
    rocke_ll_need(L, key);
}

/* ====================================================================== */
/* Type rendering (Python _llvm_type / _llvm_type_from_name)              */
/* ====================================================================== */

const char* rocke_ll_llvm_type(rocke_lower_t* L, const rocke_type_t* t)
{
    if(!t)
    {
        if(L)
        {
            rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM mapping for (null) type");
        }
        return "";
    }
    if(t->kind == ROCKE_TYPE_PTR)
    {
        const char* sp = t->space;
        if(sp && strcmp(sp, "global") == 0)
            return "ptr addrspace(1)";
        if(sp && strcmp(sp, "lds") == 0)
            return "ptr addrspace(3)";
        if(sp && strcmp(sp, "constant") == 0)
            return "ptr addrspace(4)";
        return "ptr";
    }
    if(t->kind == ROCKE_TYPE_VECTOR)
    {
        const char* elem = rocke_ll_llvm_type(L, t->elem);
        return rocke_arena_printf(&L->arena, "<%d x %s>", t->count, elem);
    }
    if(t->kind == ROCKE_TYPE_SMEM)
    {
        return "ptr addrspace(3)";
    }
    /* scalar */
    const char* n = t->name;
    if(n)
    {
        if(strcmp(n, "i1") == 0)
            return "i1";
        if(strcmp(n, "i8") == 0)
            return "i8";
        if(strcmp(n, "i16") == 0)
            return "i16";
        if(strcmp(n, "i32") == 0)
            return "i32";
        if(strcmp(n, "i64") == 0)
            return "i64";
        if(strcmp(n, "f16") == 0)
            return "half";
        if(strcmp(n, "bf16") == 0)
            return "bfloat";
        if(strcmp(n, "fp8e4m3") == 0)
            return "i8";
        if(strcmp(n, "bf8e5m2") == 0)
            return "i8";
        if(strcmp(n, "f32") == 0)
            return "float";
    }
    rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM mapping for type %s", n ? n : "(null)");
}

const char* rocke_ll_llvm_type_from_name(rocke_lower_t* L, const char* name)
{
    if(!name)
    {
        rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM type for (null)");
    }
    if(strcmp(name, "i32") == 0)
        return "i32";
    if(strcmp(name, "i64") == 0)
        return "i64";
    if(strcmp(name, "i8") == 0)
        return "i8";
    if(strcmp(name, "f16") == 0)
        return "half";
    if(strcmp(name, "bf16") == 0)
        return "bfloat";
    if(strcmp(name, "f32") == 0)
        return "float";
    if(strcmp(name, "fp8e4m3") == 0)
        return "i8";
    if(strncmp(name, "vec<", 4) == 0)
    {
        /* vec<elemxN> -> "<N x llvm_elem>" */
        const char* inner = name + 4;
        const char* xpos = strchr(inner, 'x');
        const char* end = strrchr(name, '>');
        if(xpos && end && end > xpos)
        {
            char elem[32];
            size_t elen = (size_t)(xpos - inner);
            if(elen >= sizeof elem)
            {
                elen = sizeof elem - 1;
            }
            memcpy(elem, inner, elen);
            elem[elen] = '\0';
            int count = atoi(xpos + 1);
            const char* lelem = "i32";
            if(strcmp(elem, "f32") == 0)
                lelem = "float";
            else if(strcmp(elem, "f16") == 0)
                lelem = "half";
            else if(strcmp(elem, "bf16") == 0)
                lelem = "bfloat";
            else if(strcmp(elem, "i32") == 0)
                lelem = "i32";
            else
            {
                rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM type for vec elem %s", elem);
            }
            return rocke_arena_printf(&L->arena, "<%d x %s>", count, lelem);
        }
    }
    rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM type for %s", name);
}

const char* rocke_ll_smem_storage_type(rocke_lower_t* L, const rocke_type_t* smem)
{
    if(!L || !smem)
    {
        return "";
    }
    const char* out = rocke_ll_llvm_type(L, smem->elem);
    /* Wrap from innermost (last dim) to outermost (first dim). */
    for(int d = smem->rank - 1; d >= 0; d--)
    {
        out = rocke_arena_printf(&L->arena, "[%d x %s]", smem->shape[d], out);
        if(!out)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_storage_type");
        }
    }
    return out;
}

/* ====================================================================== */
/* FP hex constants (Python _fp32_hex / _fp16_hex)                        */
/* ====================================================================== */

const char* rocke_ll_fp32_hex(rocke_lower_t* L, double x)
{
    /* LLVM spells a float hex constant as the 64-bit hex of the double value
     * of the rounded fp32 constant. */
    float f = (float)x;
    double rounded = (double)f;
    uint64_t bits;
    memcpy(&bits, &rounded, sizeof bits);
    return rocke_arena_printf(&L->arena, "0x%016llX", (unsigned long long)bits);
}

const char* rocke_ll_fp16_hex(rocke_lower_t* L, double x)
{
    /* LLVM IR: half 0xH<4 hex>. Convert double -> IEEE-754 binary16 (round to
     * nearest even) without relying on _Float16 support. */
    float f = (float)x;
    uint32_t fb;
    memcpy(&fb, &f, sizeof fb);
    uint32_t sign = (fb >> 16) & 0x8000u;
    int32_t exp = (int32_t)((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = fb & 0x7FFFFFu;
    uint16_t h;
    if(((fb >> 23) & 0xFF) == 0xFF)
    {
        /* inf / nan */
        h = (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    else if(exp >= 0x1F)
    {
        h = (uint16_t)(sign | 0x7C00u); /* overflow -> inf */
    }
    else if(exp <= 0)
    {
        if(exp < -10)
        {
            h = (uint16_t)sign; /* underflow -> 0 */
        }
        else
        {
            mant |= 0x800000u;
            uint32_t shift = (uint32_t)(14 - exp);
            uint32_t halfmant = mant >> shift;
            /* round to nearest even */
            uint32_t rem = mant & ((1u << shift) - 1u);
            uint32_t halfway = 1u << (shift - 1);
            if(rem > halfway || (rem == halfway && (halfmant & 1u)))
            {
                halfmant += 1;
            }
            h = (uint16_t)(sign | halfmant);
        }
    }
    else
    {
        uint16_t hm = (uint16_t)(mant >> 13);
        uint32_t rem = mant & 0x1FFFu;
        h = (uint16_t)(sign | ((uint16_t)exp << 10) | hm);
        if(rem > 0x1000u || (rem == 0x1000u && (hm & 1u)))
        {
            h += 1; /* carries into exponent naturally */
        }
    }
    return rocke_arena_printf(&L->arena, "0xH%04X", (unsigned)h);
}

/* ====================================================================== */
/* asm string escaping (Python _escape_llvm_asm_string)                   */
/* ====================================================================== */

const char* rocke_ll_escape_asm_string(rocke_lower_t* L, const char* s)
{
    if(!L)
    {
        return "";
    }
    if(!s)
    {
        return rocke_arena_strdup(&L->arena, "");
    }
    /* worst case 4x ("\XX"+1) -- be generous. */
    size_t n = strlen(s);
    size_t cap = n * 4 + 1;
    char* out = (char*)rocke_arena_alloc(&L->arena, cap);
    if(!out)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "escape_asm");
    }
    size_t w = 0;
    for(size_t i = 0; i < n; i++)
    {
        unsigned char c = (unsigned char)s[i];
        if(c == '\\')
        {
            w += (size_t)snprintf(out + w, cap - w, "\\5C");
        }
        else if(c == '"')
        {
            w += (size_t)snprintf(out + w, cap - w, "\\22");
        }
        else if(c >= 0x20 && c <= 0x7E)
        {
            out[w++] = (char)c;
        }
        else
        {
            w += (size_t)snprintf(out + w, cap - w, "\\%02X", c);
        }
    }
    out[w] = '\0';
    return out;
}

/* ====================================================================== */
/* Constant helpers (Python _is_constant / _eval_constant / _operand)     */
/* ====================================================================== */

bool rocke_ll_is_constant(const rocke_value_t* v)
{
    return v && v->op && v->op->opcode == ROCKE_OP_ARITH_CONSTANT;
}

int64_t rocke_ll_eval_constant(rocke_lower_t* L, const rocke_value_t* v)
{
    if(!rocke_ll_is_constant(v))
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "Value %s is not a compile-time constant",
                      (v && v->name) ? v->name : "(null)");
    }
    int64_t iv = 0;
    if(rocke_attr_get_int(&v->op->attrs, "value", &iv))
    {
        return iv;
    }
    double fv = 0.0;
    if(rocke_attr_get_float(&v->op->attrs, "value", &fv))
    {
        return (int64_t)fv;
    }
    return 0;
}

const char* rocke_ll_operand(rocke_lower_t* L, const rocke_value_t* v)
{
    if(!v)
    {
        return "";
    }
    const rocke_op_t* op = v->op;
    if(op == NULL)
    {
        return v->name;
    }
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        const char* ity = rocke_attr_get_str(&op->attrs, "ity");
        if(ity == NULL)
        {
            ity = "i32";
        }
        if(strcmp(ity, "f32") == 0)
        {
            double fv = 0.0;
            rocke_attr_get_float(&op->attrs, "value", &fv);
            return rocke_ll_fp32_hex(L, fv);
        }
        if(strcmp(ity, "f16") == 0)
        {
            double fv = 0.0;
            rocke_attr_get_float(&op->attrs, "value", &fv);
            return rocke_ll_fp16_hex(L, fv);
        }
        int64_t iv = 0;
        if(!rocke_attr_get_int(&op->attrs, "value", &iv))
        {
            double fv = 0.0;
            if(rocke_attr_get_float(&op->attrs, "value", &fv))
            {
                iv = (int64_t)fv;
            }
        }
        return rocke_arena_printf(&L->arena, "%lld", (long long)iv);
    }
    return v->name;
}

const char* rocke_ll_operand_with_type(rocke_lower_t* L, const rocke_value_t* v)
{
    if(!v)
    {
        return "";
    }
    const char* ty = rocke_ll_llvm_type(L, v->type);
    const char* op = rocke_ll_operand(L, v);
    return rocke_arena_printf(&L->arena, "%s %s", ty, op);
}

/* ====================================================================== */
/* Shared binary-op helpers (Python _binop / _vector_binop)               */
/* ====================================================================== */

void rocke_ll_binop(rocke_lower_t* L, const rocke_op_t* op, const char* llvm_op)
{
    if(!rocke_ll_live(L) || !op || op->num_results < 1 || op->num_operands < 2)
    {
        return;
    }
    const rocke_value_t* res = op->results[0];
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = %s %s %s, %s",
                   res->name,
                   llvm_op,
                   rocke_ll_llvm_type(L, res->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

void rocke_ll_vector_binop(rocke_lower_t* L, const rocke_op_t* op, const char* llvm_op)
{
    if(!rocke_ll_live(L) || !op || op->num_results < 1 || op->num_operands < 2)
    {
        return;
    }
    const rocke_value_t* res = op->results[0];
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = %s %s %s, %s",
                   res->name,
                   llvm_op,
                   rocke_ll_llvm_type(L, a->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* ====================================================================== */
/* smem pre-pass (Python _collect_smem / _smem_global_name)               */
/* ====================================================================== */

void rocke_ll_collect_smem(rocke_lower_t* L, const rocke_region_t* region)
{
    if(!L || !region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops; i++)
    {
        const rocke_op_t* op = region->ops[i];
        if(!op)
        {
            continue;
        }
        if(op->opcode == ROCKE_OP_TILE_SMEM_ALLOC && op->num_results > 0)
        {
            const rocke_value_t* res = op->results[0];
            const char* short_name = res->name;
            if(short_name && short_name[0] == '%')
            {
                short_name += 1;
            }
            const char* kname = L->kernel ? L->kernel->name : "";
            char* gname = rocke_arena_printf(
                &L->arena, "@%s.%s", short_name ? short_name : "", kname ? kname : "");
            rocke_ll_smem_global_t g;
            g.gname = gname;
            g.stype = res->type;
            int rc;
            rocke_vec_push(&L->arena, &L->smem_globals, g, rc);
            if(rc != 0)
            {
                rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_globals push");
            }
            rocke_ll_smem_name_t nm;
            nm.value_name = res->name;
            nm.gname = gname;
            rocke_vec_push(&L->arena, &L->smem_names, nm, rc);
            if(rc != 0)
            {
                rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_names push");
            }
        }
        for(int r = 0; r < op->num_regions; r++)
        {
            rocke_ll_collect_smem(L, op->regions[r]);
        }
    }
}

const char* rocke_ll_smem_global_name(rocke_lower_t* L,
                                      const rocke_value_t* smem,
                                      const rocke_type_t** out_stype)
{
    if(out_stype)
    {
        *out_stype = NULL;
    }
    if(!L || !smem)
    {
        return NULL;
    }
    for(size_t i = 0; i < L->smem_names.len; i++)
    {
        if(smem->name && L->smem_names.data[i].value_name
           && strcmp(L->smem_names.data[i].value_name, smem->name) == 0)
        {
            if(out_stype)
            {
                *out_stype = smem->type;
            }
            return L->smem_names.data[i].gname;
        }
    }
    rocke_ll_fail(
        L, ROCKE_ERR_KEY, "smem value %s never allocated", smem->name ? smem->name : "(null)");
}

/* ====================================================================== */
/* yield-stack helpers DEFINED IN CONTROL bucket -- not here.             */
/* ====================================================================== */

/* ====================================================================== */
/* Op dispatch table                                                      */
/* ====================================================================== */

rocke_ll_op_fn rocke_ll_dispatch[ROCKE_OP__COUNT];

void rocke_ll_set_handler(rocke_opcode_t opcode, rocke_ll_op_fn fn)
{
    if(opcode > ROCKE_OP_INVALID && opcode < ROCKE_OP__COUNT)
    {
        rocke_ll_dispatch[(int)opcode] = fn;
    }
}

void rocke_ll_lower_op(rocke_lower_t* L, const rocke_op_t* op)
{
    if(!rocke_ll_live(L) || !op)
    {
        return;
    }
    rocke_opcode_t oc = op->opcode;
    rocke_ll_op_fn fn = NULL;
    if(oc > ROCKE_OP_INVALID && oc < ROCKE_OP__COUNT)
    {
        fn = rocke_ll_dispatch[(int)oc];
    }
    if(fn == NULL)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "no LLVM lowering for op %s",
                      op->name ? op->name : rocke_opcode_name(oc));
    }
    fn(L, op);
}

void rocke_ll_lower_region(rocke_lower_t* L, const rocke_region_t* region)
{
    if(!L || !region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops && rocke_ll_live(L); i++)
    {
        rocke_ll_lower_op(L, region->ops[i]);
    }
}

/* Build the dispatch table once: call every per-bucket registration hook. */
static void ll_register_all(void)
{
    memset(rocke_ll_dispatch, 0, sizeof rocke_ll_dispatch);
    rocke_ll_register_arith();
    rocke_ll_register_convert();
    rocke_ll_register_mem();
    rocke_ll_register_mma();
    rocke_ll_register_crosslane();
    rocke_ll_register_vector();
}

/* ====================================================================== */
/* finalize trailers (Python _param_attrs / _format_agpr_alloc)           */
/* ====================================================================== */

const char* rocke_ll_param_attrs(rocke_lower_t* L, const rocke_param_t* p)
{
    if(!L || !p)
    {
        return "";
    }
    if(!p->type || p->type->kind != ROCKE_TYPE_PTR)
    {
        return "";
    }
    char buf[256];
    size_t w = 0;
    buf[0] = '\0';
#define LL_APPEND_ATTR(txt)                                                      \
    do                                                                           \
    {                                                                            \
        int _n = snprintf(buf + w, sizeof buf - w, "%s%s", w ? " " : "", (txt)); \
        if(_n > 0)                                                               \
        {                                                                        \
            w += (size_t)_n;                                                     \
        }                                                                        \
    } while(0)
    if(rocke_attr_get_bool(&p->attrs, "noalias", false))
        LL_APPEND_ATTR("noalias");
    if(rocke_attr_get_bool(&p->attrs, "readonly", false))
        LL_APPEND_ATTR("readonly");
    if(rocke_attr_get_bool(&p->attrs, "writeonly", false))
        LL_APPEND_ATTR("writeonly");
    if(rocke_attr_get_bool(&p->attrs, "nocapture", true))
        LL_APPEND_ATTR("nocapture");
    if(rocke_attr_get_bool(&p->attrs, "nonnull", false))
        LL_APPEND_ATTR("nonnull");
    int64_t align = 0;
    if(rocke_attr_get_int(&p->attrs, "align", &align))
    {
        char a[48];
        snprintf(a, sizeof a, "align %lld", (long long)align);
        LL_APPEND_ATTR(a);
    }
    int64_t deref = 0;
    if(rocke_attr_get_int(&p->attrs, "dereferenceable", &deref))
    {
        char d[64];
        snprintf(d, sizeof d, "dereferenceable(%lld)", (long long)deref);
        LL_APPEND_ATTR(d);
    }
#undef LL_APPEND_ATTR
    if(w == 0)
    {
        return "";
    }
    return rocke_arena_printf(&L->arena, " %s", buf);
}

const char* rocke_ll_format_agpr_alloc(rocke_lower_t* L, const rocke_attr_value_t* v)
{
    if(!L || !v)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc must be a (min, max) pair");
    }
    long lo = 0, hi = 0;
    if(v->kind == ROCKE_ATTR_STR)
    {
        const char* text = v->u.s;
        if(!text || !*text)
        {
            rocke_ll_fail(
                L, ROCKE_ERR_VALUE, "agpr_alloc string must be 'min,max', not empty/'none'");
        }
        /* skip leading ws */
        while(*text == ' ' || *text == '\t')
        {
            text++;
        }
        if(strncmp(text, "none", 4) == 0 || strncmp(text, "None", 4) == 0)
        {
            rocke_ll_fail(
                L, ROCKE_ERR_VALUE, "agpr_alloc string must be 'min,max', not empty/'none'");
        }
        const char* comma = strchr(text, ',');
        if(!comma)
        {
            rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc must contain two unsigned integers");
        }
        lo = strtol(text, NULL, 10);
        hi = strtol(comma + 1, NULL, 10);
    }
    else if(v->kind == ROCKE_ATTR_INT_LIST && v->u.ilist.count == 2)
    {
        /* A two-element list of bare ints (the (min, max) pair). */
        lo = (long)v->u.ilist.ints[0];
        hi = (long)v->u.ilist.ints[1];
    }
    else if(v->kind == ROCKE_ATTR_LIST && v->u.list.count == 2)
    {
        /* A two-element list of int maps; tolerate by reading [0]/[1] ints. */
        const rocke_attr_value_t* e0 = rocke_attr_get(v->u.list.items[0], "value");
        const rocke_attr_value_t* e1 = rocke_attr_get(v->u.list.items[1], "value");
        lo = (e0 && e0->kind == ROCKE_ATTR_INT) ? (long)e0->u.i : 0;
        hi = (e1 && e1->kind == ROCKE_ATTR_INT) ? (long)e1->u.i : 0;
    }
    else if(v->kind == ROCKE_ATTR_INT)
    {
        lo = hi = (long)v->u.i;
    }
    else
    {
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "agpr_alloc must be a (min, max) pair or 'min,max' string");
    }
    if(lo < 0 || hi < 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc values must be unsigned");
    }
    if(lo > hi)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc min must be <= max");
    }
    return rocke_arena_printf(&L->arena, "%ld,%ld", lo, hi);
}

/* ====================================================================== */
/* finalize (Python finalize)                                             */
/* ====================================================================== */

void rocke_ll_finalize(rocke_lower_t* L, rocke_strbuf_t* out)
{
    if(!L || !out)
    {
        return;
    }
    /* Terminate the current block with ret void. */
    rocke_ll_block_t* cur = rocke_ll_current(L);
    if(cur && !cur->terminated)
    {
        rocke_ll_block_emit(L, cur, " ret void");
        cur->terminated = true;
    }

    const char* dl = L->backend ? L->backend->datalayout : ROCKE_LL_DATALAYOUT;
    const char* tr = L->backend ? L->backend->triple : ROCKE_LL_TRIPLE;
    rocke_strbuf_appendf(out, "target datalayout = \"%s\"\n", dl ? dl : "");
    rocke_strbuf_appendf(out, "target triple = \"%s\"\n", tr ? tr : "");
    rocke_strbuf_append(out, "\n");

    /* smem globals. */
    for(size_t i = 0; i < L->smem_globals.len; i++)
    {
        const rocke_ll_smem_global_t* g = &L->smem_globals.data[i];
        const char* agg = rocke_ll_smem_storage_type(L, g->stype);
        const char* elem_name = (g->stype && g->stype->elem) ? g->stype->elem->name : NULL;
        bool elem_is_byte = elem_name
                            && (strcmp(elem_name, "i8") == 0 || strcmp(elem_name, "fp8e4m3") == 0
                                || strcmp(elem_name, "bf8e5m2") == 0);
        int align = elem_is_byte ? 16 : 4;
        rocke_strbuf_appendf(out,
                             "%s = internal unnamed_addr addrspace(3) global %s poison, align %d\n",
                             g->gname,
                             agg,
                             align);
    }
    if(L->smem_globals.len > 0)
    {
        rocke_strbuf_append(out, "\n");
    }

    /* Needed intrinsic declarations, in canonical TABLE order (then dynamic
     * decls). This mirrors finalize iterating self._decls in insertion order.
     *
     * Python builds self._decls as dict(_INTRINSIC_DECLS) then .update(
     * _INTRINSIC_DECLS_LLVM22_OVERRIDES) for the LLVM22 flavor. dict.update
     * REPLACES the value text for an existing key but PRESERVES that key's
     * original insertion position. So we must iterate the base table in order
     * and, per needed key, emit the override text when the flavor is LLVM22 and
     * the key has an override -- NOT emit all overrides in a separate leading
     * loop (which would float overridden keys, e.g. make.buffer.rsrc, to the
     * front of the declare block). */
    bool any_need = false;
    for(int i = 0; i < ROCKE_LL_INTRINSIC_DECLS_COUNT; i++)
    {
        const char* k = ROCKE_LL_INTRINSIC_DECLS[i].key;
        const char* decl_text = ROCKE_LL_INTRINSIC_DECLS[i].decl;
        if(L->flavor == ROCKE_LLVM_FLAVOR_LLVM22)
        {
            for(int j = 0; j < ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES_COUNT; j++)
            {
                if(strcmp(ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[j].key, k) == 0)
                {
                    decl_text = ROCKE_LL_INTRINSIC_DECLS_LLVM22_OVERRIDES[j].decl;
                    break;
                }
            }
        }
        if(ll_need_has(L, k))
        {
            rocke_strbuf_appendf(out, "%s\n", decl_text);
            any_need = true;
        }
    }
    /* Dynamic decls not in either static table. */
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        const char* k = L->dyn_decls.data[i].key;
        if(!ll_need_has(L, k))
        {
            continue;
        }
        if(ll_resolve_decl(L, k) == L->dyn_decls.data[i].decl && ll_resolve_decl(L, k) != NULL)
        {
            /* only emit if not already covered by a static table row */
            bool in_static = false;
            for(int j = 0; j < ROCKE_LL_INTRINSIC_DECLS_COUNT; j++)
            {
                if(strcmp(ROCKE_LL_INTRINSIC_DECLS[j].key, k) == 0)
                {
                    in_static = true;
                    break;
                }
            }
            if(!in_static)
            {
                rocke_strbuf_appendf(out, "%s\n", L->dyn_decls.data[i].decl);
                any_need = true;
            }
        }
    }
    if(any_need)
    {
        rocke_strbuf_append(out, "\n");
    }

    /* Function header. */
    rocke_strbuf_appendf(out, "define amdgpu_kernel void @%s(", L->kernel ? L->kernel->name : "");
    if(L->kernel)
    {
        for(int i = 0; i < L->kernel->num_params; i++)
        {
            const rocke_param_t* p = L->kernel->params[i];
            const char* tstr = rocke_ll_llvm_type(L, p->type);
            /* addr_space override (P17). */
            if(p->type && p->type->kind == ROCKE_TYPE_PTR)
            {
                const char* ovr = rocke_attr_get_str(&p->attrs, "addr_space");
                if(ovr && strcmp(ovr, "constant") == 0)
                {
                    tstr = "ptr addrspace(4)";
                }
                else if(ovr && strcmp(ovr, "global") == 0)
                {
                    tstr = "ptr addrspace(1)";
                }
            }
            const char* attrs = rocke_ll_param_attrs(L, p);
            rocke_strbuf_appendf(out, "%s%s%s %%%s", i ? ", " : "", tstr, attrs, p->name);
        }
    }
    rocke_strbuf_append(out, ") #0 {\n");

    for(size_t i = 0; i < L->blocks.len; i++)
    {
        const rocke_ll_block_t* blk = L->blocks.data[i];
        rocke_strbuf_appendf(out, "%s:\n", blk->label);
        for(size_t j = 0; j < blk->lines.len; j++)
        {
            rocke_strbuf_appendf(out, "%s\n", blk->lines.data[j]);
        }
    }
    rocke_strbuf_append(out, "}\n\n");

    /* attributes #0. */
    int max_wg = L->kernel ? rocke_kernel_max_workgroup_size(L->kernel) : 256;
    rocke_strbuf_append(out, "attributes #0 = { ");
    rocke_strbuf_append(out, "\"uniform-work-group-size\"=\"true\" ");
    rocke_strbuf_appendf(out, "\"amdgpu-flat-work-group-size\"=\"64,%d\"", max_wg);

    if(L->kernel)
    {
        /* waves_per_eu mirrors the Python lowerer: a bare int N emits "N,N",
         * a 2-element tuple (lo,hi) -- serialized as the INT_LIST l:[ i:lo, i:hi ]
         * -- emits "lo,hi". */
        const rocke_attr_value_t* wpe_v = rocke_attr_get(&L->kernel->attrs, "waves_per_eu");
        if(wpe_v && wpe_v->kind == ROCKE_ATTR_INT)
        {
            long long n = (long long)wpe_v->u.i;
            rocke_strbuf_appendf(out, " \"amdgpu-waves-per-eu\"=\"%lld,%lld\"", n, n);
        }
        else if(wpe_v && wpe_v->kind == ROCKE_ATTR_INT_LIST && wpe_v->u.ilist.count == 2)
        {
            rocke_strbuf_appendf(out,
                                 " \"amdgpu-waves-per-eu\"=\"%lld,%lld\"",
                                 (long long)wpe_v->u.ilist.ints[0],
                                 (long long)wpe_v->u.ilist.ints[1]);
        }
        const rocke_attr_value_t* agpr = rocke_attr_get(&L->kernel->attrs, "agpr_alloc");
        if(agpr)
        {
            const char* fa = rocke_ll_format_agpr_alloc(L, agpr);
            rocke_strbuf_appendf(out, " \"amdgpu-agpr-alloc\"=\"%s\"", fa);
        }
    }
    rocke_strbuf_append(out, " norecurse nounwind }\n");

    /* Python finalize does "\n".join(out) with a trailing "" element, so the
     * file ends in a single newline; when the fp-atomic metadata is present it
     * is preceded by a blank line (out.append("") before "!1 = !{}"). */
    if(L->needs_fp_atomic_md)
    {
        rocke_strbuf_append(out, "\n!1 = !{}\n");
    }
}

/* ====================================================================== */
/* Public entry points (Python lower_kernel_to_llvm)                      */
/* ====================================================================== */

/* Run the lowering against an initialized lowerer `L`. On any failure the
 * per-op handlers (and the spine helpers) raise via rocke_ll_fail -> throw, so
 * this body has no in-band error returns; success produces the heap-owned IR
 * text in *out_text. The caller owns L.arena and destroys it on both paths. */
static void ll_lower_into(rocke_lower_t* L,
                          const rocke_kernel_def_t* kernel,
                          rocke_llvm_flavor_t flavor,
                          const char* arch,
                          char** out_text)
{
    /* Resolve flavor. */
    L->flavor = (flavor == ROCKE_LLVM_FLAVOR_AUTO) ? ll_resolve_flavor() : flavor;
    if(L->flavor != ROCKE_LLVM_FLAVOR_LLVM20 && L->flavor != ROCKE_LLVM_FLAVOR_LLVM22)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "unknown LLVM flavor");
    }

    /* Resolve ISA backend. */
    rocke_status_t bst = ROCKE_OK;
    L->backend = rocke_ll_backend_for(arch, &bst);
    if(L->backend == NULL || bst != ROCKE_OK)
    {
        rocke_ll_fail(L,
                      bst != ROCKE_OK ? bst : ROCKE_ERR_KEY,
                      "unknown arch backend %s",
                      arch ? arch : "(null)");
    }
    /* The AMDGPU datalayout is FLAVOR-KEYED (Python backend.datalayout(flavor)
     * via _datalayout_for_flavor): the p8 field drifts between LLVM20 and
     * LLVM22. backend_for installs the LLVM20 form by default; rebind the
     * resolved backend's datalayout to the form for the resolved flavor.
     * L->backend points at the static LL_BACKEND_RESOLVED scratch copy, so this
     * does not mutate the canonical per-arch descriptors. */
    LL_BACKEND_RESOLVED.datalayout = rocke_ll_datalayout_for_flavor(L->flavor);

    /* Entry block (ll_make_block raises on OOM). */
    ll_make_block(L, "entry");

    /* Pre-pass + lowering. */
    rocke_ll_collect_smem(L, kernel->body);
    rocke_ll_lower_region(L, kernel->body);

    rocke_strbuf_t sb;
    if(rocke_strbuf_init(&sb, 4096) != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "out buffer");
    }
    /* finalize raises (rocke_ll_fail -> throw) on OOM; free `sb` before the
     * exception propagates so the throw path does not leak its heap buffer.
     * Codegen-neutral: the success path is unchanged. */
    try
    {
        rocke_ll_finalize(L, &sb);
    }
    catch(...)
    {
        rocke_strbuf_free(&sb);
        throw;
    }
    if(sb.oom)
    {
        rocke_strbuf_free(&sb);
        rocke_ll_fail(L, ROCKE_ERR_OOM, "finalize OOM");
    }
    char* text = rocke_strbuf_detach(&sb);
    if(text == NULL)
    {
        /* empty builder -> hand back an empty heap string */
        text = (char*)malloc(1);
        if(text == NULL)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "detach");
        }
        text[0] = '\0';
    }
    *out_text = text;
}

static rocke_status_t ll_lower_kernel_to_llvm_ex_impl(const rocke_kernel_def_t* kernel,
                                                      rocke_llvm_flavor_t flavor,
                                                      const char* arch,
                                                      char** out_text,
                                                      char* err,
                                                      size_t err_cap)
{
    if(out_text)
    {
        *out_text = NULL;
    }
    if(err && err_cap > 0)
    {
        err[0] = '\0';
    }
    if(!kernel || !out_text)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Build the dispatch table (idempotent across calls). */
    ll_register_all();

    rocke_lower_t L;
    memset(&L, 0, sizeof L);
    if(rocke_arena_init(&L.arena, 0) != 0)
    {
        return ROCKE_ERR_OOM;
    }
    L.kernel = kernel;
    L.status = ROCKE_OK;
    L.err = (char*)rocke_arena_calloc(&L.arena, ROCKE_ERR_MSG_CAP);
    L.unroll_elide_sync_op = NULL;
    L.needs_fp_atomic_md = false;
    rocke_vec_init(&L.blocks);
    rocke_vec_init(&L.needs);
    rocke_vec_init(&L.dyn_decls);
    rocke_vec_init(&L.smem_globals);
    rocke_vec_init(&L.smem_names);
    rocke_vec_init(&L.yield_stack);

    /* A failure anywhere in lowering raises a ckc::Error; catch it here so the
     * arena is always destroyed, then translate it into the legacy status code +
     * caller `err` buffer (keeping the extern "C" ABI unchanged). */
    try
    {
        ll_lower_into(&L, kernel, flavor, arch, out_text);
        rocke_arena_destroy(&L.arena);
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", e.what());
        }
        rocke_arena_destroy(&L.arena);
        return e.code();
    }
    catch(const std::bad_alloc& e)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", e.what());
        }
        rocke_arena_destroy(&L.arena);
        return ROCKE_ERR_OOM;
    }
    catch(const std::exception& e)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", e.what());
        }
        rocke_arena_destroy(&L.arena);
        return ROCKE_ERR_VALUE;
    }
    catch(...)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", "unknown C++ exception at extern \"C\" boundary");
        }
        rocke_arena_destroy(&L.arena);
        return ROCKE_ERR_VALUE;
    }
}

} /* namespace ckc */

rocke_status_t rocke_lower_kernel_to_llvm_ex(const rocke_kernel_def_t* kernel,
                                             rocke_llvm_flavor_t flavor,
                                             const char* arch,
                                             char** out_text,
                                             char* err,
                                             size_t err_cap)
{
    return ckc::ll_lower_kernel_to_llvm_ex_impl(kernel, flavor, arch, out_text, err, err_cap);
}

rocke_status_t rocke_lower_kernel_to_llvm(const rocke_kernel_def_t* kernel,
                                          rocke_llvm_flavor_t flavor,
                                          const char* arch,
                                          char** out_text)
{
    return rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_text, NULL, 0);
}
