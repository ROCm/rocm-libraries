// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_hip_core.c -- C99 port of rocke.core.lower_hip, BUCKET 0 (the SPINE):
 *
 *   - public entry point rocke_lower_kernel_to_hip (signature + prologue + body),
 *   - the op/region walkers (rocke_h_lower_op / rocke_h_lower_region),
 *   - the dispatch table that stitches every bucket's handler array together
 *     (rocke_h_dispatch), keyed by opcode,
 *   - emission + indent utilities (rocke_h_emit / rocke_h_emitf / rocke_h_emit_smem_decl
 *     / rocke_h_push_indent / rocke_h_pop_indent),
 *   - the sticky error / liveness channel (rocke_h_fail / rocke_h_live),
 *   - naming / type mapping (rocke_h_name / rocke_h_type_to_hip / rocke_h_hip_scalar /
 *     rocke_h_vec_prefix),
 *   - float literal formatting (rocke_h_f32_literal),
 *   - waitcnt encoders (rocke_h_encode_waitcnt + the two raw encoders),
 *   - arch gates (rocke_h_require_ds_read_tr / rocke_h_require_wmma_arch),
 *   - the smem storage side table (rocke_h_smem_storage / rocke_h_smem_set_storage),
 *   - and the arch seam resolver (rocke_hip_arch_from_gfx) + ROCKE_HIP_PROLOGUE.
 *
 * The per-op handlers live in the parallel bucket files (lower_hip_*.c); each
 * exports a registration table that this file stitches into a single
 * opcode-indexed dispatch array at first use. All shared helpers declared in
 * lower_hip_internal.h are DEFINED here and only called from the other buckets.
 *
 * This round's goal is a LINKING engine. The spine logic mirrors the Python
 * lowerer faithfully; a couple of side-table details (smem) use a small linear
 * arena-backed table keyed by the producing Value pointer, since the frozen IR
 * attrs must not be mutated.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error.hpp" /* ckc::Error boundary translation */
#include "rocke/ir.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_hip_internal.h"
#include "rocke/strbuf.h"
#include "rocke/vec.h"

#include <exception>
#include <new>

/* ============================== prologue ============================== */

/* Mirrors Python HIP_PROLOGUE byte-for-byte (see lower_hip.py HIP_PROLOGUE).
 * Static storage; never freed. */
const char* const ROCKE_HIP_PROLOGUE
    = "// === rocke lower_hip prologue (auto-generated) ===\n"
      "#include <hip/hip_runtime.h>\n"
      "#include <hip/hip_fp16.h>\n"
      "#include <math.h>\n"
      "#include <stdint.h>\n"
      "\n"
      "using fp16 = _Float16;\n"
      "#if defined(__BF16__) || defined(__bfloat16)\n"
      "using bf16 = __bf16;\n"
      "#else\n"
      "using bf16 = __bf16;\n"
      "#endif\n"
      "using fp8e4m3 = signed char; // raw byte storage; converted via amdgcn intrinsics\n"
      "using bf8e5m2 = signed char; // raw byte storage; e5m2 variant (different cvt intrinsic)\n"
      "\n"
      "// AMDGPU vector typedefs via Clang's ext_vector_type. Names match the\n"
      "// fNxM convention used throughout the handlers below.\n"
      "#define _ROCKE_VEC(elem_t, name, n) \\\n"
      " using name##n = elem_t __attribute__((ext_vector_type(n)))\n"
      "_ROCKE_VEC(fp16, f16x, 1); _ROCKE_VEC(fp16, f16x, 2); _ROCKE_VEC(fp16, f16x, 4);\n"
      "_ROCKE_VEC(fp16, f16x, 8); _ROCKE_VEC(fp16, f16x, 16);\n"
      "_ROCKE_VEC(bf16, bf16x, 1); _ROCKE_VEC(bf16, bf16x, 2); _ROCKE_VEC(bf16, bf16x, 4);\n"
      "_ROCKE_VEC(bf16, bf16x, 8); _ROCKE_VEC(bf16, bf16x, 16);\n"
      "_ROCKE_VEC(float, f32x, 1); _ROCKE_VEC(float, f32x, 2); _ROCKE_VEC(float, f32x, 4);\n"
      "_ROCKE_VEC(float, f32x, 8); _ROCKE_VEC(float, f32x, 16);\n"
      "_ROCKE_VEC(int, i32x, 1); _ROCKE_VEC(int, i32x, 2); _ROCKE_VEC(int, i32x, 3);\n"
      "_ROCKE_VEC(int, i32x, 4); _ROCKE_VEC(int, i32x, 8);\n"
      "_ROCKE_VEC(int16_t, i16x, 1); _ROCKE_VEC(int16_t, i16x, 2);\n"
      "_ROCKE_VEC(int16_t, i16x, 4); _ROCKE_VEC(int16_t, i16x, 8);\n"
      "_ROCKE_VEC(int8_t, i8x, 1); _ROCKE_VEC(int8_t, i8x, 2);\n"
      "_ROCKE_VEC(int8_t, i8x, 4); _ROCKE_VEC(int8_t, i8x, 8); _ROCKE_VEC(int8_t, i8x, 16);\n"
      "_ROCKE_VEC(bool, boolx, 2); _ROCKE_VEC(bool, boolx, 4); _ROCKE_VEC(bool, boolx, 8);\n"
      "_ROCKE_VEC(bool, boolx, 16);\n"
      "#undef _ROCKE_VEC\n"
      "\n"
      "// Buffer-resource descriptor opaque type. ``__builtin_amdgcn_make_buffer_rsrc``\n"
      "// returns this; the ``_ptr_`` family of buffer-load / store builtins takes\n"
      "// it as the first argument. Although the IR uses ``<4 x i32>`` to model the\n"
      "// 128-bit descriptor, at the C++ level we use the opaque builtin type so\n"
      "// type checking lines up with the intrinsics.\n"
      "using rsrc_t = __amdgpu_buffer_rsrc_t;\n"
      "\n"
      "// LLVM intrinsics that clang 20 does NOT expose as ``__builtin_amdgcn_*``\n"
      "// builtins (or whose builtins reject the size values we need). We declare\n"
      "// them as ``__device__ extern \"C\"`` with an ``__asm`` mangling that names\n"
      "// the LLVM intrinsic directly; clang lowers the call through the AMDGPU\n"
      "// backend the same way it would the missing builtin. The ``__device__``\n"
      "// attribute is required so HIP allows the call from a ``__global__``\n"
      "// kernel context.\n"
      "typedef short i16x4_raw __attribute__((ext_vector_type(4)));\n"
      "__device__ extern \"C\" i16x4_raw _llvm_amdgcn_ds_read_tr16_b64(\n"
      " const __attribute__((address_space(3))) void*)\n"
      " __asm(\"llvm.amdgcn.ds.read.tr16.b64\");\n"
      "typedef short i16x8_raw __attribute__((ext_vector_type(8)));\n"
      "__device__ extern \"C\" i16x8_raw _llvm_amdgcn_ds_read_tr16_b128(\n"
      " const __attribute__((address_space(3))) void*)\n"
      " __asm(\"llvm.amdgcn.ds.read.tr16.b128\");\n"
      "// ``__builtin_amdgcn_raw_ptr_buffer_load_lds`` restricts the size arg to\n"
      "// {1, 2, 4} bytes; the LLVM intrinsic itself accepts {1, 2, 4, 12, 16},\n"
      "// which is what async-DMA pipelines (compv4 / split-KV attention) need.\n"
      "// Calling the intrinsic directly bypasses the builtin's validation.\n"
      "__device__ extern \"C\" void _llvm_amdgcn_raw_ptr_buffer_load_lds(\n"
      " __amdgpu_buffer_rsrc_t,\n"
      " __attribute__((address_space(3))) void*,\n"
      " int /*size_bytes*/,\n"
      " int /*voffset*/,\n"
      " int /*soffset*/,\n"
      " int /*offset_imm*/,\n"
      " int /*aux_imm*/)\n"
      " __asm(\"llvm.amdgcn.raw.ptr.buffer.load.lds\");\n";

/* The HIP lowerer's private symbols live in namespace ckc; the public entry
 * points (rocke_hip_arch_from_gfx, rocke_lower_kernel_to_hip) and the public
 * ROCKE_HIP_PROLOGUE above stay at global scope under extern "C". */
namespace ckc
{

/* ============================== error / liveness ===================== */

bool rocke_h_live(const rocke_h_lowerer_t* lw)
{
    /* Internal ops raise on failure rather than latching a sticky status, so a
     * reachable non-NULL lowerer is always usable; this is now a NULL guard. */
    return lw != NULL;
}

[[noreturn]] rocke_status_t
    rocke_h_fail(rocke_h_lowerer_t* lw, rocke_status_t st, const char* fmt, ...)
{
    /* Format the reason once (bounded exactly like the legacy sink), then raise.
     * This [[noreturn]]s via ckc::raise_status. The thrown exception is caught at
     * the lowerer boundary (rocke_lower_kernel_to_hip) and translated back into the
     * status code, so the extern "C" ABI is unchanged. */
    (void)lw; /* the lowerer no longer carries a sticky error; we raise instead */
    char buf[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    buf[sizeof buf - 1] = '\0';
    ckc::raise_status(st, buf);
}

/* ============================== emission / indent ==================== */

void rocke_h_emit(rocke_h_lowerer_t* lw, const char* text)
{
    char* line;
    int width, rc;
    if(!rocke_h_live(lw))
    {
        return;
    }
    width = lw->indent > 0 ? lw->indent : 0;
    /* Python _emit: " " * indent + text */
    line = (char*)rocke_arena_alloc(&lw->b->arena, (size_t)width + strlen(text ? text : "") + 1);
    if(!line)
    {
        rocke_h_fail(lw, ROCKE_ERR_OOM, "out of memory emitting line");
        return;
    }
    memset(line, ' ', (size_t)width);
    strcpy(line + width, text ? text : "");
    rocke_vec_push(&lw->b->arena, &lw->lines, line, rc);
    if(rc != 0)
    {
        rocke_h_fail(lw, ROCKE_ERR_OOM, "out of memory appending line");
    }
}

void rocke_h_emitf(rocke_h_lowerer_t* lw, const char* fmt, ...)
{
    va_list ap;
    char* text;
    if(!rocke_h_live(lw))
    {
        return;
    }
    va_start(ap, fmt);
    {
        va_list ap2;
        int n;
        va_copy(ap2, ap);
        n = vsnprintf(NULL, 0, fmt, ap2);
        va_end(ap2);
        if(n < 0)
        {
            va_end(ap);
            rocke_h_fail(lw, ROCKE_ERR_VALUE, "vsnprintf format error");
            return;
        }
        text = (char*)rocke_arena_alloc(&lw->b->arena, (size_t)n + 1);
        if(!text)
        {
            va_end(ap);
            rocke_h_fail(lw, ROCKE_ERR_OOM, "out of memory formatting line");
            return;
        }
        vsnprintf(text, (size_t)n + 1, fmt, ap);
    }
    va_end(ap);
    rocke_h_emit(lw, text);
}

void rocke_h_emit_smem_decl(rocke_h_lowerer_t* lw, const char* decl)
{
    char* copy;
    int rc;
    if(!rocke_h_live(lw))
    {
        return;
    }
    copy = rocke_arena_strdup(&lw->b->arena, decl ? decl : "");
    if(!copy)
    {
        rocke_h_fail(lw, ROCKE_ERR_OOM, "out of memory copying smem decl");
        return;
    }
    rocke_vec_push(&lw->b->arena, &lw->smem_decls, copy, rc);
    if(rc != 0)
    {
        rocke_h_fail(lw, ROCKE_ERR_OOM, "out of memory appending smem decl");
    }
}

void rocke_h_push_indent(rocke_h_lowerer_t* lw)
{
    if(lw)
    {
        lw->indent += 1;
    }
}

void rocke_h_pop_indent(rocke_h_lowerer_t* lw)
{
    if(lw)
    {
        lw->indent -= 1;
    }
}

/* ============================== naming / types ======================= */

const char* rocke_h_name(rocke_h_lowerer_t* lw, const rocke_value_t* v)
{
    const char* n;
    if(!v || !v->name)
    {
        return "";
    }
    n = v->name;
    if(n[0] == '%')
    {
        n = n + 1;
    }
    /* Return an arena-owned copy (matches Python _name returning a str). */
    if(lw && lw->b)
    {
        char* copy = rocke_arena_strdup(&lw->b->arena, n);
        return copy ? copy : "";
    }
    return n;
}

/* The raw _HIP_TYPE dict: scalar IR name -> HIP scalar spelling. NULL if
 * unknown (caller decides whether that is an error). */
const char* rocke_h_hip_scalar(const char* ir_scalar_name)
{
    if(!ir_scalar_name)
    {
        return NULL;
    }
    if(strcmp(ir_scalar_name, "i1") == 0)
    {
        return "bool";
    }
    if(strcmp(ir_scalar_name, "i8") == 0)
    {
        return "int8_t";
    }
    if(strcmp(ir_scalar_name, "i16") == 0)
    {
        return "int16_t";
    }
    if(strcmp(ir_scalar_name, "i32") == 0)
    {
        return "int";
    }
    if(strcmp(ir_scalar_name, "i64") == 0)
    {
        return "int64_t";
    }
    if(strcmp(ir_scalar_name, "f16") == 0)
    {
        return "fp16";
    }
    if(strcmp(ir_scalar_name, "bf16") == 0)
    {
        return "bf16";
    }
    if(strcmp(ir_scalar_name, "f32") == 0)
    {
        return "float";
    }
    if(strcmp(ir_scalar_name, "fp8e4m3") == 0)
    {
        return "fp8e4m3";
    }
    if(strcmp(ir_scalar_name, "bf8e5m2") == 0)
    {
        return "bf8e5m2";
    }
    return NULL;
}

const char* rocke_h_vec_prefix(const char* ir_scalar_name, bool full_map)
{
    if(ir_scalar_name)
    {
        /* Small 2-entry map ({f16,bf16}) always available. */
        if(strcmp(ir_scalar_name, "f16") == 0)
        {
            return "f16x";
        }
        if(strcmp(ir_scalar_name, "bf16") == 0)
        {
            return "bf16x";
        }
        if(full_map)
        {
            if(strcmp(ir_scalar_name, "f32") == 0)
            {
                return "f32x";
            }
            if(strcmp(ir_scalar_name, "i32") == 0)
            {
                return "i32x";
            }
            if(strcmp(ir_scalar_name, "i16") == 0)
            {
                return "i16x";
            }
            if(strcmp(ir_scalar_name, "i8") == 0)
            {
                return "i8x";
            }
            if(strcmp(ir_scalar_name, "fp8e4m3") == 0)
            {
                return "i8x";
            }
            if(strcmp(ir_scalar_name, "bf8e5m2") == 0)
            {
                return "i8x";
            }
        }
    }
    /* Python .get(elem, "f16x") fallback. */
    return "f16x";
}

/* Python _type_to_hip(t). Returns arena-owned string; "" + sticky error on an
 * unmappable type (KeyError parity). */
const char* rocke_h_type_to_hip(rocke_h_lowerer_t* lw, const rocke_type_t* t)
{
    if(!t)
    {
        rocke_h_fail(lw, ROCKE_ERR_TYPE, "type_to_hip: NULL type");
        return "";
    }

    if(t->kind == ROCKE_TYPE_PTR)
    {
        if(t->space && (strcmp(t->space, "global") == 0 || strcmp(t->space, "lds") == 0))
        {
            const char* pointee = rocke_h_type_to_hip(lw, t->pointee);
            char* out;
            if(!rocke_h_live(lw))
            {
                return "";
            }
            out = rocke_arena_printf(&lw->b->arena, "%s*", pointee);
            return out ? out : "";
        }
        /* Other ptr spaces fall through to the scalar map (KeyError in Python). */
    }

    if(t->kind == ROCKE_TYPE_VECTOR)
    {
        const char* elem = t->elem ? t->elem->name : "";
        const char* pfx;
        char* out;
        if(strcmp(elem, "i1") == 0)
        {
            pfx = "boolx";
        }
        else
        {
            /* full 8-entry map; fp8/bf8 fold to i8x as in Python. */
            pfx = rocke_h_vec_prefix(elem, /*full_map=*/true);
            /* vec_prefix falls back to "f16x" for unknown elems, but the
             * Python code only emits the listed elems and otherwise falls
             * through to the KeyError. Detect the listed set explicitly so an
             * unknown vector elem is an error rather than silently "f16x". */
            if(strcmp(elem, "f16") != 0 && strcmp(elem, "bf16") != 0 && strcmp(elem, "f32") != 0
               && strcmp(elem, "i32") != 0 && strcmp(elem, "i16") != 0 && strcmp(elem, "i8") != 0
               && strcmp(elem, "fp8e4m3") != 0 && strcmp(elem, "bf8e5m2") != 0)
            {
                rocke_h_fail(lw, ROCKE_ERR_KEY, "type_to_hip: unmappable vector elem '%s'", elem);
                return "";
            }
        }
        out = rocke_arena_printf(&lw->b->arena, "%s%d", pfx, t->count);
        return out ? out : "";
    }

    if(t->kind == ROCKE_TYPE_SMEM)
    {
        const char* elem = rocke_h_type_to_hip(lw, t->elem);
        char* out;
        if(!rocke_h_live(lw))
        {
            return "";
        }
        out = rocke_arena_printf(&lw->b->arena, "%s*", elem);
        return out ? out : "";
    }

    /* scalar: _HIP_TYPE[t.name] */
    {
        const char* s = rocke_h_hip_scalar(t->name);
        if(!s)
        {
            rocke_h_fail(lw,
                         ROCKE_ERR_KEY,
                         "type_to_hip: unmappable type '%s'",
                         t->name ? t->name : "(null)");
            return "";
        }
        return s;
    }
}

/* ============================== literals ============================= */

const char* rocke_h_f32_literal(rocke_h_lowerer_t* lw, double val)
{
    char* out;
    if(!lw || !lw->b)
    {
        return "";
    }
    if(isnan(val))
    {
        out = rocke_arena_strdup(&lw->b->arena, "((float)NAN)");
        return out ? out : "";
    }
    if(isinf(val))
    {
        out = rocke_arena_strdup(&lw->b->arena,
                                 val < 0 ? "((float)-INFINITY)" : "((float)INFINITY)");
        return out ? out : "";
    }
    /* NOTE(port): Python emits repr(float)+"f"; %g is a close approximation but
     * NOT guaranteed byte-identical to CPython repr. Known port hazard. */
    out = rocke_arena_printf(&lw->b->arena, "%gf", val);
    return out ? out : "";
}

/* ============================== waitcnt encoders ===================== */

static int h_clamp(int v, int lo, int hi)
{
    if(v < lo)
    {
        return lo;
    }
    if(v > hi)
    {
        return hi;
    }
    return v;
}

int rocke_h_encode_waitcnt_gfx9_10(int vmcnt, int expcnt, int lgkmcnt)
{
    int vm_b = (vmcnt < 0) ? 0x3F : h_clamp(vmcnt, 0, 0x3F);
    int ec_b = (expcnt < 0) ? 0x7 : h_clamp(expcnt, 0, 0x7);
    int lk_b = (lgkmcnt < 0) ? 0xF : h_clamp(lgkmcnt, 0, 0xF);
    return (vm_b & 0xF) | (ec_b << 4) | (lk_b << 8) | (((vm_b >> 4) & 0x3) << 14);
}

int rocke_h_encode_waitcnt_gfx11(int vmcnt, int expcnt, int lgkmcnt)
{
    int vm_b = (vmcnt < 0) ? 0x3F : h_clamp(vmcnt, 0, 0x3F);
    int ec_b = (expcnt < 0) ? 0x7 : h_clamp(expcnt, 0, 0x7);
    int lk_b = (lgkmcnt < 0) ? 0x3F : h_clamp(lgkmcnt, 0, 0x3F);
    return (ec_b & 0x7) | ((lk_b & 0x3F) << 4) | ((vm_b & 0x3F) << 10);
}

int rocke_h_encode_waitcnt(const rocke_h_lowerer_t* lw, int vmcnt, int expcnt, int lgkmcnt)
{
    if(lw && lw->arch.waitcnt_family == ROCKE_HIP_WAITCNT_GFX11)
    {
        return rocke_h_encode_waitcnt_gfx11(vmcnt, expcnt, lgkmcnt);
    }
    return rocke_h_encode_waitcnt_gfx9_10(vmcnt, expcnt, lgkmcnt);
}

/* ============================== arch gates =========================== */

rocke_status_t rocke_h_require_wmma_arch(rocke_h_lowerer_t* lw, const char* op_id)
{
    if(!rocke_h_live(lw))
    {
        return lw ? lw->status : ROCKE_ERR_NOTIMPL;
    }
    /* NOTE(port): Python keys off the ArchTarget MMA catalog (mma.by_op_id);
     * that catalog is not ported, so we use lw->arch.has_wmma. */
    if(!lw->arch.has_wmma)
    {
        return rocke_h_fail(lw,
                            ROCKE_ERR_NOTIMPL,
                            "WMMA op '%s' is not available on %s "
                            "(WMMA is an RDNA/gfx11 instruction; this is a %s target). "
                            "The MFMA atoms are the matrix path on CDNA.",
                            op_id ? op_id : "(null)",
                            lw->arch.gfx ? lw->arch.gfx : "(unknown)",
                            lw->arch.family ? lw->arch.family : "cdna");
    }
    return ROCKE_OK;
}

rocke_status_t rocke_h_require_ds_read_tr(rocke_h_lowerer_t* lw, const char* op_id)
{
    if(!rocke_h_live(lw))
    {
        return lw ? lw->status : ROCKE_ERR_NOTIMPL;
    }
    if(!lw->arch.has_ds_read_tr)
    {
        return rocke_h_fail(lw,
                            ROCKE_ERR_NOTIMPL,
                            "transpose LDS read '%s' is not available on %s "
                            "(no ds_read_*_tr_* on this target); "
                            "it is a gfx950-class instruction.",
                            op_id ? op_id : "(null)",
                            lw->arch.gfx ? lw->arch.gfx : "(unknown)");
    }
    return ROCKE_OK;
}

/* ============================== smem side table ====================== */

/* The lowerer threads no extra state pointer to the buckets beyond lw itself,
 * so the smem storage side table is hung off the lowerer via a small
 * arena-backed parallel array keyed by the producing Value pointer. We grow it
 * lazily; entries are never removed. Stored in two arena-allocated arrays
 * referenced from a header stashed at the front of a dedicated arena block,
 * pointed to by a static-free design: we keep the table in the lowerer struct?
 * The struct is fixed by the header, so instead we maintain a tiny intrusive
 * arena list anchored in a per-call singly-linked node chain reachable from the
 * builder arena. To avoid mutating the public lowerer struct, the table head is
 * embedded as the first emitted smem_decls slot? No -- cleanest within the ABI:
 * keep a per-lowerer hashless linear list whose head we recover by scanning a
 * dedicated chain. Simplest correct approach: a singly linked list allocated in
 * the arena, whose head we store in a small registry keyed by lowerer pointer.
 */

typedef struct h_smem_node
{
    const rocke_value_t* key;
    const char* storage;
    struct h_smem_node* next;
} h_smem_node_t;

/* Registry mapping a lowerer instance -> its smem list head. lowerers are
 * stack-scoped and there is exactly one active per call in practice, but a
 * small linear registry keeps this robust and reentrant-enough for the port.
 * Entries are reused across calls keyed by the lowerer pointer; since lowerers
 * live on the stack, a stale pointer can only collide after the prior call
 * fully returned, at which point its list is logically dead -- we reset the
 * list head whenever a fresh lowering binds the slot (see driver below). */
#define H_SMEM_REGISTRY_CAP 16
typedef struct h_smem_reg_entry
{
    const rocke_h_lowerer_t* lw;
    h_smem_node_t* head;
} h_smem_reg_entry_t;

static h_smem_reg_entry_t g_smem_registry[H_SMEM_REGISTRY_CAP];

static h_smem_reg_entry_t* h_smem_slot(const rocke_h_lowerer_t* lw, int create)
{
    int i, free_idx = -1;
    for(i = 0; i < H_SMEM_REGISTRY_CAP; i++)
    {
        if(g_smem_registry[i].lw == lw)
        {
            return &g_smem_registry[i];
        }
        if(free_idx < 0 && g_smem_registry[i].lw == NULL)
        {
            free_idx = i;
        }
    }
    if(!create)
    {
        return NULL;
    }
    if(free_idx < 0)
    {
        /* registry full: overwrite slot 0 (best-effort; rare in practice). */
        free_idx = 0;
    }
    g_smem_registry[free_idx].lw = lw;
    g_smem_registry[free_idx].head = NULL;
    return &g_smem_registry[free_idx];
}

/* Bind a fresh, empty smem table to this lowerer for a new lowering pass. */
static void h_smem_reset(const rocke_h_lowerer_t* lw)
{
    h_smem_reg_entry_t* slot = h_smem_slot(lw, /*create=*/1);
    if(slot)
    {
        slot->head = NULL;
    }
}

/* Release this lowerer's registry slot once lowering has finished. */
static void h_smem_release(const rocke_h_lowerer_t* lw)
{
    h_smem_reg_entry_t* slot = h_smem_slot(lw, /*create=*/0);
    if(slot)
    {
        slot->lw = NULL;
        slot->head = NULL;
    }
}

void rocke_h_smem_set_storage(rocke_h_lowerer_t* lw,
                              const rocke_value_t* smem_result,
                              const char* storage_name)
{
    h_smem_reg_entry_t* slot;
    h_smem_node_t* node;
    if(!rocke_h_live(lw))
    {
        return;
    }
    slot = h_smem_slot(lw, /*create=*/1);
    if(!slot)
    {
        rocke_h_fail(lw, ROCKE_ERR_OOM, "smem registry exhausted");
        return;
    }
    node = (h_smem_node_t*)rocke_arena_alloc(&lw->b->arena, sizeof(*node));
    if(!node)
    {
        rocke_h_fail(lw, ROCKE_ERR_OOM, "out of memory recording smem storage");
        return;
    }
    node->key = smem_result;
    node->storage = rocke_arena_strdup(&lw->b->arena, storage_name ? storage_name : "");
    node->next = slot->head;
    slot->head = node;
}

const char* rocke_h_smem_storage(rocke_h_lowerer_t* lw, const rocke_value_t* smem_result)
{
    h_smem_reg_entry_t* slot = h_smem_slot(lw, /*create=*/0);
    h_smem_node_t* n;
    if(!slot)
    {
        return NULL;
    }
    for(n = slot->head; n; n = n->next)
    {
        if(n->key == smem_result)
        {
            return n->storage;
        }
    }
    return NULL;
}

/* ============================== dispatch ============================= */

/* Opcode-indexed dispatch table, assembled once from every bucket's
 * registration array. Buckets export rocke_h_handlers_*(); a NULL-handler /
 * ROCKE_OP_INVALID terminator ends each table. */
static rocke_h_handler_fn g_dispatch[ROCKE_OP__COUNT];
static int g_dispatch_ready;

static void h_install(const rocke_h_handler_entry_t* table)
{
    if(!table)
    {
        return;
    }
    for(; table->opcode != ROCKE_OP_INVALID && table->handler != NULL; table++)
    {
        if((int)table->opcode > 0 && (int)table->opcode < ROCKE_OP__COUNT)
        {
            g_dispatch[table->opcode] = table->handler;
        }
    }
}

static void h_build_dispatch(void)
{
    if(g_dispatch_ready)
    {
        return;
    }
    memset(g_dispatch, 0, sizeof(g_dispatch));
    /* core/arith + math bucket (lower_hip_ops.c) */
    h_install(rocke_h_handlers_arith());
    /* casts/conversions + gpu ids */
    h_install(rocke_h_handlers_cast());
    /* memory / LDS / buffer / async / atomics */
    h_install(rocke_h_handlers_mem());
    /* mma / cross-lane / barriers / vector / control flow */
    h_install(rocke_h_handlers_mma());
    g_dispatch_ready = 1;
}

rocke_h_handler_fn rocke_h_dispatch(rocke_opcode_t opcode)
{
    h_build_dispatch();
    if((int)opcode <= 0 || (int)opcode >= ROCKE_OP__COUNT)
    {
        return NULL;
    }
    return g_dispatch[opcode];
}

/* ============================== walkers ============================== */

rocke_status_t rocke_h_lower_op(rocke_h_lowerer_t* lw, const rocke_op_t* op)
{
    rocke_h_handler_fn fn;
    if(!rocke_h_live(lw))
    {
        return lw ? lw->status : ROCKE_ERR_NOTIMPL;
    }
    if(!op)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "lower_op: NULL op");
    }
    fn = rocke_h_dispatch(op->opcode);
    if(!fn)
    {
        /* NotImplementedError parity. */
        return rocke_h_fail(lw,
                            ROCKE_ERR_NOTIMPL,
                            "no HIP lowering for op '%s'",
                            op->name ? op->name : rocke_opcode_name(op->opcode));
    }
    return fn(lw, op);
}

rocke_status_t rocke_h_lower_region(rocke_h_lowerer_t* lw, const rocke_region_t* region)
{
    int i;
    if(!rocke_h_live(lw))
    {
        return lw ? lw->status : ROCKE_ERR_NOTIMPL;
    }
    if(!region)
    {
        return rocke_h_fail(lw, ROCKE_ERR_VALUE, "lower_region: NULL region");
    }
    for(i = 0; i < region->num_ops; i++)
    {
        rocke_h_lower_op(lw, region->ops[i]);
        if(!rocke_h_live(lw))
        {
            return lw->status;
        }
    }
    return lw->status;
}

} /* namespace ckc */

/* ============================== arch seam =========================== */

rocke_hip_arch_t rocke_hip_arch_from_gfx(const char* gfx)
{
    rocke_hip_arch_t a;

    /* Default: gfx950 CDNA facts (the byte-identical baseline). */
    a.gfx = "gfx950";
    a.waitcnt_family = ROCKE_HIP_WAITCNT_GFX9_10;
    a.has_ds_read_tr = true;
    a.has_wmma = false;
    a.family = "cdna";

    if(!gfx)
    {
        return a;
    }

    if(strcmp(gfx, "gfx950") == 0)
    {
        a.gfx = "gfx950";
        a.waitcnt_family = ROCKE_HIP_WAITCNT_GFX9_10;
        a.has_ds_read_tr = true;
        a.has_wmma = false;
        a.family = "cdna";
    }
    else if(strcmp(gfx, "gfx942") == 0 || strcmp(gfx, "gfx940") == 0 || strcmp(gfx, "gfx90a") == 0
            || strcmp(gfx, "gfx908") == 0)
    {
        /* CDNA, but no ds_read_*_tr_* and no WMMA. */
        a.gfx = gfx;
        a.waitcnt_family = ROCKE_HIP_WAITCNT_GFX9_10;
        a.has_ds_read_tr = false;
        a.has_wmma = false;
        a.family = "cdna";
    }
    else if(strncmp(gfx, "gfx11", 5) == 0)
    {
        /* RDNA3: contiguous waitcnt, WMMA, no ds_read_tr. */
        a.gfx = gfx;
        a.waitcnt_family = ROCKE_HIP_WAITCNT_GFX11;
        a.has_ds_read_tr = false;
        a.has_wmma = true;
        a.family = "rdna";
    }
    else if(strncmp(gfx, "gfx12", 5) == 0)
    {
        /* RDNA4: contiguous waitcnt family, WMMA. */
        a.gfx = gfx;
        a.waitcnt_family = ROCKE_HIP_WAITCNT_GFX11;
        a.has_ds_read_tr = false;
        a.has_wmma = true;
        a.family = "rdna";
    }
    /* Anything unrecognised keeps the gfx950 default facts (per the header). */

    return a;
}

/* ============================== public entry ======================== */

rocke_status_t rocke_lower_kernel_to_hip(rocke_ir_builder_t* b,
                                         const rocke_kernel_def_t* kernel,
                                         const rocke_lower_hip_opts_t* opts,
                                         rocke_strbuf_t* out)
{
    using namespace ckc; /* the lowerer's private symbols (state, helpers) */
    rocke_h_lowerer_t lw;
    int launch_bounds;
    bool include_prologue;
    const char* arch_name;
    rocke_strbuf_t sig;
    int i;
    size_t li;

    if(!b || !kernel || !out)
    {
        return ROCKE_ERR_VALUE;
    }

    /* ---- resolve options (Python keyword defaults) ---- */
    launch_bounds = rocke_kernel_max_workgroup_size(kernel);
    include_prologue = true;
    arch_name = NULL;
    if(opts)
    {
        if(opts->launch_bounds_set)
        {
            launch_bounds = opts->launch_bounds;
        }
        if(opts->include_prologue_set)
        {
            include_prologue = opts->include_prologue;
        }
        arch_name = opts->arch;
    }

    /* ---- init lowerer ---- */
    memset(&lw, 0, sizeof(lw));
    lw.kernel = kernel;
    lw.b = b;
    lw.arch = rocke_hip_arch_from_gfx(arch_name);
    rocke_vec_init(&lw.lines);
    rocke_vec_init(&lw.smem_decls);
    lw.indent = 1;
    lw.smem_counter = 0;
    lw.status = ROCKE_OK;
    lw.err[0] = '\0';

    h_smem_reset(&lw);

    /* ---- build the signature (Python sig_args loop) ---- */
    if(rocke_strbuf_init(&sig, 64) != 0)
    {
        h_smem_release(&lw);
        return ROCKE_ERR_OOM;
    }

    /* A failure anywhere in lowering raises a ckc::Error; catch it here so the
     * signature buffer and smem registry slot are always released, then
     * translate it into the legacy status code (keeping the extern "C" ABI
     * unchanged). */
    try
    {
        for(i = 0; i < kernel->num_params; i++)
        {
            const rocke_param_t* p = kernel->params[i];
            const char* t = rocke_h_type_to_hip(&lw, p->type);
            if(i > 0)
            {
                rocke_strbuf_append(&sig, ", ");
            }
            if(strchr(t, '*') != NULL)
            {
                rocke_strbuf_appendf(&sig, "%s __restrict__ %s", t, p->name);
            }
            else
            {
                rocke_strbuf_appendf(&sig, "%s %s", t, p->name);
            }
        }

        /* ---- lower the body ---- */
        rocke_h_lower_region(&lw, kernel->body);

        /* ---- assemble parts (Python parts list joined with '\n') ---- */
        if(include_prologue)
        {
            rocke_strbuf_append(out, ROCKE_HIP_PROLOGUE);
            rocke_strbuf_append_char(out, '\n');
        }
        /* head */
        rocke_strbuf_appendf(out,
                             "extern \"C\" __global__ __launch_bounds__(%d)\n"
                             "void %s(%s)\n{",
                             launch_bounds,
                             kernel->name ? kernel->name : "",
                             rocke_strbuf_cstr(&sig));

        /* smem block (only if non-empty, like Python `if smem_block`). */
        for(li = 0; li < lw.smem_decls.len; li++)
        {
            rocke_strbuf_append_char(out, '\n');
            rocke_strbuf_append(out, lw.smem_decls.data[li]);
        }

        /* body block (always appended, joined with '\n'). */
        for(li = 0; li < lw.lines.len; li++)
        {
            rocke_strbuf_append_char(out, '\n');
            rocke_strbuf_append(out, lw.lines.data[li]);
        }

        /* closing brace */
        rocke_strbuf_append(out, "\n}");
    }
    catch(const ckc::Error& e)
    {
        rocke_strbuf_free(&sig);
        h_smem_release(&lw);
        return e.code();
    }
    catch(const std::bad_alloc&)
    {
        rocke_strbuf_free(&sig);
        h_smem_release(&lw);
        return ROCKE_ERR_OOM;
    }
    catch(const std::exception&)
    {
        rocke_strbuf_free(&sig);
        h_smem_release(&lw);
        return ROCKE_ERR_VALUE;
    }
    catch(...)
    {
        rocke_strbuf_free(&sig);
        h_smem_release(&lw);
        return ROCKE_ERR_VALUE;
    }

    rocke_strbuf_free(&sig);
    h_smem_release(&lw);

    if(out->oom)
    {
        return ROCKE_ERR_OOM;
    }
    return ROCKE_OK;
}
