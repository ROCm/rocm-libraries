// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_gfx1250_qwen3_kv_cache.c -- C99 port of
 * rocke/instances/gfx1250/qwen3_kv_cache.py.
 *
 * Two KV-cache-side kernels: kv_dequant_smoke (fp8/bf8 read + dequant) and
 * kv_append_rope (KV append/update + optional RoPE + quantized paged store). The
 * build op order tracks the Python builds top-to-bottom; emitted IR is
 * byte-identical to the Python lowerer (see SSA-order notes: args sequenced
 * left-to-right, const_i32 never deduped).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx1250_qwen3_kv_cache.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.attention.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/lower_llvm.h"

#define ROCKE_QKV_HEADS 4
#define ROCKE_QKV_HEAD_DIM 64
#define ROCKE_QKV_BLOCK_SIZE 16

/* _dtype_ir: fp16->f16, bf16->bf16, fp8e4m3->fp8e4m3, bf8e5m2->bf8e5m2. */
static const rocke_type_t* rocke_qkv_dtype_ir(const char* dtype)
{
    if(dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "fp16") == 0)
    {
        return rocke_f16();
    }
    if(strcmp(dtype, "bf16") == 0)
    {
        return rocke_bf16();
    }
    if(strcmp(dtype, "fp8e4m3") == 0)
    {
        return rocke_fp8e4m3();
    }
    if(strcmp(dtype, "bf8e5m2") == 0)
    {
        return rocke_bf8e5m2();
    }
    return NULL;
}

/* _require_gfx1250(arch): resolves, arch=="gfx1250", wave32 + cdna. */
static bool rocke_qkv_require_gfx1250(const char* arch, char* reason, size_t reason_cap)
{
    const rocke_arch_target_t* target;
    char buf[ROCKE_ERR_MSG_CAP];

    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "unknown gfx target '%s'", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(strcmp(arch, "gfx1250") != 0)
    {
        snprintf(
            buf, sizeof(buf), "Qwen3 gfx1250 KV kernels require arch='gfx1250', got '%s'", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(target->wave_size != 32 || strcmp(target->family, "cdna") != 0)
    {
        snprintf(buf, sizeof(buf), "gfx1250 contract expected CDNA wave32 target, got %s", arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "supported");
    return true;
}

/* ===================================================================== *
 *  Qwen3KvDequantSpec
 * ===================================================================== */

rocke_qwen3_kv_dequant_gfx1250_spec_t rocke_qwen3_kv_dequant_gfx1250_spec_default(void)
{
    rocke_qwen3_kv_dequant_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.kv_storage_dtype = NULL; /* required, no Python default */
    s.output_dtype = "bf16";
    s.head_dim = ROCKE_QKV_HEAD_DIM;
    s.name = "rocke_gfx1250_qwen3_kv_dequant";
    return s;
}

rocke_status_t rocke_qwen3_kv_dequant_gfx1250_kernel_name(
    const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char d_part[32];
    const char* parts[3];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(d_part, sizeof(d_part), "d%d", spec->head_dim);
    parts[0] = d_part;
    parts[1] = spec->kv_storage_dtype;
    parts[2] = spec->output_dtype;
    return rocke_kernel_name_join(spec->name, parts, 3, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_qwen3_kv_dequant_gfx1250_is_valid_spec(const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec,
                                                  const char* arch,
                                                  char* reason,
                                                  size_t reason_cap)
{
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    /* __post_init__ */
    if(spec->kv_storage_dtype == NULL
       || (strcmp(spec->kv_storage_dtype, "fp8e4m3") != 0
           && strcmp(spec->kv_storage_dtype, "bf8e5m2") != 0))
    {
        rocke_spec_set_reason(
            reason, reason_cap, "kv_storage_dtype must be 'fp8e4m3' or 'bf8e5m2'");
        return false;
    }
    if(spec->output_dtype == NULL
       || (strcmp(spec->output_dtype, "fp16") != 0 && strcmp(spec->output_dtype, "bf16") != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "output_dtype must be 'fp16' or 'bf16'");
        return false;
    }
    if(spec->head_dim != ROCKE_QKV_HEAD_DIM)
    {
        rocke_spec_set_reason(reason, reason_cap, "Qwen3-30B-A3B head_dim must be 64");
        return false;
    }
    return rocke_qkv_require_gfx1250(arch, reason, reason_cap);
}

rocke_kernel_def_t* rocke_build_qwen3_kv_dequant_smoke_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* storage_dtype;
        const rocke_type_t* out_dtype;
        rocke_value_t* src;
        rocke_value_t* dst;
        rocke_value_t* scale;
        rocke_value_t* lane;
        rocke_value_t* base;
        rocke_value_t* raw;
        rocke_value_t* deq;
        int i;
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(!rocke_qwen3_kv_dequant_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "%s", reason);
            return NULL;
        }

        storage_dtype = rocke_qkv_dtype_ir(spec->kv_storage_dtype);
        out_dtype = rocke_qkv_dtype_ir(spec->output_dtype);

        /* b.kernel.attrs["max_workgroup_size"] = 32 */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 32);

        /* params: src/dst use readonly/writeonly + align16, NO noalias (Python). */
        {
            rocke_param_opts_t ro;
            rocke_param_opts_t wo;
            memset(&ro, 0, sizeof(ro));
            ro.readonly = true;
            ro.readonly_set = true;
            ro.align = 16;
            ro.align_set = true;
            memset(&wo, 0, sizeof(wo));
            wo.writeonly = true;
            wo.writeonly_set = true;
            wo.align = 16;
            wo.align_set = true;

            src = rocke_b_param(b, "src", rocke_ptr_type(b, storage_dtype, "global"), &ro);
            dst = rocke_b_param(b, "dst", rocke_ptr_type(b, out_dtype, "global"), &wo);
            scale = rocke_b_param(b, "scale", rocke_f32(), NULL);
        }

        /* lane = b.thread_id_x(); base = b.mul(lane, b.const_i32(8)) */
        lane = rocke_b_thread_id_x(b);
        base = rocke_b_mul(b, lane, rocke_b_const_i32(b, 8));
        /* raw = b.global_load_vN(src, base, storage_dtype, 8, align=8) */
        raw = rocke_b_global_load_vN(b, src, base, storage_dtype, 8, /*align=*/8);
        /* deq = dequant_{fp8,bf8}x8_to_dtype(b, raw, scale, out_dtype) */
        if(strcmp(spec->kv_storage_dtype, "fp8e4m3") == 0)
        {
            deq = rocke_dequant_fp8x8_to_dtype(b, raw, scale, out_dtype);
        }
        else
        {
            deq = rocke_dequant_bf8x8_to_dtype(b, raw, scale, out_dtype);
        }
        /* for i in range(8): b.global_store(dst, b.add(base, b.const_i32(i)),
         *                                   b.vec_extract(deq, i), align=2)
         * idx (add) and value (vec_extract) both side-effecting; sequence idx first. */
        for(i = 0; i < 8; ++i)
        {
            rocke_value_t* soff = rocke_b_add(b, base, rocke_b_const_i32(b, i));
            rocke_value_t* sval = rocke_b_vec_extract(b, deq, i);
            rocke_b_global_store(b, dst, soff, sval, /*align=*/2);
        }

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_qwen3_kv_dequant_smoke_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_qwen3_kv_dequant_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_qwen3_kv_dequant_smoke_gfx1250(b, spec, arch);
    });
}

rocke_status_t
    rocke_qwen3_kv_dequant_gfx1250_lower_to_llvm(const rocke_qwen3_kv_dequant_gfx1250_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    kernel = rocke_build_qwen3_kv_dequant_smoke_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n = m ? strlen(m) : 0;
            if(m == NULL)
            {
                m = "build_qwen3_kv_dequant_smoke_gfx1250 failed";
                n = strlen(m);
            }
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}

/* ===================================================================== *
 *  Qwen3KvAppendRopeSpec
 * ===================================================================== */

rocke_qwen3_kv_append_rope_gfx1250_spec_t rocke_qwen3_kv_append_rope_gfx1250_spec_default(void)
{
    rocke_qwen3_kv_append_rope_gfx1250_spec_t s;
    memset(&s, 0, sizeof(s));
    s.input_dtype = "bf16";
    s.kv_storage_dtype = "bf16";
    s.head_dim = ROCKE_QKV_HEAD_DIM;
    s.block_size = ROCKE_QKV_BLOCK_SIZE;
    s.num_kv_heads = ROCKE_QKV_HEADS;
    s.use_rope = true;
    s.name = "rocke_gfx1250_qwen3_kv_append_rope";
    return s;
}

rocke_status_t rocke_qwen3_kv_append_rope_gfx1250_kernel_name(
    const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec, char* out, size_t out_cap)
{
    char d_part[32];
    char b_part[32];
    char kvh_part[32];
    char kv_part[64];
    const char* parts[6];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(d_part, sizeof(d_part), "d%d", spec->head_dim);
    snprintf(b_part, sizeof(b_part), "b%d", spec->block_size);
    snprintf(kvh_part, sizeof(kvh_part), "kvh%d", spec->num_kv_heads);
    snprintf(kv_part, sizeof(kv_part), "kv%s", spec->kv_storage_dtype);
    parts[0] = d_part;
    parts[1] = b_part;
    parts[2] = kvh_part;
    parts[3] = spec->input_dtype;
    parts[4] = kv_part;
    /* "rope" if use_rope else "" -- kernel_name_join drops empty parts. */
    parts[5] = spec->use_rope ? "rope" : "";
    return rocke_kernel_name_join(spec->name, parts, 6, NULL, NULL, 0, out, out_cap, NULL);
}

bool rocke_qwen3_kv_append_rope_gfx1250_is_valid_spec(
    const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec,
    const char* arch,
    char* reason,
    size_t reason_cap)
{
    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(spec->input_dtype == NULL
       || (strcmp(spec->input_dtype, "fp16") != 0 && strcmp(spec->input_dtype, "bf16") != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "input_dtype must be 'fp16' or 'bf16'");
        return false;
    }
    if(spec->kv_storage_dtype == NULL
       || (strcmp(spec->kv_storage_dtype, "bf16") != 0
           && strcmp(spec->kv_storage_dtype, "fp8e4m3") != 0
           && strcmp(spec->kv_storage_dtype, "bf8e5m2") != 0))
    {
        rocke_spec_set_reason(reason, reason_cap, "kv_storage_dtype must be bf16/fp8e4m3/bf8e5m2");
        return false;
    }
    if(spec->head_dim != ROCKE_QKV_HEAD_DIM || spec->block_size != ROCKE_QKV_BLOCK_SIZE)
    {
        rocke_spec_set_reason(
            reason, reason_cap, "Qwen3-30B-A3B KV scaffold is fixed at d64 block16");
        return false;
    }
    return rocke_qkv_require_gfx1250(arch, reason, reason_cap);
}

/* _quantize_for_kv_store(b, v, storage_dtype, scale). */
static rocke_value_t* rocke_qkv_quantize_for_store(rocke_ir_builder_t* b,
                                                   rocke_value_t* v,
                                                   const char* storage_dtype,
                                                   rocke_value_t* scale)
{
    if(strcmp(storage_dtype, "bf16") == 0)
    {
        return rocke_b_cast_f32_to(b, v, rocke_bf16());
    }
    /* scaled = b.fdiv(v, scale) */
    {
        rocke_value_t* scaled = rocke_b_fdiv(b, v, scale);
        if(strcmp(storage_dtype, "fp8e4m3") == 0)
        {
            return rocke_b_cvt_f32_to_fp8(b, scaled);
        }
        return rocke_b_cvt_f32_to_bf8(b, scaled);
    }
}

rocke_kernel_def_t* rocke_build_qwen3_kv_append_rope_gfx1250(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* in_dtype;
        const rocke_type_t* storage_dtype;
        const rocke_type_t* f32;
        rocke_value_t* key_in;
        rocke_value_t* value_in;
        rocke_value_t* k_cache;
        rocke_value_t* v_cache;
        rocke_value_t* block_tables;
        rocke_value_t* slot_ids;
        rocke_value_t* cos;
        rocke_value_t* sin;
        rocke_value_t* k_scale;
        rocke_value_t* v_scale;
        rocke_value_t* token;
        rocke_value_t* kv_head;
        rocke_value_t* dim;
        rocke_value_t* slot;
        rocke_value_t* logical_block;
        rocke_value_t* token_in_block;
        rocke_value_t* physical_block;
        rocke_value_t* in_base;
        rocke_value_t* value_h;
        rocke_value_t* value_f;
        rocke_value_t* key_f;
        rocke_value_t* out_idx;
        rocke_value_t* k_store;
        rocke_value_t* v_store;
        int store_align;
        int stride_0;
        int stride_1;
        int stride_2;
        int stride_3;
        char reason[ROCKE_ERR_MSG_CAP];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(!rocke_qwen3_kv_append_rope_gfx1250_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            (void)rocke_i_set_err(b, ROCKE_ERR_NOTIMPL, "%s", reason);
            return NULL;
        }

        in_dtype = rocke_qkv_dtype_ir(spec->input_dtype);
        storage_dtype = rocke_qkv_dtype_ir(spec->kv_storage_dtype);
        f32 = rocke_f32();

        /* b.kernel.attrs["max_workgroup_size"] = 1 */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 1);

        /* params: readonly inputs + writeonly caches, align16, NO noalias. */
        {
            rocke_param_opts_t ro;
            rocke_param_opts_t wo;
            const rocke_type_t* ptr_in = rocke_ptr_type(b, in_dtype, "global");
            const rocke_type_t* ptr_st = rocke_ptr_type(b, storage_dtype, "global");
            const rocke_type_t* ptr_i32 = rocke_ptr_type(b, rocke_i32(), "global");
            const rocke_type_t* ptr_f32 = rocke_ptr_type(b, f32, "global");

            memset(&ro, 0, sizeof(ro));
            ro.readonly = true;
            ro.readonly_set = true;
            ro.align = 16;
            ro.align_set = true;
            memset(&wo, 0, sizeof(wo));
            wo.writeonly = true;
            wo.writeonly_set = true;
            wo.align = 16;
            wo.align_set = true;

            key_in = rocke_b_param(b, "key_in", ptr_in, &ro);
            value_in = rocke_b_param(b, "value_in", ptr_in, &ro);
            k_cache = rocke_b_param(b, "k_cache", ptr_st, &wo);
            v_cache = rocke_b_param(b, "v_cache", ptr_st, &wo);
            block_tables = rocke_b_param(b, "block_tables", ptr_i32, &ro);
            slot_ids = rocke_b_param(b, "slot_ids", ptr_i32, &ro);
            cos = rocke_b_param(b, "cos", ptr_f32, &ro);
            sin = rocke_b_param(b, "sin", ptr_f32, &ro);
            k_scale = rocke_b_param(b, "k_scale", f32, NULL);
            v_scale = rocke_b_param(b, "v_scale", f32, NULL);
        }

        /* token = block_id_x(); kv_head = block_id_y(); dim = block_id_z() */
        token = rocke_b_block_id_x(b);
        kv_head = rocke_b_block_id_y(b);
        dim = rocke_b_block_id_z(b);
        /* slot = b.global_load_i32(slot_ids, token) */
        slot = rocke_b_global_load_i32(b, slot_ids, token, /*align=*/-1);
        /* logical_block = b.div(slot, b.const_i32(block_size)) */
        logical_block = rocke_b_div(b, slot, rocke_b_const_i32(b, spec->block_size));
        /* token_in_block = b.mod(slot, b.const_i32(block_size)) */
        token_in_block = rocke_b_mod(b, slot, rocke_b_const_i32(b, spec->block_size));
        /* physical_block = b.global_load_i32(block_tables, logical_block) */
        physical_block = rocke_b_global_load_i32(b, block_tables, logical_block, /*align=*/-1);

        /* in_base = b.add(b.mul(b.add(b.mul(token, const(nkvh)), kv_head), const(hd)), dim)
         * Python evals each call's args left-to-right and never dedupes const_i32.
         * The const operands here are arg-2 of ops whose arg-1 is a side-effecting
         * expression, so they must be created AFTER their sibling -- sequence the
         * whole chain explicitly to match Python's emit order. */
        {
            rocke_value_t* m0 = rocke_b_mul(b, token, rocke_b_const_i32(b, spec->num_kv_heads));
            rocke_value_t* a0 = rocke_b_add(b, m0, kv_head);
            rocke_value_t* m1 = rocke_b_mul(b, a0, rocke_b_const_i32(b, spec->head_dim));
            in_base = rocke_b_add(b, m1, dim);
        }
        /* value_h = b.global_load(value_in, in_base, in_dtype, align=2) */
        value_h = rocke_b_global_load(b, value_in, in_base, in_dtype, /*align=*/2);
        /* value_f = b.cast_to_f32(value_h) */
        value_f = rocke_b_cast_to_f32(b, value_h);

        if(spec->use_rope)
        {
            rocke_value_t* pair_base;
            rocke_value_t* pair_lane;
            rocke_value_t* even_idx;
            rocke_value_t* odd_idx;
            rocke_value_t* even_f;
            rocke_value_t* odd_f;
            rocke_value_t* trig_idx;
            rocke_value_t* c;
            rocke_value_t* s;
            rocke_value_t* rot_even;
            rocke_value_t* rot_odd;
            rocke_value_t* is_odd;
            rocke_value_t* ec;
            rocke_value_t* os;
            rocke_value_t* es;
            rocke_value_t* oc;

            /* pair_base = b.mul(b.div(dim, const(2)), const(2))
             * arg-1 (div) is side-effecting; the trailing const(2) must be created
             * after it, so sequence explicitly. */
            {
                rocke_value_t* dh = rocke_b_div(b, dim, rocke_b_const_i32(b, 2));
                pair_base = rocke_b_mul(b, dh, rocke_b_const_i32(b, 2));
            }
            /* pair_lane = b.div(dim, const(2)) */
            pair_lane = rocke_b_div(b, dim, rocke_b_const_i32(b, 2));
            /* even_idx = b.add(b.mul(b.add(b.mul(token, const(nkvh)), kv_head),
             *                        const(hd)), pair_base)
             * Same const-after-operand sequencing as in_base. */
            {
                rocke_value_t* em0
                    = rocke_b_mul(b, token, rocke_b_const_i32(b, spec->num_kv_heads));
                rocke_value_t* ea0 = rocke_b_add(b, em0, kv_head);
                rocke_value_t* em1 = rocke_b_mul(b, ea0, rocke_b_const_i32(b, spec->head_dim));
                even_idx = rocke_b_add(b, em1, pair_base);
            }
            /* odd_idx = b.add(even_idx, const(1)) */
            odd_idx = rocke_b_add(b, even_idx, rocke_b_const_i32(b, 1));
            /* even_f = b.cast_to_f32(b.global_load(key_in, even_idx, in_dtype, align=2)) */
            even_f = rocke_b_cast_to_f32(
                b, rocke_b_global_load(b, key_in, even_idx, in_dtype, /*align=*/2));
            /* odd_f = b.cast_to_f32(b.global_load(key_in, odd_idx, in_dtype, align=2)) */
            odd_f = rocke_b_cast_to_f32(
                b, rocke_b_global_load(b, key_in, odd_idx, in_dtype, /*align=*/2));
            /* trig_idx = b.add(b.mul(slot, const(hd//2)), pair_lane) */
            trig_idx = rocke_b_add(
                b, rocke_b_mul(b, slot, rocke_b_const_i32(b, spec->head_dim / 2)), pair_lane);
            /* c = b.global_load(cos, trig_idx, F32, align=4) */
            c = rocke_b_global_load(b, cos, trig_idx, f32, /*align=*/4);
            /* s = b.global_load(sin, trig_idx, F32, align=4) */
            s = rocke_b_global_load(b, sin, trig_idx, f32, /*align=*/4);
            /* rot_even = b.fsub(b.fmul(even_f, c), b.fmul(odd_f, s))
             * rot_odd  = b.fadd(b.fmul(even_f, s), b.fmul(odd_f, c))
             * Both have two side-effecting fmul args; sequence L-to-R. */
            ec = rocke_b_fmul(b, even_f, c);
            os = rocke_b_fmul(b, odd_f, s);
            rot_even = rocke_b_fsub(b, ec, os);
            es = rocke_b_fmul(b, even_f, s);
            oc = rocke_b_fmul(b, odd_f, c);
            rot_odd = rocke_b_fadd(b, es, oc);
            /* is_odd = b.cmp_eq(b.mod(dim, const(2)), const(1)) -- mod first, then const(1) */
            {
                rocke_value_t* mod_dim = rocke_b_mod(b, dim, rocke_b_const_i32(b, 2));
                rocke_value_t* c_one = rocke_b_const_i32(b, 1);
                is_odd = rocke_b_cmp_eq(b, mod_dim, c_one);
            }
            /* key_f = b.select(is_odd, rot_odd, rot_even) */
            key_f = rocke_b_select(b, is_odd, rot_odd, rot_even);
        }
        else
        {
            /* key_f = b.cast_to_f32(b.global_load(key_in, in_base, in_dtype, align=2)) */
            key_f = rocke_b_cast_to_f32(
                b, rocke_b_global_load(b, key_in, in_base, in_dtype, /*align=*/2));
        }

        /* kv_desc = PagedKvDescriptor(block_size, stride_0..3); out_idx = kv_desc.offset(...).
         * Strides: s0 = bs*nkvh*hd, s1 = nkvh*hd, s2 = hd, s3 = 1. offset =
         *   physical_block*s0 + token_in_block*s1 + kv_head*s2 + dim*s3
         * (each b.mul has one side-effecting arg + an inline const; safe to nest). */
        stride_0 = spec->block_size * spec->num_kv_heads * spec->head_dim;
        stride_1 = spec->num_kv_heads * spec->head_dim;
        stride_2 = spec->head_dim;
        stride_3 = 1;
        {
            rocke_value_t* off = rocke_b_mul(b, physical_block, rocke_b_const_i32(b, stride_0));
            off = rocke_b_add(
                b, off, rocke_b_mul(b, token_in_block, rocke_b_const_i32(b, stride_1)));
            off = rocke_b_add(b, off, rocke_b_mul(b, kv_head, rocke_b_const_i32(b, stride_2)));
            off = rocke_b_add(b, off, rocke_b_mul(b, dim, rocke_b_const_i32(b, stride_3)));
            out_idx = off;
        }

        /* k_store = _quantize_for_kv_store(b, key_f, kv_storage_dtype, k_scale)
         * v_store = _quantize_for_kv_store(b, value_f, kv_storage_dtype, v_scale) */
        k_store = rocke_qkv_quantize_for_store(b, key_f, spec->kv_storage_dtype, k_scale);
        v_store = rocke_qkv_quantize_for_store(b, value_f, spec->kv_storage_dtype, v_scale);
        /* store_align = 1 if kv_storage_dtype != "bf16" else 2 */
        store_align = (strcmp(spec->kv_storage_dtype, "bf16") != 0) ? 1 : 2;
        /* b.global_store(k_cache, out_idx, k_store, align=store_align) */
        rocke_b_global_store(b, k_cache, out_idx, k_store, store_align);
        /* b.global_store(v_cache, out_idx, v_store, align=store_align) */
        rocke_b_global_store(b, v_cache, out_idx, v_store, store_align);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_qwen3_kv_append_rope_gfx1250_new(
    rocke_ir_builder_t* b, const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_qwen3_kv_append_rope_gfx1250_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_qwen3_kv_append_rope_gfx1250(b, spec, arch);
    });
}

rocke_status_t rocke_qwen3_kv_append_rope_gfx1250_lower_to_llvm(
    const rocke_qwen3_kv_append_rope_gfx1250_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel;
    rocke_status_t st;

    if(out_ll != NULL)
    {
        *out_ll = NULL;
    }
    if(spec == NULL || out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx1250";
    }
    kernel = rocke_build_qwen3_kv_append_rope_gfx1250_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            size_t n = m ? strlen(m) : 0;
            if(m == NULL)
            {
                m = "build_qwen3_kv_append_rope_gfx1250 failed";
                n = strlen(m);
            }
            if(n >= err_cap)
            {
                n = err_cap - 1;
            }
            memcpy(err, m, n);
            err[n] = '\0';
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
