// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Host-only port of library/kernels/gfx950/attention_dense.py. The shared
 * validator/decode registry remains in instances/common/attention_dense_spec.cpp.
 * No builder, launcher, or provider dependencies belong in this layer. */
#include "../common/attention_dense_spec_internal.h"
#include "rocke/arch_target.h"
#include "rocke/float_format.h"
#include "rocke/instance_gfx950_attention_dense.h"
#include "rocke/intexpr.h"
#include "rocke/sha256.h"

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <initializer_list>
#include <string>

namespace
{
bool streq(const char* a, const char* b)
{
    return a != NULL && b != NULL && strcmp(a, b) == 0;
}

bool fp8_kv(const rocke_gfx950_attention_dense_spec_t& s)
{
    return streq(s.kv_storage_dtype, "fp8e4m3");
}

void clear_reason(char* reason, size_t cap)
{
    if(reason != NULL && cap > 0)
        reason[0] = '\0';
}

/* Only the gfx950-owned part of __post_init__, for shared preflight's callback.
 * The C field types already establish emit_lse's bool and spec's concrete type. */
bool validate_arch(const void* ctx, char* reason, size_t cap)
{
    const auto* s = static_cast<const rocke_gfx950_attention_dense_spec_t*>(ctx);
    if(s == NULL)
        ROCKE_SPEC_FAIL(reason, cap, "attention dense spec is NULL");
    const auto& b = s->base;
    if(s->kv_storage_dtype != NULL && !fp8_kv(*s))
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "kv_storage_dtype must be None or one of ['fp8e4m3'], got %s",
                        s->kv_storage_dtype);
    if(s->lds_v_row_pad < 0 || s->lds_v_row_pad % 8 != 0)
        ROCKE_SPEC_FAIL(
            reason,
            cap,
            "lds_v_row_pad must be a non-negative multiple of 8 bf16 elements (16 bytes), got %d",
            s->lds_v_row_pad);

    /* _validate_codegen_knobs: ordering matters for compound-invalid specs. */
    if(s->lds_k_row_pad < 0 || s->lds_k_row_pad % 8 != 0)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "lds_k_row_pad must be a non-negative multiple of 8 elements "
                        "(16 bytes, keeps ds_read_b128 aligned), got %d",
                        s->lds_k_row_pad);
    if(s->lds_num_buffers != 2)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "lds_num_buffers: only 2 is implemented (the prologue primes two tiles; 3 "
                        "spills), got %d",
                        s->lds_num_buffers);
    if(!(s->lazy_rescale_threshold > 0.0 && s->lazy_rescale_threshold <= 8.0))
    {
        if(reason != NULL && cap > 0)
        {
            rocke_strbuf_t value{};
            rocke_append_python_float_repr(&value, s->lazy_rescale_threshold);
            if(value.oom)
                rocke_spec_set_reason(
                    reason, cap, "out of memory formatting lazy_rescale_threshold");
            else
                rocke_spec_set_reasonf(reason,
                                       cap,
                                       "lazy_rescale_threshold must be in (0, 8] log2 units "
                                       "(8 bounds P at 256), got %s",
                                       rocke_strbuf_cstr(&value));
            rocke_strbuf_free(&value);
        }
        return false;
    }
    if(s->has_exp_per_pv_step && s->exp_per_pv_step < 1)
        ROCKE_SPEC_FAIL(
            reason, cap, "exp_per_pv_step must be >= 1 or None, got %d", s->exp_per_pv_step);
    if(s->pv_priority < 0 || s->pv_priority > 3)
        ROCKE_SPEC_FAIL(reason, cap, "pv_priority must be in 0..3, got %d", s->pv_priority);
    if(s->pv_sched_fence_mask < 0 || s->pv_sched_fence_mask > 0x7ff)
        ROCKE_SPEC_FAIL(
            reason, cap, "pv_sched_fence_mask must be in 0..0x7ff, got %d", s->pv_sched_fence_mask);
    if(s->pv_sched_group_ds_read < 1)
        ROCKE_SPEC_FAIL(
            reason, cap, "pv_sched_group_ds_read must be >= 1, got %d", s->pv_sched_group_ds_read);
    if(s->has_iglp_mode && s->iglp_mode != -1 && s->iglp_mode != 0 && s->iglp_mode != 1)
        ROCKE_SPEC_FAIL(
            reason, cap, "iglp_mode must be None, -1 (off), 0 or 1, got %d", s->iglp_mode);
    if(s->pv_loop_order != NULL && !streq(s->pv_loop_order, "d_major")
       && !streq(s->pv_loop_order, "k_major"))
    {
        /* Error-only storage, sized for the caller just as shared preflight is.
         * Long invalid strings must not be silently cut before the reason writer. */
        const size_t n = rocke_py_str_repr(s->pv_loop_order, NULL, 0);
        char small[128];
        if(n < sizeof(small))
        {
            rocke_py_str_repr(s->pv_loop_order, small, sizeof(small));
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "pv_loop_order must be one of ['d_major', 'k_major'] or None, got %s",
                            small);
        }
        // The raw string may be arbitrarily long; the failure path alone allocates.
        std::string value(n, '\0');
        rocke_py_str_repr(s->pv_loop_order, &value[0], n + 1);
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "pv_loop_order must be one of ['d_major', 'k_major'] or None, got %s",
                        value.c_str());
    }
    if(s->o_store_width != 1 && s->o_store_width != 2 && s->o_store_width != 4)
        ROCKE_SPEC_FAIL(reason, cap, "o_store_width must be 1, 2 or 4, got %d", s->o_store_width);
    if(s->o_store_width != 4 && streq(b.dtype, "fp16"))
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "o_store_width=%d is bf16-only: narrower fp16 stores are not bit-identical "
                        "to the width-4 output",
                        s->o_store_width);
    if(rocke_gfx950_attention_dense_resolved_iglp_mode(s) != -1
       && (rocke_gfx950_attention_dense_resolved_pv_sched_fence(s)
           || rocke_gfx950_attention_dense_resolved_pv_sched_group_template(s)))
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "iglp_mode owns loop scheduling; disable pv_sched_fence and "
                        "pv_sched_group_template when enabling it");

    if(s->right_bound < 0)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "right_bound must be >= 0 (causal=False is the unbounded side), got %d",
                        s->right_bound);
    if(s->right_bound > 0)
    {
        if(!b.causal)
            ROCKE_SPEC_FAIL(reason, cap, "right_bound > 0 requires causal=True");
        if(b.causal_bottom_right)
            ROCKE_SPEC_FAIL(
                reason, cap, "right_bound > 0 is not supported with causal_bottom_right");
        if(b.paged || b.varlen)
            ROCKE_SPEC_FAIL(reason, cap, "right_bound > 0 is not supported with paged / varlen");
    }
    if(s->kv_storage_dtype != NULL)
    {
        if(b.persistent)
            ROCKE_SPEC_FAIL(reason, cap, "fp8 KV is not yet supported with persistent=True");
        if(b.paged)
            ROCKE_SPEC_FAIL(reason, cap, "fp8 KV is not yet supported with paged KV");
        if(b.ragged)
            ROCKE_SPEC_FAIL(reason, cap, "fp8 KV is not yet supported with ragged=True");
        if(b.varlen)
            ROCKE_SPEC_FAIL(reason, cap, "fp8 KV is not yet supported with varlen=True");
    }
    if(b.causal_bottom_right)
    {
        if(b.persistent)
            ROCKE_SPEC_FAIL(
                reason, cap, "causal_bottom_right is not supported with persistent=True");
        if(b.sliding_window > 0)
            ROCKE_SPEC_FAIL(
                reason, cap, "causal_bottom_right is not supported with sliding_window>0");
        if(b.varlen)
            ROCKE_SPEC_FAIL(reason, cap, "causal_bottom_right is not supported with varlen=True");
    }
    if(s->wide_lds_dma)
    {
        if(!b.persistent)
            ROCKE_SPEC_FAIL(reason, cap, "wide_lds_dma requires persistent=True");
        if(b.head_size != 128 || b.block_n != 64)
            ROCKE_SPEC_FAIL(reason, cap, "wide_lds_dma requires head_size=128 and block_n=64");
        if(b.ragged || b.varlen || b.paged)
            ROCKE_SPEC_FAIL(
                reason, cap, "wide_lds_dma is validated only for aligned contiguous K/V");
        if(b.lds_k_group_pad != 8 || s->lds_k_row_pad != 8 || s->lds_v_row_pad != 32)
            ROCKE_SPEC_FAIL(reason, cap, "wide_lds_dma requires K/V slab padding of 8/32 elements");
    }
    clear_reason(reason, cap);
    return true;
}

/* The largest typed, valid hash input is bounded: 14 fixed knob names, int32
 * decimals, one double, and the two legal short loop-order strings. */
struct KnobRepr
{
    char text[1024];
    size_t used = 0;
    bool add(const char* name, const char* value)
    {
        const int n = snprintf(
            text + used, sizeof(text) - used, "%s('%s', %s)", used == 0 ? "[" : ", ", name, value);
        if(n < 0 || (size_t)n >= sizeof(text) - used)
            return false;
        used += (size_t)n;
        return true;
    }
};

rocke_status_t codegen_token(const rocke_gfx950_attention_dense_spec_t* s, char out[11])
{
    out[0] = '\0';
    using Spec = rocke_gfx950_attention_dense_spec_t;
    static const Spec defaults = rocke_gfx950_attention_dense_spec_default();
    KnobRepr repr;
    bool ok = true;
    auto add_int = [&](const char* name, int value) {
        char number[32];
        snprintf(number, sizeof(number), "%d", value);
        ok = ok && repr.add(name, number);
    };
    auto add_bool = [&](const char* name, bool value) {
        ok = ok && repr.add(name, value ? "True" : "False");
    };
    auto add_optional = [&](const char* name, bool Spec::*present, auto field, auto emit) {
        if(s->*present)
        {
            if(!(defaults.*present) || s->*field != defaults.*field)
                emit(name, s->*field);
        }
        else if(defaults.*present)
            ok = ok && repr.add(name, "None");
    };
    // sorted(changed), not dataclass declaration order, and raw optional values.
    if(s->causal_diag_split != defaults.causal_diag_split)
        add_bool("causal_diag_split", s->causal_diag_split);
    add_optional("exp_per_pv_step", &Spec::has_exp_per_pv_step, &Spec::exp_per_pv_step, add_int);
    add_optional("iglp_mode", &Spec::has_iglp_mode, &Spec::iglp_mode, add_int);
    if(s->lazy_rescale_threshold != defaults.lazy_rescale_threshold)
    {
        rocke_strbuf_t number{};
        rocke_append_python_float_repr(&number, s->lazy_rescale_threshold);
        const bool oom = number.oom != 0;
        if(!oom)
            ok = ok && repr.add("lazy_rescale_threshold", rocke_strbuf_cstr(&number));
        rocke_strbuf_free(&number);
        if(oom)
            return ROCKE_ERR_OOM;
    }
    if(s->lds_num_buffers != defaults.lds_num_buffers)
        add_int("lds_num_buffers", s->lds_num_buffers);
    if(s->o_store_width != defaults.o_store_width)
        add_int("o_store_width", s->o_store_width);
    if(s->partial_vmcnt_prefetch != defaults.partial_vmcnt_prefetch)
        add_bool("partial_vmcnt_prefetch", s->partial_vmcnt_prefetch);
    if(s->pv_loop_order != defaults.pv_loop_order
       && !streq(s->pv_loop_order, defaults.pv_loop_order))
    {
        char value[32];
        if(rocke_py_str_repr(s->pv_loop_order, value, sizeof(value)) >= sizeof(value))
            return ROCKE_ERR_VALUE;
        ok = ok && repr.add("pv_loop_order", value);
    }
    if(s->pv_priority != defaults.pv_priority)
        add_int("pv_priority", s->pv_priority);
    add_optional("pv_sched_fence", &Spec::has_pv_sched_fence, &Spec::pv_sched_fence, add_bool);
    if(s->pv_sched_fence_mask != defaults.pv_sched_fence_mask)
        add_int("pv_sched_fence_mask", s->pv_sched_fence_mask);
    if(s->pv_sched_group_ds_read != defaults.pv_sched_group_ds_read)
        add_int("pv_sched_group_ds_read", s->pv_sched_group_ds_read);
    add_optional("pv_sched_group_template",
                 &Spec::has_pv_sched_group_template,
                 &Spec::pv_sched_group_template,
                 add_bool);
    if(s->use_exp2_fast != defaults.use_exp2_fast)
        add_bool("use_exp2_fast", s->use_exp2_fast);
    if(!ok || repr.used + 1 >= sizeof(repr.text))
        return ROCKE_ERR_VALUE;
    if(repr.used == 0)
        return ROCKE_OK;
    repr.text[repr.used++] = ']';
    uint8_t digest[32];
    rocke_sha256(repr.text, repr.used, digest);
    static const char hex[] = "0123456789abcdef";
    out[0] = 'c';
    out[1] = 'g';
    for(int i = 0; i < 4; ++i)
    {
        out[2 + i * 2] = hex[digest[i] >> 4];
        out[3 + i * 2] = hex[digest[i] & 15];
    }
    out[10] = '\0';
    return ROCKE_OK;
}

bool append_part(char parts[][ROCKE_ATTENTION_DENSE_NAME_PART_CAP],
                 int& count,
                 const char* prefix,
                 int value)
{
    if(count >= ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS)
        return false;
    const int n
        = snprintf(parts[count], ROCKE_ATTENTION_DENSE_NAME_PART_CAP, "%s%d", prefix, value);
    if(n < 0 || n >= ROCKE_ATTENTION_DENSE_NAME_PART_CAP)
        return false;
    ++count;
    return true;
}

bool append_literal(char parts[][ROCKE_ATTENTION_DENSE_NAME_PART_CAP],
                    int& count,
                    const char* value)
{
    if(count >= ROCKE_ATTENTION_DENSE_MAX_NAME_PARTS
       || strlen(value) >= ROCKE_ATTENTION_DENSE_NAME_PART_CAP)
        return false;
    strcpy(parts[count++], value);
    return true;
}

} // namespace

rocke_gfx950_attention_dense_spec_t rocke_gfx950_attention_dense_spec_default(void)
{
    rocke_gfx950_attention_dense_spec_t s = {};
    s.base = rocke_attention_dense_spec_default();
    s.lds_v_row_pad = 32;
    s.kv_segments = 1;
    s.lds_num_buffers = 2;
    s.lds_k_row_pad = 8;
    s.lazy_rescale_threshold = 8.0;
    s.use_exp2_fast = true;
    s.partial_vmcnt_prefetch = true;
    s.pv_priority = 1;
    s.pv_sched_group_ds_read = 2;
    s.causal_diag_split = true;
    s.o_store_width = 4;
    return s;
}

rocke_attention_dense_traits_t rocke_gfx950_attention_dense_traits(void)
{
    auto t = rocke_attention_dense_traits_base();
    t.type_name = "Gfx950AttentionDenseSpec";
    t.supports_noncausal_window = true;
    t.supports_noncausal_cross_length_ragged = true;
    return t;
}

bool rocke_gfx950_attention_dense_spec_validate(const rocke_gfx950_attention_dense_spec_t* s,
                                                char* reason,
                                                size_t cap)
{
    if(s == NULL)
        ROCKE_SPEC_FAIL(reason, cap, "attention dense spec is NULL");
    const auto t = rocke_gfx950_attention_dense_traits();
    const int kv_elem_bytes = rocke_gfx950_attention_dense_kv_elem_bytes(s);
    return rocke_dense_detail::validate(
               &s->base, &t, s->right_bound != 0, kv_elem_bytes, reason, cap)
           && validate_arch(s, reason, cap);
}

int rocke_gfx950_attention_dense_kv_elem_bytes(const rocke_gfx950_attention_dense_spec_t* s)
{
    if(s == NULL)
        return 0;
    return fp8_kv(*s) ? 1 : rocke_dense_detail::base_kv_elem_bytes;
}

int rocke_gfx950_attention_dense_resolved_exp_per_pv_step(
    const rocke_gfx950_attention_dense_spec_t* s)
{
    if(s == NULL)
        return 0;
    if(s->has_exp_per_pv_step)
        return s->exp_per_pv_step;
    if(s->base.persistent)
        return 1;
    const int64_t steps = (s->base.head_size / 32) * (int64_t)(s->base.block_n / 16);
    return steps > 0 ? (int)rocke_ceil_div((s->base.block_n / 32) * 16LL, steps) : 0;
}

bool rocke_gfx950_attention_dense_resolved_pv_sched_fence(
    const rocke_gfx950_attention_dense_spec_t* s)
{
    return s != NULL && (s->has_pv_sched_fence ? s->pv_sched_fence : !s->wide_lds_dma);
}

bool rocke_gfx950_attention_dense_resolved_pv_sched_group_template(
    const rocke_gfx950_attention_dense_spec_t* s)
{
    return s != NULL
           && (s->has_pv_sched_group_template ? s->pv_sched_group_template : !s->wide_lds_dma);
}

int rocke_gfx950_attention_dense_resolved_iglp_mode(const rocke_gfx950_attention_dense_spec_t* s)
{
    return s == NULL ? -1 : s->has_iglp_mode ? s->iglp_mode : s->wide_lds_dma ? 1 : -1;
}

const char* rocke_gfx950_attention_dense_resolved_pv_loop_order(
    const rocke_gfx950_attention_dense_spec_t* s)
{
    return s == NULL                  ? NULL
           : s->pv_loop_order != NULL ? s->pv_loop_order
           : s->wide_lds_dma          ? "k_major"
                                      : "d_major";
}

bool rocke_gfx950_attention_dense_runtime_shape(const rocke_gfx950_attention_dense_spec_t* s)
{
    if(s == NULL)
        return false;
    if(s->base.persistent)
    {
        const rocke_attention_dense_divisor_t* const* fields;
        size_t count;
        if(rocke_attention_dense_persist_divisors(&s->base, s->right_bound != 0, &fields, &count)
               != ROCKE_OK
           || count == 0)
            return false;
    }
    return !(s->base.ragged || s->base.varlen || s->base.paged
             || (s->base.causal_bottom_right && s->base.seqlen_q != s->base.seqlen_kv));
}

rocke_status_t
    rocke_gfx950_attention_dense_runtime_param_fields(const rocke_gfx950_attention_dense_spec_t* s,
                                                      rocke_gfx950_dense_fields_t* out)
{
    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    *out = {};
    if(rocke_gfx950_attention_dense_runtime_shape(s))
    {
        const char* fields[]
            = {"batch", "seqlen_q", "seqlen_kv", "num_query_heads", "num_kv_heads"};
        for(const char* field : fields)
            out->names[out->count++] = field;
    }
    return ROCKE_OK;
}

rocke_status_t
    rocke_gfx950_attention_dense_shape_kernarg_fields(const rocke_gfx950_attention_dense_spec_t* s,
                                                      rocke_gfx950_dense_fields_t* out)
{
    if(rocke_gfx950_attention_dense_runtime_param_fields(s, out) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    if(out->count == 0)
    {
        if(!s->base.persistent)
        {
            out->names[out->count++] = "batch";
            out->names[out->count++] = "seqlen_q";
            out->names[out->count++] = "seqlen_kv";
        }
    }
    else if(s->base.persistent)
    {
        const rocke_attention_dense_divisor_t* const* fields;
        size_t count;
        if(rocke_attention_dense_persist_divisors(&s->base, s->right_bound != 0, &fields, &count)
           != ROCKE_OK)
            return ROCKE_ERR_VALUE;
        for(size_t i = 0; i < count; ++i)
        {
            out->names[out->count++] = fields[i]->magic_name;
            out->names[out->count++] = fields[i]->shift_name;
        }
    }
    return ROCKE_OK;
}

rocke_status_t rocke_gfx950_attention_dense_name_parts(const rocke_gfx950_attention_dense_spec_t* s,
                                                       rocke_attention_dense_name_parts_t* out)
{
    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    if(rocke_attention_dense_name_parts_base(&s->base, s->right_bound != 0, out) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    out->head_parts = out->shape_parts = !rocke_gfx950_attention_dense_runtime_shape(s);
    if(s->base.head_size == 128)
    {
        if(s->lds_k_row_pad != 8
           && !append_part(out->layout, out->num_layout, "krpad", s->lds_k_row_pad))
            return ROCKE_ERR_VALUE;
        if(s->lds_v_row_pad != 32
           && !append_part(out->layout, out->num_layout, "vpad", s->lds_v_row_pad))
            return ROCKE_ERR_VALUE;
    }
    if(s->right_bound > 0 && !append_part(out->band, out->num_band, "rb", s->right_bound))
        return ROCKE_ERR_VALUE;
    if(s->wide_lds_dma && !append_literal(out->algorithm, out->num_algorithm, "wdma"))
        return ROCKE_ERR_VALUE;
    if(s->base.waves_per_eu != 2
       && !append_part(out->algorithm, out->num_algorithm, "wpe", s->base.waves_per_eu))
        return ROCKE_ERR_VALUE;
    char token[11];
    if(const auto status = codegen_token(s, token); status != ROCKE_OK)
        return status;
    if(token[0] != '\0' && !append_literal(out->algorithm, out->num_algorithm, token))
        return ROCKE_ERR_VALUE;
    if(s->emit_lse && !append_literal(out->algorithm, out->num_algorithm, "lse"))
        return ROCKE_ERR_VALUE;
    if(s->kv_segments != 1
       && !append_part(out->algorithm, out->num_algorithm, "kvs", s->kv_segments))
        return ROCKE_ERR_VALUE;
    return ROCKE_OK;
}

rocke_status_t rocke_gfx950_attention_dense_kernel_name(
    const rocke_gfx950_attention_dense_spec_t* s, char* out, size_t cap)
{
    rocke_attention_dense_name_parts_t parts;
    if(const auto status = rocke_gfx950_attention_dense_name_parts(s, &parts); status != ROCKE_OK)
        return status;
    return rocke_dense_detail::kernel_name(
        &s->base, &parts, fp8_kv(*s) ? "kvfp8e4m3" : NULL, out, cap);
}

rocke_status_t rocke_gfx950_dense_static_lds_bytes(int head_size,
                                                   int block_n,
                                                   int k_row_pad,
                                                   int v_row_pad,
                                                   int k_group_pad,
                                                   bool wide,
                                                   int64_t* out)
{
    if(out == NULL)
        return ROCKE_ERR_VALUE;
    if(head_size != 64 && head_size != 128)
    {
        *out = 0;
        return ROCKE_OK;
    }
    if(wide)
    {
        const int64_t rows = 4LL * (head_size / 64) * rocke_floor_div(block_n, 8);
        return rocke_intexpr_checked_mul(rows, 1024LL + k_row_pad + v_row_pad, out)
                   ? ROCKE_OK
                   : ROCKE_ERR_VALUE;
    }
    if(head_size == 128)
        return rocke_intexpr_checked_mul(4LL * block_n, 256LL + k_row_pad + v_row_pad, out)
                   ? ROCKE_OK
                   : ROCKE_ERR_VALUE;
    int64_t k, v;
    if(!rocke_intexpr_checked_mul(4LL * rocke_floor_div(block_n, 2), 128LL + k_group_pad, &k)
       || !rocke_intexpr_checked_mul(4LL * block_n, head_size, &v)
       || !rocke_intexpr_checked_add(k, v, out))
        return ROCKE_ERR_VALUE;
    return ROCKE_OK;
}

size_t rocke_gfx950_dense_tile_legality(int head_size, int out[ROCKE_GFX950_DENSE_MAX_TILES][2])
{
    if(head_size != 64 && head_size != 128)
        return 0;
    const int capacity = rocke_arch_target_from_gfx("gfx950")->lds_capacity_bytes;
    size_t count = 0;
    for(int bm : {128, 256})
        for(int bn = 32; bn <= bm; bn += 32)
        {
            if(bm % bn != 0 || bn % (bm / 32) != 0)
                continue;
            int64_t bytes;
            if(rocke_gfx950_dense_static_lds_bytes(head_size, bn, 8, 32, 8, false, &bytes)
                   != ROCKE_OK
               || bytes > capacity)
                continue;
            if(out != NULL)
            {
                out[count][0] = bm;
                out[count][1] = bn;
            }
            ++count;
        }
    return count;
}

bool rocke_gfx950_supports_attention_dense(const rocke_gfx950_attention_dense_spec_t* s,
                                           const char* arch,
                                           char* reason,
                                           size_t cap)
{
    if(arch == NULL)
        arch = "gfx950";
    if(!streq(arch, "gfx950"))
        ROCKE_SPEC_FAIL(reason, cap, "attention_dense is gfx950-only (got %s)", arch);
    if(s == NULL)
        ROCKE_SPEC_FAIL(reason, cap, "attention dense spec is NULL");
    const auto t = rocke_gfx950_attention_dense_traits();
    const auto& b = s->base;
    if(!rocke_dense_detail::preflight(&b,
                                      &t,
                                      s->right_bound != 0,
                                      rocke_gfx950_attention_dense_kv_elem_bytes(s),
                                      validate_arch,
                                      s,
                                      reason,
                                      cap))
        return false;
    bool bm_ok = false;
    for(const auto& geometry : ROCKE_DENSE_TILE_GEOMETRIES)
        bm_ok = bm_ok || b.block_m == geometry.block_m;
    if(!bm_ok)
        ROCKE_SPEC_FAIL(reason, cap, "gfx950 block_m must be one of [128, 256], got %d", b.block_m);
    const int waves = rocke_attention_dense_spec_num_waves(&b);
    if(b.block_n % waves != 0)
        ROCKE_SPEC_FAIL(
            reason,
            cap,
            "block_n=%d must be divisible by num_waves=%d so K/V DMA rows distribute evenly",
            b.block_n,
            waves);
    int64_t bytes;
    if(rocke_gfx950_dense_static_lds_bytes(b.head_size,
                                           b.block_n,
                                           s->lds_k_row_pad,
                                           s->lds_v_row_pad,
                                           b.lds_k_group_pad,
                                           s->wide_lds_dma,
                                           &bytes)
       != ROCKE_OK)
        ROCKE_SPEC_FAIL(reason, cap, "static LDS byte count is outside int64");
    const int capacity = rocke_arch_target_from_gfx("gfx950")->lds_capacity_bytes;
    if(bytes > capacity)
        ROCKE_SPEC_FAIL(reason,
                        cap,
                        "static LDS %lld bytes for head_size=%d block_n=%d exceeds gfx950's %d",
                        (long long)bytes,
                        b.head_size,
                        b.block_n,
                        capacity);
    if(!rocke_attention_dense_check_kv_segments(&b, s->kv_segments, s->emit_lse, reason, cap))
        return false;
    if(s->kv_storage_dtype != NULL)
    {
        /* Preflight established that block_n divides block_m (128 or 256).
         * Each of the wave's 64 threads loads eight flat K/V elements. */
        const int tile_elems = b.block_n * b.head_size;
        const int elems_per_load = waves * 64 * 8;
        if(tile_elems % elems_per_load != 0)
            ROCKE_SPEC_FAIL(reason,
                            cap,
                            "fp8 dense requires block_n*head_size=%d divisible by threads*8=%d "
                            "(block_n=%d, head_size=%d, num_waves=%d)",
                            tile_elems,
                            elems_per_load,
                            b.block_n,
                            b.head_size,
                            waves);
        if(b.use_sinks)
            ROCKE_SPEC_FAIL(reason, cap, "fp8 dense does not support attention sinks yet");
    }
    return true;
}

rocke_status_t rocke_gfx950_attention_dense_grid(const rocke_gfx950_attention_dense_spec_t* s,
                                                 int out[3])
{
    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    if(s->base.persistent)
    {
        out[0] = s->base.num_persistent;
        out[1] = 1;
        out[2] = 1;
        return ROCKE_OK;
    }
    int grid[3];
    if(rocke_attention_dense_nonpersist_grid(&s->base, s->right_bound != 0, grid) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    const int64_t x = (int64_t)grid[0] * s->kv_segments;
    if(x < 1 || x > INT_MAX)
        return ROCKE_ERR_VALUE;
    out[0] = (int)x;
    out[1] = grid[1];
    out[2] = grid[2];
    return ROCKE_OK;
}

rocke_status_t rocke_gfx950_attention_dense_block(const rocke_gfx950_attention_dense_spec_t* s,
                                                  int out[3])
{
    if(s == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    const int64_t threads = rocke_attention_dense_spec_num_waves(&s->base) * 64LL;
    if(threads < 1 || threads > INT_MAX)
        return ROCKE_ERR_VALUE;
    out[0] = (int)threads;
    out[1] = 1;
    out[2] = 1;
    return ROCKE_OK;
}

rocke_status_t rocke_gfx950_attention_dense_signature(rocke_arena_t* arena,
                                                      const rocke_gfx950_attention_dense_spec_t* s,
                                                      const rocke_sig_entry_t** items,
                                                      size_t* count)
{
    if(s == NULL || items == NULL || count == NULL)
        return ROCKE_ERR_VALUE;
    rocke_gfx950_dense_fields_t fields;
    if(rocke_gfx950_attention_dense_shape_kernarg_fields(s, &fields) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    rocke_signature_builder_t sig;
    if(rocke_signature_builder_init(&sig, arena) != ROCKE_OK)
        return ROCKE_ERR_VALUE;
    const bool fp8 = fp8_kv(*s);
    const char* kv_dtype = fp8 ? s->kv_storage_dtype : s->base.dtype;
    rocke_signature_builder_ptr(&sig, "q_ptr", s->base.dtype, "global");
    rocke_signature_builder_ptr(&sig, "k_ptr", kv_dtype, "global");
    rocke_signature_builder_ptr(&sig, "v_ptr", kv_dtype, "global");
    rocke_signature_builder_ptr(&sig, "o_ptr", s->base.dtype, "global");
    rocke_signature_builder_scalar(&sig, "scale", "f32");
    if(fp8)
    {
        rocke_signature_builder_scalar(&sig, "k_scale", "f32");
        rocke_signature_builder_scalar(&sig, "v_scale", "f32");
    }
    for(size_t i = 0; i < fields.count; ++i)
        rocke_signature_builder_scalar(&sig, fields.names[i], "i32");
    if(s->base.use_sinks)
        rocke_signature_builder_ptr(&sig, "sink_ptr", s->base.dtype, "global");
    if(s->base.varlen)
    {
        rocke_signature_builder_ptr(&sig, "cu_seqlens_q", "i32", "global");
        rocke_signature_builder_ptr(&sig, "cu_seqlens_kv", "i32", "global");
    }
    if(s->base.paged)
    {
        rocke_signature_builder_ptr(&sig, "block_tables", "i32", "global");
        rocke_signature_builder_ptr(&sig, "kv_lens", "i32", "global");
        rocke_signature_builder_scalar(&sig, "block_table_stride", "i32");
    }
    if(s->emit_lse)
        rocke_signature_builder_ptr(&sig, "lse_ptr", "f32", "global");
    if(s->kv_segments > 1)
        for(const char* name : ROCKE_DENSE_SEGMENT_PARAMS)
            rocke_signature_builder_ptr(&sig, name, "f32", "global");
    return rocke_signature_builder_build(&sig, items, count);
}

bool rocke_gfx950_attention_dense_runtime_args(const rocke_gfx950_attention_dense_spec_t* s,
                                               rocke_gfx950_dense_runtime_args_t* out,
                                               char* reason,
                                               size_t cap)
{
    if(s == NULL || out == NULL)
        ROCKE_SPEC_FAIL(reason, cap, "attention dense runtime args are NULL");
    *out = {};
    rocke_gfx950_dense_fields_t fields;
    if(rocke_gfx950_attention_dense_runtime_param_fields(s, &fields) != ROCKE_OK)
        ROCKE_SPEC_FAIL(reason, cap, "attention dense runtime fields are invalid");
    if(fields.count == 0 && !s->base.persistent)
        rocke_gfx950_attention_dense_shape_kernarg_fields(s, &fields);
    const int values[] = {s->base.batch,
                          s->base.seqlen_q,
                          s->base.seqlen_kv,
                          s->base.num_query_heads,
                          s->base.num_kv_heads};
    for(size_t i = 0; i < fields.count; ++i)
        out->items[out->count++] = {fields.names[i], values[i]};
    if(s->base.persistent && rocke_gfx950_attention_dense_runtime_shape(s))
    {
        rocke_attention_dense_fastdiv_values_t pairs;
        if(!rocke_attention_dense_fastdiv_values(
               &s->base, s->right_bound != 0, "gfx950", &pairs, reason, cap))
        {
            *out = {};
            return false;
        }
        const rocke_attention_dense_divisor_t* const* divisors;
        size_t n;
        if(rocke_attention_dense_persist_divisors(&s->base, s->right_bound != 0, &divisors, &n)
           != ROCKE_OK)
            ROCKE_SPEC_FAIL(reason, cap, "attention dense persistent decode is unknown");
        for(size_t i = 0; i < n; ++i)
        {
            out->items[out->count++] = {divisors[i]->magic_name, pairs.magic[i]};
            out->items[out->count++] = {divisors[i]->shift_name, pairs.shift[i]};
        }
    }
    clear_reason(reason, cap);
    return true;
}
