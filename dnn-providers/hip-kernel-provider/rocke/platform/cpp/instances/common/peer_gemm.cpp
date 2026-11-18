// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* C ABI peer_gemm.py mirror: pointer-selected panels around existing MFMA helpers. */
#include "rocke/instance_peer_gemm.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.mfma_gemm_inner.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/instance_mfma_gemm.h"
#include "rocke/ir_internal.h"

rocke_peer_gemm_spec_t rocke_peer_gemm_spec_default(void)
{
    rocke_peer_gemm_spec_t spec;
    memset(&spec, 0, sizeof(spec));
    spec.sources = 2;
    spec.dtype = "bf16";
    spec.name = "rocke_peer_gemm";
    return spec;
}

static bool peer_gemm_identifier(const char* name)
{
    if(name == NULL || !*name || strlen(name) > 128)
    {
        return false;
    }
    for(size_t i = 0; name[i]; ++i)
    {
        unsigned char c = (unsigned char)name[i];
        bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
        if(!letter && !(i && c >= '0' && c <= '9'))
        {
            return false;
        }
    }
    return true;
}

static rocke_mfma_gemm_spec_t peer_base_spec(
    const rocke_peer_gemm_spec_t* spec, char* name, size_t cap)
{
    rocke_mfma_gemm_spec_t base = rocke_mfma_gemm_spec_default();
    base.M = spec->rows_per_source;
    base.N = spec->cols_per_rank;
    base.K = spec->k;
    base.dtype = spec->dtype;
    base.tile_m = 16;
    base.tile_n = 16;
    base.kpack = false;
    snprintf(name, cap, "%s_s%d", spec->name, spec->sources);
    base.name = name;
    return base;
}

bool rocke_peer_gemm_is_valid_spec(
    const rocke_peer_gemm_spec_t* spec, const char* arch, char* reason, size_t reason_cap)
{
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    const char* why = NULL;
    char message[256];
    if(spec == NULL)
    {
        why = "null spec";
    }
    else if(strcmp(arch, "gfx942") && strcmp(arch, "gfx950"))
    {
        snprintf(message, sizeof(message), "unsupported peer GEMM arch '%s'", arch);
        why = message;
    }
    else if(spec->sources < 1 || spec->sources > 8)
    {
        why = "sources must be an integer in [1, 8]";
    }
    else if(spec->rows_per_source <= 0 || spec->cols_per_rank <= 0 || spec->k <= 0)
    {
        why = "panel dimensions must be positive i32 integers";
    }
    else if(spec->dtype == NULL || (strcmp(spec->dtype, "f16") && strcmp(spec->dtype, "bf16")))
    {
        if(spec->dtype == NULL)
        {
            why = "unsupported dtype None";
        }
        else
        {
            snprintf(message, sizeof(message), "unsupported dtype '%s'", spec->dtype);
            why = message;
        }
    }
    else if(!peer_gemm_identifier(spec->name))
    {
        why = "name must be an ASCII identifier of at most 128 characters";
    }
    else if((int64_t)spec->rows_per_source * spec->k > INT_MAX
        || (int64_t)spec->k * spec->cols_per_rank > INT_MAX
        || (int64_t)spec->rows_per_source * spec->cols_per_rank > INT_MAX)
    {
        why = "each GEMM panel element span must fit i32 indexing";
    }
    if(why != NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, why);
        return false;
    }
    char name[160];
    rocke_mfma_gemm_spec_t base = peer_base_spec(spec, name, sizeof(name));
    return rocke_mfma_gemm_is_valid_spec(&base, arch, reason, reason_cap);
}

rocke_status_t rocke_peer_gemm_kernel_name(
    const rocke_peer_gemm_spec_t* spec, char* out, size_t out_cap)
{
    if(!rocke_peer_gemm_is_valid_spec(spec, NULL, NULL, 0))
    {
        return ROCKE_ERR_VALUE;
    }
    char name[160];
    rocke_mfma_gemm_spec_t base = peer_base_spec(spec, name, sizeof(name));
    return rocke_mfma_gemm_kernel_name(&base, out, out_cap);
}

typedef struct peer_gemm_load_ctx
{
    rocke_value_t* A;
    rocke_value_t* B;
    const rocke_mfma_atom_t* atom;
    const rocke_lane_decode_t* decode;
    rocke_value_t* m_base;
    rocke_value_t* n_base;
    rocke_value_t* atom_k;
    int k;
    int n;
} peer_gemm_load_ctx_t;

static rocke_value_t* peer_load_a(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    peer_gemm_load_ctx_t* context = (peer_gemm_load_ctx_t*)user;
    rocke_value_t* k_base = rocke_b_mul(b, kt, context->atom_k);
    return rocke_load_a_row_major_contiguous(
        b, context->A, context->atom, context->decode, context->m_base, k_base, context->k);
}

static rocke_value_t* peer_load_b(rocke_ir_builder_t* b, rocke_value_t* kt, void* user)
{
    peer_gemm_load_ctx_t* context = (peer_gemm_load_ctx_t*)user;
    rocke_value_t* k_base = rocke_b_mul(b, kt, context->atom_k);
    return rocke_load_b_col_strided_scalars(
        b, context->B, context->atom, context->decode, context->n_base, k_base, context->n);
}

static rocke_kernel_def_t* peer_gemm_build(
    rocke_ir_builder_t* b, const rocke_peer_gemm_spec_t* spec, const char* arch)
{
    char reason[ROCKE_ERR_MSG_CAP];
    if(!rocke_peer_gemm_is_valid_spec(spec, arch, reason, sizeof(reason)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid peer GEMM spec: %s", reason);
        return NULL;
    }
    char base_name[160];
    rocke_mfma_gemm_spec_t base = peer_base_spec(spec, base_name, sizeof(base_name));
    const rocke_mfma_atom_t* atom = rocke_mfma_gemm_atom(&base);
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", 64);
    const rocke_type_t* element = strcmp(spec->dtype, "bf16") == 0 ? rocke_bf16() : rocke_f16();
    const rocke_type_t* panel_pointer = rocke_ptr_type(b, element, "global");
    const rocke_type_t* table_pointer = rocke_ptr_type(b, panel_pointer, "global");
    rocke_param_opts_t options;
    memset(&options, 0, sizeof(options));
    options.noalias = true;
    options.noalias_set = true;
    options.readonly = true;
    options.readonly_set = true;
    options.align = 8;
    options.align_set = true;
    rocke_value_t* a_table = rocke_b_param(b, "A_ptrs", table_pointer, &options);
    options.align = 16;
    rocke_value_t* local_b = rocke_b_param(b, "B", panel_pointer, &options);
    options.align = 8;
    rocke_value_t* c_table = rocke_b_param(b, "C_ptrs", table_pointer, &options);
    rocke_value_t* source = rocke_b_block_id_z(b);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    rocke_value_t* nonnegative = rocke_b_cmp_ge(b, source, zero);
    rocke_value_t* sources = rocke_b_const_i32(b, spec->sources);
    rocke_value_t* below_limit = rocke_b_cmp_lt(b, source, sources);
    rocke_value_t* valid = rocke_b_land(b, nonnegative, below_limit);
    rocke_if_t guard = rocke_b_scf_if(b, valid);
    rocke_b_region_enter(b, guard.then_region);
    rocke_value_t* selected_a = rocke_b_global_load(b, a_table, source, panel_pointer, 8);
    rocke_value_t* selected_c = rocke_b_global_load(b, c_table, source, panel_pointer, 8);
    rocke_value_t* lane = rocke_b_thread_id_x(b);
    rocke_value_t* bid_n = rocke_b_block_id_x(b);
    rocke_value_t* bid_m = rocke_b_block_id_y(b);
    rocke_value_t* atom_m = rocke_b_const_i32(b, atom->m);
    rocke_value_t* m_base = rocke_b_mul(b, bid_m, atom_m);
    rocke_value_t* atom_n = rocke_b_const_i32(b, atom->n);
    rocke_value_t* n_base = rocke_b_mul(b, bid_n, atom_n);
    rocke_lane_decode_t decode = rocke_decode_mfma_lanes(b, atom, lane);
    rocke_value_t* atom_k = rocke_b_const_i32(b, atom->k);
    peer_gemm_load_ctx_t context;
    context.A = selected_a;
    context.B = local_b;
    context.atom = atom;
    context.decode = &decode;
    context.m_base = m_base;
    context.n_base = n_base;
    context.atom_k = atom_k;
    context.k = spec->k;
    context.n = spec->cols_per_rank;
    rocke_value_t* acc = rocke_mfma_k_loop(
        b, spec->k, atom, peer_load_a, peer_load_b, NULL, NULL, NULL, NULL, &context);
    rocke_store_acc_to_global(b,
        selected_c,
        atom,
        &decode,
        m_base,
        n_base,
        acc,
        spec->cols_per_rank,
        spec->dtype,
        false,
        NULL,
        NULL);
    rocke_b_region_leave(b);
    rocke_b_ret(b);
    return rocke_ir_builder_ok(b) ? b->kernel : NULL;
}

rocke_kernel_def_t* rocke_build_peer_gemm_new(
    rocke_ir_builder_t* b, const rocke_peer_gemm_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b,
        [&]() -> rocke_kernel_def_t*
        {
            if(b == NULL)
            {
                return NULL;
            }
            char name[256];
            rocke_status_t status = rocke_peer_gemm_kernel_name(spec, name, sizeof(name));
            if(rocke_ir_builder_init(b, status == ROCKE_OK ? name : "rocke_peer_gemm") != ROCKE_OK)
            {
                return NULL;
            }
            return peer_gemm_build(b, spec, arch);
        });
}

rocke_status_t rocke_peer_gemm_grid(const rocke_peer_gemm_spec_t* spec, int out[3])
{
    if(out == NULL || !rocke_peer_gemm_is_valid_spec(spec, NULL, NULL, 0))
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = spec->cols_per_rank / 16;
    out[1] = spec->rows_per_source / 16;
    out[2] = spec->sources;
    return ROCKE_OK;
}

rocke_status_t rocke_peer_gemm_lower_to_llvm(const rocke_peer_gemm_spec_t* spec,
    const char* arch,
    rocke_llvm_flavor_t flavor,
    char** out_ll,
    char* err,
    size_t err_cap)
{
    if(out_ll == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    *out_ll = NULL;
    rocke_ir_builder_t b;
    memset(&b, 0, sizeof(b));
    rocke_kernel_def_t* kernel = rocke_build_peer_gemm_new(&b, spec, arch);
    rocke_status_t status = kernel == NULL
        ? ROCKE_ERR_VALUE
        : rocke_lower_kernel_to_llvm(kernel, flavor, arch == NULL ? "gfx950" : arch, out_ll);
    if(kernel == NULL)
    {
        rocke_spec_set_reason(err, err_cap, b.err);
    }
    rocke_ir_builder_free(&b);
    return status;
}
