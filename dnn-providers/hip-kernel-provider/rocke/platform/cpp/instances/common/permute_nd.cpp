// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_permute_nd.c -- C99 port of rocke/instances/common/permute_nd.py.
 *
 * Byte-identical builder-call sequence vs the Python build_permute:
 *
 *   1. is_valid_spec gate (arch max_threads_per_block, dtype, block_size, rank,
 *      perm validity, positive shapes).
 *   2. io_ir_type(dtype) -> F16/BF16.
 *   3. _build_offset_descriptor(spec): a transforms.TensorDescriptor chain
 *        naive("X", lengths=x_shape, coord_names=[i_0..i_{n-1}])
 *          .transform(unmerge("out_idx", into=[o_0..o_{n-1}], dims=y_shape),
 *                     pass_through(o_d -> i_{perm[d]}) for d in 0..n-1)
 *   4. IRBuilder(name); kernel.attrs["max_workgroup_size"] = block_size.
 *   5. b.param("X"/"Y", PtrType(io_ty,"global"), noalias/readonly|writeonly/align16).
 *   6. make_global_view over X (x_shape) and Y (y_shape) -- construction emits NO
 *      IR; for the GLOBAL address space the load/store methods reduce to the same
 *      builder primitives invoked inline below (global_load_f16/bf16,
 *      global_store, global_load_vN, global_store_vN). The view objects are
 *      therefore not materialised; we drive the identical builder calls directly.
 *   7. tid = thread_id_x(); bid = block_id_x();
 *      thread_out_base = add(mul(bid, const_i32(block_size)), tid).
 *   8. vec > 1 path:
 *        out_idx_base = mul(thread_out_base, const_i32(vec));
 *        in_bounds    = cmp_lt(out_idx_base, const_i32(total));
 *        scf_if(in_bounds) { src_off = in_desc.offset(b, out_idx=out_idx_base);
 *                            x = global_load_vN(X, src_off, io_ty, vec);
 *                            global_store_vN(Y, out_idx_base, x, vec); }
 *      vec == 1 path:
 *        out_idx   = thread_out_base;
 *        in_bounds = cmp_lt(out_idx, const_i32(total));
 *        scf_if(in_bounds) { src_off = in_desc.offset(b, out_idx=out_idx);
 *                            v = global_load_f16/bf16(X, src_off);
 *                            global_store(Y, out_idx, v); }
 *
 * NOTE on the two rocke_tensor_descriptor_t types: transforms.h and
 * tensor_view.h both define a struct/function named rocke_tensor_descriptor_t /
 * rocke_tensor_descriptor_offset with DIFFERENT shapes; they cannot be included
 * in the same TU. This kernel needs the RICH descriptor (transforms.h) for the
 * out_idx->offset algebra, and only the GLOBAL-space load/store behaviour of
 * make_global_view -- which is exactly the builder primitives above. So we
 * include transforms.h and inline the global view path, leaving tensor_view.h
 * out. The emitted IR is identical to make_global_view + load_vec_at/
 * store_vec_at/load_scalar_at/store_scalar_at on a global view.
 */

#include "rocke/instance_permute_nd.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */

/* ------------------------------------------------------------------ helpers */

rocke_permute_spec_t rocke_permute_spec_default(void)
{
    rocke_permute_spec_t s;
    memset(&s, 0, sizeof(s));
    s.rank = 0;
    s.dtype = "f16";
    s.block_size = 256;
    s.name = "rocke_permute";
    return s;
}

int rocke_permute_rank(const rocke_permute_spec_t* spec)
{
    return spec ? spec->rank : 0;
}

int rocke_permute_y_shape(const rocke_permute_spec_t* spec, int out[ROCKE_PERMUTE_MAX_RANK])
{
    int d;
    if(spec == NULL || out == NULL)
        return 0;
    /* y_shape[d] = x_shape[perm[d]] */
    for(d = 0; d < spec->rank; ++d)
    {
        out[d] = spec->x_shape[spec->perm[d]];
    }
    return spec->rank;
}

int rocke_permute_total_elements(const rocke_permute_spec_t* spec)
{
    int d;
    int total = 1;
    if(spec == NULL)
        return 0;
    /* functools.reduce(operator.mul, x_shape, 1) */
    for(d = 0; d < spec->rank; ++d)
    {
        total *= spec->x_shape[d];
    }
    return total;
}

int rocke_permute_vec_width(const rocke_permute_spec_t* spec)
{
    int inner;
    int total;
    int i;
    static const int kVecs[3] = {8, 4, 2};
    if(spec == NULL || spec->rank == 0)
        return 1;
    /* perm[-1] == rank - 1 */
    if(spec->perm[spec->rank - 1] == spec->rank - 1)
    {
        inner = spec->x_shape[spec->rank - 1];
        total = rocke_permute_total_elements(spec);
        for(i = 0; i < 3; ++i)
        {
            int v = kVecs[i];
            if(inner % v == 0 && total % v == 0)
                return v;
        }
        return 1;
    }
    /* perm[-1] != rank - 1: scalar fallback. */
    return 1;
}

rocke_status_t
    rocke_permute_kernel_name(const rocke_permute_spec_t* spec, char* out, size_t out_cap)
{
    /* Python kernel_name():
     *   shape_str = "x".join(str(d) for d in x_shape)
     *   perm_str  = "".join(str(d) for d in perm)
     *   kernel_name_join(name, f"s{shape}", f"p{perm}", dtype, f"b{block_size}",
     *                    f"v{vec}" if vec>1 else "") */
    char shape_part[64];
    char perm_part[64];
    char bs_part[24];
    char vec_part[16];
    int vec;
    int d;
    size_t off;
    const char* parts[5];
    size_t n_parts;

    if(spec == NULL || out == NULL || out_cap == 0)
        return ROCKE_ERR_VALUE;

    /* s<shape> with 'x' separators. */
    off = 0;
    if((size_t)snprintf(shape_part, sizeof(shape_part), "s") >= sizeof(shape_part))
        return ROCKE_ERR_VALUE;
    off = 1;
    for(d = 0; d < spec->rank; ++d)
    {
        int w = snprintf(shape_part + off,
                         sizeof(shape_part) - off,
                         "%s%d",
                         d == 0 ? "" : "x",
                         spec->x_shape[d]);
        if(w < 0 || (size_t)w >= sizeof(shape_part) - off)
            return ROCKE_ERR_VALUE;
        off += (size_t)w;
    }

    /* p<perm> with no separators. */
    if((size_t)snprintf(perm_part, sizeof(perm_part), "p") >= sizeof(perm_part))
        return ROCKE_ERR_VALUE;
    off = 1;
    for(d = 0; d < spec->rank; ++d)
    {
        int w = snprintf(perm_part + off, sizeof(perm_part) - off, "%d", spec->perm[d]);
        if(w < 0 || (size_t)w >= sizeof(perm_part) - off)
            return ROCKE_ERR_VALUE;
        off += (size_t)w;
    }

    if((size_t)snprintf(bs_part, sizeof(bs_part), "b%d", spec->block_size) >= sizeof(bs_part))
        return ROCKE_ERR_VALUE;

    vec = rocke_permute_vec_width(spec);
    parts[0] = shape_part;
    parts[1] = perm_part;
    parts[2] = spec->dtype;
    parts[3] = bs_part;
    n_parts = 4;
    if(vec > 1)
    {
        if((size_t)snprintf(vec_part, sizeof(vec_part), "v%d", vec) >= sizeof(vec_part))
            return ROCKE_ERR_VALUE;
        parts[4] = vec_part;
        n_parts = 5;
    }
    /* The Python passes an empty-string vec part when vec<=1; kernel_name_join
     * skips empty parts, so omitting it (n_parts=4) is equivalent. */
    return rocke_kernel_name_join(spec->name, parts, n_parts, NULL, NULL, 0, out, out_cap, NULL);
}

/* ------------------------------------------------------------- is_valid_spec */

static void permute_reason(char* reason, size_t reason_cap, const char* msg)
{
    rocke_spec_set_reason(reason, reason_cap, msg);
}

bool rocke_permute_is_valid_spec(const rocke_permute_spec_t* spec,
                                 const char* arch,
                                 char* reason,
                                 size_t reason_cap)
{
    const rocke_archtarget_t* target;
    int max_tpb;
    int legal_bs[5];
    int n_legal;
    int i;
    bool bs_ok;
    char buf[160];
    int seen[ROCKE_PERMUTE_MAX_RANK];
    int d;

    if(arch == NULL)
        arch = "gfx950";
    if(spec == NULL)
    {
        permute_reason(reason, reason_cap, "null spec");
        return false;
    }

    /* ArchTarget.from_gfx(arch); KeyError -> (False, str(e)). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        snprintf(buf, sizeof(buf), "%s", arch);
        permute_reason(reason, reason_cap, buf);
        return false;
    }
    max_tpb = rocke_archtarget_max_threads_per_block(target);

    /* dtype in ("f16", "bf16") */
    if(!(spec->dtype != NULL
         && (strcmp(spec->dtype, "f16") == 0 || strcmp(spec->dtype, "bf16") == 0)))
    {
        snprintf(buf, sizeof(buf), "unsupported dtype %s", spec->dtype ? spec->dtype : "(null)");
        permute_reason(reason, reason_cap, buf);
        return false;
    }

    /* legal block sizes = {64,128,256,512,1024} clamped to max_threads_per_block. */
    {
        static const int kBs[5] = {64, 128, 256, 512, 1024};
        n_legal = 0;
        for(i = 0; i < 5; ++i)
        {
            if(kBs[i] <= max_tpb)
                legal_bs[n_legal++] = kBs[i];
        }
    }
    bs_ok = false;
    for(i = 0; i < n_legal; ++i)
    {
        if(spec->block_size == legal_bs[i])
        {
            bs_ok = true;
            break;
        }
    }
    if(!bs_ok)
    {
        snprintf(buf, sizeof(buf), "block_size %d not legal on %s", spec->block_size, arch);
        permute_reason(reason, reason_cap, buf);
        return false;
    }

    /* rank >= 1 */
    if(spec->rank == 0)
    {
        permute_reason(reason, reason_cap, "rank must be >= 1");
        return false;
    }
    /* rank <= MAX_RANK */
    if(spec->rank > ROCKE_PERMUTE_MAX_RANK)
    {
        snprintf(
            buf, sizeof(buf), "rank %d > %d (CK Tile cap)", spec->rank, ROCKE_PERMUTE_MAX_RANK);
        permute_reason(reason, reason_cap, buf);
        return false;
    }
    /* len(perm) != rank is structurally enforced by the C struct (rank pins
     * both arrays); the Python check is preserved as a no-op here. */

    /* sorted(perm) == range(rank) */
    for(d = 0; d < spec->rank; ++d)
        seen[d] = 0;
    for(d = 0; d < spec->rank; ++d)
    {
        int p = spec->perm[d];
        if(p < 0 || p >= spec->rank || seen[p])
        {
            snprintf(buf, sizeof(buf), "perm is not a permutation of range(%d)", spec->rank);
            permute_reason(reason, reason_cap, buf);
            return false;
        }
        seen[p] = 1;
    }

    /* x_shape all positive */
    for(d = 0; d < spec->rank; ++d)
    {
        if(spec->x_shape[d] <= 0)
        {
            permute_reason(reason, reason_cap, "x_shape must be all positive");
            return false;
        }
    }

    /* total_elements > 0 */
    if(rocke_permute_total_elements(spec) <= 0)
    {
        permute_reason(reason, reason_cap, "total_elements must be positive");
        return false;
    }

    permute_reason(reason, reason_cap, "ok");
    return true;
}

/* ------------------------------------------------- _build_offset_descriptor */

/* Build the rich descriptor mapping out_idx -> input flat offset:
 *   naive("X", lengths=x_shape, coord_names=[i_0..i_{n-1}])
 *     .transform(unmerge("out_idx", into=[o_0..o_{n-1}], dims=y_shape),
 *                pass_through(o_d -> i_{perm[d]}) for d in 0..n-1)
 *
 * All name strings / arrays are arena-owned (b->arena), matching the Python
 * frozen-dataclass lifetime. Returns NULL with b's sticky error on failure. */
static const rocke_tensor_descriptor_t*
    permute_build_offset_descriptor(rocke_ir_builder_t* b, const rocke_permute_spec_t* spec)
{
    int n = spec->rank;
    int d;
    int y_shape[ROCKE_PERMUTE_MAX_RANK];
    const char** in_names;
    const char** out_names;
    int* x_lengths;
    int* y_dims;
    const rocke_tensor_descriptor_t* naive;
    const rocke_transform_t** chain;
    int n_chain;
    char nm[16];

    if(b == NULL || spec == NULL)
        return NULL;

    rocke_permute_y_shape(spec, y_shape);

    in_names = (const char**)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(const char*));
    out_names = (const char**)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(const char*));
    x_lengths = (int*)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(int));
    y_dims = (int*)rocke_arena_alloc(&b->arena, (size_t)n * sizeof(int));
    if(in_names == NULL || out_names == NULL || x_lengths == NULL || y_dims == NULL)
        return NULL;

    for(d = 0; d < n; ++d)
    {
        snprintf(nm, sizeof(nm), "i_%d", d);
        in_names[d] = rocke_arena_strdup(&b->arena, nm);
        snprintf(nm, sizeof(nm), "o_%d", d);
        out_names[d] = rocke_arena_strdup(&b->arena, nm);
        if(in_names[d] == NULL || out_names[d] == NULL)
            return NULL;
        x_lengths[d] = spec->x_shape[d];
        y_dims[d] = y_shape[d];
    }

    /* naive base over the INPUT shape (row-major strides). */
    naive = rocke_tensor_descriptor_naive(b, "X", x_lengths, n, NULL, in_names, n);
    if(naive == NULL)
        return NULL;

    /* chain = [ unmerge, pass_through x n ] */
    n_chain = 1 + n;
    chain = (const rocke_transform_t**)rocke_arena_alloc(
        &b->arena, (size_t)n_chain * sizeof(const rocke_transform_t*));
    if(chain == NULL)
        return NULL;

    chain[0] = rocke_unmerge(b, "out_idx", out_names, n, y_dims);
    if(chain[0] == NULL)
        return NULL;
    for(d = 0; d < n; ++d)
    {
        /* pass_through(upper=o_d, lower=i_{perm[d]}) */
        chain[1 + d] = rocke_pass_through(b, out_names[d], in_names[spec->perm[d]]);
        if(chain[1 + d] == NULL)
            return NULL;
    }

    return rocke_tensor_descriptor_transform(b, naive, chain, n_chain);
}

/* ----------------------------------------------------------------- build_permute */

rocke_kernel_def_t*
    rocke_build_permute(rocke_ir_builder_t* b, const rocke_permute_spec_t* spec, const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_type_t* io_ty;
        const rocke_type_t* ptr_ty;
        int total;
        int vec;
        const rocke_tensor_descriptor_t* in_desc;
        rocke_value_t* X;
        rocke_value_t* Y;
        rocke_param_opts_t xopts;
        rocke_param_opts_t yopts;
        rocke_value_t* tid;
        rocke_value_t* bid;
        rocke_value_t* thread_out_base;
        char reason[160];

        if(b == NULL || spec == NULL)
            return NULL;
        if(arch == NULL)
            arch = "gfx950";

        /* is_valid_spec gate -> ValueError on reject. */
        if(!rocke_permute_is_valid_spec(spec, arch, reason, sizeof(reason)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid permute spec: %s", reason);
            return NULL;
        }

        io_ty = rocke_b_io_ir_type(b, spec->dtype);
        if(io_ty == NULL)
            return NULL;
        total = rocke_permute_total_elements(spec);
        vec = rocke_permute_vec_width(spec);

        /* in_desc = _build_offset_descriptor(spec) */
        in_desc = permute_build_offset_descriptor(b, spec);
        if(in_desc == NULL)
            return NULL;

        /* kernel.attrs["max_workgroup_size"] = block_size. (The builder was already
         * init'd with the spec's kernel_name() by the caller.) */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

        /* PtrType(io_ty, "global"). */
        ptr_ty = rocke_ptr_type(b, io_ty, "global");

        /* X: noalias=True, readonly=True, align=16. */
        memset(&xopts, 0, sizeof(xopts));
        xopts.noalias = true;
        xopts.noalias_set = true;
        xopts.readonly = true;
        xopts.readonly_set = true;
        xopts.align = 16;
        xopts.align_set = true;
        xopts.addr_space = NULL;
        X = rocke_b_param(b, "X", ptr_ty, &xopts);

        /* Y: noalias=True, writeonly=True, align=16. */
        memset(&yopts, 0, sizeof(yopts));
        yopts.noalias = true;
        yopts.noalias_set = true;
        yopts.writeonly = true;
        yopts.writeonly_set = true;
        yopts.align = 16;
        yopts.align_set = true;
        yopts.addr_space = NULL;
        Y = rocke_b_param(b, "Y", ptr_ty, &yopts);

        /* make_global_view over X/Y emits NO IR; the global-space load/store methods
         * are inlined below as the identical builder primitives. */

        tid = rocke_b_thread_id_x(b);
        bid = rocke_b_block_id_x(b);
        /* thread_out_base = add(mul(bid, const_i32(block_size)), tid). */
        thread_out_base
            = rocke_b_add(b, rocke_b_mul(b, bid, rocke_b_const_i32(b, spec->block_size)), tid);

        if(vec > 1)
        {
            /* Vectorised path. */
            rocke_value_t* out_idx_base
                = rocke_b_mul(b, thread_out_base, rocke_b_const_i32(b, vec));
            rocke_value_t* c_total = rocke_b_const_i32(b, total);
            rocke_value_t* in_bounds = rocke_b_cmp_lt(b, out_idx_base, c_total);
            rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
            rocke_b_region_enter(b, iff.then_region);
            {
                const char* in_names[1];
                rocke_value_t* in_values[1];
                rocke_value_t* src_offset = NULL;
                rocke_value_t* _valid = NULL;
                rocke_value_t* x_vec;
                in_names[0] = "out_idx";
                in_values[0] = out_idx_base;
                /* src_offset, _valid = in_desc.offset(b, out_idx=out_idx_base) */
                if(!rocke_transforms_descriptor_offset(
                       b, in_desc, in_names, in_values, 1, &src_offset, &_valid))
                {
                    rocke_b_region_leave(b);
                    return NULL;
                }
                (void)_valid;
                /* X_view.load_vec_at(b, src_offset, n=vec) over global == global_load_vN. */
                x_vec = rocke_b_global_load_vN(b, X, src_offset, io_ty, vec, 0);
                /* Y_view.store_vec_at(b, out_idx_base, x_vec, n=vec) == global_store_vN. */
                rocke_b_global_store_vN(b, Y, out_idx_base, x_vec, vec, 0);
            }
            rocke_b_region_leave(b);
        }
        else
        {
            /* Scalar fallback path. */
            rocke_value_t* out_idx = thread_out_base;
            rocke_value_t* c_total = rocke_b_const_i32(b, total);
            rocke_value_t* in_bounds = rocke_b_cmp_lt(b, out_idx, c_total);
            rocke_if_t iff = rocke_b_scf_if(b, in_bounds);
            rocke_b_region_enter(b, iff.then_region);
            {
                const char* in_names[1];
                rocke_value_t* in_values[1];
                rocke_value_t* src_offset = NULL;
                rocke_value_t* _valid = NULL;
                rocke_value_t* val;
                in_names[0] = "out_idx";
                in_values[0] = out_idx;
                /* src_offset, _valid = in_desc.offset(b, out_idx=out_idx) */
                if(!rocke_transforms_descriptor_offset(
                       b, in_desc, in_names, in_values, 1, &src_offset, &_valid))
                {
                    rocke_b_region_leave(b);
                    return NULL;
                }
                (void)_valid;
                /* X_view.load_scalar_at(b, src_offset) over global:
                 *   f16  -> global_load_f16; bf16 -> global_load_bf16. */
                if(strcmp(spec->dtype, "f16") == 0)
                    val = rocke_b_global_load_f16(b, X, src_offset, 0);
                else
                    val = rocke_b_global_load_bf16(b, X, src_offset, 0);
                /* Y_view.store_scalar_at(b, out_idx, val) over global == global_store. */
                rocke_b_global_store(b, Y, out_idx, val, 0);
            }
            rocke_b_region_leave(b);
        }

        if(rocke_ir_builder_status(b) != ROCKE_OK)
            return NULL;
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_permute_new(rocke_ir_builder_t* b,
                                            const rocke_permute_spec_t* spec,
                                            const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
            return NULL;
        if(rocke_permute_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
            return NULL;
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
            return NULL;
        return rocke_build_permute(b, spec, arch);
    });
}

/* ----------------------------------------------------------------- permute_grid */

rocke_status_t rocke_permute_grid(const rocke_permute_spec_t* spec, int out[3])
{
    int vec;
    int total;
    int threads;
    int totals[1];
    int tiles[1];
    if(spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    vec = rocke_permute_vec_width(spec);
    total = rocke_permute_total_elements(spec);
    /* threads = (total + vec - 1) // vec */
    threads = (total + vec - 1) / vec;
    /* ceil_div_grid((threads, block_size)). */
    totals[0] = threads;
    tiles[0] = spec->block_size;
    return rocke_ceil_div_grid(totals, tiles, 1, out);
}

/* ------------------------------------------------------------ permute_signature */

rocke_status_t rocke_permute_signature(rocke_arena_t* arena,
                                       const rocke_permute_spec_t* spec,
                                       rocke_sig_entry_t* out,
                                       size_t out_cap,
                                       size_t* out_count)
{
    /* SignatureBuilder().ptr("X", dtype).ptr("Y", dtype).build() */
    rocke_signature_builder_t sb;
    const rocke_sig_entry_t* items;
    size_t count;
    rocke_status_t st;
    size_t i;

    if(arena == NULL || spec == NULL || out == NULL)
        return ROCKE_ERR_VALUE;
    st = rocke_signature_builder_init(&sb, arena);
    if(st != ROCKE_OK)
        return st;
    rocke_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Y", spec->dtype, NULL);
    st = rocke_signature_builder_build(&sb, &items, &count);
    if(st != ROCKE_OK)
        return st;
    if(count > out_cap)
        return ROCKE_ERR_VALUE;
    for(i = 0; i < count; ++i)
        out[i] = items[i];
    if(out_count != NULL)
        *out_count = count;
    return ROCKE_OK;
}

/* ------------------------------------------------------ lower-to-.ll convenience */

rocke_status_t rocke_permute_lower_to_llvm(const rocke_permute_spec_t* spec,
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
        *out_ll = NULL;
    if(spec == NULL || out_ll == NULL)
    {
        permute_reason(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
        arch = "gfx950";

    kernel = rocke_build_permute_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        permute_reason(err, err_cap, m ? m : "build_permute failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
