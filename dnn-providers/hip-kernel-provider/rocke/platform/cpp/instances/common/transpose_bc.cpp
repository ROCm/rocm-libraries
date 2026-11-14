// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ twin of rocke/instances/common/transpose_bc.py: the no-LDS 2D transpose
 * in which each lane transposes one vec x vec sub-tile in its own registers.
 *
 * The build entry mirrors the Python op-by-op so the emitted IR op stream is
 * byte-identical:
 *
 *   1. magic-division unmerge of the flat tid -> (lr, lc) lane coords;
 *   2. vec row loads of vec contiguous elements each, the row offset stepped
 *      with the TensorCoordinate move idiom (flat global or buffer-resource IO);
 *   3. the in-register transpose (vec_extract + vec_pack);
 *   4. vec row stores of the transposed sub-tile, stepped the same way.
 *
 * The buffer and global views and their load_vec_at / store_vec_at are the
 * shared tensor_view twins. The TensorCoordinate moves have no C twin yet, so
 * their op streams are reproduced inline, in Python argument order.
 */

#include "rocke/instance_transpose_bc.h"

#include <stdio.h>
#include <string.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir_internal.h"

/* Python a // b (floor division) for b != 0. */
static int tbc_floordiv(int a, int b)
{
    const int q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

rocke_transpose_bc_spec_t rocke_transpose_bc_spec_default(void)
{
    rocke_transpose_bc_spec_t s;
    s.tile_m = 64;
    s.tile_n = 64;
    s.vec = 8;
    s.dtype = "f16";
    s.use_buffer_io = false;
    s.name = "rocke_transpose_bc";
    return s;
}

int rocke_transpose_bc_block_size(const rocke_transpose_bc_spec_t* spec)
{
    if(spec == nullptr || spec->vec == 0)
    {
        return 0;
    }
    /* lanes_per_col * lanes_per_row == (tile_m // vec) * (tile_n // vec) */
    return tbc_floordiv(spec->tile_m, spec->vec) * tbc_floordiv(spec->tile_n, spec->vec);
}

/* TransposeBcSpec.kernel_name():
 *   kernel_name_join(self.name, self.dtype, f"{TM}x{TN}", f"v{vec}", "noLDS",
 *                    "buf" if self.use_buffer_io else "")
 * rocke_kernel_name_join skips the empty part, exactly like the Python. */
rocke_status_t
    rocke_transpose_bc_kernel_name(const rocke_transpose_bc_spec_t* spec, char* out, size_t out_cap)
{
    char tile[32];
    char vec[16];

    if(spec == nullptr || out == nullptr)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(tile, sizeof(tile), "%dx%d", spec->tile_m, spec->tile_n);
    snprintf(vec, sizeof(vec), "v%d", spec->vec);
    const char* parts[5] = {spec->dtype, tile, vec, "noLDS", spec->use_buffer_io ? "buf" : ""};
    return rocke_kernel_name_join(spec->name, parts, 5, nullptr, nullptr, 0, out, out_cap, nullptr);
}

/* is_valid_spec(spec, arch). ArchTarget.from_gfx's KeyError is the C lookup
 * returning nullptr, and str(e) is rocke_set_unknown_arch_reason. vec is checked
 * against {2, 4, 8} before any division, and the lane counts are only formed
 * once both tiles are multiples of vec, so C division matches the Python //. */
bool rocke_transpose_bc_is_valid_spec(const rocke_transpose_bc_spec_t* spec,
                                      const char* arch,
                                      char* reason,
                                      size_t reason_cap)
{
    char buf[256];

    if(spec == nullptr)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == nullptr)
    {
        arch = "gfx950";
    }
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == nullptr)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }

    if(spec->dtype == nullptr
       || (strcmp(spec->dtype, "f16") != 0 && strcmp(spec->dtype, "bf16") != 0))
    {
        if(spec->dtype == nullptr)
        {
            snprintf(buf, sizeof(buf), "unsupported dtype None");
        }
        else
        {
            snprintf(buf, sizeof(buf), "unsupported dtype '%s'", spec->dtype);
        }
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->vec != 2 && spec->vec != 4 && spec->vec != 8)
    {
        snprintf(buf, sizeof(buf), "vec must be in {2, 4, 8} (got %d)", spec->vec);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(spec->tile_m % spec->vec != 0 || spec->tile_n % spec->vec != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_m and tile_n must be multiples of vec");
        return false;
    }
    const int lanes_m = spec->tile_m / spec->vec;
    const int lanes_n = spec->tile_n / spec->vec;
    if(lanes_m != lanes_n)
    {
        snprintf(buf,
                 sizeof(buf),
                 "per-lane sub-tile must be square: tile_m/vec (%d) != tile_n/vec (%d)",
                 lanes_m,
                 lanes_n);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    const int bs = lanes_m * lanes_n;
    const int max_threads = rocke_arch_max_threads_per_block(target);
    if(bs > max_threads)
    {
        snprintf(buf, sizeof(buf), "block_size %d > %d hardware cap on %s", bs, max_threads, arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    const int wave = target->wave_size;
    if(bs < wave)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d < %d; use a larger tile (the kernel is designed for "
                 "one-wave-per-CTA scheduling)",
                 bs,
                 wave);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    if(bs % wave != 0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d must be a multiple of wave%d (got tile_m=%d * tile_n=%d "
                 "/ vec^2 = %d)",
                 bs,
                 wave,
                 spec->tile_m,
                 spec->tile_n,
                 bs);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* build_transpose_bc(spec, arch):
 *   ok, why = is_valid_spec(spec, arch)
 *   if not ok: raise ValueError(f"invalid transpose_bc spec: {why}")
 *   io_ty = io_ir_type(spec.dtype); b = IRBuilder(spec.kernel_name())
 *   b.kernel.attrs["max_workgroup_size"] = BS
 *   X, Y (ptr, noalias, readonly/writeonly, align=16), M, K (i32)
 *   if use_buffer_io: num_bytes = M * K * 2; make_buffer_resource(X / Y)
 *   tid; c_vec; (lr, lc) = unmerge_magic over (LANES_M, LANES_N)
 *   block_n = block_id_x * TN; block_m = block_id_y * TM
 *   my_m = block_m + lr * c_vec; my_n = block_n + lc * c_vec
 *   rows[i]: x_view.load_vec_at(in_coord.offset), coord moved by (1, 0)
 *   out_rows[i] = vec_pack([vec_extract(rows[j], i) for j])
 *   y_view.store_vec_at(out_coord.offset, out_rows[i]), coord moved by (1, 0)
 * The views and their load_vec_at / store_vec_at are the shared tensor_view
 * twins. TensorCoordinate has no C twin, so its op stream is reproduced inline,
 * in Python argument order:
 *   TensorDescriptor.offset((i0, i1)) with strides (S, 1): mul(i0, S), add(., i1)
 *   move_tensor_coordinate(c, (one, zero)): add(i0, one), add(i1, zero),
 *       then offset((one, zero)) and add(c.offset, that) */

/* The lane split, in transpose_bc_lanes.cpp: it needs
 * helper_rocke.helpers.transforms.h, whose struct rocke_tensor_descriptor has
 * the same tag as (but another definition from) the one in
 * helper_rocke.helpers.tensor_view.h, so the two cannot share a translation
 * unit. */
bool rocke_transpose_bc_lane_coords(rocke_ir_builder_t* b,
                                    rocke_value_t* tid,
                                    int lanes_m,
                                    int lanes_n,
                                    rocke_value_t** lr,
                                    rocke_value_t** lc);

/* TensorDescriptor.offset(b, (i0, i1)) for strides (s0, 1). */
static rocke_value_t*
    tbc_offset(rocke_ir_builder_t* b, rocke_value_t* i0, rocke_value_t* i1, rocke_value_t* s0)
{
    rocke_value_t* term = rocke_b_mul(b, i0, s0);
    return rocke_b_add(b, term, i1);
}

/* move_tensor_coordinate(b, coord, (one, zero)) over strides (s0, 1): the moved
 * index and the cached offset. */
static void tbc_move(rocke_ir_builder_t* b,
                     rocke_value_t* index[2],
                     rocke_value_t** off,
                     rocke_value_t* one,
                     rocke_value_t* zero,
                     rocke_value_t* s0)
{
    index[0] = rocke_b_add(b, index[0], one);
    index[1] = rocke_b_add(b, index[1], zero);
    rocke_value_t* delta = tbc_offset(b, one, zero, s0);
    *off = rocke_b_add(b, *off, delta);
}

static rocke_kernel_def_t*
    tbc_build(rocke_ir_builder_t* b, const rocke_transpose_bc_spec_t* spec, const char* arch)
{
    char why[512];
    if(!rocke_transpose_bc_is_valid_spec(spec, arch, why, sizeof(why)))
        throw ckc::ValueError(std::string("invalid transpose_bc spec: ") + why);

    const rocke_type_t* io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == nullptr)
        return nullptr;
    const int TM = spec->tile_m;
    const int TN = spec->tile_n;
    const int VEC = spec->vec;
    const int BS = rocke_transpose_bc_block_size(spec);
    const int LANES_N = TN / VEC; /* lanes_per_row */
    const int LANES_M = TM / VEC; /* lanes_per_col */

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    rocke_param_opts_t opts_x;
    memset(&opts_x, 0, sizeof(opts_x));
    opts_x.noalias = true;
    opts_x.noalias_set = true;
    opts_x.readonly = true;
    opts_x.readonly_set = true;
    opts_x.align = 16;
    opts_x.align_set = true;
    rocke_value_t* X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts_x);
    rocke_param_opts_t opts_y;
    memset(&opts_y, 0, sizeof(opts_y));
    opts_y.noalias = true;
    opts_y.noalias_set = true;
    opts_y.writeonly = true;
    opts_y.writeonly_set = true;
    opts_y.align = 16;
    opts_y.align_set = true;
    rocke_value_t* Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts_y);
    rocke_value_t* M = rocke_b_param(b, "M", rocke_i32(), nullptr);
    rocke_value_t* K = rocke_b_param(b, "K", rocke_i32(), nullptr);

    const bool use_buffer_io = spec->use_buffer_io && strcmp(spec->dtype, "f16") == 0;
    /* The (M, K) / (K, M) views with packed strides (K for X, M for Y): buffer
     * views over one resource per operand, or flat global views. */
    const int view_shape[2] = {1, 1};
    const rocke_stride_t x_strides[2] = {rocke_stride_value(K), rocke_stride_imm(1)};
    const rocke_stride_t y_strides[2] = {rocke_stride_value(M), rocke_stride_imm(1)};
    rocke_tensor_view_t x_view;
    rocke_tensor_view_t y_view;
    if(use_buffer_io)
    {
        rocke_value_t* mk = rocke_b_mul(b, M, K);
        rocke_value_t* num_bytes = rocke_b_mul(b, mk, rocke_b_const_i32(b, 2));
        rocke_tv_buffer_resource_t x_rsrc;
        rocke_tv_buffer_resource_t y_rsrc;
        if(rocke_tv_make_buffer_resource(b, &x_rsrc, X, num_bytes, nullptr) != ROCKE_OK
           || rocke_tv_make_buffer_resource(b, &y_rsrc, Y, num_bytes, nullptr) != ROCKE_OK
           || rocke_tv_make_buffer_view(&x_view, &x_rsrc, view_shape, 2, io_ty, x_strides)
                  != ROCKE_OK
           || rocke_tv_make_buffer_view(&y_view, &y_rsrc, view_shape, 2, io_ty, y_strides)
                  != ROCKE_OK)
            return nullptr;
    }
    else if(rocke_make_global_view(&x_view, X, view_shape, 2, io_ty, x_strides) != ROCKE_OK
            || rocke_make_global_view(&y_view, Y, view_shape, 2, io_ty, y_strides) != ROCKE_OK)
        return nullptr;

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* c_vec = rocke_b_const_i32(b, VEC);

    rocke_value_t* lr = nullptr;
    rocke_value_t* lc = nullptr;
    if(!rocke_transpose_bc_lane_coords(b, tid, LANES_M, LANES_N, &lr, &lc))
        return nullptr;

    rocke_value_t* bx = rocke_b_block_id_x(b);
    rocke_value_t* block_n = rocke_b_mul(b, bx, rocke_b_const_i32(b, TN));
    rocke_value_t* by = rocke_b_block_id_y(b);
    rocke_value_t* block_m = rocke_b_mul(b, by, rocke_b_const_i32(b, TM));

    rocke_value_t* my_m = rocke_b_add(b, block_m, rocke_b_mul(b, lr, c_vec));
    rocke_value_t* my_n = rocke_b_add(b, block_n, rocke_b_mul(b, lc, c_vec));

    /* Phase 1: in_coord = make_tensor_coordinate(b, x_view.desc, (my_m, my_n)). */
    rocke_value_t* in_index[2] = {my_m, my_n};
    rocke_value_t* in_off = tbc_offset(b, my_m, my_n, K);
    rocke_value_t* one = rocke_b_const_i32(b, 1);
    rocke_value_t* zero = rocke_b_const_i32(b, 0);
    std::vector<rocke_value_t*> rows;
    for(int i = 0; i < VEC; ++i)
    {
        if(i > 0)
            tbc_move(b, in_index, &in_off, one, zero, K);
        rows.push_back(rocke_tensor_view_load_vec_at(b, &x_view, in_off, VEC));
    }

    /* Phase 2: out_rows[i] = (rows[0][i], ..., rows[VEC-1][i]). */
    std::vector<rocke_value_t*> out_rows;
    for(int i = 0; i < VEC; ++i)
    {
        std::vector<rocke_value_t*> elems;
        for(int j = 0; j < VEC; ++j)
            elems.push_back(rocke_b_vec_extract(b, rows[j], i));
        out_rows.push_back(rocke_b_vec_pack(b, elems.data(), VEC, io_ty));
    }

    /* Phase 3: out_coord = make_tensor_coordinate(b, y_view.desc, (my_n, my_m)). */
    rocke_value_t* out_index[2] = {my_n, my_m};
    rocke_value_t* out_off = tbc_offset(b, my_n, my_m, M);
    for(int i = 0; i < VEC; ++i)
    {
        if(i > 0)
            tbc_move(b, out_index, &out_off, one, zero, M);
        rocke_tensor_view_store_vec_at(b, &y_view, out_off, out_rows[i], VEC);
    }

    if(b->status != ROCKE_OK)
        return nullptr;
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_transpose_bc_new(rocke_ir_builder_t* b,
                                                 const rocke_transpose_bc_spec_t* spec,
                                                 const char* arch)
{
    if(b == nullptr || spec == nullptr)
        return nullptr;
    /* IRBuilder(spec.kernel_name()); on a name that does not fit, a placeholder
     * builder carries the error so the caller can read it and free b. */
    char name[256];
    if(rocke_transpose_bc_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
    {
        if(rocke_ir_builder_init(b, "rocke_invalid_spec") == ROCKE_OK)
            rocke_i_set_err_msg(b, ROCKE_ERR_VALUE, "transpose_bc kernel name too long");
        return nullptr;
    }
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return nullptr;
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return tbc_build(b, spec, arch); });
}
