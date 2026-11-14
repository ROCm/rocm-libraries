// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ twin of rocke/instances/common/batched_transpose.py.
 *
 * The build entry mirrors build_batched_transpose2d op-by-op so the emitted IR
 * op stream is byte-identical: a coalesced global -> LDS vector copy through
 * the tensor_view helpers' TileWindow (3D views whose leading batch axis rides
 * block_id_z with the runtime batch strides), a barrier, then column-strided
 * scalar LDS reads packed into one coalesced global store per thread. The
 * TensorView / TileWindow calls go through the ported tensor_view helpers.
 */
#include "rocke/instance_batched_transpose.h"

#include <stdio.h>

#include <string>
#include <vector>

#include "rocke/arch_target.h"
#include "rocke/arena.h"
#include "rocke/error.hpp"
#include "rocke/error_boundary.hpp"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.tensor_view.h"
#include "rocke/ir_internal.h"

/* str(ZeroDivisionError) for an int // 0 or % 0. */
static const char* const BT_ZERO_DIV = "integer division or modulo by zero";

/* Python a // b (floor division) for b != 0. */
static int bt_floordiv(int a, int b)
{
    const int q = a / b;
    return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}

rocke_batched_transpose2d_spec_t rocke_batched_transpose2d_spec_default(void)
{
    rocke_batched_transpose2d_spec_t s;
    s.tile_m = 64;
    s.tile_n = 64;
    s.vec = 8;
    s.dtype = "f16";
    s.lds_pad = 8;
    s.name = "rocke_batched_transpose2d";
    return s;
}

int rocke_batched_transpose2d_block_size(const rocke_batched_transpose2d_spec_t* spec)
{
    if(spec == NULL || spec->vec == 0)
    {
        return 0;
    }
    /* (tile_m * tile_n) // vec */
    return bt_floordiv(spec->tile_m * spec->tile_n, spec->vec);
}

/* BatchedTranspose2DSpec.kernel_name():
 *   kernel_name_join(self.name, self.dtype, f"{TM}x{TN}", f"v{vec}",
 *                    f"p{lds_pad}")
 */
rocke_status_t rocke_batched_transpose2d_kernel_name(const rocke_batched_transpose2d_spec_t* spec,
                                                     char* out,
                                                     size_t out_cap)
{
    char tile_part[32];
    char vec_part[16];
    char pad_part[16];

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    snprintf(tile_part, sizeof(tile_part), "%dx%d", spec->tile_m, spec->tile_n);
    snprintf(vec_part, sizeof(vec_part), "v%d", spec->vec);
    snprintf(pad_part, sizeof(pad_part), "p%d", spec->lds_pad);
    const char* parts[4] = {spec->dtype, tile_part, vec_part, pad_part};
    return rocke_kernel_name_join(spec->name, parts, 4, NULL, NULL, 0, out, out_cap, NULL);
}

/* is_valid_spec(spec, arch). ArchTarget.from_gfx's KeyError is the C lookup
 * returning NULL, and str(e) is rocke_set_unknown_arch_reason. The Python
 * evaluates spec.block_size before validate_io, so vec == 0 raises
 * ZeroDivisionError there; here it is a reject with that error's text. */
bool rocke_batched_transpose2d_is_valid_spec(const rocke_batched_transpose2d_spec_t* spec,
                                             const char* arch,
                                             char* reason,
                                             size_t reason_cap)
{
    rocke_io_spec_rule_t rule;
    const char* why = NULL;
    char buf[160];

    if(spec == NULL)
    {
        rocke_spec_set_reason(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }
    const rocke_arch_target_t* target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        rocke_set_unknown_arch_reason(reason, reason_cap, arch);
        return false;
    }
    if(spec->vec == 0)
    {
        rocke_spec_set_reason(reason, reason_cap, BT_ZERO_DIV);
        return false;
    }

    const int block_size = rocke_batched_transpose2d_block_size(spec);
    rocke_io_spec_rule_init(&rule, spec->dtype, block_size, spec->vec);
    /* validate_io formats its reject reason into a scratch arena; without one
     * it can only return a static fallback that differs from the Python. */
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 4096) != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "out of memory");
        return false;
    }
    const bool io_ok = rocke_validate_io(&arena, &rule, &why) != 0;
    if(!io_ok)
    {
        rocke_spec_set_reason(reason, reason_cap, why != NULL ? why : "invalid io");
    }
    rocke_arena_destroy(&arena);
    if(!io_ok)
    {
        return false;
    }
    if((spec->tile_m != 16 && spec->tile_m != 32 && spec->tile_m != 64)
       || (spec->tile_n != 16 && spec->tile_n != 32 && spec->tile_n != 64))
    {
        rocke_spec_set_reason(reason, reason_cap, "tile_m/tile_n must be in {16, 32, 64}");
        return false;
    }
    if(spec->tile_m != spec->tile_n)
    {
        rocke_spec_set_reason(reason, reason_cap, "non-square tiles not yet supported");
        return false;
    }
    if((spec->tile_m * spec->tile_n) % spec->vec != 0)
    {
        rocke_spec_set_reason(reason, reason_cap, "tile area must be divisible by vec");
        return false;
    }
    const int max_threads = rocke_arch_max_threads_per_block(target);
    if(block_size > max_threads)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d > %d hardware cap on %s",
                 block_size,
                 max_threads,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    /* LDS staging buffer: [tile_m, tile_n + lds_pad] half-words (2 bytes). */
    const long lds_bytes = (long)spec->tile_m * (long)(spec->tile_n + spec->lds_pad) * 2L;
    if(!rocke_arch_fits_lds(target, lds_bytes))
    {
        snprintf(buf,
                 sizeof(buf),
                 "LDS staging %ld > %d cap on %s",
                 lds_bytes,
                 target->lds_capacity_bytes,
                 arch);
        rocke_spec_set_reason(reason, reason_cap, buf);
        return false;
    }
    rocke_spec_set_reason(reason, reason_cap, "ok");
    return true;
}

/* build_batched_transpose2d(spec, arch), after IRBuilder(spec.kernel_name()):
 *
 *   ok, why = is_valid_spec(spec, arch)
 *   if not ok: raise ValueError(f"invalid batched_transpose2d spec: {why}")
 *   io_ty = io_ir_type(spec.dtype)
 *   params X, Y (ptr, noalias, readonly/writeonly, align=16), H, W,
 *       batch_stride_x, batch_stride_y (i32)
 *   x_view / y_view: (1, TM, TN) / (1, TN, TM) global views, batch stride
 *       leading; tiles at (block_id_z, h0, w0) / (block_id_z, w0, h0)
 *   lds_tile: (TM, TN) window over a (TM, TN + lds_pad) LDS view
 *   Phase 1: row1, col1 = divmod(tid, TN // vec); x_tile.load_vec ->
 *       lds_tile.store_vec; sync
 *   Phase 2: col2, row2 = divmod(tid, TM // vec); vec scalar LDS reads packed
 *       by vec_pack; y_tile.store_vec
 *
 * Every IR-emitting call is its own statement so the op order is Python's
 * left-to-right order (C++ leaves argument evaluation order unspecified). */
static rocke_kernel_def_t*
    bt_build(rocke_ir_builder_t* b, const rocke_batched_transpose2d_spec_t* spec, const char* arch)
{
    char why[512];
    if(spec->vec == 0)
    {
        throw ckc::ValueError(BT_ZERO_DIV);
    }
    if(!rocke_batched_transpose2d_is_valid_spec(spec, arch, why, sizeof(why)))
    {
        throw ckc::ValueError(std::string("invalid batched_transpose2d spec: ") + why);
    }

    const rocke_type_t* io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    const int TM = spec->tile_m;
    const int TN = spec->tile_n;
    const int vec = spec->vec;
    const int BS = rocke_batched_transpose2d_block_size(spec);

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    rocke_param_opts_t opts_x{};
    opts_x.noalias = true;
    opts_x.noalias_set = true;
    opts_x.readonly = true;
    opts_x.readonly_set = true;
    opts_x.align = 16;
    opts_x.align_set = true;
    rocke_value_t* X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts_x);
    rocke_param_opts_t opts_y{};
    opts_y.noalias = true;
    opts_y.noalias_set = true;
    opts_y.writeonly = true;
    opts_y.writeonly_set = true;
    opts_y.align = 16;
    opts_y.align_set = true;
    rocke_value_t* Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts_y);
    rocke_value_t* H = rocke_b_param(b, "H", rocke_i32(), NULL);
    rocke_value_t* W = rocke_b_param(b, "W", rocke_i32(), NULL);
    rocke_value_t* batch_stride_x = rocke_b_param(b, "batch_stride_x", rocke_i32(), NULL);
    rocke_value_t* batch_stride_y = rocke_b_param(b, "batch_stride_y", rocke_i32(), NULL);

    rocke_value_t* tid = rocke_b_thread_id_x(b);

    rocke_value_t* c_vec = rocke_b_const_i32(b, vec);
    rocke_value_t* c_TN_chunks = rocke_b_const_i32(b, TN / vec);
    rocke_value_t* c_TM_chunks = rocke_b_const_i32(b, TM / vec);

    rocke_value_t* batch_idx = rocke_b_block_id_z(b);
    rocke_value_t* tile_x = rocke_b_block_id_x(b);
    rocke_value_t* tile_y = rocke_b_block_id_y(b);

    rocke_value_t* c_TM = rocke_b_const_i32(b, TM);
    rocke_value_t* h0 = rocke_b_mul(b, tile_y, c_TM);
    rocke_value_t* c_TN = rocke_b_const_i32(b, TN);
    rocke_value_t* w0 = rocke_b_mul(b, tile_x, c_TN);

    /* Global views and windows: host-side only, no IR. */
    const int x_shape[3] = {1, TM, TN};
    const int y_shape[3] = {1, TN, TM};
    const rocke_stride_t x_strides[3]
        = {rocke_stride_value(batch_stride_x), rocke_stride_value(W), rocke_stride_imm(1)};
    const rocke_stride_t y_strides[3]
        = {rocke_stride_value(batch_stride_y), rocke_stride_value(H), rocke_stride_imm(1)};
    rocke_tensor_view_t x_view;
    rocke_tensor_view_t y_view;
    rocke_tile_window_t x_tile;
    rocke_tile_window_t y_tile;
    rocke_value_t* const x_origin[3] = {batch_idx, h0, w0};
    rocke_value_t* const y_origin[3] = {batch_idx, w0, h0};
    if(rocke_make_global_view(&x_view, X, x_shape, 3, io_ty, x_strides) != ROCKE_OK
       || rocke_make_global_view(&y_view, Y, y_shape, 3, io_ty, y_strides) != ROCKE_OK
       || rocke_tensor_view_tile(&x_tile, &x_view, x_shape, x_origin, 3) != ROCKE_OK
       || rocke_tensor_view_tile(&y_tile, &y_view, y_shape, y_origin, 3) != ROCKE_OK)
    {
        b->status = ROCKE_ERR_VALUE;
        rocke_spec_set_reason(b->err, sizeof(b->err), "batched_transpose2d: bad view");
        return NULL;
    }

    const int lds_shape[2] = {TM, TN + spec->lds_pad};
    rocke_tensor_view_t lds_view;
    if(rocke_make_lds_view(b, &lds_view, io_ty, lds_shape, 2, "lds_xpose", NULL) != ROCKE_OK)
    {
        return NULL;
    }
    const int lds_lengths[2] = {TM, TN};
    rocke_value_t* lds_org0 = rocke_b_const_i32(b, 0);
    rocke_value_t* lds_org1 = rocke_b_const_i32(b, 0);
    rocke_value_t* const lds_origin[2] = {lds_org0, lds_org1};
    rocke_tile_window_t lds_tile;
    if(rocke_tensor_view_tile(&lds_tile, &lds_view, lds_lengths, lds_origin, 2) != ROCKE_OK)
    {
        b->status = ROCKE_ERR_VALUE;
        rocke_spec_set_reason(b->err, sizeof(b->err), "batched_transpose2d: bad lds view");
        return NULL;
    }

    /* Phase 1: coalesced global -> LDS. */
    rocke_value_t* row1 = rocke_b_div(b, tid, c_TN_chunks);
    rocke_value_t* col1_chunk = rocke_b_mod(b, tid, c_TN_chunks);
    rocke_value_t* col1 = rocke_b_mul(b, col1_chunk, c_vec);

    rocke_value_t* zero1 = rocke_b_const_i32(b, 0);
    rocke_value_t* const x_idx[3] = {zero1, row1, col1};
    rocke_value_t* x_vec = rocke_tile_window_load_vec(b, &x_tile, x_idx, 3, vec);
    rocke_value_t* const lds_idx[2] = {row1, col1};
    rocke_tile_window_store_vec(b, &lds_tile, lds_idx, 2, x_vec, vec);
    rocke_b_sync(b);

    /* Phase 2: column-strided LDS reads, coalesced global stores. */
    rocke_value_t* col2 = rocke_b_div(b, tid, c_TM_chunks);
    rocke_value_t* row2_chunk = rocke_b_mod(b, tid, c_TM_chunks);
    rocke_value_t* row2_base = rocke_b_mul(b, row2_chunk, c_vec);

    std::vector<rocke_value_t*> elems;
    for(int i = 0; i < vec; ++i)
    {
        rocke_value_t* c_i = rocke_b_const_i32(b, i);
        rocke_value_t* row_i = rocke_b_add(b, row2_base, c_i);
        rocke_value_t* const idx[2] = {row_i, col2};
        elems.push_back(rocke_tile_window_load_scalar(b, &lds_tile, idx, 2));
    }
    rocke_value_t* out_vec = rocke_b_vec_pack(b, elems.data(), vec, io_ty);
    rocke_value_t* zero2 = rocke_b_const_i32(b, 0);
    rocke_value_t* const y_idx[3] = {zero2, col2, row2_base};
    rocke_tile_window_store_vec(b, &y_tile, y_idx, 3, out_vec, vec);

    if(b->status != ROCKE_OK)
    {
        return NULL;
    }
    return b->kernel;
}

rocke_kernel_def_t* rocke_build_batched_transpose2d_new(
    rocke_ir_builder_t* b, const rocke_batched_transpose2d_spec_t* spec, const char* arch)
{
    char name[256];
    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    if(rocke_batched_transpose2d_kernel_name(spec, name, sizeof(name)) != ROCKE_OK
       || rocke_ir_builder_init(b, name) != ROCKE_OK)
    {
        return NULL;
    }
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* { return bt_build(b, spec, arch); });
}
