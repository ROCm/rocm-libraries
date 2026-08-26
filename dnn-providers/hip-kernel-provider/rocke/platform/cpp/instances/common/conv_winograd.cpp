// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// C++ engine mirror of:
//   platform/python/rocke/instances/common/conv_winograd.py
//   platform/python/rocke/instances/common/_conv_winograd_common.py
//
// Byte-identity contract: every builder here must emit the same LLVM IR as its
// Python counterpart for the same (spec, arch) inputs.
//
// Run `python tools/check_byte_identity.py --only winograd` to verify.

#include "rocke/instance_conv_winograd.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Transform matrices — mirror _conv_winograd_common.py
// ---------------------------------------------------------------------------

// F(2,3): 4×4 transform domain
static const double F2X3_B_T[4][4] = {
    {1, 0, -1, 0},
    {0, 1, 1, 0},
    {0, -1, 1, 0},
    {0, 1, 0, -1},
};
static const double F2X3_G[4][3] = {
    {1.0, 0.0, 0.0},
    {0.5, 0.5, 0.5},
    {0.5, -0.5, 0.5},
    {0.0, 0.0, 1.0},
};
static const double F2X3_A_T[2][4] = {
    {1, 1, 1, 0},
    {0, 1, -1, -1},
};

// F(4,3): 6×6 transform domain
static const double F4X3_B_T[6][6] = {
    {4, 0, -5, 0, 1, 0},
    {0, -4, -4, 1, 1, 0},
    {0, 4, -4, -1, 1, 0},
    {0, -2, -1, 2, 1, 0},
    {0, 2, -1, -2, 1, 0},
    {0, 4, 0, -5, 0, 1},
};
static const double F4X3_G[6][3] = {
    {1.0 / 4.0, 0.0, 0.0},
    {-1.0 / 6.0, -1.0 / 6.0, -1.0 / 6.0},
    {-1.0 / 6.0, 1.0 / 6.0, -1.0 / 6.0},
    {1.0 / 24.0, 1.0 / 12.0, 1.0 / 6.0},
    {1.0 / 24.0, -1.0 / 12.0, 1.0 / 6.0},
    {0.0, 0.0, 1.0},
};
static const double F4X3_A_T[4][6] = {
    {1, 1, 1, 1, 1, 0},
    {0, 1, -1, 2, -2, 0},
    {0, 1, 1, 4, 4, 0},
    {0, 1, -1, 8, -8, 1},
};

// ---------------------------------------------------------------------------
// matvec_f32 — emit SSA ops for one row of a matrix-vector product.
// Mirrors Python _matvec_f32, including the same coefficient fast-paths so the
// emitted IR is byte-identical.
//
// mat[rows][cols] — static double coefficients
// vec[cols]       — SSA values (rocke_value_t*)
// out[rows]       — written SSA results
// c_zero          — a pre-existing f32 zero constant (passed in, not recreated
//                   per row, matching Python which calls b.const_f32(0.0) once
//                   per _matvec_f32 call at the top of the function)
// ---------------------------------------------------------------------------

static rocke_value_t*
    emit_term(rocke_ir_builder_t* b, double coeff, rocke_value_t* v, rocke_value_t* c_zero)
{
    // Mirror Python _matvec_f32 coefficient dispatch exactly.
    if(coeff == 1.0)
        return v;
    if(coeff == -1.0)
        return rocke_b_fsub(b, c_zero, v);
    if(coeff == 2.0)
        return rocke_b_fadd(b, v, v);
    if(coeff == -2.0)
    {
        rocke_value_t* t = rocke_b_fadd(b, v, v);
        return rocke_b_fsub(b, c_zero, t);
    }
    if(coeff == 4.0)
    {
        rocke_value_t* t = rocke_b_fadd(b, v, v);
        return rocke_b_fadd(b, t, t);
    }
    if(coeff == -4.0)
    {
        rocke_value_t* t = rocke_b_fadd(b, v, v);
        return rocke_b_fsub(b, c_zero, rocke_b_fadd(b, t, t));
    }
    if(coeff == 5.0)
    {
        rocke_value_t* t2 = rocke_b_fadd(b, v, v);
        rocke_value_t* t4 = rocke_b_fadd(b, t2, t2);
        return rocke_b_fadd(b, t4, v);
    }
    if(coeff == -5.0)
    {
        rocke_value_t* t2 = rocke_b_fadd(b, v, v);
        rocke_value_t* t4 = rocke_b_fadd(b, t2, t2);
        return rocke_b_fsub(b, c_zero, rocke_b_fadd(b, t4, v));
    }
    if(coeff == 8.0)
    {
        rocke_value_t* t = rocke_b_fadd(b, v, v);
        t = rocke_b_fadd(b, t, t);
        return rocke_b_fadd(b, t, t);
    }
    if(coeff == -8.0)
    {
        rocke_value_t* t = rocke_b_fadd(b, v, v);
        t = rocke_b_fadd(b, t, t);
        return rocke_b_fsub(b, c_zero, rocke_b_fadd(b, t, t));
    }
    // General: emit fmul
    rocke_value_t* c_coeff = rocke_b_const_f32(b, coeff);
    return rocke_b_fmul(b, c_coeff, v);
}

// Compute one row of a (rows x cols) matrix applied to vec[cols].
// Returns NULL if the row is all-zero (caller substitutes c_zero).
static rocke_value_t* matvec_row(rocke_ir_builder_t* b,
                                 const double* row, // length cols
                                 rocke_value_t** vec,
                                 int cols,
                                 rocke_value_t* c_zero)
{
    rocke_value_t* accum = nullptr;
    for(int c = 0; c < cols; ++c)
    {
        double coeff = row[c];
        if(coeff == 0.0)
            continue;
        rocke_value_t* term = emit_term(b, coeff, vec[c], c_zero);
        accum = (accum == nullptr) ? term : rocke_b_fadd(b, accum, term);
    }
    return accum; // nullptr => row was all-zero
}

// Emit matvec_f32(b, mat[rows][cols], vec[cols]) → out[rows].
// mat_flat is row-major: mat_flat[r*cols + c].
// Mirrors the Python _matvec_f32 function exactly, including the single
// const_f32(0.0) emission at entry.
static void matvec_f32(rocke_ir_builder_t* b,
                       const double* mat_flat,
                       int rows,
                       int cols,
                       rocke_value_t** vec, // length cols
                       rocke_value_t** out) // length rows, written
{
    rocke_value_t* c_zero = rocke_b_const_f32(b, 0.0);
    for(int r = 0; r < rows; ++r)
    {
        rocke_value_t* res = matvec_row(b, mat_flat + r * cols, vec, cols, c_zero);
        out[r] = (res != nullptr) ? res : c_zero;
    }
}

// ---------------------------------------------------------------------------
// apply_transform_2d — emit Y = M * X * M^T for a square tile.
//
// mat[out_rows][in_rows] applied symmetrically:
//   step 1 (col transform): for each column of tile[in_rows][xs],
//     apply mat → mid[out_rows][xs]
//   step 2 (row transform): for each row of mid[out_rows][xs],
//     apply mat → out[out_rows][out_cols]
//
// The Python _apply_transform_2d does exactly this, using flat nested lists.
// We replicate it here with C VLAs (or fixed-size intermediates).
//
// Supports up to 8×8 tiles (F(6,3) is the largest: xs=8).  Callers for
// F(2,3) and F(4,3) use xs=4 and xs=6 respectively.
// ---------------------------------------------------------------------------

#define MAX_XS 8

static void apply_transform_2d(rocke_ir_builder_t* b,
                               const double* mat_flat, // [out_rows][in_rows]
                               int out_rows,
                               int in_rows,
                               int xs, // tile column count
                               rocke_value_t* tile[MAX_XS][MAX_XS], // [in_rows][xs]
                               rocke_value_t* out[MAX_XS][MAX_XS]) // [out_rows][out_rows]
{
    // Step 1: mid[out_rows][xs]  — apply mat to each column of tile
    rocke_value_t* mid[MAX_XS][MAX_XS] = {};
    for(int col = 0; col < xs; ++col)
    {
        rocke_value_t* col_vec[MAX_XS];
        for(int r = 0; r < in_rows; ++r)
            col_vec[r] = tile[r][col];
        rocke_value_t* col_out[MAX_XS];
        matvec_f32(b, mat_flat, out_rows, in_rows, col_vec, col_out);
        for(int r = 0; r < out_rows; ++r)
            mid[r][col] = col_out[r];
    }

    // Step 2: out[out_rows][out_rows] — apply mat to each row of mid
    for(int row = 0; row < out_rows; ++row)
    {
        rocke_value_t* row_vec[MAX_XS];
        for(int c = 0; c < xs; ++c)
            row_vec[c] = mid[row][c];
        rocke_value_t* row_out[MAX_XS];
        matvec_f32(b, mat_flat, out_rows, xs, row_vec, row_out);
        for(int c = 0; c < out_rows; ++c)
            out[row][c] = row_out[c];
    }
}

// emit_data_transform — B^T * patch * B
// tile[xs][xs] → xformed[xs][xs]
static void emit_data_transform(rocke_ir_builder_t* b,
                                int out_tile,
                                int xs,
                                rocke_value_t* tile[MAX_XS][MAX_XS],
                                rocke_value_t* xformed[MAX_XS][MAX_XS])
{
    const double* B_T_flat;
    if(out_tile == 2)
        B_T_flat = &F2X3_B_T[0][0];
    else
        B_T_flat = &F4X3_B_T[0][0];
    // B^T is (xs x xs); both row and col transforms use the same xs×xs matrix.
    apply_transform_2d(b, B_T_flat, xs, xs, xs, tile, xformed);
}

// emit_filter_transform — G * filter * G^T
// filter_patch[3][3] → xformed[xs][xs]
// Mirrors Python emit_filter_transform.
static void emit_filter_transform(rocke_ir_builder_t* b,
                                  int out_tile,
                                  int xs,
                                  int fs, // filter_size = 3
                                  rocke_value_t* filter_patch[MAX_XS][MAX_XS],
                                  rocke_value_t* xformed[MAX_XS][MAX_XS])
{
    const double* G_flat;
    if(out_tile == 2)
        G_flat = &F2X3_G[0][0];
    else
        G_flat = &F4X3_G[0][0];

    // Step 1: mid[xs][fs] — apply G[xs][fs] to each column of filter_patch[fs][fs]
    rocke_value_t* mid[MAX_XS][MAX_XS] = {};
    for(int col = 0; col < fs; ++col)
    {
        rocke_value_t* col_vec[MAX_XS];
        for(int r = 0; r < fs; ++r)
            col_vec[r] = filter_patch[r][col];
        rocke_value_t* col_out[MAX_XS];
        matvec_f32(b, G_flat, xs, fs, col_vec, col_out);
        for(int r = 0; r < xs; ++r)
            mid[r][col] = col_out[r];
    }

    // Step 2: out[xs][xs] — apply G[xs][fs] to each row of mid[xs][fs]
    // (equivalent to mid * G^T)
    for(int row = 0; row < xs; ++row)
    {
        rocke_value_t* row_vec[MAX_XS];
        for(int c = 0; c < fs; ++c)
            row_vec[c] = mid[row][c];
        rocke_value_t* row_out[MAX_XS];
        matvec_f32(b, G_flat, xs, fs, row_vec, row_out);
        for(int c = 0; c < xs; ++c)
            xformed[row][c] = row_out[c];
    }
}

// emit_output_transform — A^T * acc * A
// acc_tile[xs][xs] → out_tile_vals[ot][ot]
// Mirrors Python emit_output_transform.
static void emit_output_transform(rocke_ir_builder_t* b,
                                  int out_tile,
                                  int xs,
                                  rocke_value_t* acc_tile[MAX_XS][MAX_XS],
                                  rocke_value_t* out_vals[MAX_XS][MAX_XS])
{
    const double* A_T_flat;
    int ot = out_tile;
    if(out_tile == 2)
        A_T_flat = &F2X3_A_T[0][0];
    else
        A_T_flat = &F4X3_A_T[0][0];

    // Step 1: mid[ot][xs] — apply A_T[ot][xs] to each column of acc_tile[xs][xs]
    rocke_value_t* mid[MAX_XS][MAX_XS] = {};
    for(int col = 0; col < xs; ++col)
    {
        rocke_value_t* col_vec[MAX_XS];
        for(int r = 0; r < xs; ++r)
            col_vec[r] = acc_tile[r][col];
        rocke_value_t* col_out[MAX_XS];
        matvec_f32(b, A_T_flat, ot, xs, col_vec, col_out);
        for(int r = 0; r < ot; ++r)
            mid[r][col] = col_out[r];
    }

    // Step 2: out[ot][ot] — apply A_T[ot][xs] to each row of mid[ot][xs]
    for(int row = 0; row < ot; ++row)
    {
        rocke_value_t* row_vec[MAX_XS];
        for(int c = 0; c < xs; ++c)
            row_vec[c] = mid[row][c];
        rocke_value_t* row_out[MAX_XS];
        matvec_f32(b, A_T_flat, ot, xs, row_vec, row_out);
        for(int c = 0; c < ot; ++c)
            out_vals[row][c] = row_out[c];
    }
}

// ---------------------------------------------------------------------------
// Kernel name helper
// ---------------------------------------------------------------------------

void rocke_winograd_conv_spec_kernel_name(const rocke_winograd_conv_spec_t* s,
                                          const char* suffix,
                                          char* out,
                                          int cap)
{
    const rocke_winograd_problem_t* p = &s->problem;
    // Mirror Python kernel_name_join(name, suffix, "N{N}H{Hi}W{Wi}C{C}K{K}",
    //   "f{out_tile}x3", "bc{block_c}bk{block_k}bnhw{block_nhw}")
    // kernel_name_join uses "_" as separator and drops empty parts.
    snprintf(out,
             (size_t)cap,
             "%s_%s_N%dH%dW%dC%dK%d_f%dx3_bc%dbk%dbnhw%d",
             s->name,
             suffix,
             p->N,
             p->Hi,
             p->Wi,
             p->C,
             p->K,
             s->out_tile,
             s->block_c,
             s->block_k,
             s->block_nhw);
}

// ---------------------------------------------------------------------------
// OOB sentinel (AMD buffer descriptor returns 0 for OOB byte offsets)
// ---------------------------------------------------------------------------

static const int64_t OOB_SENTINEL = (1LL << 31) - 1;

// ---------------------------------------------------------------------------
// Kernel 1: Data transform — B^T * input_patch * B
// ---------------------------------------------------------------------------

rocke_kernel_def_t* rocke_build_winograd_data_transform_new(rocke_ir_builder_t* b,
                                                            const rocke_winograd_conv_spec_t* s,
                                                            const char* /*arch*/)
{
    char name[256];
    rocke_winograd_conv_spec_kernel_name(s, "data_xform", name, (int)sizeof(name));
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return nullptr;

    const rocke_winograd_problem_t* p = &s->problem;
    const int xs = rocke_winograd_spec_xform_size(s);
    const int ot = s->out_tile;
    const int block_nhw = s->block_nhw;
    const int block_c = s->block_c;
    const int num_tiles = rocke_winograd_spec_num_tiles(s);
    const int ntotal = p->N * num_tiles;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)(block_nhw * block_c));

    // Params — mirror Python param order and opts exactly
    rocke_param_opts_t ro_opts = {.noalias = true,
                                  .noalias_set = true,
                                  .readonly = true,
                                  .readonly_set = true,
                                  .align = 16,
                                  .align_set = true};
    rocke_param_opts_t wo_opts = {.noalias = true,
                                  .noalias_set = true,
                                  .writeonly = true,
                                  .writeonly_set = true,
                                  .align = 16,
                                  .align_set = true};
    rocke_param_opts_t no_opts = {};

    const rocke_type_t* f16_ptr = rocke_ptr_type(b, rocke_f16(), "global");
    const rocke_type_t* f32_ptr = rocke_ptr_type(b, rocke_f32(), "global");

    rocke_value_t* A_ptr = rocke_b_param(b, "A", f16_ptr, &ro_opts);
    rocke_value_t* A_bytes = rocke_b_param(b, "A_bytes", rocke_i32(), &no_opts);
    rocke_value_t* DataWs = rocke_b_param(b, "DataWs", f32_ptr, &wo_opts);
    rocke_value_t* DWs_bytes = rocke_b_param(b, "DataWs_bytes", rocke_i32(), &no_opts);

    // Constants
    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* c4 = rocke_b_const_i32(b, 4);
    rocke_value_t* c_Hi = rocke_b_const_i32(b, p->Hi);
    rocke_value_t* c_Wi = rocke_b_const_i32(b, p->Wi);
    rocke_value_t* c_C = rocke_b_const_i32(b, p->C);
    rocke_value_t* c_pH = rocke_b_const_i32(b, p->pH);
    rocke_value_t* c_pW = rocke_b_const_i32(b, p->pW);
    rocke_value_t* c_tW = rocke_b_const_i32(b, rocke_winograd_spec_tiles_w(s));
    rocke_value_t* c_ot = rocke_b_const_i32(b, ot);
    rocke_value_t* c_ntiles = rocke_b_const_i32(b, num_tiles);
    rocke_value_t* c_ntotal = rocke_b_const_i32(b, ntotal);
    rocke_value_t* c_bk = rocke_b_const_i32(b, block_nhw);
    rocke_value_t* c_bc = rocke_b_const_i32(b, block_c);

    // Thread / block indices
    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bid_nhw = rocke_b_block_id_x(b);
    rocke_value_t* bid_c = rocke_b_block_id_y(b);

    rocke_value_t* local_nhw = rocke_b_mod(b, tid, c_bk);
    rocke_value_t* local_c = rocke_b_div(b, tid, c_bk);
    rocke_value_t* nhw_idx = rocke_b_add(b, rocke_b_mul(b, bid_nhw, c_bk), local_nhw);
    rocke_value_t* c_idx = rocke_b_add(b, rocke_b_mul(b, bid_c, c_bc), local_c);

    // Decompose nhw_idx -> (n, tile_h, tile_w)
    rocke_value_t* tile_idx = rocke_b_mod(b, nhw_idx, c_ntiles);
    rocke_value_t* n_idx = rocke_b_div(b, nhw_idx, c_ntiles);
    rocke_value_t* tile_h_idx = rocke_b_div(b, tile_idx, c_tW);
    rocke_value_t* tile_w_idx = rocke_b_mod(b, tile_idx, c_tW);

    // Top-left of this tile in padded coords
    rocke_value_t* hi_base = rocke_b_sub(b, rocke_b_mul(b, tile_h_idx, c_ot), c_pH);
    rocke_value_t* wi_base = rocke_b_sub(b, rocke_b_mul(b, tile_w_idx, c_ot), c_pW);

    // Buffer resources
    rocke_value_t* a_rsrc = rocke_b_buffer_rsrc(b, A_ptr, A_bytes);
    rocke_value_t* dws_rsrc = rocke_b_buffer_rsrc(b, DataWs, DWs_bytes);

    // Guard condition
    rocke_value_t* nhw_ok = rocke_b_cmp_lt(b, nhw_idx, c_ntotal);
    rocke_value_t* c_ok = rocke_b_cmp_lt(b, c_idx, c_C);
    rocke_value_t* both_ok = rocke_b_land(b, nhw_ok, c_ok);

    rocke_if_t if_stmt = rocke_b_scf_if(b, both_ok);
    rocke_b_region_enter(b, if_stmt.then_region);
    {
        // Load xform_size × xform_size input patch
        rocke_value_t* patch[MAX_XS][MAX_XS] = {};
        for(int rr = 0; rr < xs; ++rr)
        {
            for(int cc = 0; cc < xs; ++cc)
            {
                rocke_value_t* hi = rocke_b_add(b, hi_base, rocke_b_const_i32(b, rr));
                rocke_value_t* wi = rocke_b_add(b, wi_base, rocke_b_const_i32(b, cc));

                // Bounds check
                rocke_value_t* hi_lo = rocke_b_cmp_ge(b, hi, c0);
                rocke_value_t* hi_hi = rocke_b_cmp_lt(b, hi, c_Hi);
                rocke_value_t* wi_lo = rocke_b_cmp_ge(b, wi, c0);
                rocke_value_t* wi_hi = rocke_b_cmp_lt(b, wi, c_Wi);
                rocke_value_t* hi_ok = rocke_b_land(b, hi_lo, hi_hi);
                rocke_value_t* wi_ok = rocke_b_land(b, wi_lo, wi_hi);
                rocke_value_t* in_bounds = rocke_b_land(b, hi_ok, wi_ok);

                // NHWC offset: ((n*Hi + hi)*Wi + wi)*C + c
                rocke_value_t* row_off = rocke_b_add(b, rocke_b_mul(b, n_idx, c_Hi), hi);
                rocke_value_t* col_off = rocke_b_add(b, rocke_b_mul(b, row_off, c_Wi), wi);
                rocke_value_t* elem_off = rocke_b_add(b, rocke_b_mul(b, col_off, c_C), c_idx);
                rocke_value_t* byte_off = rocke_b_mul(b, elem_off, c2);

                // OOB-safe offset: AMD buffer desc returns 0 for sentinel
                rocke_value_t* c_oob = rocke_b_const_i32(b, (int64_t)OOB_SENTINEL);
                rocke_value_t* safe = rocke_b_select(b, in_bounds, byte_off, c_oob);

                rocke_value_t* loaded_f16 = rocke_b_buffer_load_f16(b, a_rsrc, safe, c0);
                patch[rr][cc] = rocke_b_cast_to_f32(b, loaded_f16);
            }
        }

        // Apply B^T * patch * B
        rocke_value_t* xformed[MAX_XS][MAX_XS] = {};
        emit_data_transform(b, ot, xs, patch, xformed);

        // Store (xs × xs) results to data workspace
        // Layout: offset(xh, xw, nhw, c) = ((xh*xs + xw)*ntotal + nhw)*C + c
        for(int xh = 0; xh < xs; ++xh)
        {
            for(int xw = 0; xw < xs; ++xw)
            {
                rocke_value_t* xpos = rocke_b_const_i32(b, xh * xs + xw);
                rocke_value_t* nhw_lay = rocke_b_add(b, rocke_b_mul(b, xpos, c_ntotal), nhw_idx);
                rocke_value_t* ws_off = rocke_b_add(b, rocke_b_mul(b, nhw_lay, c_C), c_idx);
                rocke_value_t* ws_byte = rocke_b_mul(b, ws_off, c4);
                rocke_b_buffer_store_f32(b, dws_rsrc, ws_byte, c0, xformed[xh][xw]);
            }
        }
    }
    rocke_b_region_leave(b);

    return rocke_ir_builder_ok(b) ? b->kernel : nullptr;
}

// ---------------------------------------------------------------------------
// Kernel 2: Filter transform — G * filter * G^T
// ---------------------------------------------------------------------------

rocke_kernel_def_t* rocke_build_winograd_filter_transform_new(rocke_ir_builder_t* b,
                                                              const rocke_winograd_conv_spec_t* s,
                                                              const char* /*arch*/)
{
    char name[256];
    rocke_winograd_conv_spec_kernel_name(s, "filter_xform", name, (int)sizeof(name));
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return nullptr;

    const rocke_winograd_problem_t* p = &s->problem;
    const int xs = rocke_winograd_spec_xform_size(s);
    const int ot = s->out_tile;
    const int fs = 3; // filter_size always 3
    const int block_k = s->block_k;
    const int block_c = s->block_c;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)(block_k * block_c));

    rocke_param_opts_t ro_opts = {.noalias = true,
                                  .noalias_set = true,
                                  .readonly = true,
                                  .readonly_set = true,
                                  .align = 16,
                                  .align_set = true};
    rocke_param_opts_t wo_opts = {.noalias = true,
                                  .noalias_set = true,
                                  .writeonly = true,
                                  .writeonly_set = true,
                                  .align = 16,
                                  .align_set = true};
    rocke_param_opts_t no_opts = {};

    const rocke_type_t* f16_ptr = rocke_ptr_type(b, rocke_f16(), "global");
    const rocke_type_t* f32_ptr = rocke_ptr_type(b, rocke_f32(), "global");

    rocke_value_t* W_ptr = rocke_b_param(b, "W", f16_ptr, &ro_opts);
    rocke_value_t* W_bytes = rocke_b_param(b, "W_bytes", rocke_i32(), &no_opts);
    rocke_value_t* FilterWs = rocke_b_param(b, "FilterWs", f32_ptr, &wo_opts);
    rocke_value_t* FWs_bytes = rocke_b_param(b, "FilterWs_bytes", rocke_i32(), &no_opts);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* c4 = rocke_b_const_i32(b, 4);
    rocke_value_t* c_K = rocke_b_const_i32(b, p->K);
    rocke_value_t* c_C = rocke_b_const_i32(b, p->C);
    rocke_value_t* c_fs = rocke_b_const_i32(b, fs);
    rocke_value_t* c_bk = rocke_b_const_i32(b, block_k);
    rocke_value_t* c_bc = rocke_b_const_i32(b, block_c);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bid_k = rocke_b_block_id_x(b);
    rocke_value_t* bid_c = rocke_b_block_id_y(b);

    rocke_value_t* local_k = rocke_b_mod(b, tid, c_bk);
    rocke_value_t* local_c = rocke_b_div(b, tid, c_bk);
    rocke_value_t* k_idx = rocke_b_add(b, rocke_b_mul(b, bid_k, c_bk), local_k);
    rocke_value_t* c_idx = rocke_b_add(b, rocke_b_mul(b, bid_c, c_bc), local_c);

    rocke_value_t* w_rsrc = rocke_b_buffer_rsrc(b, W_ptr, W_bytes);
    rocke_value_t* fws_rsrc = rocke_b_buffer_rsrc(b, FilterWs, FWs_bytes);

    rocke_value_t* k_ok = rocke_b_cmp_lt(b, k_idx, c_K);
    rocke_value_t* c_ok = rocke_b_cmp_lt(b, c_idx, c_C);
    rocke_value_t* both_ok = rocke_b_land(b, k_ok, c_ok);

    rocke_if_t if_stmt = rocke_b_scf_if(b, both_ok);
    rocke_b_region_enter(b, if_stmt.then_region);
    {
        // KYXC layout: offset(k, y, x, c) = ((k*fs + y)*fs + x)*C + c
        rocke_value_t* filter_patch[MAX_XS][MAX_XS] = {};
        for(int fy = 0; fy < fs; ++fy)
        {
            for(int fx = 0; fx < fs; ++fx)
            {
                rocke_value_t* yx
                    = rocke_b_add(b, rocke_b_mul(b, k_idx, c_fs), rocke_b_const_i32(b, fy));
                rocke_value_t* yx2
                    = rocke_b_add(b, rocke_b_mul(b, yx, c_fs), rocke_b_const_i32(b, fx));
                rocke_value_t* off = rocke_b_add(b, rocke_b_mul(b, yx2, c_C), c_idx);
                rocke_value_t* byte_off = rocke_b_mul(b, off, c2);
                rocke_value_t* f16_val = rocke_b_buffer_load_f16(b, w_rsrc, byte_off, c0);
                filter_patch[fy][fx] = rocke_b_cast_to_f32(b, f16_val);
            }
        }

        rocke_value_t* xformed[MAX_XS][MAX_XS] = {};
        emit_filter_transform(b, ot, xs, fs, filter_patch, xformed);

        // Store: offset(xh, xw, k, c) = ((xh*xs + xw)*K + k)*C + c
        for(int xh = 0; xh < xs; ++xh)
        {
            for(int xw = 0; xw < xs; ++xw)
            {
                rocke_value_t* xpos = rocke_b_const_i32(b, xh * xs + xw);
                rocke_value_t* kc_lay = rocke_b_add(b, rocke_b_mul(b, xpos, c_K), k_idx);
                rocke_value_t* ws_off = rocke_b_add(b, rocke_b_mul(b, kc_lay, c_C), c_idx);
                rocke_value_t* ws_byte = rocke_b_mul(b, ws_off, c4);
                rocke_b_buffer_store_f32(b, fws_rsrc, ws_byte, c0, xformed[xh][xw]);
            }
        }
    }
    rocke_b_region_leave(b);

    return rocke_ir_builder_ok(b) ? b->kernel : nullptr;
}

// ---------------------------------------------------------------------------
// Kernel 3: Output transform — A^T * gemm_result * A
// ---------------------------------------------------------------------------

rocke_kernel_def_t* rocke_build_winograd_output_transform_new(rocke_ir_builder_t* b,
                                                              const rocke_winograd_conv_spec_t* s,
                                                              const char* /*arch*/)
{
    char name[256];
    rocke_winograd_conv_spec_kernel_name(s, "output_xform", name, (int)sizeof(name));
    if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        return nullptr;

    const rocke_winograd_problem_t* p = &s->problem;
    const int xs = rocke_winograd_spec_xform_size(s);
    const int ot = s->out_tile;
    const int block_nhw = s->block_nhw;
    const int block_k = s->block_k;
    const int num_tiles = rocke_winograd_spec_num_tiles(s);
    const int ntotal = p->N * num_tiles;

    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", (int64_t)(block_nhw * block_k));

    rocke_param_opts_t ro_opts = {.noalias = true,
                                  .noalias_set = true,
                                  .readonly = true,
                                  .readonly_set = true,
                                  .align = 16,
                                  .align_set = true};
    rocke_param_opts_t wo_opts = {.noalias = true,
                                  .noalias_set = true,
                                  .writeonly = true,
                                  .writeonly_set = true,
                                  .align = 16,
                                  .align_set = true};
    rocke_param_opts_t no_opts = {};

    const rocke_type_t* f32_ptr = rocke_ptr_type(b, rocke_f32(), "global");
    const rocke_type_t* f16_ptr = rocke_ptr_type(b, rocke_f16(), "global");

    rocke_value_t* GemmWs = rocke_b_param(b, "GemmWs", f32_ptr, &ro_opts);
    rocke_value_t* GWs_bytes = rocke_b_param(b, "GemmWs_bytes", rocke_i32(), &no_opts);
    rocke_value_t* D_ptr = rocke_b_param(b, "D", f16_ptr, &wo_opts);
    rocke_value_t* D_bytes = rocke_b_param(b, "D_bytes", rocke_i32(), &no_opts);

    rocke_value_t* c0 = rocke_b_const_i32(b, 0);
    rocke_value_t* c2 = rocke_b_const_i32(b, 2);
    rocke_value_t* c4 = rocke_b_const_i32(b, 4);
    rocke_value_t* c_Ho = rocke_b_const_i32(b, rocke_winograd_problem_Ho(p));
    rocke_value_t* c_Wo = rocke_b_const_i32(b, rocke_winograd_problem_Wo(p));
    rocke_value_t* c_K = rocke_b_const_i32(b, p->K);
    rocke_value_t* c_tW = rocke_b_const_i32(b, rocke_winograd_spec_tiles_w(s));
    rocke_value_t* c_ot = rocke_b_const_i32(b, ot);
    rocke_value_t* c_ntiles = rocke_b_const_i32(b, num_tiles);
    rocke_value_t* c_ntotal = rocke_b_const_i32(b, ntotal);
    rocke_value_t* c_bk = rocke_b_const_i32(b, block_nhw);
    rocke_value_t* c_bK = rocke_b_const_i32(b, block_k);

    rocke_value_t* tid = rocke_b_thread_id_x(b);
    rocke_value_t* bid_nhw = rocke_b_block_id_x(b);
    rocke_value_t* bid_K = rocke_b_block_id_y(b);

    rocke_value_t* local_nhw = rocke_b_mod(b, tid, c_bk);
    rocke_value_t* local_k = rocke_b_div(b, tid, c_bk);
    rocke_value_t* nhw_idx = rocke_b_add(b, rocke_b_mul(b, bid_nhw, c_bk), local_nhw);
    rocke_value_t* k_idx = rocke_b_add(b, rocke_b_mul(b, bid_K, c_bK), local_k);

    // Decompose nhw_idx -> (n, tile_h, tile_w)
    rocke_value_t* tile_idx = rocke_b_mod(b, nhw_idx, c_ntiles);
    rocke_value_t* n_idx = rocke_b_div(b, nhw_idx, c_ntiles);
    rocke_value_t* tile_h_idx = rocke_b_div(b, tile_idx, c_tW);
    rocke_value_t* tile_w_idx = rocke_b_mod(b, tile_idx, c_tW);

    rocke_value_t* gws_rsrc = rocke_b_buffer_rsrc(b, GemmWs, GWs_bytes);
    rocke_value_t* d_rsrc = rocke_b_buffer_rsrc(b, D_ptr, D_bytes);

    rocke_value_t* nhw_ok = rocke_b_cmp_lt(b, nhw_idx, c_ntotal);
    rocke_value_t* k_ok = rocke_b_cmp_lt(b, k_idx, c_K);
    rocke_value_t* both_ok = rocke_b_land(b, nhw_ok, k_ok);

    rocke_if_t if_stmt = rocke_b_scf_if(b, both_ok);
    rocke_b_region_enter(b, if_stmt.then_region);
    {
        // Load (xs × xs) tile from GEMM workspace
        // offset(xh, xw, nhw, k) = ((xh*xs + xw)*ntotal + nhw)*K + k
        rocke_value_t* acc_tile[MAX_XS][MAX_XS] = {};
        for(int xh = 0; xh < xs; ++xh)
        {
            for(int xw = 0; xw < xs; ++xw)
            {
                rocke_value_t* xpos = rocke_b_const_i32(b, xh * xs + xw);
                rocke_value_t* nhw_lay = rocke_b_add(b, rocke_b_mul(b, xpos, c_ntotal), nhw_idx);
                rocke_value_t* ws_off = rocke_b_add(b, rocke_b_mul(b, nhw_lay, c_K), k_idx);
                rocke_value_t* ws_byte = rocke_b_mul(b, ws_off, c4);
                // Generic scalar f32 buffer load via ROCKE_OP_TILE_BUFFER_LOAD_VN
                // with dwords=1 and elem_type="f32" — mirrors Python
                // b.buffer_load(rsrc, voff, soff, F32).
                {
                    rocke_value_t* operands[3] = {gws_rsrc, ws_byte, c0};
                    rocke_attr_map_t attrs = {};
                    rocke_attr_map_init(&attrs);
                    rocke_attr_set_int(b, &attrs, "dwords", 1);
                    rocke_attr_set_str(b, &attrs, "elem_type", "f32");
                    const rocke_type_t* f32_t = rocke_f32();
                    rocke_op_t* load_op = rocke_b_op(b,
                                                     ROCKE_OP_TILE_BUFFER_LOAD_VN,
                                                     operands,
                                                     3,
                                                     &f32_t,
                                                     1,
                                                     &attrs,
                                                     nullptr,
                                                     0,
                                                     "bl1",
                                                     nullptr);
                    acc_tile[xh][xw] = rocke_op_result(b, load_op);
                }
            }
        }

        // Apply A^T * acc * A → (ot × ot)
        rocke_value_t* out_vals[MAX_XS][MAX_XS] = {};
        emit_output_transform(b, ot, xs, acc_tile, out_vals);

        // Scatter into NHWK output: offset = ((n*Ho + ho)*Wo + wo)*K + k
        for(int oy = 0; oy < ot; ++oy)
        {
            for(int ox = 0; ox < ot; ++ox)
            {
                rocke_value_t* ho
                    = rocke_b_add(b, rocke_b_mul(b, tile_h_idx, c_ot), rocke_b_const_i32(b, oy));
                rocke_value_t* wo
                    = rocke_b_add(b, rocke_b_mul(b, tile_w_idx, c_ot), rocke_b_const_i32(b, ox));
                rocke_value_t* ho_ok = rocke_b_cmp_lt(b, ho, c_Ho);
                rocke_value_t* wo_ok = rocke_b_cmp_lt(b, wo, c_Wo);
                rocke_value_t* in_bounds = rocke_b_land(b, ho_ok, wo_ok);

                rocke_value_t* row_off = rocke_b_add(b, rocke_b_mul(b, n_idx, c_Ho), ho);
                rocke_value_t* col_off = rocke_b_add(b, rocke_b_mul(b, row_off, c_Wo), wo);
                rocke_value_t* elem_off = rocke_b_add(b, rocke_b_mul(b, col_off, c_K), k_idx);
                rocke_value_t* byte_off = rocke_b_mul(b, elem_off, c2);

                rocke_value_t* c_oob = rocke_b_const_i32(b, (int64_t)OOB_SENTINEL);
                rocke_value_t* safe = rocke_b_select(b, in_bounds, byte_off, c_oob);

                rocke_value_t* out_f16 = rocke_b_trunc_f32_to_f16(b, out_vals[oy][ox]);
                rocke_b_buffer_store_f16(b, d_rsrc, safe, c0, out_f16);
            }
        }
    }
    rocke_b_region_leave(b);

    return rocke_ir_builder_ok(b) ? b->kernel : nullptr;
}
