// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * rocke/instance_transpose.c -- C99 port of
 * rocke/instances/common/transpose.py.
 *
 * The build entry mirrors the Python op-by-op so the emitted IR op stream is
 * byte-identical. transpose.py is a pure memory-movement kernel:
 *
 *   1. Phase 1: magic-division unmerge of flat tid -> (row1, col1_chunk);
 *      coalesced global->LDS load via a TileDistribution (load_tile/store_tile).
 *   2. __syncthreads.
 *   3. Phase 2: magic-division unmerge of flat tid -> (col2, row2_chunk);
 *      column-strided scalar LDS reads packed into a coalesced global write.
 *
 * Rather than route through the (partial) C tensor_view / distribution helper
 * surface (which does not yet expose make_lds_view, the distribution-driven
 * load/store, or the (row,col) scalar accessors transpose.py uses), the load /
 * store op streams are reproduced inline here, op-for-op against the Python
 * TensorView / TileWindow / load_tile / store_tile / calculate_x lowerings.
 * The magic-division thread decode DOES go through the ported transforms helper
 * (rocke_tensor_descriptor_naive + rocke_unmerge_magic + unmerge_lower) so its
 * mul-hi sequence is byte-identical to the Python.
 */
#include "rocke/instance_transpose.h"

#include <stdio.h>
#include <string.h>

#include "rocke/arch_target.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"

/* ===================================================================== *
 *  Spec accessors
 * ===================================================================== */

rocke_transpose2d_spec_t rocke_transpose2d_spec_default(void)
{
    rocke_transpose2d_spec_t s;
    s.tile_m = 64;
    s.tile_n = 64;
    s.vec = 8;
    s.dtype = "f16";
    s.lds_pad = 8;
    s.grid_order = "row";
    s.name = "rocke_transpose2d";
    return s;
}

int rocke_transpose2d_block_size(const rocke_transpose2d_spec_t* spec)
{
    if(spec == NULL || spec->vec == 0)
    {
        return 0;
    }
    /* (tile_m * tile_n) // vec */
    return (spec->tile_m * spec->tile_n) / spec->vec;
}

/* Transpose2DSpec.kernel_name():
 *   kernel_name_join(self.name, self.dtype, f"{TM}x{TN}", f"v{vec}",
 *                    f"p{lds_pad}",
 *                    f"g{grid_order}" if grid_order != "row" else "")
 */
rocke_status_t
    rocke_transpose2d_kernel_name(const rocke_transpose2d_spec_t* spec, char* out, size_t out_cap)
{
    char tile_part[32];
    char vec_part[16];
    char pad_part[16];
    char grid_part[32];
    const char* parts[5];
    int w;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    w = snprintf(tile_part, sizeof(tile_part), "%dx%d", spec->tile_m, spec->tile_n);
    if(w < 0 || (size_t)w >= sizeof(tile_part))
    {
        return ROCKE_ERR_VALUE;
    }
    w = snprintf(vec_part, sizeof(vec_part), "v%d", spec->vec);
    if(w < 0 || (size_t)w >= sizeof(vec_part))
    {
        return ROCKE_ERR_VALUE;
    }
    w = snprintf(pad_part, sizeof(pad_part), "p%d", spec->lds_pad);
    if(w < 0 || (size_t)w >= sizeof(pad_part))
    {
        return ROCKE_ERR_VALUE;
    }
    /* f"g{grid_order}" if grid_order != "row" else "" -- empty parts are
     * skipped by kernel_name_join, exactly like the Python. */
    if(spec->grid_order != NULL && strcmp(spec->grid_order, "row") != 0)
    {
        w = snprintf(grid_part, sizeof(grid_part), "g%s", spec->grid_order);
        if(w < 0 || (size_t)w >= sizeof(grid_part))
        {
            return ROCKE_ERR_VALUE;
        }
    }
    else
    {
        grid_part[0] = '\0';
    }

    parts[0] = spec->dtype;
    parts[1] = tile_part;
    parts[2] = vec_part;
    parts[3] = pad_part;
    parts[4] = grid_part;

    return rocke_kernel_name_join(spec->name, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 *  is_valid_spec
 * ===================================================================== */

static void tx_copy_msg(char* dst, size_t cap, const char* msg)
{
    rocke_spec_set_reason(dst, cap, msg);
}

bool rocke_transpose2d_is_valid_spec(const rocke_transpose2d_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap)
{
    const rocke_arch_target_t* target;
    rocke_io_spec_rule_t rule;
    const char* why = NULL;
    int ok;
    int block_size;
    long lds_bytes;
    int max_threads;
    char buf[160];

    if(spec == NULL)
    {
        tx_copy_msg(reason, reason_cap, "null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    target = rocke_arch_target_from_gfx(arch);
    if(target == NULL)
    {
        /* Python: KeyError -> (False, str(e)). */
        snprintf(buf, sizeof(buf), "unknown arch %s", arch);
        tx_copy_msg(reason, reason_cap, buf);
        return false;
    }

    block_size = rocke_transpose2d_block_size(spec);

    /* validate_io(IOSpecRule(dtype, block_size, vec)). The reason string is
     * formatted into a scratch arena (it never enters the IR). */
    rocke_io_spec_rule_init(&rule, spec->dtype, block_size, spec->vec);
    ok = rocke_validate_io(NULL, &rule, &why);
    if(!ok)
    {
        tx_copy_msg(reason, reason_cap, why != NULL ? why : "invalid io");
        return false;
    }

    if((spec->tile_m != 16 && spec->tile_m != 32 && spec->tile_m != 64)
       || (spec->tile_n != 16 && spec->tile_n != 32 && spec->tile_n != 64))
    {
        tx_copy_msg(reason, reason_cap, "tile_m/tile_n must be in {16, 32, 64}");
        return false;
    }
    /* P83: rectangular tiles allowed; only require tile area % vec == 0. */
    if(spec->vec == 0 || (spec->tile_m * spec->tile_n) % spec->vec)
    {
        tx_copy_msg(reason, reason_cap, "tile area must be divisible by vec");
        return false;
    }

    max_threads = rocke_arch_max_threads_per_block(target);
    if(block_size > max_threads)
    {
        snprintf(buf,
                 sizeof(buf),
                 "block_size %d > %d hardware cap (tile_m*tile_n/vec) on %s",
                 block_size,
                 max_threads,
                 arch);
        tx_copy_msg(reason, reason_cap, buf);
        return false;
    }

    /* LDS staging buffer: [tile_m, tile_n + lds_pad] half-words (2 bytes). */
    lds_bytes = (long)spec->tile_m * (long)(spec->tile_n + spec->lds_pad) * 2L;
    if(!rocke_arch_fits_lds(target, lds_bytes))
    {
        snprintf(buf,
                 sizeof(buf),
                 "LDS staging %ld > %d cap on %s",
                 lds_bytes,
                 target->lds_capacity_bytes,
                 arch);
        tx_copy_msg(reason, reason_cap, buf);
        return false;
    }

    tx_copy_msg(reason, reason_cap, "ok");
    return true;
}

/* ===================================================================== *
 *  _decode_thread_layout
 *
 *  Python:
 *    desc = RichTensorDescriptor.naive("thread", lengths=list(dims),
 *               coord_names=[major, minor])
 *               .transform(unmerge_magic("tid", [major, minor], dims=list(dims)))
 *    coords = desc.unmerge_lower(b, tid=tid)
 *    return coords[major], coords[minor]
 *
 *  Emits the exact mul-hi magic-division sequence via the ported transforms
 *  helper. Writes *out_major / *out_minor; returns true on success.
 * ===================================================================== */
static bool tx_decode_thread_layout(rocke_ir_builder_t* b,
                                    rocke_value_t* tid,
                                    const char* major,
                                    const char* minor,
                                    int dims0,
                                    int dims1,
                                    rocke_value_t** out_major,
                                    rocke_value_t** out_minor)
{
    int lengths[2];
    const char* coord_names[2];
    const char* lowers[2];
    int dims[2];
    rocke_tensor_descriptor_t* desc;
    rocke_transform_t* xf;
    const rocke_transform_t* chain[1];
    rocke_tensor_descriptor_t* desc2;
    const char* in_names[1];
    rocke_value_t* in_values[1];
    const char* out_names[8];
    rocke_value_t* out_values[8];
    int n_out;
    int i;
    rocke_value_t* maj = NULL;
    rocke_value_t* min = NULL;

    lengths[0] = dims0;
    lengths[1] = dims1;
    coord_names[0] = major;
    coord_names[1] = minor;
    lowers[0] = major;
    lowers[1] = minor;
    dims[0] = dims0;
    dims[1] = dims1;

    desc = rocke_tensor_descriptor_naive(b, "thread", lengths, 2, NULL, coord_names, 2);
    if(desc == NULL)
    {
        return false;
    }
    xf = rocke_unmerge_magic(b, "tid", lowers, 2, dims);
    if(xf == NULL)
    {
        return false;
    }
    chain[0] = xf;
    desc2 = rocke_tensor_descriptor_transform(b, desc, chain, 1);
    if(desc2 == NULL)
    {
        return false;
    }

    in_names[0] = "tid";
    in_values[0] = tid;
    n_out = rocke_tensor_descriptor_unmerge_lower(
        b, desc2, in_names, in_values, 1, out_names, out_values, 8);
    if(n_out < 0)
    {
        return false;
    }
    for(i = 0; i < n_out; ++i)
    {
        if(strcmp(out_names[i], major) == 0)
        {
            maj = out_values[i];
        }
        else if(strcmp(out_names[i], minor) == 0)
        {
            min = out_values[i];
        }
    }
    if(maj == NULL || min == NULL)
    {
        b->status = ROCKE_ERR_VALUE;
        tx_copy_msg(b->err, sizeof(b->err), "thread decode missing coord");
        return false;
    }
    *out_major = maj;
    *out_minor = min;
    return true;
}

/* ===================================================================== *
 *  _morton_decode_bits
 *
 *  Compact the bits at positions shift, shift+2, ... of `val` into a
 *  contiguous integer with `width` significant bits, emitting the exact
 *  land/lor/div sequence the Python uses.
 * ===================================================================== */
static rocke_value_t*
    tx_morton_decode_bits(rocke_ir_builder_t* b, rocke_value_t* val, int shift, int width)
{
    rocke_value_t* v = val;
    /* Python: (1 << (2*width)) - 1 in arbitrary precision. width==16 (the
     * production path, transpose.py 279-280) gives 1<<32, which is UB for a
     * 32-bit int. When 2*width >= 32 the full 32-bit mask is wanted, i.e. -1
     * (0xFFFFFFFF), making `masks[s] & mask_lim` a no-op exactly like Python. */
    int mask_lim = (2 * width >= 32) ? -1 : ((1 << (2 * width)) - 1);

    if(shift)
    {
        v = rocke_b_div(b, v, rocke_b_const_i32(b, 1 << shift));
    }
    v = rocke_b_land(b, v, rocke_b_const_i32(b, 0x55555555 & mask_lim));
    /* Steps 1-4: v = (v | (v >> k)) & MASK.
     *
     * Python evaluates each b.land(b.lor(v, b.div(v, b.const_i32(k))),
     * b.const_i32(MASK)) strictly left-to-right, so the op stream is:
     *   const(k), div, lor, const(MASK), land
     * C function-argument evaluation order is UNSPECIFIED (and in practice
     * right-to-left here), which would emit const(MASK) before const(k) and
     * shift every SSA number. Sequence the sub-expressions into explicit
     * temporaries so the emission order matches Python exactly. */
    {
        const int divisors[4] = {2, 4, 16, 256};
        const int masks[4] = {0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF};
        int s;
        for(s = 0; s < 4; ++s)
        {
            rocke_value_t* kconst = rocke_b_const_i32(b, divisors[s]);
            rocke_value_t* quot = rocke_b_div(b, v, kconst);
            rocke_value_t* ored = rocke_b_lor(b, v, quot);
            rocke_value_t* mconst = rocke_b_const_i32(b, masks[s] & mask_lim);
            v = rocke_b_land(b, ored, mconst);
        }
    }
    return v;
}

/* ===================================================================== *
 *  build_transpose2d
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_transpose2d(rocke_ir_builder_t* b,
                                            const rocke_transpose2d_spec_t* spec,
                                            const char* arch)
{
    char reason[160];
    const rocke_type_t* io_ty;
    int TM, TN, vec, BS;
    rocke_param_opts_t opts_x, opts_y;
    rocke_value_t* X;
    rocke_value_t* Y;
    rocke_value_t* H;
    rocke_value_t* W;
    rocke_value_t* tid;
    rocke_value_t* c_vec;
    rocke_value_t* tile_x;
    rocke_value_t* tile_y;
    rocke_value_t* h0;
    rocke_value_t* w0;
    rocke_value_t* lds_smem;
    int lds_shape[2];
    /* lds_tile origin consts (b.const_i32(0), b.const_i32(0)); reused by
     * every _global_indices add against the LDS window. */
    rocke_value_t* lds_org0;
    rocke_value_t* lds_org1;
    /* Phase-1 thread coords. */
    rocke_value_t* row1 = NULL;
    rocke_value_t* col1_chunk = NULL;
    /* Phase-2 thread coords. */
    rocke_value_t* col2 = NULL;
    rocke_value_t* row2_chunk = NULL;
    rocke_value_t* row2_base;
    int i;

    if(b == NULL || spec == NULL)
    {
        return NULL;
    }
    (void)arch;

    /* Python: ok, why = is_valid_spec(spec); raise ValueError on reject.
     * is_valid_spec validates against gfx950 (the Python default). */
    if(!rocke_transpose2d_is_valid_spec(spec, "gfx950", reason, sizeof(reason)))
    {
        char msg[224];
        snprintf(msg, sizeof(msg), "invalid transpose2d spec: %s", reason);
        b->status = ROCKE_ERR_VALUE;
        tx_copy_msg(b->err, sizeof(b->err), msg);
        return NULL;
    }

    io_ty = rocke_b_io_ir_type(b, spec->dtype);
    if(io_ty == NULL)
    {
        return NULL;
    }
    TM = spec->tile_m;
    TN = spec->tile_n;
    vec = spec->vec;
    BS = rocke_transpose2d_block_size(spec);

    /* b.kernel.attrs["max_workgroup_size"] = BS */
    rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", BS);

    /* Params:
     *   X = b.param("X", PtrType(io_ty,"global"), noalias, readonly, align=16)
     *   Y = b.param("Y", PtrType(io_ty,"global"), noalias, writeonly, align=16)
     *   H = b.param("H", I32); W = b.param("W", I32)
     */
    memset(&opts_x, 0, sizeof(opts_x));
    opts_x.noalias = true;
    opts_x.noalias_set = true;
    opts_x.readonly = true;
    opts_x.readonly_set = true;
    opts_x.align = 16;
    opts_x.align_set = true;
    X = rocke_b_param(b, "X", rocke_ptr_type(b, io_ty, "global"), &opts_x);

    memset(&opts_y, 0, sizeof(opts_y));
    opts_y.noalias = true;
    opts_y.noalias_set = true;
    opts_y.writeonly = true;
    opts_y.writeonly_set = true;
    opts_y.align = 16;
    opts_y.align_set = true;
    Y = rocke_b_param(b, "Y", rocke_ptr_type(b, io_ty, "global"), &opts_y);

    H = rocke_b_param(b, "H", rocke_i32(), NULL);
    W = rocke_b_param(b, "W", rocke_i32(), NULL);

    tid = rocke_b_thread_id_x(b);

    c_vec = rocke_b_const_i32(b, vec);

    /* Decode logical (tile_x, tile_y) from block_id per grid order. */
    if(spec->grid_order != NULL && strcmp(spec->grid_order, "morton") == 0)
    {
        rocke_value_t* bid = rocke_b_block_id_x(b);
        tile_x = tx_morton_decode_bits(b, bid, 0, 16);
        tile_y = tx_morton_decode_bits(b, bid, 1, 16);
    }
    else
    {
        tile_x = rocke_b_block_id_x(b);
        tile_y = rocke_b_block_id_y(b);
    }

    /* h0 = tile_y * TM; w0 = tile_x * TN */
    h0 = rocke_b_mul(b, tile_y, rocke_b_const_i32(b, TM));
    w0 = rocke_b_mul(b, tile_x, rocke_b_const_i32(b, TN));

    /* LDS staging view: shape (TM, TN + lds_pad), packed-row-major strides
     * (TN+lds_pad, 1). make_lds_view = smem_alloc(dtype, shape). */
    lds_shape[0] = TM;
    lds_shape[1] = TN + spec->lds_pad;
    lds_smem = rocke_b_smem_alloc(b, io_ty, lds_shape, 2, "lds_xpose");

    /* lds_tile = lds_view.tile(lengths=(TM,TN),
     *                          origin=(b.const_i32(0), b.const_i32(0)))
     * TileWindow construction itself emits no IR, but the Python call site
     * materialises the two origin const_i32(0) values eagerly. They fold to
     * literal 0 in the lowered IR (so they never appear as named SSA), yet
     * they still advance the IR value counter by two -- omitting them shifts
     * every subsequent SSA number. Keep the handles: Python's _global_indices
     * reuses these very origin consts for every add against the LDS window
     * (store_tile g0/g1 and the Phase-2 load_scalar g0/g1), so reusing them
     * here -- instead of emitting fresh const(0)s -- keeps the SSA stream
     * byte-identical. */
    lds_org0 = rocke_b_const_i32(b, 0);
    lds_org1 = rocke_b_const_i32(b, 0);

    /* Phase 1 thread layout: (row1, col1_chunk), dims=(TM, TN//vec). */
    if(!tx_decode_thread_layout(b, tid, "row1", "col1_chunk", TM, TN / vec, &row1, &col1_chunk))
    {
        return NULL;
    }

    /* ---- Phase 1: coalesced global -> LDS (load_tile + store_tile) ----
     *
     * The phase-1 TileDistribution encoding splits the (TM, TN) tile as
     *   Hs = ((TM,), (TN//vec, vec))
     * with P0=row1 feeding X0 level0, P1=col1_chunk feeding X1 level0, and
     * Y0 feeding X1 level1 (the vec dim).
     *
     * make_load_store_traits picks vector_dim_y=0, scalar_per_vector=vec
     * (Y0 is stride-1 with length vec), and iterate_accesses yields a single
     * y_base=(0,). load_tile then issues ONE vector access:
     *
     *   x0 = calculate_x dim0 = add(const0, row1)
     *   x1 = calculate_x dim1 = add(add(const0, ys0=const0),
     *                               mul(col1_chunk, const_i32(vec)))
     *   load_vec_as_f32(x_view tile @ (h0,w0), x0, x1, n=vec):
     *     g0 = add(h0, x0); g1 = add(w0, x1)
     *     off = add(mul(g0, W), g1)            (x_view strides=(W,1))
     *     v   = global_load_vN(X, off, io_ty, vec)
     *     scalars[i] = cast_to_f32(vec_extract(v, i))
     *
     * store_tile recomputes calculate_x (fresh ops, same structure) and
     * writes through the LDS window:
     *   store_vec_from_f32(lds tile @ (0,0), x0', x1', values=scalars):
     *     casts[i] = cast_f32_to(scalars[i], io_ty)
     *     packed   = vec_pack(casts, io_ty)
     *     g0 = add(const0, x0'); g1 = add(const0, x1')
     *     smem_store_vN(lds_smem, [g0, g1], packed, vec)
     */
    {
        rocke_value_t* ys0;
        rocke_value_t* x0;
        rocke_value_t* x1;
        rocke_value_t* g0;
        rocke_value_t* g1;
        rocke_value_t* off;
        rocke_value_t* vv;
        rocke_value_t* scalars[8];
        rocke_value_t* casts[8];
        rocke_value_t* packed;
        rocke_value_t* idx2[2];

        /* --- load_tile: calculate_x for y_base=(0,) ---
         *
         * load_tile builds the ys list FIRST:
         *   ys = [b.const_i32(0)]              (one const per Y dim)
         * then calculate_x walks each x_dim, seeding a FRESH const_i32(0)
         * accumulator per dim (distribution.calculate_x line `x = b.const_i32(0)`).
         * The op ORDER (ys const, then x_dim0 const+add, then x_dim1 const+...)
         * and the per-dim fresh zero are load-bearing for byte-exact SSA. */
        ys0 = rocke_b_const_i32(b, 0);
        /* x_dim 0: hs=(TM,), level0 contributor P0=row1, stride==1. */
        x0 = rocke_b_add(b, rocke_b_const_i32(b, 0), row1);
        /* x_dim 1: hs=(TN//vec, vec). level1 (Y0) then level0 (P1). */
        x1 = rocke_b_add(b, rocke_b_const_i32(b, 0), ys0); /* stride==1 first */
        x1 = rocke_b_add(b, x1, rocke_b_mul(b, col1_chunk, rocke_b_const_i32(b, vec)));

        /* load_vec_as_f32 over the global x_view tile @ origin (h0, w0). */
        g0 = rocke_b_add(b, h0, x0);
        g1 = rocke_b_add(b, w0, x1);
        /* desc.offset: strides=(W, 1) -> off = add(mul(g0, W), g1). */
        off = rocke_b_add(b, rocke_b_mul(b, g0, W), g1);
        vv = rocke_b_global_load_vN(b, X, off, io_ty, vec, 0);
        for(i = 0; i < vec; ++i)
        {
            scalars[i] = rocke_b_cast_to_f32(b, rocke_b_vec_extract(b, vv, i));
        }

        /* --- store_tile: recompute calculate_x (fresh ops, same order) --- */
        ys0 = rocke_b_const_i32(b, 0);
        x0 = rocke_b_add(b, rocke_b_const_i32(b, 0), row1);
        x1 = rocke_b_add(b, rocke_b_const_i32(b, 0), ys0);
        x1 = rocke_b_add(b, x1, rocke_b_mul(b, col1_chunk, rocke_b_const_i32(b, vec)));

        /* store_vec_from_f32 over the LDS lds_tile @ origin (0, 0). */
        for(i = 0; i < vec; ++i)
        {
            casts[i] = rocke_b_cast_f32_to(b, scalars[i], io_ty);
        }
        packed = rocke_b_vec_pack(b, casts, vec, io_ty);
        /* store_tile -> _global_indices: add(lds_origin, x) reusing the origin
         * consts emitted at lds_tile creation (no fresh const_i32(0)). */
        g0 = rocke_b_add(b, lds_org0, x0);
        g1 = rocke_b_add(b, lds_org1, x1);
        idx2[0] = g0;
        idx2[1] = g1;
        rocke_b_smem_store_vN(b, lds_smem, idx2, 2, packed, vec);
    }

    /* __syncthreads */
    rocke_b_sync(b);

    /* Phase 2 thread layout: (col2, row2_chunk), dims=(TN, TM//vec). */
    if(!tx_decode_thread_layout(b, tid, "col2", "row2_chunk", TN, TM / vec, &col2, &row2_chunk))
    {
        return NULL;
    }
    /* row2_base = row2_chunk * c_vec */
    row2_base = rocke_b_mul(b, row2_chunk, c_vec);

    /* ---- Phase 2: column-strided LDS scalar reads + coalesced global write --
     *
     *   elems[i] = lds_tile.load_scalar(b, add(row2_base, const_i32(i)), col2)
     *            : lds_tile @ origin (0,0):
     *                g0 = add(const0, add(row2_base, const_i32(i)))
     *                g1 = add(const0, col2)
     *                vvec = smem_load_vN(lds_smem, [g0, g1], io_ty, n=1)
     *                elems[i] = vec_extract(vvec, 0)
     *   out_vec = vec_pack(elems, io_ty)
     *   y_tile.store_vec(b, col2, row2_base, value=out_vec, n=vec)
     *            : y_tile @ origin (w0, h0), y_view strides=(H, 1):
     *                g0 = add(w0, col2); g1 = add(h0, row2_base)
     *                off = add(mul(g0, H), g1)
     *                global_store_vN(Y, off, out_vec, vec)
     */
    {
        rocke_value_t* elems[8];
        rocke_value_t* out_vec;
        rocke_value_t* g0;
        rocke_value_t* g1;
        rocke_value_t* off;
        rocke_value_t* idx2[2];
        rocke_value_t* vvec;

        for(i = 0; i < vec; ++i)
        {
            rocke_value_t* row_i = rocke_b_add(b, row2_base, rocke_b_const_i32(b, i));
            /* TileWindow.load_scalar -> _global_indices: add(lds_origin, idx)
             * reusing the lds_tile origin consts (no fresh const_i32(0)). */
            g0 = rocke_b_add(b, lds_org0, row_i);
            g1 = rocke_b_add(b, lds_org1, col2);
            /* view.load_scalar computes the flat element offset
             *   off = desc.offset(b, [g0, g1])
             *       = add(mul(g0, stride_r), g1)   (col stride 1 omitted)
             * BEFORE dispatching on addr_space. For the LDS path the result
             * is unused by smem_load_vN (which takes the multi-index), but
             * the mul/add ops are still emitted into the IR as a side effect.
             * The LDS desc strides are (TN+lds_pad, 1). Reproduce the pair so
             * the op stream matches Python byte-for-byte. */
            (void)rocke_b_add(b, rocke_b_mul(b, g0, rocke_b_const_i32(b, TN + spec->lds_pad)), g1);
            idx2[0] = g0;
            idx2[1] = g1;
            vvec = rocke_b_smem_load_vN(b, lds_smem, idx2, 2, io_ty, 1);
            elems[i] = rocke_b_vec_extract(b, vvec, 0);
        }
        out_vec = rocke_b_vec_pack(b, elems, vec, io_ty);

        /* y_tile.store_vec(col2, row2_base). */
        g0 = rocke_b_add(b, w0, col2);
        g1 = rocke_b_add(b, h0, row2_base);
        off = rocke_b_add(b, rocke_b_mul(b, g0, H), g1);
        rocke_b_global_store_vN(b, Y, off, out_vec, vec, 0);
    }

    if(b->status != ROCKE_OK)
    {
        return NULL;
    }
    return b->kernel;
}

/* ===================================================================== *
 *  rocke_build_transpose2d_new -- init the builder with spec.kernel_name()
 *  then build. (public convenience.)
 * ===================================================================== */
rocke_kernel_def_t* rocke_build_transpose2d_new(rocke_ir_builder_t* b,
                                                const rocke_transpose2d_spec_t* spec,
                                                const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_transpose2d_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_transpose2d(b, spec, arch);
    });
}

/* ===================================================================== *
 *  transpose2d_grid
 *
 *  Python:
 *    nx = ceil(w / tile_n); ny = ceil(h / tile_m)
 *    if grid_order == "morton":
 *        side = next_pow2(max(nx, ny)); return (side*side, 1, 1)
 *    return ceil_div_grid((w, tile_n), (h, tile_m))
 * ===================================================================== */
rocke_status_t
    rocke_transpose2d_grid(int h, int w, const rocke_transpose2d_spec_t* spec, int out[3])
{
    int nx, ny;
    if(spec == NULL || out == NULL || spec->tile_m <= 0 || spec->tile_n <= 0)
    {
        return ROCKE_ERR_VALUE;
    }
    nx = (w + spec->tile_n - 1) / spec->tile_n;
    ny = (h + spec->tile_m - 1) / spec->tile_m;
    if(spec->grid_order != NULL && strcmp(spec->grid_order, "morton") == 0)
    {
        int side = 1;
        int target = nx > ny ? nx : ny;
        while(side < target)
        {
            side *= 2;
        }
        out[0] = side * side;
        out[1] = 1;
        out[2] = 1;
        return ROCKE_OK;
    }
    {
        int totals[2];
        int tiles[2];
        totals[0] = w;
        tiles[0] = spec->tile_n;
        totals[1] = h;
        tiles[1] = spec->tile_m;
        return rocke_ceil_div_grid(totals, tiles, 2, out);
    }
}

/* ===================================================================== *
 *  rocke_transpose2d_lower_to_llvm -- build + lower to .ll convenience.
 *  Owns and frees its own IRBuilder.
 * ===================================================================== */
rocke_status_t rocke_transpose2d_lower_to_llvm(const rocke_transpose2d_spec_t* spec,
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
        tx_copy_msg(err, err_cap, "lower_to_llvm: null spec/out");
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_transpose2d_new(&b, spec, arch);
    if(kernel == NULL)
    {
        const char* m = rocke_ir_builder_error(&b);
        st = rocke_ir_builder_status(&b);
        tx_copy_msg(err, err_cap, m != NULL ? m : "build_transpose2d failed");
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
