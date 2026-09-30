// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * instance_pooling.c -- C99 port of rocke/instances/common/pooling.py.
 *
 * Ported symbols:
 *   PoolingProblem (+ Ho/Wo/total_out/short)   rocke_pooling_problem_*
 *   Pooling2DSpec (+ kernel_name)              rocke_pooling2d_spec_* / _kernel_name
 *   is_valid_spec                              rocke_pooling2d_is_valid_spec
 *   build_pooling2d                            rocke_build_pooling2d (+ _new)
 *   pooling2d_grid                             rocke_pooling2d_grid
 *   pooling2d_signature                        rocke_pooling2d_signature
 *   (build + lower convenience)                rocke_pooling2d_lower_to_llvm
 *
 * The build entry reproduces the Python build_pooling2d() op-by-op so the
 * downstream IR stream is byte-identical.
 *
 * STORE-EPILOGUE (NAMED GAP): make_buffer_resource / make_buffer_view /
 * store_tile and the StaticDistributedTensor .set / TensorView .tile methods
 * from rocke.helpers.{tensor_view,distribution} are ported here as fully-wired,
 * file-local helpers (defined below) that emit the byte-identical SSA op stream
 * -- they are NOT stubs (make_static_distributed_tensor + the encoding
 * constructor + make_static_tile_distribution come from the shared helper layer
 * and are used directly). The gap is purely organisational: these file-local
 * buffer-view helpers duplicate what should eventually live in the shared
 * tensor_view / distribution C ports, and can be deleted once those expose a
 * buffer-space TensorView. The store dtype guards (f16 only in buffer space)
 * faithfully mirror the Python NotImplementedError reject paths (see below).
 */
#include "rocke/instance_pooling.h"

#include <math.h> /* INFINITY */
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/error_boundary.hpp" /* ckc::guard_builder boundary shim */
#include "rocke/helper_rocke.core.arch.h"
#include "rocke/helper_rocke.helpers.distribution.h"
#include "rocke/helper_rocke.helpers.io.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir_internal.h" /* rocke_i_set_err */
#include "rocke/lower_llvm.h"

/* ===================================================================== *
 * Store-epilogue helpers (ports of helpers/{tensor_view,distribution}.py
 * store side).
 *
 * The build entry's store step mirrors the Python:
 *
 *     out_rsrc = make_buffer_resource(b, Y, num_bytes=Y_bytes)
 *     out_view = make_buffer_view(out_rsrc, [out_total], io_ty)
 *     ...
 *     out_window = out_view.tile(lengths=[VEC], origin=[out_origin])
 *     out_dt     = make_static_distributed_tensor(out_dist, dtype=io_ty)
 *     out_dt.set([k], acc_list[k])
 *     store_tile(b, out_window, out_dt, ps=[])
 *
 * ``make_static_distributed_tensor`` is the shared distribution-port symbol
 * (helper_rocke.helpers.distribution.{h,c}); the remaining five helpers are
 * the pooling store path and are defined below with file-local concrete
 * structs. They emit the byte-identical SSA op stream:
 *
 *   make_buffer_resource  -> rsrc = b.buffer_rsrc(ptr, num_bytes);
 *                            soffset = const_i32(0)            (Python identical)
 *   make_buffer_view      -> pure host struct (TensorDescriptor.packed), no IR
 *   tile                  -> pure host struct (TileWindow),          no IR
 *   set                   -> writes storage[y_to_linear(y)],         no IR
 *   store_tile            -> distribution.calculate_x + store_vec_from_f32
 *                            (f32 demote/pack) + raw_ptr_buffer_store.
 * ===================================================================== */

/* NB: the rich named-coord rocke_tensor_descriptor (helpers.transforms.h, already
 * included above) and the simple shape/stride descriptor in
 * helpers.tensor_view.h share the struct tag ``rocke_tensor_descriptor`` but are
 * different types -- the two headers cannot coexist in one TU. The pooling store
 * path needs only a packed buffer view, so we carry the flat shape directly here
 * (a self-contained mirror of helpers/tensor_view.py's buffer TensorView) and
 * compute the packed offset inline; this keeps the byte-identical op stream
 * without dragging in the conflicting header. */
#define ROCKE_POOL_TV_MAX_RANK 8

/* helpers/tensor_view.py: BufferResource (rsrc + soffset + num_bytes). */
typedef struct rocke_buffer_resource
{
    rocke_value_t* rsrc; /* 128-bit buffer descriptor (b.buffer_rsrc(...))   */
    rocke_value_t* soffset; /* scalar byte offset; default const_i32(0)         */
    int num_bytes; /* informational (host-side int when known)         */
} rocke_buffer_resource_t;

/* helpers/tensor_view.py: a buffer-space TensorView (rsrc + packed descriptor).
 * Pooling uses a single rank-1 [out_total] axis; ``strides`` are packed
 * row-major (the trailing axis has stride 1, which the offset elides). */
typedef struct rocke_buffer_view
{
    rocke_buffer_resource_t* rsrc;
    int rank;
    int shape[ROCKE_POOL_TV_MAX_RANK];
    int strides[ROCKE_POOL_TV_MAX_RANK]; /* packed row-major element strides     */
    const rocke_type_t* dtype;
} rocke_buffer_view_t;

/* helpers/tensor_view.py: TileWindow over a buffer view (lengths + origin). */
typedef struct rocke_buffer_window
{
    const rocke_buffer_view_t* view;
    int lengths[ROCKE_POOL_TV_MAX_RANK];
    rocke_value_t* origin[ROCKE_POOL_TV_MAX_RANK];
    int rank;
} rocke_buffer_window_t;

/* make_static_distributed_tensor(distribution, dtype): shared distribution-port
 * symbol (defined in helper_rocke.helpers.distribution.c, declared in its
 * header which is included above). Used by the build entry directly. */

/* make_buffer_resource(b, ptr, num_bytes=...): rsrc = b.buffer_rsrc(ptr,
 * num_bytes); soffset defaults to const_i32(0). */
static rocke_buffer_resource_t*
    rocke_make_buffer_resource(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* num_bytes);

/* make_buffer_view(rsrc, lengths, dtype): flat single-axis buffer TensorView.
 * `lengths` is length n_lengths (pooling uses a single [out_total] axis). */
static rocke_buffer_view_t* rocke_make_buffer_view(rocke_ir_builder_t* b,
                                                   rocke_buffer_resource_t* rsrc,
                                                   const int* lengths,
                                                   int n_lengths,
                                                   const rocke_type_t* dtype);

/* TensorView.tile(lengths, origin): fixed-extent window. Pooling stores a
 * single rank-1 window of length VEC at origin [out_origin]. */
static void* rocke_buffer_view_tile(rocke_ir_builder_t* b,
                                    rocke_buffer_view_t* view,
                                    const int* lengths,
                                    int n_lengths,
                                    rocke_value_t* const* origin,
                                    int n_origin);

/* StaticDistributedTensor.set(y, value): write the f32/dtype scalar at Y
 * position `y` (here a single-element index [k]). */
static void rocke_static_distributed_tensor_set(rocke_ir_builder_t* b,
                                                rocke_static_distributed_tensor_t* dt,
                                                const int* y,
                                                int n_y,
                                                rocke_value_t* value);

/* store_tile(b, window, distributed_tensor, ps=[]): demote+pack the f32 Y-slot
 * scalars and issue the coalesced buffer_store. `ps` is the empty list here. */
static void rocke_store_tile(rocke_ir_builder_t* b,
                             void* window,
                             rocke_static_distributed_tensor_t* dt,
                             rocke_value_t* const* ps,
                             int n_ps);

/* ===================================================================== *
 * PoolingProblem
 * ===================================================================== */

rocke_pooling_problem_t rocke_pooling_problem_default(void)
{
    rocke_pooling_problem_t p;
    memset(&p, 0, sizeof(p));
    p.sH = 1;
    p.sW = 1;
    p.pH = 0;
    p.pW = 0;
    p.dH = 1;
    p.dW = 1;
    return p;
}

int rocke_pooling_problem_ho(const rocke_pooling_problem_t* p)
{
    /* (H + 2*pH - ((Y-1)*dH + 1)) // sH + 1 */
    int numer = p->H + 2 * p->pH - ((p->Y - 1) * p->dH + 1);
    /* Python // (floor division); the conv shapes keep numer >= 0 and sH > 0,
     * so C truncating division matches floor here. */
    return numer / p->sH + 1;
}

int rocke_pooling_problem_wo(const rocke_pooling_problem_t* p)
{
    int numer = p->W + 2 * p->pW - ((p->X - 1) * p->dW + 1);
    return numer / p->sW + 1;
}

int rocke_pooling_problem_total_out(const rocke_pooling_problem_t* p)
{
    return p->N * rocke_pooling_problem_ho(p) * rocke_pooling_problem_wo(p) * p->C;
}

rocke_status_t
    rocke_pooling_problem_short(const rocke_pooling_problem_t* p, char* out, size_t out_cap)
{
    int n;
    if(p == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    /* f"N{N}H{H}W{W}C{C}_Y{Y}X{X}_s{sH}x{sW}_p{pH}x{pW}" */
    n = snprintf(out,
                 out_cap,
                 "N%dH%dW%dC%d_Y%dX%d_s%dx%d_p%dx%d",
                 p->N,
                 p->H,
                 p->W,
                 p->C,
                 p->Y,
                 p->X,
                 p->sH,
                 p->sW,
                 p->pH,
                 p->pW);
    if(n < 0 || (size_t)n >= out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    return ROCKE_OK;
}

/* ===================================================================== *
 * Pooling2DSpec
 * ===================================================================== */

rocke_pooling2d_spec_t rocke_pooling2d_spec_default(void)
{
    rocke_pooling2d_spec_t s;
    memset(&s, 0, sizeof(s));
    s.problem = rocke_pooling_problem_default();
    s.dtype = "f16";
    s.op = "max";
    s.block_size = 256;
    s.vec = 1;
    s.name = "rocke_pooling2d";
    s.tile_n = 1;
    s.use_warp_xor_reduce = false;
    return s;
}

rocke_status_t
    rocke_pooling2d_kernel_name(const rocke_pooling2d_spec_t* spec, char* out, size_t out_cap)
{
    char shortbuf[128];
    char bbuf[24];
    char vbuf[24];
    const char* parts[5];
    rocke_status_t st;

    if(spec == NULL || out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_pooling_problem_short(&spec->problem, shortbuf, sizeof(shortbuf));
    if(st != ROCKE_OK)
    {
        return st;
    }
    snprintf(bbuf, sizeof(bbuf), "b%d", spec->block_size);
    snprintf(vbuf, sizeof(vbuf), "v%d", spec->vec);

    /* kernel_name_join(name, problem.short(), dtype, op, f"b{bs}", f"v{vec}") */
    parts[0] = shortbuf;
    parts[1] = spec->dtype;
    parts[2] = spec->op;
    parts[3] = bbuf;
    parts[4] = vbuf;
    return rocke_kernel_name_join(spec->name, parts, 5, NULL, NULL, 0, out, out_cap, NULL);
}

/* ===================================================================== *
 * is_valid_spec
 * ===================================================================== */

bool rocke_pooling2d_is_valid_spec(const rocke_pooling2d_spec_t* spec,
                                   const char* arch,
                                   char* reason,
                                   size_t reason_cap)
{
    const rocke_archtarget_t* target;
    int legal_bs[5];
    size_t n_legal = 0;
    int max_tpb;
    int candidate_bs[5] = {64, 128, 256, 512, 1024};
    size_t i;
    rocke_io_spec_rule_t rule;
    rocke_arena_t arena;
    const char* why = NULL;
    int ok;
    const rocke_pooling_problem_t* p;
    int Ho, Wo;

#define POOL_REASON(...)                               \
    do                                                 \
    {                                                  \
        if(reason != NULL && reason_cap > 0)           \
        {                                              \
            snprintf(reason, reason_cap, __VA_ARGS__); \
        }                                              \
    } while(0)

    if(spec == NULL)
    {
        POOL_REASON("null spec");
        return false;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    /* target = ArchTarget.from_gfx(arch) (KeyError -> reject). */
    target = rocke_archtarget_from_gfx(arch);
    if(target == NULL)
    {
        POOL_REASON("'%s'", arch);
        return false;
    }

    /* if spec.vec not in (1, 2, 4, 8): reject. */
    if(spec->vec != 1 && spec->vec != 2 && spec->vec != 4 && spec->vec != 8)
    {
        POOL_REASON("unsupported vec %d; expected one of {1, 2, 4, 8}", spec->vec);
        return false;
    }

    /* _legal_bs = tuple(bs for bs in (64,128,256,512,1024)
     *                   if bs <= target.max_threads_per_block). */
    max_tpb = rocke_archtarget_max_threads_per_block(target);
    for(i = 0; i < 5; ++i)
    {
        if(candidate_bs[i] <= max_tpb)
        {
            legal_bs[n_legal++] = candidate_bs[i];
        }
    }

    /* validate_io(IOSpecRule(dtype, block_size, vec=vec if vec>=2 else 2,
     *                        allowed_block_sizes=_legal_bs)). */
    rocke_io_spec_rule_init(&rule, spec->dtype, spec->block_size, spec->vec >= 2 ? spec->vec : 2);
    rule.allowed_block_sizes = legal_bs;
    rule.num_allowed_block_sizes = n_legal;

    /* validate_io's reject reason is surfaced only through ValueError text; a
     * scratch arena backs the formatted string (never enters the IR). */
    if(rocke_arena_init(&arena, 4096) != 0)
    {
        POOL_REASON("arena init failed");
        return false;
    }
    ok = rocke_validate_io(&arena, &rule, &why);
    if(!ok)
    {
        POOL_REASON("%s", why != NULL ? why : "invalid io");
        rocke_arena_destroy(&arena);
        return false;
    }
    rocke_arena_destroy(&arena);

    /* if spec.op not in ("max","avg","sum"): reject. */
    if(strcmp(spec->op, "max") != 0 && strcmp(spec->op, "avg") != 0 && strcmp(spec->op, "sum") != 0)
    {
        POOL_REASON("unsupported pool op '%s'", spec->op);
        return false;
    }

    p = &spec->problem;
    /* if p.Y <= 0 or p.X <= 0: reject. */
    if(p->Y <= 0 || p->X <= 0)
    {
        POOL_REASON("window dims must be positive (Y=%d, X=%d)", p->Y, p->X);
        return false;
    }
    /* if p.Ho <= 0 or p.Wo <= 0: reject. */
    Ho = rocke_pooling_problem_ho(p);
    Wo = rocke_pooling_problem_wo(p);
    if(Ho <= 0 || Wo <= 0)
    {
        POOL_REASON("output spatial dims must be positive (Ho=%d, Wo=%d); check "
                    "pad/stride/window vs H/W",
                    Ho,
                    Wo);
        return false;
    }
    /* if spec.vec >= 2 and p.C % spec.vec != 0: reject. */
    if(spec->vec >= 2 && (p->C % spec->vec) != 0)
    {
        POOL_REASON("vec=%d requires C (%d) divisible by vec; fall back to vec=1 "
                    "for partial-C cases",
                    spec->vec,
                    p->C);
        return false;
    }

    POOL_REASON("ok");
    return true;
#undef POOL_REASON
}

/* ===================================================================== *
 * _make_input_descriptor / _neutral_value / _combine
 * ===================================================================== */

/* _make_input_descriptor(p): X_nhwc naive [N,H,W,C] (coords n,hi,wi,c) with two
 * embed transforms encoding the conv-style affine spatial map. */
static rocke_tensor_descriptor_t* pool_make_input_descriptor(rocke_ir_builder_t* b,
                                                             const rocke_pooling_problem_t* p)
{
    int lengths[4];
    const char* coord_names[4];
    rocke_tensor_descriptor_t* base;
    rocke_transform_t* e_h;
    rocke_transform_t* e_w;
    const rocke_transform_t* chain[2];
    const char* upper_h[2];
    const char* upper_w[2];
    int strides_h[2];
    int strides_w[2];

    lengths[0] = p->N;
    lengths[1] = p->H;
    lengths[2] = p->W;
    lengths[3] = p->C;
    coord_names[0] = "n";
    coord_names[1] = "hi";
    coord_names[2] = "wi";
    coord_names[3] = "c";

    base = rocke_tensor_descriptor_naive(b, "X_nhwc", lengths, 4, /*strides*/ NULL, coord_names, 4);
    if(base == NULL)
    {
        return NULL;
    }

    /* embed(upper=["ho","y"], into="hi", strides=[sH,dH], offset=-pH, lo=0,
     *       hi=p.H) */
    upper_h[0] = "ho";
    upper_h[1] = "y";
    strides_h[0] = p->sH;
    strides_h[1] = p->dH;
    e_h = rocke_embed_bounded(b, upper_h, 2, "hi", strides_h, -p->pH, 0, p->H);

    /* embed(upper=["wo","x"], into="wi", strides=[sW,dW], offset=-pW, lo=0,
     *       hi=p.W) */
    upper_w[0] = "wo";
    upper_w[1] = "x";
    strides_w[0] = p->sW;
    strides_w[1] = p->dW;
    e_w = rocke_embed_bounded(b, upper_w, 2, "wi", strides_w, -p->pW, 0, p->W);

    if(e_h == NULL || e_w == NULL)
    {
        return NULL;
    }
    chain[0] = e_h;
    chain[1] = e_w;
    return rocke_tensor_descriptor_transform(b, base, chain, 2);
}

/* _neutral_value(b, op): f32 neutral element. */
static rocke_value_t* pool_neutral_value(rocke_ir_builder_t* b, const char* op)
{
    if(strcmp(op, "max") == 0)
    {
        return rocke_b_const_f32(b, -INFINITY); /* float("-inf") */
    }
    /* sum / avg -> 0.0 */
    return rocke_b_const_f32(b, 0.0);
}

/* _combine(b, op, acc, x): reduction step in f32. */
static rocke_value_t*
    pool_combine(rocke_ir_builder_t* b, const char* op, rocke_value_t* acc, rocke_value_t* x)
{
    if(strcmp(op, "max") == 0)
    {
        return rocke_b_fmax(b, acc, x);
    }
    /* sum / avg -> fadd */
    return rocke_b_fadd(b, acc, x);
}

/* ===================================================================== *
 * Store-epilogue helper definitions.
 * ===================================================================== */

/* Python helpers/tensor_view.py::_dtype_elem_bytes (the buffer-store subset). */
static int rocke_pool_dtype_elem_bytes(const rocke_type_t* dtype)
{
    const char* n = (dtype != NULL) ? dtype->name : NULL;
    if(n == NULL)
        return 0;
    if(strcmp(n, "f16") == 0 || strcmp(n, "bf16") == 0)
        return 2;
    if(strcmp(n, "f32") == 0 || strcmp(n, "i32") == 0)
        return 4;
    if(strcmp(n, "i64") == 0)
        return 8;
    return 0;
}

/* make_buffer_resource(b, ptr, num_bytes=...): rsrc = b.buffer_rsrc(ptr,
 * num_bytes); soffset defaults to const_i32(0). Mirrors the Python op order:
 * the buffer_rsrc is emitted first, then the const_i32(0) soffset. */
static rocke_buffer_resource_t*
    rocke_make_buffer_resource(rocke_ir_builder_t* b, rocke_value_t* ptr, rocke_value_t* num_bytes)
{
    rocke_buffer_resource_t* r;
    if(b == NULL)
        return NULL;
    r = (rocke_buffer_resource_t*)rocke_arena_calloc(&b->arena, sizeof(rocke_buffer_resource_t));
    if(r == NULL)
        return NULL;
    r->rsrc = rocke_b_buffer_rsrc(b, ptr, num_bytes);
    r->soffset = rocke_b_const_i32(b, 0);
    r->num_bytes = 0; /* Python keeps the int only when num_bytes is a host int */
    return r;
}

/* make_buffer_view(rsrc, shape, dtype): packed row-major buffer TensorView.
 * Pure host-side struct construction (TensorDescriptor.packed) -- emits NO IR,
 * matching Python make_buffer_view. */
static rocke_buffer_view_t* rocke_make_buffer_view(rocke_ir_builder_t* b,
                                                   rocke_buffer_resource_t* rsrc,
                                                   const int* lengths,
                                                   int n_lengths,
                                                   const rocke_type_t* dtype)
{
    rocke_buffer_view_t* v;
    if(b == NULL || rsrc == NULL)
        return NULL;
    if(n_lengths <= 0 || n_lengths > ROCKE_POOL_TV_MAX_RANK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "make_buffer_view: bad rank");
        return NULL;
    }
    v = (rocke_buffer_view_t*)rocke_arena_calloc(&b->arena, sizeof(rocke_buffer_view_t));
    if(v == NULL)
        return NULL;
    v->rsrc = rsrc;
    v->rank = n_lengths;
    v->dtype = dtype;
    {
        int i;
        int s = 1;
        for(i = 0; i < n_lengths; ++i)
            v->shape[i] = lengths[i];
        /* TensorDescriptor.packed: row-major, trailing stride 1. */
        for(i = n_lengths - 1; i >= 0; --i)
        {
            v->strides[i] = s;
            s *= lengths[i];
        }
    }
    return v;
}

/* TensorView.tile(lengths, origin): pure host-side TileWindow -- emits NO IR.
 * Python __post_init__ enforces tile rank == view rank and origin rank == view
 * rank. */
static void* rocke_buffer_view_tile(rocke_ir_builder_t* b,
                                    rocke_buffer_view_t* view,
                                    const int* lengths,
                                    int n_lengths,
                                    rocke_value_t* const* origin,
                                    int n_origin)
{
    rocke_buffer_window_t* w;
    int i;
    if(b == NULL || view == NULL)
        return NULL;
    if(n_lengths != view->rank || n_origin != view->rank || n_lengths > ROCKE_POOL_TV_MAX_RANK)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "TensorView.tile: rank mismatch");
        return NULL;
    }
    w = (rocke_buffer_window_t*)rocke_arena_calloc(&b->arena, sizeof(rocke_buffer_window_t));
    if(w == NULL)
        return NULL;
    w->view = view;
    w->rank = n_lengths;
    for(i = 0; i < n_lengths; ++i)
    {
        w->lengths[i] = lengths[i];
        w->origin[i] = origin[i];
    }
    return w;
}

/* TensorDescriptor.offset for a packed buffer view (helpers/tensor_view.py
 * TensorDescriptor.offset): sum(idx * stride), stride-1 term elided, exactly
 * the same mul/add op order. */
static rocke_value_t* rocke_pool_buffer_offset(rocke_ir_builder_t* b,
                                               const rocke_buffer_view_t* v,
                                               rocke_value_t* const* indices,
                                               int num_indices)
{
    int i;
    rocke_value_t* off = NULL;
    if(b == NULL || v == NULL || num_indices != v->rank)
        return NULL;
    for(i = 0; i < v->rank; ++i)
    {
        rocke_value_t* term;
        if(v->strides[i] == 1)
            term = indices[i];
        else
            term = rocke_b_mul(b, indices[i], rocke_b_const_i32(b, v->strides[i]));
        off = (off == NULL) ? term : rocke_b_add(b, off, term);
    }
    return off != NULL ? off : rocke_b_const_i32(b, 0);
}

/* Y_lengths[i] for a distribution encoding (Hs for major>=1, Rs for major==0).
 * Mirrors the Python TileDistribution.Y_lengths property. */
static int rocke_pool_y_length(const rocke_tile_distribution_encoding_t* e, int yi)
{
    int major = e->Ys_major[yi];
    int minor = e->Ys_minor[yi];
    if(major == 0)
        return e->Rs[minor];
    return e->Hs[major - 1].levels[minor];
}

/* TileDistribution.y_to_linear: row-major linearisation of a Y tuple. */
static int
    rocke_pool_y_to_linear(const rocke_tile_distribution_encoding_t* e, const int* y, int n_y)
{
    int off = 0;
    int i;
    for(i = 0; i < n_y; ++i)
        off = off * rocke_pool_y_length(e, i) + y[i];
    return off;
}

/* StaticDistributedTensor.set(y, value): store at storage[y_to_linear(y)].
 * Pure host-side bookkeeping -- emits NO IR. */
static void rocke_static_distributed_tensor_set(rocke_ir_builder_t* b,
                                                rocke_static_distributed_tensor_t* dt,
                                                const int* y,
                                                int n_y,
                                                rocke_value_t* value)
{
    int off;
    (void)b;
    if(dt == NULL || dt->distribution == NULL || dt->distribution->encoding == NULL)
        return;
    off = rocke_pool_y_to_linear(dt->distribution->encoding, y, n_y);
    if(off < 0 || off >= dt->num_storage)
        return;
    dt->storage[off] = value;
}

/* make_load_store_traits picker for the pooling output distribution
 * (Hs=((VEC,),), one Y mapped to the innermost H level). For this single-Y,
 * single-level encoding the stride-1 candidate is Y0 with length VEC, so
 * vector_dim_y == 0 and scalar_per_vector == largest power-of-two <= 8 dividing
 * VEC == VEC (VEC in {1,2,4,8}). iterate_accesses yields a single base (0).
 * Mirrors helpers/distribution.py::make_load_store_traits. */
static int rocke_pool_scalar_per_vector(const rocke_tile_distribution_encoding_t* e)
{
    int full_len;
    int spv;
    /* _y_x_stride for Y0: minor 0 of the innermost H level -> stride 1 always
     * for a single-level H, so Y0 is the (only) candidate. */
    full_len = rocke_pool_y_length(e, 0);
    spv = full_len < 8 ? full_len : 8;
    while(spv > 1 && (full_len % spv != 0 || (spv & (spv - 1)) != 0))
        spv /= 2;
    if(spv < 1)
        spv = 1;
    return spv;
}

/* store_tile(b, window, distributed, ps=[]): port of helpers/distribution.py
 * store_tile for the f16/bf16 buffer-window path used by pooling. Single
 * access (iterate_accesses yields one base for this distribution):
 *   x = calculate_x(b, ys=[const_i32(0)], ps=[])
 *   scalars = [distributed.get([k]) for k in range(scalar_per_vector)]
 *   store_vec_from_f32(window, *x, values=scalars)
 * store_vec_from_f32 -> cast_f32_to(...) then store_scalar (len==1) or
 * vec_pack + store_vec; the buffer branch issues raw_ptr_buffer_store and
 * rejects bf16 with the Python NotImplementedError text. */
static void rocke_store_tile(rocke_ir_builder_t* b,
                             void* window,
                             rocke_static_distributed_tensor_t* dt,
                             rocke_value_t* const* ps,
                             int n_ps)
{
    const rocke_buffer_window_t* w = (const rocke_buffer_window_t*)window;
    const rocke_tile_distribution_t* dist;
    const rocke_tile_distribution_encoding_t* enc;
    const rocke_type_t* dt_ty;
    const char* tyname;
    int spv;
    int num_x;
    int elem_bytes;
    rocke_value_t* ys[ROCKE_POOL_TV_MAX_RANK];
    rocke_value_t* x_coords[ROCKE_POOL_TV_MAX_RANK];
    rocke_value_t* gidx[ROCKE_POOL_TV_MAX_RANK];
    rocke_value_t* scalars[8];
    rocke_value_t* casts[8];
    rocke_value_t* off_elem;
    rocke_value_t* byte_off;
    int i;
    (void)ps;
    (void)n_ps;

    if(b == NULL || w == NULL || w->view == NULL || dt == NULL || dt->distribution == NULL)
        return;
    dist = dt->distribution;
    enc = dist->encoding;
    dt_ty = w->view->dtype;
    tyname = (dt_ty != NULL) ? dt_ty->name : NULL;

    /* store_tile dtype guard. Faithful mirror of tensor_view.py:
     * TensorView.store_vec_from_f32, which raises NotImplementedError(
     *   f"store_vec_from_f32 not wired for {dtype}; cast manually and use
     *    store_vec") for any dtype outside {f16, bf16}. Byte-faithful reject
     * behaviour: only dtypes Python also rejects reach here, message verbatim. */
    if(tyname == NULL || (strcmp(tyname, "f16") != 0 && strcmp(tyname, "bf16") != 0))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_NOTIMPL,
                        "store_vec_from_f32 not wired for %s; "
                        "cast manually and use store_vec",
                        tyname ? tyname : "<null>");
        return;
    }

    /* Single access: ys = [const_i32(0)] per Y dim, vector dim fixed at 0. */
    spv = rocke_pool_scalar_per_vector(enc);
    for(i = 0; i < enc->num_Y; ++i)
        ys[i] = rocke_b_const_i32(b, 0);
    if(!rocke_tile_distribution_calculate_x(b,
                                            dist,
                                            ys,
                                            enc->num_Y,
                                            /*ps*/ NULL,
                                            /*ps_counts*/ NULL,
                                            /*num_ps*/ 0,
                                            x_coords,
                                            ROCKE_POOL_TV_MAX_RANK))
        return;
    num_x = enc->num_X;

    /* scalars[k] = distributed.get([k]) (vector_dim_y == 0). */
    for(i = 0; i < spv; ++i)
    {
        int yfull[ROCKE_POOL_TV_MAX_RANK];
        int j;
        int off;
        for(j = 0; j < enc->num_Y; ++j)
            yfull[j] = 0;
        yfull[0] = i; /* vector_dim_y == 0 */
        off = rocke_pool_y_to_linear(enc, yfull, enc->num_Y);
        scalars[i] = (off >= 0 && off < dt->num_storage) ? dt->storage[off] : NULL;
    }

    /* store_vec_from_f32 (Python): cast every f32 scalar to dtype, then -- for
     * the vec case -- vec_pack, ALL before the window's global-index / offset
     * arithmetic (which store_scalar / store_vec emit). */
    for(i = 0; i < spv; ++i)
        casts[i] = rocke_b_cast_f32_to(b, scalars[i], dt_ty);

    if(spv == 1)
    {
        /* store_scalar(window, *x): _global_indices then desc.offset then
         * byte_off then raw_ptr_buffer_store.i16 (f16 only; bf16 rejects). */
        if(num_x != w->rank)
            return;
        for(i = 0; i < num_x; ++i)
            gidx[i] = rocke_b_add(b, w->origin[i], x_coords[i]);
        off_elem = rocke_pool_buffer_offset(b, w->view, gidx, num_x);
        elem_bytes = rocke_pool_dtype_elem_bytes(dt_ty);
        byte_off = rocke_b_mul(b, off_elem, rocke_b_const_i32(b, elem_bytes));
        if(strcmp(tyname, "f16") == 0)
        {
            rocke_b_buffer_store_f16(
                b, w->view->rsrc->rsrc, byte_off, w->view->rsrc->soffset, casts[0]);
        }
        else
        {
            /* Faithful mirror of tensor_view.py:store_scalar_at, which wires
             * only f16 in buffer space and raises NotImplementedError(
             *   f"buffer store_scalar_at not wired for dtype {dtype}") otherwise
             * (bf16 passes the store_vec_from_f32 gate above, then rejects here,
             * exactly as in Python). Message text matched verbatim. */
            rocke_i_set_err(
                b, ROCKE_ERR_NOTIMPL, "buffer store_scalar_at not wired for dtype %s", tyname);
        }
        return;
    }

    /* store_vec(window, *x): vec_pack first (inside store_vec_from_f32), THEN
     * _global_indices / desc.offset / byte_off / raw_ptr_buffer_store_vN. */
    {
        rocke_value_t* packed = rocke_b_vec_pack(b, casts, spv, dt_ty);
        if(num_x != w->rank)
            return;
        for(i = 0; i < num_x; ++i)
            gidx[i] = rocke_b_add(b, w->origin[i], x_coords[i]);
        off_elem = rocke_pool_buffer_offset(b, w->view, gidx, num_x);
        elem_bytes = rocke_pool_dtype_elem_bytes(dt_ty);
        byte_off = rocke_b_mul(b, off_elem, rocke_b_const_i32(b, elem_bytes));
        if(strcmp(tyname, "f16") == 0)
        {
            rocke_b_buffer_store_vN_f16(
                b, w->view->rsrc->rsrc, byte_off, w->view->rsrc->soffset, packed, spv / 2);
        }
        else
        {
            /* Faithful mirror of tensor_view.py:store_vec_at, which wires only
             * f16 in buffer space and raises NotImplementedError(
             *   f"buffer store_vec_at not wired for dtype {dtype}") otherwise.
             * Message text matched verbatim to Python. */
            rocke_i_set_err(
                b, ROCKE_ERR_NOTIMPL, "buffer store_vec_at not wired for dtype %s", tyname);
        }
    }
}

/* ===================================================================== *
 * build_pooling2d
 * ===================================================================== */

rocke_kernel_def_t* rocke_build_pooling2d(rocke_ir_builder_t* b,
                                          const rocke_pooling2d_spec_t* spec,
                                          const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        const rocke_pooling_problem_t* p;
        int VEC;
        const rocke_type_t* io_ty;
        rocke_param_opts_t opts;
        const rocke_type_t* ptr_ty;
        rocke_value_t* X;
        rocke_value_t* Y;
        rocke_value_t* X_bytes;
        rocke_value_t* Y_bytes;
        rocke_tensor_descriptor_t* in_desc;
        int C_v, total_out_v;
        rocke_value_t* c0;
        rocke_value_t* c_elem_bytes;
        rocke_value_t* c_vec;
        rocke_value_t* oob_sentinel;
        rocke_value_t* tid;
        rocke_value_t* bid;
        rocke_value_t* out_idx_v;
        rocke_tensor_descriptor_t* out_unmerge_base;
        rocke_tensor_descriptor_t* out_unmerge_desc;
        rocke_transform_t* um;
        const rocke_transform_t* um_chain[1];
        int um_lengths[4];
        const char* um_coord_names[4];
        const char* um_into[4];
        int um_dims[4];
        /* unmerge_lower output map */
        const char* dec_names[16];
        rocke_value_t* dec_values[16];
        int n_dec;
        const char* in_names[1];
        rocke_value_t* in_values[1];
        rocke_value_t* n_val = NULL;
        rocke_value_t* ho_val = NULL;
        rocke_value_t* wo_val = NULL;
        rocke_value_t* c_v_val = NULL;
        rocke_value_t* c_base;
        rocke_value_t* x_rsrc;
        rocke_value_t* neutral;
        rocke_value_t* acc_list[8];
        rocke_value_t* valid_count = NULL;
        int y_i, x_i, k, i;
        /* store side */
        int out_total;
        rocke_buffer_resource_t* out_rsrc;
        rocke_buffer_view_t* out_view;
        int out_lengths[1];
        rocke_h_row_t out_hs[1];
        int out_h_levels[1];
        int out_ys_major[1];
        int out_ys_minor[1];
        rocke_tile_distribution_encoding_t* out_enc;
        rocke_tile_distribution_t* out_dist;
        rocke_value_t* out_origin;
        void* out_window;
        rocke_value_t* origin_arr[1];
        rocke_static_distributed_tensor_t* out_dt;
        int yidx[1];

        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(arch == NULL)
        {
            arch = "gfx950";
        }

        /* ok, why = is_valid_spec(spec, arch); if not ok: raise ValueError. */
        {
            char reason[256];
            if(!rocke_pooling2d_is_valid_spec(spec, arch, reason, sizeof(reason)))
            {
                rocke_i_set_err(b, ROCKE_ERR_VALUE, "invalid pooling2d spec: %s", reason);
                return NULL;
            }
        }

        p = &spec->problem;
        VEC = spec->vec;

        /* io_ty = io_ir_type(spec.dtype) */
        io_ty = rocke_b_io_ir_type(b, spec->dtype);
        if(io_ty == NULL)
        {
            return NULL;
        }

        /* b.kernel.attrs["max_workgroup_size"] = spec.block_size */
        rocke_attr_set_int(b, &b->kernel->attrs, "max_workgroup_size", spec->block_size);

        /* X = b.param("X", PtrType(io_ty,"global"), noalias=True, readonly=True,
         *             align=16) */
        ptr_ty = rocke_ptr_type(b, io_ty, "global");
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.readonly = true;
        opts.readonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        X = rocke_b_param(b, "X", ptr_ty, &opts);

        /* Y = b.param("Y", PtrType(io_ty,"global"), noalias=True, writeonly=True,
         *             align=16) */
        memset(&opts, 0, sizeof(opts));
        opts.noalias = true;
        opts.noalias_set = true;
        opts.writeonly = true;
        opts.writeonly_set = true;
        opts.align = 16;
        opts.align_set = true;
        Y = rocke_b_param(b, "Y", ptr_ty, &opts);

        /* X_bytes = b.param("X_bytes", I32); Y_bytes = b.param("Y_bytes", I32) */
        X_bytes = rocke_b_param(b, "X_bytes", rocke_i32(), NULL);
        Y_bytes = rocke_b_param(b, "Y_bytes", rocke_i32(), NULL);

        /* in_desc = _make_input_descriptor(p) */
        in_desc = pool_make_input_descriptor(b, p);
        if(in_desc == NULL)
        {
            return NULL;
        }

        /* C_v = p.C // VEC ; total_out_v = N * Ho * Wo * C_v */
        C_v = p->C / VEC;
        total_out_v = p->N * rocke_pooling_problem_ho(p) * rocke_pooling_problem_wo(p) * C_v;

        /* c0 = const_i32(0); c_elem_bytes = const_i32(2); c_vec = const_i32(VEC);
         * oob_sentinel = const_i32((1<<31)-1) */
        c0 = rocke_b_const_i32(b, 0);
        c_elem_bytes = rocke_b_const_i32(b, 2);
        c_vec = rocke_b_const_i32(b, VEC);
        oob_sentinel = rocke_b_const_i32(b, (int64_t)((1u << 31) - 1u));

        /* tid = thread_id_x(); bid = block_id_x();
         * out_idx_v = add(mul(bid, const_i32(block_size)), tid) */
        tid = rocke_b_thread_id_x(b);
        bid = rocke_b_block_id_x(b);
        out_idx_v
            = rocke_b_add(b, rocke_b_mul(b, bid, rocke_b_const_i32(b, spec->block_size)), tid);

        /* out_unmerge_desc = TensorDescriptor.naive("pool_out_m",
         *     lengths=[N,Ho,Wo,C_v], dtype=F16, coord_names=["n","ho","wo","c_v"])
         *   .transform(unmerge_magic("out_idx_v", into=["n","ho","wo","c_v"],
         *                            dims=[N,Ho,Wo,C_v])) */
        um_lengths[0] = p->N;
        um_lengths[1] = rocke_pooling_problem_ho(p);
        um_lengths[2] = rocke_pooling_problem_wo(p);
        um_lengths[3] = C_v;
        um_coord_names[0] = "n";
        um_coord_names[1] = "ho";
        um_coord_names[2] = "wo";
        um_coord_names[3] = "c_v";
        out_unmerge_base = rocke_tensor_descriptor_naive(
            b, "pool_out_m", um_lengths, 4, NULL, um_coord_names, 4);
        if(out_unmerge_base == NULL)
        {
            return NULL;
        }
        um_into[0] = "n";
        um_into[1] = "ho";
        um_into[2] = "wo";
        um_into[3] = "c_v";
        um_dims[0] = p->N;
        um_dims[1] = rocke_pooling_problem_ho(p);
        um_dims[2] = rocke_pooling_problem_wo(p);
        um_dims[3] = C_v;
        um = rocke_unmerge_magic(b, "out_idx_v", um_into, 4, um_dims);
        if(um == NULL)
        {
            return NULL;
        }
        um_chain[0] = um;
        out_unmerge_desc = rocke_tensor_descriptor_transform(b, out_unmerge_base, um_chain, 1);
        if(out_unmerge_desc == NULL)
        {
            return NULL;
        }

        /* decoded = out_unmerge_desc.unmerge_lower(b, out_idx_v=out_idx_v) */
        in_names[0] = "out_idx_v";
        in_values[0] = out_idx_v;
        n_dec = rocke_tensor_descriptor_unmerge_lower(
            b,
            out_unmerge_desc,
            in_names,
            in_values,
            1,
            dec_names,
            dec_values,
            (int)(sizeof(dec_names) / sizeof(dec_names[0])));
        if(n_dec < 0)
        {
            return NULL;
        }
        /* n_val/ho_val/wo_val/c_v_val = decoded["n"/"ho"/"wo"/"c_v"] */
        for(i = 0; i < n_dec; ++i)
        {
            if(strcmp(dec_names[i], "n") == 0)
            {
                n_val = dec_values[i];
            }
            else if(strcmp(dec_names[i], "ho") == 0)
            {
                ho_val = dec_values[i];
            }
            else if(strcmp(dec_names[i], "wo") == 0)
            {
                wo_val = dec_values[i];
            }
            else if(strcmp(dec_names[i], "c_v") == 0)
            {
                c_v_val = dec_values[i];
            }
        }
        if(n_val == NULL || ho_val == NULL || wo_val == NULL || c_v_val == NULL)
        {
            rocke_i_set_err(b, ROCKE_ERR_KEY, "pooling2d: unmerge_lower missing decoded coord");
            return NULL;
        }

        /* c_base = mul(c_v_val, c_vec) if VEC > 1 else c_v_val */
        c_base = (VEC > 1) ? rocke_b_mul(b, c_v_val, c_vec) : c_v_val;

        /* x_rsrc = b.buffer_rsrc(X, X_bytes) */
        x_rsrc = rocke_b_buffer_rsrc(b, X, X_bytes);

        /* neutral = _neutral_value(b, op); acc_list = [neutral]*VEC */
        neutral = pool_neutral_value(b, spec->op);
        for(k = 0; k < VEC; ++k)
        {
            acc_list[k] = neutral;
        }
        /* valid_count = const_f32(0.0) if op == "avg" else None */
        if(strcmp(spec->op, "avg") == 0)
        {
            valid_count = rocke_b_const_f32(b, 0.0);
        }

        /* for y_i in range(Y): for x_i in range(X): ... window reduction */
        for(y_i = 0; y_i < p->Y; ++y_i)
        {
            rocke_value_t* c_y = rocke_b_const_i32(b, y_i);
            for(x_i = 0; x_i < p->X; ++x_i)
            {
                rocke_value_t* c_x = rocke_b_const_i32(b, x_i);
                rocke_value_t* off = NULL;
                rocke_value_t* valid = NULL;
                rocke_value_t* off_bytes;
                rocke_value_t* safe_in_off;
                const char* off_names[6];
                rocke_value_t* off_vals[6];

                /* off, valid = in_desc.offset(b, n=n_val, ho=ho_val, y=c_y,
                 *                             wo=wo_val, x=c_x, c=c_base) */
                off_names[0] = "n";
                off_vals[0] = n_val;
                off_names[1] = "ho";
                off_vals[1] = ho_val;
                off_names[2] = "y";
                off_vals[2] = c_y;
                off_names[3] = "wo";
                off_vals[3] = wo_val;
                off_names[4] = "x";
                off_vals[4] = c_x;
                off_names[5] = "c";
                off_vals[5] = c_base;
                if(!rocke_transforms_descriptor_offset(
                       b, in_desc, off_names, off_vals, 6, &off, &valid))
                {
                    return NULL;
                }

                /* off_bytes = mul(off, c_elem_bytes) */
                off_bytes = rocke_b_mul(b, off, c_elem_bytes);
                /* safe_in_off = select(valid, off_bytes, oob_sentinel) if valid is
                 *               not None else off_bytes */
                safe_in_off = (valid != NULL) ? rocke_b_select(b, valid, off_bytes, oob_sentinel)
                                              : off_bytes;

                if(VEC >= 2)
                {
                    /* loaded_vec = buffer_load_vN_f16(x_rsrc, safe_in_off, c0,
                     *                                 dwords=VEC//2) */
                    rocke_value_t* loaded_vec
                        = rocke_b_buffer_load_vN_f16(b, x_rsrc, safe_in_off, c0, VEC / 2);
                    for(k = 0; k < VEC; ++k)
                    {
                        /* raw = vec_extract(loaded_vec, k);
                         * loaded_f32 = cast_to_f32(raw) */
                        rocke_value_t* raw = rocke_b_vec_extract(b, loaded_vec, k);
                        rocke_value_t* loaded_f32 = rocke_b_cast_to_f32(b, raw);
                        /* masked = select(valid, loaded_f32, neutral) if valid else
                         *          loaded_f32 */
                        rocke_value_t* masked = (valid != NULL)
                                                    ? rocke_b_select(b, valid, loaded_f32, neutral)
                                                    : loaded_f32;
                        acc_list[k] = pool_combine(b, spec->op, acc_list[k], masked);
                    }
                }
                else
                {
                    /* loaded_raw = buffer_load_f16(x_rsrc, safe_in_off, c0);
                     * loaded_f32 = cast_to_f32(loaded_raw) */
                    rocke_value_t* loaded_raw = rocke_b_buffer_load_f16(b, x_rsrc, safe_in_off, c0);
                    rocke_value_t* loaded_f32 = rocke_b_cast_to_f32(b, loaded_raw);
                    rocke_value_t* masked = (valid != NULL)
                                                ? rocke_b_select(b, valid, loaded_f32, neutral)
                                                : loaded_f32;
                    acc_list[0] = pool_combine(b, spec->op, acc_list[0], masked);
                }

                /* if op == "avg": contrib = select(valid, 1.0, 0.0) if valid else 1.0;
                 *                 valid_count = fadd(valid_count, contrib) */
                if(strcmp(spec->op, "avg") == 0)
                {
                    rocke_value_t* contrib;
                    if(valid != NULL)
                    {
                        /* hoist select operands in Python's left-to-right order */
                        rocke_value_t* one_c = rocke_b_const_f32(b, 1.0);
                        rocke_value_t* zero_c = rocke_b_const_f32(b, 0.0);
                        contrib = rocke_b_select(b, valid, one_c, zero_c);
                    }
                    else
                    {
                        contrib = rocke_b_const_f32(b, 1.0);
                    }
                    valid_count = rocke_b_fadd(b, valid_count, contrib);
                }
            }
        }

        /* if op == "avg": safe_count = fmax(valid_count, 1.0);
         *                 rcp_count = rcp(safe_count);
         *                 acc_list = [fmul(acc, rcp_count) for acc in acc_list] */
        if(strcmp(spec->op, "avg") == 0)
        {
            rocke_value_t* safe_count = rocke_b_fmax(b, valid_count, rocke_b_const_f32(b, 1.0));
            rocke_value_t* rcp_count = rocke_b_rcp(b, safe_count);
            for(k = 0; k < VEC; ++k)
            {
                acc_list[k] = rocke_b_fmul(b, acc_list[k], rcp_count);
            }
        }

        /* ---- store: store_tile over the flat output [N*Ho*Wo*C] ---- */
        /* out_total = total_out_v * VEC */
        out_total = total_out_v * VEC;
        /* out_rsrc = make_buffer_resource(b, Y, num_bytes=Y_bytes) */
        out_rsrc = rocke_make_buffer_resource(b, Y, Y_bytes);
        /* out_view = make_buffer_view(out_rsrc, [out_total], io_ty) */
        out_lengths[0] = out_total;
        out_view = rocke_make_buffer_view(b, out_rsrc, out_lengths, 1, io_ty);

        /* out_enc = TileDistributionEncoding(Hs=((VEC,),), Ys2RHs_major=(1,),
         *                                    Ys2RHs_minor=(0,)) */
        out_h_levels[0] = VEC;
        out_hs[0].levels = out_h_levels;
        out_hs[0].count = 1;
        out_ys_major[0] = 1;
        out_ys_minor[0] = 0;
        out_enc = rocke_make_tile_distribution_encoding(b,
                                                        /*Rs*/ NULL,
                                                        0,
                                                        out_hs,
                                                        1,
                                                        /*Ps*/ NULL,
                                                        0,
                                                        out_ys_major,
                                                        out_ys_minor,
                                                        1);
        if(out_enc == NULL)
        {
            return NULL;
        }
        /* out_dist = make_static_tile_distribution(out_enc) */
        out_dist = rocke_make_static_tile_distribution(b, out_enc);
        if(out_dist == NULL)
        {
            return NULL;
        }

        /* out_origin = mul(out_idx_v, c_vec) if VEC > 1 else out_idx_v */
        out_origin = (VEC > 1) ? rocke_b_mul(b, out_idx_v, c_vec) : out_idx_v;

        /* out_window = out_view.tile(lengths=[VEC], origin=[out_origin]) */
        origin_arr[0] = out_origin;
        {
            int win_lengths[1];
            win_lengths[0] = VEC;
            out_window = rocke_buffer_view_tile(b, out_view, win_lengths, 1, origin_arr, 1);
        }

        /* out_dt = make_static_distributed_tensor(out_dist, dtype=io_ty) */
        out_dt = rocke_make_static_distributed_tensor(b, out_dist, io_ty);

        /* for k in range(VEC): out_dt.set([k], acc_list[k]) */
        for(k = 0; k < VEC; ++k)
        {
            yidx[0] = k;
            rocke_static_distributed_tensor_set(b, out_dt, yidx, 1, acc_list[k]);
        }

        /* store_tile(b, out_window, out_dt, ps=[]) */
        rocke_store_tile(b, out_window, out_dt, NULL, 0);

        if(!rocke_ir_builder_ok(b))
        {
            return NULL;
        }
        return b->kernel;
    });
}

rocke_kernel_def_t* rocke_build_pooling2d_new(rocke_ir_builder_t* b,
                                              const rocke_pooling2d_spec_t* spec,
                                              const char* arch)
{
    return ckc::guard_builder(b, [&]() -> rocke_kernel_def_t* {
        char name[256];
        if(b == NULL || spec == NULL)
        {
            return NULL;
        }
        if(rocke_pooling2d_kernel_name(spec, name, sizeof(name)) != ROCKE_OK)
        {
            return NULL;
        }
        if(rocke_ir_builder_init(b, name) != ROCKE_OK)
        {
            return NULL;
        }
        return rocke_build_pooling2d(b, spec, arch);
    });
}

/* ===================================================================== *
 * pooling2d_grid
 * ===================================================================== */

rocke_status_t rocke_pooling2d_grid(const rocke_pooling2d_spec_t* spec, int out[3])
{
    int total_v;
    int totals[1];
    int tiles[1];
    int vec;

    if(spec == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    /* total_v = total_out // max(vec, 1) */
    vec = spec->vec > 1 ? spec->vec : 1; /* max(vec, 1) */
    total_v = rocke_pooling_problem_total_out(&spec->problem) / vec;
    /* ceil_div_grid((total_v, block_size)) */
    totals[0] = total_v;
    tiles[0] = spec->block_size;
    return rocke_ceil_div_grid(totals, tiles, 1, out);
}

/* ===================================================================== *
 * pooling2d_signature
 * ===================================================================== */

rocke_status_t rocke_pooling2d_signature(struct rocke_arena* arena,
                                         const rocke_pooling2d_spec_t* spec,
                                         struct rocke_sig_entry* out,
                                         size_t out_cap,
                                         size_t* out_count)
{
    rocke_signature_builder_t sb;
    const rocke_sig_entry_t* items;
    size_t count;
    rocke_status_t st;
    size_t i;

    if(arena == NULL || spec == NULL || out == NULL || out_count == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    st = rocke_signature_builder_init(&sb, (rocke_arena_t*)arena);
    if(st != ROCKE_OK)
    {
        return st;
    }
    /* SignatureBuilder().ptr("X", dtype).ptr("Y", dtype)
     *   .scalar("X_bytes","i32").scalar("Y_bytes","i32").build() */
    rocke_signature_builder_ptr(&sb, "X", spec->dtype, NULL);
    rocke_signature_builder_ptr(&sb, "Y", spec->dtype, NULL);
    rocke_signature_builder_scalar(&sb, "X_bytes", "i32");
    rocke_signature_builder_scalar(&sb, "Y_bytes", "i32");
    st = rocke_signature_builder_build(&sb, &items, &count);
    if(st != ROCKE_OK)
    {
        return st;
    }
    if(count > out_cap)
    {
        return ROCKE_ERR_VALUE;
    }
    for(i = 0; i < count; ++i)
    {
        ((rocke_sig_entry_t*)out)[i] = items[i];
    }
    *out_count = count;
    return ROCKE_OK;
}

/* ===================================================================== *
 * rocke_pooling2d_lower_to_llvm -- build + lower to .ll convenience.
 * Owns and frees its own IRBuilder.
 * ===================================================================== */

rocke_status_t rocke_pooling2d_lower_to_llvm(const rocke_pooling2d_spec_t* spec,
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
        if(err != NULL && err_cap > 0)
        {
            snprintf(err, err_cap, "lower_to_llvm: null spec/out");
        }
        return ROCKE_ERR_VALUE;
    }
    if(arch == NULL)
    {
        arch = "gfx950";
    }

    kernel = rocke_build_pooling2d_new(&b, spec, arch);
    if(kernel == NULL)
    {
        st = rocke_ir_builder_status(&b);
        if(err != NULL && err_cap > 0)
        {
            const char* m = rocke_ir_builder_error(&b);
            if(m == NULL)
            {
                m = "build_pooling2d failed";
            }
            snprintf(err, err_cap, "%s", m);
        }
        rocke_ir_builder_free(&b);
        return (st == ROCKE_OK) ? ROCKE_ERR_VALUE : st;
    }

    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_ll, err, err_cap);
    rocke_ir_builder_free(&b);
    return st;
}
