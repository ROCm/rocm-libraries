/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_hiptensor_aot.h -- C++ mirror of the shape-generic hipTensor
 * kernels of library/kernels/common/hiptensor_aot (the kernels of hipTensor's
 * kernel pack and their JIT forms):
 *
 *   ew.EwSpec / build_ew                    rocke_htaot_ew_spec_t / rocke_build_htaot_ew
 *   reduce.ReduceSpec / build_reduce        rocke_htaot_reduce_spec_t / rocke_build_htaot_reduce
 *   rreduce.RreduceSpec / build_rreduce     rocke_htaot_rreduce_spec_t / rocke_build_htaot_rreduce
 *   rreduce.RreduceFinalSpec / build_rreduce_final
 *                                           rocke_htaot_rreduce_final_spec_t /
 *                                           rocke_build_htaot_rreduce_final
 *   jit_ew.EwJitSpec / build_ew_jit         rocke_htaot_ew_jit_spec_t / rocke_build_htaot_ew_jit
 *   ewt.EwtSpec / build_ewt                 rocke_htaot_ewt_spec_t / rocke_build_htaot_ewt
 *   ewf.EwfSpec / build_ewf                 rocke_htaot_ewf_spec_t / rocke_build_htaot_ewf
 *   ewp.EwpSpec / build_ewp                 rocke_htaot_ewp_spec_t / rocke_build_htaot_ewp
 *
 * Every builder makes the same builder calls in the same order as its Python
 * twin, so both engines lower every kernel to the same LLVM IR (enforced by the
 * hiptensor_aot_* parity families under library/tests/parity). The host-side
 * planning rules (host.py, gett_host.py, rreduce_host.py, jit_ew.jit_class_of,
 * ewt.ewt_plan)
 * live in hipTensor's C++ planner, not here.
 *
 * Dtype strings: "f16", "bf16", "f32", "f64"; NULL (or "") for an absent B / C.
 * Builders initialise the builder `b` (rocke_ir_builder_init) and return its
 * kernel, or NULL with the builder's error set (spec errors are ROCKE_ERR_VALUE
 * with Python's message).
 */
#ifndef ROCKE_INSTANCE_HIPTENSOR_AOT_H
#define ROCKE_INSTANCE_HIPTENSOR_AOT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel ABI names (Python EW_ABI, REDUCE_ABI, EW_JIT_ABI, EWV_ABI, EWT_ABI; EWF_ABI below). */
#define ROCKE_HTAOT_EW_ABI "hiptensor-aot-ew/v1"
#define ROCKE_HTAOT_REDUCE_ABI "hiptensor-aot-reduce/v1"
#define ROCKE_HTAOT_EW_JIT_ABI "hiptensor-jit-ew/v1"
/* the vector classes of the elementwise kernel (rocke_htaot_ewv_kernel_name) */
#define ROCKE_HTAOT_EWV_ABI "hiptensor-aot-ewv/v1"
#define ROCKE_HTAOT_EWT_ABI "hiptensor-aot-ewt/v2"

/* common.MAX_RANK */
#define ROCKE_HTAOT_MAX_RANK 8

/* ------------------------------------------------------------------ ew */

/* Python ew.EwSpec (frozen dataclass; defaults block 256). */
typedef struct rocke_htaot_ew_spec
{
    int arity; /* 1 (permute / pack), 2 (binary), 3 (trinary) */
    const char* a;
    const char* d;
    const char* b; /* arity 3 */
    const char* c; /* arity >= 2 */
    bool cplx;
    bool idx64;
    int block;
    bool math;
} rocke_htaot_ew_spec_t;

rocke_htaot_ew_spec_t rocke_htaot_ew_spec_default(void);
/* EwSpec.check(); false with Python's message in `reason`. */
bool rocke_htaot_ew_check(const rocke_htaot_ew_spec_t* s, char* reason, size_t cap);
/* EwSpec.kernel_name(); ROCKE_ERR_VALUE when `cap` is too small. */
rocke_status_t rocke_htaot_ew_kernel_name(const rocke_htaot_ew_spec_t* s, char* out, size_t cap);
rocke_kernel_def_t* rocke_build_htaot_ew(rocke_ir_builder_t* b, const rocke_htaot_ew_spec_t* s);

/* -------------------------------------------------------------- reduce */

/* Python reduce.ReduceSpec. mode: "row" | "col" | "final". */
typedef struct rocke_htaot_reduce_spec
{
    const char* mode;
    const char* a;
    const char* d;
    bool idx64;
    int block;
    bool math;
} rocke_htaot_reduce_spec_t;

rocke_htaot_reduce_spec_t rocke_htaot_reduce_spec_default(void);
rocke_status_t
    rocke_htaot_reduce_kernel_name(const rocke_htaot_reduce_spec_t* s, char* out, size_t cap);
rocke_kernel_def_t* rocke_build_htaot_reduce(rocke_ir_builder_t* b,
                                             const rocke_htaot_reduce_spec_t* s);

/* ------------------------------------------------------------- rreduce */

/* Python rreduce.RREDUCE_ABI */
#define ROCKE_HTAOT_RREDUCE_ABI "hiptensor-aot-rreduce/v1"

/* Python rreduce.RreduceSpec (a tuned reduction class, the geometry as kernel
 * arguments). mode: "row" | "col"; op: "add" | "mul" | "max" | "min"; nr: the
 * reduced slots (1, 2, 3, 4 or 8). */
typedef struct rocke_htaot_rreduce_spec
{
    const char* mode;
    const char* a;
    const char* d;
    const char* op;
    int nr;
    int block;
    int vec;
    int lanes;
    int unroll;
    int rows; /* col: reduction-row threads per tile (0: block / lanes) */
    bool nt; /* streaming A loads (gfx942 / gfx950) */
    bool una; /* opA other than IDENTITY */
    bool idx64;
    int wave;
} rocke_htaot_rreduce_spec_t;

rocke_htaot_rreduce_spec_t rocke_htaot_rreduce_spec_default(void);
/* RreduceSpec.check(); false with Python's message in `reason`. */
bool rocke_htaot_rreduce_check(const rocke_htaot_rreduce_spec_t* s, char* reason, size_t cap);
rocke_status_t
    rocke_htaot_rreduce_kernel_name(const rocke_htaot_rreduce_spec_t* s, char* out, size_t cap);
rocke_kernel_def_t* rocke_build_htaot_rreduce(rocke_ir_builder_t* b,
                                              const rocke_htaot_rreduce_spec_t* s);

/* Python rreduce.RreduceFinalSpec (the split finalize of a class). */
typedef struct rocke_htaot_rreduce_final_spec
{
    const char* a;
    const char* d;
    const char* op;
    int lanes;
    bool idx64;
    int block;
    int wave;
} rocke_htaot_rreduce_final_spec_t;

rocke_htaot_rreduce_final_spec_t rocke_htaot_rreduce_final_spec_default(void);
bool rocke_htaot_rreduce_final_check(const rocke_htaot_rreduce_final_spec_t* f,
                                     char* reason,
                                     size_t cap);
rocke_status_t rocke_htaot_rreduce_final_kernel_name(const rocke_htaot_rreduce_final_spec_t* f,
                                                     char* out,
                                                     size_t cap);
rocke_kernel_def_t* rocke_build_htaot_rreduce_final(rocke_ir_builder_t* b,
                                                    const rocke_htaot_rreduce_final_spec_t* f);

/* The route table of the classes (library/kernels/common/hiptensor_aot/rreduce_table.json,
 * compiled into the engine) and the functions of rreduce_host.py over it. */
const char* rocke_htaot_rreduce_table_json(void);
/* pack_classes(arch): the classes of the target's pack (its palette, then the classes of its
 * tuned problems, no duplicates). Writes up to `cap` specs (their strings are static) and
 * returns the count; -1 with the reason in `err` when the table is malformed. */
int rocke_htaot_rreduce_pack_classes(
    const char* arch, rocke_htaot_rreduce_spec_t* out, int cap, char* err, size_t err_cap);
/* final_classes(): the finalize classes of every pack. Writes up to `cap`, returns the count. */
int rocke_htaot_rreduce_final_classes(rocke_htaot_rreduce_final_spec_t* out, int cap);
/* routes(arch): the pack routes of the target's tuned problems, in the table's order
 * ("reduce.tuned.<hash>" = the launch knobs, "reduce.tuned.<hash>.key" = the canonical
 * problem), then the prototypes of the untuned choice ("reduce.proto.<i>",
 * "reduce.proto.count"). The count (-1: malformed table), then route i's name and value. */
int rocke_htaot_rreduce_route_count(const char* arch, char* err, size_t err_cap);
bool rocke_htaot_rreduce_route(
    const char* arch, int i, char* name, size_t name_cap, char* value, size_t value_cap);

/* -------------------------------------------------------------- jit_ew */

/* Python jit_ew.EwJitSpec: the generic row, the vector width and which inputs
 * (in base tensor order without D) are read with vector loads. */
typedef struct rocke_htaot_ew_jit_spec
{
    rocke_htaot_ew_spec_t base;
    int vec; /* 1, 2, 4 or 8 */
    int n_unit; /* 0 for vec 1, else the number of inputs */
    bool unit[3];
} rocke_htaot_ew_jit_spec_t;

bool rocke_htaot_ew_jit_check(const rocke_htaot_ew_jit_spec_t* s, char* reason, size_t cap);
/* EwJitSpec.kernel_name() (the JIT kernel) and ewv_kernel_name (the pack
 * class kernel). */
rocke_status_t
    rocke_htaot_ew_jit_kernel_name(const rocke_htaot_ew_jit_spec_t* s, char* out, size_t cap);
rocke_status_t
    rocke_htaot_ewv_kernel_name(const rocke_htaot_ew_jit_spec_t* s, char* out, size_t cap);
/* EwJitSpec.axes(): the geometry names in emission order; returns the count
 * (at most ROCKE_HTAOT_EW_JIT_MAX_AXES). names[i] points into static or
 * `buf`-backed storage owned by the caller's `buf` (each name < 16 bytes). */
#define ROCKE_HTAOT_EW_JIT_MAX_AXES 80
int rocke_htaot_ew_jit_axes(const rocke_htaot_ew_jit_spec_t* s,
                            char names[][16],
                            bool* is_i64,
                            int cap);
/* jit_ew.jit_vec(d): 16-byte output accesses. */
int rocke_htaot_ew_jit_vec(const char* d);
/* build_ew_jit(spec, geom): `geom` values in axes() order, or NULL for the
 * pack's class kernel (every geometry value a kernel argument). */
rocke_kernel_def_t* rocke_build_htaot_ew_jit(rocke_ir_builder_t* b,
                                             const rocke_htaot_ew_jit_spec_t* s,
                                             const int64_t* geom,
                                             int n_geom);

/* ----------------------------------------------------------------- ewt */

/* Python ewt.EwtSpec. */
typedef struct rocke_htaot_ewt_spec
{
    int arity;
    const char* a;
    const char* d;
    const char* b;
    const char* c;
    bool fold; /* several small (o, t) planes per tile */
} rocke_htaot_ewt_spec_t;

/* batch digits: every mode of a contraction's operand pack or epilogue but
 * the tile plane in one launch */
#define ROCKE_HTAOT_EWT_SLOTS_B 10
#define ROCKE_HTAOT_EWT_THREADS 256
#define ROCKE_HTAOT_EWT_TILE_O 64

rocke_status_t rocke_htaot_ewt_kernel_name(const rocke_htaot_ewt_spec_t* s, char* out, size_t cap);
int rocke_htaot_ewt_tile_t(const rocke_htaot_ewt_spec_t* s);
rocke_kernel_def_t* rocke_build_htaot_ewt(rocke_ir_builder_t* b, const rocke_htaot_ewt_spec_t* s);

/* ----------------------------------------------------------------- ewf */

/* Python ewf.EWF_ABI / EWF_UNROLL (vectors per thread). */
#define ROCKE_HTAOT_EWF_ABI "hiptensor-aot-ewf/v1"
#define ROCKE_HTAOT_EWF_UNROLL 2

/* Python ewf.EwfSpec (frozen dataclass; defaults block 256, nt false): the flat kernel of
 * tensors that are all contiguous with the same layout. */
typedef struct rocke_htaot_ewf_spec
{
    int arity;
    const char* a;
    const char* d;
    const char* b;
    const char* c;
    int block;
    bool nt; /* streaming vector loads / stores (gfx942 / gfx950) */
} rocke_htaot_ewf_spec_t;

rocke_htaot_ewf_spec_t rocke_htaot_ewf_spec_default(void);
bool rocke_htaot_ewf_check(const rocke_htaot_ewf_spec_t* s, char* reason, size_t cap);
/* EwfSpec.vec: elements per vector (16 bytes of D). */
int rocke_htaot_ewf_vec(const rocke_htaot_ewf_spec_t* s);
rocke_status_t rocke_htaot_ewf_kernel_name(const rocke_htaot_ewf_spec_t* s, char* out, size_t cap);
rocke_kernel_def_t* rocke_build_htaot_ewf(rocke_ir_builder_t* b, const rocke_htaot_ewf_spec_t* s);

/* ----------------------------------------------------------------- ewp */

/* Python ewp.EWP_ABI, SLOTS_B, MAX_VECTORS, LDS_BYTES. */
#define ROCKE_HTAOT_EWP_ABI "hiptensor-aot-ewp/v2"
#define ROCKE_HTAOT_EWP_SLOTS_B 8
#define ROCKE_HTAOT_EWP_MAX_VECTORS 32
#define ROCKE_HTAOT_EWP_LDS_BYTES 163840

/* Python ewp.EwpSpec (frozen dataclass): the tuned tiled elementwise kernel of transposing
 * layouts (tensors of one element size). */
typedef struct rocke_htaot_ewp_spec
{
    int arity;
    const char* a;
    const char* d;
    const char* b;
    const char* c;
    const char* kinds; /* one of 'T' / 'V' per input (A [B] C) */
    const char* mode; /* "t" (element transposition) or "a" (shared stride-1 mode) */
    int to; /* tile along the o run */
    int tt; /* tile along the t run */
    int tw; /* mode a: vectors (vb bytes) of the shared mode per tile */
    int block;
    int nt; /* bit 0 streaming loads, bit 1 streaming stores */
    int swz; /* XOR group of the LDS chunk index (0 / 1: none) */
    int swsh; /* chunk-index bit the swizzle key starts at */
    const char* walk; /* "o" or "t": the tile coordinate fastest across consecutive CTAs */
    int tpc; /* tiles per CTA */
    int ilp; /* phase-2 units whose loads are issued together (0: all) */
    int vb; /* bytes per vector access: 16, 8, 4 or 2 (at least the element size) */
    int rag; /* 1: ragged class (tiles may overhang the ends of the runs) */
    int tb; /* mode t: bytes per vector of the T inputs along t (0: vb) */
} rocke_htaot_ewp_spec_t;

rocke_htaot_ewp_spec_t rocke_htaot_ewp_spec_default(void);
bool rocke_htaot_ewp_check(const rocke_htaot_ewp_spec_t* s, char* reason, size_t cap);
/* EwpSpec.vec: elements per vector (vb bytes). */
int rocke_htaot_ewp_vec(const rocke_htaot_ewp_spec_t* s);
/* EwpSpec.vect: elements per vector of the T inputs along t (tb bytes, or vb). */
int rocke_htaot_ewp_vect(const rocke_htaot_ewp_spec_t* s);
/* EwpSpec.chunks(): vector chunks (vb bytes) of one LDS tile. */
int rocke_htaot_ewp_chunks(const rocke_htaot_ewp_spec_t* s);
rocke_status_t rocke_htaot_ewp_kernel_name(const rocke_htaot_ewp_spec_t* s, char* out, size_t cap);
rocke_kernel_def_t* rocke_build_htaot_ewp(rocke_ir_builder_t* b, const rocke_htaot_ewp_spec_t* s);
/* ewp_host.pack_classes(arch): the classes of the target's kernel pack in choice order; returns
 * their number (the first `cap` written to `out` when it is not NULL), 0 for a target without
 * classes, -1 for a NULL arch. */
int rocke_htaot_ewp_pack_classes(const char* arch, rocke_htaot_ewp_spec_t* out, int cap);
/* ewp_host.routes(arch): the routes of the target's tuned problems (ewp.tuned.<hash> = the class's
 * kernel name, ewp.tuned.<hash>.key = the canonical problem), sorted by name. */
int rocke_htaot_ewp_route_count(const char* arch);
bool rocke_htaot_ewp_route(const char* arch, int i, const char** name, const char** value);

/* --------------------------------------------------------------- dgett */

#define ROCKE_HTAOT_DGETT_ABI "hiptensor-aot-dgett/v1"
#define ROCKE_HTAOT_DGETT_SLOTS_MN 8
#define ROCKE_HTAOT_DGETT_SLOTS_KT 8
#define ROCKE_HTAOT_DGETT_SLOTS_KL 6
#define ROCKE_HTAOT_DGETT_SLOTS_L 4
#define ROCKE_HTAOT_DGETT_THREADS 256

/* Python dgett.DgettSpec. comp: "f16" | "bf16" | "f32" | "f64"; a_fast /
 * b_fast: "k" | "r". */
typedef struct rocke_htaot_dgett_spec
{
    const char* comp;
    bool cplx;
    const char* a_fast;
    const char* b_fast;
    bool math;
    bool idx64;
} rocke_htaot_dgett_spec_t;

/* Python dgett.DgettConfig */
typedef struct rocke_htaot_dgett_config
{
    int tm, tn, tk, wave_m, wave_n, vk, pad;
} rocke_htaot_dgett_config_t;

rocke_htaot_dgett_spec_t rocke_htaot_dgett_spec_default(void);
/* DgettSpec.check(arch); false with Python's message. */
bool rocke_htaot_dgett_check(const rocke_htaot_dgett_spec_t* s,
                             const char* arch,
                             char* reason,
                             size_t cap);
/* dgett_atom(spec, arch); NULL when the target has none. */
const struct rocke_mma_op* rocke_htaot_dgett_atom(const rocke_htaot_dgett_spec_t* s,
                                                  const char* arch);
rocke_htaot_dgett_config_t rocke_htaot_dgett_config(const rocke_htaot_dgett_spec_t* s);
rocke_status_t
    rocke_htaot_dgett_kernel_name(const rocke_htaot_dgett_spec_t* s, char* out, size_t cap);
rocke_kernel_def_t* rocke_build_htaot_dgett(rocke_ir_builder_t* b,
                                            const rocke_htaot_dgett_spec_t* s,
                                            const char* arch);

/* --------------------------------------------------------------- rgett */

#define ROCKE_HTAOT_RGETT_ABI "hiptensor-aot-rgett/v1"
#define ROCKE_HTAOT_RGETT_SLOTS_ROW 8
#define ROCKE_HTAOT_RGETT_SLOTS_KL 6
#define ROCKE_HTAOT_RGETT_SLOTS_KT 8
/* bilinear classes of the pack: C loads issued this many atom rows ahead */
#define ROCKE_HTAOT_RGETT_BILINEAR_C_PREFETCH 2

/* Python rgett.RgettSpec (the shape-generic ring GETT class). dtype: "f16" |
 * "bf16". */
typedef struct rocke_htaot_rgett_spec
{
    const char* dtype;
    bool bilinear;
    int vec_a, vec_b, vec_d;
    bool epi_pair;
    bool wide_d;
    int c_prefetch;
    int slots;
    int tile_k, dtl_stages, chiplet_wgm, chiplet_chunk_size;
} rocke_htaot_rgett_spec_t;

rocke_htaot_rgett_spec_t rocke_htaot_rgett_spec_default(void);
/* RgettSpec.check(); false with Python's message. */
bool rocke_htaot_rgett_check(const rocke_htaot_rgett_spec_t* s, char* reason, size_t cap);
rocke_status_t
    rocke_htaot_rgett_kernel_name(const rocke_htaot_rgett_spec_t* s, char* out, size_t cap);
/* RgettSpec.key(): the JIT recipe key of the class */
rocke_status_t rocke_htaot_rgett_key(const rocke_htaot_rgett_spec_t* s, char* out, size_t cap);
/* geometry_names(): the number of geometry arguments and the i-th name
 * (kernel argument order); NULL past the end. */
int rocke_htaot_rgett_geometry_count(void);
const char* rocke_htaot_rgett_geometry_name(int i);
/* rgett_classes(): the classes of hipTensor's kernel pack for one dtype
 * (NULL: f16 then bf16). Writes up to `cap` specs; returns the count. */
int rocke_htaot_rgett_classes(const char* dtype, rocke_htaot_rgett_spec_t* out, int cap);
/* build_rgett(spec, arch, geom). geom NULL: the class kernel (geometry
 * arguments); else the problem-specific JIT form with geom[i] the value of
 * geometry argument i (rocke_htaot_rgett_geometry_count() values). */
rocke_kernel_def_t* rocke_build_htaot_rgett(rocke_ir_builder_t* b,
                                            const rocke_htaot_rgett_spec_t* s,
                                            const char* arch,
                                            const int64_t* geom);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_HIPTENSOR_AOT_H */
