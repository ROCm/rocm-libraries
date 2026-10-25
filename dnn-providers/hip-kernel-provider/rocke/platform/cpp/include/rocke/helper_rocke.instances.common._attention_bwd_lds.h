/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_lds.h -- C port of the
 * LDS planning of the attention backward inner body
 * (library/kernels/common/_attention_bwd_lds.py). Host-side only: like the
 * Python module, nothing here emits IR.
 *
 *   Python (_attention_bwd_lds.py)      C (this header)
 *   ---------------------------------   ---------------------------------------
 *   LDS_LINE_BYTES, BUFFER_ALIGN_BYTES  ROCKE_ATTN_BWD_LDS_LINE_BYTES, ..._ALIGN_BYTES
 *   PHASES (PHASE_*)                    rocke_attn_bwd_lds_phase_t (bit per phase)
 *   lds_capacity_bytes                  rocke_attn_bwd_lds_capacity_bytes
 *   XtLayout (+ __post_init__)          rocke_attn_bwd_xt_layout_t (+ _init)
 *     .row_pitch_bytes / .size_bytes      rocke_attn_bwd_xt_layout_row_pitch / _size
 *     .offset / .check_vector             rocke_attn_bwd_xt_layout_offset / _check_vector
 *   RowXorLayout                        rocke_attn_bwd_row_xor_layout_t (+ _init,
 *                                       _image, _size, _offset, _check_vector)
 *   XtWriter (+ properties)             rocke_attn_bwd_xt_writer_t (+ _init, _n_dg,
 *                                       _n_sg, _d_lanes_eff, _units, _iterations,
 *                                       _write_bytes, _stores, _loads, _unit_coords)
 *   default_xt_writer                   rocke_attn_bwd_default_xt_writer
 *   xt_read_accesses / xt_write_accesses rocke_attn_bwd_xt_read_accesses / _write_
 *   xt_conflicts                        rocke_attn_bwd_xt_conflicts
 *   choose_xt_swizzle                   rocke_attn_bwd_choose_xt_swizzle
 *   planner_xt_swizzle                  rocke_attn_bwd_planner_xt_swizzle
 *   DmaSlabLayout (+ methods)           rocke_attn_bwd_dma_slab_layout_t (+ _init,
 *                                       _fold, _slabs, _size, _offset, _dma_source)
 *   dma_slab_read_accesses / _conflicts rocke_attn_bwd_dma_slab_read_accesses / _conflicts
 *   BwdLdsRequest (+ __post_init__)     rocke_attn_bwd_lds_request_t (+ _default, _init)
 *     .elem_bytes / .threads / .swizzle_of  rocke_attn_bwd_lds_request_elem_bytes / ...
 *   LdsBuffer                           rocke_attn_bwd_lds_buffer_t
 *   LdsPlan (+ methods)                 rocke_attn_bwd_lds_plan_t (+ _fits, ...)
 *   plan_bwd_lds                        rocke_attn_bwd_plan_lds
 *
 * The conflict predictions go through the wave64 profiles of
 * rocke.analysis.lds (gfx90a, gfx942, gfx950): their phase / address-class
 * rules are ported here (private to this file) for the opcodes the XT and
 * slab accesses use, with the same request checks, since a refused request
 * makes a swizzle candidate illegal. The targets, LDS capacities and rules
 * are copies: a change to rocke.analysis.lds (a profile, an opcode, a new
 * wave64 target) must be mirrored here. The parity pair
 * attention_bwd_layout_emit ("choose" configs: every candidate's predicted
 * read / write multiplicity on gfx90a / gfx942 / gfx950 / gfx1201) fails on
 * such drift, so it does not pass silently.
 *
 * A plan holds at most ROCKE_ATTN_BWD_LDS_MAX_BUFFERS buffers: Python bounds
 * them through the request's value sets (ring_depth <= 3, ds_slots <= 2;
 * 5 resident + 2 + 1 stage + 21 loop + 2 epilogue = 31).
 *
 * Not ported yet: BwdLdsRequest.from_geometry (duck-typed over
 * BwdTileGeometry, which the body unit owns; lds_request of
 * instance_attention_bwd.h builds the request of a spec).
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_LDS_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_LDS_H

#include <stdbool.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h" /* rocke_attn_bwd_err_t */

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_ATTN_BWD_LDS_LINE_BYTES 128
#define ROCKE_ATTN_BWD_BUFFER_ALIGN_BYTES 128
#define ROCKE_ATTN_BWD_LDS_MAX_BUFFERS 32
#define ROCKE_ATTN_BWD_LDS_DESC_CAP 24

/* PHASES, as bits of LdsBuffer.phases (resident buffers have all four). */
typedef enum rocke_attn_bwd_lds_phase
{
    ROCKE_ATTN_BWD_PHASE_STAGE0_K = 1,
    ROCKE_ATTN_BWD_PHASE_STAGE0_V = 2,
    ROCKE_ATTN_BWD_PHASE_LOOP = 4,
    ROCKE_ATTN_BWD_PHASE_EPILOGUE = 8
} rocke_attn_bwd_lds_phase_t;
#define ROCKE_ATTN_BWD_PHASES_ALL 15

/* "stage0_k", "stage0_v", "loop", "epilogue" for one phase bit. */
const char* rocke_attn_bwd_lds_phase_name(rocke_attn_bwd_lds_phase_t phase);

/* lds_capacity_bytes(arch). */
rocke_status_t
    rocke_attn_bwd_lds_capacity_bytes(const char* arch, int* out, rocke_attn_bwd_err_t* err);

/* ----------------------------------------------------------- XtLayout */

/* swizzle is a static spelling of _SWIZZLES after _init. */
typedef struct rocke_attn_bwd_xt_layout
{
    int rows;
    int cols;
    int elem_bytes;
    const char* swizzle;
    int chunk_bytes;
} rocke_attn_bwd_xt_layout_t;

/* XtLayout(rows, cols, elem_bytes, swizzle, chunk_bytes). */
rocke_status_t rocke_attn_bwd_xt_layout_init(rocke_attn_bwd_xt_layout_t* out,
                                             int rows,
                                             int cols,
                                             int elem_bytes,
                                             const char* swizzle,
                                             int chunk_bytes,
                                             rocke_attn_bwd_err_t* err);
int rocke_attn_bwd_xt_layout_row_pitch(const rocke_attn_bwd_xt_layout_t* l);
int rocke_attn_bwd_xt_layout_size(const rocke_attn_bwd_xt_layout_t* l);
/* .offset(row, col): the byte offset, or -1 + err outside the image. */
int rocke_attn_bwd_xt_layout_offset(const rocke_attn_bwd_xt_layout_t* l,
                                    int row,
                                    int col,
                                    rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_xt_layout_check_vector(const rocke_attn_bwd_xt_layout_t* l,
                                                     int col,
                                                     int n,
                                                     rocke_attn_bwd_err_t* err);

/* ------------------------------------------------------- RowXorLayout */

typedef struct rocke_attn_bwd_row_xor_layout
{
    int rows;
    int cols;
    int elem_bytes;
    int chunk_bytes;
} rocke_attn_bwd_row_xor_layout_t;

rocke_status_t rocke_attn_bwd_row_xor_layout_init(rocke_attn_bwd_row_xor_layout_t* out,
                                                  int rows,
                                                  int cols,
                                                  int elem_bytes,
                                                  int chunk_bytes,
                                                  rocke_attn_bwd_err_t* err);
/* .image: the "xor" XtLayout of the same shape. */
rocke_attn_bwd_xt_layout_t
    rocke_attn_bwd_row_xor_layout_image(const rocke_attn_bwd_row_xor_layout_t* l);
int rocke_attn_bwd_row_xor_layout_row_pitch(const rocke_attn_bwd_row_xor_layout_t* l);
int rocke_attn_bwd_row_xor_layout_size(const rocke_attn_bwd_row_xor_layout_t* l);
int rocke_attn_bwd_row_xor_layout_offset(const rocke_attn_bwd_row_xor_layout_t* l,
                                         int row,
                                         int col,
                                         rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_row_xor_layout_check_vector(const rocke_attn_bwd_row_xor_layout_t* l,
                                                          int col,
                                                          int n,
                                                          rocke_attn_bwd_err_t* err);

/* ----------------------------------------------------------- XtWriter */

typedef struct rocke_attn_bwd_xt_writer
{
    int seq;
    int depth;
    int threads;
    int vec_d;
    int k_per_thread;
    int d_lanes;
    int elem_bytes;
} rocke_attn_bwd_xt_writer_t;

/* XtWriter(seq, depth, threads, vec_d, k_per_thread, d_lanes, elem_bytes);
 * Python's defaults are vec_d 4, k_per_thread 2, d_lanes 16, elem_bytes 2. */
rocke_status_t rocke_attn_bwd_xt_writer_init(rocke_attn_bwd_xt_writer_t* out,
                                             int seq,
                                             int depth,
                                             int threads,
                                             int vec_d,
                                             int k_per_thread,
                                             int d_lanes,
                                             int elem_bytes,
                                             rocke_attn_bwd_err_t* err);
int rocke_attn_bwd_xt_writer_n_dg(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_n_sg(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_d_lanes_eff(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_units(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_iterations(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_write_bytes(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_stores_per_thread(const rocke_attn_bwd_xt_writer_t* w);
int rocke_attn_bwd_xt_writer_loads_per_thread(const rocke_attn_bwd_xt_writer_t* w);
/* .unit_coords(tid, iteration): true + (s0, d0), or false (Python None). */
bool rocke_attn_bwd_xt_writer_unit_coords(
    const rocke_attn_bwd_xt_writer_t* w, int tid, int iteration, int* s0, int* d0);

/* default_xt_writer(seq, depth, threads, elem_bytes=, max_vec_bytes=). */
rocke_status_t rocke_attn_bwd_default_xt_writer(rocke_attn_bwd_xt_writer_t* out,
                                                int seq,
                                                int depth,
                                                int threads,
                                                int elem_bytes,
                                                int max_vec_bytes,
                                                rocke_attn_bwd_err_t* err);

/* -------------------------------------------------- access predictions */

/* (lane, byte, width) of one LDS access. */
typedef struct rocke_attn_bwd_lds_access
{
    int lane;
    int byte;
    int width;
} rocke_attn_bwd_lds_access_t;

/* xt_read_accesses / xt_write_accesses / dma_slab_read_accesses write up to
 * 64 accesses (one per lane of a wave64) to `out`, the count to *n_out. */
rocke_status_t rocke_attn_bwd_xt_read_accesses(const rocke_attn_bwd_xt_layout_t* layout,
                                               int n0,
                                               int k0,
                                               int atom_k,
                                               int wave_size,
                                               rocke_attn_bwd_lds_access_t out[64],
                                               int* n_out,
                                               rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_xt_write_accesses(const rocke_attn_bwd_xt_layout_t* layout,
                                                const rocke_attn_bwd_xt_writer_t* writer,
                                                int wave,
                                                int iteration,
                                                int j,
                                                int wave_size,
                                                rocke_attn_bwd_lds_access_t out[64],
                                                int* n_out,
                                                rocke_attn_bwd_err_t* err);

/* xt_conflicts(layout, writer, arch, atom_k=): *has_out false for Python
 * None (no wave64 profile), else read / write multiplicities. writer may be
 * NULL (Python None). */
rocke_status_t rocke_attn_bwd_xt_conflicts(const rocke_attn_bwd_xt_layout_t* layout,
                                           const rocke_attn_bwd_xt_writer_t* writer,
                                           const char* arch,
                                           int atom_k,
                                           bool* has_out,
                                           int* read_out,
                                           int* write_out,
                                           rocke_attn_bwd_err_t* err);

/* One row of choose_xt_swizzle's table. */
typedef struct rocke_attn_bwd_xt_choice
{
    const char* swizzle; /* the candidate as given */
    bool illegal; /* table[sw] = {"illegal": msg} */
    char illegal_msg[512];
    bool has_prediction; /* False: "prediction": None */
    int read;
    int write;
    int bytes;
} rocke_attn_bwd_xt_choice_t;

#define ROCKE_ATTN_BWD_XT_CANDIDATES_MAX 16

/* choose_xt_swizzle(rows, cols, writer, arch, candidates=, elem_bytes=,
 * atom_k=). candidates NULL / n_candidates 0 is Python's default
 * ("xor", "pad8", "pad16", "pad32", "none"); more than
 * ROCKE_ATTN_BWD_XT_CANDIDATES_MAX are refused. table[i] is the row of the
 * i-th distinct candidate (dict order), *n_table their count, *best the
 * pick. */
rocke_status_t rocke_attn_bwd_choose_xt_swizzle(int rows,
                                                int cols,
                                                const rocke_attn_bwd_xt_writer_t* writer,
                                                const char* arch,
                                                const char* const* candidates,
                                                int n_candidates,
                                                int elem_bytes,
                                                int atom_k,
                                                const char** best,
                                                rocke_attn_bwd_xt_choice_t* table,
                                                int* n_table,
                                                rocke_attn_bwd_err_t* err);

/* planner_xt_swizzle(rows, cols, threads, arch, elem_bytes, atom_k): a static
 * swizzle spelling (cached like Python's @cache). */
const char* rocke_attn_bwd_planner_xt_swizzle(int rows,
                                              int cols,
                                              int threads,
                                              const char* arch,
                                              int elem_bytes,
                                              int atom_k,
                                              rocke_attn_bwd_err_t* err);

/* ------------------------------------------------------ DmaSlabLayout */

typedef struct rocke_attn_bwd_dma_slab_layout
{
    int rows;
    int cols;
    int elem_bytes;
} rocke_attn_bwd_dma_slab_layout_t;

rocke_status_t rocke_attn_bwd_dma_slab_layout_init(rocke_attn_bwd_dma_slab_layout_t* out,
                                                   int rows,
                                                   int cols,
                                                   int elem_bytes,
                                                   rocke_attn_bwd_err_t* err);
int rocke_attn_bwd_dma_slab_layout_row_bytes(const rocke_attn_bwd_dma_slab_layout_t* l);
int rocke_attn_bwd_dma_slab_layout_fold(const rocke_attn_bwd_dma_slab_layout_t* l);
int rocke_attn_bwd_dma_slab_layout_slabs(const rocke_attn_bwd_dma_slab_layout_t* l);
int rocke_attn_bwd_dma_slab_layout_size(const rocke_attn_bwd_dma_slab_layout_t* l);
int rocke_attn_bwd_dma_slab_layout_offset(const rocke_attn_bwd_dma_slab_layout_t* l,
                                          int row,
                                          int col,
                                          rocke_attn_bwd_err_t* err);
/* .dma_source(slot) -> (row, col). */
rocke_status_t rocke_attn_bwd_dma_slab_layout_dma_source(const rocke_attn_bwd_dma_slab_layout_t* l,
                                                         int slot,
                                                         int* row,
                                                         int* col,
                                                         rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_dma_slab_read_accesses(const rocke_attn_bwd_dma_slab_layout_t* l,
                                                     int m0,
                                                     int k0,
                                                     int atom_k,
                                                     int wave_size,
                                                     rocke_attn_bwd_lds_access_t out[64],
                                                     int* n_out,
                                                     rocke_attn_bwd_err_t* err);
/* dma_slab_read_conflicts(layout, arch, atom_k=): *has_out false for None. */
rocke_status_t rocke_attn_bwd_dma_slab_read_conflicts(const rocke_attn_bwd_dma_slab_layout_t* l,
                                                      const char* arch,
                                                      int atom_k,
                                                      bool* has_out,
                                                      int* worst_out,
                                                      rocke_attn_bwd_err_t* err);

/* ------------------------------------------------------- BwdLdsRequest */

/* BwdLdsRequest. Strings are the Python spellings. lds_swizzle is the
 * "name=value,..." text BwdLdsRequest.canonical_swizzle reads (NULL or "" for
 * ()); the request borrows it, so it must outlive the request and any plan
 * made from it. */
typedef struct rocke_attn_bwd_lds_request
{
    int head_size;
    int k_m0;
    int k_n0;
    int waves;
    int wave_size; /* 64 */
    const char* dtype; /* "fp16" */
    const char* kv_residency; /* "reg" */
    const char* kt_source; /* "reg" */
    const char* transpose_source; /* "xt_lds" */
    int ring_depth; /* 1 */
    const char* global_path; /* "vgpr" */
    const char* pt_route; /* "relabel" */
    bool acc_in_lds; /* false */
    const char* dq_mode; /* "atomic" */
    const char* lds_swizzle; /* NULL */
    const char* dkv_epilogue; /* "lds_staged" */
    const char* lds_order; /* "resident_low" */
    int ds_slots; /* 1 */
    int atom_k_g13; /* 16 */
    int atom_k_g4; /* 16 */
} rocke_attn_bwd_lds_request_t;

/* Python's defaults; the caller sets head_size, k_m0, k_n0 and waves. */
rocke_attn_bwd_lds_request_t rocke_attn_bwd_lds_request_default(void);
/* BwdLdsRequest.__post_init__. */
rocke_status_t rocke_attn_bwd_lds_request_init(const rocke_attn_bwd_lds_request_t* req,
                                               rocke_attn_bwd_err_t* err);
int rocke_attn_bwd_lds_request_elem_bytes(const rocke_attn_bwd_lds_request_t* req);
int rocke_attn_bwd_lds_request_threads(const rocke_attn_bwd_lds_request_t* req);
/* .swizzle_of(name, default): a pointer into static spellings or `dflt`. */
const char* rocke_attn_bwd_lds_request_swizzle_of(const rocke_attn_bwd_lds_request_t* req,
                                                  const char* name,
                                                  const char* dflt);

/* ------------------------------------------------------------- LdsPlan */

typedef enum rocke_attn_bwd_lds_layout_kind
{
    ROCKE_ATTN_BWD_LAYOUT_DESC = 0, /* no layout object: the description */
    ROCKE_ATTN_BWD_LAYOUT_XT,
    ROCKE_ATTN_BWD_LAYOUT_ROW_XOR,
    ROCKE_ATTN_BWD_LAYOUT_DMA_SLAB
} rocke_attn_bwd_lds_layout_kind_t;

/* A value of LdsPlan.layouts: a layout object or its description string. */
typedef struct rocke_attn_bwd_lds_layout
{
    const char* name; /* the buffer name */
    rocke_attn_bwd_lds_layout_kind_t kind;
    char desc[ROCKE_ATTN_BWD_LDS_DESC_CAP]; /* kind DESC */
    rocke_attn_bwd_xt_layout_t xt; /* kind XT */
    rocke_attn_bwd_row_xor_layout_t row_xor; /* kind ROW_XOR */
    rocke_attn_bwd_dma_slab_layout_t dma; /* kind DMA_SLAB */
} rocke_attn_bwd_lds_layout_t;

/* LdsBuffer: name is a static spelling; phases a mask of
 * rocke_attn_bwd_lds_phase_t (Python keeps them in PHASES order). */
typedef struct rocke_attn_bwd_lds_buffer
{
    const char* name;
    int slot;
    int offset;
    int nbytes;
    unsigned phases;
    char layout[ROCKE_ATTN_BWD_LDS_DESC_CAP];
    bool resident;
} rocke_attn_bwd_lds_buffer_t;

typedef struct rocke_attn_bwd_lds_plan
{
    const char* arch; /* static catalog spelling */
    rocke_attn_bwd_lds_request_t request;
    rocke_attn_bwd_lds_buffer_t buffers[ROCKE_ATTN_BWD_LDS_MAX_BUFFERS];
    int n_buffers;
    int resident_bytes;
    int total_bytes;
    int capacity_bytes;
    int phase_bytes[4]; /* PHASES order */
    rocke_attn_bwd_lds_layout_t layouts[ROCKE_ATTN_BWD_LDS_MAX_BUFFERS]; /* sorted by name */
    int n_layouts;
} rocke_attn_bwd_lds_plan_t;

/* plan_bwd_lds(req, arch=, strict=). */
rocke_status_t rocke_attn_bwd_plan_lds(const rocke_attn_bwd_lds_request_t* req,
                                       const char* arch,
                                       bool strict,
                                       rocke_attn_bwd_lds_plan_t* out,
                                       rocke_attn_bwd_err_t* err);

bool rocke_attn_bwd_lds_plan_fits(const rocke_attn_bwd_lds_plan_t* p);
int rocke_attn_bwd_lds_plan_phase_offset(const rocke_attn_bwd_lds_plan_t* p);
int rocke_attn_bwd_lds_plan_unaliased_bytes(const rocke_attn_bwd_lds_plan_t* p);
/* .buffer(name, slot): NULL + KeyError when absent. */
const rocke_attn_bwd_lds_buffer_t* rocke_attn_bwd_lds_plan_buffer(
    const rocke_attn_bwd_lds_plan_t* p, const char* name, int slot, rocke_attn_bwd_err_t* err);
bool rocke_attn_bwd_lds_plan_has(const rocke_attn_bwd_lds_plan_t* p, const char* name);
/* .offset(name, slot): -1 + KeyError when absent. */
int rocke_attn_bwd_lds_plan_offset(const rocke_attn_bwd_lds_plan_t* p,
                                   const char* name,
                                   int slot,
                                   rocke_attn_bwd_err_t* err);
/* .ring_offsets(name): count (up to ROCKE_ATTN_BWD_LDS_MAX_BUFFERS). */
int rocke_attn_bwd_lds_plan_ring_offsets(const rocke_attn_bwd_lds_plan_t* p,
                                         const char* name,
                                         int out[ROCKE_ATTN_BWD_LDS_MAX_BUFFERS]);
/* .ring_stride(name): the stride, or -1 + err (KeyError / ValueError). */
int rocke_attn_bwd_lds_plan_ring_stride(const rocke_attn_bwd_lds_plan_t* p,
                                        const char* name,
                                        rocke_attn_bwd_err_t* err);
/* .layout(name): NULL + KeyError when absent. */
const rocke_attn_bwd_lds_layout_t* rocke_attn_bwd_lds_plan_layout(
    const rocke_attn_bwd_lds_plan_t* p, const char* name, rocke_attn_bwd_err_t* err);
/* .check() / .check_capacity(). */
rocke_status_t rocke_attn_bwd_lds_plan_check(const rocke_attn_bwd_lds_plan_t* p,
                                             rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_lds_plan_check_capacity(const rocke_attn_bwd_lds_plan_t* p,
                                                      rocke_attn_bwd_err_t* err);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <vector>

namespace rocke_attn_bwd
{
rocke_attn_bwd_lds_plan_t
    plan_bwd_lds(const rocke_attn_bwd_lds_request_t& req, const std::string& arch, bool strict);
void lds_request_check(const rocke_attn_bwd_lds_request_t& req);
std::string planner_xt_swizzle(
    int rows, int cols, int threads, const std::string& arch, int elem_bytes, int atom_k);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_LDS_H */
