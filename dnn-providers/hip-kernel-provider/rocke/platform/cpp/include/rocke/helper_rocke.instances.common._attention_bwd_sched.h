/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_sched.h -- C port of the
 * stage-scheduler tables of the attention backward
 * (library/kernels/common/_attention_bwd_sched.py): the per-stage, per-wave,
 * per-lane instruction counts of one q step (main kernel) or key step (split
 * dQ kernel), and the scheduling hints they emit (iglp_opt, s_setprio,
 * sched_group_barrier, sched_barrier).
 *
 *   Python (_attention_bwd_sched.py)    C (this header)
 *   ---------------------------------   ---------------------------------------
 *   SCHEDS, STAGE_TABLE_SCHEDS,         ROCKE_ATTN_BWD_SCHEDS, ..._STAGE_TABLE_SCHEDS,
 *   FENCED_SCHEDS, IGLP_SCHEDS          ..._FENCED_SCHEDS, ..._IGLP_SCHEDS (+ _N_*)
 *   PIPELINES, BUILT_PIPELINES          ROCKE_ATTN_BWD_PIPELINES, ..._BUILT_PIPELINES
 *   COUNTED_LATE_DEFERRED               ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED
 *   STAGES, SPLIT_STAGES                ROCKE_ATTN_BWD_STAGES, ..._SPLIT_STAGES
 *   SPLIT_DEFERRED                      ROCKE_ATTN_BWD_SPLIT_DEFERRED
 *   DQ_SCHEDS, DQ_FENCED_SCHEDS,        ROCKE_ATTN_BWD_DQ_SCHEDS, ..._DQ_FENCED_SCHEDS,
 *   DQ_STAGES                           ..._DQ_STAGES
 *   GEMMS, STAGE_GEMMS                  ROCKE_ATTN_BWD_GEMMS, rocke_attn_bwd_stage_gemms
 *   _G4_SLICINGS (the body imports it)  ROCKE_ATTN_BWD_SCHED_G4_SLICINGS
 *   native_atom_ks                      rocke_attn_bwd_native_atom_ks
 *   instruction_k                       rocke_attn_bwd_instruction_k
 *   interleave_groups                   rocke_attn_bwd_interleave_groups
 *   GemmCounts (+ as_dict, total)       rocke_attn_bwd_gemm_counts_t (+ _total)
 *   StageCounts (+ memory, by_mask,     rocke_attn_bwd_stage_counts_t (+ _memory,
 *     census, groups)                     _by_mask, _census, _groups)
 *   StageTable                          rocke_attn_bwd_stage_table_t
 *     __post_init__ / from_params         rocke_attn_bwd_stage_table_default / _init
 *     from_geometry                       rocke_attn_bwd_stage_table_from_geometry
 *     every other method / property       rocke_attn_bwd_stage_table_<name>
 *     emit_loop_head / emit_stage_*       rocke_attn_bwd_stage_table_emit_*
 *   DqStageTable                        rocke_attn_bwd_dq_stage_table_t (same scheme)
 *
 * Masks are the ROCKE_SCHED_* classes of helper_rocke.helpers.schedule.h.
 *
 * Error model: a Python ValueError (KeyError for an unknown arch, as
 * ArchTarget.from_gfx) becomes a non-OK rocke_status_t plus str(exc) in a
 * rocke_attn_bwd_err_t (err may be NULL). The emit_* entry points record it on
 * the builder's sticky error instead and return false, before they emit
 * anything (Python computes the groups before its first builder call). One
 * documented difference: a zero divisor that Python's own arithmetic would hit
 * (ZeroDivisionError, only reachable with zero-sized grids that the checks
 * let through) is ROCKE_ERR_VALUE "integer division or modulo by zero".
 *
 * Python duck types a few inputs the C port spells as plain data: an atom is
 * its K extent (an int; ROCKE_ATTN_BWD_NONE is None, 16; 0 stays 0), the
 * geometries of from_geometry are
 * structs whose absent attributes are NULL strings, ROCKE_ATTN_BWD_NONE ints
 * or false has_* flags, and acc_in_lds / setprio / head_pack / kv_prefetch
 * are C bools (Python's "must be a bool" checks cannot fail). Strings are
 * borrowed: the caller keeps them alive as long as the table.
 *
 * Sequences a method returns are written to caller arrays: fixed-size ones
 * where the Python value is bounded by construction (at most four stages,
 * five classes, three waits), else (out, cap, n_out): *n_out is always the
 * total count, out NULL is a sizing query, and a non-NULL out with cap below
 * the total is refused (ROCKE_ERR_VALUE, nothing written), never truncated.
 *
 * Not ported: StageTable.describe (a plain-data dict for sweep records; each
 * of its values has a C function above).
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_SCHED_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_SCHED_H

#include <stdbool.h>

#include "rocke/helper_rocke.helpers.schedule.h" /* ROCKE_SCHED_* masks */
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h" /* rocke_attn_bwd_err_t */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- constants */

#define ROCKE_ATTN_BWD_N_SCHEDS 6
extern const char* const ROCKE_ATTN_BWD_SCHEDS[ROCKE_ATTN_BWD_N_SCHEDS];
#define ROCKE_ATTN_BWD_N_STAGE_TABLE_SCHEDS 2
extern const char* const ROCKE_ATTN_BWD_STAGE_TABLE_SCHEDS[ROCKE_ATTN_BWD_N_STAGE_TABLE_SCHEDS];
#define ROCKE_ATTN_BWD_N_FENCED_SCHEDS 3
extern const char* const ROCKE_ATTN_BWD_FENCED_SCHEDS[ROCKE_ATTN_BWD_N_FENCED_SCHEDS];
#define ROCKE_ATTN_BWD_N_IGLP_SCHEDS 2
extern const char* const ROCKE_ATTN_BWD_IGLP_SCHEDS[ROCKE_ATTN_BWD_N_IGLP_SCHEDS];
#define ROCKE_ATTN_BWD_N_PIPELINES 4
extern const char* const ROCKE_ATTN_BWD_PIPELINES[ROCKE_ATTN_BWD_N_PIPELINES];
#define ROCKE_ATTN_BWD_N_BUILT_PIPELINES 3
extern const char* const ROCKE_ATTN_BWD_BUILT_PIPELINES[ROCKE_ATTN_BWD_N_BUILT_PIPELINES];
extern const char* const ROCKE_ATTN_BWD_COUNTED_LATE_DEFERRED;
#define ROCKE_ATTN_BWD_N_STAGES 4
extern const char* const ROCKE_ATTN_BWD_STAGES[ROCKE_ATTN_BWD_N_STAGES];
#define ROCKE_ATTN_BWD_N_SPLIT_STAGES 3
extern const char* const ROCKE_ATTN_BWD_SPLIT_STAGES[ROCKE_ATTN_BWD_N_SPLIT_STAGES];
extern const char* const ROCKE_ATTN_BWD_SPLIT_DEFERRED;
#define ROCKE_ATTN_BWD_N_DQ_SCHEDS 6
extern const char* const ROCKE_ATTN_BWD_DQ_SCHEDS[ROCKE_ATTN_BWD_N_DQ_SCHEDS];
#define ROCKE_ATTN_BWD_N_DQ_FENCED_SCHEDS 3
extern const char* const ROCKE_ATTN_BWD_DQ_FENCED_SCHEDS[ROCKE_ATTN_BWD_N_DQ_FENCED_SCHEDS];
#define ROCKE_ATTN_BWD_N_DQ_STAGES 3
extern const char* const ROCKE_ATTN_BWD_DQ_STAGES[ROCKE_ATTN_BWD_N_DQ_STAGES];
#define ROCKE_ATTN_BWD_N_GEMMS 5
extern const char* const ROCKE_ATTN_BWD_GEMMS[ROCKE_ATTN_BWD_N_GEMMS];
#define ROCKE_ATTN_BWD_SCHED_N_G4_SLICINGS 4
extern const char* const ROCKE_ATTN_BWD_SCHED_G4_SLICINGS[ROCKE_ATTN_BWD_SCHED_N_G4_SLICINGS];

/* rocke.helpers.compile.COMGR_FRESH_PROCESS_ATTR (no C twin yet). */
#ifndef ROCKE_COMGR_FRESH_PROCESS_ATTR
#define ROCKE_COMGR_FRESH_PROCESS_ATTR "comgr_fresh_process"
#endif

/* STAGE_GEMMS[stage]: writes the GEMM names (at most 2) and returns their
 * count; -1 for a stage outside STAGES (Python KeyError). */
int rocke_attn_bwd_stage_gemms(const char* stage, const char* out[2]);

/* ------------------------------------------------------- atoms and groups */

/* native_atom_ks(arch, dtype): sorted K extents (out / cap / n_out as
 * above). */
rocke_status_t rocke_attn_bwd_native_atom_ks(
    const char* arch, const char* dtype, int* out, int cap, int* n_out, rocke_attn_bwd_err_t* err);
/* instruction_k(arch, atom_k, dtype). */
rocke_status_t rocke_attn_bwd_instruction_k(
    const char* arch, int atom_k, const char* dtype, int* out, rocke_attn_bwd_err_t* err);

/* One (mask, count) pair: a sched group, a lead / span / fixed entry, or one
 * item of a {mask: count} dict (in its insertion order). */
typedef struct rocke_attn_bwd_sched_pair
{
    int mask;
    int count;
} rocke_attn_bwd_sched_pair_t;

/* interleave_groups(mfma, mem, lead=, spans=): lead / spans are the dicts'
 * items (n 0 is None / {}; a repeated mask keeps its first position and its
 * last value, as dict() does). out / cap / n_out: see the header comment
 * (total in *n_out; out NULL sizes; cap too small is refused). */
rocke_status_t rocke_attn_bwd_interleave_groups(int mfma,
                                                const rocke_attn_bwd_sched_pair_t* mem,
                                                int n_mem,
                                                const rocke_attn_bwd_sched_pair_t* lead,
                                                int n_lead,
                                                const rocke_attn_bwd_sched_pair_t* spans,
                                                int n_spans,
                                                rocke_attn_bwd_sched_pair_t* out,
                                                int cap,
                                                int* n_out,
                                                rocke_attn_bwd_err_t* err);

/* ----------------------------------------------------------- GemmCounts */

typedef struct rocke_attn_bwd_gemm_counts
{
    int g0, g1, g2, g3, g4;
} rocke_attn_bwd_gemm_counts_t;

int rocke_attn_bwd_gemm_counts_total(const rocke_attn_bwd_gemm_counts_t* c);

/* ---------------------------------------------------------- StageCounts */

/* lead / spans / fixed are borrowed (pointer, count) arrays of any length,
 * as Python's tuples (count 0 with a NULL pointer is ()). A negative count, or
 * a NULL pointer with a positive one, is a ValueError. */
typedef struct rocke_attn_bwd_stage_counts
{
    const char* stage;
    int mfma;
    int vmem_read;
    int vmem_write;
    int ds_read;
    int ds_write;
    int n_lead;
    const rocke_attn_bwd_sched_pair_t* lead;
    int n_spans;
    const rocke_attn_bwd_sched_pair_t* spans;
    int n_fixed;
    const rocke_attn_bwd_sched_pair_t* fixed;
} rocke_attn_bwd_stage_counts_t;

/* Backing store for the lead / spans / fixed arrays of the stage_counts a
 * stages() / stage() call writes; they point into it, so it must outlive
 * them, and each call reuses it from the start. A table's stages carry at
 * most twelve pairs in total by construction (the dQ table's four); more is a
 * ValueError. */
#define ROCKE_ATTN_BWD_STAGE_STORE_PAIRS 16
typedef struct rocke_attn_bwd_stage_pairs_store
{
    int n;
    rocke_attn_bwd_sched_pair_t pairs[ROCKE_ATTN_BWD_STAGE_STORE_PAIRS];
} rocke_attn_bwd_stage_pairs_store_t;

/* memory(): (VMEM_READ, DS_READ, DS_WRITE, VMEM_WRITE) pairs. */
void rocke_attn_bwd_stage_counts_memory(const rocke_attn_bwd_stage_counts_t* s,
                                        rocke_attn_bwd_sched_pair_t out[4]);
/* by_mask(): the dict items (MFMA, VMEM_READ, VMEM_WRITE, DS_READ, DS_WRITE). */
void rocke_attn_bwd_stage_counts_by_mask(const rocke_attn_bwd_stage_counts_t* s,
                                         rocke_attn_bwd_sched_pair_t out[5]);
/* census(): by_mask plus fixed. A fixed mask outside the five classes is a
 * Python KeyError. */
rocke_status_t rocke_attn_bwd_stage_counts_census(const rocke_attn_bwd_stage_counts_t* s,
                                                  rocke_attn_bwd_sched_pair_t out[5],
                                                  rocke_attn_bwd_err_t* err);
/* groups(): see rocke_attn_bwd_interleave_groups for out / cap / n_out. */
rocke_status_t rocke_attn_bwd_stage_counts_groups(const rocke_attn_bwd_stage_counts_t* s,
                                                  rocke_attn_bwd_sched_pair_t* out,
                                                  int cap,
                                                  int* n_out,
                                                  rocke_attn_bwd_err_t* err);

/* ----------------------------------------------------------- StageTable */

typedef struct rocke_attn_bwd_stage_table
{
    const char* arch;
    int head_size;
    int k_m0;
    int k_n0;
    int k_k4;
    int waves;
    int grid_g4[2];
    int atom_g02;
    int atom_g13;
    int atom_g4;
    const char* dtype;
    const char* transpose_source;
    const char* kt_source;
    const char* global_path;
    const char* pt_route;
    int stage_vec;
    const char* dq_mode;
    const char* sched;
    bool setprio;
    const char* kv_residency;
    bool acc_in_lds;
    int grid_g02[2];
    int grid_g13[2];
    const char* pipeline;
    const char* stats_load;
    const char* dq_issue;
    const char* lds_swizzle; /* NULL: None */
    const char* g4_slicing;
    int ring_depth;
    const char* barrier_merge;
    const char* ring_issue;
    bool head_pack;
    const char* loop_sync;
    const char* counted_issue;
} rocke_attn_bwd_stage_table_t;

/* The dataclass defaults; arch / head_size / k_m0 / k_n0 / k_k4 / waves (no
 * default in Python) are NULL / 0. */
rocke_attn_bwd_stage_table_t rocke_attn_bwd_stage_table_default(void);
/* __post_init__ in place (from_params is the default plus fields, then this):
 * resolves the natural grids and checks every field. */
rocke_status_t rocke_attn_bwd_stage_table_init(rocke_attn_bwd_stage_table_t* t,
                                               rocke_attn_bwd_err_t* err);

/* The BwdTileGeometry attributes from_geometry reads. An absent attribute is
 * a NULL string, ROCKE_ATTN_BWD_NONE int, has_* false, or (has_atoms false)
 * a missing atoms table; atom_* ROCKE_ATTN_BWD_NONE is an absent atom (16),
 * and 0 stays 0 (as Python's _atom_k(0); the table checks then hit the
 * documented zero-divisor difference). An atom object or a "wmma16x16x16"
 * string (Python's other _atom_k forms) has no C spelling: resolve it to its
 * K before filling the geom. */
typedef struct rocke_attn_bwd_sched_geom
{
    int head_size;
    int k_m0;
    int k_n0;
    int k_k4;
    int waves;
    bool has_atoms;
    int atom_g02;
    int atom_g13;
    int atom_g4;
    bool has_grid_g4;
    int grid_g4[2];
    bool has_grid_g0;
    int grid_g0[2];
    bool has_grid_g1;
    int grid_g1[2];
    int ring_depth;
    const char* global_path;
    const char* barrier_merge;
    const char* transpose_source;
    const char* kt_source;
    const char* pt_route;
    int stage_vec;
    const char* dq_mode;
    const char* sched;
    bool setprio;
    const char* kv_residency;
    int acc_in_lds; /* ROCKE_ATTN_BWD_NONE: absent; else a bool */
    const char* stats_load;
    const char* dq_issue;
    bool has_lds_swizzle;
    const char* lds_swizzle; /* NULL with has_lds_swizzle: None */
    const char* g4_slicing;
    const char* ring_issue;
    bool head_pack;
    const char* loop_sync;
    const char* counted_issue;
} rocke_attn_bwd_sched_geom_t;

/* Every attribute absent. */
rocke_attn_bwd_sched_geom_t rocke_attn_bwd_sched_geom_empty(void);

/* StageTable.from_geometry(geom, arch, dtype=, stage_vec=, dq_mode=,
 * acc_in_lds=). Python's keyword defaults: "fp16", 8, "atomic", false. */
rocke_status_t rocke_attn_bwd_stage_table_from_geometry(const rocke_attn_bwd_sched_geom_t* geom,
                                                        const char* arch,
                                                        const char* dtype,
                                                        int stage_vec,
                                                        const char* dq_mode,
                                                        bool acc_in_lds,
                                                        rocke_attn_bwd_stage_table_t* out,
                                                        rocke_attn_bwd_err_t* err);

/* arch facts */
rocke_status_t rocke_attn_bwd_stage_table_wave_size(const rocke_attn_bwd_stage_table_t* t,
                                                    int* out,
                                                    rocke_attn_bwd_err_t* err);
/* instr_k: out[i] is the instruction K of GEMMS[i]. */
rocke_status_t rocke_attn_bwd_stage_table_instr_k(const rocke_attn_bwd_stage_table_t* t,
                                                  int out[5],
                                                  rocke_attn_bwd_err_t* err);
/* mma_op_id(gemm) (static storage); a gemm outside GEMMS is a KeyError. */
rocke_status_t rocke_attn_bwd_stage_table_mma_op_id(const rocke_attn_bwd_stage_table_t* t,
                                                    const char* gemm,
                                                    const char** out,
                                                    rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_stage_table_frag_len(const rocke_attn_bwd_stage_table_t* t,
                                                   const char* gemm,
                                                   int* out,
                                                   rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_stage_table_c_frag_len(const rocke_attn_bwd_stage_table_t* t,
                                                     int* out,
                                                     rocke_attn_bwd_err_t* err);

/* MMA counts */
rocke_status_t rocke_attn_bwd_stage_table_mfma(const rocke_attn_bwd_stage_table_t* t,
                                               rocke_attn_bwd_gemm_counts_t* out,
                                               rocke_attn_bwd_err_t* err);
/* stage_names: STAGES or SPLIT_STAGES; returns the count. */
int rocke_attn_bwd_stage_table_stage_names(const rocke_attn_bwd_stage_table_t* t,
                                           const char* const** out);
rocke_status_t rocke_attn_bwd_stage_table_dq_atomics_per_lane(const rocke_attn_bwd_stage_table_t* t,
                                                              int* out,
                                                              rocke_attn_bwd_err_t* err);

/* memory model */
rocke_status_t rocke_attn_bwd_stage_table_stats_loads(const rocke_attn_bwd_stage_table_t* t,
                                                      int* out,
                                                      rocke_attn_bwd_err_t* err);
/* stages(prefetch=, prev_atomics=): writes the stages (3 or 4) and their
 * count; their pair arrays live in store. */
rocke_status_t rocke_attn_bwd_stage_table_stages(const rocke_attn_bwd_stage_table_t* t,
                                                 bool prefetch,
                                                 bool prev_atomics,
                                                 rocke_attn_bwd_stage_counts_t out[4],
                                                 int* n_out,
                                                 rocke_attn_bwd_stage_pairs_store_t* store,
                                                 rocke_attn_bwd_err_t* err);
/* stage(name, prefetch=, prev_atomics=); pair arrays in store. */
rocke_status_t rocke_attn_bwd_stage_table_stage(const rocke_attn_bwd_stage_table_t* t,
                                                const char* name,
                                                bool prefetch,
                                                bool prev_atomics,
                                                rocke_attn_bwd_stage_counts_t* out,
                                                rocke_attn_bwd_stage_pairs_store_t* store,
                                                rocke_attn_bwd_err_t* err);
/* groups(name, prefetch=, prev_atomics=): see interleave_groups for the
 * out / cap / n_out contract. */
rocke_status_t rocke_attn_bwd_stage_table_groups(const rocke_attn_bwd_stage_table_t* t,
                                                 const char* name,
                                                 bool prefetch,
                                                 bool prev_atomics,
                                                 rocke_attn_bwd_sched_pair_t* out,
                                                 int cap,
                                                 int* n_out,
                                                 rocke_attn_bwd_err_t* err);

/* ring_prefetch(): present false is Python's {}. The keys a dict holds:
 * issue, where, vmem_read, ds_write, and (has_dma) dma_per_wave, dma_x4,
 * dma_x1. */
typedef struct rocke_attn_bwd_ring_prefetch
{
    bool present;
    const char* issue; /* "dma" | "loads" | "merged" */
    const char* where; /* "A" | "top" | "barrier" */
    int vmem_read;
    int ds_write;
    bool has_dma;
    int dma_per_wave[2];
    int dma_x4;
    int dma_x1;
} rocke_attn_bwd_ring_prefetch_t;

rocke_status_t rocke_attn_bwd_stage_table_ring_prefetch(const rocke_attn_bwd_stage_table_t* t,
                                                        rocke_attn_bwd_ring_prefetch_t* out,
                                                        rocke_attn_bwd_err_t* err);
/* prefetch_region(): *has false is None, else {VMEM_READ: *vmem_read}. */
rocke_status_t rocke_attn_bwd_stage_table_prefetch_region(const rocke_attn_bwd_stage_table_t* t,
                                                          bool* has,
                                                          int* vmem_read,
                                                          rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_stage_table_prefetch_fence(const rocke_attn_bwd_stage_table_t* t,
                                                         bool* out,
                                                         rocke_attn_bwd_err_t* err);
/* ring_waits(): at most one immediate; returns it via out / n_out. */
rocke_status_t rocke_attn_bwd_stage_table_ring_waits(const rocke_attn_bwd_stage_table_t* t,
                                                     int out[1],
                                                     int* n_out,
                                                     rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_stage_table_ring_atomics_per_lane(
    const rocke_attn_bwd_stage_table_t* t, int* out, rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_stage_table_vm_after_prefetch(const rocke_attn_bwd_stage_table_t* t,
                                                            bool prev_atomics,
                                                            int* out,
                                                            rocke_attn_bwd_err_t* err);

/* counted waits */
bool rocke_attn_bwd_stage_table_counted(const rocke_attn_bwd_stage_table_t* t);

/* counted_terms(): present false is {}; keys issue, loads, stores. */
typedef struct rocke_attn_bwd_counted_terms
{
    bool present;
    const char* issue;
    int loads;
    int stores;
} rocke_attn_bwd_counted_terms_t;

rocke_status_t rocke_attn_bwd_stage_table_counted_terms(const rocke_attn_bwd_stage_table_t* t,
                                                        rocke_attn_bwd_counted_terms_t* out,
                                                        rocke_attn_bwd_err_t* err);
/* counted_waits(): none or three immediates. */
rocke_status_t rocke_attn_bwd_stage_table_counted_waits(const rocke_attn_bwd_stage_table_t* t,
                                                        int out[3],
                                                        int* n_out,
                                                        rocke_attn_bwd_err_t* err);
/* step_waits(): ring_waits + counted_waits (at most four). */
rocke_status_t rocke_attn_bwd_stage_table_step_waits(const rocke_attn_bwd_stage_table_t* t,
                                                     int out[4],
                                                     int* n_out,
                                                     rocke_attn_bwd_err_t* err);

/* A hint of hint_sequence(): ("fence", 0), ("iglp", level), ("prio", level)
 * or ("group", mask, count). */
typedef enum rocke_attn_bwd_hint_kind
{
    ROCKE_ATTN_BWD_HINT_FENCE,
    ROCKE_ATTN_BWD_HINT_IGLP,
    ROCKE_ATTN_BWD_HINT_PRIO,
    ROCKE_ATTN_BWD_HINT_GROUP
} rocke_attn_bwd_hint_kind_t;

typedef struct rocke_attn_bwd_hint
{
    rocke_attn_bwd_hint_kind_t kind;
    int a; /* fence 0, iglp / prio level, group mask */
    int b; /* group count (0 otherwise) */
} rocke_attn_bwd_hint_t;

/* hint_sequence(): writes up to cap hints; *n_out is the total. */
rocke_status_t rocke_attn_bwd_stage_table_hint_sequence(const rocke_attn_bwd_stage_table_t* t,
                                                        rocke_attn_bwd_hint_t* out,
                                                        int cap,
                                                        int* n_out,
                                                        rocke_attn_bwd_err_t* err);

/* A region of regions(): (name, {mask: n}, leading reads or None). The lead
 * is dict() of a table stage's lead, at most two classes by construction;
 * more is a ValueError. */
#define ROCKE_ATTN_BWD_REGION_LEAD_MAX 4
typedef struct rocke_attn_bwd_sched_region
{
    const char* name;
    int n_ops;
    rocke_attn_bwd_sched_pair_t ops[5];
    bool has_lead; /* false: None */
    int n_lead;
    rocke_attn_bwd_sched_pair_t lead[ROCKE_ATTN_BWD_REGION_LEAD_MAX];
} rocke_attn_bwd_sched_region_t;

/* regions(): writes them and their count. At most six by construction (the
 * four stages plus the ring's prefetch region, or plus the counted step's two
 * staging / issue regions); more is a ValueError. */
#define ROCKE_ATTN_BWD_SCHED_REGIONS_MAX 8
rocke_status_t rocke_attn_bwd_stage_table_regions(
    const rocke_attn_bwd_stage_table_t* t,
    rocke_attn_bwd_sched_region_t out[ROCKE_ATTN_BWD_SCHED_REGIONS_MAX],
    int* n_out,
    rocke_attn_bwd_err_t* err);

/* ir_count_model(prefetch=, prev_atomics=). group_counts are the dict's items
 * in insertion order; stage_names[i] / stage_mfma[i] the stage_mfma dict. */
typedef struct rocke_attn_bwd_ir_count_model
{
    int sched_group_barrier;
    int sched_barrier;
    int s_setprio;
    int iglp_opt;
    int s_waitcnt;
    int n_group_counts;
    rocke_attn_bwd_sched_pair_t group_counts[5];
    rocke_attn_bwd_gemm_counts_t mfma;
    int mfma_total;
    int n_stages;
    const char* stage_names[4];
    int stage_mfma[4];
    const char* pipeline;
    int n_ring_waits;
    int ring_waits[4];
    rocke_attn_bwd_ring_prefetch_t prefetch;
} rocke_attn_bwd_ir_count_model_t;

rocke_status_t rocke_attn_bwd_stage_table_ir_count_model(const rocke_attn_bwd_stage_table_t* t,
                                                         bool prefetch,
                                                         bool prev_atomics,
                                                         rocke_attn_bwd_ir_count_model_t* out,
                                                         rocke_attn_bwd_err_t* err);

/* emission: false after an error (recorded on the builder, nothing emitted).
 * emit_loop_head marks the kernel ROCKE_COMGR_FRESH_PROCESS_ATTR for iglp1. */
bool rocke_attn_bwd_stage_table_emit_loop_head(rocke_ir_builder_t* b,
                                               const rocke_attn_bwd_stage_table_t* t);
bool rocke_attn_bwd_stage_table_emit_stage_begin(rocke_ir_builder_t* b,
                                                 const rocke_attn_bwd_stage_table_t* t,
                                                 const char* name);
bool rocke_attn_bwd_stage_table_emit_stage_end(rocke_ir_builder_t* b,
                                               const rocke_attn_bwd_stage_table_t* t,
                                               const char* name,
                                               bool prefetch,
                                               bool prev_atomics);

/* --------------------------------------------------------- DqStageTable */

typedef struct rocke_attn_bwd_dq_stage_table
{
    const char* arch;
    int head_size;
    int k_m0;
    int k_n0;
    int waves;
    int atom_g02;
    int atom_g4;
    const char* dtype;
    const char* transpose_source;
    int kv_pad;
    bool kv_prefetch;
    int stage_vec;
    const char* sched;
} rocke_attn_bwd_dq_stage_table_t;

rocke_attn_bwd_dq_stage_table_t rocke_attn_bwd_dq_stage_table_default(void);
rocke_status_t rocke_attn_bwd_dq_stage_table_init(rocke_attn_bwd_dq_stage_table_t* t,
                                                  rocke_attn_bwd_err_t* err);

/* The BwdDqGeometry attributes DqStageTable.from_geometry reads (arch ..
 * transpose_source are required; kv_pad / stage_vec ROCKE_ATTN_BWD_NONE,
 * has_kv_prefetch false and sched NULL are absent). */
typedef struct rocke_attn_bwd_dq_sched_geom
{
    const char* arch;
    int head_size;
    int k_m0;
    int k_n0;
    int waves;
    int atom_g02;
    int atom_g4;
    const char* dtype;
    const char* transpose_source;
    int kv_pad;
    bool has_kv_prefetch;
    bool kv_prefetch;
    int stage_vec;
    const char* sched;
} rocke_attn_bwd_dq_sched_geom_t;

rocke_status_t
    rocke_attn_bwd_dq_stage_table_from_geometry(const rocke_attn_bwd_dq_sched_geom_t* geom,
                                                rocke_attn_bwd_dq_stage_table_t* out,
                                                rocke_attn_bwd_err_t* err);

rocke_status_t rocke_attn_bwd_dq_stage_table_wave_size(const rocke_attn_bwd_dq_stage_table_t* t,
                                                       int* out,
                                                       rocke_attn_bwd_err_t* err);
bool rocke_attn_bwd_dq_stage_table_setprio(const rocke_attn_bwd_dq_stage_table_t* t);
bool rocke_attn_bwd_dq_stage_table_table(const rocke_attn_bwd_dq_stage_table_t* t);
bool rocke_attn_bwd_dq_stage_table_fenced(const rocke_attn_bwd_dq_stage_table_t* t);

/* mfma: {"g0", "g2", "g4"}. */
typedef struct rocke_attn_bwd_dq_mfma
{
    int g0, g2, g4;
} rocke_attn_bwd_dq_mfma_t;

rocke_status_t rocke_attn_bwd_dq_stage_table_mfma(const rocke_attn_bwd_dq_stage_table_t* t,
                                                  rocke_attn_bwd_dq_mfma_t* out,
                                                  rocke_attn_bwd_err_t* err);

/* prefetch(): present false is {}; keys vmem_read, ds_write. */
typedef struct rocke_attn_bwd_dq_prefetch
{
    bool present;
    int vmem_read;
    int ds_write;
} rocke_attn_bwd_dq_prefetch_t;

rocke_status_t rocke_attn_bwd_dq_stage_table_prefetch(const rocke_attn_bwd_dq_stage_table_t* t,
                                                      rocke_attn_bwd_dq_prefetch_t* out,
                                                      rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_dq_stage_table_stages(const rocke_attn_bwd_dq_stage_table_t* t,
                                                    rocke_attn_bwd_stage_counts_t out[3],
                                                    rocke_attn_bwd_stage_pairs_store_t* store,
                                                    rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_dq_stage_table_stage(const rocke_attn_bwd_dq_stage_table_t* t,
                                                   const char* name,
                                                   rocke_attn_bwd_stage_counts_t* out,
                                                   rocke_attn_bwd_stage_pairs_store_t* store,
                                                   rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_dq_stage_table_groups(const rocke_attn_bwd_dq_stage_table_t* t,
                                                    const char* name,
                                                    rocke_attn_bwd_sched_pair_t* out,
                                                    int cap,
                                                    int* n_out,
                                                    rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_dq_stage_table_hint_sequence(const rocke_attn_bwd_dq_stage_table_t* t,
                                                           rocke_attn_bwd_hint_t* out,
                                                           int cap,
                                                           int* n_out,
                                                           rocke_attn_bwd_err_t* err);
/* regions(): the three stages (has_lead true). */
rocke_status_t rocke_attn_bwd_dq_stage_table_regions(const rocke_attn_bwd_dq_stage_table_t* t,
                                                     rocke_attn_bwd_sched_region_t out[3],
                                                     rocke_attn_bwd_err_t* err);

/* ir_count_model(): the main table's keys without mfma_total, pipeline and
 * ring_waits; mfma is the dq dict and prefetch the dq prefetch. */
typedef struct rocke_attn_bwd_dq_ir_count_model
{
    int sched_group_barrier;
    int sched_barrier;
    int s_setprio;
    int iglp_opt;
    int s_waitcnt;
    int n_group_counts;
    rocke_attn_bwd_sched_pair_t group_counts[5];
    rocke_attn_bwd_dq_mfma_t mfma;
    int stage_mfma[3];
    rocke_attn_bwd_dq_prefetch_t prefetch;
} rocke_attn_bwd_dq_ir_count_model_t;

rocke_status_t
    rocke_attn_bwd_dq_stage_table_ir_count_model(const rocke_attn_bwd_dq_stage_table_t* t,
                                                 rocke_attn_bwd_dq_ir_count_model_t* out,
                                                 rocke_attn_bwd_err_t* err);

bool rocke_attn_bwd_dq_stage_table_emit_loop_head(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_dq_stage_table_t* t);
bool rocke_attn_bwd_dq_stage_table_emit_stage_begin(rocke_ir_builder_t* b,
                                                    const rocke_attn_bwd_dq_stage_table_t* t,
                                                    const char* name);
bool rocke_attn_bwd_dq_stage_table_emit_stage_end(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_dq_stage_table_t* t,
                                                  const char* name);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_SCHED_H */
