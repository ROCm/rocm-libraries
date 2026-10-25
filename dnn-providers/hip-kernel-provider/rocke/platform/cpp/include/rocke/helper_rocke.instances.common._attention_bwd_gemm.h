/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_gemm.h -- C port of the
 * multi-wave block GEMM of the attention backward tile step
 * (library/kernels/common/_attention_bwd_gemm.py).
 *
 *   Python (_attention_bwd_gemm.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   LdsOperand (+ __post_init__)        rocke_attn_bwd_lds_operand_t (+ _default, _check)
 *   Frags ([frag][k_step] lists)        rocke_attn_bwd_frags_t (row-major)
 *   Frags | LdsOperand                  rocke_attn_bwd_gemm_operand_t
 *   BlockGemm (__post_init__)           rocke_attn_bwd_block_gemm_t (+ _init)
 *     derived geometry                  rocke_attn_bwd_block_gemm_waves / _frags_m / ...
 *     wave_origin / zero_acc            rocke_attn_bwd_block_gemm_wave_origin / _zero_acc
 *     load_a / load_b                   rocke_attn_bwd_block_gemm_load_a / _load_b
 *     load_all_a / load_all_b           rocke_attn_bwd_block_gemm_load_all_a / _load_all_b
 *     load_operands                     rocke_attn_bwd_block_gemm_load_operands
 *     mma                               rocke_attn_bwd_block_gemm_mma (+ mma_opts)
 *     load_k_slice / mma_k_sliced       rocke_attn_bwd_block_gemm_load_k_slice / _mma_k_sliced
 *     mma_frag / acc_coords             rocke_attn_bwd_block_gemm_mma_frag / _acc_coords
 *   mma_agpr_asm / mma_agpr_asm_steps   rocke_attn_bwd_mma_agpr_asm / _mma_agpr_asm_steps
 *     steps [(a_ks, b_ks), ...]         rocke_attn_bwd_asm_step_t[] (+ _kslice_asm_steps)
 *   lane_and_wave                       rocke_attn_bwd_lane_and_wave
 *   check_relabel_grids                 rocke_attn_bwd_check_relabel_grids
 *   relabel_acc_to_a                    rocke_attn_bwd_relabel_acc_to_a
 *   kperm_b_operand_coords              rocke_attn_bwd_kperm_b_operand_coords
 *
 * Caps inputs: Python asks _attention_bwd_caps for the atom (select_bwd_atom),
 * the wave size and legal wave counts (bwd_arch_caps, check_waves), the asm
 * MFMA mnemonic (mfma_asm_mnemonic), AGPR_ASM_TAIL_NOP,
 * ASM_MFMA_VALU_LEAD_WAIT_STATES and check_transpose_read. Until the caps
 * unit lands, BlockGemm takes those answers as an explicit
 * rocke_attn_bwd_gemm_caps_t, filled by the caller for (arch, dtype, atom_k).
 * block_gemm_init checks that caps->op is what select_bwd_atom could return
 * (a 16x16xatom_k, fp32-accumulating atom of the arch's catalog for the
 * dtype, of the caps wave size) and refuses any other atom.
 *
 * Arrays: a Python list of lists is a rocke_attn_bwd_frags_t (v[r * cols + c]);
 * outputs are caller-allocated arrays of the documented size. Errors follow
 * the frag header (rocke_i_set_err raises ckc::Error; run under
 * ckc::guard_builder).
 *
 * rocke_attn_bwd_gemm_caps_fill asks the caps port
 * (helper_rocke.instances.common._attention_bwd_caps.h) those answers for
 * (arch, dtype, atom_k), as Python's BlockGemm does.
 *
 * Not ported yet: nothing of _attention_bwd_gemm.py's public surface.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_GEMM_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_GEMM_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/arch_target.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h"
#include "rocke/helper_rocke.instances.common._attention_bwd_frag.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* What _attention_bwd_caps answers for (arch, dtype, atom_k). */
typedef struct rocke_attn_bwd_gemm_caps
{
    const rocke_mma_op_t* op; /* select_bwd_atom; NULL: it raises        */
    int wave_size; /* bwd_arch_caps(arch).wave_size           */
    const int* legal_waves; /* bwd_arch_caps(arch).legal_waves         */
    int n_legal_waves; /* 0: bwd_arch_caps raises (unknown arch)  */
    const char* asm_mnemonic; /* mfma_asm_mnemonic; NULL: it raises      */
    int asm_tail_nop; /* AGPR_ASM_TAIL_NOP[atom_k]               */
    int asm_lead_wait; /* ASM_MFMA_VALU_LEAD_WAIT_STATES          */
    bool has_tr_read_b64; /* check_transpose_read(arch, 64) passes   */
} rocke_attn_bwd_gemm_caps_t;

/* The caps answers of (arch, dtype, atom_k) from the caps port: op is
 * select_bwd_atom's atom (NULL where it raises: block_gemm_init refuses it),
 * asm_mnemonic mfma_asm_mnemonic's (NULL where it raises: the asm route
 * refuses it), asm_tail_nop AGPR_ASM_TAIL_NOP[atom_k] (-1 when absent),
 * legal_waves points to static storage. An arch outside BWD_ARCHES is
 * bwd_arch_caps' ValueError (the only error; *out is zeroed then). */
rocke_status_t rocke_attn_bwd_gemm_caps_fill(const char* arch,
                                             const char* dtype,
                                             int atom_k,
                                             rocke_attn_bwd_gemm_caps_t* out,
                                             rocke_attn_bwd_err_t* err);

/* A [rows][cols] list of lists of fragments, row-major. */
typedef struct rocke_attn_bwd_frags
{
    rocke_value_t** v;
    int rows;
    int cols;
} rocke_attn_bwd_frags_t;

typedef enum rocke_attn_bwd_reader
{
    ROCKE_ATTN_BWD_READER_PLAIN = 0,
    ROCKE_ATTN_BWD_READER_TR16 = 1
} rocke_attn_bwd_reader_t;

/* LdsOperand. Exactly one of smem (a 2-D allocation) and view is non-NULL;
 * coords NULL is None. */
typedef struct rocke_attn_bwd_lds_operand
{
    rocke_value_t* smem;
    const rocke_attn_bwd_lds_view_t* view;
    rocke_attn_bwd_storage_t storage;
    int mn0;
    int k0;
    const rocke_attn_bwd_frag_coords_t* coords;
    int align;
    rocke_attn_bwd_reader_t reader;
} rocke_attn_bwd_lds_operand_t;

/* LdsOperand(smem) with the dataclass defaults (k_inner, 0, 0, None, 8, plain). */
rocke_attn_bwd_lds_operand_t rocke_attn_bwd_lds_operand_default(void);
/* __post_init__: false (error recorded) on a Python ValueError. Python runs it
 * at construction; every BlockGemm entry point that takes an LdsOperand runs
 * it again before using it, so a caller that skips it is still refused. */
bool rocke_attn_bwd_lds_operand_check(rocke_ir_builder_t* b, const rocke_attn_bwd_lds_operand_t* o);

/* Frags | LdsOperand: lds non-NULL is the LdsOperand; else regs. */
typedef struct rocke_attn_bwd_gemm_operand
{
    const rocke_attn_bwd_lds_operand_t* lds;
    rocke_attn_bwd_frags_t regs;
} rocke_attn_bwd_gemm_operand_t;

typedef struct rocke_attn_bwd_block_gemm
{
    const char* arch;
    const char* dtype;
    int m;
    int n;
    int k;
    int grid_rows;
    int grid_cols;
    int atom_k;
    rocke_attn_bwd_gemm_caps_t caps;
    const rocke_mma_op_t* op;
} rocke_attn_bwd_block_gemm_t;

bool rocke_attn_bwd_block_gemm_init(rocke_attn_bwd_block_gemm_t* g,
                                    rocke_ir_builder_t* b,
                                    const char* arch,
                                    const char* dtype,
                                    int m,
                                    int n,
                                    int k,
                                    int grid_rows,
                                    int grid_cols,
                                    int atom_k,
                                    const rocke_attn_bwd_gemm_caps_t* caps);

int rocke_attn_bwd_block_gemm_waves(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_wave_size(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_threads(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_wave_m(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_wave_n(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_frags_m(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_frags_n(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_k_steps(const rocke_attn_bwd_block_gemm_t* g);
int rocke_attn_bwd_block_gemm_mma_per_wave(const rocke_attn_bwd_block_gemm_t* g);
void rocke_attn_bwd_block_gemm_wave_rows(const rocke_attn_bwd_block_gemm_t* g,
                                         int w,
                                         int* lo,
                                         int* hi);
void rocke_attn_bwd_block_gemm_wave_cols(const rocke_attn_bwd_block_gemm_t* g,
                                         int w,
                                         int* lo,
                                         int* hi);

/* wave_origin: origin[0] = row0, origin[1] = col0. */
void rocke_attn_bwd_block_gemm_wave_origin(rocke_ir_builder_t* b,
                                           const rocke_attn_bwd_block_gemm_t* g,
                                           rocke_value_t* wave,
                                           rocke_value_t* origin[2]);
/* zero_acc: out[frags_m * frags_n]. */
void rocke_attn_bwd_block_gemm_zero_acc(rocke_ir_builder_t* b,
                                        const rocke_attn_bwd_block_gemm_t* g,
                                        rocke_value_t** out);
/* load_a: out[frags_m]; load_b: out[frags_n]. */
bool rocke_attn_bwd_block_gemm_load_a(rocke_ir_builder_t* b,
                                      const rocke_attn_bwd_block_gemm_t* g,
                                      const rocke_attn_bwd_lds_operand_t* src,
                                      rocke_value_t* lane,
                                      rocke_value_t* row0,
                                      int ks,
                                      rocke_value_t** out);
bool rocke_attn_bwd_block_gemm_load_b(rocke_ir_builder_t* b,
                                      const rocke_attn_bwd_block_gemm_t* g,
                                      const rocke_attn_bwd_lds_operand_t* src,
                                      rocke_value_t* lane,
                                      rocke_value_t* col0,
                                      int ks,
                                      rocke_value_t** out);
/* load_all_a: out[frags_m * k_steps] ([fm][ks]); load_all_b: [fn][ks]. */
bool rocke_attn_bwd_block_gemm_load_all_a(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_block_gemm_t* g,
                                          const rocke_attn_bwd_lds_operand_t* src,
                                          rocke_value_t* lane,
                                          rocke_value_t* row0,
                                          rocke_value_t** out);
bool rocke_attn_bwd_block_gemm_load_all_b(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_block_gemm_t* g,
                                          const rocke_attn_bwd_lds_operand_t* src,
                                          rocke_value_t* lane,
                                          rocke_value_t* col0,
                                          rocke_value_t** out);
/* load_operands: a_out[frags_m * k_steps], b_out[frags_n * k_steps]. lane /
 * origin NULL are None. */
bool rocke_attn_bwd_block_gemm_load_operands(rocke_ir_builder_t* b,
                                             const rocke_attn_bwd_block_gemm_t* g,
                                             const rocke_attn_bwd_gemm_operand_t* a,
                                             const rocke_attn_bwd_gemm_operand_t* bop,
                                             rocke_value_t* lane,
                                             rocke_value_t* const* origin,
                                             rocke_value_t** a_out,
                                             rocke_value_t** b_out);

typedef enum rocke_attn_bwd_mma_route
{
    ROCKE_ATTN_BWD_ROUTE_INTRINSIC = 0,
    ROCKE_ATTN_BWD_ROUTE_AGPR_ASM = 1
} rocke_attn_bwd_mma_route_t;

typedef enum rocke_attn_bwd_asm_cluster
{
    ROCKE_ATTN_BWD_CLUSTER_K_STEP = 0,
    ROCKE_ATTN_BWD_CLUSTER_GEMM = 1
} rocke_attn_bwd_asm_cluster_t;

/* Keyword arguments of BlockGemm.mma. has_* false is None. pre_issue NULL is
 * None; it is called with pre_issue_ctx. */
typedef struct rocke_attn_bwd_mma_opts
{
    rocke_attn_bwd_mma_route_t route;
    bool has_asm_tail_nop;
    int asm_tail_nop;
    rocke_attn_bwd_asm_cluster_t asm_cluster;
    bool has_asm_agpr_base;
    int asm_agpr_base;
    bool has_asm_lead_wait;
    int asm_lead_wait;
    void (*pre_issue)(rocke_ir_builder_t* b, void* ctx);
    void* pre_issue_ctx;
} rocke_attn_bwd_mma_opts_t;

rocke_attn_bwd_mma_opts_t rocke_attn_bwd_mma_opts_default(void);

/* mma. acc NULL is None (asm route only), else frags_m x frags_n; out holds
 * frags_m * frags_n. */
bool rocke_attn_bwd_block_gemm_mma(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_block_gemm_t* g,
                                   const rocke_attn_bwd_frags_t* acc,
                                   const rocke_attn_bwd_gemm_operand_t* a,
                                   const rocke_attn_bwd_gemm_operand_t* bop,
                                   rocke_value_t* lane,
                                   rocke_value_t* const* origin,
                                   const rocke_attn_bwd_mma_opts_t* opts,
                                   rocke_value_t** out);

/* One K slice: a[j * frags_m + fm], b[j * frags_n + fn] for its n steps. */
typedef struct rocke_attn_bwd_kslice
{
    rocke_value_t** a;
    rocke_value_t** b;
    int n;
} rocke_attn_bwd_kslice_t;

/* K steps per slice of k_slice keys, or 0 (error recorded). */
int rocke_attn_bwd_block_gemm_check_k_slice(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_block_gemm_t* g,
                                            int k_slice);
/* load_k_slice: out->a / out->b must hold per * frags_m / per * frags_n. */
bool rocke_attn_bwd_block_gemm_load_k_slice(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_block_gemm_t* g,
                                            const rocke_attn_bwd_gemm_operand_t* a,
                                            const rocke_attn_bwd_gemm_operand_t* bop,
                                            int s,
                                            int k_slice,
                                            rocke_value_t* lane,
                                            rocke_value_t* const* origin,
                                            rocke_attn_bwd_kslice_t* out);
/* mma_k_sliced. first NULL is None. */
bool rocke_attn_bwd_block_gemm_mma_k_sliced(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_block_gemm_t* g,
                                            const rocke_attn_bwd_frags_t* acc,
                                            const rocke_attn_bwd_gemm_operand_t* a,
                                            const rocke_attn_bwd_gemm_operand_t* bop,
                                            int k_slice,
                                            int buffers,
                                            rocke_value_t* lane,
                                            rocke_value_t* const* origin,
                                            const rocke_attn_bwd_kslice_t* first,
                                            rocke_value_t** out);
rocke_value_t* rocke_attn_bwd_block_gemm_mma_frag(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_block_gemm_t* g,
                                                  rocke_value_t* acc,
                                                  const rocke_attn_bwd_gemm_operand_t* a,
                                                  const rocke_attn_bwd_gemm_operand_t* bop,
                                                  int fm,
                                                  int fn,
                                                  rocke_value_t* lane,
                                                  rocke_value_t* const* origin);
/* acc_coords: out[slot * 2 + 0] = row, [slot * 2 + 1] = col (c_frag_len slots). */
bool rocke_attn_bwd_block_gemm_acc_coords(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_block_gemm_t* g,
                                          rocke_value_t* lane,
                                          rocke_value_t* const* origin,
                                          int fm,
                                          int fn,
                                          rocke_value_t** out);

/* One (a_ks, b_ks) pair of mma_agpr_asm_steps: a[n_a], b[n_b]. The lengths
 * are Python's len(a_ks) / len(b_ks); they must be frags_m / frags_n. */
typedef struct rocke_attn_bwd_asm_step
{
    rocke_value_t* const* a;
    int n_a;
    rocke_value_t* const* b;
    int n_b;
} rocke_attn_bwd_asm_step_t;

/* The (a_ks, b_ks) pairs of a load_k_slice result, in K order (no IR).
 * out must hold slice->n entries; returns slice->n. */
int rocke_attn_bwd_kslice_asm_steps(const rocke_attn_bwd_block_gemm_t* g,
                                    const rocke_attn_bwd_kslice_t* slice,
                                    rocke_attn_bwd_asm_step_t* out);

/* mma_agpr_asm_steps. acc NULL is None. has_* false is None; lead_wait is the
 * value (Python's default is caps.asm_lead_wait). out[frags_m * frags_n]. */
bool rocke_attn_bwd_mma_agpr_asm_steps(rocke_ir_builder_t* b,
                                       const rocke_attn_bwd_block_gemm_t* g,
                                       const rocke_attn_bwd_frags_t* acc,
                                       const rocke_attn_bwd_asm_step_t* steps,
                                       int n_steps,
                                       bool has_tail_nop,
                                       int tail_nop,
                                       bool has_agpr_base,
                                       int agpr_base,
                                       int lead_wait,
                                       rocke_value_t** out);
bool rocke_attn_bwd_mma_agpr_asm(rocke_ir_builder_t* b,
                                 const rocke_attn_bwd_block_gemm_t* g,
                                 const rocke_attn_bwd_frags_t* acc,
                                 rocke_value_t* const* a_ks,
                                 int n_a,
                                 rocke_value_t* const* b_ks,
                                 int n_b,
                                 bool has_tail_nop,
                                 int tail_nop,
                                 bool has_agpr_base,
                                 int agpr_base,
                                 int lead_wait,
                                 rocke_value_t** out);

/* lane_and_wave. tid NULL: thread_id_x is read here. */
void rocke_attn_bwd_lane_and_wave(rocke_ir_builder_t* b,
                                  int wave_size,
                                  rocke_value_t* tid,
                                  rocke_value_t** tid_out,
                                  rocke_value_t** lane,
                                  rocke_value_t** wave);

bool rocke_attn_bwd_check_relabel_grids(rocke_ir_builder_t* b,
                                        const rocke_attn_bwd_block_gemm_t* producer,
                                        const rocke_attn_bwd_block_gemm_t* consumer,
                                        rocke_attn_bwd_relabel_plan_t* plan);
/* relabel_acc_to_a: acc is the producer's [frags_m][frags_n]; out holds the
 * consumer's [frags_m][k_steps]. plan may be NULL. */
bool rocke_attn_bwd_relabel_acc_to_a(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_block_gemm_t* producer,
                                     const rocke_attn_bwd_block_gemm_t* consumer,
                                     const rocke_attn_bwd_frags_t* acc,
                                     rocke_value_t* lane,
                                     const char* bf16_cast,
                                     rocke_value_t** out,
                                     rocke_attn_bwd_relabel_plan_t* plan);
bool rocke_attn_bwd_kperm_b_operand_coords(rocke_ir_builder_t* b,
                                           const rocke_attn_bwd_relabel_plan_t* plan,
                                           const rocke_attn_bwd_block_gemm_t* producer,
                                           const rocke_attn_bwd_block_gemm_t* consumer,
                                           rocke_attn_bwd_frag_coords_t* out);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_GEMM_H */
