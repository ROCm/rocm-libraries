/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * helper_ck_dsl.helpers.schedule.h -- C99 port of ck_dsl/helpers/schedule.py.
 *
 * Faithful translation of:
 *   - HotLoopInstList  (frozen dataclass + from_geometry + derived @property)
 *   - SchedulePolicy   (frozen dataclass + for_pipeline + emit_* IR emitters)
 *
 * The emit_* methods build IR by calling the C builder (ckc_b_s_setprio,
 * ckc_b_sched_group_barrier, ckc_b_sched_barrier). They reproduce the exact
 * Python builder-call sequence so the lowered op stream is byte-identical.
 *
 * Dependency: MfmaAtom (ck_dsl.helpers.atoms) is a sibling port; we bind to its
 * declared struct ckc_mfma_atom_t / accessor prototypes in the atoms helper
 * header. Only the fields read by from_geometry are used:
 *   dtype_in, k_per_xdlops, m, n, mfma_cycle, is_f4f6.
 */
#ifndef CKC_HELPER_CK_DSL_HELPERS_SCHEDULE_H
#define CKC_HELPER_CK_DSL_HELPERS_SCHEDULE_H

#include <stdbool.h>

#include "ckc/ir.h"

/* MfmaAtom (ck_dsl.helpers.atoms) is a sibling port. We do NOT hard-include its
 * header here so this TU still syntax-checks before that peer lands; instead we
 * forward-declare the opaque atom struct and the small set of field accessors
 * from_geometry needs. When ckc/helper_ck_dsl.helpers.atoms.h is included first
 * (it defines CKC_HELPER_CK_DSL_HELPERS_ATOMS_H), these forward declarations are
 * suppressed so the real definitions win. The accessor prototypes below are the
 * contract this port binds to. */
#ifndef CKC_HELPER_CK_DSL_HELPERS_ATOMS_H
typedef struct ckc_mfma_atom ckc_mfma_atom_t;

/* MfmaAtom field/property accessors used by from_geometry. WS3 C++ build: these
 * fallback prototypes must carry the same C linkage as the real ones in
 * helper_ck_dsl.helpers.atoms.h (which are inside its extern "C" block); without
 * this guard C++ flags a conflicting-linkage redeclaration. No effect in C. */
#ifdef __cplusplus
extern "C" {
#endif
const char* ckc_mfma_atom_dtype_in(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_m(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_n(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_k_per_xdlops(const ckc_mfma_atom_t* atom);
int ckc_mfma_atom_mfma_cycle(const ckc_mfma_atom_t* atom);
bool ckc_mfma_atom_is_f4f6(const ckc_mfma_atom_t* atom);
#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* CKC_HELPER_CK_DSL_HELPERS_ATOMS_H */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- *
 * sched_group_barrier instruction-class masks (Python module constants).
 * ------------------------------------------------------------------------- */
#define CKC_SCHED_VALU 0x002      /* vector ALU                     */
#define CKC_SCHED_SALU 0x004      /* scalar ALU                     */
#define CKC_SCHED_MFMA 0x008      /* matrix-fused multiply-add      */
#define CKC_SCHED_VMEM_READ 0x020 /* global / buffer load           */
#define CKC_SCHED_VMEM_WRITE 0x040
#define CKC_SCHED_DS_READ 0x100  /* LDS load                        */
#define CKC_SCHED_DS_WRITE 0x200 /* LDS store                       */
#define CKC_SCHED_TRANS 0x400    /* transcendentals                 */

/* Element storage size (bytes) for a dtype tag. Returns -1 for an unknown
 * dtype (Python raises ValueError); callers that need the raise semantics
 * check the return and set the builder error. Mirrors _DTYPE_BYTES. */
int ckc_schedule_dtype_bytes(const char* dtype);

/* ------------------------------------------------------------------------- *
 * HotLoopInstList -- frozen dataclass.
 * ------------------------------------------------------------------------- */
typedef struct ckc_hotloop_inst_list
{
    /* --- raw geometry inputs (mirrors the C++ template params) --- */
    int block_size;
    int m_per_block;
    int n_per_block;
    int k_per_block;
    int a_buffer_load_width;
    int b_buffer_load_width;
    int a_lds_write_width;
    int b_lds_write_width;
    int a_lds_read_width;
    int b_lds_read_width;
    int m_repeat;
    int n_repeat;
    int m_per_xdl;
    int n_per_xdl;
    int k_per_xdl;
    int a_dtype_bytes;
    int b_dtype_bytes;
    int a_packed_size;
    int b_packed_size;
    int mfma_cycle;
    bool is_f4f6;

    /* --- derived instruction counts (filled by from_geometry) --- */
    int wave_num_m;
    int wave_num_n;
    int wave_size;
    int a_buffer_load_inst_num;
    int b_buffer_load_inst_num;
    int a_lds_write_inst_num;
    int b_lds_write_inst_num;
    int a_lds_read_inst_num;
    int b_lds_read_inst_num;
    int c_mfma_inst_num;
} ckc_hotloop_inst_list_t;

/* Optional-int sentinel: pass CKC_HLIL_UNSET to take the Python `None` default
 * (LDS read/write widths default to atom.k_per_xdlops). */
#define CKC_HLIL_UNSET (-1)

/* HotLoopInstList.from_geometry. The optional widths take CKC_HLIL_UNSET to
 * mean "None" (defaults to k_pack = atom.k_per_xdlops). a_dtype/b_dtype may be
 * NULL to default to atom.dtype_in.
 *
 * On an unknown operand dtype (no byte-size) the builder error is set and the
 * returned struct's a_dtype_bytes/b_dtype_bytes is -1 (Python raises
 * ValueError). Returns by value; `b` is used only for error reporting. */
ckc_hotloop_inst_list_t
ckc_hotloop_inst_list_from_geometry(ckc_ir_builder_t* b,
                                    const ckc_mfma_atom_t* atom,
                                    int block_size,
                                    int m_per_block,
                                    int n_per_block,
                                    int k_per_block,
                                    int m_repeat,
                                    int n_repeat,
                                    int a_buffer_load_width,
                                    int b_buffer_load_width,
                                    int a_lds_write_width /* CKC_HLIL_UNSET => None */,
                                    int b_lds_write_width /* CKC_HLIL_UNSET => None */,
                                    int a_lds_read_width /* CKC_HLIL_UNSET => None */,
                                    int b_lds_read_width /* CKC_HLIL_UNSET => None */,
                                    const char* a_dtype /* NULL => atom.dtype_in */,
                                    const char* b_dtype /* NULL => atom.dtype_in */,
                                    int a_packed_size /* default 1 */,
                                    int b_packed_size /* default 1 */);

/* ---- ds_read2 16-byte heuristic + issue/rate derivations (@property) ---- */
bool ckc_hlil_a_read16(const ckc_hotloop_inst_list_t* il);
bool ckc_hlil_b_read16(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_num_ds_read_inst_a(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_num_ds_read_inst_b(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_ds_read_a_issue_cycle(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_ds_read_b_issue_cycle(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_ds_read_a_mfma_rate(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_ds_read_b_mfma_rate(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_num_dsread_a_mfma(const ckc_hotloop_inst_list_t* il);
int ckc_hlil_num_dsread_b_mfma(const ckc_hotloop_inst_list_t* il);

/* ------------------------------------------------------------------------- *
 * SchedulePolicy -- frozen dataclass.
 * ------------------------------------------------------------------------- */

/* Optional-int sentinel for setprio_level (Python `None`). */
#define CKC_SCHED_SETPRIO_NONE (-1)

typedef struct ckc_schedule_policy
{
    const char* name;      /* arena/static owned; default "mem"        */
    bool emit_hints;       /* default false                            */
    int setprio_level;     /* CKC_SCHED_SETPRIO_NONE => None           */
    const char* mode;      /* "default" | "intrawave" | "interwave"    */
    int compute_high_prio; /* default 1                                */
    int compute_low_prio;  /* default 0                                */
} ckc_schedule_policy_t;

/* Default-constructed SchedulePolicy (matches the frozen dataclass defaults). */
ckc_schedule_policy_t ckc_schedule_policy_default(void);

/* SchedulePolicy.for_pipeline. On an unknown pipeline the builder error is set
 * (Python raises ValueError) and a default-constructed policy is returned. */
ckc_schedule_policy_t ckc_schedule_policy_for_pipeline(ckc_ir_builder_t* b, const char* pipeline);

void ckc_schedule_policy_emit_prologue(const ckc_schedule_policy_t* p, ckc_ir_builder_t* b);
void ckc_schedule_policy_emit_compute_prologue(const ckc_schedule_policy_t* p, ckc_ir_builder_t* b);
void ckc_schedule_policy_emit_compute_epilogue(const ckc_schedule_policy_t* p, ckc_ir_builder_t* b);

void ckc_schedule_policy_emit_after_mfma_step(const ckc_schedule_policy_t* p,
                                              ckc_ir_builder_t* b,
                                              int ds_read_count,
                                              int mfma_count);

void ckc_schedule_policy_emit_mfma_valu_pairs(const ckc_schedule_policy_t* p,
                                              ckc_ir_builder_t* b,
                                              int pairs,
                                              int valu_per_pair /* default 1 */,
                                              int group /* default 0 */);

void ckc_schedule_policy_emit_mfma_trans_pairs(const ckc_schedule_policy_t* p,
                                               ckc_ir_builder_t* b,
                                               int pairs,
                                               int trans_per_pair /* default 1 */,
                                               int group /* default 0 */);

/* emit_mfma_setprio_bookend: brackets a single MFMA emission in s_setprio(1)/(0)
 * when mode=="interwave", else calls the callback directly. The callback is a
 * no-arg-on-builder closure: emit_mfma_fn(b, user). */
void ckc_schedule_policy_emit_mfma_setprio_bookend(const ckc_schedule_policy_t* p,
                                                   ckc_ir_builder_t* b,
                                                   void (*emit_mfma_fn)(ckc_ir_builder_t* b,
                                                                        void* user),
                                                   void* user);

/* emit_hotloop_v3 / emit_compv3_hotloop (alias). force overrides emit_hints. */
void ckc_schedule_policy_emit_hotloop_v3(const ckc_schedule_policy_t* p,
                                         ckc_ir_builder_t* b,
                                         const ckc_hotloop_inst_list_t* inst_list,
                                         bool force);

/* Alias for emit_hotloop_v3 (Python: emit_compv3_hotloop = emit_hotloop_v3). */
void ckc_schedule_policy_emit_compv3_hotloop(const ckc_schedule_policy_t* p,
                                             ckc_ir_builder_t* b,
                                             const ckc_hotloop_inst_list_t* inst_list,
                                             bool force);

void ckc_schedule_policy_emit_compv4_hotloop(const ckc_schedule_policy_t* p,
                                             ckc_ir_builder_t* b,
                                             const ckc_hotloop_inst_list_t* inst_list,
                                             bool force);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CKC_HELPER_CK_DSL_HELPERS_SCHEDULE_H */
