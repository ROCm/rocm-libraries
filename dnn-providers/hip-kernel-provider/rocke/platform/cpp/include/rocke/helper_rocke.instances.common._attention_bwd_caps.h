/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_caps.h -- C port of the
 * per-arch capability facts of the attention backward kernels
 * (library/kernels/common/_attention_bwd_caps.py).
 *
 *   Python (_attention_bwd_caps.py)     C (this header)
 *   ---------------------------------   ---------------------------------------
 *   BWD_BARRIER_FENCE                   ROCKE_ATTN_BWD_BARRIER_FENCE
 *   BWD_ARCHES                          rocke_attn_bwd_arches
 *   BwdArchCaps / bwd_arch_caps         rocke_attn_bwd_caps_t /
 *                                       rocke_attn_bwd_arch_caps
 *   BwdArchCaps.max_lds_dma_dwords      rocke_attn_bwd_caps_max_lds_dma_dwords
 *   bwd_atoms                           rocke_attn_bwd_atoms
 *   select_bwd_atom                     rocke_attn_bwd_select_atom
 *   wide_k_atom_available               rocke_attn_bwd_wide_k_atom_available
 *   check_waves                         rocke_attn_bwd_check_waves
 *   check_lds_bytes                     rocke_attn_bwd_check_lds_bytes
 *   check_lds_dma_width                 rocke_attn_bwd_check_lds_dma_width
 *   check_transpose_read                rocke_attn_bwd_check_transpose_read
 *   AGPR_ASM_TAIL_NOP                   rocke_attn_bwd_agpr_asm_tail_nop
 *   ASM_MFMA_VALU_LEAD_WAIT_STATES      ROCKE_ATTN_BWD_ASM_MFMA_VALU_LEAD_WAIT_STATES
 *   mfma_result_wait_states             rocke_attn_bwd_mfma_result_wait_states
 *   asm_tail_nop_min                    rocke_attn_bwd_asm_tail_nop_min
 *   check_asm_tail_nop                  rocke_attn_bwd_check_asm_tail_nop
 *   mfma_asm_mnemonic                   rocke_attn_bwd_mfma_asm_mnemonic
 *   check_agpr_alloc                    rocke_attn_bwd_check_agpr_alloc
 *   emit_lds_dma                        rocke_attn_bwd_emit_lds_dma
 *
 * Error model: a Python ValueError (KeyError for the one dict lookup Python
 * leaves unguarded) becomes a non-OK rocke_status_t plus the Python message in
 * a rocke_attn_bwd_err_t (err may be NULL). emit_lds_dma records it on the
 * builder's sticky error instead (never throwing across the C ABI), before it
 * emits anything.
 *
 * Everything in _attention_bwd_caps.py is ported.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_CAPS_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_CAPS_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/arch_target.h" /* rocke_mma_op_t */
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Fence of every barrier of the backward kernels: the Python builders set it
 * as IRBuilder.fence_barriers, so every b.sync() carries it. The C builder has
 * no such field; every barrier of a backward kernel is emitted with
 * rocke_b_sync_fence(b, ROCKE_ATTN_BWD_BARRIER_FENCE) instead. */
#define ROCKE_ATTN_BWD_BARRIER_FENCE "compiler"

/* Python None of an Optional[int] whose 0 is a distinct value (or a distinct
 * error) in Python; also the None of an Optional[bool] (0 False, 1 True). */
#define ROCKE_ATTN_BWD_NONE INT32_MIN

/* ---------------------------------------------------------------- errors */

/* Message capacity of rocke_attn_bwd_err_t. A longer Python message is cut
 * (the message never reaches the IR). */
#define ROCKE_ATTN_BWD_ERR_CAP 2048

/* A Python exception: code ROCKE_ERR_VALUE (ValueError), ROCKE_ERR_KEY
 * (KeyError) or ROCKE_ERR_NOTIMPL (a branch the C port does not have yet),
 * and str(exc). code is ROCKE_OK when nothing was raised. */
typedef struct rocke_attn_bwd_err
{
    rocke_status_t code;
    char msg[ROCKE_ATTN_BWD_ERR_CAP];
} rocke_attn_bwd_err_t;

/* ----------------------------------------------------------------- facts */

/* BWD_ARCHES: ("gfx942", "gfx950", "gfx1151", "gfx1201"). */
#define ROCKE_ATTN_BWD_N_ARCHES 4
extern const char* const rocke_attn_bwd_arches[ROCKE_ATTN_BWD_N_ARCHES];

/* Widest tuple of BwdArchCaps (lds_dma_dwords: at most (1, 3, 4); legal_waves:
 * at most (1, 2, 4, 8)). Python's tables fix these tuples; they are not
 * inputs. */
#define ROCKE_ATTN_BWD_CAPS_TUPLE_MAX 4

/* BwdArchCaps. matrix_path is "mfma" or "wmma" (static strings). */
typedef struct rocke_attn_bwd_caps
{
    const char* arch; /* the static BWD_ARCHES spelling */
    int wave_size;
    const char* matrix_path;
    int lds_capacity_bytes;
    int lds_dma_dwords[ROCKE_ATTN_BWD_CAPS_TUPLE_MAX];
    int n_lds_dma_dwords; /* 0: () = no LDS DMA */
    bool has_tr_read_b64;
    bool has_tr_read_b128;
    bool has_agprs;
    int arch_vgprs;
    int agprs;
    bool native_fp32_atomic_add;
    int legal_waves[ROCKE_ATTN_BWD_CAPS_TUPLE_MAX];
    int n_legal_waves;
} rocke_attn_bwd_caps_t;

/* bwd_arch_caps(arch). */
rocke_status_t rocke_attn_bwd_arch_caps(const char* arch,
                                        rocke_attn_bwd_caps_t* out,
                                        rocke_attn_bwd_err_t* err);

/* BwdArchCaps.max_lds_dma_dwords (max(..., default=0)). */
int rocke_attn_bwd_caps_max_lds_dma_dwords(const rocke_attn_bwd_caps_t* caps);

/* bwd_atoms(arch, dtype): the 16x16 atoms by increasing K. Writes up to `cap`
 * op pointers (static catalog rows) to `out` and the count to *n_out; a count
 * above `cap` is refused with ROCKE_ERR_VALUE (the catalogs hold at most two
 * per dtype today). */
rocke_status_t rocke_attn_bwd_atoms(const char* arch,
                                    const char* dtype,
                                    const rocke_mma_op_t** out,
                                    int cap,
                                    int* n_out,
                                    rocke_attn_bwd_err_t* err);

/* select_bwd_atom(arch, dtype, k): the 16x16xk atom, or NULL + err. */
const rocke_mma_op_t* rocke_attn_bwd_select_atom(const char* arch,
                                                 const char* dtype,
                                                 int k,
                                                 rocke_attn_bwd_err_t* err);

/* wide_k_atom_available(arch, dtype); *out receives the bool. */
rocke_status_t rocke_attn_bwd_wide_k_atom_available(const char* arch,
                                                    const char* dtype,
                                                    bool* out,
                                                    rocke_attn_bwd_err_t* err);

/* The check_* functions: ROCKE_OK when Python returns, else the error. */
rocke_status_t rocke_attn_bwd_check_waves(const char* arch, int waves, rocke_attn_bwd_err_t* err);
rocke_status_t
    rocke_attn_bwd_check_lds_bytes(const char* arch, long long nbytes, rocke_attn_bwd_err_t* err);
rocke_status_t
    rocke_attn_bwd_check_lds_dma_width(const char* arch, int dwords, rocke_attn_bwd_err_t* err);
rocke_status_t
    rocke_attn_bwd_check_transpose_read(const char* arch, int bits, rocke_attn_bwd_err_t* err);

/* ------------------------------------------------------- asm MFMA route */

/* ASM_MFMA_VALU_LEAD_WAIT_STATES. */
#define ROCKE_ATTN_BWD_ASM_MFMA_VALU_LEAD_WAIT_STATES 2

/* AGPR_ASM_TAIL_NOP.get(atom_k): the s_nop count, or -1 for a K the mapping
 * does not hold (Python None). */
int rocke_attn_bwd_agpr_asm_tail_nop(int atom_k);

/* mfma_asm_mnemonic(arch, dtype, atom_k): the static mnemonic, or NULL + err. */
const char* rocke_attn_bwd_mfma_asm_mnemonic(const char* arch,
                                             const char* dtype,
                                             int atom_k,
                                             rocke_attn_bwd_err_t* err);

/* mfma_result_wait_states(arch, atom_k) / asm_tail_nop_min(arch, dtype,
 * atom_k): the count, or -1 + err. */
int rocke_attn_bwd_mfma_result_wait_states(const char* arch, int atom_k, rocke_attn_bwd_err_t* err);
int rocke_attn_bwd_asm_tail_nop_min(const char* arch,
                                    const char* dtype,
                                    int atom_k,
                                    rocke_attn_bwd_err_t* err);

/* check_asm_tail_nop(arch, dtype, atom_k, nop). Python also refuses a
 * non-int `nop`; a C int is always one. */
rocke_status_t rocke_attn_bwd_check_asm_tail_nop(
    const char* arch, const char* dtype, int atom_k, int nop, rocke_attn_bwd_err_t* err);

/* --------------------------------------------------------- agpr_alloc */

/* The agpr_alloc knob: None, a string (the legal one is "backend") or an
 * (a, b) pair. */
typedef enum rocke_attn_bwd_agpr_alloc_kind
{
    ROCKE_ATTN_BWD_AGPR_NONE = 0,
    ROCKE_ATTN_BWD_AGPR_STR,
    ROCKE_ATTN_BWD_AGPR_PAIR
} rocke_attn_bwd_agpr_alloc_kind_t;

typedef struct rocke_attn_bwd_agpr_alloc
{
    rocke_attn_bwd_agpr_alloc_kind_t kind;
    const char* str; /* kind STR */
    int a, b; /* kind PAIR */
} rocke_attn_bwd_agpr_alloc_t;

/* check_agpr_alloc(arch, agpr_alloc). */
rocke_status_t rocke_attn_bwd_check_agpr_alloc(const char* arch,
                                               rocke_attn_bwd_agpr_alloc_t agpr_alloc,
                                               rocke_attn_bwd_err_t* err);

/* ---------------------------------------------------------- emit_lds_dma */

/* emit_lds_dma(b, arch=, rsrc=, lds_ptr=, voffset=, soffset=, dwords=,
 * coherency=): check_lds_dma_width first (a refusal sets the builder's sticky
 * error, ROCKE_ERR_VALUE, and emits nothing), then
 * b.async_buffer_load_lds(rsrc, lds_ptr, voffset, soffset, dwords, coherency).
 * Returns ROCKE_OK, or the error recorded on the builder (a refusal, the
 * builder's own coherency check, or a builder that had already failed). */
rocke_status_t rocke_attn_bwd_emit_lds_dma(rocke_ir_builder_t* b,
                                           const char* arch,
                                           rocke_value_t* rsrc,
                                           rocke_value_t* lds_ptr,
                                           rocke_value_t* voffset,
                                           rocke_value_t* soffset,
                                           int dwords,
                                           int coherency);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* C++-only internal API for the other backward twins (throwing ckc::Error,
 * like the Python raises). */
#include <string>
#include <vector>

namespace rocke_attn_bwd
{
/* The code points of UTF-8 text (NotImplError on bytes that are not UTF-8). */
std::vector<unsigned> py_codepoints(const std::string& s);
/* Python repr() of a str (the quoting and escapes CPython uses); NotImplError
 * beyond U+00FF (the Unicode printable table is not ported). */
std::string py_repr(const std::string& s);
/* repr() of a tuple of str / int ("('a',)" for one element). */
std::string py_tuple(const std::vector<std::string>& items);
std::string py_tuple(const std::vector<int>& items);
/* repr() of a list of int. */
std::string py_list(const std::vector<int>& items);
/* Copy a caught ckc::Error (or bad_alloc) into err; returns its code. */
rocke_status_t store_err(rocke_attn_bwd_err_t* err, rocke_status_t code, const char* msg);

const rocke_attn_bwd_caps_t& arch_caps(const std::string& arch);
std::string norm_dtype(const std::string& dtype);
std::vector<const rocke_mma_op_t*> atoms(const std::string& arch, const std::string& dtype);
const rocke_mma_op_t* select_atom(const std::string& arch, const std::string& dtype, int k);
bool wide_k_atom_available(const std::string& arch, const std::string& dtype);
void check_waves(const std::string& arch, int waves);
void check_lds_bytes(const std::string& arch, long long nbytes);
void check_lds_dma_width(const std::string& arch, int dwords);
void check_transpose_read(const std::string& arch, int bits);
const char* mfma_asm_mnemonic(const std::string& arch, const std::string& dtype, int atom_k);
int mfma_result_wait_states(const std::string& arch, int atom_k);
int asm_tail_nop_min(const std::string& arch, const std::string& dtype, int atom_k);
void check_asm_tail_nop(const std::string& arch, const std::string& dtype, int atom_k, int nop);
/* repr() of an agpr_alloc value. */
std::string agpr_alloc_repr(const rocke_attn_bwd_agpr_alloc_t& a);
void check_agpr_alloc(const std::string& arch, const rocke_attn_bwd_agpr_alloc_t& a);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_CAPS_H */
