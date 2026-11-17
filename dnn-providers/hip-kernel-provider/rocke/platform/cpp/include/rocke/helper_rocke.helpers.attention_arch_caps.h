/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.attention_arch_caps.h -- C99 port of
 * rocke.helpers.attention_arch_caps: the per-arch capability table the
 * attention kernels, forward and backward, query instead of hard-coding a
 * fact (LDS-DMA widths, the transpose-read width, native fp32 global atomic
 * add, and the inline-asm MFMA mnemonics and hazard wait states).
 *
 *   Python                                  C99
 *   ATTN_CAPS_ARCHES                        rocke_attn_caps_arches[ROCKE_ATTN_CAPS_ARCHES_N]
 *   ASM_MFMA_VALU_LEAD_WAIT_STATES          ROCKE_ATTN_ASM_MFMA_VALU_LEAD_WAIT_STATES
 *   ASM_TAIL_NOP_MAX                        ROCKE_ATTN_ASM_TAIL_NOP_MAX
 *   AttnArchCaps                            rocke_attn_arch_caps_t
 *     .max_lds_dma_dwords                   rocke_attn_caps_max_lds_dma_dwords
 *   attn_arch_caps(arch).max_lds_dma_dwords rocke_attn_max_lds_dma_dwords
 *     .has_tr_read_b64                      rocke_attn_caps_has_tr_read_b64
 *   attn_arch_caps(arch)                    rocke_attn_arch_caps
 *   wide_k_atom_available(arch, dtype)      rocke_attn_wide_k_atom_available
 *   check_lds_bytes                         rocke_attn_check_lds_bytes
 *   check_lds_dma_width                     rocke_attn_check_lds_dma_width
 *   check_transpose_read                    rocke_attn_check_transpose_read
 *   mfma_asm_mnemonic                       rocke_attn_mfma_asm_mnemonic
 *   mfma_result_wait_states                 rocke_attn_mfma_result_wait_states
 *   asm_tail_nop_min                        rocke_attn_asm_tail_nop_min
 *   check_asm_tail_nop                      rocke_attn_check_asm_tail_nop
 *
 * The architectural facts come from the arch catalog (rocke/arch_target.h),
 * as in Python. Each Python ValueError becomes a false return with the
 * Python message written to an optional (err, err_cap) buffer. The atom
 * enumeration (attn_atoms / select_attn_atom) is not ported: C callers query
 * the MMA catalog directly.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_ARCH_CAPS_H
#define ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_ARCH_CAPS_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/arch_target.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_ATTN_CAPS_ARCHES_N 5
extern const char* const rocke_attn_caps_arches[ROCKE_ATTN_CAPS_ARCHES_N];

#define ROCKE_ATTN_ASM_MFMA_VALU_LEAD_WAIT_STATES 2
#define ROCKE_ATTN_ASM_TAIL_NOP_MAX 15

/* Room for every transpose-read width (only the 64-bit form today). */
#define ROCKE_ATTN_TR_READ_WIDTHS_N 1

typedef struct rocke_attn_arch_caps
{
    const char* arch;
    int wave_size;
    const char* matrix_path; /* "mfma" | "wmma" */
    int lds_capacity_bytes;
    int lds_dma_dwords[ROCKE_LDS_DMA_ENCODED_DWORDS_N]; /* ascending */
    int n_lds_dma_dwords; /* 0 = no LDS DMA */
    int tr_read_bits[ROCKE_ATTN_TR_READ_WIDTHS_N];
    int n_tr_read_bits; /* 0 = no transpose read */
    bool has_agprs;
    int arch_vgprs; /* per-lane arch-VGPR budget */
    int agprs; /* per-lane AGPR budget (0 on RDNA) */
    bool native_fp32_atomic_add;
} rocke_attn_arch_caps_t;

/* attn_arch_caps(arch). False (and the Python message) for an arch with no
 * attention capability row. */
bool rocke_attn_arch_caps(const char* arch, rocke_attn_arch_caps_t* out, char* err, size_t err_cap);

/* AttnArchCaps.max_lds_dma_dwords / .has_tr_read_b64. */
int rocke_attn_caps_max_lds_dma_dwords(const rocke_attn_arch_caps_t* caps);
bool rocke_attn_caps_has_tr_read_b64(const rocke_attn_arch_caps_t* caps);

/* attn_arch_caps(arch).max_lds_dma_dwords; 0 for an arch with no row (Python
 * raises). */
int rocke_attn_max_lds_dma_dwords(const char* arch);

/* wide_k_atom_available(arch, dtype): the native 16x16x32 atom exists. False
 * for an arch with no row (Python raises). */
bool rocke_attn_wide_k_atom_available(const char* arch, const char* dtype);

bool rocke_attn_check_lds_bytes(const char* arch, long nbytes, char* err, size_t err_cap);
bool rocke_attn_check_lds_dma_width(const char* arch, int dwords, char* err, size_t err_cap);
bool rocke_attn_check_transpose_read(const char* arch, int bits, char* err, size_t err_cap);

/* mfma_asm_mnemonic: writes the mnemonic (a static string) to *out. */
bool rocke_attn_mfma_asm_mnemonic(
    const char* arch, const char* dtype, int atom_k, const char** out, char* err, size_t err_cap);
bool rocke_attn_mfma_result_wait_states(
    const char* arch, int atom_k, int* out, char* err, size_t err_cap);
bool rocke_attn_asm_tail_nop_min(
    const char* arch, const char* dtype, int atom_k, int* out, char* err, size_t err_cap);
bool rocke_attn_check_asm_tail_nop(
    const char* arch, const char* dtype, int atom_k, int nop, char* err, size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_ARCH_CAPS_H */
