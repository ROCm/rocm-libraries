/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C99 port of rocke.helpers.attention_arch_caps; see
 * rocke/helper_rocke.helpers.attention_arch_caps.h. Error messages match the
 * Python ValueError text byte for byte.
 */
#include "rocke/helper_rocke.helpers.attention_arch_caps.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

const char* const rocke_attn_caps_arches[ROCKE_ATTN_CAPS_ARCHES_N]
    = {"gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250"};

/* Python: _NATIVE_FP32_ATOMIC_ADD (same order as rocke_attn_caps_arches). */
static const bool k_native_fp32_atomic_add[ROCKE_ATTN_CAPS_ARCHES_N]
    = {true, true, true, true, true};

/* Python: _TR_READ_B64_BITS. */
static const int k_tr_read_b64_bits = 64;

/* Python: _MFMA_RESULT_EXTRA_WAIT_STATES. */
typedef struct
{
    const char* arch;
    int extra;
} attn_wait_row_t;
static const attn_wait_row_t k_result_extra[] = {{"gfx942", 3}, {"gfx950", 4}};

/* Python: _XDL_PASSES (atom K -> passes). */
static int xdl_passes(int atom_k, bool* ok)
{
    *ok = (atom_k == 16 || atom_k == 32);
    return 4;
}

/* Python: _MFMA_ASM. */
typedef struct
{
    int k;
    const char* dtype;
    const char* mnemonic;
} attn_asm_row_t;
static const attn_asm_row_t k_mfma_asm[] = {
    {16, "fp16", "v_mfma_f32_16x16x16_f16"},
    {16, "bf16", "v_mfma_f32_16x16x16_bf16"},
    {32, "fp16", "v_mfma_f32_16x16x32_f16"},
    {32, "bf16", "v_mfma_f32_16x16x32_bf16"},
};

static bool fail(char* err, size_t err_cap, const char* fmt, ...)
{
    if(err && err_cap)
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, err_cap, fmt, ap);
        va_end(ap);
    }
    return false;
}

/* Python repr of a tuple of ints: "()", "(1,)", "(1, 3, 4)". */
static void int_tuple_repr(const int* v, int n, char* out, size_t cap)
{
    size_t used = 0;
    used += (size_t)snprintf(out + used, used < cap ? cap - used : 0, "(");
    for(int i = 0; i < n; ++i)
        used += (size_t)snprintf(out + used, used < cap ? cap - used : 0, i ? ", %d" : "%d", v[i]);
    snprintf(out + used, used < cap ? cap - used : 0, n == 1 ? ",)" : ")");
}

static int arch_index(const char* arch)
{
    for(int i = 0; arch && i < ROCKE_ATTN_CAPS_ARCHES_N; ++i)
        if(strcmp(arch, rocke_attn_caps_arches[i]) == 0)
            return i;
    return -1;
}

static bool has_mfma_ops(const rocke_arch_target_t* t)
{
    /* Python: ArchTarget.has_mfma = any(op.family == "mma" ...). */
    for(int i = 0; i < t->mma.num_ops; ++i)
        if(t->mma.ops[i].family && strcmp(t->mma.ops[i].family, "mma") == 0)
            return true;
    return false;
}

bool rocke_attn_arch_caps(const char* arch, rocke_attn_arch_caps_t* out, char* err, size_t err_cap)
{
    const int idx = arch_index(arch);
    const rocke_arch_target_t* t = idx >= 0 ? rocke_arch_target_from_gfx(arch) : NULL;
    if(!t)
        return fail(err,
                    err_cap,
                    "no attention capability row for '%s'; known: ('gfx942', 'gfx950', "
                    "'gfx1151', 'gfx1201', 'gfx1250')",
                    arch ? arch : "None");
    if(!out)
        return true;
    memset(out, 0, sizeof(*out));
    out->arch = rocke_attn_caps_arches[idx];
    out->wave_size = t->wave_size;
    out->matrix_path = has_mfma_ops(t) ? "mfma" : "wmma";
    out->lds_capacity_bytes = t->lds_capacity_bytes;
    out->n_lds_dma_dwords = rocke_arch_async_lds_dwords(t, out->lds_dma_dwords);
    if(t->memory.has_ds_read_tr)
        out->tr_read_bits[out->n_tr_read_bits++] = k_tr_read_b64_bits;
    out->agprs = t->limits.agprs;
    out->has_agprs = out->agprs > 0;
    out->arch_vgprs = t->limits.vgprs - out->agprs;
    out->native_fp32_atomic_add = k_native_fp32_atomic_add[idx];
    return true;
}

int rocke_attn_caps_max_lds_dma_dwords(const rocke_attn_arch_caps_t* caps)
{
    int m = 0;
    for(int i = 0; caps && i < caps->n_lds_dma_dwords; ++i)
        if(caps->lds_dma_dwords[i] > m)
            m = caps->lds_dma_dwords[i];
    return m;
}

int rocke_attn_max_lds_dma_dwords(const char* arch)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, NULL, 0))
        return 0;
    return rocke_attn_caps_max_lds_dma_dwords(&caps);
}

bool rocke_attn_caps_has_tr_read_b64(const rocke_attn_arch_caps_t* caps)
{
    for(int i = 0; caps && i < caps->n_tr_read_bits; ++i)
        if(caps->tr_read_bits[i] == k_tr_read_b64_bits)
            return true;
    return false;
}

bool rocke_attn_wide_k_atom_available(const char* arch, const char* dtype)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, NULL, 0) || !dtype)
        return false;
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
    const char* family = strcmp(caps.matrix_path, "mfma") == 0 ? "mma" : "wmma";
    /* Python: attn_atoms -- the 16x16 fp32-accumulating atoms of the arch's
     * wave size with an A fragment -- then any(op.k == 32). */
    enum
    {
        k_cap = 64
    };
    const rocke_mma_op_t* ops[k_cap];
    int n = rocke_mma_catalog_enumerate(&t->mma, family, dtype, dtype, "fp32", 16, 16, ops, k_cap);
    if(n > k_cap)
        n = k_cap;
    for(int i = 0; i < n; ++i)
        if(ops[i]->k == 32 && ops[i]->wave_size == caps.wave_size && ops[i]->srcs[0].frag_len > 0)
            return true;
    return false;
}

bool rocke_attn_check_lds_bytes(const char* arch, long nbytes, char* err, size_t err_cap)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, err, err_cap))
        return false;
    if(nbytes < 0 || nbytes > caps.lds_capacity_bytes)
        return fail(err,
                    err_cap,
                    "LDS footprint %ld B exceeds the %s capacity %d B",
                    nbytes,
                    arch,
                    caps.lds_capacity_bytes);
    return true;
}

bool rocke_attn_check_lds_dma_width(const char* arch, int dwords, char* err, size_t err_cap)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, err, err_cap))
        return false;
    if(caps.n_lds_dma_dwords == 0)
        return fail(err, err_cap, "%s has no global-to-LDS DMA path", arch);
    for(int i = 0; i < caps.n_lds_dma_dwords; ++i)
        if(caps.lds_dma_dwords[i] == dwords)
            return true;
    char legal[64];
    int_tuple_repr(caps.lds_dma_dwords, caps.n_lds_dma_dwords, legal, sizeof(legal));
    return fail(err,
                err_cap,
                "LDS-DMA width of %d dword(s) is not legal on %s; legal widths: %s",
                dwords,
                arch,
                legal);
}

bool rocke_attn_check_transpose_read(const char* arch, int bits, char* err, size_t err_cap)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, err, err_cap))
        return false;
    if(bits != k_tr_read_b64_bits)
        return fail(err, err_cap, "unsupported transpose-read width %d", bits);
    if(!rocke_attn_caps_has_tr_read_b64(&caps))
        return fail(err, err_cap, "%s has no ds_read_tr16_b64", arch);
    return true;
}

bool rocke_attn_mfma_asm_mnemonic(
    const char* arch, const char* dtype, int atom_k, const char** out, char* err, size_t err_cap)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, err, err_cap))
        return false;
    if(!caps.has_agprs)
        return fail(err, err_cap, "%s has no AGPRs; the asm MFMA route is MFMA-only", arch);
    const char* mnemonic = NULL;
    for(size_t i = 0; dtype && i < sizeof(k_mfma_asm) / sizeof(k_mfma_asm[0]); ++i)
        if(k_mfma_asm[i].k == atom_k && strcmp(k_mfma_asm[i].dtype, dtype) == 0)
            mnemonic = k_mfma_asm[i].mnemonic;
    if(!mnemonic)
        return fail(
            err, err_cap, "no asm MFMA for atom K %d and %s", atom_k, dtype ? dtype : "None");
    if(atom_k == 32 && !rocke_attn_wide_k_atom_available(arch, dtype))
        return fail(err, err_cap, "the 16x16x32 MFMA atom is gfx950-only");
    if(out)
        *out = mnemonic;
    return true;
}

bool rocke_attn_mfma_result_wait_states(
    const char* arch, int atom_k, int* out, char* err, size_t err_cap)
{
    rocke_attn_arch_caps_t caps;
    if(!rocke_attn_arch_caps(arch, &caps, err, err_cap))
        return false;
    int extra = -1;
    for(size_t i = 0; i < sizeof(k_result_extra) / sizeof(k_result_extra[0]); ++i)
        if(strcmp(k_result_extra[i].arch, arch) == 0)
            extra = k_result_extra[i].extra;
    if(!caps.has_agprs || extra < 0)
        return fail(err, err_cap, "no asm MFMA hazard table for %s", arch);
    bool known = false;
    const int passes = xdl_passes(atom_k, &known);
    if(!known) /* Python raises KeyError(atom_k). */
        return fail(err, err_cap, "%d", atom_k);
    if(passes == 2)
        extra = 3;
    if(out)
        *out = passes + extra;
    return true;
}

bool rocke_attn_asm_tail_nop_min(
    const char* arch, const char* dtype, int atom_k, int* out, char* err, size_t err_cap)
{
    int ws = 0;
    if(!rocke_attn_mfma_asm_mnemonic(arch, dtype, atom_k, NULL, err, err_cap))
        return false;
    if(!rocke_attn_mfma_result_wait_states(arch, atom_k, &ws, err, err_cap))
        return false;
    if(out)
        *out = ws - 1;
    return true;
}

bool rocke_attn_check_asm_tail_nop(
    const char* arch, const char* dtype, int atom_k, int nop, char* err, size_t err_cap)
{
    int lo = 0;
    if(!rocke_attn_asm_tail_nop_min(arch, dtype, atom_k, &lo, err, err_cap))
        return false;
    if(nop < lo || nop > ROCKE_ATTN_ASM_TAIL_NOP_MAX)
        return fail(err,
                    err_cap,
                    "asm tail s_nop %d is not legal on %s for the 16x16x%d %s atom; "
                    "legal: %d..%d (ISA MFMA result wait states)",
                    nop,
                    arch,
                    atom_k,
                    dtype,
                    lo,
                    ROCKE_ATTN_ASM_TAIL_NOP_MAX);
    return true;
}
