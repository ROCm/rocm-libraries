// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C port of library/kernels/common/_attention_bwd_gemm.py: the multi-wave
 * block GEMM of the attention backward tile step, its inline-asm AGPR route
 * and the C-to-A relabel between two block GEMMs.
 *
 * See rocke/helper_rocke.instances.common._attention_bwd_gemm.h for the
 * mapping and the caps inputs. Every op is created in the Python evaluation
 * order; each comment `py:` quotes the Python expression a block reproduces.
 */
#include "rocke/helper_rocke.instances.common._attention_bwd_gemm.h"

#include <stdio.h>
#include <string.h>

#include <mutex>
#include <string>
#include <vector>

#include "rocke/ir_internal.h" /* rocke_i_set_err */

static bool live(rocke_ir_builder_t* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

/* ====================================================================== *
 * LdsOperand
 * ====================================================================== */

rocke_attn_bwd_lds_operand_t rocke_attn_bwd_lds_operand_default(void)
{
    rocke_attn_bwd_lds_operand_t o;
    memset(&o, 0, sizeof(o));
    o.storage = ROCKE_ATTN_BWD_K_INNER;
    o.align = 8;
    o.reader = ROCKE_ATTN_BWD_READER_PLAIN;
    return o;
}

bool rocke_attn_bwd_lds_operand_check(rocke_ir_builder_t* b, const rocke_attn_bwd_lds_operand_t* o)
{
    if(!live(b))
        return false;
    if(o->reader != ROCKE_ATTN_BWD_READER_PLAIN && o->reader != ROCKE_ATTN_BWD_READER_TR16)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "reader must be 'plain' or 'tr16'");
        return false;
    }
    if(o->reader == ROCKE_ATTN_BWD_READER_TR16 && o->storage != ROCKE_ATTN_BWD_K_OUTER)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "the tr16 reader reads K-outer tiles only");
        return false;
    }
    return true;
}

/* ====================================================================== *
 * caps answers
 * ====================================================================== */

rocke_status_t rocke_attn_bwd_gemm_caps_fill(const char* arch,
                                             const char* dtype,
                                             int atom_k,
                                             rocke_attn_bwd_gemm_caps_t* out,
                                             rocke_attn_bwd_err_t* err)
{
    memset(out, 0, sizeof(*out));
    /* the legal wave counts of each BWD_ARCHES entry (static storage). Each
     * slot is written once, under the lock, so concurrent fills and readers
     * of out->legal_waves never race. */
    static std::mutex waves_mu;
    static bool waves_set[ROCKE_ATTN_BWD_N_ARCHES];
    static int waves[ROCKE_ATTN_BWD_N_ARCHES][ROCKE_ATTN_BWD_CAPS_TUPLE_MAX];
    rocke_attn_bwd_caps_t caps;
    rocke_status_t st = rocke_attn_bwd_arch_caps(arch, &caps, err);
    if(st != ROCKE_OK)
        return st;
    int slot = 0;
    while(slot < ROCKE_ATTN_BWD_N_ARCHES && strcmp(rocke_attn_bwd_arches[slot], caps.arch) != 0)
        ++slot;
    if(slot == ROCKE_ATTN_BWD_N_ARCHES)
    {
        if(err != NULL)
        {
            err->code = ROCKE_ERR_VALUE;
            snprintf(err->msg, sizeof(err->msg), "unknown backward arch %s", arch);
        }
        return ROCKE_ERR_VALUE;
    }
    {
        std::lock_guard<std::mutex> lock(waves_mu);
        if(!waves_set[slot])
        {
            for(int i = 0; i < caps.n_legal_waves; ++i)
                waves[slot][i] = caps.legal_waves[i];
            waves_set[slot] = true;
        }
    }
    out->wave_size = caps.wave_size;
    out->legal_waves = waves[slot];
    out->n_legal_waves = caps.n_legal_waves;
    out->has_tr_read_b64 = caps.has_tr_read_b64;
    out->asm_lead_wait = ROCKE_ATTN_BWD_ASM_MFMA_VALU_LEAD_WAIT_STATES;
    out->asm_tail_nop = rocke_attn_bwd_agpr_asm_tail_nop(atom_k);
    /* the raising lookups answer NULL (their callers refuse it) */
    rocke_attn_bwd_err_t scratch;
    out->op = rocke_attn_bwd_select_atom(arch, dtype, atom_k, &scratch);
    out->asm_mnemonic = rocke_attn_bwd_mfma_asm_mnemonic(arch, dtype, atom_k, &scratch);
    if(err != NULL)
        err->code = ROCKE_OK;
    return ROCKE_OK;
}

/* ====================================================================== *
 * BlockGemm geometry
 * ====================================================================== */

/* _attention_bwd_caps._norm_dtype; NULL where it raises. */
static const char* norm_dtype(const char* dtype)
{
    if(dtype == NULL)
        return NULL;
    if(strcmp(dtype, "fp16") == 0 || strcmp(dtype, "f16") == 0)
        return "fp16";
    if(strcmp(dtype, "bf16") == 0)
        return "bf16";
    return NULL;
}

static bool str_is(const char* s, const char* want)
{
    return s != NULL && strcmp(s, want) == 0;
}

/* The caps atom comes from the caller, which asks the caps port for it;
 * refuse any atom select_bwd_atom(arch, dtype, atom_k) could not return:
 * bwd_atoms keeps the 16x16 fp32-accumulating atoms of the arch's catalog
 * with a_dtype == b_dtype == _norm_dtype(dtype), the arch's wave size and a
 * non-empty A fragment; select_bwd_atom picks the one with k == atom_k. */
static bool caps_atom_ok(rocke_ir_builder_t* b,
                         const char* arch,
                         const char* dtype,
                         int atom_k,
                         const rocke_attn_bwd_gemm_caps_t* caps)
{
    const char* dt = norm_dtype(dtype);
    if(dt == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "backward operand dtype must be fp16 or bf16, got '%s'",
                        dtype ? dtype : "None");
        return false;
    }
    const rocke_mma_op_t* op = caps->op;
    const rocke_arch_target_t* t = rocke_arch_target_from_gfx(arch);
    bool in_catalog = false;
    if(t != NULL)
        for(int i = 0; i < t->mma.num_ops; ++i)
            in_catalog = in_catalog || &t->mma.ops[i] == op;
    bool ok = in_catalog && (str_is(op->family, "mma") || str_is(op->family, "wmma"))
              && str_is(op->a_dtype, dt) && str_is(op->b_dtype, dt) && str_is(op->c_dtype, "fp32")
              && op->m == 16 && op->n == 16 && op->k == atom_k && op->wave_size == caps->wave_size
              && op->a_frag_len > 0;
    if(!ok)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "caps atom '%s' is not the 16x16x%d %s atom of %s",
                        op->op_id ? op->op_id : "None",
                        atom_k,
                        dt,
                        arch);
        return false;
    }
    return true;
}

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
                                    const rocke_attn_bwd_gemm_caps_t* caps)
{
    memset(g, 0, sizeof(*g));
    if(!live(b))
        return false;
    g->arch = arch;
    g->dtype = dtype;
    g->m = m;
    g->n = n;
    g->k = k;
    g->grid_rows = grid_rows;
    g->grid_cols = grid_cols;
    g->atom_k = atom_k;
    if(caps == NULL || caps->n_legal_waves <= 0)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "attention backward does not target '%s'", arch ? arch : "None");
        return false;
    }
    g->caps = *caps;
    int waves = grid_rows * grid_cols;
    /* check_waves */
    bool legal = false;
    for(int i = 0; i < caps->n_legal_waves; ++i)
        legal = legal || caps->legal_waves[i] == waves;
    if(!legal)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "waves=%d is not legal on %s", waves, arch);
        return false;
    }
    /* py: self.grid not in ((1, waves), (waves, 1), (2, waves // 2)) */
    bool grid_ok = (grid_rows == 1 && grid_cols == waves) || (grid_rows == waves && grid_cols == 1)
                   || (grid_rows == 2 && grid_cols == waves / 2);
    if(!grid_ok)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "warp grid (%d, %d) is not one of (1, W), (W, 1), (2, W/2)",
                        grid_rows,
                        grid_cols);
        return false;
    }
    /* select_bwd_atom -> bwd_atoms -> _norm_dtype raises first */
    if(norm_dtype(dtype) == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "backward operand dtype must be fp16 or bf16, got '%s'",
                        dtype ? dtype : "None");
        return false;
    }
    const rocke_mma_op_t* op = caps->op;
    if(op == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "no native 16x16x%d %s atom on %s",
                        atom_k,
                        dtype ? dtype : "None",
                        arch);
        return false;
    }
    if(!caps_atom_ok(b, arch, dtype, atom_k, caps))
        return false;
    if(m % (op->m * grid_rows) || n % (op->n * grid_cols))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "block %dx%d does not split into %dx%d atoms over the warp grid (%d, %d)",
                        m,
                        n,
                        op->m,
                        op->n,
                        grid_rows,
                        grid_cols);
        return false;
    }
    if(k <= 0 || atom_k == 0 || k % atom_k)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "K=%d is not a multiple of the atom K %d", k, atom_k);
        return false;
    }
    g->op = op;
    return true;
}

int rocke_attn_bwd_block_gemm_waves(const rocke_attn_bwd_block_gemm_t* g)
{
    return g->grid_rows * g->grid_cols;
}
int rocke_attn_bwd_block_gemm_wave_size(const rocke_attn_bwd_block_gemm_t* g)
{
    return g->caps.wave_size;
}
int rocke_attn_bwd_block_gemm_threads(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_waves(g) * g->caps.wave_size;
}
int rocke_attn_bwd_block_gemm_wave_m(const rocke_attn_bwd_block_gemm_t* g)
{
    return g->m / g->grid_rows;
}
int rocke_attn_bwd_block_gemm_wave_n(const rocke_attn_bwd_block_gemm_t* g)
{
    return g->n / g->grid_cols;
}
int rocke_attn_bwd_block_gemm_frags_m(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_wave_m(g) / g->op->m;
}
int rocke_attn_bwd_block_gemm_frags_n(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_wave_n(g) / g->op->n;
}
int rocke_attn_bwd_block_gemm_k_steps(const rocke_attn_bwd_block_gemm_t* g)
{
    return g->k / g->atom_k;
}
int rocke_attn_bwd_block_gemm_mma_per_wave(const rocke_attn_bwd_block_gemm_t* g)
{
    return rocke_attn_bwd_block_gemm_frags_m(g) * rocke_attn_bwd_block_gemm_frags_n(g)
           * rocke_attn_bwd_block_gemm_k_steps(g);
}
void rocke_attn_bwd_block_gemm_wave_rows(const rocke_attn_bwd_block_gemm_t* g,
                                         int w,
                                         int* lo,
                                         int* hi)
{
    int r = w / g->grid_cols;
    *lo = r * rocke_attn_bwd_block_gemm_wave_m(g);
    *hi = (r + 1) * rocke_attn_bwd_block_gemm_wave_m(g);
}
void rocke_attn_bwd_block_gemm_wave_cols(const rocke_attn_bwd_block_gemm_t* g,
                                         int w,
                                         int* lo,
                                         int* hi)
{
    int c = w % g->grid_cols;
    *lo = c * rocke_attn_bwd_block_gemm_wave_n(g);
    *hi = (c + 1) * rocke_attn_bwd_block_gemm_wave_n(g);
}

#define FM(g) rocke_attn_bwd_block_gemm_frags_m(g)
#define FN(g) rocke_attn_bwd_block_gemm_frags_n(g)
#define KS(g) rocke_attn_bwd_block_gemm_k_steps(g)

/* ====================================================================== *
 * Emission
 * ====================================================================== */

void rocke_attn_bwd_block_gemm_wave_origin(rocke_ir_builder_t* b,
                                           const rocke_attn_bwd_block_gemm_t* g,
                                           rocke_value_t* wave,
                                           rocke_value_t* origin[2])
{
    rocke_value_t *r = NULL, *c = NULL;
    origin[0] = origin[1] = NULL;
    if(!live(b))
        return;
    if(g->grid_cols == 1)
        r = wave;
    else if(g->grid_rows == 1)
        c = wave;
    else
    {
        rocke_value_t* cc = rocke_b_const_i32(b, g->grid_cols);
        r = rocke_b_div(b, wave, cc);
        c = rocke_b_mod(b, wave, cc);
    }
    if(r != NULL)
    {
        rocke_value_t* cm = rocke_b_const_i32(b, rocke_attn_bwd_block_gemm_wave_m(g));
        origin[0] = rocke_b_mul(b, r, cm);
    }
    else
        origin[0] = rocke_b_const_i32(b, 0);
    if(c != NULL)
    {
        rocke_value_t* cn = rocke_b_const_i32(b, rocke_attn_bwd_block_gemm_wave_n(g));
        origin[1] = rocke_b_mul(b, c, cn);
    }
    else
        origin[1] = rocke_b_const_i32(b, 0);
}

void rocke_attn_bwd_block_gemm_zero_acc(rocke_ir_builder_t* b,
                                        const rocke_attn_bwd_block_gemm_t* g,
                                        rocke_value_t** out)
{
    for(int fm = 0; fm < FM(g); ++fm)
        for(int fn = 0; fn < FN(g); ++fn)
            out[fm * FN(g) + fn] = rocke_b_zero_vec_f32(b, g->op->c_frag_len);
}

/* _off(b, base, k) */
static rocke_value_t* off(rocke_ir_builder_t* b, rocke_value_t* base, int k)
{
    if(!k)
        return base;
    rocke_value_t* ck = rocke_b_const_i32(b, k);
    return rocke_b_add(b, base, ck);
}

/* BlockGemm._load */
static rocke_value_t* load_one(rocke_ir_builder_t* b,
                               const rocke_attn_bwd_block_gemm_t* g,
                               const rocke_attn_bwd_lds_operand_t* src,
                               const rocke_attn_bwd_frag_coords_t* coords,
                               rocke_value_t* lane,
                               rocke_value_t* mn0,
                               int ks)
{
    /* LdsOperand.__post_init__ ran at construction in Python. */
    if(!rocke_attn_bwd_lds_operand_check(b, src))
        return NULL;
    int k0 = src->k0 + ks * g->atom_k;
    if(src->reader == ROCKE_ATTN_BWD_READER_TR16)
        return rocke_attn_bwd_load_frag_lds_tr16(b,
                                                 coords,
                                                 src->view,
                                                 lane,
                                                 g->dtype,
                                                 g->caps.has_tr_read_b64,
                                                 rocke_attn_bwd_iv_val(mn0),
                                                 rocke_attn_bwd_iv_int(k0));
    return rocke_attn_bwd_load_frag_lds(b,
                                        coords,
                                        src->view != NULL ? NULL : src->smem,
                                        src->view,
                                        lane,
                                        g->dtype,
                                        rocke_attn_bwd_iv_val(mn0),
                                        rocke_attn_bwd_iv_int(k0),
                                        src->storage,
                                        src->align);
}

/* src.coords or operand_coords(op, role) */
static const rocke_attn_bwd_frag_coords_t* coords_of(rocke_ir_builder_t* b,
                                                     const rocke_attn_bwd_block_gemm_t* g,
                                                     const rocke_attn_bwd_lds_operand_t* src,
                                                     char role,
                                                     rocke_attn_bwd_frag_coords_t* tmp)
{
    if(src->coords != NULL)
        return src->coords;
    if(!rocke_attn_bwd_operand_coords(b, g->op, role, tmp))
        return NULL;
    return tmp;
}

static bool load_ab(rocke_ir_builder_t* b,
                    const rocke_attn_bwd_block_gemm_t* g,
                    const rocke_attn_bwd_lds_operand_t* src,
                    rocke_value_t* lane,
                    rocke_value_t* origin0,
                    int ks,
                    char role,
                    rocke_value_t** out)
{
    if(!live(b))
        return false;
    rocke_attn_bwd_frag_coords_t tmp;
    const rocke_attn_bwd_frag_coords_t* coords = coords_of(b, g, src, role, &tmp);
    if(coords == NULL)
        return false;
    int nf = role == 'a' ? FM(g) : FN(g);
    int step = role == 'a' ? g->op->m : g->op->n;
    for(int f = 0; f < nf; ++f)
    {
        rocke_value_t* mn0 = off(b, origin0, src->mn0 + f * step);
        out[f] = load_one(b, g, src, coords, lane, mn0, ks);
    }
    return live(b);
}

bool rocke_attn_bwd_block_gemm_load_a(rocke_ir_builder_t* b,
                                      const rocke_attn_bwd_block_gemm_t* g,
                                      const rocke_attn_bwd_lds_operand_t* src,
                                      rocke_value_t* lane,
                                      rocke_value_t* row0,
                                      int ks,
                                      rocke_value_t** out)
{
    return load_ab(b, g, src, lane, row0, ks, 'a', out);
}

bool rocke_attn_bwd_block_gemm_load_b(rocke_ir_builder_t* b,
                                      const rocke_attn_bwd_block_gemm_t* g,
                                      const rocke_attn_bwd_lds_operand_t* src,
                                      rocke_value_t* lane,
                                      rocke_value_t* col0,
                                      int ks,
                                      rocke_value_t** out)
{
    return load_ab(b, g, src, lane, col0, ks, 'b', out);
}

static bool load_all(rocke_ir_builder_t* b,
                     const rocke_attn_bwd_block_gemm_t* g,
                     const rocke_attn_bwd_lds_operand_t* src,
                     rocke_value_t* lane,
                     rocke_value_t* origin0,
                     char role,
                     rocke_value_t** out)
{
    int nf = role == 'a' ? FM(g) : FN(g);
    std::vector<rocke_value_t*> per((size_t)nf);
    for(int ks = 0; ks < KS(g); ++ks)
    {
        if(!load_ab(b, g, src, lane, origin0, ks, role, per.data()))
            return false;
        for(int f = 0; f < nf; ++f)
            out[f * KS(g) + ks] = per[f];
    }
    return true;
}

bool rocke_attn_bwd_block_gemm_load_all_a(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_block_gemm_t* g,
                                          const rocke_attn_bwd_lds_operand_t* src,
                                          rocke_value_t* lane,
                                          rocke_value_t* row0,
                                          rocke_value_t** out)
{
    return load_all(b, g, src, lane, row0, 'a', out);
}

bool rocke_attn_bwd_block_gemm_load_all_b(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_block_gemm_t* g,
                                          const rocke_attn_bwd_lds_operand_t* src,
                                          rocke_value_t* lane,
                                          rocke_value_t* col0,
                                          rocke_value_t** out)
{
    return load_all(b, g, src, lane, col0, 'b', out);
}

/* _check_operands */
static bool check_operands(rocke_ir_builder_t* b,
                           const rocke_attn_bwd_block_gemm_t* g,
                           const rocke_attn_bwd_gemm_operand_t* a,
                           const rocke_attn_bwd_gemm_operand_t* bop,
                           rocke_value_t* lane,
                           rocke_value_t* const* origin)
{
    const rocke_attn_bwd_gemm_operand_t* ops[2] = {a, bop};
    const char* names[2] = {"A", "B"};
    int counts[2] = {FM(g), FN(g)};
    for(int i = 0; i < 2; ++i)
    {
        if(ops[i]->lds != NULL)
        {
            /* LdsOperand.__post_init__ (Python: at construction) */
            if(!rocke_attn_bwd_lds_operand_check(b, ops[i]->lds))
                return false;
            if(lane == NULL || origin == NULL)
            {
                rocke_i_set_err(
                    b, ROCKE_ERR_VALUE, "LDS operand %s needs lane and origin", names[i]);
                return false;
            }
        }
        else if(ops[i]->regs.rows != counts[i]
                || (ops[i]->regs.rows > 0 && ops[i]->regs.cols != KS(g)))
        {
            rocke_i_set_err(b,
                            ROCKE_ERR_VALUE,
                            "register operand %s must be [%d][%d]",
                            names[i],
                            counts[i],
                            KS(g));
            return false;
        }
    }
    return true;
}

/* _step_operands: a_ks[frags_m], b_ks[frags_n]. */
static bool step_operands(rocke_ir_builder_t* b,
                          const rocke_attn_bwd_block_gemm_t* g,
                          const rocke_attn_bwd_gemm_operand_t* a,
                          const rocke_attn_bwd_gemm_operand_t* bop,
                          rocke_value_t* lane,
                          rocke_value_t* const* origin,
                          int ks,
                          rocke_value_t** a_ks,
                          rocke_value_t** b_ks)
{
    if(a->lds != NULL)
    {
        if(!rocke_attn_bwd_block_gemm_load_a(b, g, a->lds, lane, origin[0], ks, a_ks))
            return false;
    }
    else
        for(int fm = 0; fm < FM(g); ++fm)
            a_ks[fm] = a->regs.v[fm * a->regs.cols + ks];
    if(bop->lds != NULL)
    {
        if(!rocke_attn_bwd_block_gemm_load_b(b, g, bop->lds, lane, origin[1], ks, b_ks))
            return false;
    }
    else
        for(int fn = 0; fn < FN(g); ++fn)
            b_ks[fn] = bop->regs.v[fn * bop->regs.cols + ks];
    return live(b);
}

int rocke_attn_bwd_block_gemm_check_k_slice(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_block_gemm_t* g,
                                            int k_slice)
{
    if(!live(b))
        return 0;
    if(k_slice <= 0 || k_slice % g->atom_k || g->k % k_slice)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "K slice %d must be a positive multiple of the atom K %d that divides K=%d",
                        k_slice,
                        g->atom_k,
                        g->k);
        return 0;
    }
    return k_slice / g->atom_k;
}

bool rocke_attn_bwd_block_gemm_load_k_slice(rocke_ir_builder_t* b,
                                            const rocke_attn_bwd_block_gemm_t* g,
                                            const rocke_attn_bwd_gemm_operand_t* a,
                                            const rocke_attn_bwd_gemm_operand_t* bop,
                                            int s,
                                            int k_slice,
                                            rocke_value_t* lane,
                                            rocke_value_t* const* origin,
                                            rocke_attn_bwd_kslice_t* out)
{
    int per = rocke_attn_bwd_block_gemm_check_k_slice(b, g, k_slice);
    if(per == 0)
        return false;
    int n = g->k / k_slice;
    if(!(0 <= s && s < n))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "slice %d outside the %d slices of K=%d", s, n, g->k);
        return false;
    }
    if(!check_operands(b, g, a, bop, lane, origin))
        return false;
    for(int j = 0; j < per; ++j)
        if(!step_operands(b,
                          g,
                          a,
                          bop,
                          lane,
                          origin,
                          s * per + j,
                          out->a + (size_t)j * FM(g),
                          out->b + (size_t)j * FN(g)))
            return false;
    out->n = per;
    return true;
}

bool rocke_attn_bwd_block_gemm_load_operands(rocke_ir_builder_t* b,
                                             const rocke_attn_bwd_block_gemm_t* g,
                                             const rocke_attn_bwd_gemm_operand_t* a,
                                             const rocke_attn_bwd_gemm_operand_t* bop,
                                             rocke_value_t* lane,
                                             rocke_value_t* const* origin,
                                             rocke_value_t** a_out,
                                             rocke_value_t** b_out)
{
    if(!live(b))
        return false;
    int ks_n = KS(g);
    std::vector<rocke_value_t*> av((size_t)ks_n * FM(g)), bv((size_t)ks_n * FN(g));
    rocke_attn_bwd_kslice_t steps = {av.data(), bv.data(), 0};
    if(!rocke_attn_bwd_block_gemm_load_k_slice(b, g, a, bop, 0, g->k, lane, origin, &steps))
        return false;
    for(int fm = 0; fm < FM(g); ++fm)
        for(int j = 0; j < steps.n; ++j)
            a_out[fm * steps.n + j] = av[(size_t)j * FM(g) + fm];
    for(int fn = 0; fn < FN(g); ++fn)
        for(int j = 0; j < steps.n; ++j)
            b_out[fn * steps.n + j] = bv[(size_t)j * FN(g) + fn];
    return true;
}

/* ====================================================================== *
 * The inline-asm AGPR route
 * ====================================================================== */

int rocke_attn_bwd_kslice_asm_steps(const rocke_attn_bwd_block_gemm_t* g,
                                    const rocke_attn_bwd_kslice_t* slice,
                                    rocke_attn_bwd_asm_step_t* out)
{
    for(int j = 0; j < slice->n; ++j)
    {
        out[j].a = slice->a + (size_t)j * FM(g);
        out[j].n_a = FM(g);
        out[j].b = slice->b + (size_t)j * FN(g);
        out[j].n_b = FN(g);
    }
    return slice->n;
}

static bool check_agpr_base(rocke_ir_builder_t* b, const rocke_attn_bwd_block_gemm_t* g, int base)
{
    int last = base + FM(g) * FN(g) * g->op->c_frag_len - 1;
    if(base < 0 || last > 255)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "pinned AGPRs a[%d:%d] outside a0..a255", base, last);
        return false;
    }
    return true;
}

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
                                       rocke_value_t** out)
{
    if(!live(b))
        return false;
    if(n_steps <= 0)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "an asm MFMA cluster needs at least one K step");
        return false;
    }
    const char* mnem = g->caps.asm_mnemonic;
    if(mnem == NULL)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "no asm MFMA for atom K %d and %s on %s",
                        g->atom_k,
                        g->dtype,
                        g->arch);
        return false;
    }
    int nop = has_tail_nop ? tail_nop : g->caps.asm_tail_nop;
    if(!(0 <= nop && nop <= 15))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "s_nop count %d outside [0, 15]", nop);
        return false;
    }
    if(!(0 <= lead_wait && lead_wait <= 16))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "lead wait states %d outside [0, 16]", lead_wait);
        return false;
    }
    int words = g->atom_k / 8;
    const rocke_type_t* src_ty = rocke_vector_type(b, rocke_i32(), words);
    int fm_n = FM(g), fn_n = FN(g);
    int n = fm_n * fn_n;
    bool zero = acc == NULL;
    int tied = zero ? 0 : n;
    int per_step = fm_n + fn_n;
    std::vector<rocke_value_t*> srcs;
    std::vector<std::string> lines;
    char buf[256];
    if(lead_wait)
    {
        snprintf(buf, sizeof(buf), "s_nop %d", lead_wait - 1);
        lines.push_back(buf);
    }
    for(int ks = 0; ks < n_steps; ++ks)
    {
        if(steps[ks].n_a != fm_n || steps[ks].n_b != fn_n)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "K step operands do not match the block geometry");
            return false;
        }
        for(int fm = 0; fm < fm_n; ++fm)
            srcs.push_back(rocke_b_vec_bitcast(b, steps[ks].a[fm], src_ty));
        for(int fn = 0; fn < fn_n; ++fn)
            srcs.push_back(rocke_b_vec_bitcast(b, steps[ks].b[fn], src_ty));
        int a0 = n + tied + ks * per_step;
        int b0 = a0 + fm_n;
        if(ks && n == 1 && nop)
        {
            snprintf(buf, sizeof(buf), "s_nop %d", nop);
            lines.push_back(buf);
        }
        int i = 0;
        for(int fm = 0; fm < fm_n; ++fm)
            for(int fn = 0; fn < fn_n; ++fn, ++i)
            {
                std::string src_c;
                if(ks)
                    src_c = "$" + std::to_string(i);
                else if(zero)
                    src_c = "0";
                else
                    src_c = "$" + std::to_string(n + i);
                snprintf(buf,
                         sizeof(buf),
                         "%s $%d, $%d, $%d, %s",
                         mnem,
                         i,
                         a0 + fm,
                         b0 + fn,
                         src_c.c_str());
                lines.push_back(buf);
            }
    }
    if(nop)
    {
        snprintf(buf, sizeof(buf), "s_nop %d", nop);
        lines.push_back(buf);
    }
    std::vector<std::string> cons;
    if(!has_agpr_base)
        for(int i = 0; i < n; ++i)
            cons.push_back("=a");
    else
    {
        if(!check_agpr_base(b, g, agpr_base))
            return false;
        int w = g->op->c_frag_len;
        for(int i = 0; i < n; ++i)
        {
            snprintf(
                buf, sizeof(buf), "={a[%d:%d]}", agpr_base + i * w, agpr_base + (i + 1) * w - 1);
            cons.push_back(buf);
        }
    }
    for(int i = 0; i < tied; ++i)
        cons.push_back(std::to_string(i));
    for(size_t i = 0; i < srcs.size(); ++i)
        cons.push_back("v");
    std::string constraints, text;
    for(size_t i = 0; i < cons.size(); ++i)
        constraints += (i ? "," : "") + cons[i];
    for(size_t i = 0; i < lines.size(); ++i)
        text += (i ? "\n\t" : "") + lines[i];
    std::vector<rocke_value_t*> operands;
    std::vector<const rocke_type_t*> result_types;
    if(zero)
    {
        const rocke_type_t* rt = rocke_vector_type(b, rocke_f32(), g->op->c_frag_len);
        for(int i = 0; i < n; ++i)
            result_types.push_back(rt);
    }
    else
        for(int fm = 0; fm < fm_n; ++fm)
            for(int fn = 0; fn < fn_n; ++fn)
            {
                rocke_value_t* x = acc->v[fm * acc->cols + fn];
                operands.push_back(x);
                result_types.push_back(x->type);
            }
    for(rocke_value_t* s : srcs)
        operands.push_back(s);
    if(!live(b))
        return false;
    /* py: b.inline_asm_multi(..., sideeffect=True, convergent=True,
     *     result_name_hint="g13asm" if not zero else "g02asm") */
    rocke_attr_map_t attrs = rocke_i_attrs(b);
    rocke_attr_set_str(b, &attrs, "template", text.c_str());
    rocke_attr_set_str(b, &attrs, "constraints", constraints.c_str());
    rocke_attr_set_bool(b, &attrs, "sideeffect", true);
    rocke_attr_set_bool(b, &attrs, "convergent", true);
    rocke_op_t* op = rocke_b_op(b,
                                ROCKE_OP_TILE_INLINE_ASM,
                                operands.data(),
                                (int)operands.size(),
                                result_types.data(),
                                (int)result_types.size(),
                                &attrs,
                                NULL,
                                0,
                                zero ? "g02asm" : "g13asm",
                                NULL);
    if(op == NULL || !live(b))
        return false;
    for(int i = 0; i < n; ++i)
        out[i] = op->results[i];
    return true;
}

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
                                 rocke_value_t** out)
{
    rocke_attn_bwd_asm_step_t step = {a_ks, n_a, b_ks, n_b};
    return rocke_attn_bwd_mma_agpr_asm_steps(
        b, g, acc, &step, 1, has_tail_nop, tail_nop, has_agpr_base, agpr_base, lead_wait, out);
}

/* ====================================================================== *
 * mma and its variants
 * ====================================================================== */

rocke_attn_bwd_mma_opts_t rocke_attn_bwd_mma_opts_default(void)
{
    rocke_attn_bwd_mma_opts_t o;
    memset(&o, 0, sizeof(o));
    o.route = ROCKE_ATTN_BWD_ROUTE_INTRINSIC;
    o.asm_cluster = ROCKE_ATTN_BWD_CLUSTER_K_STEP;
    return o;
}

static bool acc_shape_ok(rocke_ir_builder_t* b,
                         const rocke_attn_bwd_block_gemm_t* g,
                         const rocke_attn_bwd_frags_t* acc)
{
    if(acc->rows != FM(g) || acc->cols != FN(g))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "accumulator shape does not match the block geometry");
        return false;
    }
    return true;
}

bool rocke_attn_bwd_block_gemm_mma(rocke_ir_builder_t* b,
                                   const rocke_attn_bwd_block_gemm_t* g,
                                   const rocke_attn_bwd_frags_t* acc,
                                   const rocke_attn_bwd_gemm_operand_t* a,
                                   const rocke_attn_bwd_gemm_operand_t* bop,
                                   rocke_value_t* lane,
                                   rocke_value_t* const* origin,
                                   const rocke_attn_bwd_mma_opts_t* opts,
                                   rocke_value_t** out)
{
    if(!live(b))
        return false;
    rocke_attn_bwd_mma_opts_t o = opts ? *opts : rocke_attn_bwd_mma_opts_default();
    if(o.route != ROCKE_ATTN_BWD_ROUTE_INTRINSIC && o.route != ROCKE_ATTN_BWD_ROUTE_AGPR_ASM)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "unknown MMA route %d", (int)o.route);
        return false;
    }
    if(o.asm_cluster != ROCKE_ATTN_BWD_CLUSTER_K_STEP
       && o.asm_cluster != ROCKE_ATTN_BWD_CLUSTER_GEMM)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "unknown asm cluster granularity %d", (int)o.asm_cluster);
        return false;
    }
    bool is_asm = o.route == ROCKE_ATTN_BWD_ROUTE_AGPR_ASM;
    if(!is_asm
       && (o.has_asm_tail_nop || o.asm_cluster != ROCKE_ATTN_BWD_CLUSTER_K_STEP
           || o.has_asm_agpr_base || o.has_asm_lead_wait))
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "asm_tail_nop / asm_cluster / asm_agpr_base / asm_lead_wait need "
                        "route='agpr_asm'");
        return false;
    }
    int lead_wait = o.has_asm_lead_wait ? o.asm_lead_wait : g->caps.asm_lead_wait;
    if(o.has_asm_agpr_base && !check_agpr_base(b, g, o.asm_agpr_base))
        return false;
    int fm_n = FM(g), fn_n = FN(g);
    std::vector<rocke_value_t*> cur((size_t)fm_n * fn_n, (rocke_value_t*)NULL);
    bool have_acc = acc != NULL;
    if(!have_acc)
    {
        if(!is_asm)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "a zero accumulator (acc=None) needs route='agpr_asm'");
            return false;
        }
    }
    else
    {
        if(!acc_shape_ok(b, g, acc))
            return false;
        for(int i = 0; i < fm_n * fn_n; ++i)
            cur[i] = acc->v[i];
    }
    if(!check_operands(b, g, a, bop, lane, origin))
        return false;
    bool pending = o.pre_issue != NULL;
    auto issue_point = [&]() {
        if(pending)
        {
            pending = false;
            o.pre_issue(b, o.pre_issue_ctx);
        }
    };
    bool gemm_cluster = is_asm && o.asm_cluster == ROCKE_ATTN_BWD_CLUSTER_GEMM;
    int ksn = KS(g);
    std::vector<rocke_value_t*> sa((size_t)ksn * fm_n), sb((size_t)ksn * fn_n);
    std::vector<rocke_attn_bwd_asm_step_t> steps;
    for(int ks = 0; ks < ksn; ++ks)
    {
        rocke_value_t** a_ks = sa.data() + (size_t)ks * fm_n;
        rocke_value_t** b_ks = sb.data() + (size_t)ks * fn_n;
        if(!step_operands(b, g, a, bop, lane, origin, ks, a_ks, b_ks))
            return false;
        if(gemm_cluster)
        {
            rocke_attn_bwd_asm_step_t st = {a_ks, fm_n, b_ks, fn_n};
            steps.push_back(st);
            continue;
        }
        issue_point();
        if(is_asm)
        {
            rocke_attn_bwd_frags_t fr = {cur.data(), fm_n, fn_n};
            std::vector<rocke_value_t*> nxt((size_t)fm_n * fn_n);
            if(!rocke_attn_bwd_mma_agpr_asm(b,
                                            g,
                                            have_acc ? &fr : NULL,
                                            a_ks,
                                            fm_n,
                                            b_ks,
                                            fn_n,
                                            o.has_asm_tail_nop,
                                            o.asm_tail_nop,
                                            o.has_asm_agpr_base,
                                            o.asm_agpr_base,
                                            lead_wait,
                                            nxt.data()))
                return false;
            cur = nxt;
            have_acc = true;
            continue;
        }
        for(int fm = 0; fm < fm_n; ++fm)
            for(int fn = 0; fn < fn_n; ++fn)
                cur[fm * fn_n + fn] = rocke_b_mma(
                    b, g->op->op_id, a_ks[fm], b_ks[fn], cur[fm * fn_n + fn], NULL, 0);
    }
    if(!steps.empty())
    {
        issue_point();
        rocke_attn_bwd_frags_t fr = {cur.data(), fm_n, fn_n};
        std::vector<rocke_value_t*> nxt((size_t)fm_n * fn_n);
        if(!rocke_attn_bwd_mma_agpr_asm_steps(b,
                                              g,
                                              have_acc ? &fr : NULL,
                                              steps.data(),
                                              (int)steps.size(),
                                              o.has_asm_tail_nop,
                                              o.asm_tail_nop,
                                              o.has_asm_agpr_base,
                                              o.asm_agpr_base,
                                              lead_wait,
                                              nxt.data()))
            return false;
        cur = nxt;
    }
    if(!live(b))
        return false;
    for(int i = 0; i < fm_n * fn_n; ++i)
        out[i] = cur[i];
    return true;
}

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
                                            rocke_value_t** out)
{
    if(!live(b))
        return false;
    if(buffers != 1 && buffers != 2)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "buffers must be 1 or 2, got %d", buffers);
        return false;
    }
    int per = rocke_attn_bwd_block_gemm_check_k_slice(b, g, k_slice);
    if(per == 0)
        return false;
    int n = g->k / k_slice;
    if(acc == NULL || !acc_shape_ok(b, g, acc))
    {
        if(acc == NULL)
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "accumulator shape does not match the block geometry");
        return false;
    }
    if(!check_operands(b, g, a, bop, lane, origin))
        return false;
    if(first != NULL && first->n != per)
    {
        rocke_i_set_err(
            b, ROCKE_ERR_VALUE, "first slice must hold %d K steps, got %d", per, first->n);
        return false;
    }
    int fm_n = FM(g), fn_n = FN(g);
    std::vector<rocke_value_t*> cur_acc(acc->v, acc->v + (size_t)fm_n * fn_n);
    /* two slice buffers, swapped: cur and nxt */
    std::vector<rocke_value_t*> buf_a[2], buf_b[2];
    for(int i = 0; i < 2; ++i)
    {
        buf_a[i].resize((size_t)per * fm_n);
        buf_b[i].resize((size_t)per * fn_n);
    }
    auto load = [&](int s, int slot, rocke_attn_bwd_kslice_t* ks) {
        ks->a = buf_a[slot].data();
        ks->b = buf_b[slot].data();
        return rocke_attn_bwd_block_gemm_load_k_slice(b, g, a, bop, s, k_slice, lane, origin, ks);
    };
    rocke_attn_bwd_kslice_t cur;
    int cur_slot = 0;
    if(first != NULL)
        cur = *first;
    else if(!load(0, 0, &cur))
        return false;
    for(int s = 0; s < n; ++s)
    {
        rocke_attn_bwd_kslice_t nxt;
        bool have_nxt = false;
        int nxt_slot = 1 - cur_slot;
        if(buffers == 2 && s + 1 < n)
        {
            if(!load(s + 1, nxt_slot, &nxt))
                return false;
            have_nxt = true;
        }
        for(int j = 0; j < cur.n; ++j)
            for(int fm = 0; fm < fm_n; ++fm)
                for(int fn = 0; fn < fn_n; ++fn)
                    cur_acc[fm * fn_n + fn] = rocke_b_mma(b,
                                                          g->op->op_id,
                                                          cur.a[(size_t)j * fm_n + fm],
                                                          cur.b[(size_t)j * fn_n + fn],
                                                          cur_acc[fm * fn_n + fn],
                                                          NULL,
                                                          0);
        if(s + 1 < n)
        {
            if(have_nxt)
                cur = nxt;
            else if(!load(s + 1, nxt_slot, &cur))
                return false;
            cur_slot = nxt_slot;
        }
    }
    if(!live(b))
        return false;
    for(int i = 0; i < fm_n * fn_n; ++i)
        out[i] = cur_acc[i];
    return true;
}

rocke_value_t* rocke_attn_bwd_block_gemm_mma_frag(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_block_gemm_t* g,
                                                  rocke_value_t* acc,
                                                  const rocke_attn_bwd_gemm_operand_t* a,
                                                  const rocke_attn_bwd_gemm_operand_t* bop,
                                                  int fm,
                                                  int fn,
                                                  rocke_value_t* lane,
                                                  rocke_value_t* const* origin)
{
    if(!live(b))
        return NULL;
    if(!(0 <= fm && fm < FM(g) && 0 <= fn && fn < FN(g)))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "fragment index outside the block geometry");
        return NULL;
    }
    const rocke_attn_bwd_gemm_operand_t* ops[2] = {a, bop};
    for(int i = 0; i < 2; ++i)
        if(ops[i]->lds != NULL && (lane == NULL || origin == NULL))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "LDS operands need lane and origin");
            return NULL;
        }
    for(int i = 0; i < 2; ++i)
        if(ops[i]->lds == NULL && (ops[i]->regs.rows <= (i ? fn : fm) || ops[i]->regs.cols < KS(g)))
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "register operand index out of range");
            return NULL;
        }
    for(int ks = 0; ks < KS(g); ++ks)
    {
        rocke_value_t *a_f, *b_f;
        if(a->lds != NULL)
        {
            rocke_attn_bwd_frag_coords_t tmp;
            const rocke_attn_bwd_frag_coords_t* coords = coords_of(b, g, a->lds, 'a', &tmp);
            if(coords == NULL)
                return NULL;
            rocke_value_t* mn0 = off(b, origin[0], a->lds->mn0 + fm * g->op->m);
            a_f = load_one(b, g, a->lds, coords, lane, mn0, ks);
        }
        else
            a_f = a->regs.v[fm * a->regs.cols + ks];
        if(bop->lds != NULL)
        {
            rocke_attn_bwd_frag_coords_t tmp;
            const rocke_attn_bwd_frag_coords_t* coords = coords_of(b, g, bop->lds, 'b', &tmp);
            if(coords == NULL)
                return NULL;
            rocke_value_t* mn0 = off(b, origin[1], bop->lds->mn0 + fn * g->op->n);
            b_f = load_one(b, g, bop->lds, coords, lane, mn0, ks);
        }
        else
            b_f = bop->regs.v[fn * bop->regs.cols + ks];
        acc = rocke_b_mma(b, g->op->op_id, a_f, b_f, acc, NULL, 0);
    }
    return acc;
}

bool rocke_attn_bwd_block_gemm_acc_coords(rocke_ir_builder_t* b,
                                          const rocke_attn_bwd_block_gemm_t* g,
                                          rocke_value_t* lane,
                                          rocke_value_t* const* origin,
                                          int fm,
                                          int fn,
                                          rocke_value_t** out)
{
    if(!live(b))
        return false;
    const rocke_layout_map_t* layout = rocke_mma_op_c_layout(g->op, b);
    if(layout == NULL)
        return false;
    for(int s = 0; s < layout->frag_len; ++s)
    {
        rocke_value_t *r, *c;
        if(!rocke_layout_map_coord(layout, b, lane, s, &r, &c))
            return false;
        rocke_value_t* rr = rocke_b_add(b, origin[0], r);
        out[2 * s] = off(b, rr, fm * g->op->m);
        rocke_value_t* cc = rocke_b_add(b, origin[1], c);
        out[2 * s + 1] = off(b, cc, fn * g->op->n);
    }
    return live(b);
}

/* ====================================================================== *
 * Free functions
 * ====================================================================== */

void rocke_attn_bwd_lane_and_wave(rocke_ir_builder_t* b,
                                  int wave_size,
                                  rocke_value_t* tid,
                                  rocke_value_t** tid_out,
                                  rocke_value_t** lane,
                                  rocke_value_t** wave)
{
    if(tid == NULL)
        tid = rocke_b_thread_id_x(b);
    rocke_value_t* ws = rocke_b_const_i32(b, wave_size);
    *lane = rocke_b_mod(b, tid, ws);
    rocke_value_t* q = rocke_b_div(b, tid, ws);
    *wave = rocke_b_readfirstlane(b, q);
    *tid_out = tid;
}

static bool str_eq(const char* x, const char* y)
{
    if(x == NULL || y == NULL)
        return x == y;
    return strcmp(x, y) == 0;
}

bool rocke_attn_bwd_check_relabel_grids(rocke_ir_builder_t* b,
                                        const rocke_attn_bwd_block_gemm_t* producer,
                                        const rocke_attn_bwd_block_gemm_t* consumer,
                                        rocke_attn_bwd_relabel_plan_t* plan)
{
    if(!live(b))
        return false;
    if(!str_eq(producer->arch, consumer->arch) || !str_eq(producer->dtype, consumer->dtype))
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "producer and consumer differ in arch or dtype");
        return false;
    }
    if(consumer->m != producer->n || consumer->k != producer->m)
    {
        rocke_i_set_err(b,
                        ROCKE_ERR_VALUE,
                        "consumer %dx%dx%d is not the transpose product of producer %dx%d",
                        consumer->m,
                        consumer->n,
                        consumer->k,
                        producer->m,
                        producer->n);
        return false;
    }
    int waves = rocke_attn_bwd_block_gemm_waves(producer);
    if(rocke_attn_bwd_block_gemm_waves(consumer) != waves)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "producer and consumer use different wave counts");
        return false;
    }
    for(int w = 0; w < waves; ++w)
    {
        int pl, ph, cl, ch;
        rocke_attn_bwd_block_gemm_wave_cols(producer, w, &pl, &ph);
        rocke_attn_bwd_block_gemm_wave_rows(consumer, w, &cl, &ch);
        if(pl != cl || ph != ch)
        {
            rocke_i_set_err(
                b, ROCKE_ERR_VALUE, "wave %d: producer columns differ from consumer rows", w);
            return false;
        }
        rocke_attn_bwd_block_gemm_wave_rows(producer, w, &pl, &ph);
        if(pl != 0 || ph != producer->m)
        {
            rocke_i_set_err(b, ROCKE_ERR_VALUE, "wave %d does not own every producer row", w);
            return false;
        }
    }
    rocke_attn_bwd_relabel_plan_t p;
    if(!rocke_attn_bwd_plan_c_to_a(b, producer->op, consumer->op, &p))
        return false;
    if(consumer->atom_k % producer->op->m || p.c_tiles * producer->op->m != consumer->atom_k)
    {
        rocke_i_set_err(b, ROCKE_ERR_VALUE, "consumer atom K does not stack whole producer tiles");
        return false;
    }
    *plan = p;
    return true;
}

bool rocke_attn_bwd_relabel_acc_to_a(rocke_ir_builder_t* b,
                                     const rocke_attn_bwd_block_gemm_t* producer,
                                     const rocke_attn_bwd_block_gemm_t* consumer,
                                     const rocke_attn_bwd_frags_t* acc,
                                     rocke_value_t* lane,
                                     const char* bf16_cast,
                                     rocke_value_t** out,
                                     rocke_attn_bwd_relabel_plan_t* plan_out)
{
    rocke_attn_bwd_relabel_plan_t plan;
    if(!rocke_attn_bwd_check_relabel_grids(b, producer, consumer, &plan))
        return false;
    int t = plan.c_tiles;
    int ksn = KS(consumer);
    std::vector<rocke_value_t*> tiles((size_t)t);
    for(int fm = 0; fm < FM(consumer); ++fm)
        for(int ks = 0; ks < ksn; ++ks)
        {
            for(int i = 0; i < t; ++i)
            {
                int r = ks * t + i;
                if(r >= acc->rows || fm >= acc->cols)
                {
                    rocke_i_set_err(b, ROCKE_ERR_VALUE, "producer accumulator index out of range");
                    return false;
                }
                tiles[i] = acc->v[r * acc->cols + fm];
            }
            out[fm * ksn + ks] = rocke_attn_bwd_emit_c_to_a(
                b, &plan, tiles.data(), t, consumer->dtype, lane, bf16_cast);
            if(!live(b))
                return false;
        }
    if(plan_out != NULL)
        *plan_out = plan;
    return true;
}

bool rocke_attn_bwd_kperm_b_operand_coords(rocke_ir_builder_t* b,
                                           const rocke_attn_bwd_relabel_plan_t* plan,
                                           const rocke_attn_bwd_block_gemm_t* producer,
                                           const rocke_attn_bwd_block_gemm_t* consumer,
                                           rocke_attn_bwd_frag_coords_t* out)
{
    if(!rocke_attn_bwd_relabel_plan_needs_k_perm(plan))
        return rocke_attn_bwd_operand_coords(b, consumer->op, 'b', out);
    return rocke_attn_bwd_kperm_b_coords(b, plan, producer->op, consumer->op, out);
}
