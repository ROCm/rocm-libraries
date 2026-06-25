// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.layouts.c -- C99 port of TransposeLdsReader (and its
 * bound view _BoundTransposeLdsReader) from rocke/helpers/layouts.py.
 *
 * Byte-faithful: the bind()/row() emitters issue the rocke_b_* builder calls in
 * the exact same order and with the exact same operands as the Python:
 *
 *   def bind(self, b, lane):
 *       return _BoundTransposeLdsReader(
 *           reader=self,
 *           lane=lane,
 *           lane_div_16=b.div(lane, b.const_i32(16)),
 *           lane_div_4_mod_4=b.mod(b.div(lane, b.const_i32(4)), b.const_i32(4)),
 *           col=b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4)),
 *       )
 *
 *   def row(self, b, *, k_offset, read=0):
 *       return b.add(
 *           b.const_i32(k_offset + read * 4),
 *           b.add(
 *               b.mul(self.lane_div_16, b.const_i32(self.reader.k_lanes)),
 *               self.lane_div_4_mod_4,
 *           ),
 *       )
 *
 * Note the field-init evaluation order of the dataclass call mirrors Python's
 * left-to-right keyword-argument evaluation: lane_div_16, then lane_div_4_mod_4
 * (whose own b.div is evaluated before its b.mod), then col.
 */

#include <stdarg.h>
#include <stdio.h>

#include "rocke/arena.h"
#include "rocke/error.hpp"
#include "rocke/helper_rocke.helpers.layouts.h"
#include "rocke/ir.h"

/* ----------------------------------------------------------------- helpers */

/* Set the builder's sticky error (first failure wins) and return NULL. Bound
 * only to rocke/ir.h's public struct fields (status + err); mirrors the private
 * rocke_i_set_err. */
/* Raise the failure as a ckc::Error (mirroring the Python `raise`); the public
 * entry boundary catches it and records status + message on the builder, so the
 * C ABI is unchanged. [[noreturn]] keeps the existing `return (T*)rocke_lay_set_err(...)`
 * call sites valid -- the cast/return is simply never reached. */
[[noreturn]] static void*
    rocke_lay_set_err(rocke_ir_builder_t* b, rocke_status_t st, const char* fmt, ...)
{
    (void)b;
    char msg[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    (void)vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    msg[sizeof(msg) - 1] = '\0';
    ckc::raise_status(st, msg);
}

static bool rocke_lay_live(const rocke_ir_builder_t* b)
{
    return b != NULL && b->status == ROCKE_OK;
}

/* ----------------------------------------------------- TransposeLdsReader */

rocke_transpose_lds_reader_t rocke_transpose_lds_reader_make(int K)
{
    rocke_transpose_lds_reader_t r;
    r.K = K;
    r.M = 16; /* dataclass default */
    return r;
}

int rocke_transpose_lds_reader_k_lanes(const rocke_transpose_lds_reader_t* r)
{
    /* @property k_lanes: return self.K // 4
     * (K_L = K / (64 / M) = K / 4 for the M == 16 shape these kernels use). */
    return r->K / 4;
}

int rocke_transpose_lds_reader_reads_per_k_iter(const rocke_transpose_lds_reader_t* r, int k_step)
{
    /* return max(1, k_step // self.K) */
    int q = k_step / r->K;
    return q > 1 ? q : 1;
}

/* -------------------------------------------------- _BoundTransposeLdsReader */

rocke_bound_transpose_lds_reader_t* rocke_transpose_lds_reader_bind(
    rocke_ir_builder_t* b, const rocke_transpose_lds_reader_t* r, rocke_value_t* lane)
{
    rocke_bound_transpose_lds_reader_t* bound;

    if(!rocke_lay_live(b))
    {
        return NULL;
    }
    if(r == NULL || lane == NULL)
    {
        return (rocke_bound_transpose_lds_reader_t*)rocke_lay_set_err(
            b, ROCKE_ERR_VALUE, "TransposeLdsReader.bind: NULL reader/lane");
    }

    bound = (rocke_bound_transpose_lds_reader_t*)rocke_arena_alloc(
        &b->arena, sizeof(rocke_bound_transpose_lds_reader_t));
    if(bound == NULL)
    {
        return (rocke_bound_transpose_lds_reader_t*)rocke_lay_set_err(
            b, ROCKE_ERR_OOM, "TransposeLdsReader.bind: arena OOM");
    }

    bound->reader = *r; /* frozen dataclass captured by value */
    bound->lane = lane;

    /* lane_div_16 = b.div(lane, b.const_i32(16)) */
    bound->lane_div_16 = rocke_b_div(b, lane, rocke_b_const_i32(b, 16));

    /* lane_div_4_mod_4 = b.mod(b.div(lane, b.const_i32(4)), b.const_i32(4))
     * Python evaluates left-to-right: the inner b.div (and its const_i32(4)
     * operand) is created BEFORE the outer b.mod's const_i32(4) operand. C
     * argument evaluation order is unspecified (gcc evaluates right-to-left,
     * which would create the outer const before the inner div and shift every
     * value id below by one). Sequence the inner div via an explicit temporary
     * so the const/div/const/mod creation order matches Python exactly. */
    {
        rocke_value_t* inner_div = rocke_b_div(b, lane, rocke_b_const_i32(b, 4));
        bound->lane_div_4_mod_4 = rocke_b_mod(b, inner_div, rocke_b_const_i32(b, 4));
    }

    /* col = b.mul(b.mod(lane, b.const_i32(4)), b.const_i32(4))
     * Same left-to-right sequencing requirement as lane_div_4_mod_4 above:
     * Python creates the inner b.mod (and its const operand) before the outer
     * b.mul's const operand. Force the order with an explicit temporary. */
    {
        rocke_value_t* inner_mod = rocke_b_mod(b, lane, rocke_b_const_i32(b, 4));
        bound->col = rocke_b_mul(b, inner_mod, rocke_b_const_i32(b, 4));
    }

    if(!rocke_lay_live(b))
    {
        return NULL;
    }
    return bound;
}

rocke_value_t* rocke_bound_transpose_lds_reader_row(rocke_ir_builder_t* b,
                                                    const rocke_bound_transpose_lds_reader_t* self,
                                                    int k_offset,
                                                    int read)
{
    int k_lanes;
    rocke_value_t* inner;

    if(!rocke_lay_live(b))
    {
        return NULL;
    }
    if(self == NULL)
    {
        return (rocke_value_t*)rocke_lay_set_err(
            b, ROCKE_ERR_VALUE, "_BoundTransposeLdsReader.row: NULL self");
    }

    k_lanes = rocke_transpose_lds_reader_k_lanes(&self->reader);

    /* b.add(
     *     b.const_i32(k_offset + read * 4),
     *     b.add(
     *         b.mul(self.lane_div_16, b.const_i32(self.reader.k_lanes)),
     *         self.lane_div_4_mod_4))
     *
     * Python evaluates the outer call's first arg (b.const_i32) before the
     * second (the inner b.add); the inner b.add's first arg (b.mul, whose own
     * b.const_i32 operand) is evaluated before lane_div_4_mod_4 (already an SSA
     * value, no new op). To match the op order, emit const_i32 first, then the
     * b.mul, then the adds. */
    {
        rocke_value_t* outer_const = rocke_b_const_i32(b, (int64_t)(k_offset + read * 4));
        rocke_value_t* mul
            = rocke_b_mul(b, self->lane_div_16, rocke_b_const_i32(b, (int64_t)k_lanes));
        inner = rocke_b_add(b, mul, self->lane_div_4_mod_4);
        return rocke_b_add(b, outer_const, inner);
    }
}
