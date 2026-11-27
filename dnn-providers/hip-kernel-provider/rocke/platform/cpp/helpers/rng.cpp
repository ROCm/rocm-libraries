// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * helper_rocke.helpers.rng -- C port of rocke.helpers.rng (Philox4x32-10 in
 * IR). Every builder call is its own statement, in the Python evaluation
 * order (arguments left to right), so the SSA names match.
 */

#include "rocke/helper_rocke.helpers.rng.h"

#include <stddef.h>

rocke_value_t* rocke_b_u32_const(rocke_ir_builder_t* b, uint32_t value)
{
    int64_t masked = (int64_t)value;
    if(masked >= 0x80000000LL)
        masked -= 0x100000000LL;
    return rocke_b_const_i32(b, masked);
}

rocke_value_t* rocke_b_add_u32_wrap(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c)
{
    rocke_value_t* a64 = rocke_b_zext(b, a, rocke_i64());
    rocke_value_t* c64 = rocke_b_zext(b, c, rocke_i64());
    rocke_value_t* s = rocke_b_add(b, a64, c64);
    return rocke_b_trunc(b, s, rocke_i32());
}

rocke_value_t* rocke_b_mul_lo_u32_const(rocke_ir_builder_t* b, rocke_value_t* a, uint32_t c)
{
    rocke_value_t* a64 = rocke_b_zext(b, a, rocke_i64());
    rocke_value_t* c_lo = rocke_b_const_i64(b, (int64_t)(c & 0xFFFFu));
    rocke_value_t* p_lo = rocke_b_mul(b, a64, c_lo);
    rocke_value_t* c_hi = rocke_b_const_i64(b, (int64_t)(c >> 16));
    rocke_value_t* p_hi = rocke_b_mul(b, a64, c_hi);
    rocke_value_t* mask16 = rocke_b_const_i64(b, 0xFFFF);
    rocke_value_t* p_hi_low = rocke_b_land(b, p_hi, mask16);
    rocke_value_t* sh16 = rocke_b_const_i64(b, 16);
    rocke_value_t* p_hi16 = rocke_b_shl(b, p_hi_low, sh16);
    rocke_value_t* sum = rocke_b_add(b, p_lo, p_hi16);
    return rocke_b_trunc(b, sum, rocke_i32());
}

void rocke_b_mulhilo_u32(
    rocke_ir_builder_t* b, rocke_value_t* a, uint32_t c, rocke_value_t** hi, rocke_value_t** lo)
{
    rocke_value_t* cc = rocke_b_u32_const(b, c);
    *hi = rocke_b_umul_hi_i32(b, a, cc);
    *lo = rocke_b_mul_lo_u32_const(b, a, c);
}

void rocke_b_mulhilo_u32_wide(
    rocke_ir_builder_t* b, rocke_value_t* a, uint32_t c, rocke_value_t** hi, rocke_value_t** lo)
{
    rocke_value_t* a64 = rocke_b_zext(b, a, rocke_i64());
    rocke_value_t* c_half = rocke_b_const_i64(b, (int64_t)(c >> 1));
    rocke_value_t* half = rocke_b_mul(b, a64, c_half);
    rocke_value_t* one = rocke_b_const_i64(b, 1);
    rocke_value_t* prod = rocke_b_shl(b, half, one);
    if(c & 1u)
        prod = rocke_b_add_nuw(b, prod, a64);
    rocke_value_t* sh32 = rocke_b_const_i64(b, 32);
    rocke_value_t* prod_hi = rocke_b_lshr(b, prod, sh32);
    *hi = rocke_b_trunc(b, prod_hi, rocke_i32());
    *lo = rocke_b_trunc(b, prod, rocke_i32());
}

void rocke_b_philox4x32_10(rocke_ir_builder_t* b,
                           rocke_value_t* c0,
                           rocke_value_t* c1,
                           rocke_value_t* c2,
                           rocke_value_t* c3,
                           rocke_value_t* k0,
                           rocke_value_t* k1,
                           rocke_value_t* out[4])
{
    for(int round = 0; round < ROCKE_PHILOX_ROUNDS; ++round)
    {
        rocke_value_t *hi0, *lo0, *hi1, *lo1;
        rocke_b_mulhilo_u32(b, c0, ROCKE_PHILOX_M0, &hi0, &lo0);
        rocke_b_mulhilo_u32(b, c2, ROCKE_PHILOX_M1, &hi1, &lo1);
        rocke_value_t* x0 = rocke_b_xor(b, hi1, c1);
        rocke_value_t* n0 = rocke_b_xor(b, x0, k0);
        rocke_value_t* x2 = rocke_b_xor(b, hi0, c3);
        rocke_value_t* n2 = rocke_b_xor(b, x2, k1);
        c0 = n0;
        c1 = lo1;
        c2 = n2;
        c3 = lo0;
        rocke_value_t* w0 = rocke_b_u32_const(b, ROCKE_PHILOX_W0);
        k0 = rocke_b_add_u32_wrap(b, k0, w0);
        rocke_value_t* w1 = rocke_b_u32_const(b, ROCKE_PHILOX_W1);
        k1 = rocke_b_add_u32_wrap(b, k1, w1);
    }
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
    out[3] = c3;
}

void rocke_b_philox_counter_add(rocke_ir_builder_t* b,
                                rocke_value_t* base_lo,
                                rocke_value_t* base_hi,
                                rocke_value_t* k,
                                rocke_value_t** lo,
                                rocke_value_t** hi)
{
    rocke_value_t* l = rocke_b_add_u32_wrap(b, base_lo, k);
    rocke_value_t* wrapped = rocke_b_cmp_ult(b, l, k);
    rocke_value_t* carry = rocke_b_zext(b, wrapped, rocke_i32());
    *lo = l;
    *hi = rocke_b_add_u32_wrap(b, base_hi, carry);
}

/* ------------------------------------------------------------ PhiloxCall */

void rocke_philox_call_init(rocke_philox_call_t* call,
                            rocke_ir_builder_t* b,
                            rocke_value_t* seed_lo,
                            rocke_value_t* seed_hi,
                            rocke_value_t* ctr_lo,
                            rocke_value_t* ctr_hi,
                            bool xor3,
                            bool mad64)
{
    call->b = b;
    call->xor3 = xor3;
    call->mad64 = mad64;
    call->c[0] = ctr_lo;
    call->c[1] = ctr_hi;
    call->c[2] = NULL; /* known zero */
    call->c[3] = NULL;
    call->k[0] = seed_lo;
    call->k[1] = seed_hi;
    call->rounds = 0;
}

bool rocke_philox_call_done(const rocke_philox_call_t* call)
{
    return call->rounds == ROCKE_PHILOX_ROUNDS;
}

/* PhiloxCall._mulhilo: a known-zero word gives (None, None). */
static void philox_call_mulhilo(
    rocke_philox_call_t* call, rocke_value_t* a, uint32_t m, rocke_value_t** hi, rocke_value_t** lo)
{
    if(a == NULL)
    {
        *hi = NULL;
        *lo = NULL;
        return;
    }
    if(!call->mad64)
        rocke_b_mulhilo_u32(call->b, a, m, hi, lo);
    else
        rocke_b_mulhilo_u32_wide(call->b, a, m, hi, lo);
}

/* PhiloxCall._xor: xor of the non-zero words, one v_bitop3_b32 for three. */
static rocke_value_t*
    philox_call_xor(rocke_philox_call_t* call, rocke_value_t* x, rocke_value_t* y, rocke_value_t* z)
{
    rocke_value_t* vals[3];
    int n = 0;
    if(x != NULL)
        vals[n++] = x;
    if(y != NULL)
        vals[n++] = y;
    if(z != NULL)
        vals[n++] = z;
    if(n == 3 && call->xor3)
        return rocke_b_bitop3_b32(call->b, vals[0], vals[1], vals[2], 0x96);
    rocke_value_t* out = vals[0];
    for(int i = 1; i < n; ++i)
        out = rocke_b_xor(call->b, out, vals[i]);
    return out;
}

void rocke_philox_call_step(rocke_philox_call_t* call)
{
    if(rocke_philox_call_done(call))
        return;
    rocke_ir_builder_t* b = call->b;
    rocke_value_t* c0 = call->c[0];
    rocke_value_t* c1 = call->c[1];
    rocke_value_t* c2 = call->c[2];
    rocke_value_t* c3 = call->c[3];
    rocke_value_t* k0 = call->k[0];
    rocke_value_t* k1 = call->k[1];
    rocke_value_t *hi0, *lo0, *hi1, *lo1;
    philox_call_mulhilo(call, c0, ROCKE_PHILOX_M0, &hi0, &lo0);
    philox_call_mulhilo(call, c2, ROCKE_PHILOX_M1, &hi1, &lo1);
    rocke_value_t* n0 = philox_call_xor(call, hi1, c1, k0);
    rocke_value_t* n2 = philox_call_xor(call, hi0, c3, k1);
    call->c[0] = n0;
    call->c[1] = lo1;
    call->c[2] = n2;
    call->c[3] = lo0;
    rocke_value_t* w0 = rocke_b_u32_const(b, ROCKE_PHILOX_W0);
    rocke_value_t* nk0 = rocke_b_add_u32_wrap(b, k0, w0);
    rocke_value_t* w1 = rocke_b_u32_const(b, ROCKE_PHILOX_W1);
    rocke_value_t* nk1 = rocke_b_add_u32_wrap(b, k1, w1);
    call->k[0] = nk0;
    call->k[1] = nk1;
    call->rounds += 1;
}

void rocke_philox_call_finish(rocke_philox_call_t* call, rocke_value_t* out[4])
{
    while(!rocke_philox_call_done(call))
        rocke_philox_call_step(call);
    rocke_value_t* zero = rocke_b_const_i32(call->b, 0);
    for(int i = 0; i < 4; ++i)
        out[i] = call->c[i] == NULL ? zero : call->c[i];
}
