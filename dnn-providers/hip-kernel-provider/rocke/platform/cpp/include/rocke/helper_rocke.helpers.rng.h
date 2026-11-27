/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.rng.h -- C port of rocke.helpers.rng, the
 * Philox4x32-10 counter-based RNG in IR.
 *
 * The IR's i32 add / mul carry nsw, so a wrapping u32 sum or product is built
 * in i64 from operands that cannot overflow, then truncated (see the Python
 * module). Each helper emits the Python builder calls in the same order, one
 * builder call per statement (C leaves argument evaluation order unspecified
 * and creation order decides the SSA names), so both engines emit the same IR.
 *
 * Symbol map:
 *   PHILOX_M0 / _M1 / _W0 / _W1 / _ROUNDS -> ROCKE_PHILOX_M0 / ... / ROCKE_PHILOX_ROUNDS
 *   _u32_const              -> rocke_b_u32_const
 *   add_u32_wrap            -> rocke_b_add_u32_wrap
 *   mul_lo_u32_const        -> rocke_b_mul_lo_u32_const
 *   _mulhilo32              -> rocke_b_mulhilo_u32 (out-params hi, lo)
 *   mulhilo_u32_wide        -> rocke_b_mulhilo_u32_wide (out-params hi, lo)
 *   philox4x32_10           -> rocke_b_philox4x32_10 (out[4])
 *   philox_counter_add      -> rocke_b_philox_counter_add (out-params lo, hi)
 *   PhiloxCall              -> rocke_philox_call_t + _init / _done / _step / _finish
 *
 * An errored builder makes every rocke_b_* call a NULL no-op, so these helpers
 * need no error handling of their own.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_RNG_H
#define ROCKE_HELPER_ROCKE_HELPERS_RNG_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Philox4x32-10 multipliers and key increments (Random123). */
#define ROCKE_PHILOX_M0 0xD2511F53u
#define ROCKE_PHILOX_M1 0xCD9E8D57u
#define ROCKE_PHILOX_W0 0x9E3779B9u
#define ROCKE_PHILOX_W1 0xBB67AE85u
#define ROCKE_PHILOX_ROUNDS 10

/* i32 constant holding the bit pattern of the u32 `value`. */
rocke_value_t* rocke_b_u32_const(rocke_ir_builder_t* b, uint32_t value);

/* (a + c) mod 2**32 of two i32 values read as u32, free of nsw poison. */
rocke_value_t* rocke_b_add_u32_wrap(rocke_ir_builder_t* b, rocke_value_t* a, rocke_value_t* c);

/* Low 32 bits of a * c for i32 `a` read as u32 and the u32 constant `c`. */
rocke_value_t* rocke_b_mul_lo_u32_const(rocke_ir_builder_t* b, rocke_value_t* a, uint32_t c);

/* (hi, lo) = a * c: v_mul_hi_u32 + the split low product. */
void rocke_b_mulhilo_u32(
    rocke_ir_builder_t* b, rocke_value_t* a, uint32_t c, rocke_value_t** hi, rocke_value_t** lo);

/* (hi, lo) of a * c from one u64 product (one v_mad_u64_u32). */
void rocke_b_mulhilo_u32_wide(
    rocke_ir_builder_t* b, rocke_value_t* a, uint32_t c, rocke_value_t** hi, rocke_value_t** lo);

/* Philox4x32-10 of counter words c0..c3 and key words k0, k1 (i32 as u32). */
void rocke_b_philox4x32_10(rocke_ir_builder_t* b,
                           rocke_value_t* c0,
                           rocke_value_t* c1,
                           rocke_value_t* c2,
                           rocke_value_t* c3,
                           rocke_value_t* k0,
                           rocke_value_t* k1,
                           rocke_value_t* out[4]);

/* Words (lo, hi) of the u64 counter base + k (k an i32 read as u32), mod 2**64. */
void rocke_b_philox_counter_add(rocke_ir_builder_t* b,
                                rocke_value_t* base_lo,
                                rocke_value_t* base_hi,
                                rocke_value_t* k,
                                rocke_value_t** lo,
                                rocke_value_t** hi);

/* PhiloxCall: one Philox4x32-10 call of the counter (ctr_lo, ctr_hi, 0, 0),
 * emitted a round at a time. A NULL word in c[] is a known zero (Python None):
 * no multiply or xor is emitted for it. */
typedef struct rocke_philox_call
{
    rocke_ir_builder_t* b;
    bool xor3;
    bool mad64;
    rocke_value_t* c[4];
    rocke_value_t* k[2];
    int rounds;
} rocke_philox_call_t;

void rocke_philox_call_init(rocke_philox_call_t* call,
                            rocke_ir_builder_t* b,
                            rocke_value_t* seed_lo,
                            rocke_value_t* seed_hi,
                            rocke_value_t* ctr_lo,
                            rocke_value_t* ctr_hi,
                            bool xor3,
                            bool mad64);
bool rocke_philox_call_done(const rocke_philox_call_t* call);
/* Emit the next round (no-op once all rounds are emitted). */
void rocke_philox_call_step(rocke_philox_call_t* call);
/* Emit the remaining rounds; the four output words. */
void rocke_philox_call_finish(rocke_philox_call_t* call, rocke_value_t* out[4]);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_RNG_H */
