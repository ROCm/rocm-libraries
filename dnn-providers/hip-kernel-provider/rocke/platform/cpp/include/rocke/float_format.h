/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/float_format.h -- Python's repr(float), in C++.
 *
 * The Python engine spells every float it writes with repr(): the IR printer,
 * the IR serializer and the HIP lowerer's literals. The C++ engine must write
 * the same bytes, so all three call this one function. Callers that need a
 * different spelling for some values (HIP spells nan and the infinities as
 * <math.h> macros and adds an "f" suffix to float literals) handle those values
 * themselves and pass the rest here.
 */
#ifndef ROCKE_FLOAT_FORMAT_H
#define ROCKE_FLOAT_FORMAT_H

#include "rocke/strbuf.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Append repr(value) to `out`: the shortest decimal that reads back as the
 * same double, laid out the way CPython does it.
 *
 *   1.0     0.5     -0.0     100.0     1234567890123456.0
 *   0.0001  1e-05   1e+16    5.960464477539063e-08
 *   inf     -inf    nan
 *
 * Fixed-point when the decimal exponent is in [-4, 16), exponential otherwise.
 * A fixed-point integral value keeps a ".0"; a one-digit exponential mantissa
 * has no ".0" ("1e-05"); the exponent has a sign and at least two digits. The
 * output does not depend on the C locale. On allocation failure the strbuf's
 * sticky `oom` flag is set.
 *
 * Mirrors Python: repr(value) for a float. */
void rocke_append_python_float_repr(rocke_strbuf_t* out, double value);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_FLOAT_FORMAT_H */
