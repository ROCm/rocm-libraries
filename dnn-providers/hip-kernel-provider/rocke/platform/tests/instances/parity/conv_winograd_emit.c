/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * tests/parity/conv_winograd_emit.c -- C-side emitter for the Winograd
 * convolution parity harness.
 *
 * PLACEHOLDER — The C++ engine mirror for the Winograd kernel family has not
 * been implemented yet.  Per the byte-identity rule the C++
 * builder must be added in the same PR as the Python builder before this
 * family can be declared DONE.
 *
 * Until then, this file intentionally returns exit code 1 for every config
 * index so that run_diff.py classifies every Winograd config as UNSUPPORTED
 * on the C side.  UNSUPPORTED is a gate-PASSING status (it means "C engine
 * does not know this config yet"), so the byte-identity gate stays GREEN while
 * the C++ work is in flight.
 *
 * When the C++ engine mirror lands:
 *   1. Replace the body of main() below with the real emit logic using
 *      rocke_build_winograd_{data,filter,output}_transform and
 *      rocke_lower_kernel_to_llvm (analogously to conv_implicit_gemm_emit.c).
 *   2. Add the corresponding C API declarations to
 *      platform/cpp/include/rocke/instance_conv_winograd.h.
 *   3. Run check_byte_identity.py and confirm the family transitions from
 *      UNSUPPORTED to MATCH.
 */
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;
    /* Intentional: signal "unknown config" so run_diff.py stops enumeration. */
    fprintf(stderr, "conv_winograd_emit.c: C++ engine mirror not yet implemented\n");
    return 1;
}
