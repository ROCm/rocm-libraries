/* **************************************************************************
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#include <rocblas/rocblas.h>

/* Utility macros for explicit template instantiations.

   These macros together provide a system for consisely instantiating a template with all
   possible combinations of template parameters.

   The first argument to every function is a STAMP() macro function that will be used to
   emit the text of the declaration. Every other macro function used in this technique just
   makes function calls to other macros. The stamp is the only component that directly creates
   text. The stamp function is always taken as the first argument because it must be forwarded
   through the entire chain of calls down to APPLY_STAMP(). */

/* The final function in the pipeline is always APPLY_STAMP(), which is the function that
   actually calls/expands the stamp macro. It takes the stamp function as the first argument
   and it just calls the stamp function. */
#define APPLY_STAMP(STAMP, ...) STAMP(__VA_ARGS__)

/* The FOREACH functions are best understood by examining their arguments.
   The first argument is the STAMP function, which is just forwarded to the next function.
   The second argument, F, is the next function in the pipeline.
   All remaining arguments, __VA_ARGS__, are the arguments that should be forwarded to the
   next function.

   Each function just calls the next function in the pipeline, appending whatever arguments it
   wants to add. By calling the next function twice, it can double the number of times the stamp
   is instantiated. By appending different values in each call to the next function, it can
   instantiate the stamp with multiple different values. */
#define FOREACH_BLOCKED_VARIANT(STAMP, F, ...) \
    F(STAMP, ##__VA_ARGS__, false)             \
    F(STAMP, ##__VA_ARGS__, true)
#define FOREACH_INT_TYPE(STAMP, F, ...)  \
    F(STAMP, ##__VA_ARGS__, rocblas_int) \
    F(STAMP, ##__VA_ARGS__, int64_t)
#define FOREACH_REAL_TYPE(STAMP, F, ...) \
    F(STAMP, ##__VA_ARGS__, float)       \
    F(STAMP, ##__VA_ARGS__, double)
#define FOREACH_COMPLEX_TYPE(STAMP, F, ...)        \
    F(STAMP, ##__VA_ARGS__, rocblas_float_complex) \
    F(STAMP, ##__VA_ARGS__, rocblas_double_complex)
#define FOREACH_SCALAR_TYPE(STAMP, F, ...)         \
    F(STAMP, ##__VA_ARGS__, float)                 \
    F(STAMP, ##__VA_ARGS__, double)                \
    F(STAMP, ##__VA_ARGS__, rocblas_float_complex) \
    F(STAMP, ##__VA_ARGS__, rocblas_double_complex)
#define FOREACH_MATRIX_DATA_LAYOUT(STAMP, F, ...) \
    F(STAMP, ##__VA_ARGS__, false, false) // single  \
    F(STAMP, ##__VA_ARGS__, true, true)   // batched \
    F(STAMP, ##__VA_ARGS__, false, true)  // strided_batched

/*  This macro is not strictly necessary. It's does the same thing as any of the FOREACH
    functions, but it doesn't append any values. It exists as the top-level function of the
    macro pipeline only so that all the FOREACH functions can appear as arguments, rather than
    having the first FOREACH appear different from the others. */
#define INSTANTIATE(STAMP, F, ...) F(STAMP, __VA_ARGS__)

/*  To describe what's happening within one of these macro pipelines in in another way, the list
    of arguments starts off as a list of functions. Each function pops the next function from the
    front of the list, appends the values it's adding to the end of the list, and then calls the
    next function with the list as its arguments. This continues with the number of functions at
    the head shrinking and the number of values at the tail growing. That pattern ends with the
    call to APPLY_STAMP(). At that point, there should be no functions remaining and all
    arguments are values for the STAMP. */
