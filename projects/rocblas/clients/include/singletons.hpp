/* ************************************************************************
 * Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <memory.h>
#include <string>

// global for device memory padding see d_vector.hpp

extern size_t g_DVEC_PAD;
void          d_vector_set_pad_length(size_t pad);

// Reporting for the device-memory guards in d_vector.hpp.
//
// These exist so that d_vector.hpp, and the geometry and layout of d_vector<T>, need no
// reference to GOOGLE_TEST. Its members are templates, so they carry the same mangled names
// however the translation unit was compiled; a body or a member whose presence varies with
// the macro would give one symbol two definitions, and rocblas-gemm-tune links objects built
// both ways. Only the definitions in singletons.cpp, compiled once per binary, may depend on
// the macro. (Other client headers still violate this; see AIROCBLAS-1390.)
//
// Where a binary compiles singletons.cpp *with* Google Test -- rocblas-test, rocblas-bench
// and rocblas-gemm-tune, which all link rocblas_clients_common -- a failure is recorded as a
// non-fatal test failure, and also printed when no test is running. Where it is compiled
// *without* Google Test -- the samples, which build this file themselves -- a failure is
// printed and the process exits non-zero.
//
// So a failed free is not fatal to bench or the tuner. On develop that was not a decision:
// teardown used CHECK_HIP_ERROR, which has a different definition either side of the macro,
// so which one those binaries ran was whichever the linker happened to keep. This makes it
// a deliberate choice instead.
void d_vector_report_failure(const std::string& message);

// For a guard region that failed its comparison: names how many bytes differ, the first
// differing offset, and the bytes expected and found there.
void d_vector_report_guard_corruption(const unsigned char* host,
                                      const unsigned char* reference,
                                      size_t               guard_bytes,
                                      const char*          tag);
