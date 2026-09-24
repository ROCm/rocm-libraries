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

// Reports a problem found while managing the device-memory guard regions in d_vector.hpp.
// Records a Google Test failure where there is a running test, and prints otherwise.
//
// Deliberately out of line, and declared without reference to GOOGLE_TEST. d_vector<T>'s
// members are templates, so they have the same mangled names however the translation unit
// was compiled, and a binary that links objects built both with and without GOOGLE_TEST
// keeps only one definition of each. Reporting through this function instead of a Google
// Test macro is what lets d_vector.hpp compile to the same definition either way; only the
// definition here, compiled once per binary, is allowed to care about the macro.
void d_vector_report_failure(const std::string& message);
