/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "rocsparse-types.h"

namespace rocsparse
{
    // Objective structural summary of a sparse matrix's "line" lengths, where a
    // line is the unit of the major storage direction: a row for CSR (offsets is
    // the row pointer) or a column for CSC (offsets is the column pointer). It is
    // a pure property of the sparsity pattern - no algorithm policy is baked in -
    // so any routine operating on the matrix can reuse it (e.g. SpMM and SpMV
    // both care about the row/column length distribution). It is populated once
    // by compute_line_nnz_profile (a device reduction + device->host copy) and
    // cached on the matrix descriptor, then read by cheap, pure per-routine
    // algorithm selectors.
    //
    // Only the maximum line length is tracked today (all the current SpMM gate
    // needs). The struct is the extension point for further statistics (e.g. a
    // minimum and a sum-of-squares for the standard deviation, useful to a future
    // SpMV/SpGEAM heuristic): add a field here plus its reduction in
    // compute_line_nnz_profile, with no change to callers.
    struct line_nnz_profile
    {
        bool    known{}; // whether the fields below have been computed
        int64_t nnz{}; // total non-zeros spanned by the reduction
        int64_t max{}; // longest line length
    };

    // Inputs the default-algorithm SpMM auto-selection needs, bundled into one
    // object so they can be threaded through the csrmm/cscmm analysis and compute
    // entry points without spelling them out in every signature. Passing a null
    // pointer (or a null \ref profile) opts out of auto-selection, leaving the
    // explicit or historical default algorithm in place. Future selection inputs
    // can be added here without touching those signatures.
    struct spmm_default_alg_info
    {
        line_nnz_profile* profile{}; // structural line-length profile (filled at analysis)
        bool              is_batched{}; // whether the multiply is batched
    };

    // Computes \p profile from an \p offsets array of length (nlines + 1) using a
    // two-pass, atomic-free reduction followed by one device->host copy. The copy
    // is a synchronizing operation that is illegal while the stream is captured
    // into a HIP graph, so it is meant for a non-capturing stage (in SpMM, the
    // preprocess/analysis stage). As a safety net it is also a no-op when the
    // stream is currently capturing - the profile is then left unknown and the
    // caller's selector falls back to its capture-safe default. It is likewise a
    // no-op when profile.known is already set, so the reduction runs at most once
    // per descriptor. \p offsets_indextype selects the offsets element type
    // (i32 or i64).
    rocsparse_status compute_line_nnz_profile(rocsparse_handle    handle,
                                              rocsparse_indextype offsets_indextype,
                                              int64_t             nlines,
                                              int64_t             nnz,
                                              const void*         offsets,
                                              line_nnz_profile&   profile);
}
