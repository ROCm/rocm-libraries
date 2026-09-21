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

#include "rocsparse_csrmm.hpp"

#include "rocsparse_control.hpp"

rocsparse_status rocsparse::csrmm_select_default_alg(rocsparse_operation                trans_A,
                                                     bool                               is_batched,
                                                     int32_t                            cu_count,
                                                     const rocsparse::line_nnz_profile& profile,
                                                     rocsparse_csrmm_alg&               alg)
{
    // Only the format-default algorithm is auto-tuned. Any explicit user choice
    // (row_split / nnz_split / merge_path) is honored unchanged.
    if(alg != rocsparse_csrmm_alg_default)
    {
        return rocsparse_status_success;
    }

    // The load-balanced kernels only apply to non-transposed, single-batch
    // multiplies; the profile is only computed in those cases. With nothing
    // cached (e.g. the preprocess/analysis stage was skipped, so compute runs
    // without a profile) keep the historical, capture-safe row-split default.
    if(trans_A != rocsparse_operation_none || is_batched || !profile.known || profile.nnz <= 0
       || cu_count <= 0)
    {
        return rocsparse_status_success;
    }

    // Architecture-portable load-imbalance test. Row-split assigns a fixed slice
    // of threads per line and cannot subdivide a single long line, so its tail
    // latency scales with the longest line (profile.max); a balanced kernel
    // reaches ~ nnz / (resident parallel units). The ratio of the two,
    //
    //     imbalance ~ profile.max / (nnz / P) = profile.max * P / nnz,
    //
    // with P proportional to the device's parallelism, is what decides the
    // winner. Using the compute-unit count for P makes the threshold a single
    // dimensionless constant that self-scales across architectures (a larger GPU
    // tolerates a dominant line less, so it switches sooner) instead of the
    // previous pair of absolute, gfx-specific magic numbers.
    //
    // C was chosen so the test reproduces the gfx1201 FP64 calibration (dense
    // width n=128) measured on a 56-CU Radeon RX 9070: there the crossover sits
    // near a longest-line share of ~5% of all non-zeros, i.e.
    // C ~ profile.max/nnz * cu_count ~ 0.05 * 56 ~ 3. The dense width n cancels
    // to first order because both kernels scale with it. Re-validate C if it
    // ever needs to hold on a very different regime.
    //
    // The test is evaluated in exact 64-bit integer arithmetic (rearranged to
    // avoid a division): profile.max, profile.nnz and cu_count are all bounded
    // well below 2^63 for any real problem, so no floating point is needed.
    static constexpr int64_t s_imbalance_C = 3;

    const int64_t longest_line_work = profile.max * static_cast<int64_t>(cu_count);
    const int64_t balanced_work     = s_imbalance_C * profile.nnz;
    if(longest_line_work >= balanced_work)
    {
        // One line is long enough, relative to the device's parallelism, that
        // the row-split kernel would serialize on it. Switch to the non-zero
        // split kernel, which balances work by non-zeros across wavefronts.
        alg = rocsparse_csrmm_alg_nnz_split;
    }

    return rocsparse_status_success;
}
