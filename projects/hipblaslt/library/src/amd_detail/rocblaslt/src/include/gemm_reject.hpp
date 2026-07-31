/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// GEMM rejection filter.
//
// When the environment variable HIPBLASLT_REJECT_GEMM_FILE points to a YAML
// file that lists GEMM problems (same flow-style format produced by
// hipblaslt-bench, e.g. "- { function: matmul, M: .., N: .., K: .., a_type: ..,
// ... }"), any GEMM whose size and data types match an entry in that file is
// rejected: the library returns an error status instead of running it.
//
// Only the problem *identity* fields are used for matching (size and data
// type): transA, transB, M, N, K, batch_count, a_type, b_type, c_type, d_type
// and compute_type. All other fields present in the file (leading dimensions,
// strides, alpha/beta, scales, bias, kernel/solution_index, iter, rotating,
// initialization, ...) are ignored.

#pragma once

struct RocblasltContractionProblem;

// Returns true if the given problem matches an entry in the rejection file
// referenced by HIPBLASLT_REJECT_GEMM_FILE. Returns false when the environment
// variable is unset, the file cannot be read, or no entry matches. The file is
// parsed once and cached for the lifetime of the process.
bool rocblaslt_gemm_is_rejected(const RocblasltContractionProblem& prob);
