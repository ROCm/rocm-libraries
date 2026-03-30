// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// CPU reference calculations for the GEMM example.

#pragma once

/// Naive triple-loop CPU reference GEMM: C = A * B
/// A [M x K], B [K x N], C [M x N] — layout given by per-dimension strides.
///
/// Access pattern:
///   A(m,k) = a[m * a_stride_m + k * a_stride_k]
///   B(k,n) = b[k * b_stride_k + n * b_stride_n]
///   C(m,n) = c[m * c_stride_m + n * c_stride_n]
///
/// Example strides:
///   RowMajor A[M×K]: a_stride_m=K, a_stride_k=1
///   ColMajor A[M×K]: a_stride_m=1, a_stride_k=M
void cpu_gemm(const float* a,
              const float* b,
              float* c,
              int M,
              int N,
              int K,
              int a_stride_m,
              int a_stride_k,
              int b_stride_k,
              int b_stride_n,
              int c_stride_m,
              int c_stride_n);

/// Fused epilogue reference: e[i] = c[i] + d0[i]
void cpu_gemm_add(float* e, const float* c, const float* d0, int count);

/// Fused epilogue reference: e[i] = c[i] + d0[i] + d1[i]
void cpu_gemm_add_add(float* e, const float* c, const float* d0, const float* d1, int count);

/// Fused epilogue reference: e[i] = max(0, c[i] + d0[i])
void cpu_gemm_add_relu(float* e, const float* c, const float* d0, int count);
