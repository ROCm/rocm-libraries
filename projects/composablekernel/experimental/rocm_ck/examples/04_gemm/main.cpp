// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the kpack GEMM example. Loads GEMM kernel variants
// from a kpack archive, runs each on the detected GPU, and verifies against
// CPU reference.
//
// This example demonstrates:
//   - Operator-centric Signature schema with GemmOp, AddOp, ReluOp
//   - Multi-variant iteration with per-variant grid launch parameters
//   - Epilogue composition via operator chaining (e.g., GemmOp + AddOp + ReluOp)
//   - Per-tensor layout support (Row, Col) via explicit Tensor entries
//   - Split-K scheduling (k_batch > 1) with atomic accumulation
//   - Per-type tolerance for correctness verification

#include "cpu_ref.hpp"
#include "gemm_variants.hpp"

#include <rocm_ck/args.hpp>
#include <rocm_ck/datatype_convert.hpp>
#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/hip_check.hpp>
#include <rocm_ck/kpack_module.hpp>
#include <rocm_ck/typed_buffer.hpp>

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

// ============================================================================
// Layout-aware stride helpers
// ============================================================================

/// Returns {row_stride, col_stride} for a matrix of size rows × cols.
///   RowMajor: row_stride = cols, col_stride = 1
///   ColMajor: row_stride = 1,    col_stride = rows
std::pair<int, int> layout_strides(rocm_ck::Layout ly, int rows, int cols)
{
    if(ly == rocm_ck::Layout::Row)
        return {cols, 1};
    else
        return {1, rows};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-gemm.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive ---
    rocm_ck::KpackArchive archive;
    if(!archive.open(argv[1]))
        return 1;

    // --- Problem dimensions (all divisible by tile dims for no-pad kernel) ---
    constexpr int M = 512;
    constexpr int N = 512;
    constexpr int K = 256;

    // --- Input data (fp32, initialized once) ---
    // Values kept small (0-7) so all types are exact. Worst case accumulation:
    // 256 × (7 × 7) = 12544, within fp16 exact range.
    // D0 values are small (0-3) — worst case fused result: 12544 + 3 = 12547.
    std::vector<float> ref_a(M * K);
    std::vector<float> ref_b(K * N);
    std::vector<float> ref_d0(M * N);
    std::vector<float> ref_d1(M * N);

    for(int i = 0; i < M * K; ++i)
        ref_a[i] = static_cast<float>(i % 8);
    for(int i = 0; i < K * N; ++i)
        ref_b[i] = static_cast<float>(i % 8);
    for(int i = 0; i < M * N; ++i)
        ref_d0[i] = static_cast<float>(i % 4);
    for(int i = 0; i < M * N; ++i)
        ref_d1[i] = static_cast<float>(i % 3);

    // --- Run each variant ---
    bool all_passed  = true;
    int variants_run = 0;

    // Per-variant CPU reference buffers (reused across iterations)
    std::vector<float> ref_c(M * N, 0.0f);

    // Detect GPU architecture for arch-adaptive variant filtering
    hipDeviceProp_t dev_props;
    HIP_CHECK(hipGetDeviceProperties(&dev_props, 0));
    const char* gpu_arch = dev_props.gcnArchName;

    // Batch count for the batched variant
    constexpr int BatchCount = 4;

    for(const rocm_ck::GemmVariant& variant : rocm_ck::gemm_variants)
    {
        const rocm_ck::GemmSpec& spec = variant.spec;

        // Skip arch-specific variants that don't match the current GPU
        if(std::strncmp(variant.name, "gemm_fp16_gfx90a", 16) == 0 &&
           std::strstr(gpu_arch, "gfx90a") == nullptr)
            continue;
        if(std::strncmp(variant.name, "gemm_fp16_gfx942", 16) == 0 &&
           std::strstr(gpu_arch, "gfx942") == nullptr)
            continue;

        // Skip preshuffle — requires host-side B matrix rearrangement (not yet wired)
        if(spec.pipeline == rocm_ck::Pipeline::Preshuffle)
        {
            std::printf("%s: SKIPPED (requires host-side preshuffle)\n", variant.name);
            continue;
        }

        // Batched variant uses batch dimension
        bool is_batched = (std::strcmp(variant.name, "gemm_fp16_batched") == 0);
        int batch_count = is_batched ? BatchCount : 0;

        // Per-variant grid dimensions from tile geometry
        int grid_m    = (M + spec.block_tile.m - 1) / spec.block_tile.m;
        int grid_n    = (N + spec.block_tile.n - 1) / spec.block_tile.n;
        int grid_size = grid_m * grid_n;
        int grid_y    = is_batched ? batch_count : 1; // batch dimension
        int grid_z    = spec.k_batch; // Split-K: k_batch partitions along blockIdx.z

        // Load kernel
        rocm_ck::KpackKernel kernel;
        if(!kernel.load(archive, variant.name))
            continue;

        // --- Layout-aware strides from physical tensor table ---
        auto [a_stride_m, a_stride_k] = layout_strides(spec.lhs().layout, M, K);
        auto [b_stride_k, b_stride_n] = layout_strides(spec.rhs().layout, K, N);
        auto [c_stride_m, c_stride_n] = layout_strides(spec.output().layout, M, N);

        // --- Per-variant CPU reference (layout-dependent strides) ---
        cpu_gemm(ref_a.data(),
                 ref_b.data(),
                 ref_c.data(),
                 M,
                 N,
                 K,
                 a_stride_m,
                 a_stride_k,
                 b_stride_k,
                 b_stride_n,
                 c_stride_m,
                 c_stride_n);

        // Fused epilogue references (element-wise on flat buffer — correct for RowMajor output)
        const float* ref = ref_c.data();
        std::vector<float> ref_fused;
        if(spec.has_epilogue_op(rocm_ck::EpilogueOp::Add) &&
           spec.has_epilogue_op(rocm_ck::EpilogueOp::Relu))
        {
            ref_fused.resize(M * N);
            cpu_gemm_add_relu(ref_fused.data(), ref_c.data(), ref_d0.data(), M * N);
            ref = ref_fused.data();
        }
        else if(spec.has_epilogue_op(rocm_ck::EpilogueOp::Add) && spec.num_physical_tensors > 4)
        {
            // Two D tensors: Add+Add (result = gemm + D0 + D1)
            ref_fused.resize(M * N);
            cpu_gemm_add_add(ref_fused.data(), ref_c.data(), ref_d0.data(), ref_d1.data(), M * N);
            ref = ref_fused.data();
        }
        else if(spec.has_epilogue_op(rocm_ck::EpilogueOp::Add))
        {
            ref_fused.resize(M * N);
            cpu_gemm_add(ref_fused.data(), ref_c.data(), ref_d0.data(), M * N);
            ref = ref_fused.data();
        }

        // Allocate typed device buffers from physical tensor table
        // Batched variants replicate input data across batch elements
        int buf_batch = is_batched ? batch_count : 1;

        std::vector<float> src_a(M * K * buf_batch);
        std::vector<float> src_b(K * N * buf_batch);
        for(int b = 0; b < buf_batch; ++b)
        {
            std::copy(ref_a.begin(), ref_a.end(), src_a.begin() + b * M * K);
            std::copy(ref_b.begin(), ref_b.end(), src_b.begin() + b * K * N);
        }

        rocm_ck::TypedBuffer buf_a(spec.lhs().dtype, M * K * buf_batch);
        rocm_ck::TypedBuffer buf_b(spec.rhs().dtype, K * N * buf_batch);
        rocm_ck::TypedBuffer buf_c(spec.output().dtype, M * N * buf_batch);

        buf_a.upload(src_a.data());
        buf_b.upload(src_b.data());
        buf_c.zero();

        // D tensors are present when num_physical_tensors > 3
        bool has_d0 = spec.num_physical_tensors > 3;
        bool has_d1 = spec.num_physical_tensors > 4;

        std::unique_ptr<rocm_ck::TypedBuffer> buf_d0;
        if(has_d0)
        {
            buf_d0 = std::make_unique<rocm_ck::TypedBuffer>(spec.d0().dtype, M * N);
            buf_d0->upload(ref_d0.data());
        }
        std::unique_ptr<rocm_ck::TypedBuffer> buf_d1;
        if(has_d1)
        {
            buf_d1 = std::make_unique<rocm_ck::TypedBuffer>(spec.d1().dtype, M * N);
            buf_d1->upload(ref_d1.data());
        }

        std::printf("%s: M=%d, N=%d, K=%d, grid=%dx%dx%d, block=%d%s\n",
                    variant.name,
                    M,
                    N,
                    K,
                    grid_size,
                    grid_y,
                    grid_z,
                    spec.workgroup_size,
                    is_batched ? " (batched)" : "");

        // Build generic Args — layout-aware strides from physical tensor table
        rocm_ck::Args kernel_args{};
        kernel_args.tensors[spec.lhs().args_slot] = {
            buf_a.ptr(), rocm_ck::make_shape(M, K), rocm_ck::make_strides(a_stride_m, a_stride_k)};
        kernel_args.tensors[spec.rhs().args_slot] = {
            buf_b.ptr(), rocm_ck::make_shape(K, N), rocm_ck::make_strides(b_stride_k, b_stride_n)};
        kernel_args.tensors[spec.output().args_slot] = {
            buf_c.ptr(), rocm_ck::make_shape(M, N), rocm_ck::make_strides(c_stride_m, c_stride_n)};

        // Batch parameters
        if(is_batched)
        {
            kernel_args.batch_count                            = batch_count;
            kernel_args.batch_strides[spec.lhs().args_slot]    = M * K;
            kernel_args.batch_strides[spec.rhs().args_slot]    = K * N;
            kernel_args.batch_strides[spec.output().args_slot] = M * N;
        }
        if(has_d0)
        {
            auto [d0_stride_m, d0_stride_n]          = layout_strides(spec.d0().layout, M, N);
            kernel_args.tensors[spec.d0().args_slot] = {
                buf_d0->ptr(),
                rocm_ck::make_shape(M, N),
                rocm_ck::make_strides(d0_stride_m, d0_stride_n)};
        }
        if(has_d1)
        {
            auto [d1_stride_m, d1_stride_n]          = layout_strides(spec.d1().layout, M, N);
            kernel_args.tensors[spec.d1().args_slot] = {
                buf_d1->ptr(),
                rocm_ck::make_shape(M, N),
                rocm_ck::make_strides(d1_stride_m, d1_stride_n)};
        }

        void* args_ptr          = &kernel_args;
        size_t kernel_args_size = sizeof(kernel_args);
        void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                                   args_ptr,
                                   HIP_LAUNCH_PARAM_BUFFER_SIZE,
                                   &kernel_args_size,
                                   HIP_LAUNCH_PARAM_END};

        HIP_CHECK(hipModuleLaunchKernel(kernel.function(),
                                        grid_size,
                                        grid_y,
                                        grid_z,
                                        spec.workgroup_size,
                                        1,
                                        1,
                                        0,
                                        nullptr,
                                        nullptr,
                                        launch_config));
        HIP_CHECK(hipDeviceSynchronize());

        // Download and verify (for batched, download all and verify batch 0)
        std::vector<float> result(M * N * buf_batch);
        buf_c.download(result.data());

        float tolerance = rocm_ck::tolerance_for(spec.output().dtype);
        bool passed     = true;
        for(int i = 0; i < M * N; ++i)
        {
            if(std::fabs(result[i] - ref[i]) > tolerance * std::fabs(ref[i]) + tolerance)
            {
                int row = i / N;
                int col = i % N;
                std::fprintf(stderr,
                             "  MISMATCH at (%d,%d): got %f, expected %f (diff=%e)\n",
                             row,
                             col,
                             result[i],
                             ref[i],
                             result[i] - ref[i]);
                passed = false;
                break;
            }
        }

        std::printf("%s: %s\n", variant.name, passed ? "PASSED" : "FAILED");
        if(!passed)
            all_passed = false;
        ++variants_run;
    }

    if(variants_run == 0)
    {
        std::fprintf(stderr, "No variants ran successfully\n");
        return 1;
    }

    return all_passed ? 0 : 1;
}
