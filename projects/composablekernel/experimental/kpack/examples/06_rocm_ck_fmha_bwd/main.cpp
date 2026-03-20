// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Host-side loader for the FMHA BWD OGradDotO kpack example. Loads kernel
// variants from a kpack archive and verifies each one against a CPU reference.
//
// Computes: D[b][h][s] = sum_v(dO[b][h][s][v] * O[b][h][s][v]) * p_undrop

#include "rocm_fmha_bwd_registry.hpp"

#include <rocm_ck/datatype_utils.hpp>
#include <rocm_ck/gpu_arch.hpp>
#include <rocm_ck/hip_check.hpp>

#include <hip/hip_runtime.h>

#include <rocm_kpack/kpack.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using rocm_ck::ALL_FMHA_BWD_VARIANTS;
using rocm_ck::ALL_FMHA_BWD_VARIANTS_COUNT;

// ---------------------------------------------------------------------------
// CPU reference
// ---------------------------------------------------------------------------

/// Compute D = rowSum(dO * O) * p_undrop on the host.
/// Layout: [batch, nhead, seqlen_q, hdim_v] row-major.
static void cpuOGradDotO(const std::vector<float>& O,
                         const std::vector<float>& dO,
                         std::vector<float>& D,
                         int batch,
                         int nhead,
                         int seqlen_q,
                         int hdim_v,
                         float p_undrop)
{
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            for(int s = 0; s < seqlen_q; ++s)
            {
                float acc = 0.0f;
                for(int v = 0; v < hdim_v; ++v)
                {
                    int idx = ((b * nhead + h) * seqlen_q + s) * hdim_v + v;
                    acc += dO[idx] * O[idx];
                }
                int d_idx = (b * nhead + h) * seqlen_q + s;
                D[d_idx]  = acc * p_undrop;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Batch-mode variant runner
// ---------------------------------------------------------------------------

static bool runBatchVariant(const rocm_ck::FmhaBwdVariantDescriptor& variant,
                            kpack_archive_t archive,
                            const char* gpu_arch,
                            const std::vector<float>& host_O,
                            const std::vector<float>& host_dO,
                            const std::vector<float>& ref_D,
                            int batch,
                            int nhead,
                            int seqlen_q,
                            int hdim_v,
                            float p_undrop)
{
    const auto dtype         = variant.kernel.dtype;
    const int dtype_bytes    = rocm_ck::data_type_bits(dtype) / 8;
    const size_t total_elems = static_cast<size_t>(batch) * nhead * seqlen_q * hdim_v;
    const size_t total_d     = static_cast<size_t>(batch) * nhead * seqlen_q;
    const size_t buf_size    = total_elems * dtype_bytes;
    const size_t d_size      = total_d * sizeof(float); // D is always float

    // Load kernel code object
    void* kernel_code_object       = nullptr;
    size_t kernel_code_object_size = 0;
    kpack_error_t kerr             = kpack_get_kernel(
        archive, variant.name, gpu_arch, &kernel_code_object, &kernel_code_object_size);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "  %s: no kernel for '%s' (error %d)\n", variant.name, gpu_arch, kerr);
        return false;
    }

    hipModule_t module            = nullptr;
    hipFunction_t kernel_function = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
    HIP_CHECK(hipModuleGetFunction(&kernel_function, module, variant.name));

    // Allocate device buffers
    void* dev_O  = nullptr;
    void* dev_dO = nullptr;
    void* dev_D  = nullptr;
    HIP_CHECK(hipMalloc(&dev_O, buf_size));
    HIP_CHECK(hipMalloc(&dev_dO, buf_size));
    HIP_CHECK(hipMalloc(&dev_D, d_size));

    // Convert float to typed and upload
    std::vector<char> typed_O(buf_size);
    std::vector<char> typed_dO(buf_size);
    for(size_t i = 0; i < total_elems; ++i)
    {
        rocm_ck::float_to_typed(dtype, host_O[i], typed_O.data() + i * dtype_bytes);
        rocm_ck::float_to_typed(dtype, host_dO[i], typed_dO.data() + i * dtype_bytes);
    }

    HIP_CHECK(hipMemcpy(dev_O, typed_O.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dO, typed_dO.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_D, 0, d_size));

    // Strides (row-major: [batch, nhead, seqlen_q, hdim_v])
    const rocm_ck::index_t stride_o        = hdim_v;
    const rocm_ck::index_t stride_do       = hdim_v;
    const rocm_ck::index_t nhead_stride_o  = seqlen_q * hdim_v;
    const rocm_ck::index_t nhead_stride_do = seqlen_q * hdim_v;
    const rocm_ck::index_t nhead_stride_d  = seqlen_q;
    const rocm_ck::index_t batch_stride_o  = nhead * seqlen_q * hdim_v;
    const rocm_ck::index_t batch_stride_do = nhead * seqlen_q * hdim_v;
    const rocm_ck::index_t batch_stride_d  = nhead * seqlen_q;

    rocm_ck::FmhaBwdOGradDotOBatchArgs kernel_args = {
        .o_ptr  = dev_O,
        .do_ptr = dev_dO,
        .d_ptr  = dev_D,

        .p_undrop = p_undrop,

        .seqlen_q = static_cast<rocm_ck::index_t>(seqlen_q),
        .hdim_v   = static_cast<rocm_ck::index_t>(hdim_v),

        .stride_do = stride_do,
        .stride_o  = stride_o,

        .nhead_stride_do = nhead_stride_do,
        .nhead_stride_o  = nhead_stride_o,
        .nhead_stride_d  = nhead_stride_d,

        .batch_stride_do = batch_stride_do,
        .batch_stride_o  = batch_stride_o,
        .batch_stride_d  = batch_stride_d,
    };

    // Launch
    dim3 grid      = rocm_ck::grid_size(batch, nhead, seqlen_q, variant.kernel.block_size);
    int block_size = variant.kernel.block_size;

    std::printf(
        "  %s: grid=(%u,%u,%u), block=%d\n", variant.name, grid.x, grid.y, grid.z, block_size);

    size_t kernel_args_size = sizeof(kernel_args);
    void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               &kernel_args,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &kernel_args_size,
                               HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(kernel_function,
                                    grid.x,
                                    grid.y,
                                    grid.z,
                                    block_size,
                                    1,
                                    1,
                                    0,
                                    nullptr,
                                    nullptr,
                                    launch_config));
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify D
    std::vector<float> got_D(total_d);
    HIP_CHECK(hipMemcpy(got_D.data(), dev_D, d_size, hipMemcpyDeviceToHost));

    const float tol = rocm_ck::tolerance_for(dtype);
    bool passed     = true;
    for(size_t i = 0; i < total_d; ++i)
    {
        if(std::fabs(got_D[i] - ref_D[i]) > tol)
        {
            std::fprintf(stderr,
                         "  %s: MISMATCH at D[%zu]: got %f, expected %f (diff=%e)\n",
                         variant.name,
                         i,
                         got_D[i],
                         ref_D[i],
                         got_D[i] - ref_D[i]);
            passed = false;
            break;
        }
    }

    std::printf("  %s: %s\n", variant.name, passed ? "PASSED" : "FAILED");

    HIP_CHECK(hipFree(dev_O));
    HIP_CHECK(hipFree(dev_dO));
    HIP_CHECK(hipFree(dev_D));
    HIP_CHECK(hipModuleUnload(module));
    kpack_free_kernel(kernel_code_object);

    return passed;
}

// ---------------------------------------------------------------------------
// Group-mode variant runner
// ---------------------------------------------------------------------------

static bool runGroupVariant(const rocm_ck::FmhaBwdVariantDescriptor& variant,
                            kpack_archive_t archive,
                            const char* gpu_arch,
                            const std::vector<float>& host_O,
                            const std::vector<float>& host_dO,
                            const std::vector<float>& ref_D,
                            int batch,
                            int nhead,
                            int seqlen_q,
                            int hdim_v,
                            float p_undrop)
{
    const auto dtype         = variant.kernel.dtype;
    const int dtype_bytes    = rocm_ck::data_type_bits(dtype) / 8;
    const size_t total_elems = static_cast<size_t>(batch) * nhead * seqlen_q * hdim_v;
    const size_t total_d     = static_cast<size_t>(batch) * nhead * seqlen_q;
    const size_t buf_size    = total_elems * dtype_bytes;
    const size_t d_size      = total_d * sizeof(float);

    // Load kernel
    void* kernel_code_object       = nullptr;
    size_t kernel_code_object_size = 0;
    kpack_error_t kerr             = kpack_get_kernel(
        archive, variant.name, gpu_arch, &kernel_code_object, &kernel_code_object_size);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "  %s: no kernel for '%s' (error %d)\n", variant.name, gpu_arch, kerr);
        return false;
    }

    hipModule_t module            = nullptr;
    hipFunction_t kernel_function = nullptr;
    HIP_CHECK(hipModuleLoadData(&module, kernel_code_object));
    HIP_CHECK(hipModuleGetFunction(&kernel_function, module, variant.name));

    // Allocate device buffers
    void* dev_O  = nullptr;
    void* dev_dO = nullptr;
    void* dev_D  = nullptr;
    HIP_CHECK(hipMalloc(&dev_O, buf_size));
    HIP_CHECK(hipMalloc(&dev_dO, buf_size));
    HIP_CHECK(hipMalloc(&dev_D, d_size));

    // Convert and upload
    std::vector<char> typed_O(buf_size);
    std::vector<char> typed_dO(buf_size);
    for(size_t i = 0; i < total_elems; ++i)
    {
        rocm_ck::float_to_typed(dtype, host_O[i], typed_O.data() + i * dtype_bytes);
        rocm_ck::float_to_typed(dtype, host_dO[i], typed_dO.data() + i * dtype_bytes);
    }
    HIP_CHECK(hipMemcpy(dev_O, typed_O.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dO, typed_dO.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_D, 0, d_size));

    // Build seqstart_q: cumulative physical sequence lengths [batch + 1].
    // For simplicity, all sequences have the same length (seqlen_q).
    std::vector<int32_t> seqstart_q(batch + 1);
    for(int b = 0; b <= batch; ++b)
        seqstart_q[b] = b * seqlen_q;

    int32_t* dev_seqstart_q = nullptr;
    HIP_CHECK(hipMalloc(&dev_seqstart_q, seqstart_q.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(dev_seqstart_q,
                        seqstart_q.data(),
                        seqstart_q.size() * sizeof(int32_t),
                        hipMemcpyHostToDevice));

    // Group mode layout: O/dO as [total_seq, nhead, hdim_v], D as [nhead, total_seq].
    // Strides: stride = nhead * hdim_v, nhead_stride = hdim_v, nhead_stride_d = total_seq.
    // Re-layout from batch-mode [batch, nhead, seqlen_q, hdim_v] for the test.
    const int total_seq = batch * seqlen_q;

    // Re-layout O/dO from [batch, nhead, seqlen_q, hdim_v] to [total_seq, nhead, hdim_v]
    std::vector<float> group_O(total_elems);
    std::vector<float> group_dO(total_elems);
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            for(int s = 0; s < seqlen_q; ++s)
            {
                for(int v = 0; v < hdim_v; ++v)
                {
                    int src_idx       = ((b * nhead + h) * seqlen_q + s) * hdim_v + v;
                    int dst_idx       = ((b * seqlen_q + s) * nhead + h) * hdim_v + v;
                    group_O[dst_idx]  = host_O[src_idx];
                    group_dO[dst_idx] = host_dO[src_idx];
                }
            }
        }
    }

    // Re-upload with group layout
    std::vector<char> typed_group_O(buf_size);
    std::vector<char> typed_group_dO(buf_size);
    for(size_t i = 0; i < total_elems; ++i)
    {
        rocm_ck::float_to_typed(dtype, group_O[i], typed_group_O.data() + i * dtype_bytes);
        rocm_ck::float_to_typed(dtype, group_dO[i], typed_group_dO.data() + i * dtype_bytes);
    }
    HIP_CHECK(hipMemcpy(dev_O, typed_group_O.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_dO, typed_group_dO.data(), buf_size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dev_D, 0, d_size));

    // Group layout strides
    const rocm_ck::index_t g_stride_o        = nhead * hdim_v;
    const rocm_ck::index_t g_stride_do       = nhead * hdim_v;
    const rocm_ck::index_t g_nhead_stride_o  = hdim_v;
    const rocm_ck::index_t g_nhead_stride_do = hdim_v;
    const rocm_ck::index_t g_nhead_stride_d  = total_seq; // D layout: [nhead, total_seq]

    // Re-layout reference D from [batch, nhead, seqlen_q] to [nhead, total_seq]
    std::vector<float> group_ref_D(total_d);
    for(int b = 0; b < batch; ++b)
    {
        for(int h = 0; h < nhead; ++h)
        {
            for(int s = 0; s < seqlen_q; ++s)
            {
                int src_idx          = (b * nhead + h) * seqlen_q + s;
                int dst_idx          = h * total_seq + b * seqlen_q + s;
                group_ref_D[dst_idx] = ref_D[src_idx];
            }
        }
    }

    rocm_ck::FmhaBwdOGradDotOGroupArgs kernel_args = {
        .o_ptr  = dev_O,
        .do_ptr = dev_dO,
        .d_ptr  = dev_D,

        .p_undrop = p_undrop,

        .seqlen_q = -1, // updated per-batch on device
        .hdim_v   = static_cast<rocm_ck::index_t>(hdim_v),

        .stride_do = g_stride_do,
        .stride_o  = g_stride_o,

        .nhead_stride_do = g_nhead_stride_do,
        .nhead_stride_o  = g_nhead_stride_o,
        .nhead_stride_d  = g_nhead_stride_d,

        .seqstart_q_ptr  = dev_seqstart_q,
        .seqlen_q_ptr    = nullptr, // use seqstart_q for physical length
        .cu_seqlen_q_ptr = nullptr,
    };

    // For group mode, grid uses max_seqlen_q across all batches
    dim3 grid    = rocm_ck::grid_size(batch, nhead, seqlen_q, variant.kernel.block_size);
    int block_sz = variant.kernel.block_size;

    std::printf("  %s: grid=(%u,%u,%u), block=%d (group mode)\n",
                variant.name,
                grid.x,
                grid.y,
                grid.z,
                block_sz);

    size_t kernel_args_size = sizeof(kernel_args);
    void* launch_config[]   = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                               &kernel_args,
                               HIP_LAUNCH_PARAM_BUFFER_SIZE,
                               &kernel_args_size,
                               HIP_LAUNCH_PARAM_END};

    HIP_CHECK(hipModuleLaunchKernel(kernel_function,
                                    grid.x,
                                    grid.y,
                                    grid.z,
                                    block_sz,
                                    1,
                                    1,
                                    0,
                                    nullptr,
                                    nullptr,
                                    launch_config));
    HIP_CHECK(hipDeviceSynchronize());

    // Download and verify D (in group layout)
    std::vector<float> got_D(total_d);
    HIP_CHECK(hipMemcpy(got_D.data(), dev_D, d_size, hipMemcpyDeviceToHost));

    const float tol = rocm_ck::tolerance_for(dtype);
    bool passed     = true;
    for(size_t i = 0; i < total_d; ++i)
    {
        if(std::fabs(got_D[i] - group_ref_D[i]) > tol)
        {
            std::fprintf(stderr,
                         "  %s: MISMATCH at D[%zu]: got %f, expected %f (diff=%e)\n",
                         variant.name,
                         i,
                         got_D[i],
                         group_ref_D[i],
                         got_D[i] - group_ref_D[i]);
            passed = false;
            break;
        }
    }

    std::printf("  %s: %s\n", variant.name, passed ? "PASSED" : "FAILED");

    HIP_CHECK(hipFree(dev_O));
    HIP_CHECK(hipFree(dev_dO));
    HIP_CHECK(hipFree(dev_D));
    HIP_CHECK(hipFree(dev_seqstart_q));
    HIP_CHECK(hipModuleUnload(module));
    kpack_free_kernel(kernel_code_object);

    return passed;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        std::fprintf(stderr, "Usage: %s <path-to-kernels.kpack>\n", argv[0]);
        return 1;
    }

    // --- Open the kpack archive ---
    kpack_archive_t archive = nullptr;
    kpack_error_t kerr      = kpack_open(argv[1], &archive);
    if(kerr != KPACK_SUCCESS)
    {
        std::fprintf(stderr, "Failed to open archive '%s' (error %d)\n", argv[1], kerr);
        return 1;
    }

    size_t arch_count = 0;
    kpack_get_architecture_count(archive, &arch_count);
    std::printf("Opened %s — architectures:", argv[1]);
    for(size_t i = 0; i < arch_count; ++i)
    {
        const char* arch_name = nullptr;
        kpack_get_architecture(archive, i, &arch_name);
        std::printf("%s %s", (i > 0 ? "," : ""), arch_name);
    }
    std::printf("\n");

    // --- Detect current GPU architecture ---
    std::string gpu_arch = rocm_ck::get_gpu_arch();
    if(gpu_arch.empty())
    {
        std::fprintf(stderr, "Failed to detect GPU architecture\n");
        kpack_close(archive);
        return 1;
    }
    std::printf("Detected GPU: %s\n", gpu_arch.c_str());

    // --- Test parameters ---
    const int BATCH      = 2;
    const int NHEAD      = 2;
    const int SEQLEN_Q   = 128;
    const int HDIM_V     = 128; // default; d64 variant uses 64
    const float P_UNDROP = 1.0f;

    // --- Test data (small integers exactly representable in fp16/bf16) ---
    auto make_test_data = [](int batch,
                             int nhead,
                             int seqlen_q,
                             int hdim_v,
                             std::vector<float>& O,
                             std::vector<float>& dO) {
        const int total = batch * nhead * seqlen_q * hdim_v;
        O.resize(total);
        dO.resize(total);
        for(int i = 0; i < total; ++i)
        {
            O[i]  = static_cast<float>(i % 16);       // [0..15]
            dO[i] = static_cast<float>((i * 3) % 16); // [0..15]
        }
    };

    // --- Demonstrate findVariant ---
    std::printf("\nVariant selection:\n");
    for(auto dt : {rocm_ck::DataType::FP16, rocm_ck::DataType::BF16})
    {
        const auto* best = rocm_ck::findVariant(
            {.signature = {.dtype = dt, .hdim_v = HDIM_V, .mode = rocm_ck::FmhaMode::BATCH},
             .algorithm = {.pad_seqlen_q = true, .pad_hdim_v = true}});
        if(best)
            std::printf("  %s d%d batch -> %s\n", rocm_ck::data_type_name(dt), HDIM_V, best->name);
    }
    {
        const auto* best =
            rocm_ck::findVariant({.signature = {.dtype  = rocm_ck::DataType::FP16,
                                                .hdim_v = HDIM_V,
                                                .mode   = rocm_ck::FmhaMode::BATCH},
                                  .algorithm = {.pad_seqlen_q = false, .pad_hdim_v = false}});
        if(best)
            std::printf("  FP16 d%d batch (no-pad) -> %s\n", HDIM_V, best->name);
    }

    // --- Run all variants ---
    std::printf("\nRunning all %d variants:\n", ALL_FMHA_BWD_VARIANTS_COUNT);
    bool all_passed = true;

    for(const auto& variant : ALL_FMHA_BWD_VARIANTS)
    {
        const int cur_hdim = variant.kernel.hdim_v;

        std::vector<float> host_O, host_dO;
        make_test_data(BATCH, NHEAD, SEQLEN_Q, cur_hdim, host_O, host_dO);

        // CPU reference
        const int total_d = BATCH * NHEAD * SEQLEN_Q;
        std::vector<float> ref_D(total_d, 0.0f);
        cpuOGradDotO(host_O, host_dO, ref_D, BATCH, NHEAD, SEQLEN_Q, cur_hdim, P_UNDROP);

        bool passed;
        if(variant.kernel.mode == rocm_ck::FmhaMode::GROUP)
        {
            passed = runGroupVariant(variant,
                                     archive,
                                     gpu_arch.c_str(),
                                     host_O,
                                     host_dO,
                                     ref_D,
                                     BATCH,
                                     NHEAD,
                                     SEQLEN_Q,
                                     cur_hdim,
                                     P_UNDROP);
        }
        else
        {
            passed = runBatchVariant(variant,
                                     archive,
                                     gpu_arch.c_str(),
                                     host_O,
                                     host_dO,
                                     ref_D,
                                     BATCH,
                                     NHEAD,
                                     SEQLEN_Q,
                                     cur_hdim,
                                     P_UNDROP);
        }

        if(!passed)
            all_passed = false;
    }

    // --- Cleanup ---
    kpack_close(archive);

    std::printf("\n%s\n", all_passed ? "All variants PASSED" : "Some variants FAILED");
    return all_passed ? 0 : 1;
}
