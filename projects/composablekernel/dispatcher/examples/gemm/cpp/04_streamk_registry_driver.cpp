// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * Stream-K GEMM driver through the Registry + Dispatcher (deep-core path).
 *
 * Unlike 03_streamk_gemm_driver.cpp (which calls SelectedKernel::launch()
 * DIRECTLY, bypassing the dispatcher), this driver proves the full deep-core
 * path that PR-A..PR-C built:
 *
 *     Registry::register_kernel(GeneratedStreamKKernelInstance)
 *         -> Dispatcher::run(Problem.stream_k(Atomic))
 *         -> Dispatcher::select_first_fit -> SK instance.supports()
 *         -> GeneratedStreamKKernelInstance::run -> SelectedKernel::launch()
 *
 * It registers ONE generated Stream-K kernel (force-included via
 * -include / -DCK_TILE_SINGLE_KERNEL_INCLUDE), selects it through the registry
 * by Problem::reduction_strategy, runs it, and verifies vs reference_gemm.
 *
 * Build (single-kernel include style):
 *   hipcc -std=c++17 --offload-arch=gfx942 -O3 \
 *     -DCK_TILE_SINGLE_KERNEL_INCLUDE \
 *     -I <ck>/include -I <ck>/dispatcher/include -I <generated_dir> \
 *     -include <generated_dir>/<one>_streamk.hpp \
 *     04_streamk_registry_driver.cpp -o streamk_registry_driver
 */

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"

#include "ck_tile/dispatcher/dispatcher.hpp"
#include "ck_tile/dispatcher/registry.hpp"
#include "ck_tile/dispatcher/backends/generated_tile_backend_streamk.hpp"

// The generated stream-K kernel header is injected on the command line with
// -include and -DCK_TILE_SINGLE_KERNEL_INCLUDE. It exports into the global
// namespace: SelectedKernel, ADataType, BDataType, CDataType, AccDataType,
// ALayout, BLayout, CLayout, and KERNEL_NAME.

#ifndef GFX_ARCH
#define GFX_ARCH "gfx942"
#endif

using namespace ck_tile::dispatcher;
using namespace ck_tile::dispatcher::backends;
using Priority = ck_tile::dispatcher::Registry::Priority;

template <typename Layout>
static constexpr inline auto is_row_major(Layout)
{
    return ck_tile::bool_constant<
        std::is_same_v<ck_tile::remove_cvref_t<Layout>,
                       ck_tile::tensor_layout::gemm::RowMajor>>{};
}

static std::string get_opt(int argc, char** argv, const std::string& key, const std::string& def)
{
    for(int i = 1; i < argc - 1; ++i)
        if(key == argv[i]) return argv[i + 1];
    return def;
}

// Build the KernelKey for the force-included Stream-K kernel. Only the Stream-K
// axis (streamk + reduction_strategy) governs selection; the remaining fields
// are populated for a faithful encode_identifier()/registry entry.
static KernelKey make_streamk_key(ReductionStrategy strategy)
{
    KernelKey key;
    key.signature.dtype_a             = DataType::FP16;
    key.signature.dtype_b             = DataType::FP16;
    key.signature.dtype_c             = DataType::FP16;
    key.signature.dtype_acc           = DataType::FP32;
    key.signature.layout_a            = LayoutTag::RowMajor;
    key.signature.layout_b            = LayoutTag::ColMajor;
    key.signature.layout_c            = LayoutTag::RowMajor;
    key.signature.transpose_a         = false;
    key.signature.transpose_b         = false;
    key.signature.grouped             = false;
    key.signature.split_k             = 1;
    key.signature.elementwise_op      = "PassThrough";
    key.signature.num_d_tensors       = 0;
    key.signature.structured_sparsity = false;

    key.algorithm.tile_shape      = {SelectedKernel::TileM, SelectedKernel::TileN, SelectedKernel::TileK};
    key.algorithm.warp_tile_shape = {static_cast<std::uint8_t>(SelectedKernel::WarpTileM),
                                     static_cast<std::uint8_t>(SelectedKernel::WarpTileN),
                                     static_cast<std::uint8_t>(SelectedKernel::WarpTileK)};
    key.algorithm.wave_shape      = {2, 2, 1};
    key.algorithm.pipeline        = Pipeline::CompV3;
    key.algorithm.scheduler       = Scheduler::Intrawave;
    key.algorithm.epilogue        = Epilogue::CShuffle;
    key.algorithm.block_size      = 256;
    key.algorithm.double_buffer   = false;
    key.algorithm.persistent      = false;
    key.algorithm.preshuffle      = false;
    key.algorithm.transpose_c     = false;
    key.algorithm.num_wave_groups = 1;
    key.algorithm.pad_m           = SelectedKernel::kPadM;
    key.algorithm.pad_n           = SelectedKernel::kPadN;
    key.algorithm.pad_k           = SelectedKernel::kPadK;

    // The Stream-K selection axis (the whole point of this path).
    key.algorithm.streamk             = true;
    key.algorithm.reduction_strategy  = strategy;
    key.algorithm.workspace           = (strategy != ReductionStrategy::Atomic);

    key.gfx_arch = GFX_ARCH;
    return key;
}

static ReductionStrategy parse_strategy(const std::string& s)
{
    if(s == "linear") return ReductionStrategy::Linear;
    if(s == "tree") return ReductionStrategy::Tree;
    return ReductionStrategy::Atomic;
}

int main(int argc, char** argv)
{
    const ck_tile::index_t M = std::stoll(get_opt(argc, argv, "--m", "3840"));
    const ck_tile::index_t N = std::stoll(get_opt(argc, argv, "--n", "4096"));
    const ck_tile::index_t K = std::stoll(get_opt(argc, argv, "--k", "2048"));
    const bool validate      = get_opt(argc, argv, "--validate", "1") != "0";
    const ReductionStrategy strategy =
        parse_strategy(get_opt(argc, argv, "--strategy", "atomic"));

    std::cout << "Kernel: " << KERNEL_NAME << "\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K
              << " strategy=" << to_string(strategy) << "\n";

    // --- Register the kernel into the global registry ---------------------------
    KernelKey key = make_streamk_key(strategy);
    auto kernel   = create_generated_streamk_kernel<SelectedKernel,
                                                  ADataType,
                                                  BDataType,
                                                  CDataType,
                                                  AccDataType>(key, KERNEL_NAME);
    Registry::instance().clear();
    Registry::instance().register_kernel(kernel, Priority::High);
    std::cout << "Registered kernels: " << Registry::instance().size()
              << "  identifier=" << key.encode_identifier() << "\n";

    // --- Build the problem requesting THIS Stream-K strategy --------------------
    Problem problem(M, N, K);
    problem.streamk            = true;
    problem.reduction_strategy = strategy;

    Dispatcher dispatcher;
    auto selected = dispatcher.select_kernel(problem);
    if(!selected)
    {
        std::cout << "Dispatcher selected NO kernel for the Stream-K problem -> FAIL\n";
        return 1;
    }
    std::cout << "Dispatcher selected: " << selected->get_name() << "\n";

    // --- Tensors (rcr) ---------------------------------------------------------
    const ck_tile::index_t sA = ck_tile::get_default_stride(M, K, 0, is_row_major(ALayout{}));
    const ck_tile::index_t sB = ck_tile::get_default_stride(K, N, 0, is_row_major(BLayout{}));
    const ck_tile::index_t sC = ck_tile::get_default_stride(M, N, 0, is_row_major(CLayout{}));

    ck_tile::HostTensor<ADataType> a_host(
        ck_tile::host_tensor_descriptor(M, K, sA, is_row_major(ALayout{})));
    ck_tile::HostTensor<BDataType> b_host(
        ck_tile::host_tensor_descriptor(K, N, sB, is_row_major(BLayout{})));
    ck_tile::HostTensor<CDataType> c_host(
        ck_tile::host_tensor_descriptor(M, N, sC, is_row_major(CLayout{})));

    ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_host);
    ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_host);
    c_host.SetZero();

    ck_tile::DeviceMem a_dev(a_host);
    ck_tile::DeviceMem b_dev(b_host);
    ck_tile::DeviceMem c_dev(c_host);
    c_dev.SetZero();

    // --- Run through the dispatcher (registry -> Dispatcher::run -> SK backend) -
    float ave_time = 0.f;
    try
    {
        ave_time = dispatcher.run(
            a_dev.GetDeviceBuffer(), b_dev.GetDeviceBuffer(), c_dev.GetDeviceBuffer(), problem);
    }
    catch(const std::exception& e)
    {
        std::cout << "Dispatcher::run threw: " << e.what() << " -> FAIL\n";
        return 1;
    }

    const std::size_t flop  = std::size_t(2) * M * N * K;
    const std::size_t bytes = sizeof(ADataType) * M * K + sizeof(BDataType) * K * N +
                              sizeof(CDataType) * M * N;
    const float tflops = static_cast<float>(flop) / 1.E9 / ave_time;
    const float gbps   = static_cast<float>(bytes) / 1.E6 / ave_time;
    std::cout << "Perf: " << std::setw(10) << ave_time << " ms, " << tflops << " TFlops, " << gbps
              << " GB/s\n";

    c_dev.FromDevice(c_host.data());

    bool pass = true;
    if(validate)
    {
        ck_tile::HostTensor<CDataType> ref(
            ck_tile::host_tensor_descriptor(M, N, sC, is_row_major(CLayout{})));
        ref.SetZero();
        ck_tile::reference_gemm<ADataType, BDataType, AccDataType, CDataType>(a_host, b_host, ref);
        const float maxv = *std::max_element(ref.mData.begin(), ref.mData.end());
        const auto rtol  = ck_tile::get_relative_threshold<ADataType, CDataType, AccDataType>(K);
        const auto atol =
            ck_tile::get_absolute_threshold<ADataType, CDataType, AccDataType>(maxv, K);
        pass = ck_tile::check_err(c_host, ref, "streamk_registry", rtol, atol);
        std::cout << "Verification: " << (pass ? "PASS" : "FAIL") << "\n";
    }

    return pass ? 0 : 1;
}
