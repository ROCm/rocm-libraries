// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Shared helpers for the user kernel registration tests.
//
// Registered kernels are process-global, so each test runs its DISABLED_ worker
// in a child process. This keeps them from leaking into other tests and lets
// the worker assert it starts with nothing registered.

#pragma once

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt-user-kernel.h>
#include <hipblaslt/hipblaslt.h>

#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

extern char** environ;

namespace user_kernel_test
{
    // Small M with large N and K.
    inline constexpr int64_t M = 16, N = 6144, K = 2048;

    inline constexpr uint64_t MAX_WORKSPACE = 64ull * 1024 * 1024;

    // Matches TensileLite::UserKernelIndexBase.
    inline constexpr int USER_INDEX_BASE = 3 << 29;

    // bf16 1.0. With A and B all ones, every element of D is exactly K.
    inline constexpr uint16_t BF16_ONE = 0x3F80;

    // Solution index carried at the head of an algo's opaque data.
    inline int algoIndex(hipblasLtMatmulAlgo_t const& algo)
    {
        int index = -1;
        std::memcpy(&index, algo.data, sizeof(index));
        return index;
    }

    inline std::string selfExe()
    {
        std::error_code ec;
        auto            path = std::filesystem::read_symlink("/proc/self/exe", ec);
        return ec ? std::string{} : path.string();
    }

    // PCI device id of the current GPU, e.g. "75a0", or empty if unreadable.
    // Shard filenames include the device ids they were built for.
    inline std::string devicePciId()
    {
        int device = 0;
        if(hipGetDevice(&device) != hipSuccess)
            return {};

        char bus[64] = {};
        if(hipDeviceGetPCIBusId(bus, sizeof(bus), device) != hipSuccess)
            return {};

        std::string lower(bus);
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        std::ifstream in("/sys/bus/pci/devices/" + lower + "/device");
        std::string   id;
        if(!in || !(in >> id))
            return {};
        if(id.rfind("0x", 0) == 0)
            id = id.substr(2);
        return id;
    }

    // Directories that may hold compiled kernel shards, relative to the
    // executable so the test works from the build tree or an install.
    inline std::vector<std::string> libraryRoots()
    {
        std::vector<std::string> roots;
        if(char const* env = std::getenv("HIPBLASLT_USER_KERNEL_TEST_LIB_DIR"))
            roots.emplace_back(env);

        static char const* const relative[] = {
            "Tensile/library", // build tree
            "lib/hipblaslt/library", // install tree
            "hipblaslt-install/lib/hipblaslt/library", // in-repo install
        };

        std::error_code ec;
        for(std::filesystem::path dir = std::filesystem::path(selfExe()).parent_path();
            !dir.empty() && dir != dir.root_path();
            dir = dir.parent_path())
        {
            for(auto const* suffix : relative)
            {
                auto candidate = dir / suffix;
                if(std::filesystem::is_directory(candidate, ec))
                    roots.push_back(candidate.string());
            }
        }
        return roots;
    }

    // Candidate bf16 TN payloads for this GPU, largest first since larger
    // shards are more likely to hold a usable kernel.
    inline std::vector<std::string> candidatePayloads()
    {
        const std::string pciId = devicePciId();

        std::vector<std::pair<uintmax_t, std::string>> found;
        std::error_code                                ec;

        for(auto const& root : libraryRoots())
        {
            for(auto const& arch : std::filesystem::directory_iterator(root, ec))
            {
                if(!arch.is_directory())
                    continue;
                for(auto const& entry : std::filesystem::directory_iterator(arch.path(), ec))
                {
                    std::string name = entry.path().filename().string();
                    if(name.rfind("TensileLibrary_BB_BB_", 0) != 0)
                        continue;
                    if(name.find("_Alik_Bljk_") == std::string::npos)
                        continue;

                    std::string lowerName(name);
                    std::transform(
                        lowerName.begin(), lowerName.end(), lowerName.begin(), ::tolower);
                    if(!pciId.empty() && lowerName.find(pciId) == std::string::npos)
                        continue;

                    // Pass the bare ".dat" name; the loader finds ".dat.zlib".
                    std::string stem = entry.path().string();
                    for(std::string suffix : {std::string(".zlib"), std::string(".dat")})
                        if(stem.size() > suffix.size()
                           && stem.compare(stem.size() - suffix.size(), suffix.size(), suffix) == 0)
                            stem.resize(stem.size() - suffix.size());
                    if(stem == entry.path().string())
                        continue; // not a .dat/.dat.zlib
                    if(!std::filesystem::exists(stem + ".co"))
                        continue;

                    found.emplace_back(entry.file_size(ec), stem);
                }
            }
            if(!found.empty())
                break;
        }

        std::sort(found.begin(), found.end(), [](auto const& a, auto const& b) {
            return a.first > b.first;
        });
        std::vector<std::string> stems;
        for(auto const& f : found)
            if(std::find(stems.begin(), stems.end(), f.second) == stems.end())
                stems.push_back(f.second);
        return stems;
    }

    struct Problem
    {
        int64_t                     m    = M;
        hipblasLtMatmulDesc_t       desc = nullptr;
        hipblasLtMatrixLayout_t     la = nullptr, lb = nullptr, lc = nullptr, ld = nullptr;
        hipblasLtMatmulPreference_t pref = nullptr;
    };

    // A bf16 TN GEMM at m x N x K, no epilogue.
    inline hipblasStatus_t buildProblem(Problem& p, int64_t m = M)
    {
        p.m     = m;
        auto st = hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        if(st != HIPBLAS_STATUS_SUCCESS)
            return st;

        hipblasOperation_t opT = HIPBLAS_OP_T, opN = HIPBLAS_OP_N;
        if((st = hipblasLtMatmulDescSetAttribute(
                p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT)))
           != HIPBLAS_STATUS_SUCCESS)
            return st;
        if((st = hipblasLtMatmulDescSetAttribute(
                p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)))
           != HIPBLAS_STATUS_SUCCESS)
            return st;

        if((st = hipblasLtMatrixLayoutCreate(&p.la, HIP_R_16BF, K, m, K)) != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.lb, HIP_R_16BF, K, N, K))
                  != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.lc, HIP_R_16BF, m, N, m))
                  != HIPBLAS_STATUS_SUCCESS
           || (st = hipblasLtMatrixLayoutCreate(&p.ld, HIP_R_16BF, m, N, m))
                  != HIPBLAS_STATUS_SUCCESS)
            return st;

        if((st = hipblasLtMatmulPreferenceCreate(&p.pref)) != HIPBLAS_STATUS_SUCCESS)
            return st;

        uint64_t maxWs = MAX_WORKSPACE;
        return hipblasLtMatmulPreferenceSetAttribute(
            p.pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs));
    }

    // The heuristic's top `count` choices, best first.
    inline std::vector<hipblasLtMatmulHeuristicResult_t>
        topResults(hipblasLtHandle_t handle, Problem const& p, int count)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> results(count);
        int                                           returned = 0;
        if(hipblasLtMatmulAlgoGetHeuristic(
               handle, p.desc, p.la, p.lb, p.lc, p.ld, p.pref, count, results.data(), &returned)
           != HIPBLAS_STATUS_SUCCESS)
            return {};
        results.resize(returned);
        return results;
    }

    inline std::vector<int> topIndices(hipblasLtHandle_t handle, Problem const& p, int count)
    {
        std::vector<int> indices;
        for(auto const& result : topResults(handle, p, count))
            indices.push_back(algoIndex(result.algo));
        return indices;
    }

    inline int topIndex(hipblasLtHandle_t handle, Problem const& p)
    {
        auto indices = topIndices(handle, p, 1);
        return indices.empty() ? -1 : indices[0];
    }

    // First index that resolves and can serve the problem within the workspace
    // budget, or -1.
    inline int findUsableKernel(hipblasLtHandle_t                 handle,
                                Problem const&                    p,
                                std::vector<int> const&           indices,
                                hipblasLtMatmulHeuristicResult_t& out,
                                size_t&                           workspace)
    {
        for(int idx : indices)
        {
            std::vector<int>                              one{idx};
            std::vector<hipblasLtMatmulHeuristicResult_t> got;
            if(hipblaslt_ext::getAlgosFromIndex(handle, one, got) != HIPBLAS_STATUS_SUCCESS
               || got.empty())
                continue;

            size_t ws    = 0;
            float  alpha = 1.0f, beta = 0.0f;
            if(hipblaslt_ext::matmulIsAlgoSupported(
                   handle, p.desc, &alpha, p.la, p.lb, &beta, p.lc, p.ld, got[0].algo, ws)
               != HIPBLAS_STATUS_SUCCESS)
                continue;
            if(ws > MAX_WORKSPACE)
                continue;

            out       = got[0];
            workspace = ws;
            return idx;
        }
        return -1;
    }

    struct Registered
    {
        std::string                      stem;
        int                              index     = -1;
        size_t                           workspace = 0;
        hipblasLtMatmulHeuristicResult_t result{};
    };

    // Registers payloads from `candidates`, skipping `avoid` unless nothing
    // else works, until one holds a kernel that can serve `p`. Not every shard
    // has one for this GPU and shape.
    inline Registered registerUsable(hipblasLtHandle_t               handle,
                                     Problem const&                  p,
                                     std::vector<std::string> const& candidates,
                                     std::string const&              avoid = {})
    {
        std::vector<std::string> ordered;
        for(auto const& stem : candidates)
            if(stem != avoid)
                ordered.push_back(stem);
        if(!avoid.empty())
            ordered.push_back(avoid);

        for(auto const& stem : ordered)
        {
            const std::string dat = stem + ".dat", co = stem + ".co";

            std::vector<int> indices(4096);
            int              numIndices = 0;
            if(hipblasLtUserKernelRegister(handle,
                                           dat.c_str(),
                                           co.c_str(),
                                           indices.data(),
                                           static_cast<int>(indices.size()),
                                           &numIndices)
               != HIPBLAS_STATUS_SUCCESS)
            {
                ADD_FAILURE() << "registration failed for " << stem;
                continue;
            }
            indices.resize(std::min<size_t>(numIndices, indices.size()));

            Registered r;
            r.stem  = stem;
            r.index = findUsableKernel(handle, p, indices, r.result, r.workspace);
            if(r.index >= 0)
                return r;
        }
        return {};
    }

    struct DeviceBuffer
    {
        void* ptr = nullptr;

        explicit DeviceBuffer(size_t bytes)
        {
            if(hipMalloc(&ptr, bytes) != hipSuccess)
                ptr = nullptr;
        }
        ~DeviceBuffer()
        {
            if(ptr)
                static_cast<void>(hipFree(ptr));
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
    };

    inline float bf16ToFloat(uint16_t bits)
    {
        uint32_t widened = static_cast<uint32_t>(bits) << 16;
        float    value;
        std::memcpy(&value, &widened, sizeof(value));
        return value;
    }

    // Runs `algo` on the current device with A and B all ones and checks that
    // every element of D is K.
    inline void runAndCheck(hipblasLtHandle_t            handle,
                            Problem const&               p,
                            hipblasLtMatmulAlgo_t const& algo,
                            size_t                       workspace)
    {
        const size_t          m = static_cast<size_t>(p.m);
        std::vector<uint16_t> hA(K * m, BF16_ONE), hB(K * N, BF16_ONE);
        DeviceBuffer          dA(hA.size() * 2), dB(hB.size() * 2);
        DeviceBuffer          dC(m * N * 2), dD(m * N * 2), dWs(MAX_WORKSPACE);
        ASSERT_NE(dA.ptr, nullptr);
        ASSERT_NE(dB.ptr, nullptr);
        ASSERT_NE(dC.ptr, nullptr);
        ASSERT_NE(dD.ptr, nullptr);
        ASSERT_NE(dWs.ptr, nullptr);
        ASSERT_EQ(hipMemcpy(dA.ptr, hA.data(), hA.size() * 2, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemcpy(dB.ptr, hB.data(), hB.size() * 2, hipMemcpyHostToDevice), hipSuccess);
        ASSERT_EQ(hipMemset(dC.ptr, 0, m * N * 2), hipSuccess);
        ASSERT_EQ(hipMemset(dD.ptr, 0, m * N * 2), hipSuccess);

        float alpha = 1.0f, beta = 0.0f;
        ASSERT_EQ(hipblasLtMatmul(handle,
                                  p.desc,
                                  &alpha,
                                  dA.ptr,
                                  p.la,
                                  dB.ptr,
                                  p.lb,
                                  &beta,
                                  dC.ptr,
                                  p.lc,
                                  dD.ptr,
                                  p.ld,
                                  &algo,
                                  dWs.ptr,
                                  workspace,
                                  nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        std::vector<uint16_t> hD(m * N, 0);
        ASSERT_EQ(hipMemcpy(hD.data(), dD.ptr, hD.size() * 2, hipMemcpyDeviceToHost), hipSuccess);

        bool exact = std::all_of(hD.begin(), hD.end(), [](uint16_t bits) {
            return bf16ToFloat(bits) == static_cast<float>(K);
        });
        EXPECT_TRUE(exact) << "unexpected result: D[0] = " << bf16ToFloat(hD[0]) << " expected "
                           << K;
    }

    // Re-runs this executable with only the named DISABLED_ case enabled and
    // returns its exit status, or -1 if it could not be run. Uses posix_spawn
    // because forking after HIP initialization is unsafe.
    inline int runWorker(std::string const& filter)
    {
        std::string exe = selfExe();
        if(exe.empty())
            return -1;

        std::string alsoDisabled = "--gtest_also_run_disabled_tests";
        std::string filterArg    = "--gtest_filter=" + filter;
        char*       argv[]       = {exe.data(), alsoDisabled.data(), filterArg.data(), nullptr};

        pid_t pid = 0;
        if(posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv, environ) != 0)
            return -1;

        int status = 0;
        if(waitpid(pid, &status, 0) < 0)
            return -1;
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
} // namespace user_kernel_test
