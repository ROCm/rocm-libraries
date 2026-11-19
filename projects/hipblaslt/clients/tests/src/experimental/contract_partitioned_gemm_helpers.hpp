// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Helpers shared by the PartitionedGemm tests: the contract tests in
// contract_partitioned_gemm_gtest.cpp and the other tests in ../partitioned_gemm_gtest.cpp.
//
// Part of the contract of experimental feature PARTITIONED_GEMM: the contract tests' results
// depend on these helpers, so changing this file needs a revision bump, as changing a contract
// test does (tools/scripts/check_experimental_api.py checks both).

#pragma once

#if !defined(ROCM_ENABLE_EXPERIMENTAL_APIS) || !ROCM_ENABLE_EXPERIMENTAL_APIS
#error "contract_partitioned_gemm_helpers.hpp needs ROCM_ENABLE_EXPERIMENTAL_APIS defined to 1"
#endif
#include <hipblaslt/hipblaslt-experimental.h>

#include "partition_devices.hpp"
#include "utility.hpp"
#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace partitioned_gemm_test
{
    using namespace hipblaslt_ext;
    using hipblaslt_experimental::PartitionedGemm;
    using hipblaslt_experimental::SplitDim;

    using half_t = _Float16;

    // A device buffer on one device.
    class DeviceBuf
    {
    public:
        DeviceBuf(int device, size_t bytes, const void* init = nullptr)
            : m_device(device)
        {
            EXPECT_EQ(hipSetDevice(device), hipSuccess);
            EXPECT_EQ(hipMalloc(&m_ptr, bytes), hipSuccess);
            if(init != nullptr)
                EXPECT_EQ(hipMemcpy(m_ptr, init, bytes, hipMemcpyHostToDevice), hipSuccess);
            else
                EXPECT_EQ(hipMemset(m_ptr, 0, bytes), hipSuccess);
        }
        ~DeviceBuf()
        {
            if(m_ptr != nullptr)
            {
                static_cast<void>(hipSetDevice(m_device));
                static_cast<void>(hipFree(m_ptr));
            }
        }
        DeviceBuf(const DeviceBuf&)            = delete;
        DeviceBuf& operator=(const DeviceBuf&) = delete;
        void*      get() const
        {
            return m_ptr;
        }

    private:
        int   m_device;
        void* m_ptr = nullptr;
    };

    // The pointers a test hands to PartitionedGemm::setProblem, one entry per partition. Kept as
    // lists so a test can swap or repeat entries; toGemmInputs turns them into one GemmInputs
    // per partition.
    struct Pointers
    {
        std::vector<const void*> a, b, c;
        std::vector<void*>       d;
        bool                     storePeerD = false;
        const void*              alpha      = nullptr;
        const void*              beta       = nullptr;
    };

    inline std::vector<GemmInputs> toGemmInputs(Pointers const& in)
    {
        // The lists may have the wrong length on purpose; the library reports it.
        const size_t            n = in.d.size();
        std::vector<GemmInputs> v(n);
        for(size_t i = 0; i < n; i++)
        {
            if(i < in.a.size())
                v[i].setA(in.a[i]);
            if(i < in.b.size())
                v[i].setB(in.b[i]);
            if(i < in.c.size())
                v[i].setC(in.c[i]);
            v[i].setD(in.d[i]);
            v[i].setAlpha(in.alpha);
            v[i].setBeta(in.beta);
        }
        return v;
    }

    // PartitionedGemm is validated on gfx1250 only; the tests skip on any other architecture.
    inline bool isGfx1250()
    {
        hipDeviceProp_t props;
        int             device = 0;
        if(hipGetDevice(&device) != hipSuccess
           || hipGetDeviceProperties(&props, device) != hipSuccess)
            return false;
        return std::strncmp(props.gcnArchName, "gfx1250", 7) == 0;
    }

    inline std::vector<half_t> randomHalf(size_t count, unsigned seed)
    {
        uint32_t            state = seed * 2654435761u + 1;
        std::vector<half_t> v(count);
        for(auto& x : v)
        {
            state = state * 1664525u + 1013904223u;
            x     = half_t(float(state >> 9) / float(1u << 23) - 0.5f);
        }
        return v;
    }

    // One handle, stream and workspace per partition. `devices[i]` is the device of partition i.
    struct Partitions
    {
        explicit Partitions(std::vector<int> const& devices)
            : devices(devices)
        {
            for(int device : devices)
            {
                EXPECT_EQ(hipSetDevice(device), hipSuccess);
                handles.push_back(std::make_unique<hipblaslt_local_handle>());
                hipStream_t stream = nullptr;
                EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
                streams.push_back(stream);
                workspaceBufs.push_back(std::make_unique<DeviceBuf>(device, kWorkspace));
            }
        }
        ~Partitions()
        {
            for(size_t i = 0; i < streams.size(); i++)
            {
                static_cast<void>(hipSetDevice(devices[i]));
                static_cast<void>(hipStreamSynchronize(streams[i]));
                static_cast<void>(hipStreamDestroy(streams[i]));
            }
        }

        std::vector<hipblasLtHandle_t> rawHandles() const
        {
            std::vector<hipblasLtHandle_t> out;
            for(auto const& h : handles)
                out.push_back(static_cast<const hipblasLtHandle_t&>(*h));
            return out;
        }
        std::vector<void*> workspaces() const
        {
            std::vector<void*> out;
            for(auto const& w : workspaceBufs)
                out.push_back(w->get());
            return out;
        }
        void sync() const
        {
            for(size_t i = 0; i < streams.size(); i++)
            {
                ASSERT_EQ(hipSetDevice(devices[i]), hipSuccess);
                ASSERT_EQ(hipStreamSynchronize(streams[i]), hipSuccess);
            }
        }

        static constexpr size_t                              kWorkspace = 32 << 20;
        std::vector<int>                                     devices;
        std::vector<std::unique_ptr<hipblaslt_local_handle>> handles;
        std::vector<hipStream_t>                             streams;
        std::vector<std::unique_ptr<DeviceBuf>>              workspaceBufs;
    };

    // The full problem used by every case: fp16 in/out, fp32 compute, TN, column-major.
    constexpr int64_t kM = 256, kN = 192, kK = 128;
    constexpr int64_t kLda = kK, kLdb = kK, kLdc = kM, kLdd = kM; // op(A)=T: K x M, op(B)=N: K x N

    struct Operands
    {
        std::vector<half_t> a = randomHalf(kLda * kM, 1);
        std::vector<half_t> b = randomHalf(kLdb * kN, 2);
        std::vector<half_t> c = std::vector<half_t>(kLdc * kN, half_t(0));
    };

    // D = A * B on device `device`, with the plain Gemm. The reference for every comparison.
    inline std::vector<half_t>
        referenceGemm(Operands const& op, hipblasLtHandle_t handle, int device)
    {
        std::vector<half_t> out(kLdd * kN);
        EXPECT_EQ(hipSetDevice(device), hipSuccess);
        DeviceBuf a(device, op.a.size() * 2, op.a.data()), b(device, op.b.size() * 2, op.b.data()),
            c(device, op.c.size() * 2, op.c.data()), d(device, out.size() * 2),
            ws(device, Partitions::kWorkspace);
        float           alpha = 1.f, beta = 0.f;
        Gemm            gemm(handle,
                  HIPBLAS_OP_T,
                  HIPBLAS_OP_N,
                  HIP_R_16F,
                  HIP_R_16F,
                  HIP_R_16F,
                  HIP_R_16F,
                  HIPBLAS_COMPUTE_32F);
        GemmEpilogue    epilogue;
        GemmInputs      inputs;
        GemmProblemType type(HIPBLAS_OP_T,
                             HIPBLAS_OP_N,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIP_R_16F,
                             HIPBLAS_COMPUTE_32F);
        inputs.setA(a.get());
        inputs.setB(b.get());
        inputs.setC(c.get());
        inputs.setD(d.get());
        inputs.setAlpha(&alpha);
        inputs.setBeta(&beta);
        EXPECT_EQ(gemm.setProblem(
                      kM, kN, kK, 1, kLda, kLdb, kLdc, kLdd, 0, 0, 0, 0, epilogue, inputs, type),
                  HIPBLAS_STATUS_SUCCESS);
        GemmPreference pref;
        pref.setMaxWorkspaceBytes(Partitions::kWorkspace);
        gemm.setMaxWorkspaceBytes(Partitions::kWorkspace);
        std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
        EXPECT_EQ(gemm.algoGetHeuristic(1, pref, heuristic), HIPBLAS_STATUS_SUCCESS);
        if(heuristic.empty())
            return {};
        hipStream_t stream = nullptr;
        EXPECT_EQ(hipStreamCreate(&stream), hipSuccess);
        EXPECT_EQ(gemm.initialize(heuristic[0].algo, ws.get(), true, stream),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(gemm.run(stream), HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(hipStreamSynchronize(stream), hipSuccess);
        EXPECT_EQ(hipStreamDestroy(stream), hipSuccess);
        EXPECT_EQ(hipMemcpy(out.data(), d.get(), out.size() * 2, hipMemcpyDeviceToHost),
                  hipSuccess);
        return out;
    }

    inline PartitionedGemm makeGemm(Partitions const& parts)
    {
        return PartitionedGemm(parts.rawHandles(),
                               HIPBLAS_OP_T,
                               HIPBLAS_OP_N,
                               HIP_R_16F,
                               HIP_R_16F,
                               HIP_R_16F,
                               HIP_R_16F,
                               HIPBLAS_COMPUTE_32F);
    }

    // Arguments with one whole copy of every matrix per partition (Replicated).
    struct ReplicatedBuffers
    {
        ReplicatedBuffers(Partitions const& parts, Operands const& op)
        {
            for(int device : parts.devices)
            {
                bufs.push_back(std::make_unique<DeviceBuf>(device, op.a.size() * 2, op.a.data()));
                a.push_back(bufs.back()->get());
                bufs.push_back(std::make_unique<DeviceBuf>(device, op.b.size() * 2, op.b.data()));
                b.push_back(bufs.back()->get());
                bufs.push_back(std::make_unique<DeviceBuf>(device, op.c.size() * 2, op.c.data()));
                c.push_back(bufs.back()->get());
                bufs.push_back(std::make_unique<DeviceBuf>(device, kLdd * kN * 2));
                d.push_back(bufs.back()->get());
            }
        }
        Pointers inputs()
        {
            Pointers in;
            in.a.assign(a.begin(), a.end());
            in.b.assign(b.begin(), b.end());
            in.c.assign(c.begin(), c.end());
            in.d     = d;
            in.alpha = &alpha;
            in.beta  = &beta;
            return in;
        }
        std::vector<std::unique_ptr<DeviceBuf>> bufs;
        std::vector<void*>                      a, b, c, d;
        float                                   alpha = 1.f, beta = 0.f;
    };

    inline hipblasStatus_t setReplicated(
        PartitionedGemm& gemm, SplitDim dim, Pointers const& in, int64_t m = kM, int64_t n = kN)
    {
        return gemm.setProblem(
            m, n, kK, 1, kLda, kLdb, kLdc, kLdd, 0, 0, 0, 0, dim, toGemmInputs(in), in.storePeerD);
    }

    // Copies every partition's slice of D back and compares it with the reference.
    inline void expectSlicesMatch(PartitionedGemm&           gemm,
                                  Partitions const&          parts,
                                  ReplicatedBuffers const&   buffers,
                                  std::vector<half_t> const& ref,
                                  SplitDim                   dim)
    {
        for(size_t i = 0; i < parts.devices.size(); i++)
        {
            int64_t offset = 0, size = 0;
            ASSERT_EQ(gemm.getSlice(i, offset, size), HIPBLAS_STATUS_SUCCESS);
            std::vector<half_t> got(kLdd * kN);
            ASSERT_EQ(hipSetDevice(parts.devices[i]), hipSuccess);
            ASSERT_EQ(hipMemcpy(got.data(), buffers.d[i], got.size() * 2, hipMemcpyDeviceToHost),
                      hipSuccess);
            size_t bad = 0;
            for(int64_t col = 0; col < kN; col++)
                for(int64_t row = 0; row < kM; row++)
                {
                    const int64_t split = dim == SplitDim::M ? row : col;
                    if(split < offset || split >= offset + size)
                        continue;
                    const float want = float(ref[row + col * kLdd]);
                    const float have = float(got[row + col * kLdd]);
                    if(std::fabs(want - have) > 2e-3f * std::max(1.f, std::fabs(want)))
                        bad++;
                }
            EXPECT_EQ(bad, 0u) << "partition " << i << " slice [" << offset << ", " << offset + size
                               << ")";
        }
    }

    // The partitions of a single-GPU run are extra handles on device 0.
    inline Partitions onDeviceZero(size_t count)
    {
        return Partitions(std::vector<int>(count, 0));
    }
} // namespace partitioned_gemm_test

// Skips the rest of the test unless the device is a gfx1250.
#define SKIP_UNLESS_GFX1250()                                            \
    do                                                                   \
    {                                                                    \
        if(!isGfx1250())                                                 \
            GTEST_SKIP() << "PartitionedGemm is tested on gfx1250 only"; \
    } while(0)
