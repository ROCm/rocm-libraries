// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "hipblaslt-jit-replay.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

// Helpers the JIT tests share.
namespace hipblaslt_jit_test
{
    inline void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    // Requires that f throws.
    inline void reject(const std::function<void()>& f, const std::string& message)
    {
        bool failed = false;
        try
        {
            f();
        }
        catch(const std::exception&)
        {
            failed = true;
        }
        require(failed, message);
    }

    inline std::string readFile(const std::filesystem::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        require(bool(input), "Cannot read " + path.u8string());
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    inline void writeFile(const std::filesystem::path& path, const std::string& bytes)
    {
        std::ofstream file(path, std::ios::binary);
        require(bool(file.write(bytes.data(), bytes.size())), "Cannot write " + path.u8string());
    }

    // The written bundles of the processor in a device's gcnArchName, such as
    // root/gfx942 for "gfx942:sramecc+:xnack-".
    inline std::filesystem::path deviceBundles(const std::filesystem::path& root,
                                               const std::string&           gcnArchName)
    {
        const auto bundles = root / gcnArchName.substr(0, gcnArchName.find(':'));
        require(std::filesystem::is_directory(bundles),
                "No JIT test bundles for " + gcnArchName + " in " + root.u8string());
        return bundles;
    }

    // A suffix of equal length matches. "abc" ends with "abc"; "ab" does not end with "abc".
    inline bool endsWith(const std::string& text, const std::string& suffix)
    {
        return text.size() >= suffix.size()
               && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
    }

    inline void checkHip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }

    inline void checkBlas(hipblasStatus_t status, const char* expression)
    {
        require(status == HIPBLAS_STATUS_SUCCESS,
                std::string(expression) + ": status " + std::to_string(status));
    }

    struct Device
    {
        void* pointer{};
        explicit Device(size_t bytes)
        {
            checkHip(hipMalloc(&pointer, bytes), "hipMalloc");
        }
        ~Device()
        {
            static_cast<void>(hipFree(pointer));
        }
        Device(const Device&)            = delete;
        Device& operator=(const Device&) = delete;
    };

    // A and B are small multiples of 1/8 and C of 1/4, so every product and sum is exact.
    inline std::vector<__half> fillHalf(size_t count, int a, int b, int mod, float scale)
    {
        std::vector<__half> values(count);
        for(size_t i = 0; i < count; ++i)
            values[i] = __float2half(float(int(i * a + i / 7 * b) % mod - mod / 2) * scale);
        return values;
    }

    // Host reference for an FP16 NN GEMM. Prints "PASS " and label when label is not empty.
    inline void checkFp16(const std::string&         label,
                          int                        rows,
                          int                        cols,
                          int                        k,
                          const std::vector<__half>& a,
                          const std::vector<__half>& b,
                          const std::vector<__half>& c,
                          const void*                d,
                          float                      alpha,
                          float                      beta)
    {
        std::vector<__half> out(static_cast<size_t>(rows) * cols);
        checkHip(hipMemcpy(out.data(), d, out.size() * sizeof(__half), hipMemcpyDeviceToHost),
                 "hipMemcpy D");
        for(int col = 0; col < cols; ++col)
            for(int row = 0; row < rows; ++row)
            {
                float sum = 0;
                for(int index = 0; index < k; ++index)
                    sum += __half2float(a[row + index * rows]) * __half2float(b[index + col * k]);
                const auto i        = row + col * rows;
                const auto expected = __half2float(
                    __float2half(alpha * sum + beta * __half2float(c[i])));
                const auto actual = __half2float(out[i]);
                const auto where = label.empty() ? std::string("D[") : label + ": D[";
                require(std::isfinite(actual)
                            && std::abs(actual - expected) <= 0.0005f + 0.001f * std::abs(expected),
                        where + std::to_string(i) + "] is " + std::to_string(actual)
                            + ", expected " + std::to_string(expected));
            }
        if(!label.empty())
            std::cout << "PASS " << label << '\n';
    }

    // The FP16 NN problem the JIT bundles solve, with host and device matrices.
    struct Fp16Gemm
    {
        static constexpr int rows = 256;
        static constexpr int cols = 128;

        int                        k = 0;
        std::vector<__half>        hostA, hostB, hostC;
        Device                     A, B, C, D;
        hipblasLtMatrixLayout_t    layoutA{}, layoutB{}, layoutC{}, layoutD{};

        explicit Fp16Gemm(int bound)
            : k(bound)
            , hostA(fillHalf(static_cast<size_t>(rows) * bound, 3, 5, 13, 1 / 8.0f))
            , hostB(fillHalf(static_cast<size_t>(bound) * cols, 7, 2, 11, 1 / 8.0f))
            , hostC(fillHalf(static_cast<size_t>(rows) * cols, 1, 3, 7, 1 / 4.0f))
            , A(hostA.size() * sizeof(__half))
            , B(hostB.size() * sizeof(__half))
            , C(hostC.size() * sizeof(__half))
            , D(static_cast<size_t>(rows) * cols * sizeof(__half))
        {
            checkHip(hipMemcpy(A.pointer, hostA.data(), hostA.size() * sizeof(__half),
                               hipMemcpyHostToDevice),
                     "hipMemcpy A");
            checkHip(hipMemcpy(B.pointer, hostB.data(), hostB.size() * sizeof(__half),
                               hipMemcpyHostToDevice),
                     "hipMemcpy B");
            checkHip(hipMemcpy(C.pointer, hostC.data(), hostC.size() * sizeof(__half),
                               hipMemcpyHostToDevice),
                     "hipMemcpy C");
            checkBlas(hipblasLtMatrixLayoutCreate(&layoutA, HIP_R_16F, rows, k, rows), "layout A");
            checkBlas(hipblasLtMatrixLayoutCreate(&layoutB, HIP_R_16F, k, cols, k), "layout B");
            checkBlas(hipblasLtMatrixLayoutCreate(&layoutC, HIP_R_16F, rows, cols, rows), "layout C");
            checkBlas(hipblasLtMatrixLayoutCreate(&layoutD, HIP_R_16F, rows, cols, rows), "layout D");
        }

        ~Fp16Gemm()
        {
            if(layoutA)
                hipblasLtMatrixLayoutDestroy(layoutA);
            if(layoutB)
                hipblasLtMatrixLayoutDestroy(layoutB);
            if(layoutC)
                hipblasLtMatrixLayoutDestroy(layoutC);
            if(layoutD)
                hipblasLtMatrixLayoutDestroy(layoutD);
        }

        Fp16Gemm(const Fp16Gemm&)            = delete;
        Fp16Gemm& operator=(const Fp16Gemm&) = delete;

        // Memset D, run hipblasLtMatmul, and check D against the host reference.
        void matmul(hipblasLtHandle_t           handle,
                    hipblasLtMatmulDesc_t       desc,
                    hipStream_t                 stream,
                    const hipblasLtMatmulAlgo_t* algo,
                    void*                       workspace,
                    size_t                      workspaceSize,
                    float                       alpha,
                    float                       beta,
                    const std::string&          label) const
        {
            checkHip(hipMemset(D.pointer, 0xff, static_cast<size_t>(rows) * cols * sizeof(__half)),
                     "hipMemset D");
            checkBlas(hipblasLtMatmul(handle,
                                      desc,
                                      &alpha,
                                      A.pointer,
                                      layoutA,
                                      B.pointer,
                                      layoutB,
                                      &beta,
                                      C.pointer,
                                      layoutC,
                                      D.pointer,
                                      layoutD,
                                      algo,
                                      workspace,
                                      workspaceSize,
                                      stream),
                      "hipblasLtMatmul");
            checkHip(hipStreamSynchronize(stream), "hipStreamSynchronize");
            checkFp16(label, rows, cols, k, hostA, hostB, hostC, D.pointer, alpha, beta);
        }
    };

    // Handle, descriptor, stream, and the plain-pair replay backend for this device.
    struct ReplayFixture
    {
        int                                            device = -1;
        hipDeviceProp_t                                properties{};
        std::filesystem::path                          bundles;
        hipblasLtHandle_t                              handle{};
        hipblasLtMatmulDesc_t                          desc{};
        hipStream_t                                    stream{};
        hipblaslt_ext::experimental::jit::Backend      backend{};

        explicit ReplayFixture(const std::filesystem::path& root)
        {
            checkHip(hipGetDevice(&device), "hipGetDevice");
            checkHip(hipGetDeviceProperties(&properties, device), "hipGetDeviceProperties");
            bundles = deviceBundles(root, properties.gcnArchName);
            checkBlas(hipblasLtCreate(&handle), "hipblasLtCreate");
            checkBlas(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
                      "hipblasLtMatmulDescCreate");
            checkHip(hipStreamCreate(&stream), "hipStreamCreate");
            hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
            const auto status = hipblaslt_ext::experimental::jit::replay::createBackend(
                {{(bundles / "plain-pair").u8string()}}, backend, diagnostics);
            require(status == HIPBLAS_STATUS_SUCCESS,
                    std::string("replay::createBackend: status ") + std::to_string(status)
                        + diagnostics.message);
        }

        ~ReplayFixture()
        {
            static_cast<void>(hipStreamDestroy(stream));
            hipblasLtMatmulDescDestroy(desc);
            hipblasLtDestroy(handle);
        }

        ReplayFixture(const ReplayFixture&)            = delete;
        ReplayFixture& operator=(const ReplayFixture&) = delete;
    };
}
