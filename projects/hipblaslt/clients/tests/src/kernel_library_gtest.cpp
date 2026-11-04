// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Kernel libraries are loaded once per process for each device architecture,
// and the order in which devices first use hipBLASLt must not matter. Each case
// runs in a newly started copy of this binary so that it starts with no
// library loaded.

#include "scoped_env.hpp"

#include <gtest/gtest.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext-op.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace
{
    using test_env::ScopedEnv;

    constexpr int  kSize           = 64;
    constexpr char kReportDirEnv[] = "HIPBLASLT_TEST_KERNEL_LIBRARY_DIR";

    int processId()
    {
#ifdef _WIN32
        return _getpid();
#else
        return getpid();
#endif
    }

    int deviceCount()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess)
            return 0;
        return count;
    }

    struct DeviceResult
    {
        hipblasStatus_t gemmListStatus  = HIPBLAS_STATUS_NOT_INITIALIZED;
        size_t          gemmAlgoCount   = 0;
        bool            gemmOk          = false;
        hipblasStatus_t amaxStatus      = HIPBLAS_STATUS_NOT_INITIALIZED;
        bool            amaxOk          = false;
        hipblasStatus_t transformStatus = HIPBLAS_STATUS_NOT_INITIALIZED;
        bool            transformOk     = false;

        // A device either has no kernels of a kind, or computes correct results.
        // A library without solutions for the GEMM type (for example one built
        // with a logic filter) lists no algorithms; comparing against the
        // device used alone still catches a device that loses them.
        bool valid() const
        {
            const bool gemm
                = gemmListStatus == HIPBLAS_STATUS_INVALID_VALUE
                      ? gemmAlgoCount == 0
                      : gemmListStatus == HIPBLAS_STATUS_SUCCESS && (gemmAlgoCount == 0 || gemmOk);
            const bool amax = amaxStatus == HIPBLAS_STATUS_ARCH_MISMATCH
                              || (amaxStatus == HIPBLAS_STATUS_SUCCESS && amaxOk);
            const bool transform = transformStatus == HIPBLAS_STATUS_ARCH_MISMATCH
                                   || (transformStatus == HIPBLAS_STATUS_SUCCESS && transformOk);
            return gemm && amax && transform;
        }

        std::string str() const
        {
            std::ostringstream out;
            out << gemmListStatus << ' ' << gemmAlgoCount << ' ' << gemmOk << ' ' << amaxStatus
                << ' ' << amaxOk << ' ' << transformStatus << ' ' << transformOk;
            return out.str();
        }

        // Whether a line written by str() records a GEMM that ran correctly.
        static bool ranGemm(const std::string& line)
        {
            std::istringstream in(line);
            int                listStatus = 0;
            size_t             algoCount  = 0;
            bool               ok         = false;
            return bool(in >> listStatus >> algoCount >> ok) && ok;
        }
    };

    // Lists fp16 GEMM algorithms and runs the first supported one. A and B are
    // all ones, so every element of C is kSize, exactly representable in fp16.
    void runGemm(hipblasLtHandle_t handle, DeviceResult& result)
    {
        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        result.gemmListStatus = hipblaslt_ext::getAllAlgos(handle,
                                                           hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                           HIPBLAS_OP_N,
                                                           HIPBLAS_OP_N,
                                                           HIP_R_16F,
                                                           HIP_R_16F,
                                                           HIP_R_16F,
                                                           HIP_R_16F,
                                                           HIPBLAS_COMPUTE_32F,
                                                           algos);
        result.gemmAlgoCount  = algos.size();
        if(result.gemmListStatus != HIPBLAS_STATUS_SUCCESS || algos.empty())
            return;

        const size_t        elements = kSize * kSize;
        std::vector<__half> ones(elements, __float2half(1.0f));
        std::vector<__half> c(elements);
        void *              dA = nullptr, *dB = nullptr, *dC = nullptr, *workspace = nullptr;
        const size_t        workspaceSize = 32 << 20;
        // A failed allocation or copy leaves gemmOk false, which makes a device that
        // lists algorithms invalid
        const bool ready
            = hipMalloc(&dA, elements * sizeof(__half)) == hipSuccess
              && hipMalloc(&dB, elements * sizeof(__half)) == hipSuccess
              && hipMalloc(&dC, elements * sizeof(__half)) == hipSuccess
              && hipMalloc(&workspace, workspaceSize) == hipSuccess
              && hipMemcpy(dA, ones.data(), elements * sizeof(__half), hipMemcpyHostToDevice)
                     == hipSuccess
              && hipMemcpy(dB, ones.data(), elements * sizeof(__half), hipMemcpyHostToDevice)
                     == hipSuccess;

        hipblasLtMatmulDesc_t   desc;
        hipblasLtMatrixLayout_t layout;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtMatrixLayoutCreate(&layout, HIP_R_16F, kSize, kSize, kSize);
        const float alpha = 1.0f, beta = 0.0f;

        for(auto& algo : ready ? algos : decltype(algos){})
        {
            size_t required = 0;
            if(hipblaslt_ext::matmulIsAlgoSupported(
                   handle, desc, &alpha, layout, layout, &beta, layout, layout, algo.algo, required)
                   != HIPBLAS_STATUS_SUCCESS
               || required > workspaceSize)
                continue;
            result.gemmOk = hipblasLtMatmul(handle,
                                            desc,
                                            &alpha,
                                            dA,
                                            layout,
                                            dB,
                                            layout,
                                            &beta,
                                            dC,
                                            layout,
                                            dC,
                                            layout,
                                            &algo.algo,
                                            workspace,
                                            workspaceSize,
                                            nullptr)
                                == HIPBLAS_STATUS_SUCCESS
                            && hipDeviceSynchronize() == hipSuccess;
            break;
        }
        if(result.gemmOk)
        {
            result.gemmOk
                = hipMemcpy(c.data(), dC, elements * sizeof(__half), hipMemcpyDeviceToHost)
                  == hipSuccess;
            for(const auto& value : c)
                result.gemmOk = result.gemmOk && __half2float(value) == float(kSize);
        }

        hipblasLtMatrixLayoutDestroy(layout);
        hipblasLtMatmulDescDestroy(desc);
        static_cast<void>(hipFree(dA));
        static_cast<void>(hipFree(dB));
        static_cast<void>(hipFree(dC));
        static_cast<void>(hipFree(workspace));
    }

    // AMax of a matrix whose largest absolute value is 7.5, and a transform
    // computing 2 * A.
    void runExtOpAndTransform(hipblasLtHandle_t handle, DeviceResult& result)
    {
        const uint32_t     rows = 64, cols = 32;
        std::vector<float> host(rows * cols);
        for(size_t i = 0; i < host.size(); ++i)
            host[i] = 0.25f * (i % 29);
        host[17] = -7.5f;

        float *input = nullptr, *output = nullptr;
        if(hipMalloc(&input, host.size() * sizeof(float)) != hipSuccess
           || hipMalloc(&output, host.size() * sizeof(float)) != hipSuccess
           || hipMemcpy(input, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice)
                  != hipSuccess)
        {
            static_cast<void>(hipFree(input));
            static_cast<void>(hipFree(output));
            return;
        }

        result.amaxStatus
            = hipblasltExtAMax(HIP_R_32F, HIP_R_32F, output, input, rows, cols, nullptr);
        if(result.amaxStatus == HIPBLAS_STATUS_SUCCESS && hipDeviceSynchronize() == hipSuccess)
        {
            float amax = 0;
            result.amaxOk
                = hipMemcpy(&amax, output, sizeof(float), hipMemcpyDeviceToHost) == hipSuccess
                  && amax == 7.5f;
        }

        hipblasLtMatrixTransformDesc_t desc;
        hipblasLtMatrixLayout_t        layout;
        hipblasLtMatrixTransformDescCreate(&desc, HIP_R_32F);
        hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, rows, cols, rows);
        const float alpha = 2.0f, beta = 0.0f;
        result.transformStatus = hipblasLtMatrixTransform(
            handle, desc, &alpha, input, layout, &beta, input, layout, output, layout, nullptr);
        if(result.transformStatus == HIPBLAS_STATUS_SUCCESS && hipDeviceSynchronize() == hipSuccess)
        {
            std::vector<float> out(host.size());
            result.transformOk
                = hipMemcpy(out.data(), output, out.size() * sizeof(float), hipMemcpyDeviceToHost)
                  == hipSuccess;
            for(size_t i = 0; i < host.size(); ++i)
                result.transformOk = result.transformOk && out[i] == 2.0f * host[i];
        }

        hipblasLtMatrixLayoutDestroy(layout);
        hipblasLtMatrixTransformDescDestroy(desc);
        static_cast<void>(hipFree(input));
        static_cast<void>(hipFree(output));
    }

    // An exception escaping a hipBLASLt call leaves the remaining results unset,
    // so the report still shows which device failed.
    DeviceResult runOnDevice(int device)
    {
        DeviceResult      result;
        hipblasLtHandle_t handle = nullptr;
        if(hipSetDevice(device) != hipSuccess || hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            return result;
        try
        {
            runGemm(handle, result);
            runExtOpAndTransform(handle, result);
        }
        catch(...)
        {
        }
        hipblasLtDestroy(handle);
        return result;
    }

    // Writes one line per device in device order and exits with 0 if every
    // device in ran gave valid results.
    void exitWithReport(const std::vector<DeviceResult>& results,
                        const std::vector<int>&          ran,
                        const std::string&               name)
    {
        std::ofstream report(std::filesystem::path(std::getenv(kReportDirEnv)) / name);
        for(const auto& result : results)
            report << result.str() << '\n';
        report.close();
        if(!report)
            std::exit(1);
        for(int device : ran)
            if(!results[device].valid())
                std::exit(2);
        std::exit(0);
    }

    void exitAfterDevices(const std::vector<int>& order, const std::string& name)
    {
        std::vector<DeviceResult> results(deviceCount());
        for(int device : order)
            results[device] = runOnDevice(device);
        exitWithReport(results, order, name);
    }

    void exitAfterDevicesConcurrently(const std::string& name)
    {
        std::vector<DeviceResult> results(deviceCount());
        std::vector<int>          all;
        std::vector<std::thread>  threads;
        for(int device = 0; device < deviceCount(); ++device)
        {
            all.push_back(device);
            threads.emplace_back([&results, device] { results[device] = runOnDevice(device); });
        }
        for(auto& thread : threads)
            thread.join();
        exitWithReport(results, all, name);
    }

    // Removes a directory at the end of the scope; empty for none.
    struct ScopedDirectory
    {
        std::filesystem::path path;
        ~ScopedDirectory()
        {
            if(!path.empty())
                std::filesystem::remove_all(path);
        }
    };

    std::vector<std::string> readLines(const std::filesystem::path& path)
    {
        std::ifstream            file(path);
        std::vector<std::string> lines;
        for(std::string line; std::getline(file, line);)
            lines.push_back(line);
        return lines;
    }

    // Each device must give the same results however the devices of the process
    // are used: as when it is the only device used, in forward and reverse
    // device order, and from one thread per device.
    TEST(multi_gpu_KernelLibrary, ResultsDoNotDependOnOtherDevices)
    {
        const int count = deviceCount();
        if(count < 2)
            GTEST_SKIP() << "needs at least two devices";
        GTEST_FLAG_SET(death_test_style, "threadsafe");

        // Children are new processes that rerun this test up to their
        // EXPECT_EXIT; they write to the parent's directory, which they find in
        // the environment.
        const char*              inherited = std::getenv(kReportDirEnv);
        const auto               dir       = inherited
                                                 ? std::filesystem::path(inherited)
                                                 : std::filesystem::path(::testing::TempDir())
                                   / ("hipblaslt_kernel_library_" + std::to_string(processId()));
        std::optional<ScopedEnv> reportDir;
        ScopedDirectory          cleanup;
        if(!inherited)
        {
            std::filesystem::create_directories(dir);
            cleanup.path = dir;
            reportDir.emplace(kReportDirEnv, dir.string().c_str());
        }

        std::vector<int>         forward, reverse;
        std::vector<std::string> expected;
        bool                     anyGemm = false;
        for(int device = 0; device < count; ++device)
        {
            forward.push_back(device);
            reverse.insert(reverse.begin(), device);
            const auto alone = "alone_" + std::to_string(device);
            EXPECT_EXIT(exitAfterDevices({device}, alone), ::testing::ExitedWithCode(0), "");
            const auto lines = readLines(dir / alone);
            ASSERT_EQ(lines.size(), size_t(count));
            expected.push_back(lines[device]);
            anyGemm = anyGemm || DeviceResult::ranGemm(lines[device]);
        }
        // With no device running a GEMM on its own, every order would trivially
        // give the same empty results.
        if(!anyGemm)
            GTEST_SKIP() << "no device runs a GEMM with the installed kernel libraries";

        EXPECT_EXIT(exitAfterDevices(forward, "forward"), ::testing::ExitedWithCode(0), "");
        EXPECT_EXIT(exitAfterDevices(reverse, "reverse"), ::testing::ExitedWithCode(0), "");
        EXPECT_EXIT(exitAfterDevicesConcurrently("concurrent"), ::testing::ExitedWithCode(0), "");
        for(const char* run : {"forward", "reverse", "concurrent"})
            EXPECT_EQ(readLines(dir / run), expected) << run;
    }

    // Calls MatrixTransform on the handle's device with transformDevice current and
    // returns its status, or nullopt if the test setup failed. On success, ok
    // reports whether C holds 2 * A; current is the device current afterwards.
    std::optional<hipblasStatus_t>
        transformWithDeviceCurrent(int handleDevice, int transformDevice, bool& ok, int& current)
    {
        const uint32_t     rows = 64, cols = 32;
        std::vector<float> host(rows * cols);
        for(size_t i = 0; i < host.size(); ++i)
            host[i] = 0.5f * (i % 31);

        hipblasLtHandle_t handle = nullptr;
        hipStream_t       stream = nullptr;
        float *           input = nullptr, *output = nullptr;
        bool              ready
            = hipSetDevice(handleDevice) == hipSuccess
              && hipblasLtCreate(&handle) == HIPBLAS_STATUS_SUCCESS
              && hipStreamCreate(&stream) == hipSuccess
              && hipMalloc(&input, host.size() * sizeof(float)) == hipSuccess
              && hipMalloc(&output, host.size() * sizeof(float)) == hipSuccess
              && hipMemcpy(input, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice)
                     == hipSuccess;

        hipblasLtMatrixTransformDesc_t desc;
        hipblasLtMatrixLayout_t        layout;
        hipblasLtMatrixTransformDescCreate(&desc, HIP_R_32F);
        hipblasLtMatrixLayoutCreate(&layout, HIP_R_32F, rows, cols, rows);
        const float                    alpha = 2.0f, beta = 0.0f;
        std::optional<hipblasStatus_t> status;
        ok = false;
        if(ready && hipSetDevice(transformDevice) == hipSuccess)
        {
            status = hipblasLtMatrixTransform(
                handle, desc, &alpha, input, layout, &beta, input, layout, output, layout, stream);
            if(hipGetDevice(&current) != hipSuccess)
                current = -1;
            if(*status == HIPBLAS_STATUS_SUCCESS && hipStreamSynchronize(stream) == hipSuccess)
            {
                std::vector<float> out(host.size());
                ok = hipMemcpy(
                         out.data(), output, out.size() * sizeof(float), hipMemcpyDeviceToHost)
                     == hipSuccess;
                for(size_t i = 0; i < host.size(); ++i)
                    ok = ok && out[i] == 2.0f * host[i];
            }
        }

        hipblasLtMatrixLayoutDestroy(layout);
        hipblasLtMatrixTransformDescDestroy(desc);
        static_cast<void>(hipSetDevice(handleDevice));
        static_cast<void>(hipFree(input));
        static_cast<void>(hipFree(output));
        if(stream)
            static_cast<void>(hipStreamDestroy(stream));
        if(handle)
            hipblasLtDestroy(handle);
        return status;
    }

    // MatrixTransform runs on the handle's device whichever device is current,
    // with the same outcome as when the handle's device is current, and leaves
    // the caller's device current. A device without transform kernels takes
    // the error path after the device switch.
    TEST(multi_gpu_KernelLibrary, MatrixTransformRunsOnHandleDevice)
    {
        const int count = deviceCount();
        if(count < 2)
            GTEST_SKIP() << "needs at least two devices";
        int original = 0;
        ASSERT_EQ(hipGetDevice(&original), hipSuccess);

        for(int a = 0; a < count; ++a)
        {
            bool       aloneOk = false;
            int        current = -1;
            const auto alone   = transformWithDeviceCurrent(a, a, aloneOk, current);
            ASSERT_TRUE(alone) << "setup failed on device " << a;
            ASSERT_TRUE(*alone == HIPBLAS_STATUS_ARCH_MISMATCH
                        || (*alone == HIPBLAS_STATUS_SUCCESS && aloneOk))
                << "device " << a << " status " << *alone;

            for(int b = 0; b < count; ++b)
            {
                if(b == a)
                    continue;
                bool       ok       = false;
                const auto switched = transformWithDeviceCurrent(a, b, ok, current);
                ASSERT_TRUE(switched) << "setup failed on devices " << a << ", " << b;
                EXPECT_EQ(*switched, *alone) << "handle on " << a << ", " << b << " current";
                EXPECT_EQ(ok, aloneOk) << "handle on " << a << ", " << b << " current";
                EXPECT_EQ(current, b) << "handle on " << a << ", " << b << " current";
            }
        }
        EXPECT_EQ(hipSetDevice(original), hipSuccess);
    }

    // With no Tensile library loaded, the GEMM APIs return errors.
    void exitAfterGemmWithoutLibrary()
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            std::exit(1);

        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        if(hipblaslt_ext::getAllAlgos(handle,
                                      hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                      HIPBLAS_OP_N,
                                      HIPBLAS_OP_N,
                                      HIP_R_16F,
                                      HIP_R_16F,
                                      HIP_R_16F,
                                      HIP_R_16F,
                                      HIPBLAS_COMPUTE_32F,
                                      algos)
               != HIPBLAS_STATUS_INVALID_VALUE
           || !algos.empty())
            std::exit(2);

        hipblasLtMatmulDesc_t       desc;
        hipblasLtMatrixLayout_t     layout;
        hipblasLtMatmulPreference_t preference;
        hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        hipblasLtMatrixLayoutCreate(&layout, HIP_R_16F, kSize, kSize, kSize);
        hipblasLtMatmulPreferenceCreate(&preference);
        hipblasLtMatmulHeuristicResult_t heuristic{};
        int                              returned = 0;
        const auto                       status   = hipblasLtMatmulAlgoGetHeuristic(
            handle, desc, layout, layout, layout, layout, preference, 1, &heuristic, &returned);
        if(status != HIPBLAS_STATUS_INVALID_VALUE || returned != 0)
            std::exit(3);

        hipblasLtMatmulPreferenceDestroy(preference);
        hipblasLtMatrixLayoutDestroy(layout);
        hipblasLtMatmulDescDestroy(desc);
        hipblasLtDestroy(handle);
        std::exit(0);
    }

    // With no ExtOp library loaded, the ExtOp APIs return ARCH_MISMATCH.
    void exitAfterExtOpsWithoutLibrary()
    {
        const uint32_t m = 4, n = 64;
        float *        input = nullptr, *output = nullptr, *mean = nullptr, *invvar = nullptr;
        if(hipMalloc(&input, m * n * sizeof(float)) != hipSuccess
           || hipMalloc(&output, m * n * sizeof(float)) != hipSuccess
           || hipMalloc(&mean, m * sizeof(float)) != hipSuccess
           || hipMalloc(&invvar, m * sizeof(float)) != hipSuccess)
            std::exit(4);

        int failures = 0;
        failures += hipblasltExtSoftmax(HIP_R_32F, m, n, 1, output, input, nullptr)
                    != HIPBLAS_STATUS_ARCH_MISMATCH;
        failures
            += hipblasltExtLayerNorm(
                   HIP_R_32F, output, mean, invvar, input, m, n, 1e-5f, nullptr, nullptr, nullptr)
               != HIPBLAS_STATUS_ARCH_MISMATCH;
        failures += hipblasltExtAMax(HIP_R_32F, HIP_R_32F, output, input, m, n, nullptr)
                    != HIPBLAS_STATUS_ARCH_MISMATCH;
        std::exit(failures == 0 ? 0 : 1);
    }

    // With HIPBLASLT_TENSILE_LIBPATH set, the current device finds a Tensile
    // library. A library built with a logic filter may hold no fp16 GEMMs, so
    // listing succeeds with no algorithms; without a library it fails.
    void exitIfTensileLibraryLoads()
    {
        hipblasLtHandle_t handle = nullptr;
        if(hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS)
            std::exit(1);
        std::vector<hipblasLtMatmulHeuristicResult_t> algos;
        const auto                                    status = hipblaslt_ext::getAllAlgos(handle,
                                                       hipblaslt_ext::GemmType::HIPBLASLT_GEMM,
                                                       HIPBLAS_OP_N,
                                                       HIPBLAS_OP_N,
                                                       HIP_R_16F,
                                                       HIP_R_16F,
                                                       HIP_R_16F,
                                                       HIP_R_16F,
                                                       HIPBLAS_COMPUTE_32F,
                                                       algos);
        hipblasLtDestroy(handle);
        std::exit(status == HIPBLAS_STATUS_SUCCESS ? 0 : 2);
    }

    // Whether dir holds a Tensile library, lazy-loading or not, for processor.
    bool hasTensileLibrary(const std::filesystem::path& dir, const std::string& processor)
    {
        for(const char* prefix : {"TensileLibrary_lazy_", "TensileLibrary_"})
            for(const char* extension : {".dat", ".dat.zlib", ".yaml"})
                if(std::filesystem::exists(dir / (prefix + processor + extension)))
                    return true;
        return false;
    }

    TEST(KernelLibrary, smoke_MissingTensileLibraryReturnsErrors)
    {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ScopedEnv libPath("HIPBLASLT_TENSILE_LIBPATH", "/nonexistent/hipblaslt/library");
        EXPECT_EXIT(exitAfterGemmWithoutLibrary(), ::testing::ExitedWithCode(0), "");
    }

    // HIPBLASLT_TENSILE_LIBPATH may name an installed library directory, which
    // holds one sub-directory per architecture, instead of one architecture's.
    TEST(KernelLibrary, smoke_TensileLibPathAcceptsArchitectureSubdirectories)
    {
#ifdef _WIN32
        GTEST_SKIP() << "finding the installed library directory is not implemented on Windows";
#else
        // dlsym, unlike &hipblasLtCreate, which is this binary's PLT stub, gives
        // the definition in the library, so dladdr names the library file.
        Dl_info     info{};
        void* const entry = dlsym(RTLD_DEFAULT, "hipblasLtCreate");
        if(entry == nullptr || dladdr(entry, &info) == 0 || info.dli_fname == nullptr)
            GTEST_SKIP() << "cannot locate libhipblaslt";
        const auto root = std::filesystem::weakly_canonical(info.dli_fname).parent_path()
                          / "hipblaslt" / "library";

        int             device = 0;
        hipDeviceProp_t prop{};
        ASSERT_EQ(hipGetDevice(&device), hipSuccess);
        ASSERT_EQ(hipGetDeviceProperties(&prop, device), hipSuccess);
        const std::string arch      = prop.gcnArchName;
        const std::string processor = arch.substr(0, arch.find(':'));
        if(!hasTensileLibrary(root / processor, processor) || hasTensileLibrary(root, processor))
            GTEST_SKIP() << root << " has no " << processor
                         << " sub-directory with a Tensile library";

        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ScopedEnv libPath("HIPBLASLT_TENSILE_LIBPATH", root.string().c_str());
        EXPECT_EXIT(exitIfTensileLibraryLoads(), ::testing::ExitedWithCode(0), "");
#endif
    }

    TEST(KernelLibrary, smoke_MissingExtOpLibraryReturnsArchMismatch)
    {
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ScopedEnv libPath("HIPBLASLT_EXT_OP_LIBRARY_PATH",
                          "/nonexistent/hipblasltExtOpLibrary.dat");
        EXPECT_EXIT(exitAfterExtOpsWithoutLibrary(), ::testing::ExitedWithCode(0), "");
    }
} // namespace
