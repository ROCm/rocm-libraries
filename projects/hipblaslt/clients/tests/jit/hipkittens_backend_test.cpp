// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-hipkittens.hpp"
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/Contractions.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt-ext.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace jit = hipblaslt_ext::experimental::jit;
namespace abi = hipblaslt_ext::experimental::jit::detail;
namespace hk  = hipblaslt_ext::experimental::jit::hipkittens;
namespace hj  = hipblaslt_jit;
namespace co  = hipblaslt_jit::code_object;
namespace fs  = std::filesystem;

namespace
{
    void require(bool condition, const std::string& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }
    void hip(hipError_t status, const char* expression)
    {
        require(status == hipSuccess, std::string(expression) + ": " + hipGetErrorString(status));
    }
    void blas(hipblasStatus_t status, const char* expression)
    {
        require(status == HIPBLAS_STATUS_SUCCESS,
                std::string(expression) + ": status " + std::to_string(status));
    }
#define HIP(expression) hip((expression), #expression)
#define BLAS(expression) blas((expression), #expression)

    // A BF16 and FP16 NaN that no GEMM of finite inputs writes.
    constexpr uint16_t canary = 0x7fc1;

    uint16_t toBf16(float value)
    {
        uint32_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        bits += 0x7fff + ((bits >> 16) & 1);
        return static_cast<uint16_t>(bits >> 16);
    }
    float fromBf16(uint16_t value)
    {
        const uint32_t bits = uint32_t(value) << 16;
        float          result;
        std::memcpy(&result, &bits, sizeof(result));
        return result;
    }
    // The bits of a BF16 or FP16 element.
    uint16_t toBits(hipDataType type, float value)
    {
        if(type == HIP_R_16BF)
            return toBf16(value);
        const _Float16 half = static_cast<_Float16>(value);
        uint16_t       bits;
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    }
    float fromBits(hipDataType type, uint16_t bits)
    {
        if(type == HIP_R_16BF)
            return fromBf16(bits);
        _Float16 half;
        std::memcpy(&half, &bits, sizeof(half));
        return static_cast<float>(half);
    }

    // The variant for BF16 or FP16 A, B, C and D.
    const hk::detail::Variant& variant(hipDataType type = HIP_R_16BF)
    {
        const auto& variants = hk::detail::resources().variants;
        require(variants.size() == 2, "Expected two HipKittens variants");
        const std::string prefix = type == HIP_R_16BF ? "HK_gemm_bf16_" : "HK_gemm_f16_";
        for(const auto& v : variants)
            if(std::string_view(v.kernelName).rfind(prefix, 0) == 0)
                return v;
        throw std::runtime_error("No HipKittens variant for " + prefix);
    }

    template <class T>
    struct DeviceBuffer
    {
        T* pointer{};
        explicit DeviceBuffer(size_t count)
        {
            HIP(hipMalloc(reinterpret_cast<void**>(&pointer), std::max<size_t>(count, 1) * sizeof(T)));
        }
        ~DeviceBuffer()
        {
            static_cast<void>(hipFree(pointer));
        }
        DeviceBuffer(const DeviceBuffer&)            = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    };

    // One hipBLASLt GEMM; the defaults are the HipKittens variant's domain.
    struct Config
    {
        hipblasOperation_t     opA = HIPBLAS_OP_T, opB = HIPBLAS_OP_N;
        hipDataType            typeAB = HIP_R_16BF, typeCD = HIP_R_16BF;
        int64_t                m = 1024, n = 1024, k = 1024;
        int64_t                lda = 0, ldb = 0, ldc = 0, ldd = 0; // 0 is packed
        int32_t                batch = 1;
        int64_t                gap   = 0; // elements between batches
        bool                   pointerArray = false;
        float                  alpha = 1, beta = 0;
        bool                   cIsD = false; // C aliases D
        hipblasLtPointerMode_t pointerMode = HIPBLASLT_POINTER_MODE_HOST;
        hipblasLtEpilogue_t    epilogue    = HIPBLASLT_EPILOGUE_DEFAULT;

        int64_t rowsA() const
        {
            return opA == HIPBLAS_OP_N ? m : k;
        }
        int64_t rowsB() const
        {
            return opB == HIPBLAS_OP_N ? k : n;
        }
        int64_t leadA() const
        {
            return lda ? lda : rowsA();
        }
        int64_t leadB() const
        {
            return ldb ? ldb : rowsB();
        }
        int64_t leadC() const
        {
            return ldc ? ldc : m;
        }
        int64_t leadD() const
        {
            return ldd ? ldd : m;
        }
        // Batch strides.
        int64_t strideA() const
        {
            return leadA() * (opA == HIPBLAS_OP_N ? k : m) + gap;
        }
        int64_t strideB() const
        {
            return leadB() * (opB == HIPBLAS_OP_N ? n : k) + gap;
        }
        int64_t strideC() const
        {
            return leadC() * n + gap;
        }
        int64_t strideD() const
        {
            return leadD() * n + gap;
        }
        std::string name() const
        {
            std::ostringstream out;
            out << m << 'x' << n << 'x' << k;
            if(typeAB == HIP_R_16F)
                out << " fp16";
            if(lda || ldb || ldc || ldd)
                out << " ld " << leadA() << ' ' << leadB() << ' ' << leadC() << ' ' << leadD();
            if(batch != 1)
                out << " batch " << batch;
            if(gap)
                out << " gap " << gap;
            if(alpha != 1)
                out << " alpha " << alpha;
            if(beta)
                out << " beta " << beta << (cIsD ? " C=D" : "");
            return out.str();
        }
    };

    struct Descriptors
    {
        hipblasLtMatmulDesc_t   desc{};
        hipblasLtMatrixLayout_t la{}, lb{}, lc{}, ld{};

        Descriptors(const Config& c, const void* bias)
        {
            BLAS(hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
            BLAS(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSA, &c.opA, sizeof(c.opA)));
            BLAS(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_TRANSB, &c.opB, sizeof(c.opB)));
            BLAS(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE, &c.pointerMode, sizeof(c.pointerMode)));
            BLAS(hipblasLtMatmulDescSetAttribute(
                desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &c.epilogue, sizeof(c.epilogue)));
            if(c.epilogue == HIPBLASLT_EPILOGUE_BIAS)
                BLAS(hipblasLtMatmulDescSetAttribute(
                    desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
            const auto layout = [&](hipblasLtMatrixLayout_t& l,
                                    hipDataType              type,
                                    int64_t                  rows,
                                    int64_t                  cols,
                                    int64_t                  lead,
                                    int64_t                  stride) {
                BLAS(hipblasLtMatrixLayoutCreate(&l, type, rows, cols, lead));
                BLAS(hipblasLtMatrixLayoutSetAttribute(
                    l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &c.batch, sizeof(c.batch)));
                BLAS(hipblasLtMatrixLayoutSetAttribute(
                    l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
                if(c.pointerArray)
                {
                    const hipblasLtBatchMode_t mode = HIPBLASLT_BATCH_MODE_POINTER_ARRAY;
                    BLAS(hipblasLtMatrixLayoutSetAttribute(
                        l, HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, &mode, sizeof(mode)));
                }
            };
            layout(la, c.typeAB, c.rowsA(), c.opA == HIPBLAS_OP_N ? c.k : c.m, c.leadA(),
                   c.strideA());
            layout(lb, c.typeAB, c.rowsB(), c.opB == HIPBLAS_OP_N ? c.n : c.k, c.leadB(),
                   c.strideB());
            layout(lc, c.typeCD, c.m, c.n, c.leadC(), c.strideC());
            layout(ld, c.typeCD, c.m, c.n, c.leadD(), c.strideD());
        }
        ~Descriptors()
        {
            hipblasLtMatrixLayoutDestroy(la);
            hipblasLtMatrixLayoutDestroy(lb);
            hipblasLtMatrixLayoutDestroy(lc);
            hipblasLtMatrixLayoutDestroy(ld);
            hipblasLtMatmulDescDestroy(desc);
        }
        Descriptors(const Descriptors&)            = delete;
        Descriptors& operator=(const Descriptors&) = delete;
    };

    struct Handle
    {
        hipblasLtHandle_t handle{};
        Handle()
        {
            BLAS(hipblasLtCreate(&handle));
        }
        ~Handle()
        {
            hipblasLtDestroy(handle);
        }
    };

    jit::Backend backend(const hk::Options& options = {})
    {
        jit::Backend     result;
        jit::Diagnostics diagnostics;
        const auto       status = hk::createBackend(options, result, diagnostics);
        require(status == HIPBLAS_STATUS_SUCCESS, "HipKittens backend: " + diagnostics.message);
        require(diagnostics.backend == "HipKittens", "Diagnostics named " + diagnostics.backend);
        return result;
    }

    // A gfx950 target for checks that need no gfx950 device.
    hj::DeviceTarget gfx950()
    {
        hj::DeviceTarget target;
        target.device   = 0;
        target.targetId = "gfx950:sramecc+:xnack-";
        target.isa      = "gfx950";
        return target;
    }

    // Requests whose buffers are never touched: generation reads only descriptors.
    struct Requests
    {
        Handle               h;
        DeviceBuffer<char>   buffer{256};
        float*               deviceAlpha{};
        Requests()
        {
            HIP(hipHostMalloc(reinterpret_cast<void**>(&deviceAlpha), 4096 * sizeof(float)));
            std::fill(deviceAlpha, deviceAlpha + 4096, 1.0f);
        }
        ~Requests()
        {
            static_cast<void>(hipHostFree(deviceAlpha));
        }
        jit::Request make(const Config& c)
        {
            Descriptors      d(c, buffer.pointer);
            float            beta  = c.beta;
            const void*      alpha = c.pointerMode == HIPBLASLT_POINTER_MODE_HOST
                                         ? static_cast<const void*>(&c.alpha)
                                         : deviceAlpha;
            jit::Request     request;
            jit::Diagnostics diagnostics;
            const auto       status = jit::makeGemmRequest(h.handle,
                                                     d.desc,
                                                     alpha,
                                                     buffer.pointer,
                                                     d.la,
                                                     buffer.pointer,
                                                     d.lb,
                                                     &beta,
                                                     buffer.pointer,
                                                     d.lc,
                                                     buffer.pointer,
                                                     d.ld,
                                                     request,
                                                     diagnostics);
            require(status == HIPBLAS_STATUS_SUCCESS,
                    c.name() + ": makeGemmRequest: " + diagnostics.message);
            return request;
        }
    };

    hj::Status generate(const jit::Backend&                    provider,
                        const jit::Request&                    request,
                        const hj::DeviceTarget&                target,
                        std::vector<hj::GeneratedSolution>&    solutions,
                        const std::vector<std::string>&        excluded = {})
    {
        hj::GenerationRequest generation{*abi::RequestAccess::get(request), target};
        generation.excludeKernels = excluded;
        return abi::BackendAccess::get(provider)->components().backend->generate(generation,
                                                                                solutions);
    }

    // The header directory next to the loaded libhipblaslt.
    fs::path installedHeaders()
    {
        Dl_info info{};
        require(dladdr(reinterpret_cast<void*>(&hipblasLtCreate), &info) && info.dli_fname,
                "Cannot locate libhipblaslt");
        const std::string manifest(hk::detail::resources().manifest);
        const auto        at = manifest.find("\"commit\": \"");
        require(at != std::string::npos, "The compiled-in manifest has no commit");
        return fs::canonical(info.dli_fname).parent_path() / "hipblaslt" / "hipkittens"
               / manifest.substr(at + 11, 12);
    }

    void expectUnavailable(const hk::Options& options, const std::string& cause, const char* label)
    {
        jit::Backend     result;
        jit::Diagnostics diagnostics;
        const auto       status = hk::createBackend(options, result, diagnostics);
        const std::string prefix = "JIT backend HipKittens not available: ";
        require(status != HIPBLAS_STATUS_SUCCESS && !abi::BackendAccess::get(result)
                    && diagnostics.message.rfind(prefix, 0) == 0
                    && diagnostics.message.find(cause) != std::string::npos,
                std::string(label) + ": expected \"" + cause + "\", got \"" + diagnostics.message
                    + "\"");
        std::cout << "PASS " << label << ": " << diagnostics.message << '\n';
    }

    void headers(const fs::path& scratch)
    {
        const auto staged = installedHeaders();
        std::cout << "headers: " << staged.u8string() << '\n';
        backend();
        std::cout << "PASS default discovery\n";

        const auto copy = [&](const char* name) {
            const auto target = scratch / name;
            fs::remove_all(target);
            fs::copy(staged, target, fs::copy_options::recursive);
            return target;
        };
        const auto good = copy("good");
        backend({good.u8string()});
        std::cout << "PASS Options.headers\n";
        const auto empty = scratch / "empty";
        fs::create_directories(empty);
        require(setenv("HIPBLASLT_JIT_HIPKITTENS_PATH", good.c_str(), 1) == 0, "setenv");
        backend();
        std::cout << "PASS HIPBLASLT_JIT_HIPKITTENS_PATH\n";
        require(setenv("HIPBLASLT_JIT_HIPKITTENS_PATH", empty.c_str(), 1) == 0, "setenv");
        expectUnavailable({}, "headers not found at " + empty.u8string(), "empty directory");
        require(unsetenv("HIPBLASLT_JIT_HIPKITTENS_PATH") == 0, "unsetenv");

        const auto header = fs::u8path(hk::detail::resources().headers.front().path);
        const auto missing = copy("missing");
        fs::remove(missing / header);
        expectUnavailable({missing.u8string()}, (missing / header).u8string() + " is missing",
                          "missing file");
        const auto edited = copy("edited");
        std::ofstream(edited / header, std::ios::app) << ' ';
        expectUnavailable({edited.u8string()}, "does not match manifest.json", "edited file");
        const auto other = copy("other-commit");
        {
            std::string manifest(hk::detail::resources().manifest);
            manifest.replace(manifest.find("\"commit\": \"") + 11, 1, "0");
            std::ofstream(other / "manifest.json", std::ios::trunc) << manifest;
        }
        expectUnavailable({other.u8string()}, "is not the manifest", "other commit");
        const auto linked = copy("symlink");
        const auto outside = scratch / "outside.cuh";
        fs::copy_file(staged / header, outside, fs::copy_options::overwrite_existing);
        fs::remove(linked / header);
        fs::create_symlink(outside, linked / header);
        expectUnavailable({linked.u8string()}, "leaves", "symbolic link leaving the directory");
    }

    void domain()
    {
        Requests   r;
        const auto provider = backend();
        const auto target   = gfx950();
        std::vector<hj::GeneratedSolution> solutions;

        auto status = generate(provider, r.make({}), target, solutions);
        require(status.ok() && solutions.size() == 1, "1024^3: " + status.message);
        const auto& solution = solutions.front();
        require(solution.kernelName == variant().kernelName, "Wrong kernel name");
        require(solution.hipFlags == variant().hipFlags, "Wrong HIP flags");
        require(solution.units.size() == 1
                    && solution.units[0].kind == hj::BuildUnit::Kind::Hip
                    && solution.units[0].role == hj::BuildUnit::Role::Main
                    && solution.units[0].includes.size() == hk::detail::resources().headers.size(),
                "Expected one HIP main unit with every header");
        std::cout << "PASS one solution for 1024^3\n";

        struct Case
        {
            const char* label;
            Config      config;
        };
        const auto with = [](auto change) {
            Config c;
            change(c);
            return c;
        };
        for(const auto& c : {Case{"beta 1", with([](Config& c) { c.beta = 1; })},
                             Case{"beta -0.5", with([](Config& c) { c.beta = -0.5f; })},
                             Case{"alpha 1.5", with([](Config& c) { c.alpha = 1.5f; })},
                             Case{"batch 2", with([](Config& c) { c.batch = 2; })},
                             Case{"batch 3 with gaps",
                                  with([](Config& c) { c.batch = 3, c.gap = 100; })},
                             Case{"ldA != K", with([](Config& c) { c.lda = c.k + 64; })},
                             Case{"ldB != K", with([](Config& c) { c.ldb = c.k + 64; })},
                             Case{"ldC != M", with([](Config& c) { c.ldc = c.m + 256; })},
                             Case{"ldC != M, beta 1",
                                  with([](Config& c) { c.ldc = c.m + 256, c.beta = 1; })},
                             Case{"ldD != M", with([](Config& c) { c.ldd = c.m + 256; })},
                             Case{"odd leading dimensions", with([](Config& c) {
                                      c.lda = c.k + 1, c.ldb = c.k + 3, c.ldc = c.ldd = c.m + 1;
                                  })},
                             Case{"fp16", with([](Config& c) { c.typeAB = c.typeCD = HIP_R_16F; })},
                             Case{"fp16, beta 1, batch 2, odd leading dimensions",
                                  with([](Config& c) {
                                      c.typeAB = c.typeCD = HIP_R_16F;
                                      c.beta = 1, c.batch = 2;
                                      c.lda = c.k + 1, c.ldc = c.ldd = c.m + 1;
                                  })}})
        {
            status = generate(provider, r.make(c.config), target, solutions);
            require(status.ok() && solutions.size() == 1,
                    std::string(c.label) + ": " + status.message);
            require(solutions[0].kernelName == variant(c.config.typeAB).kernelName
                        && solutions[0].hipFlags == variant(c.config.typeAB).hipFlags,
                    std::string(c.label) + ": wrong variant " + solutions[0].kernelName);
            std::cout << "PASS one solution for " << c.label << ": " << solutions[0].kernelName
                      << '\n';
        }
        const std::vector<Case> cases{
            {"NN", with([](Config& c) { c.opA = HIPBLAS_OP_N; })},
            {"NT", with([](Config& c) { c.opA = HIPBLAS_OP_N, c.opB = HIPBLAS_OP_T; })},
            {"fp32 out", with([](Config& c) { c.typeCD = HIP_R_32F; })},
            {"fp16 in, fp32 out",
             with([](Config& c) { c.typeAB = HIP_R_16F, c.typeCD = HIP_R_32F; })},
            {"fp16 NN",
             with([](Config& c) { c.typeAB = c.typeCD = HIP_R_16F, c.opA = HIPBLAS_OP_N; })},
            {"fp16 K = 192", with([](Config& c) { c.typeAB = c.typeCD = HIP_R_16F, c.k = 192; })},
            {"alpha 0 (K = 0 in hipBLASLt)", with([](Config& c) { c.alpha = 0; })},
            {"device alpha",
             with([](Config& c) { c.pointerMode = HIPBLASLT_POINTER_MODE_DEVICE; })},
            {"alpha vector",
             with([](Config& c) {
                 c.pointerMode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
             })},
            {"pointer-array batch",
             with([](Config& c) { c.batch = 2, c.pointerArray = true; })},
            {"M = 300", with([](Config& c) { c.m = 300; })},
            {"N = 384", with([](Config& c) { c.n = 384; })},
            {"K = 192", with([](Config& c) { c.k = 192; })},
            {"K = 64", with([](Config& c) { c.k = 64; })},
            {"bias", with([](Config& c) { c.epilogue = HIPBLASLT_EPILOGUE_BIAS; })},
            {"ReLU", with([](Config& c) { c.epilogue = HIPBLASLT_EPILOGUE_RELU; })},
            {"8 GiB D", with([](Config& c) { c.m = c.n = 65536, c.k = 128; })},
            {"4 GiB of D in batches",
             with([](Config& c) { c.m = c.n = 4096, c.k = 128, c.batch = 128; })},
            {"ldA * M spanning 4 GiB", with([](Config& c) { c.lda = int64_t(1) << 21; })},
        };
        for(const auto& c : cases)
        {
            status = generate(provider, r.make(c.config), target, solutions);
            require(status.code == hj::Status::Code::NotSupported && solutions.empty(),
                    std::string(c.label) + " was not rejected: " + status.message);
            std::cout << "PASS " << c.label << " not supported: " << status.message << '\n';
        }

        auto other = target;
        other.isa      = "gfx942";
        other.targetId = "gfx942:sramecc+:xnack-";
        status         = generate(provider, r.make({}), other, solutions);
        require(status.code == hj::Status::Code::TargetMismatch && solutions.empty(),
                "gfx942 was not a target mismatch: " + status.message);
        std::cout << "PASS gfx942 target mismatch\n";
        status = generate(
            provider, r.make({}), target, solutions, {std::string(variant().kernelName)});
        require(status.ok() && solutions.empty(), "An excluded kernel was generated");
        std::cout << "PASS excluded kernel\n";
    }

    void entry()
    {
        Requests                           r;
        std::vector<hj::GeneratedSolution> solutions;
        Config                             c;
        c.m = 512, c.n = 768, c.k = 384;
        c.lda = 392, c.ldb = 448, c.ldc = 513, c.ldd = 768;
        const auto status = generate(backend(), r.make(c), gfx950(), solutions);
        require(status.ok() && solutions.size() == 1, c.name() + ": " + status.message);
        using Problem = TensileLite::ContractionProblemGemm;
        const auto library
            = std::dynamic_pointer_cast<TensileLite::MasterSolutionLibrary<Problem>>(
                TensileLite::LoadLibraryData<Problem>(solutions[0].entry));
        require(library && library->solutions.size() == 1, "The entry does not load");
        const auto& solution = *library->solutions.begin()->second;
        require(!solution.customKernel.generated
                    && solution.customKernel.name == variant().kernelName
                    && solution.kernelName == variant().kernelName,
                "The entry is not the handwritten custom kernel");
        const auto all = std::dynamic_pointer_cast<TensileLite::Predicates::And<Problem>>(
            solution.problemPredicate);
        require(all != nullptr, "The problem predicate is not an And");
        std::set<std::string> types;
        for(const auto& term : all->value)
            types.insert(term->type());
        require(!types.count("BetaZero"), "The entry still requires beta 0");
        require(!types.count("AlphaValue"), "The entry still requires alpha 1");
        require(!types.count("BatchSizeEqual"), "The entry still requires one batch");
        for(const char* type : {"SizeGreaterThan",
                                "Free0SizeMultiple",
                                "Free1SizeMultiple",
                                "BoundSizeMultiple",
                                "TypesEqual",
                                "OperationIdentifierEqual"})
            require(types.count(type), std::string("The entry has no ") + type);
        for(const char* type : {"StrideAEqual", "StrideBEqual", "StrideCEqual", "StrideDEqual"})
            require(!types.count(type), std::string("The entry pins a stride with ") + type);
        namespace P = TensileLite::Predicates::Contraction;
        const auto find = [&](auto* tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            for(const auto& term : all->value)
                if(const auto found = std::dynamic_pointer_cast<T>(term))
                    return found;
            throw std::runtime_error("The entry has no " + T::Type());
        };
        constexpr size_t columns = 0xffffffff; // every column of a matrix
        const auto       load = find(static_cast<P::BufferLoadOffsetLimitCheck*>(nullptr))->value;
        require(load.depthUorMT0 == columns && load.depthUorMT1 == columns
                    && find(static_cast<P::BufferLoadOffsetLimitCheck_Beta*>(nullptr))->value
                           == columns
                    && find(static_cast<P::BufferStoreOffsetLimitCheck*>(nullptr))->value
                           == columns,
                "The entry's buffer limit checks do not span whole matrices");
        const auto k = find(static_cast<P::SizeGreaterThan*>(nullptr));
        require(k->index == 3 && k->value == 0, "The entry does not require K > 0");
        std::cout << "PASS entry: handwritten custom kernel, static predicates, whole-matrix "
                     "buffer limits and no stride pins\n";
    }

    void build()
    {
        Requests   r;
        const auto provider = backend();
        const auto target   = gfx950();
        for(const auto type : {HIP_R_16BF, HIP_R_16F})
        {
            Config c;
            c.typeAB = c.typeCD = type;
            const auto&                        v       = variant(type);
            const auto                         request = r.make(c);
            std::vector<hj::GeneratedSolution> solutions;
            auto status = generate(provider, request, target, solutions);
            require(status.ok() && solutions.size() == 1, c.name() + ": " + status.message);
            hj::GenerationRequest generation{*abi::RequestAccess::get(request), target};
            hj::BuiltSolution     built;
            status = abi::BackendAccess::get(provider)->components().builder->build(
                solutions[0], generation, built);
            require(status.ok(), "Build: " + status.message);
            const auto metadata
                = co::readMetadata(built.object.bytes.data(), built.object.bytes.size());
            require(metadata.ok(), "readMetadata: " + metadata.log);
            const auto& kernels = metadata.metadata.kernels;
            const auto  kernel  = std::find_if(kernels.begin(), kernels.end(), [&](const auto& k) {
                return k.name == v.kernelName;
            });
            require(kernel != kernels.end(), "The code object has no " + std::string(v.kernelName));
            const auto& expected = v.resources;
            std::cout << "built " << v.kernelName << ": kernarg " << kernel->kernargSegmentSize
                      << " B, LDS " << kernel->groupSegmentFixedSize << " B, VGPR "
                      << kernel->vgprCount << ", spills " << kernel->vgprSpillCount << '\n';
            require(kernel->kernargSegmentSize == expected.kernargBytes
                        && kernel->groupSegmentFixedSize == expected.ldsBytes
                        && kernel->vgprCount == expected.vgprs
                        && kernel->vgprSpillCount == expected.vgprSpills,
                    "The built kernel's resources differ from its manifest");
            require(std::none_of(kernel->arguments.begin(),
                                 kernel->arguments.end(),
                                 [](const auto& a) { return a.valueKind.rfind("hidden", 0) == 0; }),
                    "The kernel has hidden arguments");
        }
        std::cout << "PASS build matches each variant's resources\n";
    }

    // One GEMM on the device, with canaries around D and in its gaps: the
    // padding of its leading dimension and the space between its batches. C
    // holds random values, or NaN for beta 0, which must not read it, and NaN
    // in its gaps; with cIsD it is D's initial content.
    struct Gemm
    {
        static constexpr size_t guard = 4096; // elements on each side of D
        Config                  c;
        size_t                  offset; // bytes added to each base pointer
        Handle                  h;
        Descriptors             layout{c, nullptr};
        std::vector<uint16_t>   hostA, hostB, hostC;
        DeviceBuffer<char>      A, B, C, D;
        hipStream_t             stream{};

        Gemm(const Config& config, size_t offset = 0)
            : c(config)
            , offset(offset)
            , hostA(size_t(c.strideA()) * c.batch)
            , hostB(size_t(c.strideB()) * c.batch)
            , hostC(size_t(c.strideC()) * c.batch, canary)
            , A(hostA.size() * 2 + offset)
            , B(hostB.size() * 2 + offset)
            , C(hostC.size() * 2 + offset)
            , D((size_t(c.strideD()) * c.batch + 2 * guard) * 2 + offset)
        {
            HIP(hipStreamCreate(&stream));
            uint32_t seed = 12345;
            for(auto* host : {&hostA, &hostB, &hostC})
                for(size_t i = 0; i < host->size(); ++i)
                {
                    if(host == &hostC && (!c.beta || !inMatrix(i, c.leadC(), c.strideC())))
                        continue;
                    seed = seed * 1664525u + 1013904223u;
                    (*host)[i] = toBits(host == &hostC ? c.typeCD : c.typeAB,
                                        static_cast<float>(seed >> 8) / float(1 << 23) - 1.0f);
                }
            HIP(hipMemcpy(a(), hostA.data(), hostA.size() * 2, hipMemcpyHostToDevice));
            HIP(hipMemcpy(b(), hostB.data(), hostB.size() * 2, hipMemcpyHostToDevice));
            HIP(hipMemcpy(C.pointer + offset, hostC.data(), hostC.size() * 2,
                          hipMemcpyHostToDevice));
        }
        ~Gemm()
        {
            static_cast<void>(hipStreamDestroy(stream));
        }
        // Whether element i of C or D, with this leading dimension and batch
        // stride, belongs to a batch's matrix, not a gap.
        bool inMatrix(size_t i, int64_t lead, int64_t stride) const
        {
            const auto j = i % size_t(stride);
            return j < size_t(lead) * c.n && j % size_t(lead) < size_t(c.m);
        }
        void* a() const
        {
            return A.pointer + offset;
        }
        void* b() const
        {
            return B.pointer + offset;
        }
        uint16_t* base() const
        {
            return reinterpret_cast<uint16_t*>(D.pointer + offset);
        }
        void* d() const
        {
            return base() + guard;
        }
        void* cIn() const
        {
            return c.cIsD ? d() : C.pointer + offset;
        }
        jit::Request request()
        {
            jit::Request     result;
            jit::Diagnostics diagnostics;
            BLAS(jit::makeGemmRequest(h.handle,
                                      layout.desc,
                                      &c.alpha,
                                      a(),
                                      layout.la,
                                      b(),
                                      layout.lb,
                                      &c.beta,
                                      cIn(),
                                      layout.lc,
                                      d(),
                                      layout.ld,
                                      result,
                                      diagnostics));
            return result;
        }
        void poison()
        {
            std::vector<uint16_t> fill(size_t(c.strideD()) * c.batch + 2 * guard, canary);
            if(c.cIsD)
                std::copy(hostC.begin(), hostC.end(), fill.begin() + guard);
            HIP(hipMemcpy(base(), fill.data(), fill.size() * 2, hipMemcpyHostToDevice));
        }
        // Without an algorithm, hipblasLtMatmul queries the heuristic itself.
        hipblasStatus_t matmul(const hipblasLtMatmulAlgo_t* algo)
        {
            return hipblasLtMatmul(h.handle,
                                   layout.desc,
                                   &c.alpha,
                                   a(),
                                   layout.la,
                                   b(),
                                   layout.lb,
                                   &c.beta,
                                   cIn(),
                                   layout.lc,
                                   d(),
                                   layout.ld,
                                   algo,
                                   nullptr,
                                   0,
                                   stream);
        }
        // Copies D back and checks it against a CPU reference: every element
        // for small problems, a sample for large ones. Returns D.
        std::vector<uint16_t> verify(const std::string& label)
        {
            HIP(hipStreamSynchronize(stream));
            std::vector<uint16_t> all(size_t(c.strideD()) * c.batch + 2 * guard);
            HIP(hipMemcpy(all.data(), base(), all.size() * 2, hipMemcpyDeviceToHost));
            for(size_t i = 0; i < guard; ++i)
                require(all[i] == canary && all[all.size() - 1 - i] == canary,
                        label + ": wrote outside D");
            std::vector<uint16_t> out(all.begin() + guard, all.end() - guard);
            for(size_t i = 0; i < out.size(); ++i)
            {
                const bool inside = inMatrix(i, c.leadD(), c.strideD());
                require((out[i] == canary) != inside,
                        label + (inside ? ": left part of D unwritten" : ": wrote in a gap of D"));
            }
            const auto matrix  = size_t(c.m) * c.n;
            const bool full    = double(matrix) * c.k * c.batch <= double(1 << 30);
            const auto samples = full ? matrix * c.batch : size_t(8192);
            uint32_t   seed    = 777;
            const auto bound   = 0.5 * std::sqrt(double(c.k) / 8192);
            for(size_t s = 0; s < samples; ++s)
            {
                size_t row = s % c.m, col = s / c.m % c.n, batch = s / matrix;
                if(!full)
                {
                    seed = seed * 1664525u + 1013904223u;
                    row  = seed % c.m;
                    seed = seed * 1664525u + 1013904223u;
                    col  = seed % c.n;
                    seed = seed * 1664525u + 1013904223u;
                    batch = seed % c.batch;
                }
                const auto* a   = &hostA[batch * c.strideA()];
                const auto* b   = &hostB[batch * c.strideB()];
                double      sum = 0;
                for(int64_t k = 0; k < c.k; ++k)
                    sum += double(fromBits(c.typeAB, a[k + row * c.leadA()]))
                           * double(fromBits(c.typeAB, b[k + col * c.leadB()]));
                sum *= c.alpha;
                if(c.beta)
                    sum += double(c.beta)
                           * fromBits(c.typeCD,
                                      hostC[batch * c.strideC() + row + col * c.leadC()]);
                const double got
                    = fromBits(c.typeCD, out[batch * c.strideD() + row + col * c.leadD()]);
                require(std::abs(got - sum) <= bound + 0.01 * std::abs(sum),
                        label + ": D(" + std::to_string(row) + ", " + std::to_string(col) + ", "
                            + std::to_string(batch) + ") = " + std::to_string(got)
                            + ", expected " + std::to_string(sum));
            }
            return out;
        }
    };

    hipblasLtMatmulHeuristicResult_t jitAlgo(Gemm& g, const jit::Backend& provider)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        jit::Solution    solution;
        jit::Diagnostics diagnostics;
        const auto status = jit::getJitAlgo(device, g.request(), provider, 0, solution, diagnostics);
        require(status == HIPBLAS_STATUS_SUCCESS,
                g.c.name() + ": getJitAlgo: " + diagnostics.message);
        hipblasLtMatmulHeuristicResult_t result{};
        BLAS(jit::getGemmAlgo(solution, result, diagnostics));
        require(result.workspaceSize == 0, "HipKittens kernels need no workspace");
        return result;
    }

    void gemms(const jit::Backend& provider)
    {
        const int64_t shapes[][3] = {{256, 256, 128},
                                     {512, 768, 384},
                                     {1024, 1024, 1024},
                                     {256, 2048, 4096},
                                     {1280, 512, 512},
                                     {4096, 4096, 4096},
                                     {8192, 8192, 8192}};
        std::vector<Config> configs;
        for(const auto& shape : shapes)
        {
            configs.emplace_back();
            configs.back().m = shape[0], configs.back().n = shape[1], configs.back().k = shape[2];
        }
        for(const auto& [shape, alpha, beta, cIsD] : {std::tuple{shapes[1], 1.0f, 1.0f, false},
                                                      std::tuple{shapes[1], 1.0f, -0.5f, true},
                                                      std::tuple{shapes[2], 1.0f, 2.0f, true},
                                                      std::tuple{shapes[5], 1.0f, 1.0f, false},
                                                      std::tuple{shapes[1], 1.5f, 0.0f, false},
                                                      std::tuple{shapes[2], -0.25f, 1.0f, true}})
        {
            configs.emplace_back();
            auto& c = configs.back();
            c.m = shape[0], c.n = shape[1], c.k = shape[2];
            c.alpha = alpha, c.beta = beta, c.cIsD = cIsD;
        }
        for(const auto& [shape, batch, gap, beta, cIsD] :
            {std::tuple{shapes[1], 3, 0, 0.0f, false},
             std::tuple{shapes[0], 4, 64, 1.0f, true},
             std::tuple{shapes[2], 2, 512, -0.5f, false}})
        {
            configs.emplace_back();
            auto& c = configs.back();
            c.m = shape[0], c.n = shape[1], c.k = shape[2];
            c.batch = batch, c.gap = gap, c.beta = beta, c.cIsD = cIsD;
        }
        const auto add = [&](const int64_t* shape, auto change) {
            configs.emplace_back();
            auto& c = configs.back();
            c.m = shape[0], c.n = shape[1], c.k = shape[2];
            change(c);
        };
        add(shapes[1], [](Config& c) { c.lda = c.k + 8, c.ldb = c.k + 1, c.ldd = c.m + 256; });
        add(shapes[2], [](Config& c) {
            c.ldc = c.m + 1, c.ldd = c.m + 8, c.alpha = 1.5f, c.beta = 1;
        });
        add(shapes[1], [](Config& c) {
            c.lda = c.k + 64, c.ldc = c.ldd = c.m + 1, c.beta = -0.5f, c.cIsD = true;
            c.batch = 2, c.gap = 64;
        });
        const auto fp16 = [](Config& c) { c.typeAB = c.typeCD = HIP_R_16F; };
        add(shapes[2], fp16);
        add(shapes[5], fp16);
        add(shapes[1], [&](Config& c) {
            fp16(c);
            c.alpha = 1.5f, c.beta = 1, c.batch = 3, c.gap = 64;
            c.lda = c.k + 8, c.ldb = c.k + 1, c.ldc = c.m + 1, c.ldd = c.m + 256;
        });
        add(shapes[2], [&](Config& c) { fp16(c), c.alpha = -0.25f, c.beta = 2, c.cIsD = true; });
        for(const auto& c : configs)
        {
            Gemm       g(c);
            const auto algo = jitAlgo(g, provider).algo;
            g.poison();
            BLAS(g.matmul(&algo));
            const auto first = g.verify(c.name() + " hipblasLtMatmul");
            g.poison();
            BLAS(g.matmul(&algo));
            HIP(hipStreamSynchronize(g.stream));
            std::vector<uint16_t> again(first.size());
            HIP(hipMemcpy(again.data(), g.d(), again.size() * 2, hipMemcpyDeviceToHost));
            require(again == first, c.name() + ": a repeated run differs");

            hipblaslt_ext::Gemm cpp(g.h.handle,
                                    c.opA,
                                    c.opB,
                                    c.typeAB,
                                    c.typeAB,
                                    c.typeCD,
                                    c.typeCD,
                                    HIPBLAS_COMPUTE_32F);
            BLAS(cpp.setProblem(g.layout.desc,
                                &c.alpha,
                                g.a(),
                                g.layout.la,
                                g.b(),
                                g.layout.lb,
                                &c.beta,
                                g.cIn(),
                                g.layout.lc,
                                g.d(),
                                g.layout.ld));
            g.poison();
            BLAS(cpp.initialize(algo, nullptr, true, g.stream));
            BLAS(cpp.run(g.stream));
            require(g.verify(c.name() + " Gemm") == first, c.name() + ": Gemm differs");
            std::cout << "PASS " << c.name()
                      << ": hipblasLtMatmul and Gemm match the reference, canaries intact, "
                         "repeated runs identical\n";
        }
    }

    // Shapes the kernel computes wrongly must be rejected before any launch.
    void sweep(const jit::Backend& provider)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        std::vector<std::array<int64_t, 3>> shapes;
        for(int64_t k : {64, 192, 320, 448, 576, 1088, 127, 129, 136})
            shapes.push_back({256, 256, k});
        for(int64_t delta : {-128, -16, -1, 1, 16, 128})
        {
            shapes.push_back({512 + delta, 256, 256});
            shapes.push_back({256, 512 + delta, 256});
        }
        Requests r;
        for(const auto& shape : shapes)
        {
            Config c;
            c.m = shape[0], c.n = shape[1], c.k = shape[2];
            jit::Solution    solution;
            jit::Diagnostics diagnostics;
            const auto       status
                = jit::getJitAlgo(device, r.make(c), provider, 0, solution, diagnostics);
            require(status == HIPBLAS_STATUS_NOT_SUPPORTED,
                    c.name() + " was not rejected: status " + std::to_string(status));
        }
        std::cout << "PASS " << shapes.size() << " shapes outside the measured domain rejected\n";
    }

    void alignment(const jit::Backend& provider)
    {
        for(size_t offset : {2, 16})
        {
            Config c;
            c.m = 512, c.n = 768, c.k = 384;
            Gemm g(c, offset);
            const auto algo = jitAlgo(g, provider).algo;
            g.poison();
            BLAS(g.matmul(&algo));
            g.verify("offset " + std::to_string(offset));
            std::cout << "PASS base offset " << offset << " bytes\n";
        }
    }

    std::vector<int32_t> libraryAlgos(Gemm& g, const jit::Backend& provider)
    {
        int device = -1;
        HIP(hipGetDevice(&device));
        std::vector<int32_t> indices;
        jit::Diagnostics     diagnostics;
        const auto           status
            = jit::getLibraryAlgos(device, g.request(), provider, 1, 0, indices, diagnostics);
        require(status == HIPBLAS_STATUS_SUCCESS && indices.size() == 1,
                "getLibraryAlgos: " + diagnostics.message);
        return indices;
    }

    void runIndex(Gemm& g, int32_t index, const std::string& label)
    {
        std::vector<int>                              wanted{index};
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        BLAS(hipblaslt_ext::getAlgosFromIndex(g.h.handle, wanted, results));
        require(results.size() == 1, label + ": the index did not resolve");
        require(hipblaslt_ext::getKernelNameFromAlgo(g.h.handle, results[0].algo)
                    == variant().kernelName,
                label + ": the index names another kernel");
        g.poison();
        BLAS(g.matmul(&results[0].algo));
        g.verify(label);
    }

    Config libraryShape()
    {
        Config c;
        c.m = 1024, c.n = 512, c.k = 768;
        return c;
    }

    // Publishes into HIPBLASLT_JIT_LIBRARY_PATH, then a second process runs the
    // index with JIT off. Prints the index for the install check.
    void publish(const char* self)
    {
        const char* root = std::getenv("HIPBLASLT_JIT_LIBRARY_PATH");
        require(root && *root, "Set HIPBLASLT_JIT_LIBRARY_PATH to a scratch directory");
        const auto provider = backend();
        Gemm       g(libraryShape());
        const auto index = libraryAlgos(g, provider)[0];
        require(libraryAlgos(g, provider)[0] == index, "A repeated lookup returned another index");
        runIndex(g, index, "published index");
        std::cout << "PASS published and ran index " << index << '\n';

        auto strided = libraryShape();
        strided.lda  = strided.k + 64;
        {
            Gemm other(strided);
            require(libraryAlgos(other, provider)[0] == index,
                    "ldA != K got another index than ldA = K");
            runIndex(other, index, "published index with ldA != K");
        }
        std::cout << "PASS the published index serves and runs ldA != K\n";
        strided.lda = int64_t(1) << 21; // A spans 4 GiB
        {
            Requests             r;
            int                  device = -1;
            std::vector<int32_t> indices;
            jit::Diagnostics     diagnostics;
            HIP(hipGetDevice(&device));
            const auto status = jit::getLibraryAlgos(
                device, r.make(strided), provider, 1, 0, indices, diagnostics);
            require(status != HIPBLAS_STATUS_SUCCESS && indices.empty(),
                    "The published entry served an A that spans 4 GiB");
        }
        std::cout << "PASS the published entry does not serve an A that spans 4 GiB\n";

        std::cout.flush();
        const auto text  = std::to_string(index);
        const auto child = fork();
        require(child >= 0, "fork failed");
        if(child == 0)
        {
            unsetenv("HIPBLASLT_JIT");
            execl("/proc/self/exe", self, "library-reader", text.c_str(), static_cast<char*>(nullptr));
            _exit(127);
        }
        int wait = 0;
        require(waitpid(child, &wait, 0) == child && WIFEXITED(wait) && WEXITSTATUS(wait) == 0,
                "The second process failed (wait status " + std::to_string(wait) + ")");
        std::cout << "INDEX " << index << '\n';
    }

    void readIndex(int32_t index)
    {
        Gemm g(libraryShape());
        runIndex(g, index, "index from another process");
        std::cout << "PASS a second process ran the index with JIT off\n";
    }

    // The JIT heuristic, with HIPBLASLT_JIT_BACKENDS naming HipKittens first:
    // hipblasLtMatmul without an algorithm, then the C++ heuristic's first result.
    void heuristic()
    {
        Gemm g(libraryShape());
        g.poison();
        BLAS(g.matmul(nullptr));
        const auto first = g.verify("hipblasLtMatmul without an algorithm");

        hipblaslt_ext::Gemm cpp(g.h.handle,
                                g.layout.desc,
                                &g.c.alpha,
                                g.a(),
                                g.layout.la,
                                g.b(),
                                g.layout.lb,
                                &g.c.beta,
                                g.cIn(),
                                g.layout.lc,
                                g.d(),
                                g.layout.ld);
        hipblaslt_ext::GemmPreference                 preference;
        std::vector<hipblasLtMatmulHeuristicResult_t> results;
        BLAS(cpp.algoGetHeuristic(1, preference, results));
        require(results.size() == 1, "The heuristic returned no solution");
        require(hipblaslt_ext::getKernelNameFromAlgo(g.h.handle, results[0].algo)
                    == variant().kernelName,
                "The heuristic's first solution is not the HipKittens kernel");
        g.poison();
        BLAS(cpp.initialize(results[0].algo, nullptr, true, g.stream));
        BLAS(cpp.run(g.stream));
        require(g.verify("Gemm") == first,
                "The heuristic's solution differs from hipblasLtMatmul's");
        std::cout << "PASS hipblasLtMatmul without an algorithm and the heuristic run "
                  << variant().kernelName << '\n';
    }

    bool onGfx950()
    {
        int             device = -1;
        hipDeviceProp_t properties{};
        HIP(hipGetDevice(&device));
        HIP(hipGetDeviceProperties(&properties, device));
        const std::string arch = properties.gcnArchName;
        std::cout << "device: " << arch << '\n';
        return arch.rfind("gfx950", 0) == 0;
    }
}

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "";
    try
    {
        if(mode == "host" && argc == 3)
        {
            const auto scratch = fs::absolute(fs::u8path(argv[2]));
            fs::create_directories(scratch);
            headers(scratch);
            domain();
            entry();
            build();
            std::cout << "ALL HIPKITTENS HOST CHECKS PASSED\n";
        }
        else if(mode == "gpu" && argc == 2)
        {
            require(onGfx950(), "The HipKittens kernels need a gfx950 device");
            const auto provider = backend();
            gemms(provider);
            sweep(provider);
            alignment(provider);
            publish(argv[0]);
            std::cout << "ALL HIPKITTENS GPU CHECKS PASSED\n";
        }
        else if(mode == "library" && argc == 2)
        {
            require(onGfx950(), "The HipKittens kernels need a gfx950 device");
            std::cout << "headers: " << installedHeaders().u8string() << '\n';
            publish(argv[0]);
        }
        else if(mode == "library-reader" && argc == 3)
            readIndex(std::stoi(argv[2]));
        else if(mode == "heuristic" && argc == 2)
        {
            require(onGfx950(), "The HipKittens kernels need a gfx950 device");
            heuristic();
        }
        else
        {
            std::cerr << "Usage: " << argv[0] << " host SCRATCH | gpu | library | heuristic\n";
            return 2;
        }
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
