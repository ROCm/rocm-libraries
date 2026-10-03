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

    // A BF16 NaN that no GEMM of finite inputs writes.
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

    const hk::detail::Variant& variant()
    {
        const auto& variants = hk::detail::resources().variants;
        require(variants.size() == 1, "Expected one HipKittens variant");
        return variants.front();
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
        std::string name() const
        {
            std::ostringstream out;
            out << m << 'x' << n << 'x' << k;
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
                                    int64_t                  lead) {
                BLAS(hipblasLtMatrixLayoutCreate(&l, type, rows, cols, lead));
                const int64_t stride = lead * cols;
                BLAS(hipblasLtMatrixLayoutSetAttribute(
                    l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &c.batch, sizeof(c.batch)));
                BLAS(hipblasLtMatrixLayoutSetAttribute(
                    l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, sizeof(stride)));
            };
            layout(la, c.typeAB, c.rowsA(), c.opA == HIPBLAS_OP_N ? c.k : c.m, c.leadA());
            layout(lb, c.typeAB, c.rowsB(), c.opB == HIPBLAS_OP_N ? c.n : c.k, c.leadB());
            layout(lc, c.typeCD, c.m, c.n, c.leadC());
            layout(ld, c.typeCD, c.m, c.n, c.leadD());
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
                             Case{"alpha 1.5", with([](Config& c) { c.alpha = 1.5f; })}})
        {
            status = generate(provider, r.make(c.config), target, solutions);
            require(status.ok() && solutions.size() == 1,
                    std::string(c.label) + ": " + status.message);
            std::cout << "PASS one solution for " << c.label << '\n';
        }
        const std::vector<Case> cases{
            {"NN", with([](Config& c) { c.opA = HIPBLAS_OP_N; })},
            {"NT", with([](Config& c) { c.opA = HIPBLAS_OP_N, c.opB = HIPBLAS_OP_T; })},
            {"fp16 in and out", with([](Config& c) { c.typeAB = c.typeCD = HIP_R_16F; })},
            {"fp32 out", with([](Config& c) { c.typeCD = HIP_R_32F; })},
            {"alpha 0 (K = 0 in hipBLASLt)", with([](Config& c) { c.alpha = 0; })},
            {"device alpha",
             with([](Config& c) { c.pointerMode = HIPBLASLT_POINTER_MODE_DEVICE; })},
            {"alpha vector",
             with([](Config& c) {
                 c.pointerMode = HIPBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST;
             })},
            {"batch 2", with([](Config& c) { c.batch = 2; })},
            {"M = 300", with([](Config& c) { c.m = 300; })},
            {"N = 384", with([](Config& c) { c.n = 384; })},
            {"K = 192", with([](Config& c) { c.k = 192; })},
            {"K = 64", with([](Config& c) { c.k = 64; })},
            {"ldA != K", with([](Config& c) { c.lda = c.k + 64; })},
            {"ldB != K", with([](Config& c) { c.ldb = c.k + 64; })},
            {"ldC != M", with([](Config& c) { c.ldc = c.m + 256; })},
            {"ldC != M, beta 1", with([](Config& c) { c.ldc = c.m + 256, c.beta = 1; })},
            {"ldD != M", with([](Config& c) { c.ldd = c.m + 256; })},
            {"bias", with([](Config& c) { c.epilogue = HIPBLASLT_EPILOGUE_BIAS; })},
            {"ReLU", with([](Config& c) { c.epilogue = HIPBLASLT_EPILOGUE_RELU; })},
            {"8 GiB D", with([](Config& c) { c.m = c.n = 65536, c.k = 128; })},
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
        const auto status = generate(backend(), r.make(c), gfx950(), solutions);
        require(status.ok() && solutions.size() == 1, "512x768x384: " + status.message);
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
        for(const char* type : {"BatchSizeEqual",
                                "Free0SizeMultiple",
                                "Free1SizeMultiple",
                                "BoundSizeMultiple",
                                "TypesEqual",
                                "OperationIdentifierEqual"})
            require(types.count(type), std::string("The entry has no ") + type);
        namespace P = TensileLite::Predicates::Contraction;
        const auto stride = [&](auto* tag) {
            using T = std::remove_pointer_t<decltype(tag)>;
            for(const auto& term : all->value)
                if(const auto found = std::dynamic_pointer_cast<T>(term))
                    return std::make_pair(found->index, found->value);
            throw std::runtime_error("The entry has no " + T::Type());
        };
        using Pin = std::pair<size_t, size_t>;
        require(stride(static_cast<P::StrideAEqual*>(nullptr)) == Pin(1, 384)
                    && stride(static_cast<P::StrideBEqual*>(nullptr)) == Pin(1, 384)
                    && stride(static_cast<P::StrideCEqual*>(nullptr)) == Pin(1, 512)
                    && stride(static_cast<P::StrideDEqual*>(nullptr)) == Pin(1, 512),
                "The entry does not pin packed strides");
        std::cout << "PASS entry: handwritten custom kernel, static and stride predicates\n";
    }

    void build()
    {
        Requests                           r;
        std::vector<hj::GeneratedSolution> solutions;
        const auto                         provider = backend();
        const auto                         target   = gfx950();
        const auto                         request  = r.make({});
        auto status = generate(provider, request, target, solutions);
        require(status.ok() && solutions.size() == 1, "1024^3: " + status.message);
        hj::GenerationRequest generation{*abi::RequestAccess::get(request), target};
        hj::BuiltSolution     built;
        status = abi::BackendAccess::get(provider)->components().builder->build(
            solutions[0], generation, built);
        require(status.ok(), "Build: " + status.message);
        const auto metadata = co::readMetadata(built.object.bytes.data(), built.object.bytes.size());
        require(metadata.ok(), "readMetadata: " + metadata.log);
        const auto& kernels = metadata.metadata.kernels;
        const auto  kernel  = std::find_if(kernels.begin(), kernels.end(), [](const auto& k) {
            return k.name == variant().kernelName;
        });
        require(kernel != kernels.end(), "The code object has no " + std::string(variant().kernelName));
        const auto& expected = variant().resources;
        std::cout << "built: kernarg " << kernel->kernargSegmentSize << " B, LDS "
                  << kernel->groupSegmentFixedSize << " B, VGPR " << kernel->vgprCount
                  << ", spills " << kernel->vgprSpillCount << '\n';
        require(kernel->kernargSegmentSize == expected.kernargBytes
                    && kernel->groupSegmentFixedSize == expected.ldsBytes
                    && kernel->vgprCount == expected.vgprs
                    && kernel->vgprSpillCount == expected.vgprSpills,
                "The built kernel's resources differ from its manifest");
        require(std::none_of(kernel->arguments.begin(),
                             kernel->arguments.end(),
                             [](const auto& a) { return a.valueKind.rfind("hidden", 0) == 0; }),
                "The kernel has hidden arguments");
        std::cout << "PASS build matches the variant's resources\n";
    }

    // One packed GEMM on the device, with canaries around D. C holds random
    // values, or NaN for beta 0, which must not read it; with cIsD it is D's
    // initial content.
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
            , hostA(size_t(c.k) * c.m)
            , hostB(size_t(c.k) * c.n)
            , hostC(size_t(c.m) * c.n, canary)
            , A(hostA.size() * 2 + offset)
            , B(hostB.size() * 2 + offset)
            , C(hostC.size() * 2 + offset)
            , D((size_t(c.m) * c.n + 2 * guard) * 2 + offset)
        {
            HIP(hipStreamCreate(&stream));
            uint32_t seed = 12345;
            for(auto* host : {&hostA, &hostB, &hostC})
                for(auto& value : *host)
                {
                    if(host == &hostC && !c.beta)
                        break;
                    seed = seed * 1664525u + 1013904223u;
                    value = toBf16(static_cast<float>(seed >> 8) / float(1 << 23) - 1.0f);
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
            std::vector<uint16_t> fill(size_t(c.m) * c.n + 2 * guard, canary);
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
            std::vector<uint16_t> all(size_t(c.m) * c.n + 2 * guard);
            HIP(hipMemcpy(all.data(), base(), all.size() * 2, hipMemcpyDeviceToHost));
            for(size_t i = 0; i < guard; ++i)
                require(all[i] == canary && all[all.size() - 1 - i] == canary,
                        label + ": wrote outside D");
            std::vector<uint16_t> out(all.begin() + guard, all.end() - guard);
            require(std::none_of(out.begin(), out.end(), [](uint16_t v) { return v == canary; }),
                    label + ": left part of D unwritten");
            const bool full    = double(c.m) * c.n * c.k <= double(1 << 30);
            const auto samples = full ? size_t(c.m) * c.n : size_t(8192);
            uint32_t   seed    = 777;
            const auto bound   = 0.5 * std::sqrt(double(c.k) / 8192);
            for(size_t s = 0; s < samples; ++s)
            {
                size_t row = s % c.m, col = s / c.m;
                if(!full)
                {
                    seed = seed * 1664525u + 1013904223u;
                    row  = seed % c.m;
                    seed = seed * 1664525u + 1013904223u;
                    col  = seed % c.n;
                }
                double sum = 0;
                for(int64_t k = 0; k < c.k; ++k)
                    sum += double(fromBf16(hostA[k + row * c.k]))
                           * double(fromBf16(hostB[k + col * c.k]));
                sum *= c.alpha;
                if(c.beta)
                    sum += double(c.beta) * fromBf16(hostC[row + col * c.m]);
                const double got = fromBf16(out[row + col * c.m]);
                require(std::abs(got - sum) <= bound + 0.01 * std::abs(sum),
                        label + ": D(" + std::to_string(row) + ", " + std::to_string(col)
                            + ") = " + std::to_string(got) + ", expected "
                            + std::to_string(sum));
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
        Gemm                 other(strided);
        int                  device = -1;
        std::vector<int32_t> indices;
        jit::Diagnostics     diagnostics;
        HIP(hipGetDevice(&device));
        const auto status
            = jit::getLibraryAlgos(device, other.request(), provider, 1, 0, indices, diagnostics);
        require(status != HIPBLAS_STATUS_SUCCESS && indices.empty(),
                "The packed entry served a problem with ldA != K");
        std::cout << "PASS the published entry does not serve ldA != K\n";

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
