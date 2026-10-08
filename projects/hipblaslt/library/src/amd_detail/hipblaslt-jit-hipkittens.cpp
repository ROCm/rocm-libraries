// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-hipkittens.hpp"
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "rocblaslt-auxiliary.h"
#include "rocblaslt_secure_env.hpp"
#include <Tensile/Tensile.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string_view>

namespace hipblaslt_ext::experimental::jit::hipkittens
{
    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;
        using Master = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

        constexpr size_t fileLimit = size_t(1) << 20;
        // The kernels address each tensor with 32-bit byte offsets.
        constexpr size_t tensorLimit = size_t(1) << 32;

        Status unavailable(const std::string& cause)
        {
            return {Status::Code::Failed,
                    Stage::Configure,
                    "JIT backend HipKittens not available: " + cause};
        }

        std::vector<fs::path> headerCandidates(const Options& options)
        {
            if(!options.headers.empty())
                return {fs::u8path(options.headers)};
            if(const char* path = rocblaslt_secure_getenv("HIPBLASLT_JIT_HIPKITTENS_PATH");
               path && *path)
                return {fs::u8path(path)};
            std::vector<fs::path> candidates;
            try
            {
                candidates.push_back(fs::u8path(rocblaslt_internal_get_so_path()).parent_path()
                                     / fs::u8path(HIPBLASLT_JIT_HIPKITTENS_DIR));
            }
            catch(const std::runtime_error&)
            {
            }
            candidates.push_back(fs::u8path(HIPBLASLT_JIT_HIPKITTENS_FALLBACK));
            return candidates;
        }

        // SHA-256 of bytes as 64 lowercase hex digits, for the staged header manifest.
        std::string sha256Hex(std::string_view bytes)
        {
            static constexpr uint32_t k[64]
                = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
                   0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
                   0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
                   0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                   0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
                   0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
                   0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
                   0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                   0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
                   0xc67178f2};
            uint32_t h[8] = {0x6a09e667,
                             0xbb67ae85,
                             0x3c6ef372,
                             0xa54ff53a,
                             0x510e527f,
                             0x9b05688c,
                             0x1f83d9ab,
                             0x5be0cd19};
            const auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
            std::string message(bytes);
            const uint64_t bits = static_cast<uint64_t>(bytes.size()) * 8;
            message.push_back(static_cast<char>(0x80));
            while(message.size() % 64 != 56)
                message.push_back('\0');
            for(int shift = 56; shift >= 0; shift -= 8)
                message.push_back(static_cast<char>((bits >> shift) & 0xff));
            for(size_t block = 0; block < message.size(); block += 64)
            {
                uint32_t w[64];
                for(int i = 0; i < 16; ++i)
                {
                    w[i] = 0;
                    for(int j = 0; j < 4; ++j)
                        w[i] = (w[i] << 8) | static_cast<unsigned char>(message[block + 4 * i + j]);
                }
                for(int i = 16; i < 64; ++i)
                {
                    const auto s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
                    const auto s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
                    w[i]          = w[i - 16] + s0 + w[i - 7] + s1;
                }
                uint32_t v[8];
                for(int i = 0; i < 8; ++i)
                    v[i] = h[i];
                for(int i = 0; i < 64; ++i)
                {
                    const auto s1 = rotr(v[4], 6) ^ rotr(v[4], 11) ^ rotr(v[4], 25);
                    const auto t1 = v[7] + s1 + ((v[4] & v[5]) ^ (~v[4] & v[6])) + k[i] + w[i];
                    const auto s0 = rotr(v[0], 2) ^ rotr(v[0], 13) ^ rotr(v[0], 22);
                    const auto t2 = s0 + ((v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]));
                    for(int j = 7; j > 0; --j)
                        v[j] = v[j - 1];
                    v[4] += t1;
                    v[0] = t1 + t2;
                }
                for(int i = 0; i < 8; ++i)
                    h[i] += v[i];
            }
            char text[65];
            for(int i = 0; i < 8; ++i)
                std::snprintf(text + 8 * i, 9, "%08x", h[i]);
            return text;
        }

        // Reads the headers the build staged, after checking that the directory's
        // manifest is this build's and each file matches it.
        Status readHeaders(const Options& options, std::vector<hipblaslt_jit::BuildUnit>& headers)
        {
            headers.clear();
            const auto      candidates = headerCandidates(options);
            std::error_code error;
            const auto      found = std::find_if(
                candidates.begin(), candidates.end(), [&](const fs::path& directory) {
                    return fs::is_regular_file(directory / "manifest.msgpack", error);
                });
            if(found == candidates.end())
                return unavailable("headers not found at " + candidates.front().u8string()
                                   + "; set HIPBLASLT_JIT_HIPKITTENS_PATH");
            const auto  root      = fs::canonical(*found, error);
            const auto& resources = detail::resources();
            std::vector<uint8_t> manifest;
            if(error
               || !hipblaslt_jit::files::readFile(root / "manifest.msgpack", fileLimit, manifest).ok()
               || manifest != resources.manifest)
                return unavailable((*found / "manifest.msgpack").u8string()
                                   + " is not the manifest of this build's headers");
            for(const auto& header : resources.headers)
            {
                const auto path = *found / header.path;
                const auto real = fs::canonical(path, error);
                if(error || !fs::is_regular_file(real, error))
                    return unavailable(path.u8string() + " is missing");
                if(!hipblaslt_jit::source_bundle::pathStaysInside(root, real))
                    return unavailable(path.u8string() + " leaves " + found->u8string());
                hipblaslt_jit::BuildUnit file;
                // Templates include the files relative to include/.
                file.name = header.path.lexically_relative("include").generic_string();
                if(!hipblaslt_jit::files::readFile(real, fileLimit, file.bytes).ok()
                   || file.bytes.size() != header.size
                   || sha256Hex(std::string_view(
                          reinterpret_cast<const char*>(file.bytes.data()), file.bytes.size()))
                          != header.sha256)
                    return unavailable(path.u8string() + " does not match the header manifest");
                headers.push_back(std::move(file));
            }
            const auto rocm = hipblaslt_jit::code_object::rocmPath();
            const auto hip  = fs::u8path(rocm) / "include" / "hip" / "hip_runtime.h";
            if(rocm.empty() || !fs::is_regular_file(hip, error))
                return unavailable("HIP headers not found at " + hip.u8string()
                                   + "; set HIP_PATH");
            return {};
        }

        // The target's variants whose entry solves the request, or why none can.
        Status candidates(const hipblaslt_jit::OperationRequest&  request,
                          const hipblaslt_jit::DeviceTarget&      target,
                          std::vector<const detail::Variant*>&    found)
        {
            found.clear();
            const auto* gemm = dynamic_cast<const jit::detail::GemmRequest*>(&request);
            if(!gemm)
                return {Status::Code::NotSupported,
                        Stage::Generate,
                        "HipKittens generates GEMM kernels only"};
            const auto& variants = detail::resources().variants;
            if(std::none_of(variants.begin(), variants.end(), [&](const detail::Variant& variant) {
                   return variant.isa.gfx == target.isa;
               }))
                return {Status::Code::TargetMismatch,
                        Stage::Configure,
                        "HipKittens has no kernel for " + target.isa};
            const auto problem = hipblaslt_jit::lowerForJit(*gemm);
            try
            {
                hipblaslt_jit::canonicalGemm(problem);
            }
            catch(const std::runtime_error& error)
            {
                return {Status::Code::NotSupported, Stage::Generate, error.what()};
            }
            // The custom call passes base addresses and batch strides only.
            if(problem.batchMode() == TensileLite::ContractionProblemGemm::BATCHMODE::POINTER_ARRAY)
                return {Status::Code::NotSupported,
                        Stage::Generate,
                        "HipKittens kernels need strided batches"};
            for(const auto* tensor : {&problem.a(), &problem.b(), &problem.c(), &problem.d()})
                if(tensor->totalAllocatedBytes() >= tensorLimit)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "HipKittens kernels need tensors smaller than 4 GiB"};
            for(const auto& variant : variants)
            {
                if(variant.isa.gfx != target.isa)
                    continue;
                std::shared_ptr<Master> library;
                try
                {
                    library = hipblaslt_jit::loadGemmLibrary(variant.entry);
                }
                catch(const std::exception&)
                {
                    library.reset();
                }
                if(!library || library->solutions.size() != 1)
                    return {Status::Code::Failed,
                            Stage::Generate,
                            "The HipKittens entry of " + variant.name + " does not load"};
                if((*library->solutions.begin()->second->problemPredicate)(problem))
                    found.push_back(&variant);
            }
            if(found.empty())
                return {Status::Code::NotSupported,
                        Stage::Generate,
                        "No HipKittens kernel solves this problem"};
            return {};
        }

        class HipKittensBackend final : public hipblaslt_jit::Backend
        {
        public:
            explicit HipKittensBackend(std::vector<hipblaslt_jit::BuildUnit> headers)
                : m_headers(std::move(headers))
                , m_info{"hipkittens", "HipKittens", {}}
            {
                // Compatibility is per solution. The backend version stays this
                // literal so a rebuilt bundle still opens the same library directory.
                m_info.version = "hipkittens";
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status generate(const hipblaslt_jit::GenerationRequest&        request,
                            std::vector<hipblaslt_jit::GeneratedSolution>& solutions) const override
            {
                solutions.clear();
                std::vector<const detail::Variant*> found;
                if(auto status = candidates(request.request, request.target, found); !status.ok())
                    return status;
                for(const auto* variant : found)
                {
                    const auto& skip = request.excludeKernels;
                    if(solutions.size() >= request.count
                       || std::find(skip.begin(), skip.end(), variant->kernelName) != skip.end())
                        continue;
                    hipblaslt_jit::GeneratedSolution solution;
                    solution.entry       = variant->entry;
                    solution.kernelNames = {variant->kernelName};
                    solution.hipFlags    = variant->hipFlags;
                    hipblaslt_jit::BuildUnit unit = variant->source;
                    unit.kind                     = hipblaslt_jit::BuildUnit::Kind::Hip;
                    unit.includes                 = m_headers;
                    solution.units.push_back(std::move(unit));
                    solutions.push_back(std::move(solution));
                }
                return {};
            }

        private:
            std::vector<hipblaslt_jit::BuildUnit> m_headers;
            hipblaslt_jit::BackendInfo              m_info;
        };
    }

    hipblasStatus_t
        createBackend(const Options& options, Backend& backend, Diagnostics& diagnostics)
    {
        backend     = {};
        diagnostics = {"HipKittens", ""};
        try
        {
            std::vector<hipblaslt_jit::BuildUnit> headers;
            if(auto status = readHeaders(options, headers); !status.ok())
            {
                diagnostics.message = status.message;
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            backend = Backend::create(hipblaslt_jit::makeJit(
                std::make_shared<const HipKittensBackend>(std::move(headers))));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate HipKittens backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        catch(const std::exception& error)
        {
            diagnostics.message = error.what();
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
    }
}
