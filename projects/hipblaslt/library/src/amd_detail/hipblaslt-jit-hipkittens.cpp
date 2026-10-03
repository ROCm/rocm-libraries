// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-hipkittens.hpp"
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-heuristic.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "rocblaslt-auxiliary.h"
#include "rocblaslt_secure_env.hpp"
#include <Tensile/Tensile.hpp>
#include <algorithm>
#include <filesystem>
#include <stdexcept>

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

        std::vector<uint8_t> bytes(std::string_view text)
        {
            return {text.begin(), text.end()};
        }

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

        // Reads the headers the build staged, after checking that the directory's
        // manifest is this build's and each file matches it.
        Status readHeaders(const Options& options, std::vector<hipblaslt_jit::IncludeFile>& headers)
        {
            headers.clear();
            const auto      candidates = headerCandidates(options);
            std::error_code error;
            const auto      found      = std::find_if(
                candidates.begin(), candidates.end(), [&](const fs::path& directory) {
                    return fs::is_regular_file(directory / "manifest.json", error);
                });
            if(found == candidates.end())
                return unavailable("headers not found at " + candidates.front().u8string()
                                   + "; set HIPBLASLT_JIT_HIPKITTENS_PATH");
            const auto  root      = fs::canonical(*found, error);
            const auto& resources = detail::resources();
            std::vector<uint8_t> manifest;
            if(error || !hipblaslt_jit::files::readFile(root / "manifest.json", fileLimit, manifest).ok()
               || std::string_view(reinterpret_cast<const char*>(manifest.data()), manifest.size())
                      != resources.manifest)
                return unavailable((*found / "manifest.json").u8string()
                                   + " is not the manifest of this build's headers");
            for(const auto& header : resources.headers)
            {
                const auto path = *found / fs::u8path(header.path);
                const auto real = fs::canonical(path, error);
                if(error || !fs::is_regular_file(real, error))
                    return unavailable(path.u8string() + " is missing");
                const auto inside = real.lexically_relative(root);
                if(inside.empty() || *inside.begin() == "..")
                    return unavailable(path.u8string() + " leaves " + found->u8string());
                hipblaslt_jit::IncludeFile file;
                // Templates include the files relative to include/.
                file.name = fs::u8path(header.path).lexically_relative("include").generic_u8string();
                if(!hipblaslt_jit::files::readFile(real, fileLimit, file.bytes).ok()
                   || file.bytes.size() != header.size
                   || hipblaslt_jit::sha256Hex(std::string_view(
                          reinterpret_cast<const char*>(file.bytes.data()), file.bytes.size()))
                          != header.sha256)
                    return unavailable(path.u8string() + " does not match manifest.json");
                headers.push_back(std::move(file));
            }
            const auto rocm = hipblaslt_jit::code_object::rocmPath();
            const auto hip  = fs::u8path(rocm) / "include" / "hip" / "hip_runtime.h";
            if(rocm.empty() || !fs::is_regular_file(hip, error))
                return unavailable("HIP headers not found at " + hip.u8string()
                                   + "; set HIP_PATH");
            return {};
        }

        struct Candidate
        {
            const detail::Variant* variant;
            std::vector<uint8_t>   entry; // with the packed strides pinned
        };

        // The target's variants whose entry solves the request, or why none can.
        Status candidates(const hipblaslt_jit::OperationRequest& request,
                          const hipblaslt_jit::DeviceTarget&     target,
                          std::vector<Candidate>&                found)
        {
            found.clear();
            const auto* gemm = dynamic_cast<const jit::detail::GemmRequest*>(&request);
            if(!gemm)
                return {Status::Code::NotSupported,
                        Stage::Generate,
                        "HipKittens generates GEMM kernels only"};
            const auto& variants = detail::resources().variants;
            if(std::none_of(variants.begin(), variants.end(), [&](const detail::Variant& v) {
                   return v.isa == target.isa;
               }))
                return {Status::Code::TargetMismatch,
                        Stage::Configure,
                        "HipKittens has no kernel for " + target.isa};
            const auto problem = hipblaslt_jit::lowerForJit(*gemm);
            hipblaslt_jit::CanonicalGemm shape;
            try
            {
                shape = hipblaslt_jit::canonicalGemm(problem);
            }
            catch(const std::runtime_error& e)
            {
                return {Status::Code::NotSupported, Stage::Generate, e.what()};
            }
            for(const auto* tensor : {&problem.a(), &problem.b(), &problem.c(), &problem.d()})
                if(tensor->totalAllocatedBytes() >= tensorLimit)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "HipKittens kernels need tensors smaller than 4 GiB"};
            // The kernels read packed A, B and C and write packed D; JIT library
            // rows match only sizes, so the entry also pins the strides, C's even
            // for beta 0 because the entry serves every beta.
            const std::vector<hipblaslt_jit::msgpack_io::IndexedPredicate> packed{
                {"StrideAEqual", 1, shape.k},
                {"StrideBEqual", 1, shape.k},
                {"StrideCEqual", 1, shape.m},
                {"StrideDEqual", 1, shape.m}};
            for(const auto& variant : variants)
            {
                if(variant.isa != target.isa)
                    continue;
                Candidate candidate{&variant, {}};
                if(auto status = hipblaslt_jit::msgpack_io::appendEntryPredicates(
                       bytes(variant.entry), packed, candidate.entry);
                   !status.ok())
                    return status;
                const auto library = std::dynamic_pointer_cast<Master>(
                    TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                        candidate.entry));
                if(!library || library->solutions.size() != 1)
                    return {Status::Code::Failed,
                            Stage::Generate,
                            "The HipKittens entry of " + std::string(variant.name)
                                + " does not load"};
                if((*library->solutions.begin()->second->problemPredicate)(problem))
                    found.push_back(std::move(candidate));
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
            explicit HipKittensBackend(std::vector<hipblaslt_jit::IncludeFile> headers)
                : m_headers(std::move(headers))
                , m_info{"hipkittens", "HipKittens", ""}
            {
                // The headers match the manifest, so it stands for their content.
                const auto&          resources = detail::resources();
                hipblaslt_jit::Fnv1a version;
                version.add(resources.manifest);
                for(const auto& variant : resources.variants)
                {
                    version.add(variant.name).add(variant.source).add(variant.entry);
                    for(const auto& flag : variant.hipFlags)
                        version.add(flag);
                }
                m_info.version = "hipkittens:" + version.hex();
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status accepts(const hipblaslt_jit::OperationRequest& request,
                           const hipblaslt_jit::DeviceTarget&     target) const override
            {
                std::vector<Candidate> found;
                return candidates(request, target, found);
            }

            Status generate(const hipblaslt_jit::GenerationRequest&        request,
                            std::vector<hipblaslt_jit::GeneratedSolution>& solutions) const override
            {
                solutions.clear();
                std::vector<Candidate> found;
                if(auto status = candidates(request.request, request.target, found); !status.ok())
                    return status;
                for(auto& candidate : found)
                {
                    const auto& variant = *candidate.variant;
                    const auto& skip    = request.excludeKernels;
                    if(solutions.size() >= request.count
                       || std::find(skip.begin(), skip.end(), variant.kernelName) != skip.end())
                        continue;
                    hipblaslt_jit::GeneratedSolution solution;
                    solution.entry      = std::move(candidate.entry);
                    solution.kernelName = std::string(variant.kernelName);
                    solution.hipFlags   = variant.hipFlags;
                    hipblaslt_jit::BuildUnit unit;
                    unit.role     = hipblaslt_jit::BuildUnit::Role::Main;
                    unit.kind     = hipblaslt_jit::BuildUnit::Kind::Hip;
                    unit.name     = std::string(variant.name) + ".hip";
                    unit.bytes    = bytes(variant.source);
                    unit.includes = m_headers;
                    solution.units.push_back(std::move(unit));
                    solutions.push_back(std::move(solution));
                }
                return {};
            }

        private:
            std::vector<hipblaslt_jit::IncludeFile> m_headers;
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
            std::vector<hipblaslt_jit::IncludeFile> headers;
            if(auto status = readHeaders(options, headers); !status.ok())
            {
                diagnostics.message = status.message;
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            backend = jit::detail::BackendAccess::make(
                std::make_shared<const hipblaslt_jit::Jit>(hipblaslt_jit::Jit::Components{
                    std::make_shared<const HipKittensBackend>(std::move(headers)),
                    nullptr,
                    nullptr,
                    hipblaslt_jit::makeComgrBuilder(),
                    hipblaslt_jit::makeTensileLoader(),
                    nullptr}));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate HipKittens backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        catch(const std::exception& e)
        {
            diagnostics.message = e.what();
            return HIPBLAS_STATUS_INTERNAL_ERROR;
        }
    }

    Status detail::makeProcessBackend(hipblaslt_jit::ProcessBackend& made)
    {
        std::vector<hipblaslt_jit::IncludeFile> headers;
        auto                                    status = readHeaders({}, headers);
        if(status.ok())
            made.backend = std::make_shared<const HipKittensBackend>(std::move(headers));
        return status;
    }
}
