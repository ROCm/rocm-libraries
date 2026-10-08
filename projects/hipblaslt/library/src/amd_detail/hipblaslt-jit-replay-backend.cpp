// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-replay.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <algorithm>
#include <stdexcept>

namespace hipblaslt_ext::experimental::jit::replay
{
    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;

        bool sameFile(const hipblaslt_jit::Artifact& left, const hipblaslt_jit::Artifact& right)
        {
            return left.name == right.name && left.kind == right.kind && left.bytes == right.bytes;
        }

        bool sameSource(const hipblaslt_jit::SourceCodeBundle& left,
                        const hipblaslt_jit::SourceCodeBundle& right)
        {
            if(!sameFile(left.mainKernel, right.mainKernel) || left.hipFlags != right.hipFlags
               || left.helpers.size() != right.helpers.size()
               || left.includes.size() != right.includes.size())
                return false;
            for(size_t i = 0; i < left.helpers.size(); ++i)
                if(!sameFile(left.helpers[i], right.helpers[i]))
                    return false;
            for(size_t i = 0; i < left.includes.size(); ++i)
                if(!sameFile(left.includes[i], right.includes[i]))
                    return false;
            return true;
        }

        struct Replayed
        {
            hipblaslt_jit::SourceCodeBundle                source;
            std::vector<uint8_t>                           record;
            std::shared_ptr<hipblaslt_jit::TensileLibrary> library;
        };

        class ReplayBackend final : public hipblaslt_jit::Backend
        {
        public:
            explicit ReplayBackend(const Options& options)
                : m_info{"replay", "replay", ""}
            {
                if(options.replay.empty())
                    throw std::invalid_argument("The replay backend has no bundle to replay");
                for(const auto& path : options.replay)
                {
                    Replayed replayed;
                    const auto directory = fs::u8path(path);
                    auto       read      = hipblaslt_jit::readTensileSourceBundle(directory);
                    replayed.source      = std::move(read.source);
                    namespace artifacts  = hipblaslt_jit::source_bundle;
                    replayed.record = artifacts::readArtifact(
                        artifacts::artifact(directory, "library/TensileLibrary.dat"));
                    replayed.library = hipblaslt_jit::loadGemmLibrary(replayed.record);
                    if(!replayed.library || replayed.library->solutions.empty())
                        throw std::invalid_argument("The bundle " + path
                                                    + " holds no GEMM solution");
                    m_replayed.push_back(std::move(replayed));
                }
                // Compatibility is the code-object version and the kernarg version
                // stored on each solution. The backend version stays this literal
                // so a rebuilt bundle still opens the same library directory.
                m_info.version = "replay";
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status generate(const hipblaslt_jit::GenerationRequest&       request,
                            std::vector<hipblaslt_jit::SourceCodeBundle>& solutions) const override
            {
                solutions.clear();
                const auto problem
                    = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(request.request);
                const auto excluded = [&](const std::string& kernel) {
                    return hipblaslt_jit::excludedKernel(request.excludeKernels, kernel);
                };
                bool   targeted = false, solves = false;
                size_t solving  = 0;
                for(const auto& replayed : m_replayed)
                {
                    if(solving >= request.count)
                        break;
                    size_t entrySolving = 0;
                    bool   fresh        = false;
                    for(const auto& [index, solution] : replayed.library->solutions)
                    {
                        if(!request.target.hardware
                           || !(*solution->hardwarePredicate)(*request.target.hardware))
                            continue;
                        targeted = true;
                        if(!(*solution->problemPredicate)(problem))
                            continue;
                        solves = true;
                        ++entrySolving;
                        fresh = fresh || !excluded(solution->kernelName);
                    }
                    if(!fresh)
                        continue;
                    solving += entrySolving;
                    solutions.push_back(replayed.source);
                }
                if(!targeted)
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "No replayed solution targets " + request.target.isa};
                if(!solves)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "No replayed solution solves this problem"};
                return {Status::Code::Success,
                        Stage::Generate,
                        "Replayed " + std::to_string(solutions.size())
                            + (solutions.size() == 1 ? " bundle" : " bundles")};
            }

            std::vector<std::string>
                kernelNames(const hipblaslt_jit::SourceCodeBundle& source) const override
            {
                const auto* replayed = find(source);
                if(!replayed)
                    return {};
                std::vector<std::string> names;
                for(const auto& [index, solution] : replayed->library->solutions)
                    if(std::find(names.begin(), names.end(), solution->kernelName) == names.end())
                        names.push_back(solution->kernelName);
                return names;
            }

            std::vector<uint8_t>
                solutionRecord(const hipblaslt_jit::SourceCodeBundle& source) const override
            {
                const auto* replayed = find(source);
                return replayed ? replayed->record : std::vector<uint8_t>{};
            }

        private:
            const Replayed* find(const hipblaslt_jit::SourceCodeBundle& source) const
            {
                for(const auto& replayed : m_replayed)
                    if(sameSource(replayed.source, source))
                        return &replayed;
                return nullptr;
            }

            hipblaslt_jit::BackendInfo m_info;
            std::vector<Replayed>      m_replayed;
        };
    }

    std::shared_ptr<const hipblaslt_jit::Backend> makeBackend(const Options& options)
    {
        return std::make_shared<const ReplayBackend>(options);
    }

    hipblasStatus_t
        createBackend(const Options& options, Backend& backend, Diagnostics& diagnostics)
    {
        backend     = {};
        diagnostics = {"replay", ""};
        try
        {
            backend = Backend::create(hipblaslt_jit::makeJit(makeBackend(options)));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate the replay backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        catch(const std::exception& e)
        {
            diagnostics.message = e.what();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
    }
}
