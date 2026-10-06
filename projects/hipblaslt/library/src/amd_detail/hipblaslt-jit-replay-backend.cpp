// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-replay.hpp"
#include <stdexcept>
#include <string_view>

namespace hipblaslt_ext::experimental::jit::replay
{
    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;

        struct Replayed
        {
            hipblaslt_jit::GeneratedSolution        solution;
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
                const auto text = [](const std::vector<uint8_t>& bytes) {
                    return std::string_view(reinterpret_cast<const char*>(bytes.data()),
                                            bytes.size());
                };
                hipblaslt_jit::Fnv1a version;
                for(const auto& path : options.replay)
                {
                    Replayed replayed;
                    auto     read     = hipblaslt_jit::readTensileSourceBundle(fs::u8path(path));
                    replayed.solution = std::move(read.solution);
                    replayed.library  = std::move(read.library);
                    if(!replayed.library || replayed.library->solutions.empty())
                        throw std::invalid_argument("The bundle " + path
                                                    + " holds no GEMM solution");
                    const auto& solution = replayed.solution;
                    version.add(text(solution.entry));
                    for(const auto& kernel : solution.kernelNames)
                        version.add(kernel);
                    for(const auto& unit : solution.units)
                    {
                        version.add(unit.name).add(text(unit.bytes));
                        for(const auto& include : unit.includes)
                            version.add(include.name).add(text(include.bytes));
                    }
                    m_replayed.push_back(std::move(replayed));
                }
                m_info.version = "replay:" + version.hex();
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status generate(const hipblaslt_jit::GenerationRequest&        request,
                            std::vector<hipblaslt_jit::GeneratedSolution>& solutions) const override
            {
                solutions.clear();
                const auto* gemm = dynamic_cast<const detail::GemmRequest*>(&request.request);
                if(!gemm)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "The replay backend replays a GEMM solution"};
                const auto problem  = hipblaslt_jit::lowerForJit(*gemm);
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
                    solutions.push_back(replayed.solution);
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

        private:
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
            backend = detail::BackendAccess::make(hipblaslt_jit::makeJit(makeBackend(options)));
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
