// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-mock.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include <Tensile/Tensile.hpp>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string_view>

namespace hipblaslt_ext::experimental::jit::mock
{
    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;
        using Master = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;
        using Role   = hipblaslt_jit::BuildUnit::Role;

        struct Replayed
        {
            hipblaslt_jit::GeneratedSolution solution;
            std::shared_ptr<Master>          library;
        };

        // One line per request: kind, device, problem sizes, count and excluded kernels.
        std::string describe(const hipblaslt_jit::GenerationRequest& request,
                             const detail::GemmRequest&              gemm)
        {
            std::string line(request.request.kind());
            line += ' ' + request.target.isa + " sizes=";
            const auto sizes = hipblaslt_jit::problemSizes(hipblaslt_jit::lowerForJit(gemm));
            for(size_t i = 0; i < sizes.size(); ++i)
                line += (i ? "," : "") + std::to_string(sizes[i]);
            line += " count=" + std::to_string(request.count) + " exclude=";
            for(size_t i = 0; i < request.excludeKernels.size(); ++i)
                line += (i ? "," : "") + request.excludeKernels[i];
            return line;
        }

        class MockBackend final : public hipblaslt_jit::Backend
        {
        public:
            explicit MockBackend(const Options& options)
                : m_fault(options.fault)
                , m_record(options.record)
                , m_info{"mock", "mock", options.contract, ""}
            {
                if(options.replay.empty())
                    throw std::invalid_argument("The mock backend has no bundle to replay");
                if(m_fault == Options::Fault::Record && m_record.empty())
                    throw std::invalid_argument("The mock record fault has no file to record to");
                const auto text = [](const std::vector<uint8_t>& bytes) {
                    return std::string_view(reinterpret_cast<const char*>(bytes.data()),
                                            bytes.size());
                };
                hipblaslt_jit::Fnv1a version;
                for(const auto& path : options.replay)
                {
                    Replayed replayed;
                    replayed.solution = hipblaslt_jit::readTensileSourceBundle(fs::u8path(path));
                    replayed.library  = std::dynamic_pointer_cast<Master>(
                        TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                            replayed.solution.entry));
                    if(!replayed.library || replayed.library->solutions.empty())
                        throw std::invalid_argument("The bundle " + path
                                                    + " holds no GEMM solution");
                    const auto& solution = replayed.solution;
                    version.add(text(solution.entry)).add(solution.kernelName);
                    for(const auto& unit : solution.units)
                    {
                        version.add(unit.name).add(text(unit.bytes));
                        for(const auto& include : unit.includes)
                            version.add(include.name).add(text(include.bytes));
                    }
                    m_replayed.push_back(std::move(replayed));
                }
                m_info.version = "mock:" + version.hex();
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status generate(const hipblaslt_jit::GenerationRequest&        request,
                            std::vector<hipblaslt_jit::GeneratedSolution>& solutions) const override
            {
                solutions.clear();
                if(m_fault == Options::Fault::Trap)
                    std::abort();
                const auto* gemm = dynamic_cast<const detail::GemmRequest*>(&request.request);
                if(!gemm)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "The mock backend replays a GEMM solution"};
                if(m_fault == Options::Fault::Record)
                {
                    std::ofstream file(fs::u8path(m_record), std::ios::app);
                    file << describe(request, *gemm) << '\n';
                    return {Status::Code::Failed,
                            Stage::Generate,
                            (file ? "Mock generation recorded its request in "
                                  : "Mock generation could not record its request in ")
                                + m_record};
                }
                if(m_fault == Options::Fault::Generate)
                {
                    Status     failure{Status::Code::Failed, Stage::Generate, "Mock generation fault"};
                    const auto log = request.scratch / "mock.log";
                    std::ofstream(log) << failure.message << '\n' << describe(request, *gemm) << '\n';
                    std::error_code error;
                    if(fs::exists(log, error))
                        failure.logPath = log.u8string();
                    return failure;
                }
                const auto problem  = hipblaslt_jit::lowerForJit(*gemm);
                const auto excluded = [&](const std::string& kernel) {
                    const auto& names = request.excludeKernels;
                    return std::find(names.begin(), names.end(), kernel) != names.end();
                };
                bool targeted = false, solves = false;
                for(const auto& replayed : m_replayed)
                {
                    if(solutions.size() == request.count)
                        break;
                    const auto& solution = *replayed.library->solutions.at(0);
                    if(!request.target.hardware
                       || !(*solution.hardwarePredicate)(*request.target.hardware))
                        continue;
                    targeted = true;
                    if(!(*solution.problemPredicate)(problem))
                        continue;
                    solves = true;
                    if(excluded(replayed.solution.kernelName))
                        continue;
                    solutions.push_back(replayed.solution);
                    if(m_fault == Options::Fault::Build)
                    {
                        const std::string invalid = "s_not_an_instruction\n";
                        for(auto& unit : solutions.back().units)
                            if(unit.role == Role::Main)
                                unit.bytes.assign(invalid.begin(), invalid.end());
                    }
                }
                if(!targeted)
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "No replayed solution targets " + request.target.isa};
                if(!solves)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "No replayed solution solves this problem"};
                std::string summary = "The mock replayed " + std::to_string(solutions.size())
                                      + (solutions.size() == 1 ? " bundle" : " bundles");
                if(request.prediction)
                    summary += " for " + std::to_string(request.prediction->ranked.size())
                               + " ranked candidates";
                return {Status::Code::Success, Stage::Generate, std::move(summary)};
            }

        private:
            Options::Fault             m_fault;
            std::string                m_record;
            hipblaslt_jit::BackendInfo m_info;
            std::vector<Replayed>      m_replayed;
        };
    }

    std::shared_ptr<const hipblaslt_jit::Backend> makeBackend(const Options& options)
    {
        return std::make_shared<const MockBackend>(options);
    }

    hipblasStatus_t
        createBackend(const Options& options, Backend& backend, Diagnostics& diagnostics)
    {
        backend     = {};
        diagnostics = {"mock", ""};
        try
        {
            const bool predicted = !options.contract.empty();
            backend              = detail::BackendAccess::make(
                std::make_shared<const hipblaslt_jit::Jit>(hipblaslt_jit::Jit::Components{
                    makeBackend(options),
                    predicted ? hipblaslt_jit::makeOrigamiPredictor() : nullptr,
                    predicted ? hipblaslt_jit::makeCatalogKnowledge() : nullptr,
                    hipblaslt_jit::makeComgrBuilder(),
                    hipblaslt_jit::makeTensileLoader(),
                    nullptr}));
            return HIPBLAS_STATUS_SUCCESS;
        }
        catch(const std::bad_alloc&)
        {
            diagnostics.message = "Cannot allocate mock backend";
            return HIPBLAS_STATUS_ALLOC_FAILED;
        }
        catch(const std::exception& e)
        {
            diagnostics.message = e.what();
            return HIPBLAS_STATUS_INVALID_VALUE;
        }
    }
}
