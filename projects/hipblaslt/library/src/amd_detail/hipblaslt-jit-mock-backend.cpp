// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-mock.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include <Tensile/Tensile.hpp>
#include <algorithm>
#include <cstdlib>
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

        class MockBackend final : public hipblaslt_jit::Backend
        {
        public:
            explicit MockBackend(const Options& options)
                : m_fault(options.fault)
                , m_info{"mock", "mock", ""}
                , m_solution(hipblaslt_jit::readTensileSourceBundle(fs::u8path(options.replay)))
                , m_library(std::dynamic_pointer_cast<Master>(
                      TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                          m_solution.entry)))
            {
                const auto text = [](const std::vector<uint8_t>& bytes) {
                    return std::string_view(reinterpret_cast<const char*>(bytes.data()),
                                            bytes.size());
                };
                hipblaslt_jit::Fnv1a version;
                version.add(text(m_solution.entry)).add(m_solution.kernelName);
                for(const auto& unit : m_solution.units)
                {
                    version.add(unit.name).add(text(unit.bytes));
                    for(const auto& include : unit.includes)
                        version.add(include.name).add(text(include.bytes));
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
                if(m_fault == Options::Fault::Generate)
                    return {Status::Code::Failed, Stage::Generate, "Mock generation fault"};
                const auto* gemm = dynamic_cast<const detail::GemmRequest*>(&request.request);
                if(!gemm)
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "The mock backend replays a GEMM solution"};
                const auto& solution = *m_library->solutions.at(0);
                if(!request.target.hardware
                   || !(*solution.hardwarePredicate)(*request.target.hardware))
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "The replayed solution does not target " + request.target.isa};
                const auto problem = hipblaslt_jit::lowerForJit(*gemm);
                if(!(*solution.problemPredicate)(problem))
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "The replayed solution does not solve this problem"};
                const auto& excluded = request.excludeKernels;
                if(std::find(excluded.begin(), excluded.end(), m_solution.kernelName)
                   != excluded.end())
                    return {};
                solutions.push_back(m_solution);
                if(m_fault == Options::Fault::Build)
                {
                    const std::string invalid = "s_not_an_instruction\n";
                    for(auto& unit : solutions.back().units)
                        if(unit.role == Role::Main)
                            unit.bytes.assign(invalid.begin(), invalid.end());
                }
                return {};
            }

        private:
            Options::Fault                   m_fault;
            hipblaslt_jit::BackendInfo       m_info;
            hipblaslt_jit::GeneratedSolution m_solution;
            std::shared_ptr<Master>          m_library;
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
            backend = detail::BackendAccess::make(
                std::make_shared<const hipblaslt_jit::Jit>(hipblaslt_jit::Jit::Components{
                    makeBackend(options),
                    nullptr,
                    nullptr,
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
