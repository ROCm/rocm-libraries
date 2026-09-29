// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-mock.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-tensilelite-artifacts.hpp"
#include <Tensile/Tensile.hpp>
#include <algorithm>
#include <cstdlib>
#include <set>

namespace hipblaslt_ext::experimental::jit::mock
{
    namespace
    {
        namespace fs      = std::filesystem;
        namespace reading = tensilelite::detail::artifacts;
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
            {
                const auto               bundle = fs::canonical(fs::u8path(options.replay));
                std::vector<std::string> objects;
                const auto tree = reading::readEnvelope(bundle / "loader.bin", objects);
                m_isa           = reading::field(tree, "architecture.resolved");
                m_isa           = m_isa.substr(0, m_isa.find(':'));
                m_solution.kernelName = reading::field(tree, "main_kernel.name");
                m_solution.entry      = reading::readLibrary(
                    reading::artifact(bundle, reading::field(tree, "library.path")));
                m_library = std::dynamic_pointer_cast<Master>(
                    TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                        m_solution.entry));
                reading::require(m_library && m_library->solutions.size() == 1
                                     && m_library->solutions.count(0),
                                 "The replay bundle must hold exactly one solution");
                const auto main
                    = reading::artifact(bundle, reading::field(tree, "main_kernel.code_object"));
                std::set<fs::path> paths;
                for(const auto& object : objects)
                    paths.insert(reading::artifact(bundle, object));
                paths.insert(main);
                for(const auto& path : paths)
                    m_solution.units.push_back({path == main ? Role::Main : Role::Helper,
                                                path.filename().u8string(),
                                                reading::readArtifact(path)});
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
                if(request.target.isa != m_isa)
                    return {Status::Code::TargetMismatch,
                            Stage::Configure,
                            "The replayed solution targets " + m_isa};
                const auto problem = hipblaslt_jit::lowerForJit(*gemm);
                if(!(*m_library->solutions.at(0)->problemPredicate)(problem))
                    return {Status::Code::NotSupported,
                            Stage::Generate,
                            "The replayed solution does not solve this problem"};
                const auto& excluded = request.excludeKernels;
                if(std::find(excluded.begin(), excluded.end(), m_solution.kernelName)
                   != excluded.end())
                    return {};
                solutions.push_back(m_solution);
                if(m_fault == Options::Fault::Build)
                    for(auto& unit : solutions.back().units)
                        if(unit.role == Role::Main)
                            unit.bytes.resize(4);
                return {};
            }

        private:
            Options::Fault                   m_fault;
            hipblaslt_jit::BackendInfo       m_info;
            std::string                      m_isa;
            hipblaslt_jit::GeneratedSolution m_solution;
            std::shared_ptr<Master>          m_library;
        };
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
                    std::make_shared<const MockBackend>(options),
                    nullptr,
                    nullptr,
                    hipblaslt_jit::makePrebuiltBuilder(),
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
