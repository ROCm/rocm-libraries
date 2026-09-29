// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include "hipblaslt-jit-tensilelite-artifacts.hpp"
#include <Tensile/Tensile.hpp>
#include <stdexcept>

namespace hipblaslt_jit
{
    namespace
    {
        using Master = TensileLite::MasterSolutionLibrary<TensileLite::ContractionProblemGemm>;

        void require(bool condition, const std::string& message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

        void checkHip(hipError_t status, const char* operation)
        {
            if(status != hipSuccess)
                throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
        }

        class TensileLoader final : public SolutionLoader
        {
        public:
            Status support(const BuiltSolution&    built,
                           const OperationRequest& request,
                           const DeviceTarget&     target,
                           size_t                  workspaceLimit) const override
            {
                const auto  bundle    = parse(built, target);
                size_t      workspace = 0;
                Diagnostics diagnostics;
                const auto  status
                    = bundle->support(request, workspaceLimit, workspace, diagnostics);
                if(status == HIPBLAS_STATUS_SUCCESS)
                    return {};
                return {status == HIPBLAS_STATUS_NOT_SUPPORTED ? Status::Code::NotSupported
                                                               : Status::Code::Failed,
                        Stage::Support,
                        diagnostics.message};
            }

            Status load(const BuiltSolution& built,
                        const OperationRequest&,
                        const DeviceTarget& target,
                        size_t,
                        std::shared_ptr<const KernelBundle>& out) const override
            {
                out.reset();
                auto bundle     = parse(built, target);
                bundle->adapter = std::make_shared<TensileLite::hip::SolutionAdapter>(false, "jit-gemm");
                checkHip(bundle->adapter->loadCodeObjectBytes(built.object.bytes),
                         "Load generated code object");
                for(const auto& helper : built.helpers)
                    checkHip(bundle->adapter->loadCodeObjectBytes(helper.bytes),
                             "Load generated code object");
                checkHip(bundle->adapter->initKernel(bundle->kernel),
                         "Resolve generated kernel symbol");
                out = std::move(bundle);
                return {};
            }

        private:
            using Diagnostics = TensileBundle::Diagnostics;

            static std::shared_ptr<TensileBundle> parse(const BuiltSolution& built,
                                                        const DeviceTarget&  target)
            {
                require(target.hardware != nullptr,
                        "A device target with Tensile hardware is required");
                auto bundle      = std::make_shared<TensileBundle>();
                bundle->hardware = target.hardware;
                bundle->kernel   = built.generated.kernelName;
                bundle->library  = std::dynamic_pointer_cast<Master>(
                    TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                        built.generated.entry));
                require(bundle->library && bundle->library->solutions.size() == 1
                            && bundle->library->solutions.count(0),
                        "Expected a non-lazy library containing only local solution 0");
                const auto solution = bundle->library->solutions.at(0);
                require(solution && solution->index == 0 && solution->kernelName == bundle->kernel,
                        "Library solution identity does not match the generated kernel");
                return bundle;
            }
        };
    }

    GeneratedSolution readTensileSourceBundle(const std::filesystem::path& bundle)
    {
        namespace artifacts = hipblaslt_ext::experimental::jit::tensilelite::detail::artifacts;
        auto              sources = artifacts::readSourceBundle(bundle);
        GeneratedSolution result;
        result.entry       = std::move(sources.library);
        const auto library = std::dynamic_pointer_cast<Master>(
            TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(result.entry));
        require(library && library->solutions.size() == 1 && library->solutions.count(0)
                    && library->solutions.at(0),
                "Expected a non-lazy library containing only local solution 0");
        result.kernelName = library->solutions.at(0)->kernelName;
        for(auto& file : sources.assembly)
            result.units.push_back({BuildUnit::Role::Main,
                                    std::move(file.name),
                                    std::move(file.bytes),
                                    BuildUnit::Kind::Assembly,
                                    {}});
        std::vector<IncludeFile> includes;
        for(auto& header : sources.headers)
            includes.push_back({std::move(header.name), std::move(header.bytes)});
        for(auto& file : sources.helpers)
            result.units.push_back({BuildUnit::Role::Helper,
                                    std::move(file.name),
                                    std::move(file.bytes),
                                    BuildUnit::Kind::Hip,
                                    includes});
        return result;
    }

    std::shared_ptr<const SolutionLoader> makeTensileLoader()
    {
        return std::make_shared<const TensileLoader>();
    }
}
