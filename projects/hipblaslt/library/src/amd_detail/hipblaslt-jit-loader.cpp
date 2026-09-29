// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-loader.hpp"
#include <Tensile/Tensile.hpp>
#include <cstring>
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

        class PrebuiltBuilder final : public CodeObjectBuilder
        {
        public:
            Status build(const GeneratedSolution& solution,
                         const GenerationRequest&,
                         BuiltSolution& built) const override
            {
                built           = {};
                built.generated = solution;
                size_t mains    = 0;
                for(const auto& unit : solution.units)
                {
                    // An ELF64 header is 64 bytes and starts with the ELF magic.
                    if(unit.bytes.size() < 64 || std::memcmp(unit.bytes.data(), "\x7f" "ELF", 4))
                        return {Status::Code::Failed,
                                Stage::Build,
                                "Code object " + unit.name + " is not an ELF file"};
                    if(unit.role == BuildUnit::Role::Main)
                    {
                        built.object.bytes = unit.bytes;
                        ++mains;
                    }
                    else
                        built.helpers.push_back({unit.bytes});
                }
                if(mains != 1)
                    return {Status::Code::Failed,
                            Stage::Build,
                            "A generated solution needs exactly one main code object"};
                return {};
            }
        };

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

    std::shared_ptr<const CodeObjectBuilder> makePrebuiltBuilder()
    {
        return std::make_shared<const PrebuiltBuilder>();
    }

    std::shared_ptr<const SolutionLoader> makeTensileLoader()
    {
        return std::make_shared<const TensileLoader>();
    }
}
