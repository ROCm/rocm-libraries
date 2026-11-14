// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-hipkittens.hpp"
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-component.hpp"
#include "hipblaslt-jit-gemm-internal.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/Tensile.hpp>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace hipblaslt_ext::experimental::jit::hipkittens
{
    namespace detail
    {
        const std::vector<Kernel>& kernels()
        {
            static const std::vector<Kernel> value = [] {
                const auto kernel = [](const std::string&       name,
                                       const std::string&       kernelName,
                                       rocisa::DataType         type,
                                       std::vector<std::string> hipFlags) {
                    Kernel made;
                    made.name       = name + ".hip";
                    made.kind       = hipblaslt_jit::Artifact::Kind::Hip;
                    made.bytes      = resources().source;
                    made.processor  = hipblaslt_jit::Processor::gfx950;
                    made.kernelName = kernelName;
                    made.type       = type;
                    made.hipFlags   = std::move(hipFlags);
                    return made;
                };
                // Both compile gemm_tn_256x256x64_gfx950.hip; HK_FP16 selects the FP16
                // kernel name in that template.
                return std::vector<Kernel>{
                    kernel("gemm_bf16_tn_256x256x64_gfx950",
                           "HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5",
                           rocisa::DataType::BFloat16,
                           {"-std=c++20", "-DKITTENS_CDNA4", "-w"}),
                    kernel("gemm_f16_tn_256x256x64_gfx950",
                           "HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5",
                           rocisa::DataType::Half,
                           {"-std=c++20", "-DKITTENS_CDNA4", "-DHK_FP16", "-w"})};
            }();
            return value;
        }

        namespace
        {
            template <typename P>
            std::shared_ptr<P> predicate(decltype(P::value) value)
            {
                auto made   = std::make_shared<P>();
                made->value = std::move(value);
                return made;
            }

            template <typename P>
            std::shared_ptr<P> sizePredicate(int64_t index, size_t value)
            {
                auto made   = std::make_shared<P>();
                made->index = index;
                made->value = value;
                return made;
            }
        }

        // Solves K > 0 and a multiple of 128, with M and N multiples of the 256
        // tile: odd numbers of 64-wide K tiles give wrong results, and the kernel
        // has no tail handling for M or N. The custom kernel launches with the
        // HIP kernel's arguments, threads and macro tile; sizeMapping is the rest
        // of what TensileLite reads from a solution.
        std::shared_ptr<TensileLite::ContractionSolution> Kernel::solution() const
        {
            using namespace TensileLite;
            using namespace TensileLite::Predicates::Contraction;
            using rocisa::DataType;
            using Problem = ContractionProblemGemm;
            // Every column, so a buffer-limit check accepts an offset anywhere in the
            // matrix: the HIP kernel addresses each one with a single buffer descriptor.
            constexpr size_t allColumns = 4294967295;

            auto solution          = std::make_shared<ContractionSolution>();
            solution->kernelName   = kernelName;
            solution->solutionName = kernelName;

            auto& problemType                      = solution->problemType;
            problemType.operationIdentifier        = "Contraction_l_Alik_Bljk_Cijk_Dijk";
            problemType.transA                     = true;
            problemType.transB                     = false;
            problemType.aType                      = type;
            problemType.bType                      = type;
            problemType.cType                      = type;
            problemType.dType                      = type;
            problemType.eType                      = type;
            problemType.computeInputTypeA          = type;
            problemType.computeInputTypeB          = type;
            problemType.computeType                = DataType::Float;
            problemType.highPrecisionAccumulate    = true;
            problemType.biasSrcWhiteList           = {3};
            problemType.biasDataTypeWhiteList      = {DataType::Float, type};
            problemType.supportDeviceUserArguments = true;

            const auto arg = [](CustomArgType type, CustomArgSemantic semantic) {
                return CustomArgDefinition{type, semantic};
            };
            auto& custom = solution->customKernel;
            custom.name  = kernelName;
            custom.args  = {arg(CustomArgType::address, CustomArgSemantic::AddressA),
                            arg(CustomArgType::address, CustomArgSemantic::AddressB),
                            arg(CustomArgType::address, CustomArgSemantic::AddressC),
                            arg(CustomArgType::address, CustomArgSemantic::AddressD),
                            arg(CustomArgType::uint32, CustomArgSemantic::SizeFree0),
                            arg(CustomArgType::uint32, CustomArgSemantic::SizeFree1),
                            arg(CustomArgType::uint32, CustomArgSemantic::SizeSum),
                            arg(CustomArgType::float32, CustomArgSemantic::Alpha),
                            arg(CustomArgType::float32, CustomArgSemantic::Beta),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideA0),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideB0),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideC0),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideD0),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideA1),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideB1),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideC1),
                            arg(CustomArgType::uint32, CustomArgSemantic::StrideD1)};
            // Each work-group reduces its whole tile over K from the first K tile
            // up, in the same instruction order in every wave, with no split K,
            // atomics or stagger, so every row of D sums in the same order.
            custom.uniformSummationOrder = true;
            custom.macrotile             = {256, 256, 64};
            custom.threads               = {512, 1, 1};
            custom.grid = {CustomGridSize::TilesXY, CustomGridSize::One, CustomGridSize::Batch};

            auto& sizes                    = solution->sizeMapping;
            sizes.waveNum                  = 8;
            sizes.workGroupSize            = {16, 16, 1};
            sizes.threadTile               = {4, 4, 1};
            sizes.macroTile                = {256, 256, 1};
            sizes.matrixInstruction        = {16, 16, 32, 1};
            sizes.waveGroup                = {2, 4};
            sizes.depthU                   = 64;
            sizes.globalSplitUPGR          = 16;
            sizes.globalSplitU             = 1;
            sizes.workGroupMapping         = 8;
            sizes.magicDivAlg              = 2;
            sizes.globalAccumulation       = 2;
            sizes.workGroupMappingXCC      = 1;
            sizes.workGroupMappingXCCGroup = -1;
            sizes.CUOccupancy              = -1;
            sizes.PrefetchGlobalRead       = 1;
            sizes.synchronizerSizePerWG    = 32768;
            sizes.customMainLoopScheduling = -1;
            sizes.NumLoadsCoalescedA       = 1;
            sizes.NumLoadsCoalescedB       = 1;
            sizes.VectorWidthA             = -1;
            sizes.VectorWidthB             = -1;

            solution->problemPredicate = std::make_shared<Predicates::And<Problem>>(
                std::vector<std::shared_ptr<Predicates::Predicate<Problem>>>{
                    sizePredicate<BoundSizeMultiple>(-1, 128),
                    sizePredicate<Free0SizeMultiple>(0, 256),
                    sizePredicate<Free1SizeMultiple>(0, 256),
                    predicate<OperationIdentifierEqual>(problemType.operationIdentifier),
                    predicate<BiasDataTypeWhiteList>(problemType.biasDataTypeWhiteList),
                    predicate<GateResidualDataTypeWhiteList>({}),
                    predicate<BiasSrcWhiteList>(problemType.biasSrcWhiteList),
                    predicate<AmaxDCheck>(false),
                    predicate<TypesEqual>({type, type, type, type, type, type}),
                    predicate<HighPrecisionAccumulateEqual>(true),
                    predicate<ActivationCheck>(ActivationType::None),
                    predicate<ActivationComputeTypeEqual>(DataType::Float),
                    predicate<ActivationNoGuardEqual>(false),
                    predicate<UseGradientEqual>(false),
                    predicate<UseBiasCheck>(0),
                    predicate<UseEEqual>(false),
                    predicate<UseGateResidualCheck>(false),
                    predicate<DataTypeEEqual>(type),
                    predicate<StridedBatchedEqual>(true),
                    predicate<GroupedGemmEqual>(false),
                    predicate<UseScaleABCheck>(""),
                    predicate<UseScaleCDCheck>(false),
                    predicate<UseScaleAlphaVecCheck>(0),
                    predicate<Sparse>(0),
                    predicate<F32XdlMathOpEqual>(DataType::Float),
                    predicate<SupportDeviceUserArguments>(true),
                    predicate<SwizzleTensorA>(false),
                    predicate<SwizzleTensorB>(false),
                    predicate<FusedGemmA2A>(false),
                    predicate<MXBlockA>(0),
                    predicate<MXBlockB>(0),
                    predicate<LeadingFree0SizesGreaterOrEqual>(1),
                    predicate<LeadingFree1SizesGreaterOrEqual>(1),
                    predicate<KernelLanguageCompatible>(KernelLanguage::Assembly),
                    predicate<BufferLoadOffsetLimitCheck>({0, 0, allColumns, allColumns}),
                    predicate<BufferLoadOffsetLimitCheck_Beta>(allColumns),
                    predicate<BufferStoreOffsetLimitCheck>(allColumns),
                    predicate<GlobalSplitUCheckMinK>({32, 1}),
                    predicate<WorkgroupMappingXCCCheck>({1, -1}),
                    sizePredicate<SizeGreaterThan>(3, 0)});
            solution->hardwarePredicate
                = std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(
                    std::make_shared<Predicates::GPU::ProcessorEqual>(processor));
            solution->taskPredicate = std::make_shared<LaunchLimits>();
            return solution;
        }
    }

    namespace
    {
        namespace fs = std::filesystem;
        using hipblaslt_jit::Stage;
        using hipblaslt_jit::Status;

        // The kernels address each tensor with 32-bit byte offsets.
        constexpr size_t tensorLimit = size_t(1) << 32;
        // Increment when a HipKittens solution can change.
        constexpr uint32_t hipkittensVersion = 1;

        Status unavailable(const std::string& cause)
        {
            return {Status::Code::Failed,
                    Stage::Configure,
                    "JIT backend HipKittens not available: " + cause};
        }

        Status findHipHeaders()
        {
            std::error_code error;
            const auto      rocm = hipblaslt_jit::code_object::rocmPath();
            const auto      hip  = fs::u8path(rocm) / "include" / "hip" / "hip_runtime.h";
            if(rocm.empty() || !fs::is_regular_file(hip, error))
                return unavailable("HIP headers not found at " + hip.u8string() + "; set HIP_PATH");
            return {};
        }

        using Candidate
            = std::pair<const detail::Kernel*, std::shared_ptr<TensileLite::ContractionSolution>>;

        // The target's kernels whose solution solves the request, each with that
        // solution, or why none can.
        Status candidates(const hipblaslt_jit::GemmRequest&  request,
                          const hipblaslt_jit::DeviceTarget& target,
                          std::vector<Candidate>&            found)
        {
            found.clear();
            const auto& kernels = detail::kernels();
            if(std::none_of(kernels.begin(), kernels.end(), [&](const detail::Kernel& kernel) {
                   return kernel.processor == target.processor;
               }))
                return {Status::Code::TargetMismatch,
                        Stage::Configure,
                        "HipKittens has no kernel for " + target.processorName()};
            const auto problem = hipblaslt_jit::RocblasltToTensileContractionProblemGemm(request);
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
            for(const auto& kernel : kernels)
            {
                if(kernel.processor != target.processor)
                    continue;
                auto solution = kernel.solution();
                if((*solution->problemPredicate)(problem))
                    found.emplace_back(&kernel, std::move(solution));
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
            HipKittensBackend()
                : m_info{"hipkittens", hipkittensVersion}
            {
            }

            const hipblaslt_jit::BackendInfo& info() const noexcept override
            {
                return m_info;
            }

            Status generate(const hipblaslt_jit::GenerationRequest&       request,
                            std::vector<hipblaslt_jit::SourceCodeBundle>& bundles) const override
            {
                bundles.clear();
                std::vector<Candidate> found;
                if(auto status = candidates(request.request, request.target, found); !status.ok())
                    return status;
                for(auto& [kernel, solution] : found)
                {
                    if(bundles.size() >= request.count
                       || hipblaslt_jit::excludedKernel(request.excludeKernels, kernel->kernelName))
                        continue;
                    hipblaslt_jit::SourceCodeBundle bundle;
                    bundle.mainKernel        = *kernel;
                    bundle.includes          = detail::resources().headers;
                    bundle.hipFlags          = kernel->hipFlags;
                    bundle.target            = request.target;
                    bundle.codeObjectVersion = request.codeObjectVersion;
                    bundle.solutions         = {std::move(solution)};
                    bundles.push_back(std::move(bundle));
                }
                return {};
            }

        private:
            hipblaslt_jit::BackendInfo m_info;
        };
    }

    hipblasStatus_t createBackend(std::shared_ptr<const hipblaslt_jit::Jit>& jit,
                                  Diagnostics&                               diagnostics)
    {
        jit         = {};
        diagnostics = {};
        try
        {
            if(auto status = findHipHeaders(); !status.ok())
            {
                diagnostics.message = status.message;
                return HIPBLAS_STATUS_INVALID_VALUE;
            }
            jit = hipblaslt_jit::makeJit(std::make_shared<const HipKittensBackend>());
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

namespace hipblaslt_jit
{
    // The process's HipKittens backend, created on first use, or null after one
    // warning when it is not available.
    std::shared_ptr<const Jit> makeHipKittensJit()
    {
        static const std::shared_ptr<const Jit> process = [] {
            std::shared_ptr<const Jit>                    jit;
            hipblaslt_ext::experimental::jit::Diagnostics diagnostics;
            if(hipblaslt_ext::experimental::jit::hipkittens::createBackend(jit, diagnostics)
               != HIPBLAS_STATUS_SUCCESS)
            {
                std::cerr << "hipblaslt warning: " << diagnostics.message << std::endl;
                return std::shared_ptr<const Jit>();
            }
            return jit;
        }();
        return process;
    }
}
