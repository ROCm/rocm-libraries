/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <filesystem>

#ifdef ROCROLLER_USE_HIP
#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#endif /* ROCROLLER_USE_HIP */

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Operations/CommandArgument_fwd.hpp>
#include <rocRoller/Operations/OperationTag.hpp>
#include <rocRoller/Serialization/Enum.hpp>
#include <rocRoller/Serialization/GPUArchitecture.hpp>
#include <rocRoller/Serialization/YAML.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>
#include <rocRoller/Utilities/Timer.hpp>
#include <rocRoller/Utilities/Utils.hpp>
#include <rocRoller/Utilities/Version.hpp>

#include <common/Utilities.hpp>
#include <common/mxDataGen.hpp>

#include "include/DataParallelGEMMSolution.hpp"
#include "include/GEMMParameters.hpp"
#include "include/StreamKGEMMSolution.hpp"

#include <CLI/CLI.hpp>

using namespace rocRoller;

const std::string clientName = "GEMMv00";

enum ReturnCodes : int
{
    OK                         = 0,
    GenerateFailure            = 1,
    CorrectnessFailure         = 2,
    SolutionNotSupportedOnArch = 3
};

template <typename IO>
struct rocRoller::Serialization::
    MappingTraits<Client::GEMMClient::Result, IO, rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, Client::GEMMClient::Result& result)
    {
        iot::mapRequired(io, "client", clientName);
        iot::mapRequired(io, "device", result.benchmarkResults.runParams.device);
        iot::mapRequired(io, "M", result.problemParams.m);
        iot::mapRequired(io, "N", result.problemParams.n);
        iot::mapRequired(io, "K", result.problemParams.k);
        iot::mapRequired(io, "alpha", result.problemParams.alpha);
        iot::mapRequired(io, "beta", result.problemParams.beta);
        iot::mapRequired(io, "trans_A", Client::GEMMClient::toString(result.solutionParams.transA));
        iot::mapRequired(io, "trans_B", Client::GEMMClient::toString(result.solutionParams.transB));

        iot::mapRequired(io, "type_A", result.solutionParams.typeA);
        iot::mapRequired(io, "type_B", result.solutionParams.typeB);
        iot::mapRequired(io, "type_C", result.solutionParams.typeC);
        iot::mapRequired(io, "type_D", result.solutionParams.typeD);
        iot::mapRequired(io, "type_acc", result.solutionParams.typeAcc);

        iot::mapRequired(io, "scale_A", result.solutionParams.scaleA);
        iot::mapRequired(io, "scale_B", result.solutionParams.scaleB);
        iot::mapRequired(io, "scaleBlockSize", result.solutionParams.scaleBlockSize);
        iot::mapRequired(io, "loadLDSScale_A", result.solutionParams.loadLDSScaleA);
        iot::mapRequired(io, "loadLDSScale_B", result.solutionParams.loadLDSScaleB);
        iot::mapRequired(io, "swizzleScale", result.solutionParams.swizzleScale);
        iot::mapRequired(io, "prefetchScale", result.solutionParams.prefetchScale);

        iot::mapRequired(io, "mac_m", result.solutionParams.macM);
        iot::mapRequired(io, "mac_n", result.solutionParams.macN);
        iot::mapRequired(io, "mac_k", result.solutionParams.macK);
        iot::mapRequired(io, "wave_m", result.solutionParams.waveM);
        iot::mapRequired(io, "wave_n", result.solutionParams.waveN);
        iot::mapRequired(io, "wave_k", result.solutionParams.waveK);
        iot::mapRequired(io, "wave_b", result.solutionParams.waveB);
        iot::mapRequired(io, "workgroup_size_x", result.solutionParams.workgroupSizeX);
        iot::mapRequired(io, "workgroup_size_y", result.solutionParams.workgroupSizeY);
        iot::mapRequired(io, "workgroupMapping", result.solutionParams.workgroupMapping);
        iot::mapRequired(io, "workgroupRemapXCC", result.solutionParams.workgroupRemapXCC);
        iot::mapRequired(
            io, "workgroupRemapXCCValue", result.solutionParams.workgroupRemapXCCValue);
        iot::mapRequired(io, "unroll_x", result.solutionParams.unrollX);
        iot::mapRequired(io, "unroll_y", result.solutionParams.unrollY);
        iot::mapRequired(io, "loadLDS_A", result.solutionParams.loadLDSA);
        iot::mapRequired(io, "loadLDS_B", result.solutionParams.loadLDSB);
        iot::mapRequired(io, "storeLDS_D", result.solutionParams.storeLDSD);
        iot::mapRequired(io, "direct2LDS_A", result.solutionParams.direct2LDSA);
        iot::mapRequired(io, "direct2LDS_B", result.solutionParams.direct2LDSB);
        iot::mapRequired(io, "prefetch", result.solutionParams.prefetch);
        iot::mapRequired(io, "prefetchInFlight", result.solutionParams.prefetchInFlight);
        iot::mapRequired(io, "prefetchLDSFactor", result.solutionParams.prefetchLDSFactor);
        iot::mapRequired(io, "prefetchMixMemOps", result.solutionParams.prefetchMixMemOps);
        iot::mapRequired(io, "betaInFma", result.solutionParams.betaInFma);
        iot::mapRequired(io, "scheduler", result.solutionParams.scheduler);

        iot::mapRequired(io, "streamK", result.solutionParams.streamK);
        iot::mapRequired(io, "numWGs", result.benchmarkResults.runParams.numWGs);
        iot::mapRequired(io, "streamKTwoTile", result.solutionParams.streamKTwoTile);

        iot::mapRequired(io, "numWarmUp", result.benchmarkResults.runParams.numWarmUp);
        iot::mapRequired(io, "numOuter", result.benchmarkResults.runParams.numOuter);
        iot::mapRequired(io, "numInner", result.benchmarkResults.runParams.numInner);

        iot::mapRequired(io, "kernelGenerate", result.benchmarkResults.kernelGenerate);
        iot::mapRequired(io, "kernelAssemble", result.benchmarkResults.kernelAssemble);
        iot::mapRequired(io, "kernelExecute", result.benchmarkResults.kernelExecute);

        iot::mapRequired(io, "checked", result.benchmarkResults.checked);
        iot::mapRequired(io, "correct", result.benchmarkResults.correct);
        iot::mapRequired(io, "rnorm", result.benchmarkResults.rnorm);
    }

    static void mapping(IO& io, Client::GEMMClient::Result& result, EmptyContext& ctx)
    {
        mapping(io, result);
    }
};

template <typename IO>
struct rocRoller::Serialization::MappingTraits<Client::GEMMClient::SolutionParameters,
                                               IO,
                                               rocRoller::Serialization::EmptyContext>
{
    static const bool flow = false;
    using iot              = IOTraits<IO>;

    static void mapping(IO& io, Client::GEMMClient::SolutionParameters& params)
    {
        iot::mapRequired(io, "architecture", params.architecture);

        iot::mapRequired(io, "mac_m", params.macM);
        iot::mapRequired(io, "mac_n", params.macN);
        iot::mapRequired(io, "mac_k", params.macK);
        iot::mapRequired(io, "wave_m", params.waveM);
        iot::mapRequired(io, "wave_n", params.waveN);
        iot::mapRequired(io, "wave_k", params.waveK);
        iot::mapRequired(io, "wave_b", params.waveB);
        iot::mapRequired(io, "workgroup_size_x", params.workgroupSizeX);
        iot::mapRequired(io, "workgroup_size_y", params.workgroupSizeY);
        iot::mapRequired(io, "workgroupMapping", params.workgroupMapping);
        iot::mapRequired(io, "workgroupRemapXCC", params.workgroupRemapXCC);
        iot::mapRequired(io, "workgroupRemapXCCValue", params.workgroupRemapXCCValue);
        iot::mapRequired(io, "unroll_x", params.unrollX);
        iot::mapRequired(io, "unroll_y", params.unrollY);
        iot::mapRequired(io, "loadLDS_A", params.loadLDSA);
        iot::mapRequired(io, "loadLDS_B", params.loadLDSB);
        iot::mapRequired(io, "storeLDS_D", params.storeLDSD);
        iot::mapRequired(io, "direct2LDS_A", params.direct2LDSA);
        iot::mapRequired(io, "direct2LDS_B", params.direct2LDSB);
        iot::mapRequired(io, "prefetch", params.prefetch);
        iot::mapRequired(io, "prefetchInFlight", params.prefetchInFlight);
        iot::mapRequired(io, "prefetchLDSFactor", params.prefetchLDSFactor);
        iot::mapRequired(io, "prefetchMixMemOps", params.prefetchMixMemOps);
        iot::mapRequired(io, "betaInFma", params.betaInFma);
        iot::mapRequired(io, "scheduler", params.scheduler);
        iot::mapRequired(io, "matchMemoryAccess", params.matchMemoryAccess);

        iot::mapRequired(io, "trans_A", params.transA);
        iot::mapRequired(io, "trans_B", params.transB);

        iot::mapRequired(io, "type_A", params.typeA);
        iot::mapRequired(io, "type_B", params.typeB);
        iot::mapRequired(io, "type_C", params.typeC);
        iot::mapRequired(io, "type_D", params.typeD);
        iot::mapRequired(io, "type_acc", params.typeAcc);

        iot::mapRequired(io, "scale_A", params.scaleA);
        iot::mapRequired(io, "scale_B", params.scaleB);
        iot::mapRequired(io, "scaleBlockSize", params.scaleBlockSize);
        iot::mapRequired(io, "loadScaleLDS_A", params.loadLDSScaleA);
        iot::mapRequired(io, "loadScaleLDS_B", params.loadLDSScaleB);
        iot::mapRequired(io, "swizzleScale", params.swizzleScale);
        iot::mapRequired(io, "prefetchScale", params.prefetchScale);

        iot::mapRequired(io, "streamK", params.streamK);
        iot::mapRequired(io, "streamKTwoTile", params.streamKTwoTile);

        iot::mapOptional(io, "version", params.version);
    }

    static void mapping(IO& io, Client::GEMMClient::SolutionParameters& params, EmptyContext& ctx)
    {
        mapping(io, params);
    }
};

namespace rocRoller::Client::GEMMClient
{
    using GEMMSolutionPtr = std::shared_ptr<Client::GEMMClient::GEMMSolution>;

    template <typename A, typename B, typename C, typename D>
    std::pair<bool, double>
        validate(std::vector<typename PackedTypeOf<A>::type> const&      h_A,
                 std::vector<typename PackedTypeOf<B>::type> const&      h_B,
                 std::vector<C> const&                                   h_C,
                 std::vector<D> const&                                   h_D,
                 std::vector<uint8_t> const&                             h_scaleA,
                 std::vector<uint8_t> const&                             h_scaleB,
                 rocRoller::Client::GEMMClient::ProblemParameters const& problemParams,
                 GPUArchitecture const&                                  arch)
    {
        using namespace rocRoller::Client::GEMMClient;

        // Host result
        std::vector<D> h_result(problemParams.m * problemParams.n, static_cast<D>(0.0));

        if(!h_scaleA.empty() || !h_scaleB.empty())
        {
            rocRoller::ScaledCPUMM(h_result,
                                   h_C,
                                   h_A,
                                   h_B,
                                   h_scaleA,
                                   h_scaleB,
                                   problemParams.m,
                                   problemParams.n,
                                   problemParams.k,
                                   problemParams.alpha,
                                   problemParams.beta,
                                   problemParams.transA == TransposeType::T,
                                   problemParams.transB == TransposeType::T);
        }
        else
        {
            CPUMM(h_result,
                  h_C,
                  h_A,
                  h_B,
                  problemParams.m,
                  problemParams.n,
                  problemParams.k,
                  problemParams.alpha,
                  problemParams.beta,
                  problemParams.transA == TransposeType::T,
                  problemParams.transB == TransposeType::T);
        }

        auto tol = gemmAcceptableError<A, B, D>(
            problemParams.m, problemParams.n, problemParams.k, arch.target());
        auto res = compare(h_D, h_result, tol);

        Log::debug(res.message());

        std::cout << "Result: " << (res.ok ? "Correct" : "Incorrect") << std::endl;
        std::cout << "RNorm: " << res.relativeNormL2 << std::endl;
        if(!res.ok)
        {
            std::cerr << "WARNING: Result incorrect.  " << res.message() << std::endl;
        }
        return {res.ok, res.relativeNormL2};
    }

    // D (MxN) = alpha * A (MxK) X B (KxN) + beta * C (MxN)
    template <typename A, typename B, typename C, typename D>
    Client::BenchmarkResults GEMM(CommandPtr               command,
                                  CommandKernelPtr         commandKernel,
                                  GEMMSolutionPtr          gemm,
                                  ProblemParameters const& problemParams,
                                  RunParameters const&     runParams,
                                  GPUArchitecture const&   arch)
    {
        using namespace rocRoller::Client;
        using namespace rocRoller::Client::GEMMClient;

        // Host Data
        std::cout << "Generating input data..." << std::endl;

        TensorDescriptor descA(fromString<DataType>(problemParams.typeA),
                               {static_cast<unsigned long>(problemParams.m),
                                static_cast<unsigned long>(problemParams.k)},
                               problemParams.transA == TransposeType::T ? "T" : "N");
        TensorDescriptor descB(fromString<DataType>(problemParams.typeB),
                               {static_cast<unsigned long>(problemParams.k),
                                static_cast<unsigned long>(problemParams.n)},
                               problemParams.transB == TransposeType::T ? "T" : "N");
        TensorDescriptor descC(fromString<DataType>(problemParams.typeC),
                               {static_cast<unsigned long>(problemParams.m),
                                static_cast<unsigned long>(problemParams.n)},
                               "N");

        using PackedTypeA = typename PackedTypeOf<A>::type;
        using PackedTypeB = typename PackedTypeOf<B>::type;
        std::vector<PackedTypeA> hostA;
        std::vector<PackedTypeB> hostB;
        std::vector<C>           hostC;
        std::vector<D>           hostD(problemParams.m * problemParams.n, D{});
        std::vector<uint8_t>     hostScaleA, hostScaleB;

        auto seed = 31415u;
        DGenInput(seed,
                  hostA,
                  descA,
                  hostB,
                  descB,
                  hostC,
                  descC,
                  hostScaleA,
                  hostScaleB,
                  problemParams.scaleA == Operations::ScaleMode::Separate,
                  problemParams.scaleB == Operations::ScaleMode::Separate);

        auto deviceA = make_shared_device(hostA);
        auto deviceB = make_shared_device(hostB);
        auto deviceC = make_shared_device(hostC);
        auto deviceD = make_shared_device<D>(problemParams.m * problemParams.n, D{});

        std::shared_ptr<uint8_t> deviceScaleA, deviceScaleB;
        AssertFatal(problemParams.scaleA == Operations::ScaleMode::None
                        || problemParams.scaleA == Operations::ScaleMode::SingleScale
                        || problemParams.scaleA == Operations::ScaleMode::Separate,
                    "Scale mode not supported!",
                    ShowValue(problemParams.scaleA));
        AssertFatal(problemParams.scaleB == Operations::ScaleMode::None
                        || problemParams.scaleB == Operations::ScaleMode::SingleScale
                        || problemParams.scaleB == Operations::ScaleMode::Separate,
                    "Scale mode not supported!",
                    ShowValue(problemParams.scaleB));
        if(problemParams.scaleA == Operations::ScaleMode::Separate)
        {
            deviceScaleA = make_shared_device(hostScaleA);
        }
        if(problemParams.scaleB == Operations::ScaleMode::Separate)
        {
            deviceScaleB = make_shared_device(hostScaleB);
        }

        std::cout << "Generating lauch parameters and runtime arguments..." << std::endl;

        commandKernel->loadKernel();

        auto commandArgs = gemm->commandArguments(command, problemParams, runParams);

        auto [aTag, bTag, cTag, dTag] = gemm->getABCDTags();
        commandArgs.setArgument(aTag, ArgumentType::Value, (A*)deviceA.get());
        commandArgs.setArgument(bTag, ArgumentType::Value, (B*)deviceB.get());
        commandArgs.setArgument(cTag, ArgumentType::Value, (C*)deviceC.get());
        commandArgs.setArgument(dTag, ArgumentType::Value, (D*)deviceD.get());

        if(problemParams.scaleA == Operations::ScaleMode::Separate)
        {
            auto scaleBlockSize = problemParams.scaleBlockSize;
            AssertFatal(scaleBlockSize != -1, "scaleBlockSize must be set to scale A.");
            AssertFatal(arch.isSupportedScaleBlockSize(scaleBlockSize),
                        fmt::format("Architecture {} does not support block scaling (size: {}).",
                                    arch.target().toString(),
                                    scaleBlockSize));
            AssertFatal(problemParams.k % scaleBlockSize == 0,
                        fmt::format("K: {} must be a multiple of the scale block size: {}",
                                    problemParams.k,
                                    scaleBlockSize));
            auto dataTypeA = TypeInfo<A>::Var.dataType;
            auto descAScale
                = TensorDescriptor(dataTypeA,
                                   {static_cast<size_t>(problemParams.m),
                                    static_cast<size_t>(problemParams.k / scaleBlockSize)},
                                   problemParams.transA == TransposeType::T ? "T" : "N");
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            setCommandTensorArg(commandArgs, aScaleTag.value(), descAScale, deviceScaleA.get());
        }
        else if(problemParams.scaleA == Operations::ScaleMode::SingleScale)
        {
            auto scaleValue             = floatToScale(problemParams.scaleValueA);
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            commandArgs.setArgument(aScaleTag.value(), ArgumentType::Value, scaleValue);

            hostScaleA = {scaleValue};
        }

        if(problemParams.scaleB == Operations::ScaleMode::Separate)
        {
            auto scaleBlockSize = problemParams.scaleBlockSize;
            AssertFatal(scaleBlockSize != -1, "scaleBlockSize must be set to scale B.");
            AssertFatal(arch.isSupportedScaleBlockSize(scaleBlockSize),
                        fmt::format("Architecture {} does not support block scaling (size: {}).",
                                    arch.target().toString(),
                                    scaleBlockSize));
            AssertFatal(problemParams.k % scaleBlockSize == 0,
                        fmt::format("K: {} must be a multiple of the scale block size: {}",
                                    problemParams.k,
                                    scaleBlockSize));
            auto dataTypeB = TypeInfo<A>::Var.dataType;
            auto descBScale
                = TensorDescriptor(dataTypeB,
                                   {static_cast<size_t>(problemParams.k / scaleBlockSize),
                                    static_cast<size_t>(problemParams.n)},
                                   problemParams.transB == TransposeType::T ? "T" : "N");
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            setCommandTensorArg(commandArgs, bScaleTag.value(), descBScale, deviceScaleB.get());
        }
        else if(problemParams.scaleB == Operations::ScaleMode::SingleScale)
        {
            auto scaleValue             = floatToScale(problemParams.scaleValueB);
            auto [aScaleTag, bScaleTag] = gemm->getABScaleTags();
            commandArgs.setArgument(bScaleTag.value(), ArgumentType::Value, scaleValue);

            hostScaleB = {scaleValue};
        }

        gemm->validateRunParameters(command, problemParams, runParams, commandKernel);

        auto runtimeArgs = commandArgs.runtimeArguments();

        // Note: the lifetime of deviceScratch needs to exceed kernel executions
        std::shared_ptr<uint8_t> deviceScratch;
        {
            auto scratchSpaceRequired = commandKernel->scratchSpaceRequired(runtimeArgs);
            if(scratchSpaceRequired > 0)
            {
                deviceScratch = make_shared_device<uint8_t>(scratchSpaceRequired, 0);
                commandArgs.setArgument(
                    gemm->getScratchTag(), ArgumentType::Value, deviceScratch.get());
            }
        }

        if(runParams.visualize)
        {
            Client::visualize(command, *commandKernel, commandArgs);
        }

        std::cout << std::endl;
        std::cout << "Problem:" << std::endl;
        std::cout << problemParams << std::endl;

        std::cout << "Launching GPU kernel(s)..." << std::endl;

        BenchmarkResults result;
        result.runParams = runParams;

        // Benchmark runs
        for(int outer = 0; outer < runParams.numOuter; ++outer)
        {
            // Warmup runs
            for(int i = 0; i < runParams.numWarmUp; ++i)
            {
                commandKernel->launchKernel(runtimeArgs);
            }

            HIP_TIMER(t_kernel, "GEMM", runParams.numInner);
            for(int inner = 0; inner < runParams.numInner; ++inner)
            {
                commandKernel->launchKernel(runtimeArgs, t_kernel, inner);
            }
            HIP_SYNC(t_kernel);
            t_kernel->sleep(50);
            auto nanoseconds = t_kernel->allNanoseconds();
            result.kernelExecute.insert(
                result.kernelExecute.end(), nanoseconds.begin(), nanoseconds.end());
        }

        double totalTime = 0;
        for(auto ke : result.kernelExecute)
            totalTime += static_cast<double>(ke) / 1.e9;
        double averageTime = totalTime / (runParams.numInner * runParams.numOuter);

        std::cout << "Average runtime (s): " << averageTime << std::endl;
        std::cout << "Average GFLOPS:      "
                  << (double)problemParams.m * problemParams.n * problemParams.k * 2.0 / averageTime
                         * 1.e-9
                  << std::endl;

        result.kernelAssemble = TimerPool::nanoseconds("CommandKernel::assembleKernel");
        result.kernelGenerate = TimerPool::nanoseconds("CommandKernel::generateKernel");

        if(runParams.check)
        {
            AssertFatal(hipMemcpy(hostD.data(),
                                  deviceD.get(),
                                  problemParams.m * problemParams.n * sizeof(D),
                                  hipMemcpyDeviceToHost)
                        == (hipError_t)HIP_SUCCESS);

            auto [correct, rnorm] = validate<A, B, C, D>(
                hostA, hostB, hostC, hostD, hostScaleA, hostScaleB, problemParams, arch);

            result.checked = true;
            result.correct = correct;
            result.rnorm   = rnorm;
        }

        return result;
    }

    template <typename A, typename C, typename D>
    Client::BenchmarkResults GEMMMixed(CommandPtr               command,
                                       CommandKernelPtr         commandKernel,
                                       GEMMSolutionPtr          gemm,
                                       ProblemParameters const& problemParams,
                                       RunParameters const&     runParams,
                                       GPUArchitecture const&   arch,
                                       auto                     typeB)
    {
        if(typeB == "fp8")
        {
            return GEMM<A, FP8, C, D>(command, commandKernel, gemm, problemParams, runParams, arch);
        }
        else if(typeB == "bf8")
        {
            return GEMM<A, BF8, C, D>(command, commandKernel, gemm, problemParams, runParams, arch);
        }
        else if(typeB == "fp6")
        {
            return GEMM<A, FP6, C, D>(command, commandKernel, gemm, problemParams, runParams, arch);
        }
        else if(typeB == "bf6")
        {
            return GEMM<A, BF6, C, D>(command, commandKernel, gemm, problemParams, runParams, arch);
        }
        else if(typeB == "fp4")
        {
            return GEMM<A, FP4, C, D>(command, commandKernel, gemm, problemParams, runParams, arch);
        }
        else
            Throw<FatalError>("Invalid type for Mixed GEMM.");
    }

    template <typename C, typename D>
    Client::BenchmarkResults GEMMMixed(CommandPtr               command,
                                       CommandKernelPtr         commandKernel,
                                       GEMMSolutionPtr          gemm,
                                       ProblemParameters const& problemParams,
                                       RunParameters const&     runParams,
                                       GPUArchitecture const&   arch,
                                       auto                     typeA,
                                       auto                     typeB)
    {
        if(typeA == "fp8")
        {
            return GEMMMixed<FP8, C, D>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeB);
        }
        else if(typeA == "bf8")
        {
            return GEMMMixed<BF8, C, D>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeB);
        }
        else if(typeA == "fp6")
        {
            return GEMMMixed<FP6, C, D>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeB);
        }
        else if(typeA == "bf6")
        {
            return GEMMMixed<BF6, C, D>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeB);
        }
        else if(typeA == "fp4")
        {
            return GEMMMixed<FP4, C, D>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeB);
        }
        else
            Throw<FatalError>("Invalid type for Mixed GEMM.");
    }

    template <typename AB>
    Client::BenchmarkResults GEMMUniform(CommandPtr               command,
                                         CommandKernelPtr         commandKernel,
                                         GEMMSolutionPtr          gemm,
                                         ProblemParameters const& problemParams,
                                         RunParameters const&     runParams,
                                         GPUArchitecture const&   arch,
                                         auto                     typeCD)
    {
        if(typeCD == "float")
        {
            return GEMM<AB, AB, float, float>(
                command, commandKernel, gemm, problemParams, runParams, arch);
        }
        if(typeCD == "half")
        {
            return GEMM<AB, AB, Half, Half>(
                command, commandKernel, gemm, problemParams, runParams, arch);
        }
        if(typeCD == "bf16")
        {
            return GEMM<AB, AB, BFloat16, BFloat16>(
                command, commandKernel, gemm, problemParams, runParams, arch);
        }
        Throw<FatalError>("Invalid CD type for uniform GEMM.");
    }

    Client::BenchmarkResults GEMMUniform(CommandPtr               command,
                                         CommandKernelPtr         commandKernel,
                                         GEMMSolutionPtr          gemm,
                                         ProblemParameters const& problemParams,
                                         RunParameters const&     runParams,
                                         GPUArchitecture const&   arch,
                                         auto                     typeAB,
                                         auto                     typeCD)
    {
        if(typeAB == "float")
        {
            return GEMMUniform<float>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "half")
        {
            return GEMMUniform<Half>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "bf16")
        {
            return GEMMUniform<BFloat16>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "fp8")
        {
            return GEMMUniform<FP8>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "bf8")
        {
            return GEMMUniform<BF8>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "fp6")
        {
            return GEMMUniform<FP6>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "bf6")
        {
            return GEMMUniform<BF6>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        if(typeAB == "fp4")
        {
            return GEMMUniform<FP4>(
                command, commandKernel, gemm, problemParams, runParams, arch, typeCD);
        }
        Throw<FatalError>("Invalid AB type for uniform GEMM.");
    }

    /*
     * Generate an instance of GEMMSolution and call generateSolution.
     *
     * The kind of GEMMSolution returned is based on the parameters in
     * solutionParams.
     */
    std::pair<GEMMSolutionPtr, int> createGEMMSolution(rocRoller::ContextPtr     context,
                                                       SolutionParameters const& solution)
    {
        GEMMSolutionPtr gemmSolution;

        auto const& arch = context->targetArchitecture().target();

        if(solution.streamK)
        {
            if(context->targetArchitecture().HasCapability(GPUCapability::ArchAccUnifiedRegs))
            {
                gemmSolution = std::make_shared<Client::GEMMClient::StreamKGEMMSolution>(context);
            }
            else
            {
                std::cout << "Not running StreamK for " << arch.toString() << std::endl;
                return {nullptr, ReturnCodes::SolutionNotSupportedOnArch};
            }
        }
        else
        {
            gemmSolution = std::make_shared<Client::GEMMClient::DataParallelGEMMSolution>(context);
        }

        AssertFatal(gemmSolution, "No solution!");

        return {gemmSolution, ReturnCodes::OK};
    }

    struct ArchitectureParameters
    {
        GPUArchitectureTarget target;
    };

    struct TypeParameters
    {
        std::string typeA   = "float";
        std::string typeB   = "float";
        std::string typeC   = "float";
        std::string typeD   = "float";
        std::string typeAcc = "float";

        Client::GEMMClient::TransposeType transA = Client::GEMMClient::TransposeType::N;
        Client::GEMMClient::TransposeType transB = Client::GEMMClient::TransposeType::N;

        Operations::ScaleMode scaleA = Operations::ScaleMode::None;
        Operations::ScaleMode scaleB = Operations::ScaleMode::None;

        int scaleBlockSize = -1;
    };

    struct IOParameters
    {
        bool        doSaveAsm, doSaveCO;
        std::string saveAsmPath, loadAsmPath;
        std::string saveCOPath, loadCOPath;
        std::string resultsPath, timersPath;
    };

    void writeFile(std::filesystem::path const& filename, std::vector<char> const& x)
    {
        std::ofstream file(filename, std::ios::out | std::ios::binary);
        std::copy(x.cbegin(), x.cend(), std::ostream_iterator<unsigned char>(file));
    }

    int RunGEMMCLI(bool                   doGenerate,
                   bool                   doValidate,
                   bool                   doBenchmark,
                   bool                   doInfo,
                   ArchitectureParameters architecture,
                   SolutionParameters     solution,
                   ProblemParameters      problem,
                   TypeParameters         types,
                   RunParameters          run,
                   IOParameters           io)
    {
        GEMMSolutionPtr  gemm;
        CommandPtr       command;
        CommandKernelPtr commandKernel;

        if(doInfo)
        {
            std::cout << "Loading kernel from: " << io.loadCOPath << std::endl;
            auto yaml = readMetaDataFromCodeObject(io.loadCOPath);
            std::cout << yaml << std::endl;

            auto kernelFromYAML = AssemblyKernels::fromYAML(yaml).kernels[0];
            std::cout << *kernelFromYAML.command() << std::endl;

            return ReturnCodes::OK;
        }

        // Changing settings has to go before creating the context :(
        if(io.doSaveAsm)
        {
            if(io.saveAsmPath.empty())
                io.saveAsmPath = solution.generateKernelName() + ".s";

            Settings::getInstance()->set(Settings::SaveAssembly, true);
            Settings::getInstance()->set(Settings::AssemblyFile, std::string(io.saveAsmPath));
        }

        // When loading from .s or .co, currently need to load
        // solution from a .yaml file.
        if(!io.loadAsmPath.empty())
        {
            auto yamlPath = std::filesystem::path{io.loadAsmPath};
            yamlPath.replace_extension(".yaml");

            solution            = Serialization::readYAMLFile<SolutionParameters>(yamlPath);
            architecture.target = solution.architecture;
            if(solution.version != rocRoller::Version::Git())
            {
                std::cout << "Warning: this version of rocRoller (" << rocRoller::Version::Git()
                          << ") differs from the one that generated the kernel." << std::endl;
            }
        }

        auto const arch = GPUArchitectureLibrary::getInstance()->GetArch(architecture.target);

        if(solution.scheduler != "")
        {
            auto schedulerValue = fromString<Scheduling::SchedulerProcedure>(solution.scheduler);
            Settings::getInstance()->set(Settings::Scheduler, schedulerValue);
        }

        auto context = Context::ForTarget(arch, solution.generateKernelName());

        bool willRunOnGPU = doValidate || doBenchmark;
        if(willRunOnGPU)
        {
            std::cout << "Setting HIP device to " << run.device << std::endl;
            HIP_CHECK(hipSetDevice(run.device));
        }

        if(willRunOnGPU && solution.streamK)
        {
            if(run.numWGs == 0)
            {
                hipDeviceProp_t deviceProperties;
                AssertFatal(hipGetDeviceProperties(&deviceProperties, 0)
                            == (hipError_t)HIP_SUCCESS);
                run.numWGs = deviceProperties.multiProcessorCount;
            }
            AssertFatal(!solution.streamKTwoTile || solution.streamK);
        }

        if(doGenerate)
        {
            std::cout << "Generating for architecture: "
                      << context->targetArchitecture().target().toString() << std::endl;

            std::cout << std::endl;
            std::cout << "Solution:" << std::endl;
            std::cout << solution << std::endl;

            std::cout << "Generating: " << solution.generateKernelName() << "..." << std::endl;

            int reason;
            std::tie(gemm, reason) = createGEMMSolution(context, solution);
            if(!gemm)
                return reason;
            command       = gemm->makeCommand(solution);
            commandKernel = gemm->generateCommandKernel(command, solution);

            std::string basePath;
            if(io.doSaveAsm)
                basePath = io.saveAsmPath;
            if(io.doSaveCO)
                basePath = io.saveCOPath;

            if(io.doSaveAsm || io.doSaveCO)
            {
                // When saveing ASM, also need code-object so that
                // Command (ie, workgroup size, argument mapping etc)
                // can be de-serialized later.

                auto codeObject = commandKernel->assembleKernel();

                std::filesystem::path codeObjectPath{basePath};
                codeObjectPath.replace_extension(".co");

                std::filesystem::path yamlPath{basePath};
                yamlPath.replace_extension(".yaml");
                if(!std::filesystem::exists(yamlPath))
                {
                    std::ofstream file(yamlPath);
                    Serialization::writeYAML(file, solution);
                    std::cout << "Wrote: " << yamlPath.string() << std::endl;
                }

                writeFile(codeObjectPath, codeObject);
                std::cout << "Wrote: " << codeObjectPath.string() << std::endl;
            }

            if(io.doSaveAsm)
            {
                std::filesystem::path assemblyPath{basePath};
                assemblyPath.replace_extension(".s");

                // We don't explicitly write here (the code-gen does),
                // but we still emit a message here.
                std::cout << "Wrote: " << assemblyPath.string() << std::endl;
            }
        }
        else
        {
            int reason;
            std::tie(gemm, reason) = createGEMMSolution(context, solution);

            command = gemm->makeCommand(solution);

            if(!io.loadAsmPath.empty())
            {
                // When loading ASM, we also load code-object so that
                // Command (ie, workgroup size, argument mapping etc)
                // can be de-serialized.
                std::filesystem::path codeObjectPath{io.loadAsmPath};
                codeObjectPath.replace_extension(".co");

                std::cout << "Loading kernel meta-data from: " << codeObjectPath.string()
                          << std::endl;
                commandKernel = std::make_shared<CommandKernel>();
                commandKernel->setContext(context);
                auto kernel = commandKernel->loadKernelFromCodeObject(
                    codeObjectPath, solution.generateKernelName());
                command = kernel->command();

                std::cout << "Loading kernel from: " << io.loadAsmPath << std::endl;
                commandKernel
                    = std::make_shared<CommandKernel>(command, solution.generateKernelName());
                commandKernel->setContext(context);
                commandKernel->loadKernelFromAssembly(io.loadAsmPath,
                                                      solution.generateKernelName());
            }
            else if(!io.loadCOPath.empty())
            {
                std::cout << "Loading kernel from: " << io.loadCOPath << std::endl;

                commandKernel = std::make_shared<CommandKernel>();
                commandKernel->setContext(context);
                auto kernel = commandKernel->loadKernelFromCodeObject(
                    io.loadCOPath, solution.generateKernelName());

                command = kernel->command();
            }
        }

        if(!gemm || !command || !commandKernel)
            return ReturnCodes::GenerateFailure;

        if(doValidate || doBenchmark)
        {
            std::cout << "Running..." << std::endl;

            if(doValidate)
            {
                run.check     = true;
                run.numWarmUp = 0;
                run.numOuter  = 1;
                run.numInner  = 1;
            }

            auto isF8F6F4 = [](auto dtype) {
                return (dtype == "fp8" || dtype == "bf8" || dtype == "fp6" || dtype == "bf6"
                        || dtype == "fp4");
            };

            Client::GEMMClient::Result result;

            result.problemParams              = problem;
            result.solutionParams             = solution;
            result.benchmarkResults.runParams = run;

            if(types.typeA == types.typeB && types.typeC == types.typeD)
            {
                result.benchmarkResults = GEMMUniform(
                    command, commandKernel, gemm, problem, run, arch, types.typeA, types.typeC);
            }
            else if((problem.typeA != problem.typeB) && isF8F6F4(problem.typeA)
                    && isF8F6F4(problem.typeB))
            {
                result.benchmarkResults = GEMMMixed<float, float>(
                    command, commandKernel, gemm, problem, run, arch, problem.typeA, problem.typeB);
            }
            else
            {
                Throw<FatalError>("Unsupported combination of datatypes for GEMM");
            }

            if(!io.resultsPath.empty())
            {
                std::ofstream file(io.resultsPath);
                Serialization::writeYAML(file, result);
            }

            if(!result.benchmarkResults.correct)
                return ReturnCodes::CorrectnessFailure;
        }

        // Dump timers
        if(!io.timersPath.empty())
        {
            std::ofstream dfile;
            dfile.open(io.timersPath, std::ofstream::out | std::ofstream::trunc);
            dfile << rocRoller::TimerPool::CSV();
            dfile.close();
        }

        return ReturnCodes::OK;
    }

    void overwriteTypesFromSolution(TypeParameters& types, SolutionParameters const& solution)
    {
        if((types.typeA != solution.typeA) || (types.typeB != solution.typeB)
           || (types.typeC != solution.typeC) || (types.typeD != solution.typeD)
           || (types.typeAcc != solution.typeAcc))
        {
            std::cout << "NOTE: Types have been superceded by solution." << std::endl;
        }
        if((types.transA != solution.transA) || (types.transB != solution.transB))
        {
            std::cout << "NOTE: Transposes have been superceded by solution." << std::endl;
        }
        if((types.scaleA != solution.scaleA) || (types.scaleA != solution.scaleB))
        {
            std::cout << "NOTE: MX Scalings have been superceded by solution." << std::endl;
        }
        if(types.scaleBlockSize != solution.scaleBlockSize)
        {
            std::cout << "NOTE: MX scale block size has been superceded by solution." << std::endl;
        }

        types.typeA          = solution.typeA;
        types.typeB          = solution.typeB;
        types.typeC          = solution.typeC;
        types.typeD          = solution.typeD;
        types.typeAcc        = solution.typeAcc;
        types.transA         = solution.transA;
        types.transB         = solution.transB;
        types.scaleA         = solution.scaleA;
        types.scaleB         = solution.scaleB;
        types.scaleBlockSize = solution.scaleBlockSize;
    }
}

constexpr bool PARSE_SUCCESS = true;
constexpr bool PARSE_FAILURE = false;

static bool ParseMI(const std::string&                                 arg,
                    rocRoller::Client::GEMMClient::SolutionParameters& solution)
{
    if(arg.empty())
        return PARSE_FAILURE;

    bool fail = false;
    try
    {
        std::istringstream iss(arg);
        std::string        token;

        iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
        std::getline(iss, token, 'x');
        solution.waveM = std::stoi(token);
        std::getline(iss, token, 'x');
        solution.waveN = std::stoi(token);
        std::getline(iss, token, 'x');
        solution.waveK = std::stoi(token);
        iss.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        std::getline(iss, token, 'x');
        solution.waveB = std::stoi(token);
    }
    catch(const std::invalid_argument&)
    {
        fail = true;
    }
    catch(const std::ios_base::failure&)
    {
        fail = true;
    }

    fail |= (solution.waveM < 1) || (solution.waveN < 1) || (solution.waveK < 1)
            || (solution.waveB < 1);

    if(fail)
    {
        std::cerr << "Invalid format for MI instruction." << std::endl;
        std::cerr << std::endl;
        std::cerr << "The MI argument should be formatted like:" << std::endl;
        std::cerr << std::endl;
        std::cerr << "    --mi=MxNxKxB" << std::endl;
        std::cerr << std::endl;
        std::cerr << "For example: --mi=32x32x2x1" << std::endl;

        return PARSE_FAILURE;
    }

    return PARSE_SUCCESS;
}

static bool ParseWorkgroupMapping(const std::string&                                 arg,
                                  rocRoller::Client::GEMMClient::SolutionParameters& solution)
{
    if(arg.empty())
        return PARSE_FAILURE;

    bool fail = false;
    try
    {
        std::istringstream iss(arg);
        std::string        token;

        iss.exceptions(std::ifstream::eofbit | std::ifstream::failbit | std::ifstream::badbit);
        std::getline(iss, token, ',');
        auto dim = std::stoi(token);
        iss.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        std::getline(iss, token, ',');
        auto size                 = std::stoi(token);
        solution.workgroupMapping = {dim, size};

        fail |= (dim != -1 && dim != 0 && dim != 1);
        fail |= (size != -1 && size < 1);
    }
    catch(const std::invalid_argument&)
    {
        fail = true;
    }
    catch(const std::ios_base::failure&)
    {
        fail = true;
    }

    if(fail)
    {
        std::cerr << "Invalid format for workgroupMapping." << std::endl;
        std::cerr << std::endl;
        std::cerr << "The workgroupMapping argument should be formatted like:" << std::endl;
        std::cerr << std::endl;
        std::cerr << "    --workgroupMapping=d,s" << std::endl;
        std::cerr << std::endl;
        std::cerr << "where d is 0 or 1, and s is a positive integer." << std::endl;
        std::cerr << std::endl;
        std::cerr << "For example: --workgroupMapping=0,6" << std::endl;

        return PARSE_FAILURE;
    }

    return PARSE_SUCCESS;
}

/*
 * Parse the command line and dispatch.
 */
int main(int argc, const char* argv[])
{
    CLI::App app{"GEMM Driver: D (MxN) = alpha * A (MxK) * B (KxN) + beta * C (MxN)"};
    app.footer(Settings::getInstance()->help());

    //
    // Parameters
    //
    std::string                                           architectureName;
    rocRoller::Client::GEMMClient::ArchitectureParameters architecture;

    rocRoller::Client::GEMMClient::SolutionParameters solution{
        .macM = 64,
        .macN = 64,
        .macK = 64,

        .waveM = -1,
        .waveN = -1,
        .waveK = -1,
        .waveB = -1,

        .workgroupSizeX         = 128,
        .workgroupSizeY         = 2,
        .workgroupMapping       = {-1, -1},
        .workgroupRemapXCC      = false,
        .workgroupRemapXCCValue = -1,

        .scaleA = Operations::ScaleMode::None,
        .scaleB = Operations::ScaleMode::None,

        .scaleBlockSize = -1,

        .loadLDSScaleA = false,
        .loadLDSScaleB = false,

        .swizzleScale  = false,
        .prefetchScale = false,

        .loadLDSA  = true,
        .loadLDSB  = true,
        .storeLDSD = true,

        .direct2LDSA = false,
        .direct2LDSB = false,

        .prefetch          = false,
        .prefetchInFlight  = 0,
        .prefetchLDSFactor = 0,
        .prefetchMixMemOps = false,

        .betaInFma = true,

        .unrollX = 0,
        .unrollY = 0,

        .scheduler         = "Priority",
        .matchMemoryAccess = true,

        .streamK        = false,
        .streamKTwoTile = false,

        .version = rocRoller::Version::Git(),
    };

    rocRoller::Client::GEMMClient::ProblemParameters problem{
        .m = 3072,
        .n = 4096,
        .k = 4096,

        .alpha = 2.0f,
        .beta  = 0.5f,

        .scaleValueA = 1.0f,
        .scaleValueB = 1.0f,
    };

    rocRoller::Client::GEMMClient::TypeParameters types;

    rocRoller::Client::RunParameters run{
        .device    = 0,
        .numWarmUp = 3,
        .numOuter  = 5,
        .numInner  = 2,
        .check     = true,
        .visualize = false,
        .numWGs    = 0,
    };

    rocRoller::Client::GEMMClient::IOParameters io{
        .doSaveAsm   = false,
        .doSaveCO    = false,
        .saveAsmPath = "",
        .loadAsmPath = "",
        .saveCOPath  = "",
        .loadCOPath  = "",
        .resultsPath = "",
    };

    //
    // Architecture
    //
    app.option_defaults()->ignore_case()->group("Architecture parameters");
    app.add_option("--arch", architectureName, "GPU architecture name (eg, gfx90a).");

    //
    // Problem definition
    //
    app.option_defaults()->ignore_case()->group("Problem parameters");
    app.add_option("-M,--M", problem.m, "Tensor size M.");
    app.add_option("-N,--N", problem.n, "Tensor size N.");
    app.add_option("-K,--K", problem.k, "Tensor size K.");
    app.add_option("--alpha", problem.alpha, "Alpha scalar.");
    app.add_option("--beta", problem.beta, "Beta scalar.");
    app.add_option("--scaleValue_A", problem.scaleValueA, "Single scale value for A.");
    app.add_option("--scaleValue_B", problem.scaleValueB, "Single scale value for B.");

    //
    // Problem types
    //
    app.option_defaults()->ignore_case()->group("Type parameters");
    app.add_option("--type_A",
                   types.typeA,
                   "Datatype of A matrix [float | half | bf16 | fp8 | bf8 | fp6 | bf6 | fp4].  "
                   "Default: float.");
    app.add_option("--type_B",
                   types.typeB,
                   "Datatype of B matrix [float | half | bf16 | fp8 | bf8 | fp6 | bf6 | fp4].  "
                   "Default: float.");
    app.add_option(
        "--type_C", types.typeC, "Datatype of C matrix [float | half | bf16].  Default: float.");
    app.add_option(
        "--type_D", types.typeD, "Datatype of D matrix [float | half | bf16].  Default: float.");
    app.add_option("--type_acc",
                   types.typeAcc,
                   "Datatype of accumulation [float | half | bf16].  Default: float");
    app.add_option(
        "--trans_A",
        [&types](auto res) -> bool {
            types.transA = fromString<Client::GEMMClient::TransposeType>(res[0]);
            return true;
        },
        "N: A is not to be transposed.  T: A is to be transposed.",
        "N");
    app.add_option(
        "--trans_B",
        [&types](auto res) -> bool {
            types.transB = fromString<Client::GEMMClient::TransposeType>(res[0]);
            return true;
        },
        "N: B is not to be transposed.  T: B is to be transposed.",
        "N");
    app.add_option(
        "--scale_A",
        [&types](auto res) -> bool {
            types.scaleA = fromString<Operations::ScaleMode>(res[0]);
            return true;
        },
        "Enable MX scaling of A matrix [None | Separate | SingleScale].",
        "Default: None.");
    app.add_option(
        "--scale_B",
        [&types](auto res) -> bool {
            types.scaleB = fromString<Operations::ScaleMode>(res[0]);
            return true;
        },
        "Enable MX scaling of B matrix [None | Separate | SingleScale].",
        "Default: None.");
    app.add_option("--scaleBlockSize",
                   types.scaleBlockSize,
                   "Set MX scaling block size for A and B. (default: 32)");

    //
    // Kernel options
    //
    app.option_defaults()->ignore_case()->group("Solution parameters");
    app.add_option("--mac_m", solution.macM, "(Macro) Tile size M.");
    app.add_option("--mac_n", solution.macN, "(Macro) Tile size N.");
    app.add_option("--mac_k", solution.macK, "(Macro) Tile size K.");
    app.add_option("--wave_m", solution.waveM, "(MI) Tile size M.");
    app.add_option("--wave_n", solution.waveN, "(MI) Tile size N.");
    app.add_option("--wave_k", solution.waveK, "(MI) Tile size K.");
    app.add_option("--wave_b", solution.waveB, "(MI) Tile size K.");

    app.add_option(
        "--mi",
        [&solution](auto& args) -> bool { return ParseMI(args[0], solution); },
        "MI instruction to use.  Default 32x32x2x1 for floats, 32x32x8x1 for halfs.");

    app.add_option(
        "--workgroup_size_x", solution.workgroupSizeX, "Workgroup size in the x dimension.");
    app.add_option(
        "--workgroup_size_y", solution.workgroupSizeY, "Workgroup size in the y dimension.");
    app.add_option(
        "--workgroupMapping",
        [&solution](auto& args) -> bool { return ParseWorkgroupMapping(args[0], solution); },
        "Workgroup mapping dimension and size.");
    app.add_flag(
        "--workgroupRemapXCC", solution.workgroupRemapXCC, "Use an XCC-aware workgroup remapping.");
    app.add_option("--workgroupRemapXCCValue",
                   solution.workgroupRemapXCCValue,
                   "Force an XCC-aware workgroup remapping value. (Optional)");
    app.add_option("--unroll_x", solution.unrollX, "Unroll size in X.");
    app.add_option("--unroll_y", solution.unrollY, "Unroll size in Y.");
    app.add_flag("--loadLDS_A", solution.loadLDSA, "Use LDS when loading A.");
    app.add_flag("--loadLDS_B", solution.loadLDSB, "Use LDS when loading B.");
    app.add_flag("--storeLDS_D", solution.storeLDSD, "Use LDS when storing D.");
    app.add_flag("--direct2LDS_A", solution.direct2LDSA, "Use direct-to-LDS when loading A.");
    app.add_flag("--direct2LDS_B", solution.direct2LDSB, "Use direct-to-LDS when loading B.");
    app.add_flag(
        "--betaInFma", solution.betaInFma, "Use beta in FMA instruction instead of alpha.");
    app.add_option("--scheduler", solution.scheduler, "Which scheduler to use.");
    app.add_flag("--match_memory_access",
                 solution.matchMemoryAccess,
                 "Match memory access to transpose.  Currently decreases performance.");
    app.add_flag("--prefetch", solution.prefetch, "Enable prefetching (UnrollK=2 implied).");
    app.add_option("--prefetchInFlight",
                   solution.prefetchInFlight,
                   "Number of prefetches in flight at the same time");
    app.add_option("--prefetchLDSFactor",
                   solution.prefetchLDSFactor,
                   "Prefetch 1/prefetchLDSFactor of MacroTile from LDS");
    auto prefetchMixMemOpsFlag
        = app.add_flag("--prefetchMixMemOps",
                       solution.prefetchMixMemOps,
                       "Mix global and LDS memory operations during prefetching.");
    app.add_flag("--streamK", solution.streamK, "Enable StreamK algorithm.");
    app.add_flag("--streamKTwoTile", solution.streamKTwoTile, "Enable two-tile StreamK algorithm.");

    app.add_flag("--loadLDSScale_A", solution.loadLDSScaleA, "Use LDS when loading A scale.");
    app.add_flag("--loadLDSScale_B", solution.loadLDSScaleB, "Use LDS when loading B scale.");

    app.add_flag(
        "--swizzleScale", solution.swizzleScale, "Use Swizzle when loading A and B scale.");
    app.add_flag("--prefetchScale",
                 solution.prefetchScale,
                 "Prefetch scale values with using Swizzled scales.");

    //
    // Benchmarking options
    //
    app.option_defaults()->ignore_case()->group("Benchmarking parameters");
    app.add_option("--num_warmup", run.numWarmUp, "Number of warm-up runs.");
    app.add_option("--num_outer", run.numOuter, "Number of outer runs.");
    app.add_option("--num_inner", run.numInner, "Number of inner runs.");
    app.add_option("--device", run.device, "GPU device ordinal");
    app.add_option("--numWGs",
                   run.numWGs,
                   "Number of workgroups to use with StreamK algorithm.  Defaults to number of WGs "
                   "present on local device.");

    //
    // Client params and shortcuts
    //

    app.option_defaults()->ignore_case()->group("Client options and shortcuts");

    bool noCheckResult = false;

    std::string loadPath, examplePath;

    app.add_flag(
        "--hgemm",
        [&types](auto res) -> bool {
            types.typeA   = "half";
            types.typeB   = "half";
            types.typeC   = "half";
            types.typeD   = "half";
            types.typeAcc = "float";
            return true;
        },
        "Overwrite types to: --type_A=half --type_B=half --type_C=half --type_D=half "
        "--type_acc=float.");

    app.add_flag(
        "--visualize", run.visualize, "Dump out volumes describing memory access patterns.");

    app.add_flag("--no-check", noCheckResult, "Do not verify GEMM results against OpenBLAS.");

    app.add_option("--yaml", io.resultsPath, "Save results to file.");

    app.add_option("--timers", io.timersPath, "Save timers to CSV file.");

    //
    // generate sub-command
    //

    std::string loadConfigPath;

    auto generate = app.add_subcommand("generate", "Generate a GEMM solution.")->fallthrough();

    auto asmOption = generate->add_option("--asm", io.saveAsmPath, "Save yaml+assembly to files.")
                         ->expected(0, 1);
    auto coOption
        = generate->add_option("--co", io.saveCOPath, "Save code-object to file.")->expected(0, 1);
    generate
        ->add_option(
            "--config", loadConfigPath, "Load solution generation parameters from YAML file.")
        ->expected(1, 1);

    //
    // validate sub-command
    //

    auto validate
        = app.add_subcommand("validate",
                             "Run and validate a GEMM solution (only runs the solution once).")
              ->fallthrough();

    validate->add_option(
        "--load", loadPath, "Load solution from code-object (.co) or assembly (.s) file.");

    //
    // benchmark sub-command
    //

    auto benchmark
        = app.add_subcommand("benchmark",
                             "Benchmark a GEMM solution (may run the solution many times).")
              ->fallthrough();

    benchmark->add_option(
        "--load", loadPath, "Load solution from code-object (.co) or assembly (.s) file.");

    //
    // info sub-command
    //

    auto info = app.add_subcommand("info", "Dump info about a GEMM solution.")->fallthrough();

    info->add_option("load", loadPath, "Load solution from code-object (.co).")->required();

    //
    // example sub-command
    //
    auto example = app.add_subcommand("example", "Save example generation parameters to YAML file.")
                       ->fallthrough();

    example->add_option("save", examplePath, "Example config path.")->required();

    //
    // Parse and update/validate problem definition
    //

    CLI11_PARSE(app, argc, argv);

    if(architectureName.empty())
        architecture.target
            = GPUArchitectureLibrary::getInstance()->GetDefaultHipDeviceArch(run.device).target();
    else
        architecture.target = GPUArchitectureTarget::fromString(architectureName);

    solution.architecture = architecture.target;

    if(!loadConfigPath.empty())
    {
        // THIS OVERWRITES COMMAND LINE OPTIONS
        solution = Serialization::readYAMLFile<rocRoller::Client::GEMMClient::SolutionParameters>(
            loadConfigPath);

        if(solution.architecture.gfx == GPUArchitectureGFX::UNKNOWN)
            solution.architecture = architecture.target;

        overwriteTypesFromSolution(types, solution);
    }

    if(!loadPath.empty())
    {
        auto path = std::filesystem::path(loadPath);
        if(path.extension() == ".s" || path.extension() == ".yaml")
        {
            io.loadAsmPath = path.string();
        }
        else if(path.extension() == ".co")
        {
            io.loadCOPath = path.string();
        }
        else
        {
            Throw<FatalError>("Extension not supported.  Can not load solution from ", loadPath);
        }
    }

    if(!io.loadAsmPath.empty() || !io.loadCOPath.empty())
    {
        std::filesystem::path yamlPath;
        if(!io.loadAsmPath.empty())
            yamlPath = std::filesystem::path{io.loadAsmPath};
        if(!io.loadCOPath.empty())
            yamlPath = std::filesystem::path{io.loadCOPath};
        yamlPath.replace_extension(".yaml");

        solution = Serialization::readYAMLFile<rocRoller::Client::GEMMClient::SolutionParameters>(
            yamlPath);

        overwriteTypesFromSolution(types, solution);
    }

    if(types.scaleA != Operations::ScaleMode::None && types.scaleB == Operations::ScaleMode::None)
    {
        types.scaleB        = Operations::ScaleMode::SingleScale;
        problem.scaleValueB = 1.0f;
    }

    if(types.scaleB != Operations::ScaleMode::None && types.scaleA == Operations::ScaleMode::None)
    {
        types.scaleA        = Operations::ScaleMode::SingleScale;
        problem.scaleValueA = 1.0f;
    }

    auto const& arch = GPUArchitectureLibrary::getInstance()->GetArch(solution.architecture);
    if(types.scaleBlockSize == -1
       && (types.scaleA == Operations::ScaleMode::Separate
           || types.scaleB == Operations::ScaleMode::Separate))
    {
        AssertFatal(arch.HasCapability(GPUCapability::HasBlockScaling32),
                    fmt::format("Architecture {} does not support block scaling.",
                                arch.target().toString()));
        types.scaleBlockSize   = arch.GetCapability(GPUCapability::DefaultScaleBlockSize);
        problem.scaleBlockSize = types.scaleBlockSize;
    }

    AssertFatal((types.typeAcc == "float") || (types.typeAcc == "half")
                || (types.typeAcc == "bf16"));

    problem.typeA          = types.typeA;
    problem.typeB          = types.typeB;
    problem.typeC          = types.typeC;
    problem.typeD          = types.typeD;
    problem.typeAcc        = types.typeAcc;
    problem.transA         = types.transA;
    problem.transB         = types.transB;
    problem.scaleA         = types.scaleA;
    problem.scaleB         = types.scaleB;
    problem.scaleBlockSize = types.scaleBlockSize;

    solution.typeA          = types.typeA;
    solution.typeB          = types.typeB;
    solution.typeC          = types.typeC;
    solution.typeD          = types.typeD;
    solution.typeAcc        = types.typeAcc;
    solution.transA         = types.transA;
    solution.transB         = types.transB;
    solution.scaleA         = types.scaleA;
    solution.scaleB         = types.scaleB;
    solution.scaleBlockSize = types.scaleBlockSize;

    // TODO: Reevaluate the relationship between problem and solution params.
    problem.workgroupMapping = solution.workgroupMapping;

    run.check = !noCheckResult;

    io.doSaveAsm = asmOption->count() > 0;
    io.doSaveCO  = coOption->count() > 0;

    // Set default MI sizes
    if(arch.HasCapability(GPUCapability::HasMFMA))
    {
        if(problem.typeA == "float" && problem.typeB == "float" && problem.typeC == "float"
           && problem.typeD == "float")
        {
            if(solution.waveM == -1)
                solution.waveM = 32;
            if(solution.waveN == -1)
                solution.waveN = 32;
            if(solution.waveK == -1)
                solution.waveK = 2;
            if(solution.waveB == -1)
                solution.waveB = 1;
        }
        else if(solution.typeA == "half" && solution.typeB == "half")
        {
            if(solution.waveM == -1)
                solution.waveM = 32;
            if(solution.waveN == -1)
                solution.waveN = 32;
            if(solution.waveK == -1)
                solution.waveK = 8;
            if(solution.waveB == -1)
                solution.waveB = 1;
        }
        else if(solution.typeA == "bf16" && solution.typeB == "bf16")
        {
            if(solution.waveM == -1)
                solution.waveM = 16;
            if(solution.waveN == -1)
                solution.waveN = 16;
            if(solution.waveK == -1)
                solution.waveK = 8;
            if(solution.waveB == -1)
                solution.waveB = 1;
        }
        else if((solution.typeA == "fp8" && solution.typeB == "fp8")
                || (solution.typeA == "bf8" && solution.typeB == "bf8"))
        {
            if(solution.waveM == -1)
                solution.waveM = 16;
            if(solution.waveN == -1)
                solution.waveN = 16;
            if(solution.waveK == -1)
                solution.waveK = 32;
            if(solution.waveB == -1)
                solution.waveB = 1;
        }
    }
    else if(arch.HasCapability(GPUCapability::HasWMMA))
    {
        if(arch.target().isRDNA4GPU())
        {
            if((solution.typeA == "half" && solution.typeB == "half")
               || (solution.typeA == "bf16" && solution.typeB == "bf16")
               || (solution.typeA == "fp8" && solution.typeB == "fp8")
               || (solution.typeA == "bf8" && solution.typeB == "bf8")
               || (solution.typeA == "bf8" && solution.typeB == "fp8")
               || (solution.typeA == "fp8" && solution.typeB == "bf8"))
            {
                if(solution.waveM == -1)
                    solution.waveM = 16;
                if(solution.waveN == -1)
                    solution.waveN = 16;
                if(solution.waveK == -1)
                    solution.waveK = 16;
                if(solution.waveB == -1)
                    solution.waveB = 1;
            }
            else
            {
                // Override default settings for the `example` and `generate` subcommands.
                if(example->parsed() || generate->parsed())
                {
                    solution.typeA = "half";
                    solution.typeB = "half";
                    solution.typeC = "half";
                    solution.typeD = "half";
                    solution.waveM = 16;
                    solution.waveN = 16;
                    solution.waveK = 16;
                    solution.waveB = 1;
                }
                else
                {
                    Throw<FatalError>("Unsupported MI on: ",
                                      arch.target().toString(),
                                      ShowValue(problem.typeA),
                                      ShowValue(problem.typeB),
                                      ShowValue(problem.typeC),
                                      ShowValue(problem.typeD),
                                      ShowValue(problem.typeAcc));
                }
            }
            // TODO Support prefetch on gfx12
            solution.prefetch = false;
        }
        else
        {
            Throw<FatalError>("Unsupported arch for GEMM client: ", arch.target().toString());
        }
    }
    else
    {
        Throw<FatalError>("Unsupported arch for GEMM client: ", arch.target().toString());
    }

    // Set default prefetchMixMemOps
    if(prefetchMixMemOpsFlag->count() == 0)
    {
        solution.prefetchMixMemOps = false;

        if(solution.prefetchLDSFactor != 0)
            solution.prefetchMixMemOps = true;

        if(solution.scaleB == Operations::ScaleMode::Separate && !solution.loadLDSScaleB)
            solution.prefetchMixMemOps = false;

        if(solution.scaleA == Operations::ScaleMode::Separate && !solution.loadLDSScaleA)
            solution.prefetchMixMemOps = false;

        // TODO: enable (prefetchMixMemOps == true && prefetchLDSFactor == 2 && direct2LDSA/B = true)
        if(solution.prefetchLDSFactor == 2 && (solution.direct2LDSA || solution.direct2LDSB))
            solution.prefetchMixMemOps = false;
    }

    //
    // Run!
    //
    if(example->parsed())
    {
        std::ofstream file(examplePath);
        Serialization::writeYAML(file, solution);
        return 0;
    }

    // If no subcommands were given, default behaviour is: generate
    // and benchmark.
    bool generateAndBenchmark
        = !generate->parsed() && !validate->parsed() && !benchmark->parsed() && !info->parsed();

    bool doGenerate  = generateAndBenchmark || generate->parsed();
    bool doValidate  = validate->parsed();
    bool doBenchmark = generateAndBenchmark || benchmark->parsed();
    bool doInfo      = info->parsed();

    return rocRoller::Client::GEMMClient::RunGEMMCLI(doGenerate,
                                                     doValidate,
                                                     doBenchmark,
                                                     doInfo,
                                                     architecture,
                                                     solution,
                                                     problem,
                                                     types,
                                                     run,
                                                     io);
}
