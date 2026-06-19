// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <any>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp"
#include <Tensile/ContractionProblem.hpp>

namespace TensileLite::testing
{
    namespace detail
    {
        inline void setDataInitArg(Client::po::variables_map& args,
                                   std::string               key,
                                   std::any                  value)
        {
            args[std::move(key)] = Client::po::variable_value(std::move(value));
        }
    } // namespace detail

    struct DataInitConfig
    {
        // Must include the largest problem a test later passes to prepareCPUInputs()
        // or prepareGPUInputs(), because ClientProblemFactory sizes allocations from
        // these values.
        std::vector<std::vector<size_t>> problemSizes;

        std::string problemIdentifier = "Contraction_l_Alik_Bjlk_Cijk_Dijk";

        rocisa::DataType type      = rocisa::DataType::Float;
        rocisa::DataType aType     = rocisa::DataType::Float;
        rocisa::DataType bType     = rocisa::DataType::Float;
        rocisa::DataType cType     = rocisa::DataType::Float;
        rocisa::DataType dType     = rocisa::DataType::Float;
        rocisa::DataType eType     = rocisa::DataType::None;
        rocisa::DataType amaxDType = rocisa::DataType::None;
        rocisa::DataType alphaType = rocisa::DataType::Float;
        rocisa::DataType betaType  = rocisa::DataType::Float;

        bool stridedBatched          = false;
        bool groupedGemm             = false;
        bool highPrecisionAccumulate = false;
        bool deterministicMode       = false;
        bool cEqualD                 = false;
        int  sparse                  = 0;

        int mxABlock      = 0;
        int mxBBlock      = 0;
        int mxScaleFormat = 0;
        rocisa::DataType mxAType = rocisa::DataType::E8;
        rocisa::DataType mxBType = rocisa::DataType::E8;
        bool swizzleTensorA = false;
        bool swizzleTensorB = false;

        size_t      maxWorkspaceSize = 32 * 1024 * 1024;
        int         useBias          = 0;
        int         biasSource       = 3;
        std::string useScaleAB;
        bool        useScaleCD       = false;
        int         useScaleAlphaVec = 0;
        bool        useE             = false;
        bool        useGradient      = false;
        bool        outputAmaxD      = false;
        bool        useUserArgs      = false;

        std::vector<rocisa::DataType> biasTypeArgs = {rocisa::DataType::None};
        std::vector<int>              factorDimArgs = {0};
        std::vector<int>              streamKHybridMode = {0};

        ActivationType activationType = ActivationType::None;
        bool           activationNoGuard = false;
        std::vector<ActivationType> activationEnumArgs = {ActivationType::None};
        rocisa::DataType            activationComputeType = rocisa::DataType::None;
        rocisa::DataType            computeInputTypeA     = rocisa::DataType::None;
        rocisa::DataType            computeInputTypeB     = rocisa::DataType::None;
        rocisa::DataType            f32XdlMathOp          = rocisa::DataType::None;

        int                           numElementsToValidate = 0;
        bool                          pristineOnGpu         = true;
        Client::PruneSparseMode       pruneMode             = Client::PruneSparseMode::PruneRandom;
        int32_t                       rotatingBufferSize    = 0;
        int32_t                       rotatingBufferMode    = 0;
        Client::BoundsCheckMode       boundsCheck           = Client::BoundsCheckMode::Disable;

        int    numBenchmarks        = 0;
        int    numEnqueuesPerSync   = 0;
        int    maxEnqueuesPerSync   = -1;
        size_t minFlopsPerSync      = 0;
        int    numSyncsPerBenchmark = 0;
        int    numWarmups           = 0;

        bool printValids               = false;
        int  printMax                  = -1;
        bool printTensorA              = false;
        bool printTensorB              = false;
        bool printTensorC              = false;
        bool printTensorD              = false;
        bool printTensorRef            = false;
        bool printTensorBias           = false;
        bool printTensorAmaxD          = false;

        KernelLanguage    kernelLanguage    = KernelLanguage::Any;
        PerformanceMetric performanceMetric  = PerformanceMetric::DeviceEfficiency;
        int               metadataLayout    = 0;
        TensorOps         aOps;
        TensorOps         bOps;
        TensorOps         cOps;
        TensorOps         dOps;

        Client::InitMode initA            = Client::InitMode::Random;
        Client::InitMode initB            = Client::InitMode::Random;
        Client::InitMode initC            = Client::InitMode::Random;
        Client::InitMode initD            = Client::InitMode::Zero;
        Client::InitMode initE            = Client::InitMode::Zero;
        Client::InitMode initBias         = Client::InitMode::One;
        Client::InitMode initScaleA       = Client::InitMode::Two;
        Client::InitMode initScaleB       = Client::InitMode::Two;
        Client::InitMode initScaleC       = Client::InitMode::Two;
        Client::InitMode initScaleD       = Client::InitMode::Two;
        Client::InitMode initScaleAlphaVec = Client::InitMode::One;
        Client::InitMode initMxA          = Client::InitMode::One;
        Client::InitMode initMxB          = Client::InitMode::One;
        Client::InitMode initAlpha        = Client::InitMode::Two;
        Client::InitMode initBeta         = Client::InitMode::Two;
    };

    using BaseDataInitArgsOptions = DataInitConfig;

    inline Client::po::variables_map
        buildBaseDataInitArgs(DataInitConfig const& options)
    {
        if(options.problemSizes.empty())
        {
            throw std::invalid_argument(
                "buildBaseDataInitArgs requires at least one problem size.");
        }

        Client::po::variables_map args;

        detail::setDataInitArg(args,
                               "problem-identifier",
                               std::any(options.problemIdentifier));
        detail::setDataInitArg(args, "problem-size", std::any(options.problemSizes));

        detail::setDataInitArg(args, "type", std::any(options.type));
        detail::setDataInitArg(args, "a-type", std::any(options.aType));
        detail::setDataInitArg(args, "b-type", std::any(options.bType));
        detail::setDataInitArg(args, "c-type", std::any(options.cType));
        detail::setDataInitArg(args, "d-type", std::any(options.dType));
        detail::setDataInitArg(args, "e-type", std::any(options.eType));
        detail::setDataInitArg(args, "amaxD-type", std::any(options.amaxDType));
        detail::setDataInitArg(args, "alpha-type", std::any(options.alphaType));
        detail::setDataInitArg(args, "beta-type", std::any(options.betaType));

        detail::setDataInitArg(args, "strided-batched", std::any(options.stridedBatched));
        detail::setDataInitArg(args, "grouped-gemm", std::any(options.groupedGemm));
        detail::setDataInitArg(args,
                               "high-precision-accumulate",
                               std::any(options.highPrecisionAccumulate));
        detail::setDataInitArg(args, "deterministic-mode", std::any(options.deterministicMode));
        detail::setDataInitArg(args, "c-equal-d", std::any(options.cEqualD));
        detail::setDataInitArg(args, "sparse", std::any(options.sparse));
        detail::setDataInitArg(args, "kernel-language", std::any(options.kernelLanguage));
        detail::setDataInitArg(args,
                               "performance-metric",
                               std::any(options.performanceMetric));
        detail::setDataInitArg(args, "metadata-layout", std::any(options.metadataLayout));
        detail::setDataInitArg(args, "a-ops", std::any(options.aOps));
        detail::setDataInitArg(args, "b-ops", std::any(options.bOps));
        detail::setDataInitArg(args, "c-ops", std::any(options.cOps));
        detail::setDataInitArg(args, "d-ops", std::any(options.dOps));

        detail::setDataInitArg(args, "mx-a-block", std::any(options.mxABlock));
        detail::setDataInitArg(args, "mx-b-block", std::any(options.mxBBlock));
        detail::setDataInitArg(args, "mx-scale-format", std::any(options.mxScaleFormat));
        detail::setDataInitArg(args, "mx-a-type", std::any(options.mxAType));
        detail::setDataInitArg(args, "mx-b-type", std::any(options.mxBType));
        detail::setDataInitArg(args, "swizzle-tensor-a", std::any(options.swizzleTensorA));
        detail::setDataInitArg(args, "swizzle-tensor-b", std::any(options.swizzleTensorB));
        detail::setDataInitArg(args, "init-mx-a", std::any(options.initMxA));
        detail::setDataInitArg(args, "init-mx-b", std::any(options.initMxB));

        detail::setDataInitArg(args, "max-workspace-size", std::any(options.maxWorkspaceSize));
        detail::setDataInitArg(args, "use-bias", std::any(options.useBias));
        detail::setDataInitArg(args, "bias-source", std::any(options.biasSource));
        detail::setDataInitArg(args, "use-scaleAB", std::any(options.useScaleAB));
        detail::setDataInitArg(args, "use-scaleCD", std::any(options.useScaleCD));
        detail::setDataInitArg(args, "use-scaleAlphaVec", std::any(options.useScaleAlphaVec));
        detail::setDataInitArg(args, "use-e", std::any(options.useE));
        detail::setDataInitArg(args, "use-gradient", std::any(options.useGradient));
        detail::setDataInitArg(args, "output-amaxD", std::any(options.outputAmaxD));
        detail::setDataInitArg(args, "use-user-args", std::any(options.useUserArgs));
        detail::setDataInitArg(args, "bias-type-args", std::any(options.biasTypeArgs));
        detail::setDataInitArg(args, "factor-dim-args", std::any(options.factorDimArgs));
        detail::setDataInitArg(args,
                               "streamk-hybrid-mode",
                               std::any(options.streamKHybridMode));
        detail::setDataInitArg(args, "activation-type", std::any(options.activationType));
        detail::setDataInitArg(args, "activation-no-guard", std::any(options.activationNoGuard));
        detail::setDataInitArg(args,
                               "activation-enum-args",
                               std::any(options.activationEnumArgs));
        detail::setDataInitArg(args,
                               "activation-compute-type",
                               std::any(options.activationComputeType));
        detail::setDataInitArg(args,
                               "compute-input-type-A",
                               std::any(options.computeInputTypeA));
        detail::setDataInitArg(args,
                               "compute-input-type-B",
                               std::any(options.computeInputTypeB));
        detail::setDataInitArg(args, "f32-xdl-math-op", std::any(options.f32XdlMathOp));

        detail::setDataInitArg(args,
                               "num-elements-to-validate",
                               std::any(options.numElementsToValidate));
        detail::setDataInitArg(args, "print-valids", std::any(options.printValids));
        detail::setDataInitArg(args, "print-max", std::any(options.printMax));
        detail::setDataInitArg(args, "print-tensor-a", std::any(options.printTensorA));
        detail::setDataInitArg(args, "print-tensor-b", std::any(options.printTensorB));
        detail::setDataInitArg(args, "print-tensor-c", std::any(options.printTensorC));
        detail::setDataInitArg(args, "print-tensor-d", std::any(options.printTensorD));
        detail::setDataInitArg(args, "print-tensor-ref", std::any(options.printTensorRef));
        detail::setDataInitArg(args, "print-tensor-bias", std::any(options.printTensorBias));
        detail::setDataInitArg(args, "print-tensor-amaxd", std::any(options.printTensorAmaxD));
        detail::setDataInitArg(args, "pristine-on-gpu", std::any(options.pristineOnGpu));
        detail::setDataInitArg(args, "prune-mode", std::any(options.pruneMode));
        detail::setDataInitArg(args,
                               "rotating-buffer-size",
                               std::any(options.rotatingBufferSize));
        detail::setDataInitArg(args,
                               "rotating-buffer-mode",
                               std::any(options.rotatingBufferMode));
        detail::setDataInitArg(args, "bounds-check", std::any(options.boundsCheck));
        detail::setDataInitArg(args, "num-benchmarks", std::any(options.numBenchmarks));
        detail::setDataInitArg(args,
                               "num-enqueues-per-sync",
                               std::any(options.numEnqueuesPerSync));
        detail::setDataInitArg(args,
                               "max-enqueues-per-sync",
                               std::any(options.maxEnqueuesPerSync));
        detail::setDataInitArg(args,
                               "min-flops-per-sync",
                               std::any(options.minFlopsPerSync));
        detail::setDataInitArg(args,
                               "num-syncs-per-benchmark",
                               std::any(options.numSyncsPerBenchmark));
        detail::setDataInitArg(args, "num-warmups", std::any(options.numWarmups));

        detail::setDataInitArg(args, "init-a", std::any(options.initA));
        detail::setDataInitArg(args, "init-b", std::any(options.initB));
        detail::setDataInitArg(args, "init-c", std::any(options.initC));
        detail::setDataInitArg(args, "init-d", std::any(options.initD));
        detail::setDataInitArg(args, "init-e", std::any(options.initE));
        detail::setDataInitArg(args, "init-bias", std::any(options.initBias));
        detail::setDataInitArg(args, "init-scaleA", std::any(options.initScaleA));
        detail::setDataInitArg(args, "init-scaleB", std::any(options.initScaleB));
        detail::setDataInitArg(args, "init-scaleC", std::any(options.initScaleC));
        detail::setDataInitArg(args, "init-scaleD", std::any(options.initScaleD));
        detail::setDataInitArg(args,
                               "init-scaleAlphaVec",
                               std::any(options.initScaleAlphaVec));
        detail::setDataInitArg(args, "init-alpha", std::any(options.initAlpha));
        detail::setDataInitArg(args, "init-beta", std::any(options.initBeta));

        return args;
    }

    inline Client::po::variables_map
        buildBaseDataInitArgs(std::vector<std::vector<size_t>> problemSizes)
    {
        DataInitConfig config;
        config.problemSizes = std::move(problemSizes);
        return buildBaseDataInitArgs(config);
    }

    inline Client::po::variables_map
        buildRingArgs(Client::po::variables_map args, int elementsToValidate = 1)
    {
        detail::setDataInitArg(args, "num-benchmarks", std::any(int(0)));
        detail::setDataInitArg(args, "num-enqueues-per-sync", std::any(int(0)));
        detail::setDataInitArg(args, "num-syncs-per-benchmark", std::any(int(0)));
        detail::setDataInitArg(args, "num-warmups", std::any(int(0)));

        detail::setDataInitArg(args,
                               "num-elements-to-validate",
                               std::any(int(elementsToValidate)));
        detail::setDataInitArg(args, "print-valids", std::any(false));
        detail::setDataInitArg(args, "print-max", std::any(int(-1)));

        detail::setDataInitArg(args, "pristine-on-gpu", std::any(true));
        detail::setDataInitArg(args,
                               "bounds-check",
                               std::any(Client::BoundsCheckMode::Disable));
        detail::setDataInitArg(args, "rotating-buffer-size", std::any(int32_t(0)));
        detail::setDataInitArg(args, "rotating-buffer-mode", std::any(int32_t(0)));

        detail::setDataInitArg(args, "sparse", std::any(int(0)));
        detail::setDataInitArg(args,
                               "bias-type-args",
                               std::any(std::vector<rocisa::DataType>{rocisa::DataType::None}));
        detail::setDataInitArg(args, "mx-a-block", std::any(int(0)));
        detail::setDataInitArg(args, "mx-b-block", std::any(int(0)));
        detail::setDataInitArg(args, "mx-a-type", std::any(rocisa::DataType::E8));
        detail::setDataInitArg(args, "mx-b-type", std::any(rocisa::DataType::E8));
        detail::setDataInitArg(args, "mx-scale-format", std::any(int(0)));

        detail::setDataInitArg(args, "init-a", std::any(Client::InitMode::Random));
        detail::setDataInitArg(args, "init-b", std::any(Client::InitMode::Random));
        detail::setDataInitArg(args, "init-c", std::any(Client::InitMode::Random));
        detail::setDataInitArg(args, "init-d", std::any(Client::InitMode::Zero));
        detail::setDataInitArg(args, "init-e", std::any(Client::InitMode::Zero));
        detail::setDataInitArg(args, "init-alpha", std::any(Client::InitMode::Two));
        detail::setDataInitArg(args, "init-beta", std::any(Client::InitMode::Two));
        detail::setDataInitArg(args, "init-bias", std::any(Client::InitMode::One));
        detail::setDataInitArg(args, "init-scaleA", std::any(Client::InitMode::Two));
        detail::setDataInitArg(args, "init-scaleB", std::any(Client::InitMode::Two));
        detail::setDataInitArg(args, "init-scaleC", std::any(Client::InitMode::Two));
        detail::setDataInitArg(args, "init-scaleD", std::any(Client::InitMode::Two));
        detail::setDataInitArg(args,
                               "init-scaleAlphaVec",
                               std::any(Client::InitMode::One));
        detail::setDataInitArg(args, "init-mx-a", std::any(Client::InitMode::One));
        detail::setDataInitArg(args, "init-mx-b", std::any(Client::InitMode::One));

        return args;
    }

    inline Client::po::variables_map
        buildRingArgs(std::vector<std::vector<size_t>> problemSizes,
                      int                               elementsToValidate = 1)
    {
        return buildRingArgs(buildBaseDataInitArgs(std::move(problemSizes)),
                             elementsToValidate);
    }

    // -----------------------------------------------------------------------
    // Plain problem factory
    // -----------------------------------------------------------------------

    struct PlainProblemSpec
    {
        size_t m     = 128;
        size_t n     = 128;
        size_t k     = 256;
        size_t batch = 1;

        bool transA = true;
        bool transB = false;

        rocisa::DataType aType = rocisa::DataType::Float;
        rocisa::DataType bType = rocisa::DataType::Float;
        rocisa::DataType cType = rocisa::DataType::Float;
        rocisa::DataType dType = rocisa::DataType::Float;

        rocisa::DataType computeInputTypeA = rocisa::DataType::Float;
        rocisa::DataType computeInputTypeB = rocisa::DataType::Float;
        rocisa::DataType alphaType         = rocisa::DataType::Float;
        rocisa::DataType betaType          = rocisa::DataType::Float;

        double beta = 0.0;
    };

    inline ContractionProblemGemm makePlainProblem(PlainProblemSpec const& spec = {})
    {
        size_t const lda     = spec.transA ? spec.k : spec.m;
        size_t const aStride = spec.transA ? spec.k * spec.m : spec.m * spec.k;
        size_t const ldb     = spec.transB ? spec.n : spec.k;
        size_t const bStride = spec.transB ? spec.n * spec.k : spec.k * spec.n;
        size_t const ldc     = spec.m;
        size_t const cStride = spec.m * spec.n;
        size_t const ldd     = spec.m;
        size_t const dStride = spec.m * spec.n;

        auto problem = ContractionProblemGemm::GEMM_Strides(
            spec.transA,
            spec.transB,
            spec.aType,
            spec.bType,
            spec.cType,
            spec.dType,
            spec.m,
            spec.n,
            spec.k,
            spec.batch,
            lda,
            aStride,
            ldb,
            bStride,
            ldc,
            cStride,
            ldd,
            dStride,
            spec.beta);

        problem.setComputeInputTypeA(spec.computeInputTypeA);
        problem.setComputeInputTypeB(spec.computeInputTypeB);
        problem.setAlphaType(spec.alphaType);
        problem.setBetaType(spec.betaType);
        problem.setStridedBatched(true);
        return problem;
    }

    inline ContractionProblemGemm makePlainProblem(size_t m, size_t n, size_t k)
    {
        PlainProblemSpec spec;
        spec.m = m;
        spec.n = n;
        spec.k = k;
        return makePlainProblem(spec);
    }

    // -----------------------------------------------------------------------
    // Batched problem factory
    // -----------------------------------------------------------------------

    inline ContractionProblemGemm makeBatchedProblem(size_t m,
                                                     size_t n,
                                                     size_t k,
                                                     size_t batch)
    {
        auto f32 = rocisa::DataType::Float;

        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                             false,
                                                             f32,
                                                             f32,
                                                             f32,
                                                             f32,
                                                             m,
                                                             n,
                                                             k,
                                                             batch,
                                                             m,
                                                             m * k,
                                                             k,
                                                             k * n,
                                                             m,
                                                             m * n,
                                                             m,
                                                             m * n,
                                                             0.0);
        problem.setStridedBatched(false);
        return problem;
    }
} // namespace TensileLite::testing
