// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <Tensile/ContractionProblem.hpp>
#include <cstddef>
#include <optional>

namespace hipblaslt_ext::experimental::jit::detail
{
    struct GemmRequest;
}

namespace hipblaslt_jit
{
    // The TensileLite problem hipBLASLt solves for the request.
    TensileLite::ContractionProblemGemm
        lowerForJit(const hipblaslt_ext::experimental::jit::detail::GemmRequest& request);

    // A single strided batched GEMM in TensileLite's canonical index form.
    struct CanonicalGemm
    {
        bool   transA = false, transB = false;
        bool   conjugateA = false, conjugateB = false;
        size_t m = 0, n = 0, k = 0, batch = 0;
    };

    // Why JIT does not implement the problem, or nullptr when it does. A
    // TensileLite ProblemType has no fused all-to-all, so JIT neither generates
    // nor stores solutions for fused GEMM and all-to-all.
    const char* notImplemented(const TensileLite::ContractionProblemGemm& problem);

    // Throws std::runtime_error when a TensileLite ProblemType cannot describe the
    // problem or notImplemented names a reason.
    CanonicalGemm canonicalGemm(const TensileLite::ContractionProblemGemm& problem);

    // The problem's TensileLite ProblemType. Fields that the problem omits stay
    // empty. Throws like canonicalGemm. problemTypeKey derives the library key
    // from this value.
    struct ProblemType
    {
        // TensileLite's UseScaleAB string: empty, "Scalar" or "Vector".
        enum class Scale
        {
            None,
            Scalar,
            Vector,
        };

        bool batched        = false;
        bool stridedBatched = false;
        bool transposeA     = false;
        bool transposeB     = false;
        bool complexConjugateA = false;
        bool complexConjugateB = false;

        rocisa::DataType dataType        = rocisa::DataType::None;
        rocisa::DataType dataTypeA       = rocisa::DataType::None;
        rocisa::DataType dataTypeB       = rocisa::DataType::None;
        rocisa::DataType macDataTypeA    = rocisa::DataType::None;
        rocisa::DataType macDataTypeB    = rocisa::DataType::None;
        rocisa::DataType destDataType    = rocisa::DataType::None;
        rocisa::DataType computeDataType = rocisa::DataType::None;
        rocisa::DataType f32XdlMathOp    = rocisa::DataType::None;

        bool  highPrecisionAccumulate = false;
        int   useBias                 = 0;
        bool  useE                    = false;
        bool  gradient                = false;
        Scale scale                   = Scale::None;
        bool  useScaleCD              = false;
        int   useScaleAlphaVec        = 0;
        bool  outputAmaxD             = false;
        int   mxBlockA                = 0;
        int   mxBlockB                = 0;
        int   sparse                  = 0;
        bool  swizzleTensorA          = false;
        bool  swizzleTensorB          = false;
        bool  useGateResidual         = false;

        std::optional<rocisa::DataType>                        dataTypeMxsa;
        std::optional<rocisa::DataType>                        dataTypeMxsb;
        std::optional<rocisa::DataType>                        biasDataType;
        std::optional<TensileLite::ContractionProblemGemm::TENSOR> biasSrc;
        std::optional<rocisa::DataType>                        gateResidualDataType;
        std::optional<rocisa::DataType>                        dataTypeE;
        std::optional<rocisa::DataType>                        dataTypeAmaxD;
        std::optional<TensileLite::ActivationType>             activation;
        std::optional<rocisa::DataType>                        activationComputeDataType;
    };

    ProblemType problemTypeFields(const TensileLite::ContractionProblemGemm& problem);
}
