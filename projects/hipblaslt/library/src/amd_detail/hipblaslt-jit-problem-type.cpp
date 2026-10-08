// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-problem-type.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <stdexcept>
#include <string>

namespace hipblaslt_jit
{
    namespace
    {
        void require(bool condition, const char* message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }

        ProblemType::Scale scaleOf(const std::string& value)
        {
            if(value.empty())
                return ProblemType::Scale::None;
            if(value == "Scalar")
                return ProblemType::Scale::Scalar;
            if(value == "Vector")
                return ProblemType::Scale::Vector;
            throw std::runtime_error("UseScaleAB '" + value + "' is not none, scalar or vector");
        }
    }

    const char* notImplemented(const TensileLite::ContractionProblemGemm& problem)
    {
        return problem.fusedGemmA2A() ? "fused GEMM and all-to-all is not implemented" : nullptr;
    }

    CanonicalGemm canonicalGemm(const TensileLite::ContractionProblemGemm& problem)
    {
        if(const auto reason = notImplemented(problem))
            throw std::runtime_error(reason);
        require(problem.stridedBatched() && !problem.groupedGemm(),
                "the JIT solution library stores a single strided GEMM; grouped GEMM is not "
                "implemented");
        require(problem.c().dataType() == problem.d().dataType(),
                "TensileLite ProblemType requires matching C and D datatypes");
        const auto conjugate = [](const TensileLite::TensorOps& ops) {
            require(ops.empty()
                        || (ops.size() == 1
                            && ops.front().type == TensileLite::TensorOp::Type::ComplexConjugate),
                    "the input tensor operation cannot be represented by TensileLite ProblemType");
            return !ops.empty();
        };
        CanonicalGemm gemm;
        gemm.conjugateA = conjugate(problem.aOps());
        gemm.conjugateB = conjugate(problem.bOps());
        require(problem.cOps().empty() && problem.dOps().empty(),
                "TensileLite ProblemType does not describe C/D tensor operations");
        require(problem.freeIndicesA().size() == 1 && problem.freeIndicesB().size() == 1
                    && problem.boundIndices().size() == 1 && problem.batchIndices().size() == 1
                    && !problem.transposeC01(),
                "expected canonical batched GEMM indices");
        for(const auto* tensor : {&problem.a(), &problem.b(), &problem.c(), &problem.d()})
            require(tensor->dimensions() == 3 && tensor->strides().at(0) == 1,
                    "expected three-dimensional column-major tensor descriptors");
        gemm.transA = problem.freeIndicesA()[0].i == 1;
        gemm.transB = problem.freeIndicesB()[0].i == 0;
        gemm.m      = problem.freeSizeA(0);
        gemm.n      = problem.freeSizeB(0);
        gemm.k      = problem.boundSize(0);
        gemm.batch  = problem.batchSize(0);
        return gemm;
    }

    ProblemType problemTypeFields(const TensileLite::ContractionProblemGemm& problem)
    {
        const auto gemm = canonicalGemm(problem);
        ProblemType fields;
        fields.batched             = true;
        fields.stridedBatched      = true;
        fields.transposeA          = gemm.transA;
        fields.transposeB          = gemm.transB;
        fields.complexConjugateA   = gemm.conjugateA;
        fields.complexConjugateB   = gemm.conjugateB;
        fields.dataType            = problem.computeInputTypeA();
        fields.dataTypeA           = problem.a().dataType();
        fields.dataTypeB           = problem.b().dataType();
        fields.macDataTypeA        = problem.computeInputTypeA();
        fields.macDataTypeB        = problem.computeInputTypeB();
        fields.destDataType        = problem.d().dataType();
        fields.computeDataType     = problem.computeType();
        fields.f32XdlMathOp        = problem.f32XdlMathOp();
        fields.highPrecisionAccumulate = problem.highPrecisionAccumulate();
        fields.useBias             = problem.useBias();
        fields.useE                = problem.useE();
        fields.gradient            = problem.useGradient();
        fields.scale               = scaleOf(problem.useScaleAB());
        fields.useScaleCD          = problem.useScaleCD();
        fields.useScaleAlphaVec    = problem.useScaleAlphaVec();
        fields.outputAmaxD         = problem.outputAmaxD();
        fields.mxBlockA            = static_cast<int>(problem.mxBlockA());
        fields.mxBlockB            = static_cast<int>(problem.mxBlockB());
        fields.sparse              = problem.sparse();
        fields.swizzleTensorA      = problem.swizzleTensorA();
        fields.swizzleTensorB      = problem.swizzleTensorB();
        fields.useGateResidual     = problem.useGateResidual();
        if(problem.mxBlockA())
            fields.dataTypeMxsa = problem.mxTypeA();
        if(problem.mxBlockB())
            fields.dataTypeMxsb = problem.mxTypeB();
        if(problem.useBias())
        {
            fields.biasDataType = problem.bias().dataType();
            fields.biasSrc      = problem.biasSrc();
        }
        if(problem.useGateResidual())
            fields.gateResidualDataType = problem.gateResidual().dataType();
        if(problem.useE())
            fields.dataTypeE = problem.e().dataType();
        if(problem.outputAmaxD())
            fields.dataTypeAmaxD = problem.amaxd().dataType();
        if(problem.activationType() != TensileLite::ActivationType::None)
        {
            fields.activation                  = problem.activationType();
            fields.activationComputeDataType   = problem.activationComputeType();
        }
        return fields;
    }
}
