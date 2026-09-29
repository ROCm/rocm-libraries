// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-problem-type.hpp"
#include "hipblaslt-jit-json.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <stdexcept>

namespace hipblaslt_jit
{
    namespace
    {
        void require(bool condition, const char* message)
        {
            if(!condition)
                throw std::runtime_error(message);
        }
    }

    CanonicalGemm canonicalGemm(const TensileLite::ContractionProblemGemm& problem)
    {
        require(problem.stridedBatched() && !problem.groupedGemm(),
                "prediction requires a single strided GEMM; grouped GEMM is not implemented");
        require(problem.c().dataType() == problem.d().dataType(),
                "Tensile ProblemType requires matching C and D datatypes");
        const auto conjugate = [](const TensileLite::TensorOps& ops) {
            require(ops.empty()
                        || (ops.size() == 1
                            && ops.front().type == TensileLite::TensorOp::Type::ComplexConjugate),
                    "the input tensor operation cannot be represented by Tensile ProblemType");
            return !ops.empty();
        };
        CanonicalGemm gemm;
        gemm.conjugateA = conjugate(problem.aOps());
        gemm.conjugateB = conjugate(problem.bOps());
        require(problem.cOps().empty() && problem.dOps().empty(),
                "Tensile ProblemType does not describe C/D tensor operations");
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

    std::vector<std::pair<std::string, std::string>>
        problemTypeFields(const TensileLite::ContractionProblemGemm& problem)
    {
        using json::literal;
        using Type      = rocisa::DataType;
        const auto gemm = canonicalGemm(problem);
        std::vector<std::pair<std::string, std::string>> fields{
            {"OperationType", json::quote("GEMM")},
            {"Batched", literal(true)},
            {"StridedBatched", literal(true)},
            {"TransposeA", literal(gemm.transA)},
            {"TransposeB", literal(gemm.transB)},
            {"ComplexConjugateA", literal(gemm.conjugateA)},
            {"ComplexConjugateB", literal(gemm.conjugateB)},
        };
        const auto add = [&](const char* name, std::string value) {
            fields.emplace_back(name, std::move(value));
        };
        const auto dataType
            = [&](const char* name, Type value) { add(name, literal(static_cast<int>(value))); };
        dataType("DataType", problem.computeInputTypeA());
        dataType("DataTypeA", problem.a().dataType());
        dataType("DataTypeB", problem.b().dataType());
        dataType("MacDataTypeA", problem.computeInputTypeA());
        dataType("MacDataTypeB", problem.computeInputTypeB());
        dataType("DestDataType", problem.d().dataType());
        dataType("ComputeDataType", problem.computeType());
        dataType("F32XdlMathOp", problem.f32XdlMathOp());
        if(problem.mxBlockA())
            dataType("DataTypeMXSA", problem.mxTypeA());
        if(problem.mxBlockB())
            dataType("DataTypeMXSB", problem.mxTypeB());
        add("HighPrecisionAccumulate", literal(problem.highPrecisionAccumulate()));
        add("UseBias", literal(problem.useBias()));
        add("UseE", literal(problem.useE()));
        add("Gradient", literal(problem.useGradient()));
        add("UseScaleAB", json::quote(problem.useScaleAB()));
        add("UseScaleCD", literal(problem.useScaleCD()));
        add("UseScaleAlphaVec", literal(problem.useScaleAlphaVec()));
        add("OutputAmaxD", literal(problem.outputAmaxD()));
        add("MXBlockA", literal(problem.mxBlockA()));
        add("MXBlockB", literal(problem.mxBlockB()));
        add("Sparse", literal(problem.sparse()));
        add("SwizzleTensorA", literal(problem.swizzleTensorA()));
        add("SwizzleTensorB", literal(problem.swizzleTensorB()));
        add("UseGateResidual", literal(problem.useGateResidual()));
        if(problem.useBias())
        {
            add("BiasDataTypeList",
                "[" + literal(static_cast<int>(problem.bias().dataType())) + "]");
            add("BiasSrc", json::quote(std::string(1, 'A' + problem.biasSrc())));
        }
        if(problem.useGateResidual())
            add("GateResidualDataTypeList",
                "[" + literal(static_cast<int>(problem.gateResidual().dataType())) + "]");
        if(problem.useE())
            dataType("DataTypeE", problem.e().dataType());
        if(problem.outputAmaxD())
            dataType("DataTypeAmaxD", problem.amaxd().dataType());
        if(problem.activationType() != TensileLite::ActivationType::None)
        {
            add("Activation", literal(true));
            add("ActivationType", json::quote("hipblaslt_all"));
            dataType("ActivationComputeDataType", problem.activationComputeType());
        }
        return fields;
    }
}
