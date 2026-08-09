/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Product-private TensileLite descriptor adapter for roc::host-validation.

#include <cmath>
#include <complex>
#include <cstddef>
#include <iostream>
#include <optional>
#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>
#include <roc/host_validation/adapters/tensilelite/Reference.hpp>
#include <roc/host_validation/backends/tiled.hpp>
#include <roc/host_validation/validation.hpp>
#include <span>
#include <utility>

#include "Tensile/TensorDescriptor_fwd.hpp"
#include "Tensile/Utils.hpp"
#include "TimingInstrumentation.hpp"

namespace TensileLite
{

    namespace
    {

        /** One MX scale tensor element as float (E8 / E5M3 / Float8 E4M3).
         *
         * Returns the magnitude of the scale (std::fabs). MX scales are interpreted
         * as positive multipliers per the OCP MX spec; for the canonical UE8M0
         * (E8) type this is a no-op, but for E5M3 / Float8 (E4M3) used as scale
         * elements it preserves the prior reference behaviour that explicitly
         * applied abs() before folding the scale into the accumulator.
         */
        inline float mxScaleElementAsFloat(rocisa::DataType mxType, void const* base, size_t index)
        {
            float v;
            switch(mxType)
            {
            case rocisa::DataType::E8:
                v = static_cast<float>(static_cast<E8 const*>(base)[index]);
                break;
            case rocisa::DataType::E5M3:
                v = static_cast<float>(static_cast<E5M3 const*>(base)[index]);
                break;
            case rocisa::DataType::Float8:
                v = static_cast<float>(static_cast<Float8 const*>(base)[index]);
                break;
            default:
                throw std::runtime_error(concatenate(
                    "Reference MX scale: unsupported element type ", static_cast<int>(mxType)));
            }
            return std::fabs(v);
        }

    }

    namespace Client
    {
        inline bool isMXFP4Problem(const ContractionProblemGemm& problem)
        {
            return (problem.a().dataType() == rocisa::DataType::Float4
                    && problem.mxBlockA() > 0)
                || (problem.b().dataType() == rocisa::DataType::Float4
                    && problem.mxBlockB() > 0);
        }

        bool tryRuntimeGemm(
            ContractionProblemGemm const& problem,
            ContractionInputs const& inputs,
            size_t elementsToValidate,
            const roc::host_validation::GemmBackendImplementation* backendImplementation) {
            using namespace roc::host_validation;

            if (problem.boundIndices().size() != 1 || problem.freeIndicesA().size() != 1 ||
                problem.freeIndicesB().size() != 1 || problem.batchIndices().size() != 1)
                return false;
            if (problem.useGradient() && problem.useBias() &&
                problem.biasSrc() != ContractionProblemGemm::A &&
                problem.biasSrc() != ContractionProblemGemm::B &&
                problem.biasSrc() != ContractionProblemGemm::D)
                return false;
            const bool useStandaloneEpilogue = problem.useGradient() ||
                                               problem.outputAmaxD() || problem.useE() ||
                                               problem.useScaleCD() ||
                                               problem.useGateResidual();
            if ((problem.mxBlockA() > 0) != (problem.mxBlockB() > 0)) return false;
            if ((problem.useBias() && inputs.bias == nullptr && inputs.batchBias == nullptr) ||
                (problem.useScaleAlphaVec() && inputs.scaleAlphaVec == nullptr) ||
                ((problem.useScaleAB() == "Scalar" || problem.useScaleAB() == "Vector") &&
                 (inputs.scaleA == nullptr || inputs.scaleB == nullptr)) ||
                (problem.mxBlockA() > 0 && (inputs.mxsa == nullptr || inputs.mxsb == nullptr)))
                return false;
            if ((problem.useE() && inputs.e == nullptr) ||
                (problem.outputAmaxD() && inputs.amaxD == nullptr) ||
                (problem.useScaleCD() &&
                 (inputs.scaleC == nullptr || inputs.scaleD == nullptr)) ||
                (problem.useGateResidual() && inputs.gateResidual == nullptr &&
                 inputs.batchGateResidual == nullptr))
                return false;
            if ((inputs.a == nullptr && inputs.batchA == nullptr) ||
                (inputs.b == nullptr && inputs.batchB == nullptr) ||
                (inputs.c == nullptr && inputs.batchC == nullptr) ||
                (inputs.d == nullptr && inputs.batchD == nullptr))
                return false;
            ScalarType typeA;
            ScalarType typeB;
            ScalarType typeC;
            ScalarType typeD;
            ScalarType accumulatorType;
            ScalarType computeTypeA;
            ScalarType computeTypeB;
            ScalarType betaType;
            try {
                typeA = toHostValidationScalarType(problem.a().dataType());
                typeB = toHostValidationScalarType(problem.b().dataType());
                typeC = toHostValidationScalarType(problem.c().dataType());
                typeD = toHostValidationScalarType(problem.d().dataType());
                accumulatorType = toHostValidationScalarType(problem.computeType());
                betaType = toHostValidationScalarType(problem.betaType());
                computeTypeA = problem.computeInputTypeA() == rocisa::DataType::None
                                   ? typeA
                                   : toHostValidationScalarType(problem.computeInputTypeA());
                computeTypeB = problem.computeInputTypeB() == rocisa::DataType::None
                                   ? typeB
                                   : toHostValidationScalarType(problem.computeInputTypeB());
            } catch (std::invalid_argument const&) {
                return false;
            }
            if (backendImplementation != nullptr &&
                backendImplementation->backend() == GemmBackend::Tiled &&
                (accumulatorType == ScalarType::Float16 ||
                 accumulatorType == ScalarType::BFloat16))
                accumulatorType = ScalarType::Float32;
            if (accumulatorType != ScalarType::Float16 &&
                accumulatorType != ScalarType::BFloat16 &&
                accumulatorType != ScalarType::Float32 && accumulatorType != ScalarType::Float64 &&
                accumulatorType != ScalarType::Int32 &&
                accumulatorType != ScalarType::ComplexFloat32 &&
                accumulatorType != ScalarType::ComplexFloat64)
                return false;
            if (useStandaloneEpilogue && accumulatorType != ScalarType::Float32 &&
                accumulatorType != ScalarType::Float64 &&
                accumulatorType != ScalarType::Int32)
                return false;
            const bool preQuantizationScaleA =
                scalarTypeInfo(typeA).storageBits > scalarTypeInfo(computeTypeA).storageBits;
            const bool preQuantizationScaleB =
                scalarTypeInfo(typeB).storageBits > scalarTypeInfo(computeTypeB).storageBits;
            auto scalarValue = [](rocisa::DataType type,
                                  ConstantVariant const& value) -> std::complex<double> {
                switch (type) {
                    case rocisa::DataType::Float:
                    case rocisa::DataType::XFloat32:
                        return {constVariantCast<float>(value), 0.0};
                    case rocisa::DataType::Double:
                        return {constVariantCast<double>(value), 0.0};
                    case rocisa::DataType::Half:
                        return {static_cast<double>(constVariantCast<Half>(value)), 0.0};
                    case rocisa::DataType::BFloat16:
                        return {static_cast<double>(constVariantCast<BFloat16>(value)), 0.0};
                    case rocisa::DataType::Int32:
                        return {static_cast<double>(constVariantCast<int32_t>(value)), 0.0};
                    case rocisa::DataType::ComplexFloat: {
                        auto converted = constVariantCast<std::complex<float>>(value);
                        return {converted.real(), converted.imag()};
                    }
                    case rocisa::DataType::ComplexDouble:
                        return constVariantCast<std::complex<double>>(value);
                    default:
                        throw std::invalid_argument(
                            "Runtime canonical bridge scalar type is unsupported.");
                }
            };
            std::complex<double> alpha;
            std::complex<double> beta;
            try {
                alpha = scalarValue(problem.alphaType(), inputs.alpha);
                beta = scalarValue(problem.betaType(), inputs.beta);
            } catch (std::invalid_argument const&) {
                return false;
            }

            ActivationType concreteActivation = problem.activationType();
            if (concreteActivation == ActivationType::All ||
                concreteActivation == ActivationType::Hipblaslt_all)
                concreteActivation = problem.getParams().activationEnum();
            roc::host_validation::Activation activation;
            try {
                activation =
                    toHostValidationActivation(concreteActivation, problem.useGradient());
            } catch (std::invalid_argument const&) {
                return false;
            }
            if ((accumulatorType == ScalarType::ComplexFloat32 ||
                 accumulatorType == ScalarType::ComplexFloat64) &&
                activation != roc::host_validation::Activation::None)
                return false;

            auto storageSpan = [](ScalarType type, void const* pointer, size_t elements) {
                const size_t bits = scalarTypeInfo(type).storageBits;
                const size_t bytes = (elements * bits + 7) / 8;
                return std::span<const std::byte>(static_cast<const std::byte*>(pointer), bytes);
            };
            auto scalarFromStorage = [&](ScalarType type,
                                         void const* pointer) -> std::complex<double> {
                TensorView view(type, Layout::contiguous(Shape{1}), storageSpan(type, pointer, 1));
                if (scalarTypeInfo(type).category == ScalarCategory::Complex)
                    return view.loadAs<std::complex<double>>({0});
                return {view.loadAs<double>({0}), 0.0};
            };

            ScalarType alphaType;
            try {
                alphaType = toHostValidationScalarType(problem.alphaType());
            } catch (std::invalid_argument const&) {
                return false;
            }
            if (problem.useScaleAB() == "Scalar") {
                if (!preQuantizationScaleA)
                    alpha *= scalarFromStorage(alphaType, inputs.scaleA);
                if (!preQuantizationScaleB)
                    alpha *= scalarFromStorage(alphaType, inputs.scaleB);
            }

            std::complex<double> outputScale = {1.0, 0.0};
            if (problem.useScaleCD()) {
                if (beta != std::complex<double>(0.0, 0.0))
                    beta *= scalarFromStorage(betaType, inputs.scaleC);
                outputScale = scalarFromStorage(betaType, inputs.scaleD);
            }

            double activationParameter0 = 0.0;
            double activationParameter1 = 0.0;
            try {
                if (!inputs.activationArgs.empty())
                    activationParameter0 =
                        scalarValue(problem.computeType(), inputs.activationArgs[0]).real();
                if (inputs.activationArgs.size() > 1)
                    activationParameter1 =
                        scalarValue(problem.computeType(), inputs.activationArgs[1]).real();
            } catch (std::invalid_argument const&) {
                return false;
            }

            const size_t indexMA = problem.freeIndicesA()[0].i;
            const size_t indexKA = problem.boundIndices()[0].a;
            const size_t indexNB = problem.freeIndicesB()[0].i;
            const size_t indexKB = problem.boundIndices()[0].b;
            const size_t indexMD = problem.freeIndices()[0].d;
            const size_t indexND = problem.freeIndices()[1].d;
            const size_t batchA = problem.batchIndices()[0].a;
            const size_t batchB = problem.batchIndices()[0].b;
            const size_t batchC = problem.batchIndices()[0].c;
            const size_t batchD = problem.batchIndices()[0].d;
            const size_t mxBlockA = problem.mxBlockA();
            const size_t mxBlockB = problem.mxBlockB();
            const size_t strideMxsaM = mxBlockA > 0 ? problem.mxsa().strides()[indexMA] : 0;
            const size_t strideMxsaBlock = mxBlockA > 0 ? problem.mxsa().strides()[indexKA] : 0;
            const size_t strideMxsbN = mxBlockB > 0 ? problem.mxsb().strides()[indexNB] : 0;
            const size_t strideMxsbBlock = mxBlockB > 0 ? problem.mxsb().strides()[indexKB] : 0;
            const size_t strideBatchMxsa = mxBlockA > 0 ? problem.mxsa().strides()[batchA] : 0;
            const size_t strideBatchMxsb = mxBlockB > 0 ? problem.mxsb().strides()[batchB] : 0;

            const size_t m = problem.freeSizeA(0);
            const size_t n = problem.freeSizeB(0);
            const size_t k = problem.boundSize(0);
            const size_t batches = problem.batchSize(0);
            const OutputSelection globalSelection =
                problem.useGradient() && problem.useBias()
                    ? OutputSelection::all()
                    : OutputSelection::primeStride(problem.d().totalLogicalElements(),
                                                   problem.d().totalAllocatedElements(),
                                                   elementsToValidate);
            std::vector<std::vector<size_t>> selectedByBatch;
            if(!globalSelection.selectsAll())
            {
                selectedByBatch.resize(batches);
                for(size_t logicalIndex :
                    globalSelection.indices(problem.d().totalLogicalElements()))
                {
                    std::vector<int64_t> coordinate(problem.d().dimensions());
                    CoordNumbered(logicalIndex,
                                  coordinate.begin(),
                                  coordinate.end(),
                                  problem.d().sizes().begin(),
                                  problem.d().sizes().end());
                    const size_t selectedBatch = static_cast<size_t>(coordinate[batchD]);
                    const size_t row           = static_cast<size_t>(coordinate[indexMD]);
                    const size_t column        = static_cast<size_t>(coordinate[indexND]);
                    selectedByBatch.at(selectedBatch).push_back(row * n + column);
                }
            }
            bool aConjugate = false;
            bool bConjugate = false;
            for (auto const& op : problem.aOps())
                if (op.type == TensorOp::Type::ComplexConjugate) aConjugate = true;
            for (auto const& op : problem.bOps())
                if (op.type == TensorOp::Type::ComplexConjugate) bConjugate = true;
            auto descriptorStorageBytes = [](ScalarType type,
                                             TensorDescriptor const& descriptor) {
                std::vector<ptrdiff_t> strides(descriptor.strides().begin(),
                                               descriptor.strides().end());
                return storageBytesForLayout(
                    type, Layout(Shape(descriptor.sizes()), std::move(strides)));
            };
            const auto aStorage = inputs.batchA == nullptr
                ? std::span<const std::byte>(
                      static_cast<const std::byte*>(inputs.a) + inputs.batchOffsetA,
                      descriptorStorageBytes(typeA, problem.a()))
                : std::span<const std::byte>{};
            const auto bStorage = inputs.batchB == nullptr
                ? std::span<const std::byte>(
                      static_cast<const std::byte*>(inputs.b) + inputs.batchOffsetB,
                      descriptorStorageBytes(typeB, problem.b()))
                : std::span<const std::byte>{};
            const auto cStorage = inputs.batchC == nullptr
                ? std::span<const std::byte>(
                      static_cast<const std::byte*>(inputs.c) + inputs.batchOffsetC,
                      descriptorStorageBytes(typeC, problem.c()))
                : std::span<const std::byte>{};
            const auto dStorage = inputs.batchD == nullptr
                ? std::span<std::byte>(
                      static_cast<std::byte*>(inputs.d) + inputs.batchOffsetD,
                      descriptorStorageBytes(typeD, problem.d()))
                : std::span<std::byte>{};
            std::optional<ScalarType> biasType;
            std::span<const std::byte> biasStorage;
            std::span<std::byte> biasOutputStorage;
            if (problem.useBias()) {
                try {
                    biasType = toHostValidationScalarType(problem.bias().dataType());
                } catch (std::invalid_argument const&) {
                    return false;
                }
                if (inputs.batchBias == nullptr)
                    biasStorage =
                        std::span<const std::byte>(static_cast<const std::byte*>(inputs.bias),
                                                   descriptorStorageBytes(*biasType,
                                                                          problem.bias()));
                if (problem.useGradient() && inputs.batchBias == nullptr)
                    biasOutputStorage = std::span<std::byte>(
                        static_cast<std::byte*>(const_cast<void*>(inputs.bias)),
                        descriptorStorageBytes(*biasType, problem.bias()));
            }
            std::optional<ScalarType> auxiliaryType;
            std::span<std::byte> auxiliaryStorage;
            if (problem.useE()) {
                auto const& auxiliary =
                    problem.tensors()[ContractionProblemGemm::TENSOR::E];
                try {
                    auxiliaryType = toHostValidationScalarType(auxiliary.dataType());
                } catch (std::invalid_argument const&) {
                    return false;
                }
                auxiliaryStorage = std::span<std::byte>(
                    static_cast<std::byte*>(inputs.e),
                    descriptorStorageBytes(*auxiliaryType, auxiliary));
            }
            std::optional<ScalarType> gateType;
            std::span<const std::byte> gateStorage;
            if (problem.useGateResidual()) {
                auto const& gate =
                    problem.tensors()[ContractionProblemGemm::TENSOR::GATE_RESIDUAL];
                try {
                    gateType = toHostValidationScalarType(gate.dataType());
                } catch (std::invalid_argument const&) {
                    return false;
                }
                if (inputs.batchGateResidual == nullptr)
                    gateStorage = std::span<const std::byte>(
                        static_cast<const std::byte*>(inputs.gateResidual),
                        descriptorStorageBytes(*gateType, gate));
            }
            std::optional<ScalarType> amaxType;
            std::span<std::byte> amaxStorage;
            if (problem.outputAmaxD()) {
                auto const& amax =
                    problem.tensors()[ContractionProblemGemm::TENSOR::AMAXD];
                try {
                    amaxType = toHostValidationScalarType(amax.dataType());
                } catch (std::invalid_argument const&) {
                    return false;
                }
                amaxStorage = std::span<std::byte>(
                    static_cast<std::byte*>(inputs.amaxD),
                    descriptorStorageBytes(*amaxType, amax));
            }
            const size_t scaleAlphaLength = problem.getParams().factorDim() == 0 ? m : n;
            std::span<const std::byte> scaleAlphaStorage;
            if (problem.useScaleAlphaVec())
                scaleAlphaStorage = storageSpan(alphaType, inputs.scaleAlphaVec, scaleAlphaLength);
            std::span<const std::byte> scaleAStorage;
            std::span<const std::byte> scaleBStorage;
            if (problem.useScaleAB() == "Vector") {
                scaleAStorage = storageSpan(alphaType, inputs.scaleA, m);
                scaleBStorage = storageSpan(alphaType, inputs.scaleB, n);
            }

            for (size_t batch = 0; batch < batches; ++batch) {
                if ((inputs.batchA != nullptr && inputs.batchA[batch] == nullptr) ||
                    (inputs.batchB != nullptr && inputs.batchB[batch] == nullptr) ||
                    (inputs.batchC != nullptr && inputs.batchC[batch] == nullptr) ||
                    (inputs.batchD != nullptr && inputs.batchD[batch] == nullptr) ||
                    (problem.useBias() && inputs.batchBias != nullptr &&
                     inputs.batchBias[batch] == nullptr) ||
                    (problem.useGateResidual() && inputs.batchGateResidual != nullptr &&
                     inputs.batchGateResidual[batch] == nullptr))
                    return false;
                ptrdiff_t offsetA = inputs.batchA == nullptr
                    ? static_cast<ptrdiff_t>(batch * problem.a().strides()[batchA])
                    : 0;
                ptrdiff_t offsetB = inputs.batchB == nullptr
                    ? static_cast<ptrdiff_t>(batch * problem.b().strides()[batchB])
                    : 0;
                const ptrdiff_t offsetC = inputs.batchC == nullptr
                    ? static_cast<ptrdiff_t>(batch * problem.c().strides()[batchC])
                    : 0;
                const ptrdiff_t offsetD = inputs.batchD == nullptr
                    ? static_cast<ptrdiff_t>(batch * problem.d().strides()[batchD])
                    : 0;
                ptrdiff_t strideKA =
                    static_cast<ptrdiff_t>(problem.a().strides()[indexKA]);
                ptrdiff_t strideKB =
                    static_cast<ptrdiff_t>(problem.b().strides()[indexKB]);
                if (problem.boundIndices()[0].aMirror && k != 0) {
                    offsetA += static_cast<ptrdiff_t>(k - 1) * strideKA;
                    strideKA = -strideKA;
                }
                if (problem.boundIndices()[0].bMirror && k != 0) {
                    offsetB += static_cast<ptrdiff_t>(k - 1) * strideKB;
                    strideKB = -strideKB;
                }
                const Layout layoutA(
                    Shape{m, k},
                    {static_cast<ptrdiff_t>(problem.a().strides()[indexMA]), strideKA},
                    offsetA);
                const Layout layoutB(
                    Shape{k, n},
                    {strideKB, static_cast<ptrdiff_t>(problem.b().strides()[indexNB])},
                    offsetB);
                const Layout layoutC(
                    Shape{m, n},
                    {static_cast<ptrdiff_t>(problem.c().strides()[indexMD]),
                     static_cast<ptrdiff_t>(problem.c().strides()[indexND])},
                    offsetC);
                const Layout layoutD(
                    Shape{m, n},
                    {static_cast<ptrdiff_t>(problem.d().strides()[indexMD]),
                     static_cast<ptrdiff_t>(problem.d().strides()[indexND])},
                    offsetD);
                const auto currentAStorage = inputs.batchA == nullptr
                    ? aStorage
                    : std::span<const std::byte>(
                          static_cast<const std::byte*>(inputs.batchA[batch]) +
                              inputs.batchOffsetA,
                          storageBytesForLayout(typeA, layoutA));
                const auto currentBStorage = inputs.batchB == nullptr
                    ? bStorage
                    : std::span<const std::byte>(
                          static_cast<const std::byte*>(inputs.batchB[batch]) +
                              inputs.batchOffsetB,
                          storageBytesForLayout(typeB, layoutB));
                const auto currentCStorage = inputs.batchC == nullptr
                    ? cStorage
                    : std::span<const std::byte>(
                          static_cast<const std::byte*>(inputs.batchC[batch]) +
                              inputs.batchOffsetC,
                          storageBytesForLayout(typeC, layoutC));
                const auto currentDStorage = inputs.batchD == nullptr
                    ? dStorage
                    : std::span<std::byte>(
                          static_cast<std::byte*>(inputs.batchD[batch]) +
                              inputs.batchOffsetD,
                          storageBytesForLayout(typeD, layoutD));
                GemmOperand operandA(
                    TensorView(typeA, layoutA, currentAStorage));
                GemmOperand operandB(
                    TensorView(typeB, layoutB, currentBStorage));
                if (computeTypeA != typeA) operandA.computeType = computeTypeA;
                if (computeTypeB != typeB) operandB.computeType = computeTypeB;
                if (problem.useScaleAB() == "Scalar" && preQuantizationScaleA)
                    operandA.preQuantizationScales.push_back(VectorBinding{
                        TensorView(alphaType,
                                   Layout::contiguous(Shape{1}),
                                   storageSpan(alphaType, inputs.scaleA, 1)),
                        MatrixAxis::Row});
                if (problem.useScaleAB() == "Scalar" && preQuantizationScaleB)
                    operandB.preQuantizationScales.push_back(VectorBinding{
                        TensorView(alphaType,
                                   Layout::contiguous(Shape{1}),
                                   storageSpan(alphaType, inputs.scaleB, 1)),
                        MatrixAxis::Column});
                if (problem.useScaleAB() == "Vector" && preQuantizationScaleA)
                    operandA.preQuantizationScales.push_back(VectorBinding{
                        TensorView(alphaType, Layout::contiguous(Shape{m}), scaleAStorage),
                        MatrixAxis::Row});
                if (problem.useScaleAB() == "Vector" && preQuantizationScaleB)
                    operandB.preQuantizationScales.push_back(VectorBinding{
                        TensorView(alphaType, Layout::contiguous(Shape{n}), scaleBStorage),
                        MatrixAxis::Column});
                operandA.conjugate = aConjugate;
                operandB.conjugate = bConjugate;
                std::optional<Tensor> runtimeScaleA;
                std::optional<Tensor> runtimeScaleB;
                if (mxBlockA > 0) {
                    const size_t blockCountA = k / mxBlockA + (k % mxBlockA != 0 ? 1 : 0);
                    const size_t blockCountB = k / mxBlockB + (k % mxBlockB != 0 ? 1 : 0);
                    runtimeScaleA.emplace(ScalarType::Float32, Shape{m, blockCountA});
                    runtimeScaleB.emplace(ScalarType::Float32, Shape{n, blockCountB});
                    for (size_t row = 0; row < m; ++row) {
                        for (size_t block = 0; block < blockCountA; ++block) {
                            const size_t sourceBlock = problem.boundIndices()[0].aMirror
                                ? blockCountA - 1 - block
                                : block;
                            const size_t index = batch * strideBatchMxsa + row * strideMxsaM +
                                                 sourceBlock * strideMxsaBlock;
                            runtimeScaleA->mutableView().storeFrom(
                                {row, block},
                                mxScaleElementAsFloat(problem.mxTypeA(), inputs.mxsa, index));
                        }
                    }
                    for (size_t column = 0; column < n; ++column) {
                        for (size_t block = 0; block < blockCountB; ++block) {
                            const size_t sourceBlock = problem.boundIndices()[0].bMirror
                                ? blockCountB - 1 - block
                                : block;
                            const size_t index = batch * strideBatchMxsb + column * strideMxsbN +
                                                 sourceBlock * strideMxsbBlock;
                            runtimeScaleB->mutableView().storeFrom(
                                {column, block},
                                mxScaleElementAsFloat(problem.mxTypeB(), inputs.mxsb, index));
                        }
                    }
                    operandA.blockScale = BlockScaleBinding{runtimeScaleA->view(), mxBlockA};
                    operandB.blockScale = BlockScaleBinding{runtimeScaleB->view(), mxBlockB};
                }

                const TensorView logicalA = operandA.values;
                const TensorView logicalB = operandB.values;
                MutableTensorView productOutput(
                    typeD, layoutD, currentDStorage);
                std::optional<Tensor> intermediate;
                MutableTensorView gemmOutput = productOutput;
                if (useStandaloneEpilogue) {
                    intermediate.emplace(accumulatorType, Shape{m, n});
                    gemmOutput = intermediate->mutableView();
                }

                GemmProblem runtimeProblem(
                    std::move(operandA), std::move(operandB),
                    TensorView(typeC, layoutC, currentCStorage),
                    gemmOutput,
                    accumulatorType);
                runtimeProblem.epilogue.alpha = alpha;
                runtimeProblem.epilogue.beta = beta;
                if (!useStandaloneEpilogue) {
                    runtimeProblem.epilogue.activation = activation;
                    runtimeProblem.epilogue.activationParameter0 = activationParameter0;
                    runtimeProblem.epilogue.activationParameter1 = activationParameter1;
                }
                if (problem.useScaleAlphaVec())
                    runtimeProblem.epilogue.scaleAlpha = VectorBinding{
                        TensorView(alphaType, Layout::contiguous(Shape{scaleAlphaLength}),
                                   scaleAlphaStorage),
                        problem.getParams().factorDim() == 0 ? MatrixAxis::Row
                                                             : MatrixAxis::Column};
                if (problem.useScaleAB() == "Vector") {
                    if (!preQuantizationScaleA)
                        runtimeProblem.epilogue.scaleA =
                            TensorView(alphaType, Layout::contiguous(Shape{m}), scaleAStorage);
                    if (!preQuantizationScaleB)
                        runtimeProblem.epilogue.scaleB =
                            TensorView(alphaType, Layout::contiguous(Shape{n}), scaleBStorage);
                }
                std::optional<VectorBinding> runtimeBias;
                ptrdiff_t runtimeBiasOffset = 0;
                size_t runtimeBiasLength = 0;
                MatrixAxis runtimeBiasAxis = MatrixAxis::Row;
                std::span<std::byte> currentBiasOutputStorage = biasOutputStorage;
                if (problem.useBias()) {
                    std::vector<int64_t> biasCoordinate(problem.bias().dimensions(), 0);
                    if (biasCoordinate.size() > 2 && inputs.batchBias == nullptr)
                        biasCoordinate[2] = static_cast<int64_t>(batch);
                    runtimeBiasOffset =
                        static_cast<ptrdiff_t>(problem.bias().index(biasCoordinate));
                    runtimeBiasLength = problem.bias().sizes()[0];
                    runtimeBiasAxis =
                        problem.getParams().factorDim() == 0 ? MatrixAxis::Row : MatrixAxis::Column;
                    if (runtimeBiasLength == m && runtimeBiasLength != n)
                        runtimeBiasAxis = MatrixAxis::Row;
                    else if (runtimeBiasLength == n && runtimeBiasLength != m)
                        runtimeBiasAxis = MatrixAxis::Column;
                    std::span<const std::byte> currentBiasStorage = biasStorage;
                    if (inputs.batchBias != nullptr) {
                        runtimeBiasOffset = 0;
                        const Layout layout = Layout::contiguous(Shape{runtimeBiasLength});
                        currentBiasStorage = std::span<const std::byte>(
                            static_cast<const std::byte*>(inputs.batchBias[batch]),
                            storageBytesForLayout(*biasType, layout));
                        if (problem.useGradient())
                            currentBiasOutputStorage = std::span<std::byte>(
                                static_cast<std::byte*>(
                                    const_cast<void*>(inputs.batchBias[batch])),
                                storageBytesForLayout(*biasType, layout));
                    }
                    runtimeBias = VectorBinding{
                        TensorView(*biasType,
                                   Layout(Shape{runtimeBiasLength}, {1}, runtimeBiasOffset),
                                   currentBiasStorage),
                        runtimeBiasAxis};
                    if (!useStandaloneEpilogue) runtimeProblem.epilogue.bias = runtimeBias;
                }
                runtimeProblem.mathMode =
                    accumulatorType == ScalarType::Float32 &&
                            problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                        ? MathMode::XFloat32
                        : MathMode::Default;
                if(!globalSelection.selectsAll())
                    runtimeProblem.outputSelection
                        = OutputSelection::explicitIndices(selectedByBatch[batch]);
                GemmInvocation invocation(std::move(runtimeProblem));
                if (backendImplementation != nullptr) {
                    const GemmBackend backend = backendImplementation->backend();
                    invocation.execution = {
                        .backend = backend,
                        .requireRequestedBackend = true,
                        .backendImplementation = backendImplementation,
                    };
                    const GemmSupportInfo support = queryGemmSupport(invocation);
                    if (!support) return false;
                }
                referenceGemm(invocation);

                if (useStandaloneEpilogue) {
                    EpilogueProblem epilogue(
                        intermediate->view(), productOutput, accumulatorType);
                    if (!problem.useGradient()) epilogue.bias = runtimeBias;
                    epilogue.activation = activation;
                    epilogue.activationParameter0 = activationParameter0;
                    epilogue.activationParameter1 = activationParameter1;
                    epilogue.outputScale = outputScale;
                    epilogue.outputSelection = invocation.problem.outputSelection;
                    std::optional<Tensor> gradientAuxiliary;
                    std::optional<Tensor> biasWorkspace;
                    if (problem.useGradient() && problem.useBias() &&
                        problem.biasSrc() == ContractionProblemGemm::D) {
                        biasWorkspace.emplace(accumulatorType, Shape{m, n});
                        epilogue.rawOutput = biasWorkspace->mutableView();
                    }
                    if (problem.useE()) {
                        auto const& auxiliary =
                            problem.tensors()[ContractionProblemGemm::TENSOR::E];
                        const ptrdiff_t offsetE =
                            static_cast<ptrdiff_t>(batch * auxiliary.strides()[batchD]);
                        const Layout auxiliaryLayout(
                            Shape{m, n},
                            {static_cast<ptrdiff_t>(auxiliary.strides()[indexMD]),
                             static_cast<ptrdiff_t>(auxiliary.strides()[indexND])},
                            offsetE);
                        if (problem.useGradient()) {
                            epilogue.auxiliaryInput = TensorView(
                                *auxiliaryType, auxiliaryLayout, auxiliaryStorage);
                        } else {
                            epilogue.auxiliaryOutput = MutableTensorView(
                                *auxiliaryType, auxiliaryLayout, auxiliaryStorage);
                        }
                    }
                    if (problem.useGradient()) {
                        epilogue.activationApplication = ActivationApplication::Gradient;
                        if (!epilogue.auxiliaryInput) {
                            gradientAuxiliary.emplace(accumulatorType, Shape{m, n});
                            epilogue.auxiliaryInput = gradientAuxiliary->view();
                        }
                    }
                    if (problem.useGateResidual()) {
                        auto const& gate = problem.tensors()
                            [ContractionProblemGemm::TENSOR::GATE_RESIDUAL];
                        const ptrdiff_t offsetGate = inputs.batchGateResidual == nullptr
                            ? static_cast<ptrdiff_t>(batch * gate.strides()[batchD])
                            : 0;
                        const Layout gateLayout(
                            Shape{m, n},
                            {static_cast<ptrdiff_t>(gate.strides()[indexMD]),
                             static_cast<ptrdiff_t>(gate.strides()[indexND])},
                            offsetGate);
                        std::span<const std::byte> currentGateStorage = gateStorage;
                        if (inputs.batchGateResidual != nullptr)
                            currentGateStorage = std::span<const std::byte>(
                                static_cast<const std::byte*>(
                                    inputs.batchGateResidual[batch]),
                                storageBytesForLayout(*gateType, gateLayout));
                        epilogue.gateResidual = TensorView(
                            *gateType, gateLayout, currentGateStorage);
                    }
                    if (problem.outputAmaxD()) {
                        epilogue.amax = MutableTensorView(
                            *amaxType, Layout::contiguous(Shape{1}), amaxStorage);
                        epilogue.accumulateAmax = batch != 0;
                    }
                    referenceEpilogue(epilogue);

                    if (problem.useGradient() && problem.useBias()) {
                        MutableTensorView biasOutput(
                            *biasType,
                            Layout(Shape{runtimeBiasLength}, {1}, runtimeBiasOffset),
                            currentBiasOutputStorage);
                        if (problem.biasSrc() == ContractionProblemGemm::D) {
                            referenceSum(ReductionProblem(
                                biasWorkspace->view(),
                                biasOutput,
                                accumulatorType,
                                {runtimeBiasAxis == MatrixAxis::Row ? size_t(1) : size_t(0)}));
                        } else if (problem.biasSrc() == ContractionProblemGemm::A) {
                            referenceSum(ReductionProblem(
                                logicalA, biasOutput, accumulatorType, {1}));
                        } else {
                            referenceSum(ReductionProblem(
                                logicalB, biasOutput, accumulatorType, {0}));
                        }
                    }
                }
            }
            return true;
        }

        bool tryRuntimeCanonicalGemm(ContractionProblemGemm const& problem,
                                     ContractionInputs const& inputs, size_t elementsToValidate) {
            return tryRuntimeGemm(problem, inputs, elementsToValidate, nullptr);
        }

        bool tryRuntimeTiledGemm(ContractionProblemGemm const& problem,
                                 ContractionInputs const& inputs, size_t elementsToValidate) {
            static const roc::host_validation::TiledGemmBackend tiledBackend;
            return tryRuntimeGemm(problem, inputs, elementsToValidate, &tiledBackend);
        }

        bool isFastPathEligible(ContractionProblemGemm const& problem)
        {

            auto rejectFast = [](const char* reason) {
                if (false) {  // Re-enable when testing to find reason.
                    std::clog << "FAST_PATH_REJECT: " << reason << std::endl;
                }
                return false;
            };

            auto isSupportedOutputType = [](rocisa::DataType t) {
                return t == rocisa::DataType::Float || t == rocisa::DataType::Double
                       || t == rocisa::DataType::Half || t == rocisa::DataType::BFloat16;
            };

            auto isSupportedInputType = [&](rocisa::DataType t) {
                if(isSupportedOutputType(t))
                    return true;
#ifdef TENSILE_USE_FP8_BF8
                if(t == rocisa::DataType::Float8
                   || t == rocisa::DataType::BFloat8
                   || t == rocisa::DataType::Float8_fnuz
                   || t == rocisa::DataType::BFloat8_fnuz)
                    return true;
#endif
#ifndef _WIN32
                if(t == rocisa::DataType::Float4)
                    return true;
#endif
                return false;
            };

            if(!isSupportedInputType(problem.a().dataType())
               || !isSupportedInputType(problem.b().dataType())
               || !isSupportedOutputType(problem.c().dataType())
               || !isSupportedOutputType(problem.d().dataType()))
            {
                std::string detail = "unsupported_type"
                    " A=" + TensileLite::ToString(problem.a().dataType())
                    + " B=" + TensileLite::ToString(problem.b().dataType())
                    + " C=" + TensileLite::ToString(problem.c().dataType())
                    + " D=" + TensileLite::ToString(problem.d().dataType());
                return rejectFast(detail.c_str());
            }

            constexpr size_t FAST_BLOCK_K = 8;
            size_t mxBlockA    = problem.mxBlockA();
            size_t mxBlockB    = problem.mxBlockB();

#ifndef _WIN32
            if(isMXFP4Problem(problem) && problem.a().dataType() != problem.b().dataType())
                return rejectFast("mixed_mxfp4_input_types");
#endif

            if(mxBlockA > 0 || mxBlockB > 0)
            {
                // One-sided MX (only A or only B scaled) is not supported. The
                // canonical component reference also rejects this case, so the
                // two paths agree on what "MX" means.
                if((mxBlockA > 0) != (mxBlockB > 0))
                    return rejectFast("one_sided_mx_not_supported");

                if(mxBlockA > 0 && mxBlockA % FAST_BLOCK_K != 0)
                    return rejectFast("mxBlockA_not_aligned_to_BLOCK_K");
                if(mxBlockB > 0 && mxBlockB % FAST_BLOCK_K != 0)
                    return rejectFast("mxBlockB_not_aligned_to_BLOCK_K");

                size_t sizeK = problem.boundSize(0);
                if(mxBlockA > 0 && sizeK % mxBlockA != 0)
                    return rejectFast("K_not_multiple_of_mxBlockA");
                if(mxBlockB > 0 && sizeK % mxBlockB != 0)
                    return rejectFast("K_not_multiple_of_mxBlockB");

                // The fast path reinterprets the scale tensors as E8 (UE8M0)
                // unconditionally (see solveGemmMXFastPath). Non-E8 scale
                // formats (E5M3 / Float8-E4M3) would be misdecoded as e8m0
                // exponents (e.g. byte 0xFF -> NaN), so defer to the slow path,
                // which decodes each scale via mxScaleElementAsFloat().
                if(mxBlockA > 0 && problem.mxsa().dataType() != rocisa::DataType::E8)
                    return rejectFast("mxScaleA_not_E8");
                if(mxBlockB > 0 && problem.mxsb().dataType() != rocisa::DataType::E8)
                    return rejectFast("mxScaleB_not_E8");
            }

            if(problem.boundIndices().size() >= 1)
            {
                if(problem.boundIndices()[0].aMirror || problem.boundIndices()[0].bMirror)
                    return rejectFast("mirror_indices");
            }

            if(problem.batchIndices().empty())
            {
                return rejectFast("no_batch_indices");
            }

            if(problem.useGradient())
            {
                return rejectFast("gradient");
            }

            if(problem.outputAmaxD())
            {
                return rejectFast("amaxD");
            }

            if(problem.useE())
            {
                return rejectFast("useE");
            }

            if(problem.useScaleCD())
            {
                return rejectFast("scaleCD");
            }

            if(problem.boundIndices().size() != 1 || problem.freeIndicesA().size() != 1
               || problem.freeIndicesB().size() != 1)
            {
                return rejectFast("multi_index");
            }

            // Layout validation — index accesses are safe because the
            // index-structure check above verified exactly 1 element in each.
            size_t indexMA = problem.freeIndicesA()[0].i;
            size_t indexKA = problem.boundIndices()[0].a;
            size_t indexNB = problem.freeIndicesB()[0].i;
            size_t indexKB = problem.boundIndices()[0].b;
            size_t indexMD = problem.freeIndices()[0].d;

            size_t strideMA = problem.a().strides()[indexMA];
            size_t strideKA = problem.a().strides()[indexKA];
            size_t strideNB = problem.b().strides()[indexNB];
            size_t strideKB = problem.b().strides()[indexKB];

            bool isPackedA = (strideMA == 1 || strideKA == 1);
            bool isPackedB = (strideNB == 1 || strideKB == 1);
            bool isPackedD = (problem.d().strides()[indexMD] == 1);
            if(!isPackedA || !isPackedB || !isPackedD)
            {
                return rejectFast("layout");
            }

            return true;
        }

        void SolveGemmCPU(ContractionProblemGemm const& problem,
                          ContractionInputs const&      inputs,
                          size_t                        elementsToValidate,
                          bool                          tryFastPath)
        {

            // The fast solver computes all elements. If the number of elements to validate
            // is in [0, sparsityThreshold * totalElements), skip this solver, falling through to another
            // solver that handles the partial validation sparsity efficiently.
            double sparsityThreshold        = 0.2;
            bool   isDenseEnoughForFastPath = true;
            if(elementsToValidate >= 0
               && elementsToValidate < sparsityThreshold * problem.d().totalLogicalElements())
            {
                isDenseEnoughForFastPath = false;
            }

            if(tryFastPath && isDenseEnoughForFastPath && isFastPathEligible(problem))
            {
                ScopedTimer timer("solve_cpu_fast");
                if(tryRuntimeTiledGemm(problem, inputs, elementsToValidate)) return;
            }

            if (tryRuntimeCanonicalGemm(problem, inputs, elementsToValidate)) return;

            throw std::runtime_error(concatenate(
                "Unsupported host-validation GEMM descriptor: ",
                problem.operationIdentifier(),
                ". The product-local typed reference fallback has been disabled."));
        }

        void SolveCPU(ContractionProblem const* problem,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate)
        {

            if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionGroupedInputs const*>(inputs);
                if(!refInput)
                {
                    throw std::runtime_error("Unable to cast input to ContractionGroupedInputs.");
                }
                if(groupedProblem->gemms.size() != refInput->grouped.size())
                {
                    throw std::runtime_error("Mismatched number of grouped problems and inputs.");
                }
                for(uint64_t i = 0; i < groupedProblem->gemms.size(); ++i)
                {
                    ContractionProblemGemm problem = groupedProblem->gemms[i];
                    ContractionInputs      input   = refInput->grouped[i];
                    SolveGemmCPU(problem, input, elementsToValidate);
                }
                return;
            }

            else if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionInputs const*>(inputs);
                if(!refInput)
                {
                    throw std::runtime_error("Unable to cast input to ContractionInputs.");
                }
                SolveGemmCPU(*gemmProblem, *refInput, elementsToValidate);
            }

            else
            {
                throw std::runtime_error(
                    "[Reference] Failed to cast to any ContractionProblem");
            }
        }
    } // namespace Client
} // namespace TensileLite
