// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/adapters/tensilelite/GemmInvocationAdapter.hpp>
#include <roc/host_validation/adapters/tensilelite/HostValidationBridge.hpp>

#include <Tensile/TensorDescriptor_fwd.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TensileLite::Client::reference_adapter
{
    namespace detail
    {
        using namespace roc::host_validation;
        using TensileLite::Client::checkedHostValidationPtrdiff;
        using TensileLite::Client::hostValidationLayout;

        inline TranslationFailure failure(TranslationFailureCode code, std::string reason)
        {
            return {code, std::move(reason)};
        }

        inline ptrdiff_t checkedMultiply(size_t left, size_t right)
        {
            constexpr uintmax_t maximum
                = static_cast<uintmax_t>(std::numeric_limits<ptrdiff_t>::max());
            const uintmax_t unsignedLeft  = static_cast<uintmax_t>(left);
            const uintmax_t unsignedRight = static_cast<uintmax_t>(right);
            if(unsignedLeft != 0 && unsignedRight > maximum / unsignedLeft)
                throw std::overflow_error("TensileLite adapter offset multiplication overflow.");
            return static_cast<ptrdiff_t>(unsignedLeft * unsignedRight);
        }

        inline ptrdiff_t checkedAdd(ptrdiff_t left, ptrdiff_t right)
        {
            if((right > 0 && left > std::numeric_limits<ptrdiff_t>::max() - right)
               || (right < 0 && left < std::numeric_limits<ptrdiff_t>::min() - right))
                throw std::overflow_error("TensileLite adapter offset addition overflow.");
            return left + right;
        }

        inline bool supportedAccumulator(ScalarType type)
        {
            return type == ScalarType::Float16 || type == ScalarType::BFloat16
                   || type == ScalarType::Float32 || type == ScalarType::Float64
                   || type == ScalarType::Int32 || type == ScalarType::ComplexFloat32
                   || type == ScalarType::ComplexFloat64;
        }

        inline bool supportedStandaloneEpilogueAccumulator(ScalarType type)
        {
            return type == ScalarType::Float32 || type == ScalarType::Float64
                   || type == ScalarType::Int32;
        }

        inline std::complex<double> scalarValue(rocisa::DataType type, ConstantVariant const& value)
        {
            switch(type)
            {
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
            case rocisa::DataType::ComplexFloat:
            {
                const auto converted = constVariantCast<std::complex<float>>(value);
                return {converted.real(), converted.imag()};
            }
            case rocisa::DataType::ComplexDouble:
                return constVariantCast<std::complex<double>>(value);
            default:
                throw std::invalid_argument(
                    "TensileLite scalar type has no host-validation conversion.");
            }
        }

        inline std::span<const std::byte>
            storageSpan(ScalarType type, const void* pointer, size_t elements)
        {
            const Layout layout = Layout::contiguous(Shape{elements});
            return {static_cast<const std::byte*>(pointer), storageBytesForLayout(type, layout)};
        }

        inline std::complex<double> scalarFromStorage(ScalarType type, const void* pointer)
        {
            const Tensor view(type, Layout::contiguous(Shape{1}), storageSpan(type, pointer, 1));
            if(scalarTypeInfo(type).category == ScalarCategory::Complex)
                return view.loadAs<std::complex<double>>({0});
            return {view.loadAs<double>({0}), 0.0};
        }

        inline size_t descriptorStorageBytes(ScalarType type, TensorDescriptor const& descriptor)
        {
            return storageBytesForLayout(type, hostValidationLayout(descriptor));
        }

        inline std::span<const std::byte> descriptorStorage(ScalarType              type,
                                                            TensorDescriptor const& descriptor,
                                                            const void*             pointer,
                                                            ptrdiff_t               byteOffset = 0)
        {
            return {static_cast<const std::byte*>(pointer) + byteOffset,
                    descriptorStorageBytes(type, descriptor)};
        }

        inline std::span<std::byte> mutableDescriptorStorage(ScalarType              type,
                                                             TensorDescriptor const& descriptor,
                                                             void*                   pointer,
                                                             ptrdiff_t               byteOffset = 0)
        {
            return {static_cast<std::byte*>(pointer) + byteOffset,
                    descriptorStorageBytes(type, descriptor)};
        }

        inline MatrixAxis inferBiasAxis(size_t length, size_t rows, size_t columns, int factorDim)
        {
            MatrixAxis axis = factorDim == 0 ? MatrixAxis::Row : MatrixAxis::Column;
            if(length == rows && length != columns)
                axis = MatrixAxis::Row;
            else if(length == columns && length != rows)
                axis = MatrixAxis::Column;
            return axis;
        }

        inline float mxScaleElementAsFloat(rocisa::DataType type, const void* pointer, size_t index)
        {
            float value;
            switch(type)
            {
            case rocisa::DataType::E8:
                // TensileLite's E8 encoding intentionally treats raw zero as
                // numeric zero rather than OCP E8M0's smallest finite scale.
                value = static_cast<float>(static_cast<E8 const*>(pointer)[index]);
                break;
            case rocisa::DataType::E5M3:
                value = static_cast<float>(static_cast<E5M3 const*>(pointer)[index]);
                break;
            case rocisa::DataType::Float8:
                value = static_cast<float>(static_cast<Float8 const*>(pointer)[index]);
                break;
            default:
                throw std::invalid_argument(concatenate(
                    "Reference MX scale has unsupported element type ", static_cast<int>(type)));
            }
            return std::fabs(value);
        }

        struct BatchInputs
        {
            Tensor                              a;
            Tensor                              b;
            Tensor                              c;
            Tensor                              d;
            OutputSelection                     outputSelection;
            std::optional<VectorBinding>        bias;
            std::optional<Tensor>               biasOutput;
            std::optional<Tensor>               auxiliaryInput;
            std::optional<Tensor>               auxiliaryOutput;
            std::optional<Tensor>               gateResidual;
            std::span<std::byte>                dDestination;
            std::optional<std::span<std::byte>> biasOutputDestination;
            std::optional<std::span<std::byte>> auxiliaryOutputDestination;
        };
    } // namespace detail

    struct GemmInvocationAdapter::State
    {
        using Activation = roc::host_validation::Activation;
        using ScalarType = roc::host_validation::ScalarType;
        using Tensor     = roc::host_validation::Tensor;

        State(ContractionProblemGemm const& problem_,
              ContractionInputs const&      inputs_,
              size_t                        elementsToValidate_)
            : problem(problem_)
            , inputs(inputs_)
            , elementsToValidate(elementsToValidate_)
        {
        }

        std::optional<TranslationFailure> preflight()
        {
            using namespace roc::host_validation;
            using detail::failure;

            if(problem.boundIndices().size() != 1 || problem.freeIndicesA().size() != 1
               || problem.freeIndicesB().size() != 1 || problem.batchIndices().size() != 1)
            {
                return failure(
                    TranslationFailureCode::UnsupportedContraction,
                    "Host validation requires one A free index, one B free index, one batch "
                    "index, and one bound index.");
            }
            if(problem.useGradient() && problem.useBias()
               && problem.biasSrc() != ContractionProblemGemm::A
               && problem.biasSrc() != ContractionProblemGemm::B
               && problem.biasSrc() != ContractionProblemGemm::D)
            {
                return failure(TranslationFailureCode::UnsupportedBiasSource,
                               "Gradient bias source must be A, B, or D.");
            }
            if((problem.mxBlockA() > 0) != (problem.mxBlockB() > 0))
            {
                return failure(TranslationFailureCode::InvalidScaleConfiguration,
                               "One-sided MX block scaling is unsupported.");
            }
            if(problem.useBias() && inputs.bias == nullptr && inputs.batchBias == nullptr)
                return failure(TranslationFailureCode::MissingInput, "Bias input is missing.");
            if(problem.useScaleAlphaVec() && inputs.scaleAlphaVec == nullptr)
            {
                return failure(TranslationFailureCode::MissingInput,
                               "Scale-alpha vector input is missing.");
            }
            if((problem.useScaleAB() == "Scalar" || problem.useScaleAB() == "Vector")
               && (inputs.scaleA == nullptr || inputs.scaleB == nullptr))
            {
                return failure(TranslationFailureCode::MissingInput,
                               "A and B scale inputs are required together.");
            }
            if(problem.mxBlockA() > 0 && (inputs.mxsa == nullptr || inputs.mxsb == nullptr))
                return failure(TranslationFailureCode::MissingInput, "MX scale input is missing.");
            if(problem.useE() && inputs.e == nullptr)
                return failure(TranslationFailureCode::MissingInput,
                               "Auxiliary E input is missing.");
            if(problem.outputAmaxD() && inputs.amaxD == nullptr)
                return failure(TranslationFailureCode::MissingInput, "Amax(D) output is missing.");
            if(problem.useScaleCD() && (inputs.scaleC == nullptr || inputs.scaleD == nullptr))
            {
                return failure(TranslationFailureCode::MissingInput,
                               "C and D scale inputs are required together.");
            }
            if(problem.useGateResidual() && inputs.gateResidual == nullptr
               && inputs.batchGateResidual == nullptr)
            {
                return failure(TranslationFailureCode::MissingInput,
                               "Gate-residual input is missing.");
            }
            if((inputs.a == nullptr && inputs.batchA == nullptr)
               || (inputs.b == nullptr && inputs.batchB == nullptr)
               || (inputs.c == nullptr && inputs.batchC == nullptr)
               || (inputs.d == nullptr && inputs.batchD == nullptr))
            {
                return failure(TranslationFailureCode::MissingInput,
                               "A, B, C, or D input is missing.");
            }

            try
            {
                typeA                    = toHostValidationScalarType(problem.a().dataType());
                typeB                    = toHostValidationScalarType(problem.b().dataType());
                typeC                    = toHostValidationScalarType(problem.c().dataType());
                typeD                    = toHostValidationScalarType(problem.d().dataType());
                operationAccumulatorType = toHostValidationScalarType(problem.computeType());
                betaType                 = toHostValidationScalarType(problem.betaType());
                alphaType                = toHostValidationScalarType(problem.alphaType());
                computeTypeA             = problem.computeInputTypeA() == rocisa::DataType::None
                                               ? typeA
                                               : toHostValidationScalarType(problem.computeInputTypeA());
                computeTypeB             = problem.computeInputTypeB() == rocisa::DataType::None
                                               ? typeB
                                               : toHostValidationScalarType(problem.computeInputTypeB());
            }
            catch(std::invalid_argument const& error)
            {
                return failure(TranslationFailureCode::UnsupportedDataType, error.what());
            }

            if(!detail::supportedAccumulator(operationAccumulatorType))
            {
                return failure(TranslationFailureCode::UnsupportedAccumulator,
                               "TensileLite compute type is unsupported by host validation.");
            }

            useStandaloneEpilogue = problem.useGradient() || problem.outputAmaxD() || problem.useE()
                                    || problem.useScaleCD() || problem.useGateResidual();
            preQuantizationScaleA
                = scalarTypeInfo(typeA).storageBits > scalarTypeInfo(computeTypeA).storageBits;
            preQuantizationScaleB
                = scalarTypeInfo(typeB).storageBits > scalarTypeInfo(computeTypeB).storageBits;

            try
            {
                alpha = detail::scalarValue(problem.alphaType(), inputs.alpha);
                beta  = detail::scalarValue(problem.betaType(), inputs.beta);
            }
            catch(std::invalid_argument const& error)
            {
                return failure(TranslationFailureCode::UnsupportedDataType, error.what());
            }

            ActivationType concreteActivation = problem.activationType();
            if(concreteActivation == ActivationType::All
               || concreteActivation == ActivationType::Hipblaslt_all)
                concreteActivation = problem.getParams().activationEnum();
            try
            {
                activation = toHostValidationActivation(concreteActivation, problem.useGradient());
            }
            catch(std::invalid_argument const& error)
            {
                return failure(TranslationFailureCode::UnsupportedActivation, error.what());
            }
            if((operationAccumulatorType == ScalarType::ComplexFloat32
                || operationAccumulatorType == ScalarType::ComplexFloat64)
               && activation != Activation::None)
            {
                return failure(TranslationFailureCode::UnsupportedActivation,
                               "Complex GEMM activation is unsupported.");
            }

            if(problem.useScaleAB() == "Scalar")
            {
                if(!preQuantizationScaleA)
                    alpha *= detail::scalarFromStorage(alphaType, inputs.scaleA);
                if(!preQuantizationScaleB)
                    alpha *= detail::scalarFromStorage(alphaType, inputs.scaleB);
            }

            if(problem.useScaleCD())
            {
                if(beta != std::complex<double>(0.0, 0.0))
                    beta *= detail::scalarFromStorage(betaType, inputs.scaleC);
                outputScale = detail::scalarFromStorage(betaType, inputs.scaleD);
            }

            try
            {
                if(!inputs.activationArgs.empty())
                {
                    activationParameter0
                        = detail::scalarValue(problem.computeType(), inputs.activationArgs[0])
                              .real();
                }
                if(inputs.activationArgs.size() > 1)
                {
                    activationParameter1
                        = detail::scalarValue(problem.computeType(), inputs.activationArgs[1])
                              .real();
                }
            }
            catch(std::invalid_argument const& error)
            {
                return failure(TranslationFailureCode::UnsupportedDataType, error.what());
            }

            indexMA = problem.freeIndicesA()[0].i;
            indexKA = problem.boundIndices()[0].a;
            indexNB = problem.freeIndicesB()[0].i;
            indexKB = problem.boundIndices()[0].b;
            indexMD = problem.freeIndices()[0].d;
            indexND = problem.freeIndices()[1].d;
            batchA  = problem.batchIndices()[0].a;
            batchB  = problem.batchIndices()[0].b;
            batchC  = problem.batchIndices()[0].c;
            batchD  = problem.batchIndices()[0].d;
            m       = problem.freeSizeA(0);
            n       = problem.freeSizeB(0);
            k       = problem.boundSize(0);
            batches = problem.batchSize(0);

            mxBlockA = problem.mxBlockA();
            mxBlockB = problem.mxBlockB();
            if(mxBlockA > 0)
            {
                strideMxsaM     = problem.mxsa().strides()[indexMA];
                strideMxsaBlock = problem.mxsa().strides()[indexKA];
                strideMxsbN     = problem.mxsb().strides()[indexNB];
                strideMxsbBlock = problem.mxsb().strides()[indexKB];
                strideBatchMxsa = problem.mxsa().strides()[batchA];
                strideBatchMxsb = problem.mxsb().strides()[batchB];
            }

            for(auto const& operation : problem.aOps())
                aConjugate |= operation.type == TensorOp::Type::ComplexConjugate;
            for(auto const& operation : problem.bOps())
                bConjugate |= operation.type == TensorOp::Type::ComplexConjugate;

            const bool requiresCompleteD = problem.outputAmaxD()
                                           || (problem.useGradient() && problem.useBias()
                                               && problem.biasSrc() == ContractionProblemGemm::D);
            const OutputSelection globalSelection
                = requiresCompleteD
                      ? OutputSelection::all()
                      : OutputSelection::primeStride(problem.d().totalLogicalElements(),
                                                     problem.d().totalAllocatedElements(),
                                                     elementsToValidate);
            std::vector<std::vector<size_t>> selectedByBatch;
            if(!globalSelection.selectsAll())
            {
                selectedByBatch.resize(batches);
                for(const size_t logicalIndex :
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

            for(size_t batch = 0; batch < batches; ++batch)
            {
                if((inputs.batchA != nullptr && inputs.batchA[batch] == nullptr)
                   || (inputs.batchB != nullptr && inputs.batchB[batch] == nullptr)
                   || (inputs.batchC != nullptr && inputs.batchC[batch] == nullptr)
                   || (inputs.batchD != nullptr && inputs.batchD[batch] == nullptr)
                   || (problem.useBias() && inputs.batchBias != nullptr
                       && inputs.batchBias[batch] == nullptr)
                   || (problem.useGateResidual() && inputs.batchGateResidual != nullptr
                       && inputs.batchGateResidual[batch] == nullptr))
                {
                    return failure(TranslationFailureCode::InvalidBatchPointer,
                                   "A pointer-array batch contains a null tensor pointer.");
                }
            }

            const ptrdiff_t batchOffsetA
                = detail::checkedHostValidationPtrdiff(inputs.batchOffsetA);
            const ptrdiff_t batchOffsetB
                = detail::checkedHostValidationPtrdiff(inputs.batchOffsetB);
            const ptrdiff_t batchOffsetC
                = detail::checkedHostValidationPtrdiff(inputs.batchOffsetC);
            const ptrdiff_t batchOffsetD
                = detail::checkedHostValidationPtrdiff(inputs.batchOffsetD);

            std::span<const std::byte> aStorage;
            std::span<const std::byte> bStorage;
            std::span<const std::byte> cStorage;
            std::span<std::byte>       dStorage;
            std::optional<Tensor>      aTensor;
            std::optional<Tensor>      bTensor;
            std::optional<Tensor>      cTensor;
            std::optional<Tensor>      dTensor;
            const bool readC = beta != std::complex<double>(0.0, 0.0);
            const auto makeAddendTensor = [&](const Layout& layout,
                                              std::span<const std::byte> source) {
                return readC ? Tensor(typeC, layout, source)
                             : Tensor(typeC, layout, TensorStorage::allocateUninitialized);
            };
            const bool initializeOutput = globalSelection.selectsAll()
                                          || scalarTypeInfo(typeD).storageBits % 8 != 0;
            const auto makeOutputTensor = [&](const Layout& layout,
                                              std::span<std::byte> destination) {
                return initializeOutput ? Tensor(typeD, layout, destination)
                                        : Tensor(typeD,
                                                 layout,
                                                 TensorStorage::allocateUninitialized);
            };
            if(inputs.batchA == nullptr)
            {
                aStorage = detail::descriptorStorage(typeA, problem.a(), inputs.a, batchOffsetA);
                aTensor.emplace(typeA, detail::hostValidationLayout(problem.a()), aStorage);
            }
            if(inputs.batchB == nullptr)
            {
                bStorage = detail::descriptorStorage(typeB, problem.b(), inputs.b, batchOffsetB);
                bTensor.emplace(typeB, detail::hostValidationLayout(problem.b()), bStorage);
            }
            if(inputs.batchC == nullptr)
            {
                cStorage = detail::descriptorStorage(typeC, problem.c(), inputs.c, batchOffsetC);
                cTensor.emplace(
                    makeAddendTensor(detail::hostValidationLayout(problem.c()), cStorage));
            }
            if(inputs.batchD == nullptr)
            {
                dStorage
                    = detail::mutableDescriptorStorage(typeD, problem.d(), inputs.d, batchOffsetD);
                dTensor.emplace(
                    makeOutputTensor(detail::hostValidationLayout(problem.d()), dStorage));
            }

            std::optional<ScalarType>  biasType;
            std::span<const std::byte> biasStorage;
            std::span<std::byte>       biasOutputStorage;
            std::optional<Tensor>      biasTensor;
            std::optional<Tensor>      biasOutputTensor;
            if(problem.useBias())
            {
                try
                {
                    biasType = toHostValidationScalarType(problem.bias().dataType());
                }
                catch(std::invalid_argument const& error)
                {
                    return failure(TranslationFailureCode::UnsupportedDataType, error.what());
                }
                if(inputs.batchBias == nullptr)
                {
                    biasStorage = detail::descriptorStorage(*biasType, problem.bias(), inputs.bias);
                    biasTensor.emplace(
                        *biasType, detail::hostValidationLayout(problem.bias()), biasStorage);
                    if(problem.useGradient())
                    {
                        biasOutputStorage = detail::mutableDescriptorStorage(
                            *biasType, problem.bias(), const_cast<void*>(inputs.bias));
                        biasOutputTensor.emplace(*biasType,
                                                 detail::hostValidationLayout(problem.bias()),
                                                 biasOutputStorage);
                    }
                }
            }

            std::optional<ScalarType> auxiliaryType;
            std::span<std::byte>      auxiliaryStorage;
            TensorDescriptor const*   auxiliaryDescriptor = nullptr;
            std::optional<Tensor>     auxiliaryTensor;
            if(problem.useE())
            {
                auxiliaryDescriptor = &problem.tensors()[ContractionProblemGemm::TENSOR::E];
                try
                {
                    auxiliaryType = toHostValidationScalarType(auxiliaryDescriptor->dataType());
                }
                catch(std::invalid_argument const& error)
                {
                    return failure(TranslationFailureCode::UnsupportedDataType, error.what());
                }
                auxiliaryStorage = detail::mutableDescriptorStorage(
                    *auxiliaryType, *auxiliaryDescriptor, inputs.e);
                auxiliaryTensor.emplace(*auxiliaryType,
                                        detail::hostValidationLayout(*auxiliaryDescriptor),
                                        auxiliaryStorage);
            }

            std::optional<ScalarType>  gateType;
            std::span<const std::byte> gateStorage;
            TensorDescriptor const*    gateDescriptor = nullptr;
            std::optional<Tensor>      gateTensor;
            if(problem.useGateResidual())
            {
                gateDescriptor = &problem.tensors()[ContractionProblemGemm::TENSOR::GATE_RESIDUAL];
                try
                {
                    gateType = toHostValidationScalarType(gateDescriptor->dataType());
                }
                catch(std::invalid_argument const& error)
                {
                    return failure(TranslationFailureCode::UnsupportedDataType, error.what());
                }
                if(inputs.batchGateResidual == nullptr)
                {
                    gateStorage = detail::descriptorStorage(
                        *gateType, *gateDescriptor, inputs.gateResidual);
                    gateTensor.emplace(
                        *gateType, detail::hostValidationLayout(*gateDescriptor), gateStorage);
                }
            }

            if(problem.outputAmaxD())
            {
                auto const& descriptor = problem.tensors()[ContractionProblemGemm::TENSOR::AMAXD];
                try
                {
                    const ScalarType amaxType = toHostValidationScalarType(descriptor.dataType());
                    const auto       storage
                        = detail::mutableDescriptorStorage(amaxType, descriptor, inputs.amaxD);
                    amax            = Tensor(amaxType, Layout::contiguous(Shape{1}), storage);
                    amaxDestination = storage;
                }
                catch(std::invalid_argument const& error)
                {
                    return failure(TranslationFailureCode::UnsupportedDataType, error.what());
                }
            }

            const size_t scaleAlphaLength = problem.getParams().factorDim() == 0 ? m : n;
            if(problem.useScaleAlphaVec())
            {
                scaleAlpha = Tensor(
                    alphaType,
                    Layout::contiguous(Shape{scaleAlphaLength}),
                    detail::storageSpan(alphaType, inputs.scaleAlphaVec, scaleAlphaLength));
            }
            if(problem.useScaleAB() == "Scalar")
            {
                scaleA = Tensor(alphaType,
                                Layout::contiguous(Shape{1}),
                                detail::storageSpan(alphaType, inputs.scaleA, 1));
                scaleB = Tensor(alphaType,
                                Layout::contiguous(Shape{1}),
                                detail::storageSpan(alphaType, inputs.scaleB, 1));
            }
            else if(problem.useScaleAB() == "Vector")
            {
                scaleA = Tensor(alphaType,
                                Layout::contiguous(Shape{m}),
                                detail::storageSpan(alphaType, inputs.scaleA, m));
                scaleB = Tensor(alphaType,
                                Layout::contiguous(Shape{n}),
                                detail::storageSpan(alphaType, inputs.scaleB, n));
            }

            translatedBatches.reserve(batches);
            for(size_t batch = 0; batch < batches; ++batch)
            {
                ptrdiff_t offsetA
                    = inputs.batchA == nullptr
                          ? detail::checkedMultiply(batch, problem.a().strides()[batchA])
                          : 0;
                ptrdiff_t offsetB
                    = inputs.batchB == nullptr
                          ? detail::checkedMultiply(batch, problem.b().strides()[batchB])
                          : 0;
                const ptrdiff_t offsetC
                    = inputs.batchC == nullptr
                          ? detail::checkedMultiply(batch, problem.c().strides()[batchC])
                          : 0;
                const ptrdiff_t offsetD
                    = inputs.batchD == nullptr
                          ? detail::checkedMultiply(batch, problem.d().strides()[batchD])
                          : 0;
                ptrdiff_t strideKA
                    = detail::checkedHostValidationPtrdiff(problem.a().strides()[indexKA]);
                ptrdiff_t strideKB
                    = detail::checkedHostValidationPtrdiff(problem.b().strides()[indexKB]);
                if(problem.boundIndices()[0].aMirror && k != 0)
                {
                    const ptrdiff_t mirroredOffset
                        = detail::checkedMultiply(k - 1, problem.a().strides()[indexKA]);
                    offsetA  = detail::checkedAdd(offsetA, mirroredOffset);
                    strideKA = -strideKA;
                }
                if(problem.boundIndices()[0].bMirror && k != 0)
                {
                    const ptrdiff_t mirroredOffset
                        = detail::checkedMultiply(k - 1, problem.b().strides()[indexKB]);
                    offsetB  = detail::checkedAdd(offsetB, mirroredOffset);
                    strideKB = -strideKB;
                }

                const Layout layoutA(
                    Shape{m, k},
                    {detail::checkedHostValidationPtrdiff(problem.a().strides()[indexMA]),
                     strideKA},
                    offsetA);
                const Layout layoutB(
                    Shape{k, n},
                    {strideKB,
                     detail::checkedHostValidationPtrdiff(problem.b().strides()[indexNB])},
                    offsetB);
                const Layout layoutC(
                    Shape{m, n},
                    {detail::checkedHostValidationPtrdiff(problem.c().strides()[indexMD]),
                     detail::checkedHostValidationPtrdiff(problem.c().strides()[indexND])},
                    offsetC);
                const Layout layoutD(
                    Shape{m, n},
                    {detail::checkedHostValidationPtrdiff(problem.d().strides()[indexMD]),
                     detail::checkedHostValidationPtrdiff(problem.d().strides()[indexND])},
                    offsetD);

                const auto currentAStorage
                    = inputs.batchA == nullptr
                          ? aStorage
                          : std::span<const std::byte>(
                              static_cast<const std::byte*>(inputs.batchA[batch]) + batchOffsetA,
                              storageBytesForLayout(typeA, layoutA));
                const auto currentBStorage
                    = inputs.batchB == nullptr
                          ? bStorage
                          : std::span<const std::byte>(
                              static_cast<const std::byte*>(inputs.batchB[batch]) + batchOffsetB,
                              storageBytesForLayout(typeB, layoutB));
                const auto currentCStorage
                    = inputs.batchC == nullptr
                          ? cStorage
                          : std::span<const std::byte>(
                              static_cast<const std::byte*>(inputs.batchC[batch]) + batchOffsetC,
                              storageBytesForLayout(typeC, layoutC));
                const auto currentDStorage
                    = inputs.batchD == nullptr
                          ? dStorage
                          : std::span<std::byte>(static_cast<std::byte*>(inputs.batchD[batch])
                                                     + batchOffsetD,
                                                 storageBytesForLayout(typeD, layoutD));
                Tensor currentA = inputs.batchA == nullptr
                                      ? aTensor->alias(layoutA)
                                      : Tensor(typeA, layoutA, currentAStorage);
                Tensor currentB = inputs.batchB == nullptr
                                      ? bTensor->alias(layoutB)
                                      : Tensor(typeB, layoutB, currentBStorage);
                Tensor currentC = inputs.batchC == nullptr
                                      ? cTensor->alias(layoutC)
                                      : makeAddendTensor(layoutC, currentCStorage);
                Tensor currentD = inputs.batchD == nullptr
                                      ? dTensor->alias(layoutD)
                                      : makeOutputTensor(layoutD, currentDStorage);

                std::optional<VectorBinding>        runtimeBias;
                std::optional<Tensor>               runtimeBiasOutput;
                std::optional<std::span<std::byte>> runtimeBiasOutputDestination;
                if(problem.useBias())
                {
                    ptrdiff_t        runtimeBiasOffset = 0;
                    const size_t     runtimeBiasLength = problem.bias().sizes()[0];
                    const MatrixAxis runtimeBiasAxis   = detail::inferBiasAxis(
                        runtimeBiasLength, m, n, problem.getParams().factorDim());
                    std::span<const std::byte> currentBiasStorage       = biasStorage;
                    std::span<std::byte>       currentBiasOutputStorage = biasOutputStorage;
                    if(inputs.batchBias == nullptr)
                    {
                        if(problem.bias().dimensions() > 2)
                            runtimeBiasOffset
                                = detail::checkedMultiply(batch, problem.bias().strides()[2]);
                    }
                    else
                    {
                        const Layout layout = Layout::contiguous(Shape{runtimeBiasLength});
                        currentBiasStorage  = std::span<const std::byte>(
                            static_cast<const std::byte*>(inputs.batchBias[batch]),
                            storageBytesForLayout(*biasType, layout));
                        if(problem.useGradient())
                        {
                            currentBiasOutputStorage = std::span<std::byte>(
                                static_cast<std::byte*>(const_cast<void*>(inputs.batchBias[batch])),
                                storageBytesForLayout(*biasType, layout));
                        }
                    }
                    const Layout biasLayout(Shape{runtimeBiasLength}, {1}, runtimeBiasOffset);
                    runtimeBias
                        = VectorBinding{inputs.batchBias == nullptr
                                            ? biasTensor->alias(biasLayout)
                                            : Tensor(*biasType, biasLayout, currentBiasStorage),
                                        runtimeBiasAxis};
                    if(problem.useGradient())
                    {
                        runtimeBiasOutput
                            = inputs.batchBias == nullptr
                                  ? biasOutputTensor->alias(biasLayout)
                                  : Tensor(*biasType, biasLayout, currentBiasOutputStorage);
                        runtimeBiasOutputDestination = currentBiasOutputStorage;
                    }
                }

                std::optional<Tensor> auxiliaryInput;
                std::optional<Tensor> auxiliaryOutput;
                if(problem.useE())
                {
                    const ptrdiff_t offsetE
                        = detail::checkedMultiply(batch, auxiliaryDescriptor->strides()[batchD]);
                    const Layout layoutE(Shape{m, n},
                                         {detail::checkedHostValidationPtrdiff(
                                              auxiliaryDescriptor->strides()[indexMD]),
                                          detail::checkedHostValidationPtrdiff(
                                              auxiliaryDescriptor->strides()[indexND])},
                                         offsetE);
                    if(problem.useGradient())
                        auxiliaryInput = auxiliaryTensor->alias(layoutE);
                    else
                        auxiliaryOutput = auxiliaryTensor->alias(layoutE);
                }

                std::optional<Tensor> runtimeGate;
                if(problem.useGateResidual())
                {
                    const ptrdiff_t offsetGate
                        = inputs.batchGateResidual == nullptr
                              ? detail::checkedMultiply(batch, gateDescriptor->strides()[batchD])
                              : 0;
                    const Layout gateLayout(
                        Shape{m, n},
                        {detail::checkedHostValidationPtrdiff(gateDescriptor->strides()[indexMD]),
                         detail::checkedHostValidationPtrdiff(gateDescriptor->strides()[indexND])},
                        offsetGate);
                    std::span<const std::byte> currentGateStorage = gateStorage;
                    if(inputs.batchGateResidual != nullptr)
                    {
                        currentGateStorage = std::span<const std::byte>(
                            static_cast<const std::byte*>(inputs.batchGateResidual[batch]),
                            storageBytesForLayout(*gateType, gateLayout));
                    }
                    runtimeGate = inputs.batchGateResidual == nullptr
                                      ? gateTensor->alias(gateLayout)
                                      : Tensor(*gateType, gateLayout, currentGateStorage);
                }

                OutputSelection outputSelection = OutputSelection::all();
                if(!globalSelection.selectsAll())
                    outputSelection = OutputSelection::explicitIndices(selectedByBatch[batch]);

                translatedBatches.push_back(detail::BatchInputs{
                    std::move(currentA),
                    std::move(currentB),
                    std::move(currentC),
                    std::move(currentD),
                    std::move(outputSelection),
                    std::move(runtimeBias),
                    std::move(runtimeBiasOutput),
                    std::move(auxiliaryInput),
                    std::move(auxiliaryOutput),
                    std::move(runtimeGate),
                    currentDStorage,
                    runtimeBiasOutputDestination,
                    problem.useE() && !problem.useGradient()
                        ? std::optional<std::span<std::byte>>(auxiliaryStorage)
                        : std::nullopt,
                });
            }

            return std::nullopt;
        }

        ContractionProblemGemm const& problem;
        ContractionInputs const&      inputs;
        size_t                        elementsToValidate;

        ScalarType typeA                    = ScalarType::Float32;
        ScalarType typeB                    = ScalarType::Float32;
        ScalarType typeC                    = ScalarType::Float32;
        ScalarType typeD                    = ScalarType::Float32;
        ScalarType operationAccumulatorType = ScalarType::Float32;
        ScalarType computeTypeA             = ScalarType::Float32;
        ScalarType computeTypeB             = ScalarType::Float32;
        ScalarType alphaType                = ScalarType::Float32;
        ScalarType betaType                 = ScalarType::Float32;

        std::complex<double> alpha                = {1.0, 0.0};
        std::complex<double> beta                 = {0.0, 0.0};
        std::complex<double> outputScale          = {1.0, 0.0};
        Activation           activation           = Activation::None;
        double               activationParameter0 = 0.0;
        double               activationParameter1 = 0.0;

        bool useStandaloneEpilogue = false;
        bool preQuantizationScaleA = false;
        bool preQuantizationScaleB = false;
        bool aConjugate            = false;
        bool bConjugate            = false;

        size_t indexMA = 0;
        size_t indexKA = 0;
        size_t indexNB = 0;
        size_t indexKB = 0;
        size_t indexMD = 0;
        size_t indexND = 0;
        size_t batchA  = 0;
        size_t batchB  = 0;
        size_t batchC  = 0;
        size_t batchD  = 0;
        size_t m       = 0;
        size_t n       = 0;
        size_t k       = 0;
        size_t batches = 0;

        size_t mxBlockA        = 0;
        size_t mxBlockB        = 0;
        size_t strideMxsaM     = 0;
        size_t strideMxsaBlock = 0;
        size_t strideMxsbN     = 0;
        size_t strideMxsbBlock = 0;
        size_t strideBatchMxsa = 0;
        size_t strideBatchMxsb = 0;

        std::optional<Tensor>               scaleAlpha;
        std::optional<Tensor>               scaleA;
        std::optional<Tensor>               scaleB;
        std::optional<Tensor>               amax;
        std::optional<std::span<std::byte>> amaxDestination;
        std::vector<detail::BatchInputs>    translatedBatches;
    };

    GemmInvocationAdapter::GemmInvocationAdapter(std::unique_ptr<const State> state)
        : m_state(std::move(state))
    {
    }

    GemmInvocationAdapter::~GemmInvocationAdapter() = default;

    GemmInvocationAdapter::GemmInvocationAdapter(GemmInvocationAdapter&&) noexcept = default;

    GemmInvocationAdapter&
        GemmInvocationAdapter::operator=(GemmInvocationAdapter&&) noexcept = default;

    void TranslatedGemmBatch::copyOutputs() const
    {
        for(const auto& copyBack : copyBacks)
        {
            if(!copyBack.selection || copyBack.selection->selectsAll())
                copyBack.source.copyTo(copyBack.destination);
            else
            {
                const auto selected = copyBack.selection->indices(copyBack.source.size());
                copyBack.source.copyTo(copyBack.destination, selected);
            }
        }
    }

    size_t GemmInvocationAdapter::batchCount() const
    {
        return m_state->batches;
    }

    roc::host_validation::ScalarType GemmInvocationAdapter::operationAccumulatorType() const
    {
        return m_state->operationAccumulatorType;
    }

    bool GemmInvocationAdapter::usesStandaloneEpilogue() const
    {
        return m_state->useStandaloneEpilogue;
    }

    std::variant<TranslatedGemmBatch, TranslationFailure> GemmInvocationAdapter::translateBatch(
        size_t batch, roc::host_validation::ScalarType accumulatorType) const
    {
        using namespace roc::host_validation;
        using detail::failure;

        if(batch >= m_state->translatedBatches.size())
        {
            return failure(TranslationFailureCode::InvalidBatchIndex,
                           "Requested reference batch index is out of range.");
        }
        if(!detail::supportedAccumulator(accumulatorType))
        {
            return failure(TranslationFailureCode::UnsupportedAccumulator,
                           "Requested host-validation accumulator type is unsupported.");
        }
        if(m_state->useStandaloneEpilogue
           && !detail::supportedStandaloneEpilogueAccumulator(accumulatorType))
        {
            return failure(TranslationFailureCode::UnsupportedAccumulator,
                           "Standalone epilogue requires F32, F64, or I32 accumulation.");
        }

        try
        {
            const auto&         source = m_state->translatedBatches[batch];
            TranslatedGemmBatch translated;

            GemmOperand operandA(source.a);
            GemmOperand operandB(source.b);
            if(m_state->computeTypeA != m_state->typeA)
                operandA.computeType = m_state->computeTypeA;
            if(m_state->computeTypeB != m_state->typeB)
                operandB.computeType = m_state->computeTypeB;

            if(m_state->problem.useScaleAB() == "Scalar" && m_state->preQuantizationScaleA)
            {
                operandA.preQuantizationScales.push_back(
                    VectorBinding{*m_state->scaleA, MatrixAxis::Row});
            }
            if(m_state->problem.useScaleAB() == "Scalar" && m_state->preQuantizationScaleB)
            {
                operandB.preQuantizationScales.push_back(
                    VectorBinding{*m_state->scaleB, MatrixAxis::Column});
            }
            if(m_state->problem.useScaleAB() == "Vector" && m_state->preQuantizationScaleA)
            {
                operandA.preQuantizationScales.push_back(
                    VectorBinding{*m_state->scaleA, MatrixAxis::Row});
            }
            if(m_state->problem.useScaleAB() == "Vector" && m_state->preQuantizationScaleB)
            {
                operandB.preQuantizationScales.push_back(
                    VectorBinding{*m_state->scaleB, MatrixAxis::Column});
            }
            operandA.conjugate = m_state->aConjugate;
            operandB.conjugate = m_state->bConjugate;

            if(m_state->mxBlockA > 0)
            {
                const size_t blockCountA = m_state->k / m_state->mxBlockA
                                           + (m_state->k % m_state->mxBlockA != 0 ? 1 : 0);
                const size_t blockCountB = m_state->k / m_state->mxBlockB
                                           + (m_state->k % m_state->mxBlockB != 0 ? 1 : 0);
                translated.runtimeScaleA.emplace(ScalarType::Float32,
                                                 Shape{m_state->m, blockCountA});
                translated.runtimeScaleB.emplace(ScalarType::Float32,
                                                 Shape{m_state->n, blockCountB});

                for(size_t row = 0; row < m_state->m; ++row)
                {
                    for(size_t block = 0; block < blockCountA; ++block)
                    {
                        const size_t sourceBlock = m_state->problem.boundIndices()[0].aMirror
                                                       ? blockCountA - 1 - block
                                                       : block;
                        const size_t index       = batch * m_state->strideBatchMxsa
                                             + row * m_state->strideMxsaM
                                             + sourceBlock * m_state->strideMxsaBlock;
                        translated.runtimeScaleA->storeFrom(
                            {row, block},
                            detail::mxScaleElementAsFloat(
                                m_state->problem.mxTypeA(), m_state->inputs.mxsa, index));
                    }
                }
                for(size_t column = 0; column < m_state->n; ++column)
                {
                    for(size_t block = 0; block < blockCountB; ++block)
                    {
                        const size_t sourceBlock = m_state->problem.boundIndices()[0].bMirror
                                                       ? blockCountB - 1 - block
                                                       : block;
                        const size_t index       = batch * m_state->strideBatchMxsb
                                             + column * m_state->strideMxsbN
                                             + sourceBlock * m_state->strideMxsbBlock;
                        translated.runtimeScaleB->storeFrom(
                            {column, block},
                            detail::mxScaleElementAsFloat(
                                m_state->problem.mxTypeB(), m_state->inputs.mxsb, index));
                    }
                }
                operandA.blockScale
                    = BlockScaleBinding{*translated.runtimeScaleA, m_state->mxBlockA};
                operandB.blockScale
                    = BlockScaleBinding{*translated.runtimeScaleB, m_state->mxBlockB};
            }

            Tensor productOutput = source.d;
            translated.copyBacks.push_back(
                {source.dDestination, productOutput, source.outputSelection});
            if(m_state->amax && m_state->amaxDestination)
                translated.copyBacks.push_back(
                    {*m_state->amaxDestination, *m_state->amax, std::nullopt});
            if(source.biasOutput && source.biasOutputDestination)
                translated.copyBacks.push_back(
                    {*source.biasOutputDestination, *source.biasOutput, std::nullopt});
            if(source.auxiliaryOutput && source.auxiliaryOutputDestination)
                translated.copyBacks.push_back({*source.auxiliaryOutputDestination,
                                                *source.auxiliaryOutput,
                                                source.outputSelection});
            Tensor gemmOutput = productOutput;
            if(m_state->useStandaloneEpilogue)
            {
                translated.intermediate.emplace(accumulatorType, Shape{m_state->m, m_state->n});
                gemmOutput = *translated.intermediate;
            }

            translated.gemmRequest.emplace(
                std::move(operandA), std::move(operandB), source.c, gemmOutput, accumulatorType);
            auto& request          = translated.gemm();
            request.epilogue.alpha = m_state->alpha;
            request.epilogue.beta  = m_state->beta;
            if(!m_state->useStandaloneEpilogue && m_state->typeD == ScalarType::Int8)
                request.epilogue.outputConversion = OutputConversion::SaturatingInt8;
            if(!m_state->useStandaloneEpilogue)
            {
                request.epilogue.activation           = m_state->activation;
                request.epilogue.activationParameter0 = m_state->activationParameter0;
                request.epilogue.activationParameter1 = m_state->activationParameter1;
            }
            if(m_state->scaleAlpha)
            {
                request.epilogue.scaleAlpha = VectorBinding{
                    *m_state->scaleAlpha,
                    m_state->problem.getParams().factorDim() == 0 ? MatrixAxis::Row
                                                                  : MatrixAxis::Column};
            }
            if(m_state->problem.useScaleAB() == "Vector")
            {
                if(!m_state->preQuantizationScaleA)
                    request.epilogue.scaleA = m_state->scaleA;
                if(!m_state->preQuantizationScaleB)
                    request.epilogue.scaleB = m_state->scaleB;
            }
            if(source.bias && !m_state->useStandaloneEpilogue)
                request.epilogue.bias = source.bias;
            request.mathMode
                = accumulatorType == ScalarType::Float32
                          && m_state->problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                      ? MathMode::XFloat32
                      : MathMode::Default;
            request.outputSelection = source.outputSelection;

            if(m_state->useStandaloneEpilogue)
            {
                EpilogueRequest epilogue(*translated.intermediate, productOutput, accumulatorType);
                if(!m_state->problem.useGradient())
                    epilogue.bias = source.bias;
                epilogue.activation           = m_state->activation;
                epilogue.activationParameter0 = m_state->activationParameter0;
                epilogue.activationParameter1 = m_state->activationParameter1;
                epilogue.outputScale          = m_state->outputScale;
                if(m_state->typeD == ScalarType::Int8)
                    epilogue.outputConversion = OutputConversion::SaturatingInt8;
                epilogue.outputSelection = request.outputSelection;

                if(m_state->problem.useGradient() && m_state->problem.useBias()
                   && m_state->problem.biasSrc() == ContractionProblemGemm::D)
                {
                    translated.biasWorkspace.emplace(accumulatorType,
                                                     Shape{m_state->m, m_state->n});
                    epilogue.rawOutput     = *translated.biasWorkspace;
                    epilogue.rawOutputType = epilogue.rawOutput->type();
                }
                epilogue.auxiliaryInput  = source.auxiliaryInput;
                epilogue.auxiliaryOutput = source.auxiliaryOutput;
                if(epilogue.auxiliaryOutput)
                    epilogue.auxiliaryOutputType = epilogue.auxiliaryOutput->type();
                if(m_state->problem.useGradient())
                {
                    epilogue.activationApplication = ActivationApplication::Gradient;
                    if(!epilogue.auxiliaryInput)
                    {
                        translated.gradientAuxiliary.emplace(accumulatorType,
                                                             Shape{m_state->m, m_state->n});
                        epilogue.auxiliaryInput = *translated.gradientAuxiliary;
                    }
                }
                epilogue.gateResidual = source.gateResidual;
                if(m_state->amax)
                {
                    epilogue.amax           = m_state->amax;
                    epilogue.amaxType       = epilogue.amax->type();
                    epilogue.accumulateAmax = batch != 0;
                }
                translated.epilogue.emplace(std::move(epilogue));

                if(m_state->problem.useGradient() && m_state->problem.useBias())
                {
                    const auto biasOutput = *source.biasOutput;
                    const auto biasAxis   = source.bias->axis;
                    if(m_state->problem.biasSrc() == ContractionProblemGemm::D)
                    {
                        translated.biasReduction.emplace(
                            *translated.biasWorkspace,
                            biasOutput,
                            accumulatorType,
                            std::vector<size_t>{biasAxis == MatrixAxis::Row ? size_t(1)
                                                                            : size_t(0)});
                    }
                    else if(m_state->problem.biasSrc() == ContractionProblemGemm::A)
                    {
                        translated.biasReduction.emplace(
                            request.a.values, biasOutput, accumulatorType, std::vector<size_t>{1});
                    }
                    else
                    {
                        translated.biasReduction.emplace(
                            request.b.values, biasOutput, accumulatorType, std::vector<size_t>{0});
                    }
                }
            }

            return translated;
        }
        catch(std::invalid_argument const& error)
        {
            return failure(TranslationFailureCode::InvalidDescriptor, error.what());
        }
        catch(std::out_of_range const& error)
        {
            return failure(TranslationFailureCode::InvalidDescriptor, error.what());
        }
        catch(std::overflow_error const& error)
        {
            return failure(TranslationFailureCode::InvalidDescriptor, error.what());
        }
    }

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                size_t                        elementsToValidate)
    {
        try
        {
            auto state = std::make_unique<GemmInvocationAdapter::State>(
                problem, inputs, elementsToValidate);
            if(auto preflightFailure = state->preflight())
                return std::move(*preflightFailure);
            return GemmInvocationAdapter(std::move(state));
        }
        catch(std::invalid_argument const& error)
        {
            return detail::failure(TranslationFailureCode::InvalidDescriptor, error.what());
        }
        catch(std::out_of_range const& error)
        {
            return detail::failure(TranslationFailureCode::InvalidDescriptor, error.what());
        }
        catch(std::overflow_error const& error)
        {
            return detail::failure(TranslationFailureCode::InvalidDescriptor, error.what());
        }
    }
} // namespace TensileLite::Client::reference_adapter
