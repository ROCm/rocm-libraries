// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <TensileLite/Client/HostNumerics/GemmInvocationAdapter.hpp>
#include <TensileLite/Client/HostNumerics/HostNumericsBridge.hpp>

#include <Tensile/TensorDescriptor_fwd.hpp>
#include <Tensile/Utils.hpp>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace TensileLite::Client::HostNumerics
{
    namespace detail
    {
        using namespace roc::host_numerics;
        using TensileLite::Client::checkedHostNumericsPtrdiff;
        using TensileLite::Client::hostNumericsLayout;

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
                    "TensileLite scalar type has no host-numerics conversion.");
            }
        }

        inline std::span<const std::byte>
            storageSpan(ScalarType type, const void* pointer, size_t elements)
        {
            const Layout layout = Layout::contiguousLastDimensionFastest(Shape{elements});
            return {static_cast<const std::byte*>(pointer), storageBytesForLayout(type, layout)};
        }

        inline std::complex<double> scalarFromStorage(ScalarType type, const void* pointer)
        {
            const Tensor view = Tensor::copyEncodedBackingStorage(
                type, Layout::contiguousLastDimensionFastest(Shape{1}),
                storageSpan(type, pointer, 1));
            if(scalarTypeInfo(type).category == ScalarCategory::Complex)
                return view.loadAs<std::complex<double>>({0});
            return {view.loadAs<double>({0}), 0.0};
        }

        inline size_t descriptorStorageBytes(ScalarType type, TensorDescriptor const& descriptor)
        {
            return storageBytesForLayout(type, hostNumericsLayout(descriptor));
        }

        inline std::span<std::byte> mutableDescriptorStorage(ScalarType              type,
                                                             TensorDescriptor const& descriptor,
                                                             void*                   pointer,
                                                             ptrdiff_t               byteOffset = 0)
        {
            return {static_cast<std::byte*>(pointer) + byteOffset,
                    descriptorStorageBytes(type, descriptor)};
        }

        enum class MatrixAxis
        {
            Row,
            Column,
        };

        inline MatrixAxis inferBiasAxis(size_t length, size_t rows, size_t columns, int factorDim)
        {
            MatrixAxis axis = factorDim == 0 ? MatrixAxis::Row : MatrixAxis::Column;
            if(length == rows && length != columns)
                axis = MatrixAxis::Row;
            else if(length == columns && length != rows)
                axis = MatrixAxis::Column;
            return axis;
        }

        inline Tensor broadcastVectorAsMatrix(const Tensor& values, MatrixAxis axis)
        {
            return values.expandDims(axis == MatrixAxis::Row ? 1 : 0);
        }

        enum class ScaleABMode
        {
            None,
            Scalar,
            Vector,
        };

        struct BatchInputs
        {
            Tensor                              a;
            Tensor                              b;
            Tensor                              c;
            Tensor                              d;
            OutputSelection                     outputSelection;
            std::optional<Tensor>                bias;
            std::optional<MatrixAxis>            biasAxis;
            std::optional<Tensor>               biasOutput;
            std::optional<Tensor>               auxiliaryInput;
            std::optional<Tensor>               auxiliaryOutput;
            std::optional<Tensor>               gateResidual;
            std::span<std::byte>                dDestination;
            std::optional<std::span<std::byte>> biasOutputDestination;
            std::optional<std::span<std::byte>> auxiliaryOutputDestination;
        };

        struct BorrowedConstTensor
        {
            Layout                     layout;
            std::span<const std::byte> storage;
        };

        struct BorrowedMutableTensor
        {
            Layout               layout;
            std::span<std::byte> storage;
        };

        struct BatchPlan
        {
            BorrowedConstTensor                   a;
            BorrowedConstTensor                   b;
            BorrowedConstTensor                   c;
            BorrowedMutableTensor                 d;
            OutputSelection                       outputSelection;
            std::optional<BorrowedConstTensor>    bias;
            std::optional<BorrowedMutableTensor>  biasOutput;
            MatrixAxis                            biasAxis = MatrixAxis::Row;
            std::optional<BorrowedConstTensor>    auxiliaryInput;
            std::optional<BorrowedMutableTensor>  auxiliaryOutput;
            std::optional<BorrowedConstTensor>    gateResidual;
            std::optional<BorrowedConstTensor>    blockScaleA;
            std::optional<BorrowedConstTensor>    blockScaleB;
        };

        // Rebase one affine batch to the smallest byte-aligned storage window.
        // Packed formats may retain a few prefix bits so element zero remains
        // correctly aligned within its first byte.
        inline std::pair<Layout, ptrdiff_t>
            rebaseLayoutToByteAlignedStorage(ScalarType type, const Layout& layout)
        {
            if(layout.shape().elementCount() == 0)
                return {Layout(layout.shape(),
                               std::vector<ptrdiff_t>(layout.strides().begin(),
                                                      layout.strides().end())),
                        0};

            std::vector<size_t> lowerCorner(layout.shape().rank(), 0);
            for(size_t dimension = 0; dimension < layout.shape().rank(); ++dimension)
                if(layout.stride(dimension) < 0)
                    lowerCorner[dimension] = layout.shape()[dimension] - 1;
            const ptrdiff_t lowerOffset = layout.elementOffset(lowerCorner);
            if(lowerOffset < 0)
                throw std::invalid_argument(
                    "TensileLite adapter layout addresses before its batch storage base.");

            const size_t storageBits = scalarTypeInfo(type).storageBits;
            const size_t alignmentElements = 8 / std::gcd<size_t>(8, storageBits);
            const size_t alignedOffset =
                static_cast<size_t>(lowerOffset) / alignmentElements * alignmentElements;
            const size_t bytesPerAlignedGroup = alignmentElements * storageBits / 8;
            const ptrdiff_t byteOffset =
                checkedMultiply(alignedOffset / alignmentElements, bytesPerAlignedGroup);
            const ptrdiff_t rebasedOffset =
                checkedAdd(layout.offset(), -static_cast<ptrdiff_t>(alignedOffset));
            return {Layout(layout.shape(),
                           std::vector<ptrdiff_t>(layout.strides().begin(),
                                                  layout.strides().end()),
                           rebasedOffset),
                    byteOffset};
        }

        inline BorrowedConstTensor makeBorrowedConstTensor(ScalarType type,
                                                            Layout layout,
                                                            const void* storage,
                                                            ptrdiff_t byteOffset = 0)
        {
            auto [rebasedLayout, layoutByteOffset]
                = rebaseLayoutToByteAlignedStorage(type, layout);
            const auto* bytes = static_cast<const std::byte*>(storage)
                                + checkedAdd(byteOffset, layoutByteOffset);
            const size_t requiredBytes = storageBytesForLayout(type, rebasedLayout);
            return {std::move(rebasedLayout),
                    std::span<const std::byte>(bytes, requiredBytes)};
        }

        inline BorrowedMutableTensor makeBorrowedMutableTensor(ScalarType type,
                                                                Layout layout,
                                                                void* storage,
                                                                ptrdiff_t byteOffset = 0)
        {
            auto [rebasedLayout, layoutByteOffset]
                = rebaseLayoutToByteAlignedStorage(type, layout);
            auto* bytes = static_cast<std::byte*>(storage)
                          + checkedAdd(byteOffset, layoutByteOffset);
            const size_t requiredBytes = storageBytesForLayout(type, rebasedLayout);
            return {std::move(rebasedLayout),
                    std::span<std::byte>(bytes, requiredBytes)};
        }
    } // namespace detail

    struct GemmInvocationAdapter::State
    {
        using Activation = roc::host_numerics::Activation;
        using Layout     = roc::host_numerics::Layout;
        using MathMode   = roc::host_numerics::MathMode;
        using MatrixAxis      = detail::MatrixAxis;
        using OutputSelection = roc::host_numerics::OutputSelection;
        using ScalarType = roc::host_numerics::ScalarType;
        using Tensor     = roc::host_numerics::Tensor;

        Tensor makeAddendTensor(const Layout& layout, std::span<const std::byte> source) const
        {
            return readC ? Tensor::copyEncodedBackingStorage(typeC, layout, source)
                         : Tensor(typeC,
                                  Layout(layout.shape(),
                                         std::vector<ptrdiff_t>(layout.shape().rank(), 0)));
        }

        static Tensor makeOutputTensorForType(ScalarType type, const Layout& layout)
        {
            return scalarTypeInfo(type).isPacked() ? Tensor(type, layout)
                                                   : Tensor::allocateUninitialized(type, layout);
        }

        Tensor makeOutputTensor(const Layout& layout) const
        {
            return makeOutputTensorForType(typeD, layout);
        }

        std::optional<TranslationFailure>
            normalizeProblem(ContractionProblemGemm const& problem,
                             OutputSelection               outputSelection)
        {
            using namespace roc::host_numerics;
            using detail::failure;

            if(problem.boundIndices().size() != 1 || problem.freeIndicesA().size() != 1
               || problem.freeIndicesB().size() != 1 || problem.batchIndices().size() != 1)
            {
                return failure(
                    TranslationFailureCode::UnsupportedContraction,
                    "Host numerics requires one A free index, one B free index, one batch "
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
            if(problem.useScaleAB() == "Scalar")
                scaleABMode = detail::ScaleABMode::Scalar;
            else if(problem.useScaleAB() == "Vector")
                scaleABMode = detail::ScaleABMode::Vector;
            else if(!problem.useScaleAB().empty())
            {
                return failure(TranslationFailureCode::InvalidScaleConfiguration,
                               "ScaleAB mode must be empty, Scalar, or Vector.");
            }

            try
            {
                typeA                    = toHostNumericsScalarType(problem.a().dataType());
                typeB                    = toHostNumericsScalarType(problem.b().dataType());
                typeC                    = toHostNumericsScalarType(problem.c().dataType());
                typeD                    = toHostNumericsScalarType(problem.d().dataType());
                operationAccumulatorType = toHostNumericsScalarType(problem.computeType());
                betaType                 = toHostNumericsScalarType(problem.betaType());
                alphaType                = toHostNumericsScalarType(problem.alphaType());
                computeTypeA             = problem.computeInputTypeA() == rocisa::DataType::None
                                               ? typeA
                                               : toHostNumericsScalarType(problem.computeInputTypeA());
                computeTypeB             = problem.computeInputTypeB() == rocisa::DataType::None
                                               ? typeB
                                               : toHostNumericsScalarType(problem.computeInputTypeB());
                if(problem.useBias())
                    biasType = toHostNumericsScalarType(problem.bias().dataType());
                if(problem.useE())
                    auxiliaryType = toHostNumericsScalarType(
                        problem.tensors()[ContractionProblemGemm::TENSOR::E].dataType());
                if(problem.useGateResidual())
                    gateType = toHostNumericsScalarType(
                        problem.tensors()[ContractionProblemGemm::TENSOR::GATE_RESIDUAL].dataType());
            }
            catch(std::invalid_argument const& error)
            {
                return failure(TranslationFailureCode::UnsupportedDataType, error.what());
            }

            useGradient    = problem.useGradient();
            useBias        = problem.useBias();
            biasSource     = problem.biasSrc();
            scaleAlphaAxis = problem.getParams().factorDim() == 0 ? MatrixAxis::Row
                                                                  : MatrixAxis::Column;
            mathMode       = operationAccumulatorType == ScalarType::Float32
                                       && problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                                 ? MathMode::XFloat32
                                 : MathMode::Default;
            useStandaloneEpilogue
                = useGradient || problem.outputAmaxD() || problem.useE()
                  || problem.useGateResidual();
            preQuantizationScaleA
                = scalarTypeInfo(typeA).storageBits > scalarTypeInfo(computeTypeA).storageBits;
            preQuantizationScaleB
                = scalarTypeInfo(typeB).storageBits > scalarTypeInfo(computeTypeB).storageBits;

            ActivationType concreteActivation = problem.activationType();
            if(concreteActivation == ActivationType::All
               || concreteActivation == ActivationType::Hipblaslt_all)
                concreteActivation = problem.getParams().activationEnum();
            try
            {
                activation = toHostNumericsActivation(concreteActivation, useGradient);
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

            const auto& freeIndexA = problem.freeIndicesA()[0];
            const auto& freeIndexB = problem.freeIndicesB()[0];
            indexMA                = freeIndexA.i;
            indexKA                = problem.boundIndices()[0].a;
            indexNB                = freeIndexB.i;
            indexKB                = problem.boundIndices()[0].b;
            indexMC                = freeIndexA.c;
            indexNC                = freeIndexB.c;
            indexMD                = freeIndexA.d;
            indexND                = freeIndexB.d;
            batchA                 = problem.batchIndices()[0].a;
            batchB                 = problem.batchIndices()[0].b;
            batchC                 = problem.batchIndices()[0].c;
            batchD                 = problem.batchIndices()[0].d;
            m                      = problem.freeSizeA(0);
            n                      = problem.freeSizeB(0);
            k                      = problem.boundSize(0);
            batches                = problem.batchSize(0);

            mxBlockA = problem.mxBlockA();
            mxBlockB = problem.mxBlockB();
            if(mxBlockA > 0)
            {
                try
                {
                    mxScaleTypeA = toHostNumericsMxScaleType(problem.mxTypeA());
                    mxScaleTypeB = toHostNumericsMxScaleType(problem.mxTypeB());
                }
                catch(std::invalid_argument const& error)
                {
                    return failure(TranslationFailureCode::UnsupportedDataType, error.what());
                }
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

            const bool requiresCompleteD
                = problem.outputAmaxD()
                  || (useGradient && useBias && biasSource == ContractionProblemGemm::D);
            const OutputSelection globalSelection
                = requiresCompleteD ? OutputSelection::all(outputSelection.indexOrder())
                                    : std::move(outputSelection);
            selectAllOutputs = globalSelection.selectsAll();
            selectedByBatch.clear();
            if(!selectAllOutputs)
            {
                selectedByBatch.resize(batches);
                const Shape outputShape(problem.d().sizes());
                for(const size_t logicalIndex :
                    globalSelection.indices(problem.d().totalLogicalElements()))
                {
                    const std::vector<size_t> coordinate
                        = outputShape.coordinates(logicalIndex, globalSelection.indexOrder());
                    const size_t selectedBatch = coordinate[batchD];
                    const size_t row           = coordinate[indexMD];
                    const size_t column        = coordinate[indexND];
                    selectedByBatch.at(selectedBatch).push_back(row * n + column);
                }
            }
            return std::nullopt;
        }

        std::optional<TranslationFailure> bindInputs(ContractionProblemGemm const& problem,
                                                     ContractionInputs const&      inputs)
        {
            using namespace roc::host_numerics;
            using detail::failure;

            if(problem.useBias() && inputs.bias == nullptr && inputs.batchBias == nullptr)
                return failure(TranslationFailureCode::MissingInput, "Bias input is missing.");
            if(problem.useScaleAlphaVec() && inputs.scaleAlphaVec == nullptr)
            {
                return failure(TranslationFailureCode::MissingInput,
                               "Scale-alpha vector input is missing.");
            }
            if(scaleABMode != detail::ScaleABMode::None
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
                alpha = detail::scalarValue(problem.alphaType(), inputs.alpha);
                beta  = detail::scalarValue(problem.betaType(), inputs.beta);
            }
            catch(std::invalid_argument const& error)
            {
                return failure(TranslationFailureCode::UnsupportedDataType, error.what());
            }

            const bool computeProduct = alpha != std::complex<double>(0.0, 0.0);

            if(problem.useScaleCD())
            {
                scaleC = detail::scalarFromStorage(betaType, inputs.scaleC);
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
                = detail::checkedHostNumericsPtrdiff(inputs.batchOffsetA);
            const ptrdiff_t batchOffsetB
                = detail::checkedHostNumericsPtrdiff(inputs.batchOffsetB);
            const ptrdiff_t batchOffsetC
                = detail::checkedHostNumericsPtrdiff(inputs.batchOffsetC);
            const ptrdiff_t batchOffsetD
                = detail::checkedHostNumericsPtrdiff(inputs.batchOffsetD);

            readA = computeProduct
                    || (useGradient && useBias && biasSource == ContractionProblemGemm::A);
            readB = computeProduct
                    || (useGradient && useBias && biasSource == ContractionProblemGemm::B);
            readC = beta != std::complex<double>(0.0, 0.0);
            TensorDescriptor const* auxiliaryDescriptor = nullptr;
            if(problem.useE())
                auxiliaryDescriptor = &problem.tensors()[ContractionProblemGemm::TENSOR::E];

            TensorDescriptor const* gateDescriptor = nullptr;
            if(problem.useGateResidual())
                gateDescriptor = &problem.tensors()[ContractionProblemGemm::TENSOR::GATE_RESIDUAL];

            if(problem.outputAmaxD())
            {
                auto const& descriptor = problem.tensors()[ContractionProblemGemm::TENSOR::AMAXD];
                try
                {
                    const ScalarType amaxType = toHostNumericsScalarType(descriptor.dataType());
                    const auto       storage
                        = detail::mutableDescriptorStorage(amaxType, descriptor, inputs.amaxD);
                    amax = Tensor::copyEncodedBackingStorage(
                        amaxType, Layout::contiguousLastDimensionFastest(Shape{1}), storage);
                    amaxDestination = storage;
                }
                catch(std::invalid_argument const& error)
                {
                    return failure(TranslationFailureCode::UnsupportedDataType, error.what());
                }
            }

            const size_t scaleAlphaLength = scaleAlphaAxis == MatrixAxis::Row ? m : n;
            if(computeProduct && problem.useScaleAlphaVec())
            {
                scaleAlpha = Tensor::copyEncodedBackingStorage(
                    alphaType,
                    Layout::contiguousLastDimensionFastest(Shape{scaleAlphaLength}),
                    detail::storageSpan(alphaType, inputs.scaleAlphaVec, scaleAlphaLength));
            }
            if(computeProduct && scaleABMode == detail::ScaleABMode::Scalar)
            {
                scaleA = Tensor::copyEncodedBackingStorage(
                    alphaType, Layout::contiguousLastDimensionFastest(Shape{1}),
                    detail::storageSpan(alphaType, inputs.scaleA, 1));
                scaleB = Tensor::copyEncodedBackingStorage(
                    alphaType, Layout::contiguousLastDimensionFastest(Shape{1}),
                    detail::storageSpan(alphaType, inputs.scaleB, 1));
            }
            else if(computeProduct && scaleABMode == detail::ScaleABMode::Vector)
            {
                scaleA = Tensor::copyEncodedBackingStorage(
                    alphaType, Layout::contiguousLastDimensionFastest(Shape{m}),
                    detail::storageSpan(alphaType, inputs.scaleA, m));
                scaleB = Tensor::copyEncodedBackingStorage(
                    alphaType, Layout::contiguousLastDimensionFastest(Shape{n}),
                    detail::storageSpan(alphaType, inputs.scaleB, n));
            }

            batchPlans.reserve(batches);
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
                    = detail::checkedHostNumericsPtrdiff(problem.a().strides()[indexKA]);
                ptrdiff_t strideKB
                    = detail::checkedHostNumericsPtrdiff(problem.b().strides()[indexKB]);
                if(problem.boundIndices()[0].aMirror && k != 0)
                {
                    offsetA = detail::checkedAdd(
                        offsetA, detail::checkedMultiply(k - 1, problem.a().strides()[indexKA]));
                    strideKA = -strideKA;
                }
                if(problem.boundIndices()[0].bMirror && k != 0)
                {
                    offsetB = detail::checkedAdd(
                        offsetB, detail::checkedMultiply(k - 1, problem.b().strides()[indexKB]));
                    strideKB = -strideKB;
                }

                const Layout layoutA(
                    Shape{m, k},
                    {detail::checkedHostNumericsPtrdiff(problem.a().strides()[indexMA]), strideKA},
                    offsetA);
                const Layout layoutB(
                    Shape{k, n},
                    {strideKB, detail::checkedHostNumericsPtrdiff(problem.b().strides()[indexNB])},
                    offsetB);
                const Layout layoutC(
                    Shape{m, n},
                    {detail::checkedHostNumericsPtrdiff(problem.c().strides()[indexMC]),
                     detail::checkedHostNumericsPtrdiff(problem.c().strides()[indexNC])},
                    offsetC);
                const Layout layoutD(
                    Shape{m, n},
                    {detail::checkedHostNumericsPtrdiff(problem.d().strides()[indexMD]),
                     detail::checkedHostNumericsPtrdiff(problem.d().strides()[indexND])},
                    offsetD);

                const void* currentA = inputs.batchA == nullptr ? inputs.a : inputs.batchA[batch];
                const void* currentB = inputs.batchB == nullptr ? inputs.b : inputs.batchB[batch];
                const void* currentC = inputs.batchC == nullptr ? inputs.c : inputs.batchC[batch];
                void*       currentD = inputs.batchD == nullptr ? inputs.d : inputs.batchD[batch];

                detail::BatchPlan plan{
                    .a = detail::makeBorrowedConstTensor(
                        typeA, layoutA, currentA, batchOffsetA),
                    .b = detail::makeBorrowedConstTensor(
                        typeB, layoutB, currentB, batchOffsetB),
                    .c = detail::makeBorrowedConstTensor(
                        typeC, layoutC, currentC, batchOffsetC),
                    .d = detail::makeBorrowedMutableTensor(
                        typeD, layoutD, currentD, batchOffsetD),
                    .outputSelection = selectAllOutputs
                                           ? OutputSelection::all()
                                           : OutputSelection::explicitIndices(
                                                 selectedByBatch[batch]),
                };

                if(problem.useBias())
                {
                    ptrdiff_t runtimeBiasOffset = 0;
                    const size_t runtimeBiasLength = problem.bias().sizes()[0];
                    if(inputs.batchBias == nullptr && problem.bias().dimensions() > 2)
                        runtimeBiasOffset
                            = detail::checkedMultiply(batch, problem.bias().strides()[2]);
                    const Layout biasLayout(Shape{runtimeBiasLength}, {1}, runtimeBiasOffset);
                    const void* currentBias
                        = inputs.batchBias == nullptr ? inputs.bias : inputs.batchBias[batch];
                    plan.biasAxis = detail::inferBiasAxis(
                        runtimeBiasLength, m, n, problem.getParams().factorDim());
                    if(useGradient)
                    {
                        void* currentBiasOutput = inputs.batchBias == nullptr
                                                      ? const_cast<void*>(inputs.bias)
                                                      : const_cast<void*>(inputs.batchBias[batch]);
                        plan.biasOutput = detail::makeBorrowedMutableTensor(
                            *biasType, biasLayout, currentBiasOutput);
                    }
                    else
                    {
                        plan.bias = detail::makeBorrowedConstTensor(
                            *biasType, biasLayout, currentBias);
                    }
                }

                if(problem.useE())
                {
                    const ptrdiff_t offsetE
                        = detail::checkedMultiply(batch, auxiliaryDescriptor->strides()[batchD]);
                    const Layout layoutE(
                        Shape{m, n},
                        {detail::checkedHostNumericsPtrdiff(
                             auxiliaryDescriptor->strides()[indexMD]),
                         detail::checkedHostNumericsPtrdiff(
                             auxiliaryDescriptor->strides()[indexND])},
                        offsetE);
                    if(useGradient)
                        plan.auxiliaryInput = detail::makeBorrowedConstTensor(
                            *auxiliaryType, layoutE, inputs.e);
                    else
                        plan.auxiliaryOutput = detail::makeBorrowedMutableTensor(
                            *auxiliaryType, layoutE, inputs.e);
                }

                if(problem.useGateResidual())
                {
                    const ptrdiff_t offsetGate
                        = inputs.batchGateResidual == nullptr
                              ? detail::checkedMultiply(batch, gateDescriptor->strides()[batchD])
                              : 0;
                    const Layout gateLayout(
                        Shape{m, n},
                        {detail::checkedHostNumericsPtrdiff(gateDescriptor->strides()[indexMD]),
                         detail::checkedHostNumericsPtrdiff(gateDescriptor->strides()[indexND])},
                        offsetGate);
                    const void* currentGate = inputs.batchGateResidual == nullptr
                                                  ? inputs.gateResidual
                                                  : inputs.batchGateResidual[batch];
                    plan.gateResidual = detail::makeBorrowedConstTensor(
                        *gateType, gateLayout, currentGate);
                }

                if(computeProduct && mxBlockA > 0)
                {
                    const size_t blockCountA
                        = k / mxBlockA + static_cast<size_t>(k % mxBlockA != 0);
                    const size_t blockCountB
                        = k / mxBlockB + static_cast<size_t>(k % mxBlockB != 0);
                    const ptrdiff_t positiveBlockStrideA
                        = detail::checkedHostNumericsPtrdiff(strideMxsaBlock);
                    const ptrdiff_t positiveBlockStrideB
                        = detail::checkedHostNumericsPtrdiff(strideMxsbBlock);
                    const Layout scaleLayoutA(
                        Shape{m, blockCountA},
                        {detail::checkedHostNumericsPtrdiff(strideMxsaM),
                         problem.boundIndices()[0].aMirror ? -positiveBlockStrideA
                                                           : positiveBlockStrideA},
                        problem.boundIndices()[0].aMirror && blockCountA != 0
                            ? detail::checkedMultiply(blockCountA - 1, strideMxsaBlock)
                            : 0);
                    const Layout scaleLayoutB(
                        Shape{n, blockCountB},
                        {detail::checkedHostNumericsPtrdiff(strideMxsbN),
                         problem.boundIndices()[0].bMirror ? -positiveBlockStrideB
                                                           : positiveBlockStrideB},
                        problem.boundIndices()[0].bMirror && blockCountB != 0
                            ? detail::checkedMultiply(blockCountB - 1, strideMxsbBlock)
                            : 0);
                    plan.blockScaleA = detail::makeBorrowedConstTensor(
                        mxScaleTypeA,
                        scaleLayoutA,
                        inputs.mxsa,
                        detail::checkedMultiply(batch, strideBatchMxsa));
                    plan.blockScaleB = detail::makeBorrowedConstTensor(
                        mxScaleTypeB,
                        scaleLayoutB,
                        inputs.mxsb,
                        detail::checkedMultiply(batch, strideBatchMxsb));
                }

                batchPlans.push_back(std::move(plan));
            }

            return std::nullopt;
        }

        detail::BatchInputs materializeBatch(size_t batch) const
        {
            using namespace roc::host_numerics;
            const detail::BatchPlan& plan = batchPlans.at(batch);
            Tensor currentA = readA ? Tensor::copyEncodedBackingStorage(
                                          typeA, plan.a.layout, plan.a.storage)
                                    : Tensor(typeA, Layout(Shape{m, k}, {0, 0}));
            Tensor currentB = readB ? Tensor::copyEncodedBackingStorage(
                                          typeB, plan.b.layout, plan.b.storage)
                                    : Tensor(typeB, Layout(Shape{k, n}, {0, 0}));
            Tensor currentC = makeAddendTensor(plan.c.layout, plan.c.storage);
            Tensor currentD = makeOutputTensor(plan.d.layout);

            std::optional<Tensor>               runtimeBias;
            std::optional<Tensor>               runtimeBiasOutput;
            std::optional<std::span<std::byte>> runtimeBiasOutputDestination;
            if(plan.bias)
            {
                runtimeBias = detail::broadcastVectorAsMatrix(
                    Tensor::copyEncodedBackingStorage(
                        *biasType, plan.bias->layout, plan.bias->storage),
                    plan.biasAxis);
            }
            if(plan.biasOutput)
            {
                runtimeBiasOutput = makeOutputTensorForType(
                    *biasType, plan.biasOutput->layout);
                runtimeBiasOutputDestination = plan.biasOutput->storage;
            }

            std::optional<Tensor> auxiliaryInput;
            std::optional<Tensor> auxiliaryOutput;
            if(plan.auxiliaryInput)
                auxiliaryInput = Tensor::copyEncodedBackingStorage(
                    *auxiliaryType, plan.auxiliaryInput->layout, plan.auxiliaryInput->storage);
            if(plan.auxiliaryOutput)
                auxiliaryOutput = makeOutputTensorForType(
                    *auxiliaryType, plan.auxiliaryOutput->layout);

            std::optional<Tensor> runtimeGate;
            if(plan.gateResidual)
                runtimeGate = Tensor::copyEncodedBackingStorage(
                    *gateType, plan.gateResidual->layout, plan.gateResidual->storage);

            return {
                std::move(currentA),
                std::move(currentB),
                std::move(currentC),
                std::move(currentD),
                plan.outputSelection,
                std::move(runtimeBias),
                useBias ? std::optional<MatrixAxis>(plan.biasAxis) : std::nullopt,
                std::move(runtimeBiasOutput),
                std::move(auxiliaryInput),
                std::move(auxiliaryOutput),
                std::move(runtimeGate),
                plan.d.storage,
                runtimeBiasOutputDestination,
                plan.auxiliaryOutput
                    ? std::optional<std::span<std::byte>>(plan.auxiliaryOutput->storage)
                    : std::nullopt,
            };
        }

        ScalarType typeA                    = ScalarType::Float32;
        ScalarType typeB                    = ScalarType::Float32;
        ScalarType typeC                    = ScalarType::Float32;
        ScalarType typeD                    = ScalarType::Float32;
        ScalarType operationAccumulatorType = ScalarType::Float32;
        ScalarType computeTypeA             = ScalarType::Float32;
        ScalarType computeTypeB             = ScalarType::Float32;
        ScalarType mxScaleTypeA             = ScalarType::E8M0;
        ScalarType mxScaleTypeB             = ScalarType::E8M0;
        ScalarType alphaType                = ScalarType::Float32;
        ScalarType betaType                 = ScalarType::Float32;

        std::complex<double> alpha                = {1.0, 0.0};
        std::complex<double> beta                 = {0.0, 0.0};
        std::complex<double> scaleC               = {1.0, 0.0};
        std::complex<double> outputScale          = {1.0, 0.0};
        Activation           activation           = Activation::None;
        double               activationParameter0 = 0.0;
        double               activationParameter1 = 0.0;

        bool useStandaloneEpilogue = false;
        bool useGradient           = false;
        bool useBias               = false;
        bool preQuantizationScaleA = false;
        bool preQuantizationScaleB = false;
        bool aConjugate            = false;
        bool bConjugate            = false;

        ContractionProblemGemm::TENSOR biasSource = ContractionProblemGemm::D;
        MatrixAxis                     scaleAlphaAxis = MatrixAxis::Row;
        MathMode                       mathMode        = MathMode::Default;
        detail::ScaleABMode            scaleABMode     = detail::ScaleABMode::None;

        size_t m = 0;
        size_t n = 0;
        size_t k = 0;
        size_t batches = 0;

        size_t indexMA = 0;
        size_t indexKA = 0;
        size_t indexNB = 0;
        size_t indexKB = 0;
        size_t indexMC = 0;
        size_t indexNC = 0;
        size_t indexMD = 0;
        size_t indexND = 0;
        size_t batchA  = 0;
        size_t batchB  = 0;
        size_t batchC  = 0;
        size_t batchD  = 0;

        size_t mxBlockA = 0;
        size_t mxBlockB = 0;

        size_t strideMxsaM     = 0;
        size_t strideMxsaBlock = 0;
        size_t strideMxsbN     = 0;
        size_t strideMxsbBlock = 0;
        size_t strideBatchMxsa = 0;
        size_t strideBatchMxsb = 0;

        std::vector<std::vector<size_t>> selectedByBatch;
        bool                             selectAllOutputs = true;

        std::optional<Tensor>               scaleAlpha;
        std::optional<Tensor>               scaleA;
        std::optional<Tensor>               scaleB;
        std::optional<Tensor>               amax;
        std::optional<std::span<std::byte>> amaxDestination;
        bool                                readA        = true;
        bool                                readB        = true;
        bool                                readC        = true;
        std::optional<ScalarType>           biasType;
        std::optional<ScalarType>           auxiliaryType;
        std::optional<ScalarType>           gateType;
        std::vector<detail::BatchPlan>       batchPlans;
    };

    GemmInvocationAdapter::GemmInvocationAdapter(std::unique_ptr<const State> state)
        : m_state(std::move(state))
    {
    }

    GemmInvocationAdapter::~GemmInvocationAdapter() = default;

    GemmInvocationAdapter::GemmInvocationAdapter(GemmInvocationAdapter&&) noexcept = default;

    GemmInvocationAdapter&
        GemmInvocationAdapter::operator=(GemmInvocationAdapter&&) noexcept = default;

    roc::host_numerics::GemmBackend
        TranslatedGemmBatch::runGemm(roc::host_numerics::GemmBackend backend) const
    {
        return roc::host_numerics::referenceGemmInto(a, b, c, d, options, backend);
    }

    void TranslatedGemmBatch::runPostGemmOperationsAndCopyOutputs() const
    {
        if(epilogue)
            referenceEpilogueInto(epilogue->input, epilogue->outputs, epilogue->options);
        if(biasReduction)
            referenceSumInto(biasReduction->input,
                             biasReduction->output,
                             biasReduction->axes,
                             biasReduction->accumulatorType);
        for(const auto& copyBack : copyBacks)
        {
            if(copyBack.selection.selectsAll())
                copyBack.source.copyLogicalElementsToEncodedStorage(copyBack.destination);
            else
            {
                const auto selected =
                    copyBack.selection.indices(copyBack.source.elementCount());
                copyBack.source.copySelectedElementsToEncodedStorage(
                    copyBack.destination,
                    selected,
                    roc::host_numerics::IndexOrder::LastDimensionFastest);
            }
        }
    }

    size_t GemmInvocationAdapter::batchCount() const
    {
        return m_state->batchPlans.size();
    }

    roc::host_numerics::GemmBackend
        GemmInvocationAdapter::execute(roc::host_numerics::GemmBackend backend) const
    {
        using namespace roc::host_numerics;

        GemmBackend combined   = GemmBackend::Pointwise;
        bool        hasBackend = false;
        for(size_t batch = 0; batch < batchCount(); ++batch)
        {
            auto translation = translateBatch(batch);
            if(std::holds_alternative<TranslationFailure>(translation))
            {
                const auto& failure = std::get<TranslationFailure>(translation);
                throw std::invalid_argument("TensileLite host-numerics translation failed: "
                                            + failure.reason);
            }

            TranslatedGemmBatch translated
                = std::move(std::get<TranslatedGemmBatch>(translation));
            const GemmBackend backendUsed = translated.runGemm(backend);
            translated.runPostGemmOperationsAndCopyOutputs();

            if(!hasBackend)
            {
                combined   = backendUsed;
                hasBackend = true;
            }
            else if(combined != backendUsed)
            {
                combined = GemmBackend::Mixed;
            }
        }
        return combined;
    }

    std::variant<TranslatedGemmBatch, TranslationFailure> GemmInvocationAdapter::translateBatch(
        size_t batch) const
    {
        using namespace roc::host_numerics;
        using detail::failure;

        const ScalarType accumulatorType = m_state->operationAccumulatorType;

        if(batch >= m_state->batchPlans.size())
        {
            return failure(TranslationFailureCode::InvalidBatchIndex,
                           "Requested reference batch index is out of range.");
        }
        try
        {
            const auto&         plan   = m_state->batchPlans.at(batch);
            const auto          source = m_state->materializeBatch(batch);

            Tensor                productOutput = source.d;
            Tensor                gemmOutput    = productOutput;
            std::optional<Tensor> intermediate;
            if(m_state->useStandaloneEpilogue)
            {
                intermediate.emplace(accumulatorType, Shape{m_state->m, m_state->n});
                gemmOutput = *intermediate;
            }

            TranslatedGemmBatch translated(
                source.a, source.b, source.c, gemmOutput, accumulatorType);
            translated.copyBacks.push_back(
                {source.dDestination, productOutput, source.outputSelection});
            if(m_state->amax && m_state->amaxDestination)
                translated.copyBacks.push_back(
                    {*m_state->amaxDestination, *m_state->amax, OutputSelection::all()});
            if(source.biasOutput && source.biasOutputDestination)
                translated.copyBacks.push_back(
                    {*source.biasOutputDestination, *source.biasOutput, OutputSelection::all()});
            if(source.auxiliaryOutput && source.auxiliaryOutputDestination)
                translated.copyBacks.push_back({*source.auxiliaryOutputDestination,
                                                *source.auxiliaryOutput,
                                                source.outputSelection});
            auto& request           = translated.gemmOptions();
            request.computeTypeA    = m_state->computeTypeA != m_state->typeA
                                          ? std::optional<ScalarType>(m_state->computeTypeA)
                                          : std::nullopt;
            request.computeTypeB    = m_state->computeTypeB != m_state->typeB
                                          ? std::optional<ScalarType>(m_state->computeTypeB)
                                          : std::nullopt;
            if(m_state->scaleA && m_state->preQuantizationScaleA)
                request.preQuantizationScalesA.push_back(m_state->scaleA->expandDims(1));
            if(m_state->scaleB && m_state->preQuantizationScaleB)
                request.preQuantizationScalesB.push_back(m_state->scaleB->expandDims(0));
            request.conjugateA = m_state->aConjugate;
            request.conjugateB = m_state->bConjugate;
            if(plan.blockScaleA && plan.blockScaleB)
            {
                request.blockScaleA = Tensor::copyEncodedBackingStorage(
                    m_state->mxScaleTypeA, plan.blockScaleA->layout, plan.blockScaleA->storage);
                request.blockScaleB = Tensor::copyEncodedBackingStorage(
                    m_state->mxScaleTypeB, plan.blockScaleB->layout, plan.blockScaleB->storage);
                request.blockSizeA = m_state->mxBlockA;
                request.blockSizeB = m_state->mxBlockB;
            }
            request.alpha  = m_state->alpha;
            request.beta   = m_state->beta;
            request.scaleC = m_state->scaleC;
            if(!m_state->useStandaloneEpilogue && m_state->typeD == ScalarType::Int8)
                request.outputConversion = OutputConversion::SaturatingInt8;
            if(!m_state->useStandaloneEpilogue)
            {
                request.activation           = m_state->activation;
                request.activationParameter0 = m_state->activationParameter0;
                request.activationParameter1 = m_state->activationParameter1;
                request.outputScale          = m_state->outputScale;
            }
            if(m_state->scaleAlpha)
                request.scaleAlpha = detail::broadcastVectorAsMatrix(*m_state->scaleAlpha,
                                                                     m_state->scaleAlphaAxis);
            if(!m_state->preQuantizationScaleA)
                request.scaleA = m_state->scaleA
                                     ? std::optional<Tensor>(m_state->scaleA->expandDims(1))
                                     : std::nullopt;
            if(!m_state->preQuantizationScaleB)
                request.scaleB = m_state->scaleB
                                     ? std::optional<Tensor>(m_state->scaleB->expandDims(0))
                                     : std::nullopt;
            if(source.bias && !m_state->useStandaloneEpilogue)
                request.bias = source.bias;
            request.mathMode       = m_state->mathMode;
            request.outputSelection = source.outputSelection;

            if(m_state->useStandaloneEpilogue)
            {
                TranslatedGemmBatch::BoundEpilogue epilogue(
                    *intermediate, productOutput, accumulatorType);
                if(!m_state->useGradient)
                    epilogue.options.bias = source.bias;
                epilogue.options.activation           = m_state->activation;
                epilogue.options.activationParameter0 = m_state->activationParameter0;
                epilogue.options.activationParameter1 = m_state->activationParameter1;
                epilogue.options.outputScale          = m_state->outputScale;
                if(m_state->typeD == ScalarType::Int8)
                    epilogue.options.outputConversion = OutputConversion::SaturatingInt8;
                epilogue.options.outputSelection = request.outputSelection;

                if(m_state->useGradient && m_state->useBias
                   && m_state->biasSource == ContractionProblemGemm::D)
                {
                    epilogue.outputs.rawOutput.emplace(accumulatorType,
                                                       Shape{m_state->m, m_state->n});
                }
                epilogue.options.auxiliaryInput  = source.auxiliaryInput;
                epilogue.outputs.auxiliaryOutput = source.auxiliaryOutput;
                if(m_state->useGradient)
                {
                    epilogue.options.activationApplication = ActivationApplication::Gradient;
                    if(!epilogue.options.auxiliaryInput)
                        epilogue.options.auxiliaryInput.emplace(accumulatorType,
                                                                Shape{m_state->m, m_state->n});
                }
                epilogue.options.gateResidual = source.gateResidual;
                if(m_state->amax)
                {
                    epilogue.outputs.amax           = m_state->amax;
                    epilogue.options.accumulateAmax = batch != 0;
                }
                translated.epilogue.emplace(std::move(epilogue));

                if(m_state->useGradient && m_state->useBias)
                {
                    const auto biasOutput = *source.biasOutput;
                    const auto biasAxis   = *source.biasAxis;
                    if(m_state->biasSource == ContractionProblemGemm::D)
                    {
                        translated.biasReduction.emplace(
                            *translated.epilogue->outputs.rawOutput,
                            biasOutput,
                            accumulatorType,
                            std::vector<size_t>{biasAxis == detail::MatrixAxis::Row ? size_t(1)
                                                                                    : size_t(0)});
                    }
                    else if(m_state->biasSource == ContractionProblemGemm::A)
                    {
                        translated.biasReduction.emplace(
                            translated.a, biasOutput, accumulatorType, std::vector<size_t>{1});
                    }
                    else
                    {
                        translated.biasReduction.emplace(
                            translated.b, biasOutput, accumulatorType, std::vector<size_t>{0});
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
                                roc::host_numerics::OutputSelection outputSelection)
    {
        try
        {
            auto state = std::make_unique<GemmInvocationAdapter::State>();
            if(auto normalizationFailure
               = state->normalizeProblem(problem, std::move(outputSelection)))
                return std::move(*normalizationFailure);
            if(auto bindingFailure = state->bindInputs(problem, inputs))
                return std::move(*bindingFailure);
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

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                size_t                        elementsToValidate)
    {
        return translateGemmInvocation(
            problem,
            inputs,
            referenceOutputSelection(problem.d(), elementsToValidate));
    }
} // namespace TensileLite::Client::HostNumerics
