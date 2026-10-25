// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <TensileLite/Client/hostnumerics/GemmInvocationAdapter.hpp>
#include <TensileLite/Client/hostnumerics/HostNumericsBridge.hpp>
#include <roc/hostnumerics/detail/shape_access.hpp>

#include <Tensile/TensorDescriptor_fwd.hpp>
#include <Tensile/Utils.hpp>
#include <roc/hostnumerics/tensor_operations.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace TensileLite::Client::hostnumerics
{
    namespace detail
    {
        using roc::hostnumerics::ActivationFunction;
        using roc::hostnumerics::IndexSelection;
        using roc::hostnumerics::Layout;
        using roc::hostnumerics::MathMode;
        using roc::hostnumerics::ScalarCategory;
        using roc::hostnumerics::ScalarType;
        using roc::hostnumerics::scalarTypeInfo;
        using roc::hostnumerics::Shape;
        using roc::hostnumerics::storageBytesForLayout;
        using roc::hostnumerics::Tensor;
        using TensileLite::Client::hostnumericsLayout;

        inline TranslationFailure failure(TranslationFailureCode code, std::string reason)
        {
            return {code, std::move(reason)};
        }

        inline std::optional<ptrdiff_t> getMultiply(size_t left, size_t right) noexcept
        {
            if(!std::in_range<ptrdiff_t>(right))
                return std::nullopt;
            return roc::hostnumerics::detail::LayoutOffset::getOffsetMultiply(
                left, static_cast<ptrdiff_t>(right));
        }

        inline Expected<Layout, DescriptorFailure> matrixLayout(const TensorDescriptor& descriptor,
                                                                size_t                  rows,
                                                                size_t                  columns,
                                                                size_t                  rowAxis,
                                                                size_t                  columnAxis,
                                                                size_t                  batchAxis,
                                                                size_t                  batch,
                                                                bool pointerArray,
                                                                bool mirrorRows,
                                                                bool mirrorColumns)
        {
            const auto& strides = descriptor.strides();
            if(rowAxis >= strides.size() || columnAxis >= strides.size()
               || batchAxis >= strides.size())
            {
                return Unexpected(DescriptorFailure(DescriptorError::DimensionOutOfRange));
            }
            if(!std::in_range<ptrdiff_t>(strides[rowAxis])
               || !std::in_range<ptrdiff_t>(strides[columnAxis]))
            {
                return Unexpected(DescriptorFailure(DescriptorError::OffsetOutOfRange));
            }
            ptrdiff_t rowStride    = static_cast<ptrdiff_t>(strides[rowAxis]);
            ptrdiff_t columnStride = static_cast<ptrdiff_t>(strides[columnAxis]);
            auto      offset       = pointerArray ? std::optional<ptrdiff_t>(0)
                                                  : getMultiply(batch, strides[batchAxis]);
            if(!offset)
            {
                return Unexpected(DescriptorFailure(DescriptorError::OffsetMultiplyOverflow));
            }
            auto reverse = [&](size_t     extent,
                               ptrdiff_t& stride,
                               bool       mirror) -> std::optional<DescriptorFailure> {
                if(!mirror || extent == 0)
                    return std::nullopt;
                const auto delta = roc::hostnumerics::detail::LayoutOffset::getOffsetMultiply(
                    extent - 1, stride);
                if(!delta)
                {
                    return DescriptorFailure(DescriptorError::OffsetMultiplyOverflow);
                }
                offset = roc::hostnumerics::detail::LayoutOffset::getOffsetAdd(*offset, *delta);
                if(!offset)
                {
                    return DescriptorFailure(DescriptorError::OffsetAddOverflow);
                }
                stride = -stride;
                return std::nullopt;
            };
            if(auto failure = reverse(rows, rowStride, mirrorRows))
                return Unexpected(*failure);
            if(auto failure = reverse(columns, columnStride, mirrorColumns))
                return Unexpected(*failure);
            return Layout(Shape{rows, columns}, {rowStride, columnStride}, *offset);
        }

        inline bool supportsScalarValue(rocisa::DataType type) noexcept
        {
            switch(type)
            {
            case rocisa::DataType::Float:
            case rocisa::DataType::XFloat32:
            case rocisa::DataType::Double:
            case rocisa::DataType::Half:
            case rocisa::DataType::BFloat16:
            case rocisa::DataType::Int32:
            case rocisa::DataType::ComplexFloat:
            case rocisa::DataType::ComplexDouble:
                return true;
            default:
                return false;
            }
        }

        inline std::optional<std::complex<double>> scalarValue(rocisa::DataType       type,
                                                               ConstantVariant const& value)
        {
            const bool complex
                = type == rocisa::DataType::ComplexFloat || type == rocisa::DataType::ComplexDouble;
            const bool compatible = std::visit(
                [&](const auto& item) {
                    using Value = std::remove_cvref_t<decltype(item)>;
                    if constexpr(std::is_same_v<
                                     Value,
                                     std::complex<
                                         float>> || std::is_same_v<Value, std::complex<double>>)
                        return complex;
                    else if(complex)
                        return std::is_same_v<Value, float> || std::is_same_v<Value, double>;
                    else
                        return std::is_same_v<Value, float> || std::is_same_v<Value, double> || std::is_same_v<Value, Half> || std::is_same_v<Value, int32_t> || std::is_same_v<Value, BFloat16> || std::is_same_v<Value, int8_t> || std::is_same_v<Value, Float8> || std::is_same_v<Value, BFloat8> || std::is_same_v<Value, Float8_fnuz> || std::is_same_v<Value, BFloat8_fnuz>;
                },
                value);
            if(!compatible)
                return std::nullopt;
            switch(type)
            {
            case rocisa::DataType::Float:
            case rocisa::DataType::XFloat32:
                return std::complex<double>{constVariantCast<float>(value), 0.0};
            case rocisa::DataType::Double:
                return std::complex<double>{constVariantCast<double>(value), 0.0};
            case rocisa::DataType::Half:
                return std::complex<double>{static_cast<double>(constVariantCast<Half>(value)),
                                            0.0};
            case rocisa::DataType::BFloat16:
                return std::complex<double>{static_cast<double>(constVariantCast<BFloat16>(value)),
                                            0.0};
            case rocisa::DataType::Int32:
            {
                const double real = constVariantCast<double>(value);
                if(!std::isfinite(real))
                    return std::nullopt;
                const auto converted = roc::hostnumerics::convertScalar<int32_t>(
                    real, roc::hostnumerics::detail::implicitNativeConversionOptions());
                return std::complex<double>{static_cast<double>(converted), 0.0};
            }
            case rocisa::DataType::ComplexFloat:
            {
                const auto converted = constVariantCast<std::complex<float>>(value);
                return std::complex<double>{converted.real(), converted.imag()};
            }
            case rocisa::DataType::ComplexDouble:
                return constVariantCast<std::complex<double>>(value);
            default:
                return std::nullopt;
            }
        }

        inline std::span<const std::byte>
            storageSpan(ScalarType type, const void* pointer, size_t elements)
        {
            const Layout layout = Layout::contiguousLastDimensionFastest(Shape{elements});
            return {static_cast<const std::byte*>(pointer), storageBytesForLayout(type, layout)};
        }

        inline Expected<Tensor, DescriptorFailure>
            getSnapshot(ScalarType type, const void* pointer, size_t elements)
        {
            auto layout = Layout::getContiguousLayoutLastDimensionFastest(Shape{elements});
            if(!layout)
                return Unexpected(DescriptorFailure(layout.error()));
            const auto bytes = roc::hostnumerics::detail::getStorageBytesForLayout(type, *layout);
            if(!bytes)
                return Unexpected(DescriptorFailure(bytes.error()));
            if(*bytes != 0 && pointer == nullptr)
            {
                return Unexpected(DescriptorFailure(DescriptorError::NullStorage));
            }
            return Tensor::copyEncodedBackingStorage(
                type, std::move(*layout), {static_cast<const std::byte*>(pointer), *bytes});
        }

        inline std::complex<double> scalarFromStorage(ScalarType type, const void* pointer)
        {
            const Tensor view = Tensor::copyEncodedBackingStorage(
                type,
                Layout::contiguousLastDimensionFastest(Shape{1}),
                storageSpan(type, pointer, 1));
            if(scalarTypeInfo(type).category == ScalarCategory::Complex)
                return view.loadAs<std::complex<double>>({0});
            return {view.loadAs<double>({0}), 0.0};
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
            Tensor                    a;
            Tensor                    b;
            Tensor                    c;
            Tensor                    d;
            IndexSelection            outputSelection = IndexSelection::allLastDimensionFastest();
            std::optional<Tensor>     bias;
            std::optional<MatrixAxis> biasAxis;
            std::optional<Tensor>     biasOutput;
            std::optional<Tensor>     auxiliaryInput;
            std::optional<Tensor>     auxiliaryOutput;
            std::optional<Tensor>     gateResidual;
            std::span<std::byte>      dDestination;
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
            BorrowedConstTensor   a;
            BorrowedConstTensor   b;
            BorrowedConstTensor   c;
            BorrowedMutableTensor d;
            Layout                outputLayout;
            IndexSelection        outputSelection = IndexSelection::allLastDimensionFastest();
            std::optional<BorrowedConstTensor>   bias;
            std::optional<BorrowedMutableTensor> biasOutput;
            MatrixAxis                           biasAxis = MatrixAxis::Row;
            std::optional<BorrowedConstTensor>   auxiliaryInput;
            std::optional<BorrowedMutableTensor> auxiliaryOutput;
            std::optional<BorrowedConstTensor>   gateResidual;
            std::optional<BorrowedConstTensor>   blockScaleA;
            std::optional<BorrowedConstTensor>   blockScaleB;
        };

        // Rebase one affine batch to the smallest byte-aligned storage window.
        // Packed formats may retain a few prefix bits so element zero remains
        // correctly aligned within its first byte.
        inline Expected<std::pair<Layout, ptrdiff_t>, DescriptorFailure>
            rebaseLayoutToByteAlignedStorage(ScalarType type, const Layout& layout)
        {
            const auto bounds = roc::hostnumerics::detail::getElementBounds(layout);
            if(!bounds)
                return Unexpected(DescriptorFailure(bounds.error()));
            if(bounds->second < bounds->first)
                return std::pair{
                    Layout(layout.shape(),
                           std::vector<ptrdiff_t>(layout.strides().begin(), layout.strides().end()),
                           0),
                    ptrdiff_t{0}};
            const ptrdiff_t lowerOffset = bounds->first;
            if(lowerOffset < 0)
            {
                return Unexpected(DescriptorFailure(DescriptorError::BeforeBatchStorage));
            }
            const size_t bits          = scalarTypeInfo(type).storageBits;
            const size_t alignment     = 8 / std::gcd<size_t>(8, bits);
            const size_t alignedOffset = static_cast<size_t>(lowerOffset) / alignment * alignment;
            const auto   byteOffset = getMultiply(alignedOffset / alignment, alignment * bits / 8);
            const auto   rebasedOffset = roc::hostnumerics::detail::LayoutOffset::getOffsetAdd(
                layout.offset(), -static_cast<ptrdiff_t>(alignedOffset));
            if(!byteOffset || !rebasedOffset)
            {
                return Unexpected(DescriptorFailure(DescriptorError::OffsetOverflow));
            }
            return std::pair{
                Layout(layout.shape(),
                       std::vector<ptrdiff_t>(layout.strides().begin(), layout.strides().end()),
                       *rebasedOffset),
                *byteOffset};
        }

        template <typename Borrowed, typename Pointer>
        Expected<Borrowed, DescriptorFailure> makeBorrowedTensor(ScalarType    type,
                                                                 const Layout& layout,
                                                                 Pointer       storage,
                                                                 ptrdiff_t     byteOffset)
        {
            auto rebased = rebaseLayoutToByteAlignedStorage(type, layout);
            if(!rebased)
                return Unexpected(DescriptorFailure(rebased.error()));
            const auto required
                = roc::hostnumerics::detail::getStorageBytesForLayout(type, rebased->first);
            if(!required)
                return Unexpected(DescriptorFailure(required.error()));
            if(*required == 0)
                return Borrowed{std::move(rebased->first), {}};
            if(storage == nullptr)
            {
                return Unexpected(DescriptorFailure(DescriptorError::NullStorage));
            }
            const auto offset = roc::hostnumerics::detail::LayoutOffset::getOffsetAdd(
                byteOffset, rebased->second);
            if(!offset)
            {
                return Unexpected(DescriptorFailure(DescriptorError::OffsetAddOverflow));
            }
            using Byte = std::conditional_t<std::is_const_v<std::remove_pointer_t<Pointer>>,
                                            const std::byte,
                                            std::byte>;
            return Borrowed{std::move(rebased->first),
                            std::span<Byte>(static_cast<Byte*>(storage) + *offset, *required)};
        }

    } // namespace detail

    struct GemmInvocationAdapter::State
    {
        using ActivationFunction = roc::hostnumerics::ActivationFunction;
        using Layout             = roc::hostnumerics::Layout;
        using MathMode           = roc::hostnumerics::MathMode;
        using MatrixAxis         = detail::MatrixAxis;
        using IndexSelection     = roc::hostnumerics::IndexSelection;
        using ScalarType         = roc::hostnumerics::ScalarType;
        using Shape              = roc::hostnumerics::Shape;
        using Tensor             = roc::hostnumerics::Tensor;

        Tensor makeAddendTensor(const Layout& layout, std::span<const std::byte> source) const
        {
            return readC ? Tensor::copyEncodedBackingStorage(typeC, layout, source)
                         : Tensor(typeC,
                                  Layout(layout.shape(),
                                         std::vector<ptrdiff_t>(layout.shape().rank(), 0),
                                         0));
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

        std::optional<TranslationFailure> normalizeProblem(ContractionProblemGemm const& problem,
                                                           IndexSelection outputSelection)
        {
            using detail::failure;
            using roc::hostnumerics::IdentityActivation;
            using roc::hostnumerics::scalarTypeInfo;

            if(problem.boundIndices().size() != 1 || problem.freeIndicesA().size() != 1
               || problem.freeIndicesB().size() != 1 || problem.batchIndices().size() != 1)
            {
                return failure(
                    TranslationFailureCode::UnsupportedContraction,
                    "HostNumerics requires one A free index, one B free index, one batch "
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

            auto mapType    = [](rocisa::DataType type) { return getHostNumericsScalarType(type); };
            auto assignType = [&](ScalarType& destination, rocisa::DataType source) {
                auto mapped = mapType(source);
                if(!mapped)
                    return false;
                destination = *mapped;
                return true;
            };
            if(!assignType(typeA, problem.a().dataType())
               || !assignType(typeB, problem.b().dataType())
               || !assignType(typeC, problem.c().dataType())
               || !assignType(typeD, problem.d().dataType())
               || !assignType(operationAccumulatorType, problem.computeType())
               || !assignType(betaType, problem.betaType())
               || !assignType(alphaType, problem.alphaType()))
                return failure(TranslationFailureCode::UnsupportedDataType,
                               "rocisa data type has no scalar HostNumerics mapping.");
            switch(operationAccumulatorType)
            {
            case ScalarType::Float16:
            case ScalarType::BFloat16:
            case ScalarType::Float32:
            case ScalarType::Float64:
            case ScalarType::Int32:
            case ScalarType::ComplexFloat32:
            case ScalarType::ComplexFloat64:
                break;
            default:
                return failure(TranslationFailureCode::UnsupportedDataType,
                               "Unsupported HostNumerics GEMM accumulator type.");
            }
            computeTypeA = typeA;
            computeTypeB = typeB;
            if((problem.computeInputTypeA() != rocisa::DataType::None
                && !assignType(computeTypeA, problem.computeInputTypeA()))
               || (problem.computeInputTypeB() != rocisa::DataType::None
                   && !assignType(computeTypeB, problem.computeInputTypeB())))
                return failure(TranslationFailureCode::UnsupportedDataType,
                               "rocisa data type has no scalar HostNumerics mapping.");
            auto optionalType
                = [&](std::optional<ScalarType>& destination, rocisa::DataType source) {
                      destination = mapType(source);
                      return destination.has_value();
                  };
            if((problem.useBias() && !optionalType(biasType, problem.bias().dataType()))
               || (problem.useE()
                   && !optionalType(
                       auxiliaryType,
                       problem.tensors()[ContractionProblemGemm::TENSOR::E].dataType()))
               || (problem.useGateResidual()
                   && !optionalType(gateType,
                                    problem.tensors()[ContractionProblemGemm::TENSOR::GATE_RESIDUAL]
                                        .dataType())))
                return failure(TranslationFailureCode::UnsupportedDataType,
                               "rocisa data type has no scalar HostNumerics mapping.");

            useGradient = problem.useGradient();
            useBias     = problem.useBias();
            biasSource  = problem.biasSrc();
            scaleAlphaAxis
                = problem.getParams().factorDim() == 0 ? MatrixAxis::Row : MatrixAxis::Column;
            mathMode = operationAccumulatorType == ScalarType::Float32
                               && problem.f32XdlMathOp() == rocisa::DataType::XFloat32
                           ? MathMode::XFloat32
                           : MathMode::Default;
            preQuantizationScaleA
                = scalarTypeInfo(typeA).storageBits > scalarTypeInfo(computeTypeA).storageBits;
            preQuantizationScaleB
                = scalarTypeInfo(typeB).storageBits > scalarTypeInfo(computeTypeB).storageBits;

            ActivationType concreteActivation = problem.activationType();
            if(concreteActivation == ActivationType::All
               || concreteActivation == ActivationType::Hipblaslt_all)
                concreteActivation = problem.getParams().activationEnum();
            auto mappedActivation = getHostNumericsActivation(concreteActivation, useGradient);
            if(!mappedActivation)
                return failure(TranslationFailureCode::UnsupportedActivation,
                               "Activation has no runtime HostNumerics mapping.");
            activation = std::move(*mappedActivation);

            if((operationAccumulatorType == ScalarType::ComplexFloat32
                || operationAccumulatorType == ScalarType::ComplexFloat64)
               && !std::holds_alternative<IdentityActivation>(activation))
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
            auto validAxes
                = [](const TensorDescriptor& descriptor, size_t row, size_t column, size_t batch) {
                      return row < descriptor.dimensions() && column < descriptor.dimensions()
                             && batch < descriptor.dimensions();
                  };
            if(!validAxes(problem.a(), indexMA, indexKA, batchA)
               || !validAxes(problem.b(), indexKB, indexNB, batchB)
               || !validAxes(problem.c(), indexMC, indexNC, batchC)
               || !validAxes(problem.d(), indexMD, indexND, batchD))
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "TensileLite adapter dimension is outside its descriptor.");
            m       = problem.freeSizeA(0);
            n       = problem.freeSizeB(0);
            k       = problem.boundSize(0);
            batches = problem.batchSize(0);

            if(batches != 0 && m != 0 && n != 0)
            {
                auto contiguous = Layout::getContiguousLayoutLastDimensionFastest(Shape{m, n});
                if(!contiguous)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   contiguous.error().message());
                const auto bytes = roc::hostnumerics::detail::getStorageBytesForLayout(
                    operationAccumulatorType, *contiguous);
                if(!bytes)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(bytes.error())));
            }

            mxBlockA = problem.mxBlockA();
            mxBlockB = problem.mxBlockB();
            if(mxBlockA > 0)
            {
                const auto mappedA = getHostNumericsMxScaleType(problem.mxTypeA());
                const auto mappedB = getHostNumericsMxScaleType(problem.mxTypeB());
                if(!mappedA || !mappedB)
                    return failure(TranslationFailureCode::UnsupportedDataType,
                                   "rocisa data type has no MX scale HostNumerics mapping.");
                mxScaleTypeA = *mappedA;
                mxScaleTypeB = *mappedB;
            }

            for(auto const& operation : problem.aOps())
                aConjugate |= operation.type == TensorOp::Type::ComplexConjugate;
            for(auto const& operation : problem.bOps())
                bConjugate |= operation.type == TensorOp::Type::ComplexConjugate;

            const bool requiresCompleteD
                = problem.outputAmaxD()
                  || (useGradient && useBias && biasSource == ContractionProblemGemm::D);
            std::vector<size_t> globalOrder;
            try
            {
                globalOrder = outputSelection.indexOrder(problem.d().dimensions());
            }
            catch(std::runtime_error const& error)
            {
                return failure(TranslationFailureCode::InvalidDescriptor, error.what());
            }
            const IndexSelection globalSelection
                = requiresCompleteD ? IndexSelection::all(globalOrder) : std::move(outputSelection);
            const auto logicalCount
                = roc::hostnumerics::detail::ShapeAccess::getElementCount(problem.d().sizes());
            if(!logicalCount)
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "Tensor shape element count exceeds size_t.");
            if(auto error
               = roc::hostnumerics::detail::indexSelectionError(globalSelection, *logicalCount))
                return failure(TranslationFailureCode::InvalidDescriptor,
                               std::string(message(*error)));
            selectAllOutputs = globalSelection.selectsAll();
            selectedByBatch.clear();
            if(!selectAllOutputs)
            {
                selectedByBatch.resize(batches);
                const Shape outputShape(problem.d().sizes());
                for(const size_t logicalIndex : globalSelection.indices(*logicalCount))
                {
                    const std::vector<size_t> coordinate
                        = outputShape.coordinates(logicalIndex, globalOrder);
                    const size_t selectedBatch = coordinate[batchD];
                    const size_t row           = coordinate[indexMD];
                    const size_t column        = coordinate[indexND];
                    if(selectedBatch >= selectedByBatch.size())
                        return failure(
                            TranslationFailureCode::InvalidDescriptor,
                            "Selected output batch exceeds the contraction batch count.");
                    selectedByBatch[selectedBatch].push_back(row * n + column);
                }
            }
            return std::nullopt;
        }

        std::optional<TranslationFailure> bindInputs(ContractionProblemGemm const& problem,
                                                     ContractionInputs const&      inputs)
        {
            using detail::failure;

            if(batches == 0 || m == 0 || n == 0)
                return std::nullopt;

            if(!detail::supportsScalarValue(problem.alphaType())
               || !detail::supportsScalarValue(problem.betaType())
               || (!inputs.activationArgs.empty()
                   && !detail::supportsScalarValue(problem.computeType())))
                return failure(TranslationFailureCode::UnsupportedDataType,
                               "Unsupported TensileLite scalar value type.");

            const auto mappedAlpha = detail::scalarValue(problem.alphaType(), inputs.alpha);
            const auto mappedBeta  = detail::scalarValue(problem.betaType(), inputs.beta);
            if(!mappedAlpha || !mappedBeta)
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "Scalar value cannot be converted to its declared type.");
            alpha = *mappedAlpha;
            beta  = *mappedBeta;

            const bool computeProduct = k != 0 && alpha != std::complex<double>(0.0, 0.0);
            readA                     = computeProduct
                    || (useGradient && useBias && biasSource == ContractionProblemGemm::A);
            readB = computeProduct
                    || (useGradient && useBias && biasSource == ContractionProblemGemm::B);
            readC                   = beta != std::complex<double>(0.0, 0.0);
            const bool aHasElements = m != 0 && k != 0;
            const bool bHasElements = k != 0 && n != 0;

            if(problem.useBias() && inputs.bias == nullptr && inputs.batchBias == nullptr)
                return failure(TranslationFailureCode::MissingInput, "Bias input is missing.");
            if(computeProduct && problem.useScaleAlphaVec() && inputs.scaleAlphaVec == nullptr)
            {
                return failure(TranslationFailureCode::MissingInput,
                               "Scale-alpha vector input is missing.");
            }
            if(computeProduct && scaleABMode != detail::ScaleABMode::None
               && (inputs.scaleA == nullptr || inputs.scaleB == nullptr))
            {
                return failure(TranslationFailureCode::MissingInput,
                               "A and B scale inputs are required together.");
            }
            if(computeProduct && problem.mxBlockA() > 0
               && (inputs.mxsa == nullptr || inputs.mxsb == nullptr))
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
            if((readA && aHasElements && inputs.a == nullptr && inputs.batchA == nullptr)
               || (readB && bHasElements && inputs.b == nullptr && inputs.batchB == nullptr)
               || (readC && inputs.c == nullptr && inputs.batchC == nullptr)
               || (inputs.d == nullptr && inputs.batchD == nullptr))
            {
                return failure(TranslationFailureCode::MissingInput,
                               "A, B, C, or D input is missing.");
            }

            if(problem.useScaleCD())
            {
                scaleC      = detail::scalarFromStorage(betaType, inputs.scaleC);
                outputScale = detail::scalarFromStorage(betaType, inputs.scaleD);
            }

            for(size_t parameter = 0; parameter < std::min<size_t>(2, inputs.activationArgs.size());
                ++parameter)
            {
                auto mapped
                    = detail::scalarValue(problem.computeType(), inputs.activationArgs[parameter]);
                if(!mapped)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   "Scalar value cannot be converted to its declared type.");
                (parameter == 0 ? activationParameter0 : activationParameter1) = mapped->real();
            }

            for(size_t batch = 0; batch < batches; ++batch)
            {
                if((readA && aHasElements && inputs.batchA != nullptr
                    && inputs.batchA[batch] == nullptr)
                   || (readB && bHasElements && inputs.batchB != nullptr
                       && inputs.batchB[batch] == nullptr)
                   || (readC && inputs.batchC != nullptr && inputs.batchC[batch] == nullptr)
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

            if(!std::in_range<ptrdiff_t>(inputs.batchOffsetA))
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "TensileLite adapter offset exceeds ptrdiff_t.");
            const ptrdiff_t batchOffsetA = static_cast<ptrdiff_t>(inputs.batchOffsetA);
            if(!std::in_range<ptrdiff_t>(inputs.batchOffsetB))
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "TensileLite adapter offset exceeds ptrdiff_t.");
            const ptrdiff_t batchOffsetB = static_cast<ptrdiff_t>(inputs.batchOffsetB);
            if(!std::in_range<ptrdiff_t>(inputs.batchOffsetC))
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "TensileLite adapter offset exceeds ptrdiff_t.");
            const ptrdiff_t batchOffsetC = static_cast<ptrdiff_t>(inputs.batchOffsetC);
            if(!std::in_range<ptrdiff_t>(inputs.batchOffsetD))
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "TensileLite adapter offset exceeds ptrdiff_t.");
            const ptrdiff_t batchOffsetD = static_cast<ptrdiff_t>(inputs.batchOffsetD);

            TensorDescriptor const* auxiliaryDescriptor = nullptr;
            if(problem.useE())
                auxiliaryDescriptor = &problem.tensors()[ContractionProblemGemm::TENSOR::E];

            TensorDescriptor const* gateDescriptor = nullptr;
            if(problem.useGateResidual())
                gateDescriptor = &problem.tensors()[ContractionProblemGemm::TENSOR::GATE_RESIDUAL];

            if(problem.outputAmaxD())
            {
                auto const& descriptor = problem.tensors()[ContractionProblemGemm::TENSOR::AMAXD];
                ScalarType  amaxType;
                const auto  mappedType = getHostNumericsScalarType(descriptor.dataType());
                if(!mappedType)
                    return failure(TranslationFailureCode::UnsupportedDataType,
                                   "rocisa data type has no scalar HostNumerics mapping.");
                amaxType              = *mappedType;
                auto descriptorLayout = getHostnumericsLayout(descriptor);
                if(!descriptorLayout)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(descriptorLayout.error())));
                const auto bytes = roc::hostnumerics::detail::getStorageBytesForLayout(
                    amaxType, *descriptorLayout);
                if(!bytes)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(bytes.error())));
                const Layout scalarLayout = Layout::contiguousLastDimensionFastest(Shape{1});
                const size_t required     = storageBytesForLayout(amaxType, scalarLayout);
                if(*bytes < required)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   "Encoded Tensor storage is too small for its layout.");
                std::span<std::byte> storage(static_cast<std::byte*>(inputs.amaxD), *bytes);
                amax = Tensor::copyEncodedBackingStorage(amaxType, scalarLayout, storage);
                amaxDestination = storage;
            }

            const size_t scaleAlphaLength = scaleAlphaAxis == MatrixAxis::Row ? m : n;
            if(computeProduct && problem.useScaleAlphaVec())
            {
                auto scaleAlphaResult
                    = detail::getSnapshot(alphaType, inputs.scaleAlphaVec, scaleAlphaLength);
                if(!scaleAlphaResult)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(scaleAlphaResult.error())));
                scaleAlpha = std::move(*scaleAlphaResult);
            }
            if(computeProduct && scaleABMode != detail::ScaleABMode::None)
            {
                const bool vector = scaleABMode == detail::ScaleABMode::Vector;
                auto scaleAResult = detail::getSnapshot(alphaType, inputs.scaleA, vector ? m : 1);
                if(!scaleAResult)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(scaleAResult.error())));
                scaleA            = std::move(*scaleAResult);
                auto scaleBResult = detail::getSnapshot(alphaType, inputs.scaleB, vector ? n : 1);
                if(!scaleBResult)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(scaleBResult.error())));
                scaleB = std::move(*scaleBResult);
            }

            batchPlans.reserve(batches);
            for(size_t batch = 0; batch < batches; ++batch)
            {
                auto candidateA = detail::matrixLayout(problem.a(),
                                                       m,
                                                       k,
                                                       indexMA,
                                                       indexKA,
                                                       batchA,
                                                       batch,
                                                       inputs.batchA != nullptr,
                                                       false,
                                                       problem.boundIndices()[0].aMirror);
                if(!candidateA)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(candidateA.error())));
                const Layout& layoutA    = *candidateA;
                auto          candidateB = detail::matrixLayout(problem.b(),
                                                       k,
                                                       n,
                                                       indexKB,
                                                       indexNB,
                                                       batchB,
                                                       batch,
                                                       inputs.batchB != nullptr,
                                                       problem.boundIndices()[0].bMirror,
                                                       false);
                if(!candidateB)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(candidateB.error())));
                const Layout& layoutB    = *candidateB;
                auto          candidateC = detail::matrixLayout(problem.c(),
                                                       m,
                                                       n,
                                                       indexMC,
                                                       indexNC,
                                                       batchC,
                                                       batch,
                                                       inputs.batchC != nullptr,
                                                       false,
                                                       false);
                if(!candidateC)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(candidateC.error())));
                const Layout& layoutC    = *candidateC;
                auto          candidateD = detail::matrixLayout(problem.d(),
                                                       m,
                                                       n,
                                                       indexMD,
                                                       indexND,
                                                       batchD,
                                                       batch,
                                                       inputs.batchD != nullptr,
                                                       false,
                                                       false);
                if(!candidateD)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(candidateD.error())));
                const Layout& layoutD  = *candidateD;
                const void*   currentA = inputs.batchA == nullptr ? inputs.a : inputs.batchA[batch];
                const void*   currentB = inputs.batchB == nullptr ? inputs.b : inputs.batchB[batch];
                const void*   currentC = inputs.batchC == nullptr ? inputs.c : inputs.batchC[batch];
                void*         currentD = inputs.batchD == nullptr ? inputs.d : inputs.batchD[batch];

                auto borrowedA = readA ? detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                                     typeA, layoutA, currentA, batchOffsetA)
                                       : detail::Expected<detail::BorrowedConstTensor,
                                                          detail::DescriptorFailure>(
                                           detail::BorrowedConstTensor{layoutA, {}});
                if(!borrowedA)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(borrowedA.error())));
                auto borrowedB = readB ? detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                                     typeB, layoutB, currentB, batchOffsetB)
                                       : detail::Expected<detail::BorrowedConstTensor,
                                                          detail::DescriptorFailure>(
                                           detail::BorrowedConstTensor{layoutB, {}});
                if(!borrowedB)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(borrowedB.error())));
                auto borrowedC = readC ? detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                                     typeC, layoutC, currentC, batchOffsetC)
                                       : detail::Expected<detail::BorrowedConstTensor,
                                                          detail::DescriptorFailure>(
                                           detail::BorrowedConstTensor{layoutC, {}});
                if(!borrowedC)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(borrowedC.error())));
                auto borrowedD = detail::makeBorrowedTensor<detail::BorrowedMutableTensor>(
                    typeD, layoutD, currentD, batchOffsetD);
                if(!borrowedD)
                    return failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(borrowedD.error())));
                detail::BatchPlan plan{
                    .a               = std::move(*borrowedA),
                    .b               = std::move(*borrowedB),
                    .c               = std::move(*borrowedC),
                    .d               = std::move(*borrowedD),
                    .outputLayout    = layoutD,
                    .outputSelection = selectAllOutputs
                                           ? IndexSelection::allLastDimensionFastest()
                                           : IndexSelection::explicitIndicesLastDimensionFastest(
                                               selectedByBatch[batch]),
                };

                if(problem.useBias())
                {
                    ptrdiff_t    runtimeBiasOffset = 0;
                    const size_t runtimeBiasLength = problem.bias().sizes()[0];
                    if(inputs.batchBias == nullptr && problem.bias().dimensions() > 2)
                    {
                        auto offset = detail::getMultiply(batch, problem.bias().strides()[2]);
                        if(!offset)
                            return failure(TranslationFailureCode::InvalidDescriptor,
                                           "TensileLite adapter offset multiplication overflow.");
                        runtimeBiasOffset = *offset;
                    }
                    const Layout biasLayout(Shape{runtimeBiasLength}, {1}, runtimeBiasOffset);
                    const void*  currentBias
                        = inputs.batchBias == nullptr ? inputs.bias : inputs.batchBias[batch];
                    plan.biasAxis = detail::inferBiasAxis(
                        runtimeBiasLength, m, n, problem.getParams().factorDim());
                    if(useGradient)
                    {
                        void* currentBiasOutput = inputs.batchBias == nullptr
                                                      ? const_cast<void*>(inputs.bias)
                                                      : const_cast<void*>(inputs.batchBias[batch]);
                        auto  biasOutputResult
                            = detail::makeBorrowedTensor<detail::BorrowedMutableTensor>(
                                *biasType, biasLayout, currentBiasOutput, 0);
                        if(!biasOutputResult)
                            return failure(TranslationFailureCode::InvalidDescriptor,
                                           std::string(message(biasOutputResult.error())));
                        plan.biasOutput = std::move(*biasOutputResult);
                    }
                    else
                    {
                        auto biasResult = detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                            *biasType, biasLayout, currentBias, 0);
                        if(!biasResult)
                            return failure(TranslationFailureCode::InvalidDescriptor,
                                           std::string(message(biasResult.error())));
                        plan.bias = std::move(*biasResult);
                    }
                }

                if(problem.useE())
                {
                    auto layoutE = detail::matrixLayout(*auxiliaryDescriptor,
                                                        m,
                                                        n,
                                                        indexMD,
                                                        indexND,
                                                        batchD,
                                                        batch,
                                                        false,
                                                        false,
                                                        false);
                    if(!layoutE)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(layoutE.error())));
                    if(useGradient)
                    {
                        auto auxiliaryInputResult
                            = detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                                *auxiliaryType, *layoutE, static_cast<const void*>(inputs.e), 0);
                        if(!auxiliaryInputResult)
                            return failure(TranslationFailureCode::InvalidDescriptor,
                                           std::string(message(auxiliaryInputResult.error())));
                        plan.auxiliaryInput = std::move(*auxiliaryInputResult);
                    }
                    else
                    {
                        auto auxiliaryOutputResult
                            = detail::makeBorrowedTensor<detail::BorrowedMutableTensor>(
                                *auxiliaryType, *layoutE, inputs.e, 0);
                        if(!auxiliaryOutputResult)
                            return failure(TranslationFailureCode::InvalidDescriptor,
                                           std::string(message(auxiliaryOutputResult.error())));
                        plan.auxiliaryOutput = std::move(*auxiliaryOutputResult);
                    }
                }

                if(problem.useGateResidual())
                {
                    auto layoutGate = detail::matrixLayout(*gateDescriptor,
                                                           m,
                                                           n,
                                                           indexMD,
                                                           indexND,
                                                           batchD,
                                                           batch,
                                                           inputs.batchGateResidual != nullptr,
                                                           false,
                                                           false);
                    if(!layoutGate)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(layoutGate.error())));
                    const void* currentGate = inputs.batchGateResidual == nullptr
                                                  ? inputs.gateResidual
                                                  : inputs.batchGateResidual[batch];
                    auto        gateResidualResult
                        = detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                            *gateType, *layoutGate, currentGate, 0);
                    if(!gateResidualResult)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(gateResidualResult.error())));
                    plan.gateResidual = std::move(*gateResidualResult);
                }

                if(computeProduct && mxBlockA > 0)
                {
                    const size_t blockCountA
                        = k / mxBlockA + static_cast<size_t>(k % mxBlockA != 0);
                    const size_t blockCountB
                        = k / mxBlockB + static_cast<size_t>(k % mxBlockB != 0);
                    auto layoutScaleA = detail::matrixLayout(problem.mxsa(),
                                                             m,
                                                             blockCountA,
                                                             indexMA,
                                                             indexKA,
                                                             batchA,
                                                             batch,
                                                             false,
                                                             false,
                                                             problem.boundIndices()[0].aMirror);
                    if(!layoutScaleA)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(layoutScaleA.error())));
                    auto layoutScaleB = detail::matrixLayout(problem.mxsb(),
                                                             n,
                                                             blockCountB,
                                                             indexNB,
                                                             indexKB,
                                                             batchB,
                                                             batch,
                                                             false,
                                                             false,
                                                             problem.boundIndices()[0].bMirror);
                    if(!layoutScaleB)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(layoutScaleB.error())));
                    auto blockScaleAResult
                        = detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                            mxScaleTypeA, *layoutScaleA, inputs.mxsa, 0);
                    if(!blockScaleAResult)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(blockScaleAResult.error())));
                    plan.blockScaleA = std::move(*blockScaleAResult);
                    auto blockScaleBResult
                        = detail::makeBorrowedTensor<detail::BorrowedConstTensor>(
                            mxScaleTypeB, *layoutScaleB, inputs.mxsb, 0);
                    if(!blockScaleBResult)
                        return failure(TranslationFailureCode::InvalidDescriptor,
                                       std::string(message(blockScaleBResult.error())));
                    plan.blockScaleB = std::move(*blockScaleBResult);
                }

                batchPlans.push_back(std::move(plan));
            }

            return std::nullopt;
        }

        detail::BatchInputs materializeBatch(size_t batch, bool referenceOutputs) const
        {
            const detail::BatchPlan& plan = batchPlans[batch];
            Tensor                   currentA
                = readA ? Tensor::copyEncodedBackingStorage(typeA, plan.a.layout, plan.a.storage)
                        : Tensor(typeA, Layout(Shape{m, k}, {0, 0}, 0));
            Tensor currentB
                = readB ? Tensor::copyEncodedBackingStorage(typeB, plan.b.layout, plan.b.storage)
                        : Tensor(typeB, Layout(Shape{k, n}, {0, 0}, 0));
            Tensor currentC = makeAddendTensor(plan.c.layout, plan.c.storage);
            Tensor currentD
                = referenceOutputs ? makeOutputTensor(plan.d.layout) : Tensor(typeD, Shape{0});

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
            if(referenceOutputs && plan.biasOutput)
            {
                runtimeBiasOutput = makeOutputTensorForType(*biasType, plan.biasOutput->layout);
                runtimeBiasOutputDestination = plan.biasOutput->storage;
            }

            std::optional<Tensor> auxiliaryInput;
            std::optional<Tensor> auxiliaryOutput;
            if(plan.auxiliaryInput)
                auxiliaryInput = Tensor::copyEncodedBackingStorage(
                    *auxiliaryType, plan.auxiliaryInput->layout, plan.auxiliaryInput->storage);
            if(referenceOutputs && plan.auxiliaryOutput)
                auxiliaryOutput
                    = makeOutputTensorForType(*auxiliaryType, plan.auxiliaryOutput->layout);

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

        std::complex<double> alpha       = {1.0, 0.0};
        std::complex<double> beta        = {0.0, 0.0};
        std::complex<double> scaleC      = {1.0, 0.0};
        std::complex<double> outputScale = {1.0, 0.0};
        ActivationFunction   activation;
        double               activationParameter0 = 0.0;
        double               activationParameter1 = 0.0;

        bool useGradient           = false;
        bool useBias               = false;
        bool preQuantizationScaleA = false;
        bool preQuantizationScaleB = false;
        bool aConjugate            = false;
        bool bConjugate            = false;

        ContractionProblemGemm::TENSOR biasSource     = ContractionProblemGemm::D;
        MatrixAxis                     scaleAlphaAxis = MatrixAxis::Row;
        MathMode                       mathMode       = MathMode::Default;
        detail::ScaleABMode            scaleABMode    = detail::ScaleABMode::None;

        size_t m       = 0;
        size_t n       = 0;
        size_t k       = 0;
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

        std::vector<std::vector<size_t>> selectedByBatch;
        bool                             selectAllOutputs = true;

        std::optional<Tensor>               scaleAlpha;
        std::optional<Tensor>               scaleA;
        std::optional<Tensor>               scaleB;
        std::optional<Tensor>               amax;
        std::optional<std::span<std::byte>> amaxDestination;
        bool                                readA = true;
        bool                                readB = true;
        bool                                readC = true;
        std::optional<ScalarType>           biasType;
        std::optional<ScalarType>           auxiliaryType;
        std::optional<ScalarType>           gateType;
        std::vector<detail::BatchPlan>      batchPlans;
    };

    GemmInvocationAdapter::GemmInvocationAdapter(std::unique_ptr<const State> state)
        : m_state(std::move(state))
    {
    }

    GemmInvocationAdapter::~GemmInvocationAdapter() = default;

    GemmInvocationAdapter::GemmInvocationAdapter(GemmInvocationAdapter&&) noexcept = default;

    GemmInvocationAdapter&
        GemmInvocationAdapter::operator=(GemmInvocationAdapter&&) noexcept = default;

    void TranslatedGemmBatch::runGemm(roc::hostnumerics::GemmBackend backend) const
    {
        using roc::hostnumerics::matmulInto;
        using roc::hostnumerics::Tensor;

        const auto isZero = [](const Tensor& value) {
            return value.item<std::complex<double>>() == std::complex<double>(0.0, 0.0);
        };
        if(!isZero(alpha) && a.shape()[1] != 0)
            matmulInto(a, b, d, options, outputSelection, backend);
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
                const auto selected = copyBack.selection.indices(copyBack.source.elementCount());
                copyBack.source.copySelectedElementsToEncodedStorage(
                    copyBack.destination,
                    selected,
                    copyBack.selection.indexOrder(copyBack.source.shape().rank()));
            }
        }
    }

    size_t GemmInvocationAdapter::batchCount() const
    {
        return m_state->batchPlans.size();
    }

    void GemmInvocationAdapter::execute(roc::hostnumerics::GemmBackend backend) const
    {
        for(size_t batch = 0; batch < batchCount(); ++batch)
        {
            auto translation = translateBatch(batch);
            if(std::holds_alternative<TranslationFailure>(translation))
            {
                const auto& failure = std::get<TranslationFailure>(translation);
                throw std::runtime_error("TensileLite HostNumerics translation failed: "
                                         + failure.reason);
            }

            TranslatedGemmBatch translated = std::move(std::get<TranslatedGemmBatch>(translation));
            translated.runGemm(backend);
            translated.runPostGemmOperationsAndCopyOutputs();
        }
    }

    roc::hostnumerics::Layout GemmInvocationAdapter::outputLayout(size_t batch) const
    {
        if(batch >= m_state->batchPlans.size())
            throw std::runtime_error("Requested reference batch index is out of range.");
        return m_state->batchPlans[batch].outputLayout;
    }

    roc::hostnumerics::FreivaldsPreparation GemmInvocationAdapter::prepareFreivalds(
        size_t                                          batch,
        const roc::hostnumerics::FreivaldsProbeOptions& probes,
        const roc::hostnumerics::FreivaldsLimits&       limits) const
    {
        using namespace roc::hostnumerics;
        if(batch >= m_state->batchPlans.size())
            return FreivaldsPreparationFailure{VerificationStatus::Unsupported,
                                               "Requested reference batch index is out of range."};
        const auto& plan = m_state->batchPlans[batch];
        if(m_state->useGradient || m_state->amax || plan.auxiliaryOutput)
            return FreivaldsPreparationFailure{
                VerificationStatus::Unsupported,
                "TensileLite Freivalds does not yet verify auxiliary, amax or gradient outputs."};
        auto translated = translateBatch(batch, false);
        if(const auto* failure = std::get_if<TranslationFailure>(&translated))
            return FreivaldsPreparationFailure{VerificationStatus::Unsupported, failure->reason};
        const auto& numerical = std::get<TranslatedGemmBatch>(translated);
        return roc::hostnumerics::prepareFreivalds(numerical.a,
                                                   numerical.b,
                                                   numerical.options,
                                                   numerical.epilogue->options,
                                                   m_state->typeD,
                                                   probes,
                                                   limits);
    }

    std::variant<TranslatedGemmBatch, TranslationFailure>
        GemmInvocationAdapter::translateBatch(size_t batch) const
    {
        return translateBatch(batch, true);
    }

    std::variant<TranslatedGemmBatch, TranslationFailure>
        GemmInvocationAdapter::translateBatch(size_t batch, bool referenceOutputs) const
    {
        using detail::failure;
        using roc::hostnumerics::ActivationApplication;
        using roc::hostnumerics::IndexSelection;
        using roc::hostnumerics::OutputConversion;
        using roc::hostnumerics::ScalarType;
        using roc::hostnumerics::Shape;
        using roc::hostnumerics::Tensor;

        const ScalarType accumulatorType = m_state->operationAccumulatorType;

        if(batch >= m_state->batchPlans.size())
        {
            return failure(TranslationFailureCode::InvalidBatchIndex,
                           "Requested reference batch index is out of range.");
        }

        // Scalar snapshots are recoverable descriptor errors, so check their
        // conversion before entering the throwing Tensor convenience API.
        const bool complex = accumulatorType == ScalarType::ComplexFloat32
                             || accumulatorType == ScalarType::ComplexFloat64;
        for(const auto value :
            {m_state->alpha, m_state->beta, m_state->scaleC, m_state->outputScale})
        {
            if(!complex && value.imag() != 0.0)
                return failure(TranslationFailureCode::InvalidDescriptor,
                               "A complex value with a nonzero imaginary component cannot be "
                               "converted to real.");
            if(accumulatorType == ScalarType::Int32 && !std::isfinite(value.real()))
            {
                return failure(TranslationFailureCode::InvalidDescriptor,
                               std::isnan(value.real())
                                   ? "NaN cannot be converted to an integer destination."
                                   : "Infinity cannot be converted to an integer destination.");
            }
        }

        const auto& plan   = m_state->batchPlans[batch];
        const auto  source = m_state->materializeBatch(batch, referenceOutputs);

        Tensor productOutput = source.d;
        Tensor gemmOutput(accumulatorType,
                          referenceOutputs ? Shape{m_state->m, m_state->n} : Shape{0});

        TranslatedGemmBatch translated(source.a, source.b, source.c, gemmOutput, accumulatorType);
        translated.copyBacks.push_back(
            {source.dDestination, productOutput, source.outputSelection});
        if(m_state->amax && m_state->amaxDestination)
            translated.copyBacks.push_back({*m_state->amaxDestination,
                                            *m_state->amax,
                                            IndexSelection::allLastDimensionFastest()});
        if(source.biasOutput && source.biasOutputDestination)
            translated.copyBacks.push_back({*source.biasOutputDestination,
                                            *source.biasOutput,
                                            IndexSelection::allLastDimensionFastest()});
        if(source.auxiliaryOutput && source.auxiliaryOutputDestination)
            translated.copyBacks.push_back({*source.auxiliaryOutputDestination,
                                            *source.auxiliaryOutput,
                                            source.outputSelection});
        auto& request        = translated.matmulOptions();
        request.computeTypeA = m_state->computeTypeA != m_state->typeA
                                   ? std::optional<ScalarType>(m_state->computeTypeA)
                                   : std::nullopt;
        request.computeTypeB = m_state->computeTypeB != m_state->typeB
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
        translated.alpha  = Tensor::scalar(accumulatorType, m_state->alpha);
        translated.beta   = Tensor::scalar(accumulatorType, m_state->beta);
        translated.scaleC = Tensor::scalar(accumulatorType, m_state->scaleC);
        if(m_state->scaleAlpha)
            translated.scaleAlpha
                = detail::broadcastVectorAsMatrix(*m_state->scaleAlpha, m_state->scaleAlphaAxis);
        if(!m_state->preQuantizationScaleA)
            translated.scaleA = m_state->scaleA
                                    ? std::optional<Tensor>(m_state->scaleA->expandDims(1))
                                    : std::nullopt;
        if(!m_state->preQuantizationScaleB)
            translated.scaleB = m_state->scaleB
                                    ? std::optional<Tensor>(m_state->scaleB->expandDims(0))
                                    : std::nullopt;
        request.mathMode           = m_state->mathMode;
        translated.outputSelection = source.outputSelection;

        {
            TranslatedGemmBatch::BoundEpilogue epilogue(gemmOutput, productOutput, accumulatorType);
            if(!m_state->useGradient)
                epilogue.options.bias = source.bias;
            if(m_state->alpha != std::complex<double>(0.0, 0.0) && translated.a.shape()[1] != 0)
            {
                if(translated.scaleA)
                    epilogue.options.inputScaleFactors.push_back(*translated.scaleA);
                if(translated.scaleB)
                    epilogue.options.inputScaleFactors.push_back(*translated.scaleB);
                if(translated.scaleAlpha)
                    epilogue.options.inputScaleFactors.push_back(*translated.scaleAlpha);
                epilogue.options.inputScale = translated.alpha;
            }
            if(m_state->beta != std::complex<double>(0.0, 0.0))
            {
                epilogue.options.addend      = translated.c;
                epilogue.options.addendScale = multiply(translated.beta,
                                                        translated.scaleC,
                                                        accumulatorType,
                                                        accumulatorType,
                                                        std::nullopt);
            }
            epilogue.options.activation = m_state->activation;
            std::visit(
                [&](auto& function) {
                    using Function = std::remove_cvref_t<decltype(function)>;
                    if constexpr(std::is_same_v<Function, roc::hostnumerics::ClippedReluActivation>)
                    {
                        function.lower = m_state->activationParameter0;
                        function.upper = m_state->activationParameter1;
                    }
                    else if constexpr(std::is_same_v<Function,
                                                     roc::hostnumerics::GeluScalingActivation>)
                        function.scale = m_state->activationParameter0;
                    else if constexpr(std::is_same_v<Function,
                                                     roc::hostnumerics::LeakyReluActivation>)
                        function.negativeSlope = m_state->activationParameter0;
                    else if constexpr(std::is_same_v<Function, roc::hostnumerics::TanhActivation>)
                    {
                        function.inputScale  = m_state->activationParameter0;
                        function.outputScale = m_state->activationParameter1;
                    }
                    else if constexpr(std::is_same_v<Function, roc::hostnumerics::SwishActivation>)
                        function.beta = m_state->activationParameter0;
                    else if constexpr(std::is_same_v<Function, roc::hostnumerics::ClampActivation>)
                    {
                        function.minimum = m_state->activationParameter0;
                        function.maximum = m_state->activationParameter1;
                    }
                },
                epilogue.options.activation);
            epilogue.options.outputScale = Tensor::scalar(accumulatorType, m_state->outputScale);
            if(m_state->typeD == ScalarType::Int8)
                epilogue.options.outputConversion = OutputConversion::SaturatingInt8;
            epilogue.options.outputSelection = translated.outputSelection;

            if(m_state->useGradient && m_state->useBias
               && m_state->biasSource == ContractionProblemGemm::D)
            {
                epilogue.outputs.rawOutput.emplace(accumulatorType, Shape{m_state->m, m_state->n});
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

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const&     problem,
                                ContractionInputs const&          inputs,
                                roc::hostnumerics::IndexSelection outputSelection)
    {
        auto state = std::make_unique<GemmInvocationAdapter::State>();
        if(auto normalizationFailure = state->normalizeProblem(problem, std::move(outputSelection)))
            return std::move(*normalizationFailure);
        if(auto bindingFailure = state->bindInputs(problem, inputs))
            return std::move(*bindingFailure);
        return GemmInvocationAdapter(std::move(state));
    }

    std::variant<GemmInvocationAdapter, TranslationFailure>
        translateGemmInvocation(ContractionProblemGemm const& problem,
                                ContractionInputs const&      inputs,
                                size_t                        elementsToValidate)
    {
        auto selection = getReferenceIndexSelection(problem.d(), elementsToValidate);
        if(!selection)
            return detail::failure(TranslationFailureCode::InvalidDescriptor,
                                   std::string(message(selection.error())));
        return translateGemmInvocation(problem, inputs, std::move(*selection));
    }
} // namespace TensileLite::Client::hostnumerics
