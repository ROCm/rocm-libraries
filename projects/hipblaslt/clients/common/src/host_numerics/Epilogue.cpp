// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_numerics/Epilogue.hpp>
#include <hipblaslt/host_numerics/Types.hpp>

#include <cstddef>
#include <span>
#include <stdexcept>

namespace hipblaslt::host_numerics
{
    using namespace ::roc::host_numerics;

    namespace
    {
        Tensor scalarValue(const void* pointer, ScalarType type)
        {
            const size_t storageBytes = (scalarTypeInfo(type).storageBits + 7U) / 8U;
            return Tensor::copyEncodedBackingStorage(
                type,
                Layout::contiguousLastDimensionFastest(Shape{}),
                std::span<const std::byte>(static_cast<const std::byte*>(pointer), storageBytes));
        }

        ActivationFunction activationFunction(Activation activation,
                                              double     parameter0,
                                              double     parameter1)
        {
            switch(activation)
            {
            case Activation::None:
                return IdentityActivation{};
            case Activation::Absolute:
                return AbsoluteActivation{};
            case Activation::ClippedRelu:
                return ClippedReluActivation{parameter0, parameter1};
            case Activation::Relu:
                return ReluActivation{};
            case Activation::Gelu:
                return GeluActivation{};
            case Activation::GeluDerivative:
                return GeluDerivativeActivation{};
            case Activation::GeluScaling:
                return GeluScalingActivation{parameter0};
            case Activation::LeakyRelu:
                return LeakyReluActivation{parameter0};
            case Activation::ReluDerivative:
                return ReluDerivativeActivation{};
            case Activation::Sigmoid:
                return SigmoidActivation{};
            case Activation::Tanh:
                return TanhActivation{parameter0, parameter1};
            case Activation::Silu:
                return SiluActivation{};
            case Activation::Swish:
                return SwishActivation{parameter0};
            case Activation::Clamp:
                return ClampActivation{parameter0, parameter1};
            }
            throw std::invalid_argument("Unsupported hipBLASLt epilogue activation.");
        }
    } // namespace

    void referenceEpilogue(const EpilogueArguments& arguments)
    {
        if(arguments.rows < 0 || arguments.columns < 0 || arguments.leadingDimension < 0)
            throw std::invalid_argument("hipBLASLt epilogue dimensions must be nonnegative.");

        const size_t rows             = static_cast<size_t>(arguments.rows);
        const size_t columns          = static_cast<size_t>(arguments.columns);
        const size_t leadingDimension = static_cast<size_t>(arguments.leadingDimension);
        if(rows != 0 && leadingDimension < rows)
            throw std::invalid_argument(
                "hipBLASLt epilogue leading dimension is smaller than its row count.");

        const ScalarType computeType = scalarType(arguments.computeType);
        const ScalarType outputType  = scalarType(arguments.outputType);
        const Layout     matrixLayout(Shape{rows, columns},
                                      {1, static_cast<ptrdiff_t>(leadingDimension)});

        Tensor input = copyTensorFromEncodedStorage(arguments.input, computeType, matrixLayout);
        EpilogueOutputs outputs{
            .output = copyTensorFromEncodedStorage(arguments.output, outputType, matrixLayout)};
        EpilogueOptions options(computeType);

        if(arguments.rawOutput != nullptr)
            outputs.rawOutput
                = copyTensorFromEncodedStorage(arguments.rawOutput, computeType, matrixLayout);

        if(arguments.auxiliary != nullptr)
        {
            const ScalarType auxiliaryType = scalarType(arguments.auxiliaryType);
            if(arguments.activationApplication == ActivationApplication::Gradient)
                options.auxiliaryInput = copyTensorFromEncodedStorage(
                    arguments.auxiliary, auxiliaryType, matrixLayout);
            else
                outputs.auxiliaryOutput = copyTensorFromEncodedStorage(
                    arguments.auxiliary, auxiliaryType, matrixLayout);
        }

        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguousLastDimensionFastest(Shape{1});
            outputs.amax = copyTensorFromEncodedStorage(arguments.amax, computeType, amaxLayout);
            options.accumulateAmax = arguments.accumulateAmax;
        }

        if(arguments.bias != nullptr)
        {
            const ScalarType biasType   = scalarType(arguments.biasType);
            const Layout     biasLayout = Layout::contiguousLastDimensionFastest(Shape{rows});
            Tensor bias  = copyTensorFromEncodedStorage(arguments.bias, biasType, biasLayout);
            options.bias = bias.expandDims(1);
        }

        if(arguments.outputScale != nullptr)
            options.outputScale = scalarValue(arguments.outputScale, computeType);
        if(arguments.auxiliaryScale != nullptr)
            options.auxiliaryScale = scalarValue(arguments.auxiliaryScale, computeType);
        if(outputType == ScalarType::Int8)
            options.outputConversion = OutputConversion::SaturatingInt8;
        options.activation            = activationFunction(arguments.activation,
                                                           arguments.activationParameter0,
                                                           arguments.activationParameter1);
        options.activationApplication = arguments.activationApplication;
        roc::host_numerics::referenceEpilogueInto(input, outputs, options);
        copyTensorEncodedBackingStorageToBuffer(
            arguments.output, storageBytesForLayout(outputType, matrixLayout), outputs.output);
        if(arguments.rawOutput != nullptr)
            copyTensorEncodedBackingStorageToBuffer(
                arguments.rawOutput,
                storageBytesForLayout(computeType, matrixLayout),
                *outputs.rawOutput);
        if(arguments.auxiliary != nullptr
           && arguments.activationApplication != ActivationApplication::Gradient)
            copyTensorEncodedBackingStorageToBuffer(
                arguments.auxiliary,
                storageBytesForLayout(scalarType(arguments.auxiliaryType), matrixLayout),
                *outputs.auxiliaryOutput);
        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguousLastDimensionFastest(Shape{1});
            copyTensorEncodedBackingStorageToBuffer(
                arguments.amax, storageBytesForLayout(computeType, amaxLayout), *outputs.amax);
        }
    }
} // namespace hipblaslt::host_numerics
