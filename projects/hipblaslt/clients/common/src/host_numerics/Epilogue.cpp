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
        Scalar scalarValue(const void* pointer, ScalarType type)
        {
            const size_t storageBytes = (scalarTypeInfo(type).storageBits + 7U) / 8U;
            return Scalar::fromStorage(
                type,
                std::span<const std::byte>(static_cast<const std::byte*>(pointer), storageBytes));
        }
    } // namespace

    EpilogueRunInfo referenceEpilogue(const EpilogueArguments& arguments)
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

        EpilogueRequest request(
            copyTensorFromEncodedStorage(arguments.input, computeType, matrixLayout),
            copyTensorFromEncodedStorage(arguments.output, outputType, matrixLayout),
            computeType);

        if(arguments.rawOutput != nullptr)
        {
            request.rawOutput
                = copyTensorFromEncodedStorage(arguments.rawOutput, computeType, matrixLayout);
            request.rawOutputType = request.rawOutput->type();
        }

        if(arguments.auxiliary != nullptr)
        {
            const ScalarType auxiliaryType = scalarType(arguments.auxiliaryType);
            if(arguments.activationApplication == ActivationApplication::Gradient)
                request.auxiliaryInput = copyTensorFromEncodedStorage(
                    arguments.auxiliary, auxiliaryType, matrixLayout);
            else
            {
                request.auxiliaryOutput = copyTensorFromEncodedStorage(
                    arguments.auxiliary, auxiliaryType, matrixLayout);
                request.auxiliaryOutputType = request.auxiliaryOutput->type();
            }
        }

        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguousLastDimensionFastest(Shape{1});
            request.amax = copyTensorFromEncodedStorage(arguments.amax, computeType, amaxLayout);
            request.amaxType       = request.amax->type();
            request.accumulateAmax = arguments.accumulateAmax;
        }

        if(arguments.bias != nullptr)
        {
            const ScalarType biasType     = scalarType(arguments.biasType);
            const size_t     biasElements = arguments.biasAxis == MatrixAxis::Row ? rows : columns;
            const Layout biasLayout = Layout::contiguousLastDimensionFastest(Shape{biasElements});
            request.bias
                = VectorBinding{copyTensorFromEncodedStorage(arguments.bias, biasType, biasLayout),
                                arguments.biasAxis};
        }

        if(arguments.outputScale != nullptr)
            request.outputScale = scalarValue(arguments.outputScale, computeType);
        if(arguments.auxiliaryScale != nullptr)
            request.auxiliaryScale = scalarValue(arguments.auxiliaryScale, computeType);
        if(outputType == ScalarType::Int8)
            request.outputConversion = OutputConversion::SaturatingInt8;
        request.activation            = arguments.activation;
        request.activationApplication = arguments.activationApplication;
        request.activationParameter0  = arguments.activationParameter0;
        request.activationParameter1  = arguments.activationParameter1;
        const EpilogueRunInfo run     = roc::host_numerics::referenceEpilogue(request);
        copyTensorEncodedBackingStorageToBuffer(
            arguments.output, storageBytesForLayout(outputType, matrixLayout), request.output);
        if(arguments.rawOutput != nullptr)
            copyTensorEncodedBackingStorageToBuffer(
                arguments.rawOutput,
                storageBytesForLayout(computeType, matrixLayout),
                *request.rawOutput);
        if(arguments.auxiliary != nullptr
           && arguments.activationApplication != ActivationApplication::Gradient)
            copyTensorEncodedBackingStorageToBuffer(
                arguments.auxiliary,
                storageBytesForLayout(scalarType(arguments.auxiliaryType), matrixLayout),
                *request.auxiliaryOutput);
        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguousLastDimensionFastest(Shape{1});
            copyTensorEncodedBackingStorageToBuffer(
                arguments.amax, storageBytesForLayout(computeType, amaxLayout), *request.amax);
        }
        return run;
    }
} // namespace hipblaslt::host_numerics
