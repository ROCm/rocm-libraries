// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_validation/Epilogue.hpp>
#include <hipblaslt/host_validation/Types.hpp>

#include <complex>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>

namespace hipblaslt::host_validation
{
    using namespace ::roc::host_validation;

    namespace
    {
        std::span<const std::byte> constStorage(
            const void* pointer, ScalarType type, const Layout& layout, const char* name)
        {
            const size_t bytes = storageBytesForLayout(type, layout);
            if(pointer == nullptr && bytes != 0)
                throw std::invalid_argument(std::string("Null hipBLASLt epilogue ") + name + ".");
            return {static_cast<const std::byte*>(pointer), bytes};
        }

        std::span<std::byte> mutableStorage(
            void* pointer, ScalarType type, const Layout& layout, const char* name)
        {
            const size_t bytes = storageBytesForLayout(type, layout);
            if(pointer == nullptr && bytes != 0)
                throw std::invalid_argument(std::string("Null hipBLASLt epilogue ") + name + ".");
            return {static_cast<std::byte*>(pointer), bytes};
        }

        std::complex<double> scalarValue(const void* pointer, ScalarType type, const char* name)
        {
            const Layout layout = Layout::contiguousLastDimensionFastest(Shape{1});
            Tensor value =
                Tensor::copyEncodedBackingStorage(type, layout, constStorage(pointer, type, layout, name));
            return {value.loadAs<double>({0}), 0.0};
        }

        void copyBack(void* destination, const Tensor& tensor)
        {
            if(!tensor.rawEncodedBackingStorage().empty())
                std::memcpy(destination, tensor.rawEncodedBackingStorage().data(), tensor.rawEncodedBackingStorage().size());
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
            Tensor::copyEncodedBackingStorage(
                computeType, matrixLayout,
                constStorage(arguments.input, computeType, matrixLayout, "input")),
            Tensor::copyEncodedBackingStorage(
                outputType, matrixLayout,
                mutableStorage(arguments.output, outputType, matrixLayout, "output")),
            computeType);

        if(arguments.rawOutput != nullptr)
        {
            request.rawOutput = Tensor::copyEncodedBackingStorage(
                computeType,
                matrixLayout,
                mutableStorage(arguments.rawOutput, computeType, matrixLayout, "raw output"));
            request.rawOutputType = request.rawOutput->type();
        }

        if(arguments.auxiliary != nullptr)
        {
            const ScalarType auxiliaryType = scalarType(arguments.auxiliaryType);
            if(arguments.activationApplication == ActivationApplication::Gradient)
                request.auxiliaryInput = Tensor::copyEncodedBackingStorage(
                    auxiliaryType,
                    matrixLayout,
                    constStorage(
                        arguments.auxiliary, auxiliaryType, matrixLayout, "auxiliary input"));
            else
            {
                request.auxiliaryOutput = Tensor::copyEncodedBackingStorage(
                    auxiliaryType,
                    matrixLayout,
                    mutableStorage(
                        arguments.auxiliary, auxiliaryType, matrixLayout, "auxiliary output"));
                request.auxiliaryOutputType = request.auxiliaryOutput->type();
            }
        }

        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguousLastDimensionFastest(Shape{1});
            request.amax
                = Tensor::copyEncodedBackingStorage(
                    computeType, amaxLayout,
                    mutableStorage(arguments.amax, computeType, amaxLayout, "AMax output"));
            request.amaxType       = request.amax->type();
            request.accumulateAmax = arguments.accumulateAmax;
        }

        if(arguments.bias != nullptr)
        {
            const ScalarType biasType     = scalarType(arguments.biasType);
            const size_t     biasElements = arguments.biasAxis == MatrixAxis::Row ? rows : columns;
            const Layout     biasLayout   = Layout::contiguousLastDimensionFastest(Shape{biasElements});
            request.bias
                = VectorBinding{Tensor::copyEncodedBackingStorage(
                                    biasType, biasLayout,
                                    constStorage(arguments.bias, biasType, biasLayout, "bias")),
                                arguments.biasAxis};
        }

        if(arguments.outputScale != nullptr)
            request.outputScale = scalarValue(arguments.outputScale, computeType, "output scale");
        if(arguments.auxiliaryScale != nullptr)
            request.auxiliaryScale
                = scalarValue(arguments.auxiliaryScale, computeType, "auxiliary scale");
        if(outputType == ScalarType::Int8)
            request.outputConversion = OutputConversion::SaturatingInt8;
        request.activation            = arguments.activation;
        request.activationApplication = arguments.activationApplication;
        request.activationParameter0  = arguments.activationParameter0;
        request.activationParameter1  = arguments.activationParameter1;
        const EpilogueRunInfo run     = roc::host_validation::referenceEpilogue(request);
        copyBack(arguments.output, request.output);
        if(arguments.rawOutput != nullptr)
            copyBack(arguments.rawOutput, *request.rawOutput);
        if(arguments.auxiliary != nullptr
           && arguments.activationApplication != ActivationApplication::Gradient)
            copyBack(arguments.auxiliary, *request.auxiliaryOutput);
        if(arguments.amax != nullptr)
            copyBack(arguments.amax, *request.amax);
        return run;
    }
} // namespace hipblaslt::host_validation
