// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/adapters/hipblaslt/Epilogue.hpp>
#include <roc/host_validation/adapters/hipblaslt/Types.hpp>

#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

namespace roc::host_validation::hipblaslt_adapter
{
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
            const Layout layout = Layout::contiguous(Shape{1});
            TensorView   value(type, layout, constStorage(pointer, type, layout, name));
            return {value.loadAs<double>({0}), 0.0};
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

        EpilogueProblem problem(
            TensorView(computeType,
                       matrixLayout,
                       constStorage(arguments.input, computeType, matrixLayout, "input")),
            MutableTensorView(
                outputType,
                matrixLayout,
                mutableStorage(arguments.output, outputType, matrixLayout, "output")),
            computeType);

        if(arguments.rawOutput != nullptr)
            problem.rawOutput = MutableTensorView(
                computeType,
                matrixLayout,
                mutableStorage(arguments.rawOutput, computeType, matrixLayout, "raw output"));

        if(arguments.auxiliary != nullptr)
        {
            const ScalarType auxiliaryType = scalarType(arguments.auxiliaryType);
            if(arguments.activationApplication == ActivationApplication::Gradient)
                problem.auxiliaryInput = TensorView(auxiliaryType,
                                                    matrixLayout,
                                                    constStorage(arguments.auxiliary,
                                                                 auxiliaryType,
                                                                 matrixLayout,
                                                                 "auxiliary input"));
            else
                problem.auxiliaryOutput = MutableTensorView(auxiliaryType,
                                                            matrixLayout,
                                                            mutableStorage(arguments.auxiliary,
                                                                           auxiliaryType,
                                                                           matrixLayout,
                                                                           "auxiliary output"));
        }

        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguous(Shape{1});
            problem.amax = MutableTensorView(
                computeType,
                amaxLayout,
                mutableStorage(arguments.amax, computeType, amaxLayout, "AMax output"));
            problem.accumulateAmax = arguments.accumulateAmax;
        }

        if(arguments.bias != nullptr)
        {
            const ScalarType biasType     = scalarType(arguments.biasType);
            const size_t     biasElements = arguments.biasAxis == MatrixAxis::Row ? rows : columns;
            const Layout     biasLayout   = Layout::contiguous(Shape{biasElements});
            problem.bias                  = VectorBinding{
                TensorView(biasType,
                           biasLayout,
                           constStorage(arguments.bias, biasType, biasLayout, "bias")),
                arguments.biasAxis};
        }

        if(arguments.outputScale != nullptr)
            problem.outputScale = scalarValue(arguments.outputScale, computeType, "output scale");
        if(arguments.auxiliaryScale != nullptr)
            problem.auxiliaryScale
                = scalarValue(arguments.auxiliaryScale, computeType, "auxiliary scale");
        if(outputType == ScalarType::Int8)
            problem.outputConversion = OutputConversion::SaturatingInt8;
        problem.activation            = arguments.activation;
        problem.activationApplication = arguments.activationApplication;
        problem.activationParameter0  = arguments.activationParameter0;
        problem.activationParameter1  = arguments.activationParameter1;
        return roc::host_validation::referenceEpilogue(problem);
    }
} // namespace roc::host_validation::hipblaslt_adapter
