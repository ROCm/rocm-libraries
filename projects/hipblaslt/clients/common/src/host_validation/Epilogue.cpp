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
            const Layout layout = Layout::contiguous(Shape{1});
            Tensor       value(type, layout, constStorage(pointer, type, layout, name));
            return {value.loadAs<double>({0}), 0.0};
        }

        void copyBack(void* destination, const Tensor& tensor)
        {
            if(!tensor.storage().empty())
                std::memcpy(destination, tensor.storage().data(), tensor.storage().size());
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
            Tensor(computeType,
                   matrixLayout,
                   constStorage(arguments.input, computeType, matrixLayout, "input")),
            Tensor(outputType,
                   matrixLayout,
                   mutableStorage(arguments.output, outputType, matrixLayout, "output")),
            computeType);

        if(arguments.rawOutput != nullptr)
            problem.rawOutput = Tensor(
                computeType,
                matrixLayout,
                mutableStorage(arguments.rawOutput, computeType, matrixLayout, "raw output"));

        if(arguments.auxiliary != nullptr)
        {
            const ScalarType auxiliaryType = scalarType(arguments.auxiliaryType);
            if(arguments.activationApplication == ActivationApplication::Gradient)
                problem.auxiliaryInput = Tensor(
                    auxiliaryType,
                    matrixLayout,
                    constStorage(
                        arguments.auxiliary, auxiliaryType, matrixLayout, "auxiliary input"));
            else
                problem.auxiliaryOutput = Tensor(
                    auxiliaryType,
                    matrixLayout,
                    mutableStorage(
                        arguments.auxiliary, auxiliaryType, matrixLayout, "auxiliary output"));
        }

        if(arguments.amax != nullptr)
        {
            const Layout amaxLayout = Layout::contiguous(Shape{1});
            problem.amax
                = Tensor(computeType,
                         amaxLayout,
                         mutableStorage(arguments.amax, computeType, amaxLayout, "AMax output"));
            problem.accumulateAmax = arguments.accumulateAmax;
        }

        if(arguments.bias != nullptr)
        {
            const ScalarType biasType     = scalarType(arguments.biasType);
            const size_t     biasElements = arguments.biasAxis == MatrixAxis::Row ? rows : columns;
            const Layout     biasLayout   = Layout::contiguous(Shape{biasElements});
            problem.bias
                = VectorBinding{Tensor(biasType,
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
        const EpilogueRunInfo run     = roc::host_validation::referenceEpilogue(problem);
        copyBack(arguments.output, problem.output);
        if(arguments.rawOutput != nullptr)
            copyBack(arguments.rawOutput, *problem.rawOutput);
        if(arguments.auxiliary != nullptr
           && arguments.activationApplication != ActivationApplication::Gradient)
            copyBack(arguments.auxiliary, *problem.auxiliaryOutput);
        if(arguments.amax != nullptr)
            copyBack(arguments.amax, *problem.amax);
        return run;
    }
} // namespace hipblaslt::host_validation
