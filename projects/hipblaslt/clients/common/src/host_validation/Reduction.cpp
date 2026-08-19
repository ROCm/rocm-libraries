// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_validation/Reduction.hpp>
#include <hipblaslt/host_validation/Types.hpp>

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
        std::span<const std::byte>
            constStorage(const void* pointer, ScalarType type, const Layout& layout)
        {
            const size_t bytes = storageBytesForLayout(type, layout);
            if(pointer == nullptr && bytes != 0)
                throw std::invalid_argument("Null hipBLASLt reduction input.");
            return {static_cast<const std::byte*>(pointer), bytes};
        }

        std::span<std::byte> mutableStorage(void* pointer, ScalarType type, const Layout& layout)
        {
            const size_t bytes = storageBytesForLayout(type, layout);
            if(pointer == nullptr && bytes != 0)
                throw std::invalid_argument("Null hipBLASLt reduction output.");
            return {static_cast<std::byte*>(pointer), bytes};
        }
    } // namespace

    ReductionRunInfo referenceSum(const ReductionArguments& arguments)
    {
        if(arguments.rows < 0 || arguments.columns < 0)
            throw std::invalid_argument("hipBLASLt reduction dimensions must be nonnegative.");

        const ScalarType inputType       = scalarType(arguments.inputType);
        const ScalarType outputType      = scalarType(arguments.outputType);
        const ScalarType accumulatorType = scalarType(arguments.accumulatorType);
        const Layout     inputLayout(
            Shape{static_cast<size_t>(arguments.rows), static_cast<size_t>(arguments.columns)},
            {static_cast<ptrdiff_t>(arguments.rowStride),
             static_cast<ptrdiff_t>(arguments.columnStride)});
        const Layout outputLayout(Shape{static_cast<size_t>(arguments.rows)},
                                  {static_cast<ptrdiff_t>(arguments.outputStride)});

        Tensor output(
            outputType, outputLayout, mutableStorage(arguments.output, outputType, outputLayout));
        const ReductionRunInfo run = roc::host_validation::referenceSum(ReductionRequest(
            Tensor(inputType, inputLayout, constStorage(arguments.input, inputType, inputLayout)),
            output,
            accumulatorType,
            {1}));
        if(!output.storage().empty())
            std::memcpy(arguments.output, output.storage().data(), output.storage().size());
        return run;
    }
} // namespace hipblaslt::host_validation
