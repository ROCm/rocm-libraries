// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipblaslt/host_numerics/Reduction.hpp>
#include <hipblaslt/host_numerics/Types.hpp>

#include <cstddef>
#include <stdexcept>

namespace hipblaslt::host_numerics
{
    using namespace ::roc::host_numerics;

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

        Tensor output = copyTensorFromEncodedStorage(arguments.output, outputType, outputLayout);
        const ReductionRunInfo run = roc::host_numerics::referenceSum(
            ReductionRequest(copyTensorFromEncodedStorage(arguments.input, inputType, inputLayout),
                             output,
                             accumulatorType,
                             {1}));
        copyTensorEncodedBackingStorageToBuffer(
            arguments.output, storageBytesForLayout(outputType, outputLayout), output);
        return run;
    }
} // namespace hipblaslt::host_numerics
