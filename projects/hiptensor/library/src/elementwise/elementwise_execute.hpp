/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

#ifndef HIPTENSOR_ELEMENTWISE_EXECUTE_HPP
#define HIPTENSOR_ELEMENTWISE_EXECUTE_HPP

#include <cstddef>
#include <vector>

#include <hiptensor/hiptensor_types.h>

#include "../backend/launch_config.hpp"
#include "data_types.hpp"
#include "plancache_autotune.hpp"

namespace hiptensor
{
    // One tensor of an elementwise operation as the kernels address it.
    struct ElementwiseTensor
    {
        hiptensorTensorDescriptor const* desc;
        std::vector<int32_t>             modes;
        void const*                      data;
    };

    // Whether the elementwise kernels implement the data types of the operation
    // (HIPTENSOR_PERMUTATION, HIPTENSOR_ELEMENTWISE_BINARY or HIPTENSOR_ELEMENTWISE_TRINARY):
    // HIPTENSOR_STATUS_NOT_SUPPORTED otherwise. Used when the plan is created and when it runs.
    // These are the rows of the Composable Kernel backend; the API accepts further rows (a bf16
    // permutation, an f16 or bf16 trinary operation with f32 scalars, ...) that the rocKE
    // backend serves (rocke_plan.cpp, apiElementwiseRow), so this backend declines them.
    hiptensorStatus_t checkElementwiseDataTypes(hiptensorOperationDescriptor const& desc,
                                                char const*                         apiFuncName);

    // Plan-time validation of an elementwise operation: the data types, and that a kernel
    // exists for the output's rank and the operand types. HIPTENSOR_STATUS_NOT_SUPPORTED
    // otherwise, so that hiptensorCreatePlan reports what execution would only report as
    // HIPTENSOR_STATUS_INTERNAL_ERROR.
    hiptensorStatus_t elementwiseCheckPlan(hiptensorOperationDescriptor const& desc);

    // cuTENSOR semantics for zero scaling factors: an input whose scaling factor is zero is
    // neither read nor transformed. Its operator becomes HIPTENSOR_OP_INTERNAL_ZERO, so the term
    // is exactly zero whatever the input holds, and when its type is the output's, the kernel
    // reads the output's own element (each element is read and written by the same thread)
    // instead of the input, which may then be nullptr. `opIndex[i]` is the position of input i's
    // operator in `operators`. A zero-scaled nullptr input whose type is not the output's cannot
    // be redirected and would still be loaded by the kernel: HIPTENSOR_STATUS_NOT_SUPPORTED.
    hiptensorStatus_t applyZeroScales(std::vector<double> const&        scalars,
                                      std::vector<ElementwiseTensor>&   inputs,
                                      std::vector<std::size_t> const&   opIndex,
                                      std::vector<hiptensorOperator_t>& operators,
                                      ElementwiseTensor const&          output);

    // Whether every input's scaling factor is zero (a NaN scaling factor is not zero).
    bool elementwiseAllScalesZero(std::vector<double> const& scalars);

    // cuTENSOR semantics when every input's scaling factor is zero: no input is read and no
    // operator is applied, and ADD, MUL, MAX and MIN of zeros are zero, so the result does not
    // depend on the inputs: D = +0 at each of its elements. Writes them through D's own strides
    // on `stream`, with one memset when D's elements occupy a contiguous range, otherwise
    // element by element (the memory between the elements of a padded D is not written). Both
    // backends call it instead of running an elementwise kernel over the whole operation.
    hiptensorStatus_t
        elementwiseZeroOutput(hiptensorTensorDescriptor const& descD, void* D, hipStream_t stream);

    // Selects and runs an elementwise kernel: the plan's kernel from an earlier execution if it
    // supports this call, otherwise the best supporting kernel of the registry. `operators` is
    // in the order of the solution's initArgs, `queryOperators` in the order of the registry's
    // query. Records the selection in the plan and the plan cache.
    hiptensorStatus_t runElementwise(hiptensorHandle_t                       handle,
                                     hiptensorPlan_t                         plan,
                                     AutotuneOps                             autotuneOp,
                                     char const*                             apiFuncName,
                                     std::vector<double> const&              scalars,
                                     std::vector<ElementwiseTensor> const&   inputs,
                                     ElementwiseTensor const&                output,
                                     std::vector<hiptensorOperator_t> const& operators,
                                     std::vector<hiptensorOperator_t> const& queryOperators,
                                     std::size_t                             flopsPerElement,
                                     backend::LaunchConfig const&            cfg);

    // hiptensorElementwiseBinaryExecute on explicit operands: D = opAC(alpha * opA(A),
    // gamma * opC(C)), with the data types and scalar type of the plan's operation.
    hiptensorStatus_t elementwiseBinaryExecute(hiptensorHandle_t                handle,
                                               hiptensorPlan_t                  plan,
                                               hiptensorTensorDescriptor const* descA,
                                               std::vector<int32_t> const&      modeA,
                                               hiptensorOperator_t              opA,
                                               hiptensorTensorDescriptor const* descC,
                                               std::vector<int32_t> const&      modeC,
                                               hiptensorOperator_t              opC,
                                               hiptensorTensorDescriptor const* descD,
                                               std::vector<int32_t> const&      modeD,
                                               hiptensorOperator_t              opAC,
                                               void const*                      alpha,
                                               void const*                      A,
                                               void const*                      gamma,
                                               void const*                      C,
                                               void*                            D,
                                               backend::LaunchConfig const&     cfg,
                                               char const*                      apiFuncName);
} // namespace hiptensor

#endif // HIPTENSOR_ELEMENTWISE_EXECUTE_HPP
