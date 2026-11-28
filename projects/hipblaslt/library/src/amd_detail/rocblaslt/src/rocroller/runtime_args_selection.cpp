// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "analytical_utils.hpp"
#include "gemm.hpp"
#include "rocroller_interop.hpp"
#include "runtime_args_selection.hpp"

#include <Tensile/RocRollerStreamKGrid.hpp>

int chooseStreamKGridSize(std::shared_ptr<GemmKernel> gemm, const RocblasltContractionProblem& prob)
{
    // One implementation, shared with the rocRoller-off launch. Filling the
    // inputs here and calling TensileLite::rocRollerStreamKGrid means an edit
    // to the origami sequence cannot land on only one of the two paths.
    size_t elementSizeA_bits
        = rocRoller::DataTypeInfo::Get(hipblaslt::hip2RocRollerType(gemm->params->kernelType.typeA))
              .elementBits;
    size_t elementSizeB_bits
        = rocRoller::DataTypeInfo::Get(hipblaslt::hip2RocRollerType(gemm->params->kernelType.typeB))
              .elementBits;
    size_t elementSizeAcc = rocRoller::DataTypeInfo::Get(
                                hipblaslt::hip2RocRollerType(gemm->params->kernelType.typeAcc))
                                .elementBytes;

    TensileLite::RocRollerStreamKGridInputs inputs;
    inputs.m                     = prob.m;
    inputs.n                     = prob.n;
    inputs.k                     = prob.k;
    inputs.batch                 = prob.batch_count;
    inputs.numCUs                = static_cast<size_t>(prob.sm_count_target);
    inputs.tileM                 = static_cast<size_t>(gemm->params->workgroupTile.m);
    inputs.tileN                 = static_cast<size_t>(gemm->params->workgroupTile.n);
    inputs.tileK                 = static_cast<size_t>(gemm->params->workgroupTile.k);
    inputs.occupancy             = gemm->occupancy;
    inputs.workspaceSize         = prob.workspaceSize;
    inputs.workspaceSizePerElemC = elementSizeAcc;
    inputs.aDtype                = hipDataType_to_analytical_type(gemm->params->kernelType.typeA);
    inputs.bDtype                = hipDataType_to_analytical_type(gemm->params->kernelType.typeB);
    inputs.aElementBits          = elementSizeA_bits;
    inputs.bElementBits          = elementSizeB_bits;

    const origami::hardware_t analytical_hardware = origami::hardware_t::get_hardware_for_device(0);
    return static_cast<int>(TensileLite::rocRollerStreamKGrid(inputs, analytical_hardware));
}
