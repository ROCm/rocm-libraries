// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// The MIOpen C API as consumed by the hipDNN MIOpen provider.
//
// Provider translation units include this header instead of <miopen/miopen.h>
// directly, for one reason: it also declares the three
// miopenConvolution*GetWorkSpaceSizeRange entry points, which the MIOpen
// implementation library exports but the public <miopen/miopen.h> intentionally
// does not declare.
//
// Provider source always spells the public entry-point names; which library
// those names bind to is a build-time choice. By default MIOpen is a single
// library and they bind it directly. Under MIOPEN_ENABLE_HIPDNN_WRAPPER=ON the
// provider links libMIOpen_private.so, whose miopen.h entry points carry an
// _impl suffix, and CMakeLists.txt force-includes MiopenApiPrivateRename.hpp to
// rewrite those names to their _impl form before this header is parsed. Keeping
// the rename out of the source is what lets one set of call sites serve both
// configurations.
//
// The three GetWorkSpaceSizeRange entry points below sit outside that scheme:
// they are not part of the miopen.h contract, so they keep their original names
// on libMIOpen_private.so and these declarations bind the same symbol either way.
#pragma once

#include <miopen/miopen.h>

// Exported from the MIOpen implementation library but intentionally not declared
// in the public miopen.h header.
// The signatures are copied verbatim from MIOpen, whose no-op top-level `const` on
// the pointer-typedef parameters trips clang-tidy, so we suppress those checks to
// keep the prototypes identical.
//
// These have C linkage, so a divergence from MIOpen's definitions would link
// cleanly and corrupt arguments at run time. The copy is therefore machine-checked
// against them by projects/miopen/script/check_public_abi.py check-headers, which
// runs in MIOpen CI and as a pre-commit hook on this file.
// NOLINTBEGIN(misc-misplaced-const,readability-avoid-const-params-in-decls)
extern "C" {
miopenStatus_t
    miopenConvolutionForwardGetWorkSpaceSizeRange(miopenHandle_t handle,
                                                  const miopenTensorDescriptor_t wDesc,
                                                  const miopenTensorDescriptor_t xDesc,
                                                  const miopenConvolutionDescriptor_t convDesc,
                                                  const miopenTensorDescriptor_t yDesc,
                                                  size_t* minWorkspaceSize,
                                                  size_t* maxWorkspaceSize);

miopenStatus_t
    miopenConvolutionBackwardDataGetWorkSpaceSizeRange(miopenHandle_t handle,
                                                       const miopenTensorDescriptor_t dyDesc,
                                                       const miopenTensorDescriptor_t wDesc,
                                                       const miopenConvolutionDescriptor_t convDesc,
                                                       const miopenTensorDescriptor_t dxDesc,
                                                       size_t* minWorkspaceSize,
                                                       size_t* maxWorkspaceSize);

miopenStatus_t miopenConvolutionBackwardWeightsGetWorkSpaceSizeRange(
    miopenHandle_t handle,
    const miopenTensorDescriptor_t dyDesc,
    const miopenTensorDescriptor_t xDesc,
    const miopenConvolutionDescriptor_t convDesc,
    const miopenTensorDescriptor_t dwDesc,
    size_t* minWorkspaceSize,
    size_t* maxWorkspaceSize);
}
// NOLINTEND(misc-misplaced-const,readability-avoid-const-params-in-decls)
