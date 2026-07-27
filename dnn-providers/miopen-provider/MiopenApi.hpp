// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// The MIOpen C API as consumed by the hipDNN MIOpen provider.
//
// Provider translation units include this header instead of <miopen/miopen.h>
// directly, for one reason: it also declares the three
// miopenConvolution*GetWorkSpaceSizeRange entry points, which are exported from
// libMIOpen but intentionally absent from the public <miopen/miopen.h>.
//
// The provider always calls the public entry-point names. When MIOpen is built
// with the public/private split (MIOPEN_ENABLE_HIPDNN_WRAPPER=ON) the provider
// links libMIOpen_private.so, whose symbols carry an _impl suffix; in that build
// the compiler force-includes MiopenApiPrivateRename.hpp (see CMakeLists.txt),
// which renames every public name — including the declarations pulled in here — to
// its _impl form before this header is parsed. With the split OFF nothing is
// renamed and the provider binds the public libMIOpen exactly as before the split.
#pragma once

#include <miopen/miopen.h>

// Exported from libMIOpen but intentionally not declared in the public miopen.h.
// The signatures are copied verbatim from MIOpen, whose no-op top-level `const` on
// the pointer-typedef parameters trips clang-tidy, so we suppress those checks to
// keep the prototypes identical.
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
