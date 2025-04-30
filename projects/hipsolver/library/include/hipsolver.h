/* ************************************************************************
 * Copyright (C) 2020-2024 Advanced Micro Devices, Inc.
 * ************************************************************************ */

// HIP = Heterogeneous-compute Interface for Portability
//
// Define a extremely thin runtime layer that allows source code to be compiled unmodified
// through either AMD HCC or NVCC.   Key features tend to be in the spirit
// and terminology of CUDA, but with a portable path to other accelerators as well.
//
// This is the master include file for hipsolver, wrapping around rocsolver and cusolver
//
#ifndef HIPSOLVER_H
#define HIPSOLVER_H

#include "internal/hipsolver-export.h"
#include "internal/hipsolver-version.h"

/* Defines types used across the hipSOLVER library. */
#include "internal/hipsolver-types.h"

/* Defines functions with the hipsolver prefix. APIs differ from cuSOLVER in some cases
 * in order to enable better rocSOLVER performance.
 */
#include "internal/hipsolver-functions.h"

/* Defines functions with the hipsolverDn prefix. APIs match those from cuSOLVER but may
 * result in degraded rocSOLVER performance.
 */
#include "internal/hipsolver-dense.h"
#include "internal/hipsolver-dense64.h"

/* Defines functions and types with the hipsolverRf prefix. APIs match those from cuSOLVER.
 */
#include "internal/hipsolver-refactor.h"

/* Defines functions and types with the hipsolverSp prefix. APIs match those from cuSOLVER.
 */
#include "internal/hipsolver-sparse.h"

#endif // HIPSOLVER_H
