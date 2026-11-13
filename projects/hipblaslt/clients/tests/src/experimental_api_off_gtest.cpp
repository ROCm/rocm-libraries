// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Without ROCM_ENABLE_EXPERIMENTAL_APIS, the experimental header must declare nothing, so a
// consumer's feature checks take the fallback path. Checked at compile time.
#define ROCM_ENABLE_EXPERIMENTAL_APIS 0
#include <hipblaslt/hipblaslt-experimental.h>

#if defined(HIPBLASLT_EXPERIMENTAL_DEMO) || defined(HIPBLASLT_EXPERIMENTAL_ENUM_BIT)
#error "hipblaslt-experimental.h declared experimental APIs with ROCM_ENABLE_EXPERIMENTAL_APIS=0"
#endif
