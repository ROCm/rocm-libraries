// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm.hpp"
#include "solution_cache.hpp"

#include "rocblaslt.h"

void preloadCustomKernels(SolutionCache& cache);
