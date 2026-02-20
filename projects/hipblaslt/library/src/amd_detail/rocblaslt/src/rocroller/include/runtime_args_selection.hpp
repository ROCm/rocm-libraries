// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm.hpp"
#include "rocblaslt.h"

const int DEFAULT_WGM = 2;

int chooseStreamKGridSize(std::shared_ptr<GemmKernel>        gemm,
    const RocblasltContractionProblem& prob);
