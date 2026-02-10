#pragma once

#include "gemm.hpp"
#include "solution_cache.hpp"

#include "rocblaslt.h"

void preloadCustomKernels(SolutionCache& cache);

rocblaslt_status runCustomKernel(std::shared_ptr<GemmKernel> gemm, const RocblasltContractionProblem& prob);
