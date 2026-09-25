#pragma once

#include "hipconv/conv2d_params.hpp"
#include "hipconv/tolerance.hpp"

#include <cstddef>

namespace hipconv
{

float get_unit_roundoff(DataType dtype);

size_t get_accumulation_depth(const Conv2dParams& par);

// Compute tolerance for convolution kernels that use mixed-precision
// matrix multiply (low-precision inputs, fp32 accumulation).
//
// rtol is TOLERANCE_UNAVAILABLE at a depth past gamma's hypothesis, where no bound exists.
void get_mixed_precision_tolerance(const Conv2dParams& par, float& atol, float& rtol);

// The same, for a kernel whose accumulation is blocked.
//
// `depth` is the number of fp32 roundings on the longest path from a product to the result, which
// for a blocked summation is far below get_accumulation_depth(par); see
// docs/algorithms/direct/direct-wgrad-tolerance.md.
void get_mixed_precision_tolerance(const Conv2dParams& par, size_t depth, float& atol, float& rtol);

} // namespace hipconv
