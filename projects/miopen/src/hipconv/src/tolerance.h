#pragma once

#include "hipconv/conv2d_params.hpp"

#include <cstddef>

namespace hipconv
{

float get_unit_roundoff(DataType dtype);

size_t get_accumulation_depth(const Conv2dParams& par);

// Compute tolerance for convolution kernels that use mixed-precision
// matrix multiply (low-precision inputs, fp32 accumulation).
void get_mixed_precision_tolerance(const Conv2dParams& par, float& atol, float& rtol);

} // namespace hipconv
