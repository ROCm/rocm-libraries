#pragma once

// Stub algorithm for architectures without a grouped convolution implementation.
// Included by per-arch .cpp files that define STUB_ALGORITHM_NAME before including.

#include "algorithm.h"

#ifndef __HIP_DEVICE_COMPILE__

namespace
{
bool stub_is_applicable(const hipconv::Conv2dParams&)
{
    return false;
}
} // namespace

extern const hipconv::ConvAlgorithm STUB_ALGORITHM_NAME{stub_is_applicable, {}};

#endif // __HIP_DEVICE_COMPILE__
