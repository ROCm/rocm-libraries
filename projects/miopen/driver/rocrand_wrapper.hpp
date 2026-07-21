// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once
#ifndef GUARD_ROCRAND_WRAPPER_HPP
#define GUARD_ROCRAND_WRAPPER_HPP

// The wrapper is necessary because direct inclusion of rocrand.hpp into the driver source leads to
// build errors like this: "amd_hip_fp16.h:1743:19: error: type alias redefinition with different
// types ('__half' vs 'half_float::half')." The reason is that the driver uses the definition of
// half from half.hpp, while rocrand uses definition of half type from HIP headers, i.e. the
// definitions are different.

#include <miopen/demangle.hpp>
#include <half/half.hpp>

#include <iostream>
#include <typeinfo>

namespace gpumemrand {

int gen_0_1(double* buf, size_t sz);
int gen_0_1(float* buf, size_t sz);
int gen_0_1(half_float::half* buf, size_t sz);

template <typename T>
int gen_0_1(T* buf, size_t sz)
{
    std::cout << "Warning: gpumemrand functions are supported only for double, float and half. GPU "
                 "buffer { "
              << static_cast<void*>(buf) << ", " << sz << ", " << miopen::demangle(typeid(T).name())
              << " } remains uninitialized." << std::endl;
    return 0;
}

} // namespace gpumemrand

#endif // GUARD_ROCRAND_WRAPPER_HPP
