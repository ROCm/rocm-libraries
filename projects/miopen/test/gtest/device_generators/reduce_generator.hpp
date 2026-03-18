// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_REDUCE_GENERATOR_HPP
#define GUARD_MIOPEN_REDUCE_GENERATOR_HPP

#include <cstdint>

class bfloat16;

namespace miopen {
struct TensorDescriptor;
} // namespace miopen

namespace half_float {
class half;
} // namespace half_float

using hipStream_t = struct ihipStream_t*;

namespace test::gtest {

/**
 * @brief Generate deterministic reduction-test inputs on GPU (grid-invariant).
 *
 * Uses Philox seeded by (seed, i) per element.
 *
 * Operator mapping:
 * - ADD/AVG:  sign*(rand_val/max_val) + 0.01  (avoid frequent exact zeros)
 * - MUL:      values near 1.0 to avoid huge/zero products across many elements
 * - AMAX:     +/- (rand_val + 0.5)           (avoid zeros for abs-max)
 * - NORM1/2:  rand_val*sign*(0.1 + 0.9*u)       ratio in [0.1, 1.0)
 * - MIN/MAX:  rand_val*sign
 */
template <typename T>
void ReduceGenerator(T* output,
                     std::size_t n,
                     uint64_t seed,
                     int op,
                     uint64_t max_val,
                     const miopen::TensorDescriptor& desc,
                     hipStream_t stream);

extern template void ReduceGenerator<float>(
    float*, std::size_t, uint64_t, int, uint64_t, const miopen::TensorDescriptor&, hipStream_t);
extern template void ReduceGenerator<double>(
    double*, std::size_t, uint64_t, int, uint64_t, const miopen::TensorDescriptor&, hipStream_t);
extern template void ReduceGenerator<half_float::half>(half_float::half*,
                                                       std::size_t,
                                                       uint64_t,
                                                       int,
                                                       uint64_t,
                                                       const miopen::TensorDescriptor&,
                                                       hipStream_t);
extern template void ReduceGenerator<bfloat16>(
    bfloat16*, std::size_t, uint64_t, int, uint64_t, const miopen::TensorDescriptor&, hipStream_t);
extern template void ReduceGenerator<int8_t>(
    int8_t*, std::size_t, uint64_t, int, uint64_t, const miopen::TensorDescriptor&, hipStream_t);
extern template void ReduceGenerator<int32_t>(
    int32_t*, std::size_t, uint64_t, int, uint64_t, const miopen::TensorDescriptor&, hipStream_t);

} // namespace test::gtest

#endif // GUARD_MIOPEN_DEVICE_PRNG_HPP
