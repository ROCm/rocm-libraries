#ifndef RANDOMH
#define RANDOMH

#include <random>
#include <hiprand/hiprand.hpp>

#ifdef __HIP_DEVICE_COMPILE__
using RandState = hiprandState;
#else
using RandState = void;
#endif

__host__ double random_double([[maybe_unused]] void *state) {
    static thread_local std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static thread_local std::mt19937 generator;
    return distribution(generator);
}

__device__ inline void setup_random(uint32_t tId, hiprandState *state)
{
    hiprand_init(1984+tId, 0, 0, state);
}

__device__ double random_double(hiprandState *state) {
    return hiprand_uniform( state );
}

#endif
