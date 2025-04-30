// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>

#include <hiprand/hiprand.h>

namespace
{

hiprandStatus_t to_hiprand_status(rocrand_status status)
{
    switch(status)
    {
        case ROCRAND_STATUS_SUCCESS: return HIPRAND_STATUS_SUCCESS;
        case ROCRAND_STATUS_NOT_CREATED: return HIPRAND_STATUS_NOT_INITIALIZED;
        case ROCRAND_STATUS_VERSION_MISMATCH: return HIPRAND_STATUS_VERSION_MISMATCH;
        case ROCRAND_STATUS_ALLOCATION_FAILED: return HIPRAND_STATUS_ALLOCATION_FAILED;
        case ROCRAND_STATUS_TYPE_ERROR: return HIPRAND_STATUS_TYPE_ERROR;
        case ROCRAND_STATUS_OUT_OF_RANGE: return HIPRAND_STATUS_OUT_OF_RANGE;
        case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE: return HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
        case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case ROCRAND_STATUS_LAUNCH_FAILURE: return HIPRAND_STATUS_LAUNCH_FAILURE;
        // case ROCRAND_STATUS_PREEXISTING_FAILURE:
        //     return HIPRAND_STATUS_PREEXISTING_FAILURE;
        // case ROCRAND_STATUS_INITIALIZATION_FAILED:
        //     return HIPRAND_STATUS_INITIALIZATION_FAILED;
        // case ROCRAND_STATUS_ARCH_MISMATCH:
        //     return HIPRAND_STATUS_ARCH_MISMATCH;
        case ROCRAND_STATUS_INTERNAL_ERROR: return HIPRAND_STATUS_INTERNAL_ERROR;
        default: return HIPRAND_STATUS_INTERNAL_ERROR;
    }
}

rocrand_rng_type to_rocrand_rng_type(hiprandRngType_t rng_type)
{
    switch(rng_type)
    {
        case HIPRAND_RNG_PSEUDO_DEFAULT: return ROCRAND_RNG_PSEUDO_DEFAULT;
        case HIPRAND_RNG_PSEUDO_XORWOW: return ROCRAND_RNG_PSEUDO_XORWOW;
        case HIPRAND_RNG_PSEUDO_MRG32K3A: return ROCRAND_RNG_PSEUDO_MRG32K3A;
        case HIPRAND_RNG_PSEUDO_MTGP32: return ROCRAND_RNG_PSEUDO_MTGP32;
        case HIPRAND_RNG_PSEUDO_PHILOX4_32_10: return ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
        case HIPRAND_RNG_PSEUDO_MT19937: return ROCRAND_RNG_PSEUDO_MT19937;
        case HIPRAND_RNG_QUASI_DEFAULT: return ROCRAND_RNG_QUASI_DEFAULT;
        case HIPRAND_RNG_QUASI_SOBOL32: return ROCRAND_RNG_QUASI_SOBOL32;
        case HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32: return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
        case HIPRAND_RNG_QUASI_SOBOL64: return ROCRAND_RNG_QUASI_SOBOL64;
        case HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64: return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
        default: throw HIPRAND_STATUS_TYPE_ERROR;
    }
}

rocrand_direction_vector_set to_rocrand_direction_vector_set_type(hiprandDirectionVectorSet_t set)
{
    switch(set)
    {
        case HIPRAND_DIRECTION_VECTORS_32_JOEKUO6: return ROCRAND_DIRECTION_VECTORS_32_JOEKUO6;
        case HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            return ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
        case HIPRAND_DIRECTION_VECTORS_64_JOEKUO6: return ROCRAND_DIRECTION_VECTORS_64_JOEKUO6;
        case HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            return ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
    }
    throw HIPRAND_STATUS_TYPE_ERROR;
}

rocrand_ordering to_rocrand_ordering(hiprandOrdering_t ordering)
{
    switch(ordering)
    {
        case HIPRAND_ORDERING_PSEUDO_BEST: return ROCRAND_ORDERING_PSEUDO_BEST;
        case HIPRAND_ORDERING_PSEUDO_DEFAULT: return ROCRAND_ORDERING_PSEUDO_DEFAULT;
        case HIPRAND_ORDERING_PSEUDO_SEEDED: return ROCRAND_ORDERING_PSEUDO_SEEDED;
        case HIPRAND_ORDERING_PSEUDO_LEGACY: return ROCRAND_ORDERING_PSEUDO_LEGACY;
        case HIPRAND_ORDERING_PSEUDO_DYNAMIC: return ROCRAND_ORDERING_PSEUDO_DYNAMIC;
        case HIPRAND_ORDERING_QUASI_DEFAULT: return ROCRAND_ORDERING_QUASI_DEFAULT;
    }
    throw HIPRAND_STATUS_TYPE_ERROR;
}

} // namespace

hiprandStatus_t HIPRANDAPI hiprandCreateGenerator(hiprandGenerator_t* generator,
                                                  hiprandRngType_t    rng_type)
{
    try
    {
        return to_hiprand_status(rocrand_create_generator((rocrand_generator*)(generator),
                                                          to_rocrand_rng_type(rng_type)));
    }
    catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI hiprandCreateGeneratorHost(hiprandGenerator_t* generator,
                                                      hiprandRngType_t    rng_type)
{
    try
    {
        // cuRAND's host generator does not enqueue the generation on the stream
        return to_hiprand_status(
            rocrand_create_generator_host_blocking((rocrand_generator*)generator,
                                                   to_rocrand_rng_type(rng_type)));
    }
    catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI hiprandDestroyGenerator(hiprandGenerator_t generator)
{
    return to_hiprand_status(rocrand_destroy_generator((rocrand_generator)(generator)));
}

hiprandStatus_t HIPRANDAPI
    hiprandGenerate(hiprandGenerator_t generator, unsigned int* output_data, size_t n)
{
    return to_hiprand_status(rocrand_generate((rocrand_generator)(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI
    hiprandGenerateChar(hiprandGenerator_t generator, unsigned char* output_data, size_t n)
{
    return to_hiprand_status(rocrand_generate_char((rocrand_generator)(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI
    hiprandGenerateShort(hiprandGenerator_t generator, unsigned short* output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate_short((rocrand_generator)(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateLongLong(hiprandGenerator_t      generator,
                                                   unsigned long long int* output_data,
                                                   size_t                  n)
{
    return to_hiprand_status(
        rocrand_generate_long_long(reinterpret_cast<rocrand_generator>(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI
    hiprandGenerateUniform(hiprandGenerator_t generator, float* output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate_uniform((rocrand_generator)(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI
    hiprandGenerateUniformDouble(hiprandGenerator_t generator, double* output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate_uniform_double((rocrand_generator)(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI
    hiprandGenerateUniformHalf(hiprandGenerator_t generator, half* output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate_uniform_half((rocrand_generator)(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateNormal(
    hiprandGenerator_t generator, float* output_data, size_t n, float mean, float stddev)
{
    return to_hiprand_status(
        rocrand_generate_normal((rocrand_generator)(generator), output_data, n, mean, stddev));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateNormalDouble(
    hiprandGenerator_t generator, double* output_data, size_t n, double mean, double stddev)
{
    return to_hiprand_status(rocrand_generate_normal_double((rocrand_generator)(generator),
                                                            output_data,
                                                            n,
                                                            mean,
                                                            stddev));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateNormalHalf(
    hiprandGenerator_t generator, half* output_data, size_t n, half mean, half stddev)
{
    return to_hiprand_status(
        rocrand_generate_normal_half((rocrand_generator)(generator), output_data, n, mean, stddev));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateLogNormal(
    hiprandGenerator_t generator, float* output_data, size_t n, float mean, float stddev)
{
    return to_hiprand_status(
        rocrand_generate_log_normal((rocrand_generator)(generator), output_data, n, mean, stddev));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateLogNormalDouble(
    hiprandGenerator_t generator, double* output_data, size_t n, double mean, double stddev)
{
    return to_hiprand_status(rocrand_generate_log_normal_double((rocrand_generator)(generator),
                                                                output_data,
                                                                n,
                                                                mean,
                                                                stddev));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateLogNormalHalf(
    hiprandGenerator_t generator, half* output_data, size_t n, half mean, half stddev)
{
    return to_hiprand_status(rocrand_generate_log_normal_half((rocrand_generator)(generator),
                                                              output_data,
                                                              n,
                                                              mean,
                                                              stddev));
}

hiprandStatus_t HIPRANDAPI hiprandGeneratePoisson(hiprandGenerator_t generator,
                                                  unsigned int*      output_data,
                                                  size_t             n,
                                                  double             lambda)
{
    return to_hiprand_status(
        rocrand_generate_poisson((rocrand_generator)(generator), output_data, n, lambda));
}

hiprandStatus_t HIPRANDAPI hiprandGenerateSeeds(hiprandGenerator_t generator)
{
    return to_hiprand_status(rocrand_initialize_generator((rocrand_generator)(generator)));
}

hiprandStatus_t HIPRANDAPI hiprandSetStream(hiprandGenerator_t generator, hipStream_t stream)
{
    return to_hiprand_status(rocrand_set_stream((rocrand_generator)(generator), stream));
}

hiprandStatus_t HIPRANDAPI hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator,
                                                               unsigned long long seed)
{
    return to_hiprand_status(rocrand_set_seed((rocrand_generator)(generator), seed));
}

hiprandStatus_t HIPRANDAPI hiprandSetGeneratorOffset(hiprandGenerator_t generator,
                                                     unsigned long long offset)
{
    return to_hiprand_status(rocrand_set_offset((rocrand_generator)(generator), offset));
}

hiprandStatus_t HIPRANDAPI hiprandSetGeneratorOrdering(hiprandGenerator_t generator,
                                                       hiprandOrdering_t  order)
{
    try
    {
        return to_hiprand_status(
            rocrand_set_ordering((rocrand_generator)(generator), to_rocrand_ordering(order)));
    }
    catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator,
                                                                    unsigned int       dimensions)
{
    return to_hiprand_status(
        rocrand_set_quasi_random_generator_dimensions((rocrand_generator)(generator), dimensions));
}

hiprandStatus_t HIPRANDAPI hiprandGetVersion(int* version)
{
    return to_hiprand_status(rocrand_get_version(version));
}

hiprandStatus_t HIPRANDAPI
    hiprandCreatePoissonDistribution(double                         lambda,
                                     hiprandDiscreteDistribution_t* discrete_distribution)
{
    return to_hiprand_status(rocrand_create_poisson_distribution(lambda, discrete_distribution));
}

hiprandStatus_t HIPRANDAPI
    hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution)
{
    return to_hiprand_status(rocrand_destroy_discrete_distribution(discrete_distribution));
}

hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors32(hiprandDirectionVectors32_t** vectors,
                                                        hiprandDirectionVectorSet_t   set)
{
    using internal = unsigned int;

    // The memory layout  between rocRAND and cuRAND  is the same. Both
    // contain a series of unsigned ints (long long for 64-bit variant)
    // However, the accessing type is different:
    // - rocRAND uses a const c-style array.
    // - cuRAND  uses a non-const c-style array of c-style arrays.
    // - hipRAND uses cuRAND's style for consistency.
    // Since 1) this pointer is only used to transfer data from host to
    // device, 2) the memory layout  between rocRAND and hipRAND is the
    // same, and 3) the data being referenced should not be modified on
    // host, reinterpret and const cast is OK.
    const internal** raw_ptr = const_cast<const internal**>(reinterpret_cast<internal**>(vectors));
    return to_hiprand_status(
        rocrand_get_direction_vectors32(raw_ptr, to_rocrand_direction_vector_set_type(set)));
}

hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors64(hiprandDirectionVectors64_t** vectors,
                                                        hiprandDirectionVectorSet_t   set)
{
    using internal = unsigned long long;

    // See 'hiprandGetDirectionVectors32'.
    const internal** raw_ptr = const_cast<const internal**>(reinterpret_cast<internal**>(vectors));
    return to_hiprand_status(
        rocrand_get_direction_vectors64(raw_ptr, to_rocrand_direction_vector_set_type(set)));
}

hiprandStatus_t HIPRANDAPI hiprandGetScrambleConstants32(const unsigned int** constants)
{
    return to_hiprand_status(rocrand_get_scramble_constants32(constants));
}

hiprandStatus_t HIPRANDAPI hiprandGetScrambleConstants64(const unsigned long long** constants)
{
    return to_hiprand_status(rocrand_get_scramble_constants64(constants));
}
