// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <curand.h>

#include <hiprand/hiprand.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

hiprandStatus_t to_hiprand_status(curandStatus_t status)
{
    switch(status)
    {
        case CURAND_STATUS_SUCCESS:
            return HIPRAND_STATUS_SUCCESS;
        case CURAND_STATUS_NOT_INITIALIZED:
            return HIPRAND_STATUS_NOT_INITIALIZED;
        case CURAND_STATUS_VERSION_MISMATCH:
            return HIPRAND_STATUS_VERSION_MISMATCH;
        case CURAND_STATUS_ALLOCATION_FAILED:
            return HIPRAND_STATUS_ALLOCATION_FAILED;
        case CURAND_STATUS_TYPE_ERROR:
            return HIPRAND_STATUS_TYPE_ERROR;
        case CURAND_STATUS_OUT_OF_RANGE:
            return HIPRAND_STATUS_OUT_OF_RANGE;
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            return HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case CURAND_STATUS_LAUNCH_FAILURE:
            return HIPRAND_STATUS_LAUNCH_FAILURE;
        case CURAND_STATUS_PREEXISTING_FAILURE:
            return HIPRAND_STATUS_PREEXISTING_FAILURE;
        case CURAND_STATUS_INITIALIZATION_FAILED:
            return HIPRAND_STATUS_INITIALIZATION_FAILED;
        case CURAND_STATUS_ARCH_MISMATCH:
            return HIPRAND_STATUS_ARCH_MISMATCH;
        case CURAND_STATUS_INTERNAL_ERROR:
            return HIPRAND_STATUS_INTERNAL_ERROR;
        default:
            return HIPRAND_STATUS_INTERNAL_ERROR;
    }
}

curandRngType_t to_curand_rng_type(hiprandRngType_t rng_type)
{
    switch(rng_type)
    {
        case HIPRAND_RNG_PSEUDO_DEFAULT:
            return CURAND_RNG_PSEUDO_DEFAULT;
        case HIPRAND_RNG_PSEUDO_XORWOW:
            return CURAND_RNG_PSEUDO_XORWOW;
        case HIPRAND_RNG_PSEUDO_MRG32K3A:
            return CURAND_RNG_PSEUDO_MRG32K3A;
        case HIPRAND_RNG_PSEUDO_MTGP32:
            return CURAND_RNG_PSEUDO_MTGP32;
        case HIPRAND_RNG_PSEUDO_PHILOX4_32_10:
            return CURAND_RNG_PSEUDO_PHILOX4_32_10;
        case HIPRAND_RNG_PSEUDO_MT19937:
            return CURAND_RNG_PSEUDO_MT19937;
        case HIPRAND_RNG_QUASI_DEFAULT:
            return CURAND_RNG_QUASI_DEFAULT;
        case HIPRAND_RNG_QUASI_SOBOL32:
            return CURAND_RNG_QUASI_SOBOL32;
        case HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
            return CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
        case HIPRAND_RNG_QUASI_SOBOL64:
            return CURAND_RNG_QUASI_SOBOL64;
        case HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
            return CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
        default:
            throw HIPRAND_STATUS_TYPE_ERROR;
    }
}

curandDirectionVectorSet_t to_curand_direction_vector_set_type(hiprandDirectionVectorSet_t set)
{
    switch(set)
    {
        case HIPRAND_DIRECTION_VECTORS_32_JOEKUO6: return CURAND_DIRECTION_VECTORS_32_JOEKUO6;
        case HIPRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            return CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6;
        case HIPRAND_DIRECTION_VECTORS_64_JOEKUO6: return CURAND_DIRECTION_VECTORS_64_JOEKUO6;
        case HIPRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            return CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6;
    }
    throw HIPRAND_STATUS_TYPE_ERROR;
}

curandOrdering_t to_curand_ordering(hiprandOrdering_t ordering)
{
    switch(ordering)
    {
        case HIPRAND_ORDERING_PSEUDO_BEST: return CURAND_ORDERING_PSEUDO_BEST;
        case HIPRAND_ORDERING_PSEUDO_DEFAULT: return CURAND_ORDERING_PSEUDO_DEFAULT;
        case HIPRAND_ORDERING_PSEUDO_SEEDED: return CURAND_ORDERING_PSEUDO_SEEDED;
        case HIPRAND_ORDERING_PSEUDO_LEGACY: return CURAND_ORDERING_PSEUDO_LEGACY;
        case HIPRAND_ORDERING_PSEUDO_DYNAMIC: return CURAND_ORDERING_PSEUDO_DYNAMIC;
        case HIPRAND_ORDERING_QUASI_DEFAULT: return CURAND_ORDERING_QUASI_DEFAULT;
    }
    throw HIPRAND_STATUS_TYPE_ERROR;
}

hiprandStatus_t HIPRANDAPI
hiprandCreateGenerator(hiprandGenerator_t * generator, hiprandRngType_t rng_type)
{
    try
    {
        return to_hiprand_status(
            curandCreateGenerator(
                (curandGenerator_t *)(generator),
                to_curand_rng_type(rng_type)
            )
        );
    } catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI
hiprandCreateGeneratorHost(hiprandGenerator_t * generator, hiprandRngType_t rng_type)
{
    try
    {
        return to_hiprand_status(
            curandCreateGeneratorHost(
                (curandGenerator_t *)(generator),
                to_curand_rng_type(rng_type)
            )
        );
    } catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI
hiprandDestroyGenerator(hiprandGenerator_t generator)
{
    return to_hiprand_status(
        curandDestroyGenerator(
            (curandGenerator_t)(generator)
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerate(hiprandGenerator_t generator,
                unsigned int * output_data, size_t n)
{
    return to_hiprand_status(
        curandGenerate(
            (curandGenerator_t)(generator),
            output_data, n
        )
    );
}


hiprandStatus_t HIPRANDAPI
hiprandGenerateChar(hiprandGenerator_t generator,
                    unsigned char * output_data, size_t n)
{
    (void) generator;
    (void) output_data;
    (void) n;
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateShort(hiprandGenerator_t generator,
                     unsigned short * output_data, size_t n)
{
    (void) generator;
    (void) output_data;
    (void) n;
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t HIPRANDAPI hiprandGenerateLongLong(hiprandGenerator_t      generator,
                                                   unsigned long long int* output_data,
                                                   size_t                  n)
{
    return to_hiprand_status(
        curandGenerateLongLong(reinterpret_cast<curandGenerator_t>(generator), output_data, n));
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateUniform(hiprandGenerator_t generator,
                       float * output_data, size_t n)
{
    return to_hiprand_status(
        curandGenerateUniform(
            (curandGenerator_t)(generator),
            output_data, n
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateUniformDouble(hiprandGenerator_t generator,
                             double * output_data, size_t n)
{
    return to_hiprand_status(
        curandGenerateUniformDouble(
            (curandGenerator_t)(generator),
            output_data, n
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateUniformHalf(hiprandGenerator_t generator,
                           half * output_data, size_t n)
{
    (void) generator;
    (void) output_data;
    (void) n;
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateNormal(hiprandGenerator_t generator,
                      float * output_data, size_t n,
                      float mean, float stddev)
{
    return to_hiprand_status(
        curandGenerateNormal(
            (curandGenerator_t)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateNormalDouble(hiprandGenerator_t generator,
                            double * output_data, size_t n,
                            double mean, double stddev)
{
    return to_hiprand_status(
        curandGenerateNormalDouble(
            (curandGenerator_t)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateNormalHalf(hiprandGenerator_t generator,
                          half * output_data, size_t n,
                          half mean, half stddev)
{
    (void) generator;
    (void) output_data;
    (void) n;
    (void) mean;
    (void) stddev;
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormal(hiprandGenerator_t generator,
                         float * output_data, size_t n,
                         float mean, float stddev)
{
    return to_hiprand_status(
        curandGenerateLogNormal(
            (curandGenerator_t)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormalDouble(hiprandGenerator_t generator,
                               double * output_data, size_t n,
                               double mean, double stddev)
{
    return to_hiprand_status(
        curandGenerateLogNormalDouble(
            (curandGenerator_t)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormalHalf(hiprandGenerator_t generator,
                             half * output_data, size_t n,
                             half mean, half stddev)
{
    (void) generator;
    (void) output_data;
    (void) n;
    (void) mean;
    (void) stddev;
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t HIPRANDAPI
hiprandGeneratePoisson(hiprandGenerator_t generator,
                       unsigned int * output_data, size_t n,
                       double lambda)
{
    return to_hiprand_status(
        curandGeneratePoisson(
            (curandGenerator_t)(generator),
            output_data, n,
            lambda
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateSeeds(hiprandGenerator_t generator)
{
    return to_hiprand_status(
        curandGenerateSeeds(
            (curandGenerator_t)(generator)
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetStream(hiprandGenerator_t generator, hipStream_t stream)
{
    return to_hiprand_status(
        curandSetStream(
            (curandGenerator_t)(generator),
            stream
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator, unsigned long long seed)
{
    return to_hiprand_status(
        curandSetPseudoRandomGeneratorSeed(
            (curandGenerator_t)(generator),
            seed
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetGeneratorOffset(hiprandGenerator_t generator, unsigned long long offset)
{
    return to_hiprand_status(
        curandSetGeneratorOffset(
            (curandGenerator_t)(generator),
            offset
        )
    );
}

hiprandStatus_t HIPRANDAPI hiprandSetGeneratorOrdering(hiprandGenerator_t generator,
                                                       hiprandOrdering_t  order)
{
    try
    {
        return to_hiprand_status(
            curandSetGeneratorOrdering((curandGenerator_t)(generator), to_curand_ordering(order)));
    }
    catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI
hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator, unsigned int dimensions)
{
    return to_hiprand_status(
        curandSetQuasiRandomGeneratorDimensions(
            (curandGenerator_t)(generator),
            dimensions
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGetVersion(int * version)
{
    return to_hiprand_status(
        curandGetVersion(version)
    );
}

hiprandStatus_t HIPRANDAPI
hiprandCreatePoissonDistribution(double lambda, hiprandDiscreteDistribution_t * discrete_distribution)
{
    return to_hiprand_status(
        curandCreatePoissonDistribution(lambda, discrete_distribution)
    );
}

hiprandStatus_t HIPRANDAPI
hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution)
{
    return to_hiprand_status(
        curandDestroyDistribution(discrete_distribution)
    );
}

hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors32(hiprandDirectionVectors32_t** vectors,
                                                        hiprandDirectionVectorSet_t   set)
{
    return to_hiprand_status(
        curandGetDirectionVectors32(vectors, to_curand_direction_vector_set_type(set)));
}

hiprandStatus_t HIPRANDAPI hiprandGetDirectionVectors64(hiprandDirectionVectors64_t** vectors,
                                                        hiprandDirectionVectorSet_t   set)
{
    return to_hiprand_status(
        curandGetDirectionVectors64(vectors, to_curand_direction_vector_set_type(set)));
}

hiprandStatus_t HIPRANDAPI hiprandGetScrambleConstants32(const unsigned int** constants)
{
    return to_hiprand_status(curandGetScrambleConstants32(const_cast<unsigned int**>(constants)));
}

hiprandStatus_t HIPRANDAPI hiprandGetScrambleConstants64(const unsigned long long** constants)
{
    return to_hiprand_status(
        curandGetScrambleConstants64(const_cast<unsigned long long**>(constants)));
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
