// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rng/distribution/discrete.hpp"
#include "rng/distribution/poisson.hpp"
#include "rng/generator_type.hpp"
#include "rng/generator_types.hpp"

#include <new>
#include <rocrand/rocrand.h>

template<bool UseHostFunc>
rocrand_status create_generator_host(rocrand_generator* generator, rocrand_rng_type rng_type)
{
    using namespace rocrand_impl::host;
    try
    {
        // clang-format off
        switch(rng_type)
        {
            case ROCRAND_RNG_PSEUDO_LFSR113:
                *generator = new generator_type<lfsr113_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_PHILOX4_32_10:
                *generator = new generator_type<philox4x32_10_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG31K3P:
                *generator = new generator_type<mrg31k3p_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG32K3A:
                *generator = new generator_type<mrg32k3a_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20:
                *generator = new generator_type<threefry2x32_20_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20:
                *generator = new generator_type<threefry2x64_20_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20:
                *generator = new generator_type<threefry4x32_20_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20:
                *generator = new generator_type<threefry4x64_20_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_QUASI_DEFAULT:
            case ROCRAND_RNG_QUASI_SOBOL32:
                *generator = new generator_type<sobol32_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
                *generator = new generator_type<scrambled_sobol32_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_QUASI_SOBOL64:
                *generator = new generator_type<sobol64_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
                *generator = new generator_type<scrambled_sobol64_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_DEFAULT:
            case ROCRAND_RNG_PSEUDO_XORWOW:
                *generator = new generator_type<xorwow_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_MTGP32:
                *generator = new generator_type<mtgp32_generator_host<UseHostFunc>>();
                break;
            case ROCRAND_RNG_PSEUDO_MT19937:
                *generator = new generator_type<mt19937_generator_host<UseHostFunc>>();
                break;
            default:
                return ROCRAND_STATUS_TYPE_ERROR;
        }
        // clang-format on
    }
    catch(const std::bad_alloc& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status create_generator_host(rocrand_generator* generator,
                                     rocrand_rng_type   rng_type,
                                     bool               use_host_func)
{
    return use_host_func ? create_generator_host<true>(generator, rng_type)
                         : create_generator_host<false>(generator, rng_type);
}

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

rocrand_status ROCRANDAPI rocrand_create_generator(rocrand_generator* generator,
                                                   rocrand_rng_type   rng_type)
{
    using namespace rocrand_impl::host;
    try
    {
        // clang-format off
        switch(rng_type)
        {
            case ROCRAND_RNG_PSEUDO_PHILOX4_32_10:
                *generator = new generator_type<philox4x32_10_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG31K3P:
                *generator = new generator_type<mrg31k3p_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG32K3A:
                *generator = new generator_type<mrg32k3a_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_DEFAULT:
            case ROCRAND_RNG_PSEUDO_XORWOW:
                *generator = new generator_type<xorwow_generator>();
                break;
            case ROCRAND_RNG_QUASI_DEFAULT:
            case ROCRAND_RNG_QUASI_SOBOL32:
                *generator = new generator_type<sobol32_generator>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
                *generator = new generator_type<scrambled_sobol32_generator>();
                break;
            case ROCRAND_RNG_QUASI_SOBOL64:
                *generator = new generator_type<sobol64_generator>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
                *generator = new generator_type<scrambled_sobol64_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_MTGP32:
                *generator = new generator_type<mtgp32_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_LFSR113:
                *generator = new generator_type<lfsr113_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_MT19937:
                *generator = new generator_type<mt19937_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20:
                *generator = new generator_type<threefry2x32_20_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20:
                *generator = new generator_type<threefry2x64_20_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20:
                *generator = new generator_type<threefry4x32_20_generator>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20:
                *generator = new generator_type<threefry4x64_20_generator>();
                break;
            default:
                return ROCRAND_STATUS_TYPE_ERROR;
        }
        // clang-format on
    }
    catch(const std::bad_alloc& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_create_generator_host(rocrand_generator* generator,
                                                        rocrand_rng_type   rng_type)
{
    return create_generator_host(generator, rng_type, true);
}

rocrand_status ROCRANDAPI rocrand_create_generator_host_blocking(rocrand_generator* generator,
                                                                 rocrand_rng_type   rng_type)
{
    return create_generator_host(generator, rng_type, false);
}

rocrand_status ROCRANDAPI rocrand_destroy_generator(rocrand_generator generator)
{
    try
    {
        delete generator;
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_generate(rocrand_generator generator,
                                           unsigned int*     output_data,
                                           size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_int(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_long_long(rocrand_generator       generator,
                                                     unsigned long long int* output_data,
                                                     size_t                  n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_long(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_char(rocrand_generator generator,
                                                unsigned char*    output_data,
                                                size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_char(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_short(rocrand_generator generator,
                                                 unsigned short*   output_data,
                                                 size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_short(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_uniform(rocrand_generator generator,
                                                   float*            output_data,
                                                   size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_uniform_float(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_uniform_double(rocrand_generator generator,
                                                          double*           output_data,
                                                          size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_uniform_double(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_uniform_half(rocrand_generator generator,
                                                        half*             output_data,
                                                        size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_uniform_half(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_normal(
    rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_normal_float(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_normal_double(
    rocrand_generator generator, double* output_data, size_t n, double mean, double stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_normal_double(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_normal_half(
    rocrand_generator generator, half* output_data, size_t n, half mean, half stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_normal_half(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_log_normal(
    rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_log_normal_float(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_log_normal_double(
    rocrand_generator generator, double* output_data, size_t n, double mean, double stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_log_normal_double(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_log_normal_half(
    rocrand_generator generator, half* output_data, size_t n, half mean, half stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_log_normal_half(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_poisson(rocrand_generator generator,
                                                   unsigned int*     output_data,
                                                   size_t            n,
                                                   double            lambda)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }
    if(lambda <= 0.0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    return generator->generate_poisson(output_data, n, lambda);
}

rocrand_status ROCRANDAPI rocrand_initialize_generator(rocrand_generator generator)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->init();
}

rocrand_status ROCRANDAPI rocrand_set_stream(rocrand_generator generator, hipStream_t stream)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_stream(stream);
}

rocrand_status ROCRANDAPI rocrand_set_seed(rocrand_generator generator, unsigned long long seed)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    generator->set_seed(seed);
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_set_seed_uint4(rocrand_generator generator, uint4 seed)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_seed_uint4(seed);
}

rocrand_status ROCRANDAPI rocrand_set_offset(rocrand_generator generator, unsigned long long offset)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_offset(offset);
}

rocrand_status ROCRANDAPI rocrand_set_ordering(rocrand_generator generator, rocrand_ordering order)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_order(order);
}

rocrand_status ROCRANDAPI rocrand_set_quasi_random_generator_dimensions(rocrand_generator generator,
                                                                        unsigned int dimensions)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_dimensions(dimensions);
}

rocrand_status ROCRANDAPI rocrand_get_version(int* version)
{
    if(version == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    *version = ROCRAND_VERSION;
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_create_poisson_distribution(
    double lambda, rocrand_discrete_distribution* discrete_distribution)
{
    using namespace rocrand_impl::host;
    if(discrete_distribution == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }
    if(lambda <= 0.0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    unsigned int              size;
    unsigned int              offset;
    const std::vector<double> poisson_probabilities
        = calculate_poisson_probabilities(lambda, size, offset);

    rocrand_discrete_distribution_st h_dis;
    rocrand_status                   status
        = discrete_distribution_factory<DISCRETE_METHOD_UNIVERSAL>::create(poisson_probabilities,
                                                                           size,
                                                                           offset,
                                                                           h_dis);
    if(status != ROCRAND_STATUS_SUCCESS)
    {
        return status;
    }

    hipError_t error = hipMalloc(discrete_distribution, sizeof(rocrand_discrete_distribution_st));
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_ALLOCATION_FAILED;
    }
    error = hipMemcpy(*discrete_distribution,
                      &h_dis,
                      sizeof(rocrand_discrete_distribution_st),
                      hipMemcpyDefault);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI
    rocrand_create_discrete_distribution(const double*                  probabilities,
                                         unsigned int                   size,
                                         unsigned int                   offset,
                                         rocrand_discrete_distribution* discrete_distribution)
{
    using namespace rocrand_impl::host;
    if(discrete_distribution == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }
    if(size == 0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    rocrand_discrete_distribution_st h_dis;
    rocrand_status status = discrete_distribution_factory<DISCRETE_METHOD_UNIVERSAL>::create(
        std::vector<double>(probabilities, probabilities + size),
        size,
        offset,
        h_dis);
    if(status != ROCRAND_STATUS_SUCCESS)
    {
        return status;
    }

    hipError_t error;
    error = hipMalloc(discrete_distribution, sizeof(rocrand_discrete_distribution_st));
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_ALLOCATION_FAILED;
    }
    error = hipMemcpy(*discrete_distribution,
                      &h_dis,
                      sizeof(rocrand_discrete_distribution_st),
                      hipMemcpyDefault);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI
    rocrand_destroy_discrete_distribution(rocrand_discrete_distribution discrete_distribution)
{
    using namespace rocrand_impl::host;
    if(discrete_distribution == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    rocrand_discrete_distribution_st h_dis;

    hipError_t error;
    error = hipMemcpy(&h_dis,
                      discrete_distribution,
                      sizeof(rocrand_discrete_distribution_st),
                      hipMemcpyDefault);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    const rocrand_status status
        = discrete_distribution_factory<DISCRETE_METHOD_UNIVERSAL>::deallocate(h_dis);
    if(status != ROCRAND_STATUS_SUCCESS)
    {
        return status;
    }

    error = hipFree(discrete_distribution);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_get_direction_vectors32(const unsigned int**         vectors,
                                                          rocrand_direction_vector_set set)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    switch(set)
    {
        case ROCRAND_DIRECTION_VECTORS_32_JOEKUO6:
            *vectors = rocrand_h_sobol32_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        case ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            *vectors = rocrand_h_scrambled_sobol32_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        default: return ROCRAND_STATUS_OUT_OF_RANGE;
    }
#pragma clang diagnostic pop
}

rocrand_status ROCRANDAPI rocrand_get_direction_vectors64(const unsigned long long**   vectors,
                                                          rocrand_direction_vector_set set)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    switch(set)
    {
        case ROCRAND_DIRECTION_VECTORS_64_JOEKUO6:
            *vectors = rocrand_h_sobol64_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        case ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            *vectors = rocrand_h_scrambled_sobol64_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        default: return ROCRAND_STATUS_OUT_OF_RANGE;
    }
#pragma clang diagnostic pop
}

rocrand_status ROCRANDAPI rocrand_get_scramble_constants32(const unsigned int** constants)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    *constants = h_scrambled_sobol32_constants;
    return ROCRAND_STATUS_SUCCESS;
#pragma clang diagnostic pop
}

rocrand_status ROCRANDAPI rocrand_get_scramble_constants64(const unsigned long long** constants)
{
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    *constants = h_scrambled_sobol64_constants;
    return ROCRAND_STATUS_SUCCESS;
#pragma clang diagnostic pop
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
