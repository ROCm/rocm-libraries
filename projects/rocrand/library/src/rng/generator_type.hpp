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

#ifndef ROCRAND_RNG_GENERATOR_TYPE_H_
#define ROCRAND_RNG_GENERATOR_TYPE_H_

#include <rocrand/rocrand.h>

#include <hip/hip_runtime_api.h>
#include <hip/hip_vector_types.h>

struct rocrand_generator_base_type
{
    rocrand_generator_base_type() = default;

    virtual ~rocrand_generator_base_type() = default;

    virtual rocrand_rng_type type() const = 0;

    virtual unsigned long long get_seed() const                  = 0;
    virtual void               set_seed(unsigned long long seed) = 0;
    virtual rocrand_status     set_seed_uint4(uint4 seed)        = 0;

    virtual unsigned long long get_offset() const                    = 0;
    virtual rocrand_status     set_offset(unsigned long long offset) = 0;

    virtual rocrand_ordering get_order() const                 = 0;
    virtual rocrand_status   set_order(rocrand_ordering order) = 0;

    virtual hipStream_t get_stream() const             = 0;
    virtual rocrand_status set_stream(hipStream_t stream) = 0;

    virtual rocrand_status set_dimensions(unsigned int dimensions) = 0;

    virtual rocrand_status init() = 0;

    // clang-format off
    virtual rocrand_status generate_char(unsigned char* output_data, size_t n) = 0;
    virtual rocrand_status generate_short(unsigned short* output_data, size_t n) = 0;
    virtual rocrand_status generate_int(unsigned int* output_data, size_t n) = 0;
    virtual rocrand_status generate_long(unsigned long long int* output_data, size_t n) = 0;

    virtual rocrand_status generate_uniform_half(half* output_data, size_t n) = 0;
    virtual rocrand_status generate_uniform_float(float* output_data, size_t n) = 0;
    virtual rocrand_status generate_uniform_double(double* output_data, size_t n) = 0;

    virtual rocrand_status generate_normal_half(half* output_data, size_t n, half mean, half stddev) = 0;
    virtual rocrand_status generate_normal_float(float* output_data, size_t n, float mean, float stddev) = 0;
    virtual rocrand_status generate_normal_double(double* output_data, size_t n, double mean, double stddev) = 0;

    virtual rocrand_status generate_log_normal_half(half* output_data, size_t n, half mean, half stddev) = 0;
    virtual rocrand_status generate_log_normal_float(float* output_data, size_t n, float mean, float stddev) = 0;
    virtual rocrand_status generate_log_normal_double(double* output_data, size_t n, double mean, double stddev) = 0;

    virtual rocrand_status generate_poisson(unsigned int* output_data, size_t n, double lambda) = 0;
    // clang-format on
};

namespace rocrand_impl::host
{

/// This wrapper provides support for the different types of generator functions,
/// while calling into a generic function in the actual generator implementation.
/// This saves us to write the same code for the distributions every time.
template<typename Generator>
struct generator_type : rocrand_generator_base_type
{
    generator_type() : m_generator() {}

    rocrand_rng_type type() const override final
    {
        return Generator::type();
    }

    unsigned long long get_seed() const override final
    {
        return m_generator.get_seed();
    }

    void set_seed(unsigned long long seed) override final
    {
        m_generator.set_seed(seed);
    }

    rocrand_status set_seed_uint4(uint4 seed) override final
    {
        return m_generator.set_seed_uint4(seed);
    }

    unsigned long long get_offset() const override final
    {
        return m_generator.get_offset();
    }

    rocrand_status set_offset(unsigned long long offset) override final
    {
        return m_generator.set_offset(offset);
    }

    rocrand_ordering get_order() const override final
    {
        return m_generator.get_order();
    }

    rocrand_status set_order(rocrand_ordering order) override final
    {
        return m_generator.set_order(order);
    }

    hipStream_t get_stream() const override final
    {
        return m_generator.get_stream();
    }

    rocrand_status set_stream(hipStream_t stream) override final
    {
        return m_generator.set_stream(stream);
    }

    rocrand_status set_dimensions(unsigned int dimensions) override final
    {
        return m_generator.set_dimensions(dimensions);
    }

    rocrand_status init() override final
    {
        return m_generator.init();
    }

    rocrand_status generate_char(unsigned char* output_data, size_t n) override final
    {
        return m_generator.generate(output_data, n);
    }

    rocrand_status generate_short(unsigned short* output_data, size_t n) override final
    {
        return m_generator.generate(output_data, n);
    }

    rocrand_status generate_int(unsigned int* output_data, size_t n) override final
    {
        return m_generator.generate(output_data, n);
    }

    rocrand_status generate_long(unsigned long long int* output_data, size_t n) override final
    {
        return m_generator.generate(output_data, n);
    }

    rocrand_status generate_uniform_half(half* output_data, size_t n) override final
    {
        return m_generator.generate_uniform(output_data, n);
    }

    rocrand_status generate_uniform_float(float* output_data, size_t n) override final
    {
        return m_generator.generate_uniform(output_data, n);
    }

    rocrand_status generate_uniform_double(double* output_data, size_t n) override final
    {
        return m_generator.generate_uniform(output_data, n);
    }

    rocrand_status
        generate_normal_half(half* output_data, size_t n, half mean, half stddev) override final
    {
        return m_generator.generate_normal(output_data, n, mean, stddev);
    }

    rocrand_status
        generate_normal_float(float* output_data, size_t n, float mean, float stddev) override final
    {
        return m_generator.generate_normal(output_data, n, mean, stddev);
    }

    rocrand_status generate_normal_double(double* output_data,
                                          size_t  n,
                                          double  mean,
                                          double  stddev) override final
    {
        return m_generator.generate_normal(output_data, n, mean, stddev);
    }

    rocrand_status
        generate_log_normal_half(half* output_data, size_t n, half mean, half stddev) override final
    {
        return m_generator.generate_log_normal(output_data, n, mean, stddev);
    }

    rocrand_status generate_log_normal_float(float* output_data,
                                             size_t n,
                                             float  mean,
                                             float  stddev) override final
    {
        return m_generator.generate_log_normal(output_data, n, mean, stddev);
    }
    rocrand_status generate_log_normal_double(double* output_data,
                                              size_t  n,
                                              double  mean,
                                              double  stddev) override final
    {
        return m_generator.generate_log_normal(output_data, n, mean, stddev);
    }

    rocrand_status
        generate_poisson(unsigned int* output_data, size_t n, double lambda) override final
    {
        return m_generator.generate_poisson(output_data, n, lambda);
    }

private:
    Generator m_generator;
};

/// \brief This type provides some default implementations for the methods
/// that are required by the `Generator` parameter of `rocrand_generator`.
/// It can be used, but it is not required. It only exists as utility.
struct generator_impl_base
{
    generator_impl_base(rocrand_ordering order, unsigned long long offset, hipStream_t stream)
        : m_order(order), m_offset(offset), m_stream(stream)
    {}

    virtual ~generator_impl_base() = default;

    virtual void reset() = 0;

    virtual rocrand_status set_seed_uint4(uint4 seed)
    {
        // For most generators, we can only set the seed as long long.
        // Generators that have support for this should override this method.
        (void)seed;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    unsigned long long get_offset() const
    {
        return m_offset;
    }

    rocrand_status set_offset(unsigned long long offset)
    {
        m_offset = offset;
        reset();
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_ordering get_order() const
    {
        return m_order;
    }

    hipStream_t get_stream() const
    {
        return m_stream;
    }

    void set_stream(hipStream_t stream)
    {
        m_stream = stream;
    }

    virtual rocrand_status set_dimensions(unsigned int dimensions)
    {
        // This method should be overridden for generators that support it.
        (void)dimensions;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

protected:
    rocrand_ordering   m_order;
    unsigned long long m_offset;
    hipStream_t        m_stream;
};

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_GENERATOR_TYPE_H_
