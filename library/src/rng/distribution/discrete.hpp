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

#ifndef ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_
#define ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_

#include "../common.hpp" // IWYU pragma: keep

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_discrete.h>
#include <rocrand/rocrand_discrete_types.h>

#include <algorithm>
#include <iterator>
#include <vector>

// Alias method
//
// Walker, A. J.
// An Efficient Method for Generating Discrete Random Variables with General Distributions, 1977
//
// Vose M. D.
// A Linear Algorithm For Generating Random Numbers With a Given Distribution, 1991

namespace rocrand_impl::host
{

enum discrete_method
{
    DISCRETE_METHOD_ALIAS     = 1,
    DISCRETE_METHOD_CDF       = 2,
    DISCRETE_METHOD_UNIVERSAL = DISCRETE_METHOD_ALIAS | DISCRETE_METHOD_CDF
};

/// \brief Encapsulates a `rocrand_discrete_distribution_st` and makes it possible
/// to sample the discrete distribution in the host generators.
template<discrete_method Method>
class discrete_distribution_base
{
public:
    static constexpr inline unsigned int input_width  = 1;
    static constexpr inline unsigned int output_width = 1;

    // rocrand_discrete_distribution_st is a struct
    explicit discrete_distribution_base(const rocrand_discrete_distribution_st& distribution)
        : m_distribution(distribution)
    {}

    template<class T>
    __forceinline__ __host__ __device__ unsigned int operator()(T x) const
    {
        if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
        {
            return rocrand_device::detail::discrete_alias(x, m_distribution);
        }
        else
        {
            return rocrand_device::detail::discrete_cdf(x, m_distribution);
        }
    }

    template<class T>
    __forceinline__ __host__ __device__
    void operator()(const T (&input)[1], unsigned int output[1]) const
    {
        output[0] = (*this)(input[0]);
    }

private:
    rocrand_discrete_distribution_st m_distribution;
};

/// \brief A collection of static methods for constructing and destroying
/// instances of `rocrand_discrete_distribution_st`.
/// \tparam Method Controls which members of the produced `rocrand_discrete_distribution_st`
/// are populated.
/// \tparam IsHostSide Controls whether the allocated and filled memory blocks reside
/// on the host or on the device.
template<discrete_method Method, bool IsHostSide = false>
class discrete_distribution_factory
{
public:
    /// \brief Allocates and populates an instance of `rocrand_discrete_distribution_st`.
    /// \note `allocate` and `normalize` are called by this function, therefore those
    /// doesn't need to be called separately.
    /// \note The produced `rocrand_discrete_distribution_st` MUST be deallocated by the matching
    /// `deallocate` function when it's no longer used.
    /// \param p The probability array of the discrete distribution.
    /// \param size The size of the input probability array. This must not exceed the size of `p`.
    /// \param offset The offset of the input probability array.
    /// \param distribution [out] The allocated and populated discrete distribution instance.
    /// \return `ROCRAND_STATUS_SUCCESS` if the operation is successful, otherwise an error code from the
    /// first failing procedure.
    static rocrand_status create(std::vector<double>               p,
                                 const unsigned int                size,
                                 const unsigned int                offset,
                                 rocrand_discrete_distribution_st& distribution)
    {
        rocrand_status status = allocate(size, offset, distribution);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        normalize(p, size);
        if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
        {
            std::vector<double>       h_probability(size);
            std::vector<unsigned int> h_alias(size);
            create_alias_table(p, size, h_probability.begin(), h_alias.begin());
            status = copy_alias_table(distribution, h_probability, h_alias);
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
        {
            std::vector<double> h_cdf(size);
            create_cdf(p, size, h_cdf.begin());
            status = copy_cdf(distribution, h_cdf);
            if(status != ROCRAND_STATUS_SUCCESS)
            {
                return status;
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    /// \brief Frees the allocated memory associated with the passed distribution that was
    /// previously created by `create` or `allocate`.
    /// \param [in,out] distribution The distribution to deallocate.
    /// The fields of the distribution are set to default values.
    /// \return `ROCRAND_STATUS_SUCCESS` if the operation is successful, otherwise an error code from the
    /// first failing procedure.
    static rocrand_status deallocate(rocrand_discrete_distribution_st& distribution)
    {
        if constexpr(IsHostSide)
        {
            delete[] distribution.probability;
            delete[] distribution.alias;
            delete[] distribution.cdf;
        }
        else
        {
            hipError_t error;
            error = hipFree(distribution.probability);
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
            error = hipFree(distribution.alias);
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
            error = hipFree(distribution.cdf);
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }

        distribution = {};
        return ROCRAND_STATUS_SUCCESS;
    }

    /// \brief Normalizes the values in probability vector `p`.
    /// \param p [in,out] p The probability vector to normalize.
    /// \param size The size of the probability vector.
    /// It MUST NOT be larger than the size of `p`.
    static void normalize(std::vector<double>& p, const unsigned int size)
    {
        double sum = 0.0;
        for(unsigned int i = 0; i < size; i++)
        {
            sum += p[i];
        }
        for(unsigned int i = 0; i < size; i++)
        {
            p[i] /= sum;
        }
    }

    /// \brief Computes the alias table from the probability vector for a discrete distribution.
    /// \tparam ProbabilityIt The type of the output iterator to which the calculated probabilities are written.
    /// Must be a RandomAccessIterator.
    /// \tparam AliasIt The type of the output iterator to which the calculated aliases are written.
    /// Must be a RandomAccessIterator.
    /// \param p The normalized probability vector.
    /// \param size The size of the probability vector.
    /// It MUST NOT be larger than the size of `p`.
    /// \param h_probability Probabilities output iterator.
    /// \param h_alias Aliases output iterator.
    template<class ProbabilityIt, class AliasIt>
    static void create_alias_table(std::vector<double> p,
                                   const unsigned int  size,
                                   ProbabilityIt       h_probability,
                                   AliasIt             h_alias)
    {
        static_assert(
            std::is_same_v<double, typename std::iterator_traits<ProbabilityIt>::value_type>);
        static_assert(
            std::is_same_v<unsigned int, typename std::iterator_traits<AliasIt>::value_type>);

        const double average = 1.0 / size;

        std::vector<unsigned int> small;
        std::vector<unsigned int> large;

        small.reserve(size);
        large.reserve(size);

        for (unsigned int i = 0; i < size; i++)
        {
            if (p[i] >= average)
                large.push_back(i);
            else
                small.push_back(i);
        }

        while (!small.empty() && !large.empty())
        {
            const unsigned int less = small.back();
            small.pop_back();
            const unsigned int more = large.back();
            large.pop_back();

            h_probability[less] = p[less] * size;
            h_alias[less] = more;

            p[more] = (p[more] + p[less]) - average;

            if (p[more] >= average)
                large.push_back(more);
            else
                small.push_back(more);
        }

        for (unsigned int i : small)
        {
            h_probability[i] = 1.0;
        }
        for (unsigned int i : large)
        {
            h_probability[i] = 1.0;
        }
    }

    /// \brief Computes the CDF (cumulative distribution function) table from the
    /// probability vector for a discrete distribution.
    /// \tparam CdfIt The type of the output iterator to which the calculated CDF values are written.
    /// Must be a RandomAccessIterator.
    /// \param p The normalized probability vector.
    /// \param size The size of the probability vector.
    /// It MUST NOT be larger than the size of `p`.
    /// \param h_cdf CDF output iterator.
    template<class CdfIt>
    static void create_cdf(const std::vector<double>& p, const unsigned int size, CdfIt h_cdf)
    {
        static_assert(std::is_same_v<double, typename std::iterator_traits<CdfIt>::value_type>);

        double sum = 0.0;
        for(unsigned int i = 0; i < size; i++)
        {
            sum += p[i];
            h_cdf[i] = sum;
        }
    }

    /// \brief Allocates the required amount of memory for a `rocrand_discrete_distribution_st`.
    /// \param size The size of the input probability array.
    /// \param offset The offset of the input probability array.
    /// \param [out] distribution The distribution to allocate.
    /// \return `ROCRAND_STATUS_SUCCESS` if the operation is successful, otherwise an error code from the
    /// first failing procedure.
    static rocrand_status allocate(const unsigned int                size,
                                   const unsigned int                offset,
                                   rocrand_discrete_distribution_st& distribution)
    {
        distribution        = {};
        distribution.size   = size;
        distribution.offset = offset;
        if constexpr(IsHostSide)
        {
            if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
            {
                distribution.probability = new double[distribution.size];
                distribution.alias       = new unsigned int[distribution.size];
            }
            if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
            {
                distribution.cdf = new double[distribution.size];
            }
        }
        else
        {
            hipError_t error;
            if constexpr((Method & DISCRETE_METHOD_ALIAS) != 0)
            {
                error = hipMalloc(&distribution.probability, sizeof(double) * distribution.size);
                if(error != hipSuccess)
                {
                    return ROCRAND_STATUS_ALLOCATION_FAILED;
                }
                error = hipMalloc(&distribution.alias, sizeof(unsigned int) * distribution.size);
                if(error != hipSuccess)
                {
                    return ROCRAND_STATUS_ALLOCATION_FAILED;
                }
            }
            if constexpr((Method & DISCRETE_METHOD_CDF) != 0)
            {
                error = hipMalloc(&distribution.cdf, sizeof(double) * distribution.size);
                if(error != hipSuccess)
                {
                    return ROCRAND_STATUS_ALLOCATION_FAILED;
                }
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }

private:
    static rocrand_status copy_alias_table(const rocrand_discrete_distribution_st& distribution,
                                           const std::vector<double>&              h_probability,
                                           const std::vector<unsigned int>&        h_alias)
    {
        if constexpr(IsHostSide)
        {
            std::copy(h_probability.begin(), h_probability.end(), distribution.probability);
            std::copy(h_alias.begin(), h_alias.end(), distribution.alias);
        }
        else
        {
            hipError_t error;
            error = hipMemcpy(distribution.probability,
                              h_probability.data(),
                              sizeof(double) * distribution.size,
                              hipMemcpyHostToDevice);
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
            error = hipMemcpy(distribution.alias,
                              h_alias.data(),
                              sizeof(unsigned int) * distribution.size,
                              hipMemcpyHostToDevice);
            if(error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    static rocrand_status copy_cdf(const rocrand_discrete_distribution_st& distribution,
                                   const std::vector<double>&              h_cdf)
    {
        if constexpr(IsHostSide)
        {
            std::copy(h_cdf.begin(), h_cdf.end(), distribution.cdf);
        }
        else
        {
            hipError_t error;
            error = hipMemcpy(distribution.cdf,
                              h_cdf.data(),
                              sizeof(double) * distribution.size,
                              hipMemcpyHostToDevice);
            if (error != hipSuccess)
            {
                return ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
        return ROCRAND_STATUS_SUCCESS;
    }
};

} // namespace rocrand_impl::host

#endif // ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_
