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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include <gtest/gtest.h>
#include <stdio.h>

#include <random>
#include <vector>

#include <rng/distribution/poisson.hpp>

using namespace rocrand_impl::host;

class poisson_distribution_tests : public ::testing::TestWithParam<double>
{};

TEST_P(poisson_distribution_tests, mean_var)
{
    const double lambda = GetParam();

    std::random_device rd;
    std::mt19937       gen(rd());

    using distribution_factory_t = discrete_distribution_factory<DISCRETE_METHOD_ALIAS, true>;

    unsigned int              size;
    unsigned int              offset;
    const std::vector<double> poisson_probabilities
        = calculate_poisson_probabilities(lambda, size, offset);
    rocrand_discrete_distribution_st discrete_dist;
    ROCRAND_CHECK(
        distribution_factory_t::create(poisson_probabilities, size, offset, discrete_dist));

    poisson_distribution<DISCRETE_METHOD_ALIAS> dis(discrete_dist);

    const size_t samples_count = static_cast<size_t>(std::max(2.0, sqrt(lambda))) * 100000;
    std::vector<unsigned int> values(samples_count);

    for(size_t si = 0; si < samples_count; si++)
    {
        const unsigned int v = dis(static_cast<unsigned int>(gen()));
        values[si]           = v;
    }

    distribution_factory_t::deallocate(discrete_dist);

    const double mean     = get_mean(values);
    const double variance = get_variance(values, mean);

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-2));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-2));
}

TEST_P(poisson_distribution_tests, histogram_compare)
{
    const double lambda = GetParam();

    const unsigned int seed = std::random_device{}();
    SCOPED_TRACE(testing::Message() << "with seed = " << seed);
    std::mt19937 gen(seed);

    using distribution_factory_t = discrete_distribution_factory<DISCRETE_METHOD_ALIAS, true>;
    unsigned int              size;
    unsigned int              offset;
    const std::vector<double> poisson_probabilities
        = calculate_poisson_probabilities(lambda, size, offset);
    rocrand_discrete_distribution_st discrete_dist;
    ROCRAND_CHECK(
        distribution_factory_t::create(poisson_probabilities, size, offset, discrete_dist));

    poisson_distribution<DISCRETE_METHOD_ALIAS> dis(discrete_dist);

    const size_t samples_count = static_cast<size_t>(std::max(2.0, sqrt(lambda))) * 100000;
    const size_t bin_size      = static_cast<size_t>(std::max(2.0, sqrt(lambda)));
    const size_t bins_count    = static_cast<size_t>((2.0 * lambda + 10.0) / bin_size);
    std::vector<unsigned int> histogram_rocrand(bins_count);

    for(size_t si = 0; si < samples_count; si++)
    {
        const unsigned int v   = dis(static_cast<unsigned int>(gen()));
        const size_t       bin = v / bin_size;
        if(bin < bins_count)
        {
            histogram_rocrand[bin]++;
        }
    }

    distribution_factory_t::deallocate(discrete_dist);

    // for small lambda, histogram test is inaccurate due to relatively large bins
    // for large lambda, expected value calculation is inaccurate due to non-finite terms
    if(lambda <= 50.0)
    {
        for(size_t bi = 0; bi < bins_count; bi++)
        {
            const unsigned int h   = histogram_rocrand[bi];
            double             tmp = 0.0;
            for(size_t i = 0; i < bin_size; ++i)
            {
                const int k = bi * bin_size + i;
                tmp += std::pow(lambda, k) * std::exp(-lambda) / std::tgamma(k + 1.0);
            }
            const unsigned int actual = std::roundl(samples_count * tmp);

            // Very loose comparison
            EXPECT_NEAR(h, actual, std::max(samples_count * 1e-3, actual * 1e-1));
        }
    }
    else
    {
        std::poisson_distribution<unsigned int> host_dis(lambda);
        std::vector<unsigned int>               histogram_stl(bins_count);
        for(size_t si = 0; si < samples_count; si++)
        {
            const unsigned int v   = host_dis(gen);
            const size_t       bin = v / bin_size;
            if(bin < bins_count)
            {
                histogram_stl[bin]++;
            }
        }

        // Very loose comparison
        for(size_t bi = 0; bi < bins_count; bi++)
        {
            const unsigned int h0 = histogram_rocrand[bi];
            const unsigned int h1 = histogram_stl[bi];
            EXPECT_NEAR(h0, h1, std::max(samples_count * 1e-3, std::max(h0, h1) * 1e-1));
        }
    }
}

const double lambdas[] = {1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0};

INSTANTIATE_TEST_SUITE_P(poisson_distribution_tests,
                         poisson_distribution_tests,
                         ::testing::ValuesIn(lambdas));
