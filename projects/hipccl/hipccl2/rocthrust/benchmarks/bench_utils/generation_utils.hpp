// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef ROCTHRUST_BENCHMARKS_BENCH_UTILS_GENERATION_UTILS_HPP_
#define ROCTHRUST_BENCHMARKS_BENCH_UTILS_GENERATION_UTILS_HPP_

// Utils
#include "common/types.hpp"

// Thrust
#include <thrust/copy.h>
#include <thrust/detail/config.h>
#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

// rocPRIM
#include <rocprim/rocprim.hpp>

// rocRAND
#include <rocrand/rocrand.h>

// Google Benchmark
#include <benchmark/benchmark.h>

// STL
#include <algorithm>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <type_traits>

namespace bench_utils
{
/// \brief Provides a sequence of seeds.
class managed_seed
{
public:
    /// \param[in] seed_string Either "random" to get random seeds,
    ///   or an unsigned integer to get (a sequence) of deterministic seeds.
    managed_seed(const std::string& seed_string)
    {
        is_random = seed_string == "random";
        if(!is_random)
        {
            const unsigned int seed = std::stoul(seed_string);
            std::seed_seq      seq {seed};
            seq.generate(seeds.begin(), seeds.end());
        }
    }

    managed_seed()
        : managed_seed("random") {};

    unsigned int get_0() const
    {
        return is_random ? std::random_device {}() : seeds[0];
    }

    unsigned int get_1() const
    {
        return is_random ? std::random_device {}() : seeds[1];
    }

    unsigned int get_2() const
    {
        return is_random ? std::random_device {}() : seeds[2];
    }

private:
    std::array<unsigned int, 3> seeds;
    bool                        is_random;
};

float get_entropy_percentage(int entropy_reduction)
{
    switch(entropy_reduction)
    {
    case 0:
        return 100.0;
    case 1:
        return 81.1;
    case 2:
        return 54.4;
    case 3:
        return 33.7;
    case 4:
        return 20.1;
    default:
        return 0;
    }
}

template <typename T>
T value_from_entropy(float64_t percentage)
{
    if(percentage == 100.0)
    {
        return std::numeric_limits<T>::max();
    }

    percentage /= 100; // convert percentage to per one

    // Select value from the line between the lowest and the highest representable
    // values of type T based on the entropy value.
    const auto max_val = static_cast<double>(std::numeric_limits<T>::max());
    const auto min_val = static_cast<double>(std::numeric_limits<T>::lowest());
    const auto result  = min_val + percentage * (max_val - min_val);
    return static_cast<T>(result);
}

const int entropy_reductions[] = {0, 2, 4, 6};

namespace detail
{
    // std::uniform_int_distribution is undefined for anything other than:
    // short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
    template <typename T>
    struct is_valid_for_int_distribution
        : std::integral_constant<
              bool,
              std::is_same<short, T>::value || std::is_same<unsigned short, T>::value
                  || std::is_same<int, T>::value || std::is_same<unsigned int, T>::value
                  || std::is_same<long, T>::value || std::is_same<unsigned long, T>::value
                  || std::is_same<long long, T>::value
                  || std::is_same<unsigned long long, T>::value>
    {
    };

    template <class T, class Enable = void>
    struct random_to_item_t
    {
    };

    // Floating point types
    template <typename T>
    struct random_to_item_t<T, typename std::enable_if<std::is_floating_point<T>::value>::type>
    {
        double m_min;
        double m_max;

        __host__ __device__ random_to_item_t(T min, T max)
            : m_min(static_cast<double>(min))
            , m_max(static_cast<double>(max))
        {
        }

        __host__ __device__ auto operator()(double random_value) const
        {
            return static_cast<T>((m_max - m_min) * random_value + m_min);
        }
    };

    // Integral types
    template <typename T>
    struct random_to_item_t<T, typename std::enable_if<!std::is_floating_point<T>::value>::type>
    {
#if THRUST_BENCHMARKS_HAVE_INT128_SUPPORT
        using CastT = typename std::conditional<
            std::is_same<T, int128_t>::value || std::is_same<T, uint128_t>::value,
            typename std::conditional<std::is_signed<T>::value, long, unsigned long>::type,
            T>::type;
#else
        using CastT = typename std::conditional<std::is_signed<T>::value, long, unsigned long>::type;
#endif


        double m_min;
        double m_max;

        __host__ __device__ random_to_item_t(T min, T max)
            : m_min(static_cast<double>(min))
            , m_max(static_cast<double>(max))
        {
        }

        __host__ __device__ auto operator()(double random_value) const
        {
            return static_cast<CastT>(floor((m_max - m_min + 1) * random_value + m_min));
        }
    };

    struct and_t
    {
        template <class T>
        __host__ __device__ T operator()(T a, T b) const
        {
            return a & b;
        }

        __host__ __device__ float operator()(float a, float b) const
        {
            const std::uint32_t result
                = reinterpret_cast<std::uint32_t&>(a) & reinterpret_cast<std::uint32_t&>(b);
            return reinterpret_cast<const float&>(result);
        }

        __host__ __device__ double operator()(double a, double b) const
        {
            const std::uint64_t result
                = reinterpret_cast<std::uint64_t&>(a) & reinterpret_cast<std::uint64_t&>(b);
            return reinterpret_cast<const double&>(result);
        }
    };

    template <class T>
    struct geq_t
    {
        T val;

        __host__ __device__ bool operator()(T x)
        {
            return x >= val;
        }
    };

    template <class T>
    class value_wrapper_t
    {
        T m_val {};

    public:
        explicit value_wrapper_t(T val)
            : m_val(val)
        {
        }

        T get() const
        {
            return m_val;
        }

        value_wrapper_t& operator++()
        {
            m_val++;
            return *this;
        }
    };

    class seed_t : public value_wrapper_t<unsigned long long int>
    {
    public:
        using value_wrapper_t::value_wrapper_t;
        using value_wrapper_t::operator++;

        seed_t()
            : value_wrapper_t(42)
        {
        }
    };

    struct device_generator_base_t
    {
        const std::size_t elements {0};
        const std::string seed_type {"random"};
        seed_t            seed {};
        const int         entropy_reduction {0 /*bit_entropy::_1_000*/};

        device_generator_base_t(std::size_t        m_elements,
                                const std::string& m_seed_type,
                                int                m_entropy_reduction)
            : elements(m_elements)
            , seed_type(m_seed_type)
            , entropy_reduction(m_entropy_reduction)
        {
            rocrand_create_generator(&gen, ROCRAND_RNG_PSEUDO_DEFAULT);
            const managed_seed managed_seed {seed_type};
            seed = seed_t {managed_seed.get_0()};
        }

        ~device_generator_base_t()
        {
            rocrand_destroy_generator(gen);
        }

        template <typename T>
        thrust::device_vector<T> generate(T min, T max)
        {
            thrust::device_vector<T>       data(elements);
            const thrust::detail::device_t policy {};

            if(entropy_reduction == 0) /*bit_entropy::_1_000*/
            {
                const double* uniform_distribution
                    = this->new_uniform_distribution(seed, data.size());

                thrust::transform(policy,
                                  uniform_distribution,
                                  uniform_distribution + data.size(),
                                  data.data(),
                                  random_to_item_t<T>(min, max));
                return data;
            }
            else if(entropy_reduction >= 5) /*bit_entropy::_0_000*/
            {
                std::mt19937 rng;
                rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
                std::uniform_real_distribution<float> dist(0.0f, 1.0f);
                T random_value = detail::random_to_item_t<T>(min, max)(dist(rng));
                thrust::fill(policy, data.data(), data.data() + data.size(), random_value);
                return data;
            }
            else
            {
                const double* uniform_distribution
                    = this->new_uniform_distribution(seed, data.size());
                ++seed;

                thrust::transform(policy,
                                  uniform_distribution,
                                  uniform_distribution + data.size(),
                                  data.data(),
                                  random_to_item_t<T>(min, max));

                const int number_of_steps = entropy_reduction;

                thrust::device_vector<T> tmp(data.size());

                for(int i = 0; i < number_of_steps; i++, ++seed)
                {
                    this->no_entropy_generate(tmp, min, max);

                    thrust::transform(policy,
                                      data.data(),
                                      data.data() + data.size(),
                                      tmp.data(),
                                      data.data(),
                                      detail::and_t {});
                }
                return data;
            }
        }

        const double* new_uniform_distribution(seed_t seed, std::size_t num_items)
        {
            distribution.resize(num_items);
            double* d_distribution = thrust::raw_pointer_cast(distribution.data());

            rocrand_set_seed(gen, seed.get());
            rocrand_generate_uniform_double(gen, d_distribution, num_items);

            hipError_t error = hipDeviceSynchronize();
            if(error != hipSuccess)
            {
                std::cout << "HIP error: " << hipGetErrorString(error) << " file: " << __FILE__
                        << " line: " << __LINE__ << std::endl;
                exit(error);
            }   

            return d_distribution;
        }

    private:
        rocrand_generator             gen;
        thrust::device_vector<double> distribution;

        template <typename T, typename Policy = thrust::detail::device_t>
        void no_entropy_generate(thrust::device_vector<T>& data, T m_min, T m_max)
        {
            const double* uniform_distribution = this->new_uniform_distribution(seed, data.size());

            thrust::transform(Policy {},
                              uniform_distribution,
                              uniform_distribution + data.size(),
                              data.data(),
                              detail::random_to_item_t<T>(m_min, m_max));
            return;
        }
    };

    template <class T>
    struct device_vector_generator_t : device_generator_base_t
    {
        const T min {std::numeric_limits<T>::min()};
        const T max {std::numeric_limits<T>::max()};

        device_vector_generator_t(std::size_t        m_elements,
                                  const std::string& m_seed_type,
                                  int                m_entropy_reduction,
                                  T                  m_min,
                                  T                  m_max)
            : device_generator_base_t(m_elements, m_seed_type, m_entropy_reduction)
            , min(m_min)
            , max(m_max)
        {
        }

        operator thrust::device_vector<T>()
        {
            return device_generator_base_t::generate(min, max);
        }
    };

    template <>
    struct device_vector_generator_t<void> : device_generator_base_t
    {
        device_vector_generator_t(std::size_t        m_elements,
                                  const std::string& m_seed_type,
                                  int                m_entropy_reduction)
            : device_generator_base_t(m_elements, m_seed_type, m_entropy_reduction)
        {
        }

        template <typename T>
        operator thrust::device_vector<T>()
        {
            return device_generator_base_t::generate(std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());
        }
    };

    template <typename T>
    std::size_t gen_uniform_offsets(const std::string         seed_type,
                                    thrust::device_vector<T>& segment_offsets,
                                    const std::size_t         min_segment_size,
                                    const std::size_t         max_segment_size)
    {
        const T elements = segment_offsets.size() - 2;

        segment_offsets
            = device_generator_base_t(segment_offsets.size(), seed_type, 0 /*bit_entropy::_1_000*/)
                  .generate(static_cast<T>(min_segment_size), static_cast<T>(max_segment_size));

        // Find the range of contiguous offsets starting from index 0 which sum is greater or
        // equal than 'elements'.
        const thrust::detail::device_t policy {};

        // Add the offset 'elements + 1' to the array of segment offsets to make sure that
        // there is at least one offset greater than 'elements'.
        thrust::fill_n(policy, segment_offsets.data() + elements, 1, elements + 1);

        // Perform an exclusive prefix sum scan with first value 0, so what we compute is
        // scan[i + 1] = \sum_{i=0}^{i} segment_offsets[i] for i \in [0, elements+1]
        // and scan[0] = 0.
        thrust::exclusive_scan(policy,
                               segment_offsets.data(),
                               segment_offsets.data() + segment_offsets.size(),
                               segment_offsets.data() /*, thrust::plus<>{}*/);

        // Find first sum of offsets greater than 'elements', we are sure that there is
        // going to be one because we added elements + 1 at the end of the segment_offsets.
        auto iter = thrust::find_if(policy,
                                    segment_offsets.data(),
                                    segment_offsets.data() + segment_offsets.size(),
                                    geq_t<T> {elements});

        // Compute the element's index.
        auto dist = thrust::distance(segment_offsets.data(), iter);
        // Fill next item with 'elements'.
        thrust::fill_n(policy, segment_offsets.data() + dist, 1, elements);
        // Return next item's index.
        return dist + 1;
    }

    template <class T>
    struct constant_op
    {
        std::size_t val;

        __device__ __forceinline__ T operator()(const T& /*key*/) const
        {
            return static_cast<T>(val);
        }
    };

    template <class T>
    struct idx_to_op
    {
        T*           keys            = nullptr;
        std::size_t* segment_offsets = nullptr;

        __device__ __forceinline__ std::size_t operator()(std::size_t i)
        {
            const std::size_t init_offset = segment_offsets[i];
            const std::size_t end_offset  = segment_offsets[i + 1];
            thrust::transform(thrust::device,
                              keys + init_offset,
                              keys + end_offset,
                              keys + init_offset,
                              constant_op<T> {i});
            return i;
        }
    };

    // Temporal approach for generation of key segments.
    template <typename T>
    void gen_key_segments(thrust::device_vector<T>&      keys,
                          thrust::device_vector<size_t>& segment_offsets)
    {
        const std::size_t total_segments = segment_offsets.size() - 1;

        thrust::counting_iterator<int>     iota(0);
        thrust::device_vector<std::size_t> segment_indices(iota, iota + total_segments);

        idx_to_op<T> op {thrust::raw_pointer_cast(keys.data()),
                         thrust::raw_pointer_cast(segment_offsets.data())};

        thrust::transform(
            segment_indices.begin(), segment_indices.end(), segment_indices.begin(), op);
    }

    // TODO: use this approach for gen_key_segments when rocPRIM allows it.
    // template <class T>
    // struct repeat_index_t
    // {
    // __host__ __device__ __forceinline__ thrust::constant_iterator<T> operator()(std::size_t i)
    // {
    //     return thrust::constant_iterator<T>(static_cast<T>(i));
    // }
    // };

    // template <typename T>
    // struct offset_to_iterator_t
    // {
    //     T* base_it;

    //     __host__ __device__ __forceinline__ T* operator()(std::size_t offset) const
    //     {
    //         return base_it + offset;
    //     }
    // };

    // struct offset_to_size_t
    // {
    //     std::size_t* offsets = nullptr;

    //     __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
    //     {
    //         return offsets[i + 1] - offsets[i];
    //     }
    // };
    //
    // template <typename T>
    // void gen_key_segments(thrust::device_vector<T>&           keys,
    //                       thrust::device_vector<std::size_t>& segment_offsets)
    // {

    //     const std::size_t total_segments = segment_offsets.size() - 1;

    //     thrust::counting_iterator<int> iota(0);
    //     repeat_index_t<T>       src_transform_op {};
    //     offset_to_iterator_t<T> dst_transform_op {thrust::raw_pointer_cast(keys.data())};
    //     offset_to_size_t size_transform_op {thrust::raw_pointer_cast(segment_offsets.data())};

    //     auto d_range_srcs = thrust::make_transform_iterator(iota, src_transform_op);
    //     auto d_range_dsts
    //         = thrust::make_transform_iterator(segment_offsets.begin(), dst_transform_op);
    //     auto d_range_sizes = thrust::make_transform_iterator(iota, size_transform_op);

    //     std::uint8_t*     d_temp_storage     = nullptr;
    //     std::size_t       temp_storage_bytes = 0;

    //     rocprim::batch_copy(d_temp_storage,
    //                         temp_storage_bytes,
    //                         d_range_srcs,
    //                         d_range_dsts,
    //                         d_range_sizes,
    //                         total_segments);

    //     thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    //     d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    //     rocprim::batch_copy(d_temp_storage,
    //                         temp_storage_bytes,
    //                         d_range_srcs,
    //                         d_range_dsts,
    //                         d_range_sizes,
    //                         total_segments);
    //     hipDeviceSynchronize();
    // }

    struct device_uniform_key_segments_generator_t
    {
        const std::size_t elements {0};
        const std::string seed_type {"random"};
        const std::size_t min_segment_size {0};
        const std::size_t max_segment_size {0};

        device_uniform_key_segments_generator_t(std::size_t       m_elements,
                                                const std::string m_seed_type,
                                                const std::size_t m_min_segment_size,
                                                const std::size_t m_max_segment_size)
            : elements(m_elements)
            , seed_type(m_seed_type)
            , min_segment_size(m_min_segment_size)
            , max_segment_size(m_max_segment_size)
        {
        }

        template <class KeyT>
        operator thrust::device_vector<KeyT>()
        {
            thrust::device_vector<KeyT> keys(elements);

            thrust::device_vector<std::size_t> segment_offsets(keys.size() + 2);
            const std::size_t                  offsets_size = gen_uniform_offsets(
                seed_type, segment_offsets, min_segment_size, max_segment_size);
            segment_offsets.resize(offsets_size);

            gen_key_segments(keys, segment_offsets);

            return keys;
        }
    };

    struct gen_uniform_key_segments_t
    {
        device_uniform_key_segments_generator_t operator()(const std::size_t elements,
                                                           const std::string seed_type,
                                                           const std::size_t min_segment_size,
                                                           const std::size_t max_segment_size) const
        {
            return {elements, seed_type, min_segment_size, max_segment_size};
        }
    };

    struct gen_uniform_t
    {
        gen_uniform_key_segments_t key_segments {};
    };

    struct gen_t
    {
        template <class T>
        device_vector_generator_t<T> operator()(std::size_t       elements,
                                                const std::string seed_type,
                                                const int         entropy = 0 /*100*/,
                                                T                 min = std::numeric_limits<T>::min,
                                                T max = std::numeric_limits<T>::max()) const
        {
            return {elements, seed_type, entropy, min, max};
        }

        device_vector_generator_t<void> operator()(std::size_t       elements,
                                                   const std::string seed_type,
                                                   const int         entropy = 0 /*100*/) const
        {
            return {elements, seed_type, entropy};
        }

        gen_uniform_t uniform {};
    };
} // namespace detail

detail::gen_t generate;

} // namespace bench_utils
#endif // ROCTHRUST_BENCHMARKS_BENCH_UTILS_GENERATION_UTILS_HPP_
