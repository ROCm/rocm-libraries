// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_BENCHMARK_UTILS_HPP_
#define ROCPRIM_BENCHMARK_UTILS_HPP_

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#ifdef WIN32
#include <numeric>
#endif

#include "benchmark/benchmark.h"
#include <rocprim/rocprim.hpp>

#define HIP_CHECK(condition)                                                                \
    {                                                                                       \
        hipError_t error = condition;                                                       \
        if(error != hipSuccess)                                                             \
        {                                                                                   \
            std::cout << "HIP error: " << hipGetErrorString(error) << " line: " << __LINE__ \
                      << std::endl;                                                         \
            exit(error);                                                                    \
        }                                                                                   \
    }

#define TUNING_SHARED_MEMORY_MAX 65536u
// Support half operators on host side

ROCPRIM_HOST inline
rocprim::native_half half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const rocprim::native_half *>(&x);
}

ROCPRIM_HOST inline
rocprim::half native_to_half(const rocprim::native_half& x)
{
    return *reinterpret_cast<const rocprim::half *>(&x);
}

struct half_less
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b;
        #else
        return half_to_native(a) < half_to_native(b);
        #endif
    }
};

struct half_plus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::half operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return native_to_half(half_to_native(a) + half_to_native(b));
        #endif
    }
};

struct half_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::half& a, const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a == b;
        #else
        return half_to_native(a) == half_to_native(b);
        #endif
    }
};

// std::uniform_int_distribution is undefined for anything other than:
// short, int, long, long long, unsigned short, unsigned int, unsigned long, or unsigned long long
template <typename T>
struct is_valid_for_int_distribution :
    std::integral_constant<bool,
        std::is_same<short, T>::value ||
        std::is_same<unsigned short, T>::value ||
        std::is_same<int, T>::value ||
        std::is_same<unsigned int, T>::value ||
        std::is_same<long, T>::value ||
        std::is_same<unsigned long, T>::value ||
        std::is_same<long long, T>::value ||
        std::is_same<unsigned long long, T>::value
    > {};

using engine_type = std::default_random_engine;

// get_random_data() generates only part of sequence and replicates it,
// because benchmarks usually do not need "true" random sequence.
template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<rocprim::is_integral<T>::value, std::vector<T>>::type
{
    engine_type gen{std::random_device{}()};
    using dis_type = typename std::conditional<
        is_valid_for_int_distribution<T>::value,
        T,
        typename std::conditional<std::is_signed<T>::value,
            int,
            unsigned int>::type
        >::type;
    std::uniform_int_distribution<dis_type> distribution((T)min, (T)max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T, class U, class V>
inline auto get_random_data(size_t size, U min, V max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<rocprim::is_floating_point<T>::value, std::vector<T>>::type
{
    engine_type gen{std::random_device{}()};
    // Generate floats when T is half
    using dis_type = typename std::conditional<std::is_same<rocprim::half, T>::value, float, T>::type;
    std::uniform_real_distribution<dis_type> distribution((dis_type)min, (dis_type)max);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline std::vector<T> get_random_data01(size_t size, float p, size_t max_random_size = 1024 * 1024)
{
    engine_type gen{std::random_device{}()};
    std::bernoulli_distribution distribution(p);
    std::vector<T> data(size);
    std::generate(
        data.begin(), data.begin() + std::min(size, max_random_size),
        [&]() { return distribution(gen); }
    );
    for(size_t i = max_random_size; i < size; i += max_random_size)
    {
        std::copy_n(data.begin(), std::min(size - i, max_random_size), data.begin() + i);
    }
    return data;
}

template<class T>
inline T get_random_value(T min, T max)
{
    return get_random_data(1, min, max)[0];
}

template<class T, class U = T>
struct custom_type
{
    using first_type = T;
    using second_type = U;

    T x;
    U y;

    ROCPRIM_HOST_DEVICE inline
    custom_type(T xx = 0, U yy = 0) : x(xx), y(yy)
    {
    }

    ROCPRIM_HOST_DEVICE inline
    ~custom_type() = default;

    ROCPRIM_HOST_DEVICE inline
    custom_type operator+(const custom_type& rhs) const
    {
        return custom_type(x + rhs.x, y + rhs.y);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator<(const custom_type& rhs) const
    {
        // intentionally suboptimal choice for short-circuting,
        // required to generate more performant device code
        return ((x == rhs.x && y < rhs.y) || x < rhs.x);
    }

    ROCPRIM_HOST_DEVICE inline
    bool operator==(const custom_type& rhs) const
    {
        return x == rhs.x && y == rhs.y;
    }
};

template<typename>
struct is_custom_type : std::false_type {};

template<class T, class U>
struct is_custom_type<custom_type<T,U>> : std::true_type {};

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<is_custom_type<T>::value, std::vector<T>>::type
{
    using first_type = typename T::first_type;
    using second_type = typename T::second_type;
    std::vector<T> data(size);
    auto fdata = get_random_data<first_type>(size, min.x, max.x, max_random_size);
    auto sdata = get_random_data<second_type>(size, min.y, max.y, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(fdata[i], sdata[i]);
    }
    return data;
}

template<class T>
inline auto get_random_data(size_t size, T min, T max, size_t max_random_size = 1024 * 1024)
    -> typename std::enable_if<!is_custom_type<T>::value && !std::is_same<decltype(max.x), void>::value, std::vector<T>>::type
{
    // NOTE 1: post-increment operator required, because HIP has different typedefs for vector field types
    //         when using HCC or HIP-Clang. Using HIP-Clang members are accessed as fields of a struct via
    //         a union, but in HCC mode they are proxy types (Scalar_accessor). STL algorithms don't
    //         always tolerate proxies. Unfortunately, Scalar_accessor doesn't have any member typedefs to
    //         conveniently obtain the inner stored type. All operations on it (operator+, operator+=,
    //         CTOR, etc.) return a reference to an accessor, it is only the post-increment operator that
    //         returns a copy of the stored type, hence we take the decltype of that.
    //
    // NOTE 2: decltype() is unevaluated context. We don't really modify max, just compute the type of the
    //         expression if we were to actually call it.
    using field_type = decltype(max.x++);
    std::vector<T> data(size);
    auto field_data = get_random_data<field_type>(size, min.x, max.x, max_random_size);
    for(size_t i = 0; i < size; i++)
    {
        data[i] = T(field_data[i]);
    }
    return data;
}

inline bool is_warp_size_supported(const unsigned int required_warp_size)
{
    return ::rocprim::host_warp_size() >= required_warp_size;
}

template<unsigned int LogicalWarpSize>
struct DeviceSelectWarpSize
{
    static constexpr unsigned int value = ::rocprim::device_warp_size() >= LogicalWarpSize
        ? LogicalWarpSize
        : ::rocprim::device_warp_size();
};

template<typename T>
std::vector<T> get_random_segments(const size_t size,
                                   const size_t max_segment_length,
                                   const int seed_value)
{
    static_assert(std::is_arithmetic<T>::value, "Key type must be arithmetic");

    std::default_random_engine prng(seed_value);
    std::uniform_int_distribution<size_t> segment_length_distribution(max_segment_length);
    using key_distribution_type = std::conditional_t<
        std::is_integral<T>::value,
        std::uniform_int_distribution<T>,
        std::uniform_real_distribution<T>
    >;
    key_distribution_type key_distribution(std::numeric_limits<T>::max());
    std::vector<T> keys(size);

    size_t keys_start_index = 0;
    while (keys_start_index < size)
    {
        const size_t new_segment_length = segment_length_distribution(prng);
        const size_t new_segment_end = std::min(size, keys_start_index + new_segment_length);
        const T key = key_distribution(prng);
        std::fill(
            keys.begin() + keys_start_index,
            keys.begin() + new_segment_end,
            key
        );
        keys_start_index += new_segment_length;
    }
    return keys;
}

template <typename T, T, typename>
struct make_index_range_impl;

template <typename T, T Start, T... I>
struct make_index_range_impl<T, Start, std::integer_sequence<T, I...>>
{
    using type = std::integer_sequence<T, (Start + I)...>;
};

// make a std::integer_sequence with values from Start to End inclusive
template <typename T, T Start, T End>
using make_index_range =
    typename make_index_range_impl<T, Start, std::make_integer_sequence<T, End - Start + 1>>::type;

template<typename T, template<T> class Function, T... I, typename... Args>
void static_for_each_impl(std::integer_sequence<T, I...>, Args&&... args)
{
    int a[] = {(Function<I>{}(std::forward<Args>(args)...), 0)...};
    static_cast<void>(a);
}

// call the supplied template with all values of the std::integer_sequence Indices
template<typename Indices, template<typename Indices::value_type> class Function, typename... Args>
void static_for_each(Args&&... args)
{
    static_for_each_impl<typename Indices::value_type, Function>(Indices{},
                                                                 std::forward<Args>(args)...);
}

#define REGISTER_BENCHMARK(benchmarks, size, stream, instance)                     \
    benchmark::internal::Benchmark* benchmark = benchmark::RegisterBenchmark(      \
        instance.name().c_str(),                                                   \
        [instance](benchmark::State& state, size_t size, const hipStream_t stream) \
        { instance.run(state, size, stream); },                                    \
        size,                                                                      \
        stream);                                                                   \
    benchmarks.emplace_back(benchmark)

struct config_autotune_interface
{
    virtual std::string name() const                               = 0;
    virtual std::string sort_key() const
    {
        return name();
    };
    virtual ~config_autotune_interface()                           = default;
    virtual void run(benchmark::State&, size_t, hipStream_t) const = 0;
};

struct config_autotune_register
{
    static std::vector<std::unique_ptr<config_autotune_interface>>& vector() {
        static std::vector<std::unique_ptr<config_autotune_interface>> storage;
        return storage;
    }

    template <typename T>
    static config_autotune_register create() {
        vector().push_back(std::make_unique<T>());
        return config_autotune_register();
    }

    template<typename BulkCreateFunction>
    static config_autotune_register create_bulk(BulkCreateFunction&& f)
    {
        std::forward<BulkCreateFunction>(f)(vector());
        return config_autotune_register();
    }

    // Register a subset of all created benchmarks for the current parallel instance and add to vector.
    static void register_benchmark_subset(std::vector<benchmark::internal::Benchmark*>& benchmarks,
                                          int               parallel_instance_index,
                                          int               parallel_instance_count,
                                          size_t            size,
                                          const hipStream_t stream)
    {
        std::vector<std::unique_ptr<config_autotune_interface>>& configs = vector();
        // sorting to get a consistent order because order of initialization of static variables is undefined by the C++ standard.
        std::sort(configs.begin(),
                  configs.end(),
                  [](const auto& l, const auto& r) { return l->sort_key() < r->sort_key(); });
        size_t configs_per_instance
            = (configs.size() + parallel_instance_count - 1) / parallel_instance_count;
        size_t start = std::min(parallel_instance_index * configs_per_instance, configs.size());
        size_t end = std::min((parallel_instance_index + 1) * configs_per_instance, configs.size());
        for(size_t i = start; i < end; i++)
        {
            std::unique_ptr<config_autotune_interface>& uniq_ptr         = configs.at(i);
            config_autotune_interface*                  tuning_benchmark = uniq_ptr.get();
            benchmark::internal::Benchmark*             benchmark = benchmark::RegisterBenchmark(
                tuning_benchmark->name().c_str(),
                [tuning_benchmark](benchmark::State& state, size_t size, const hipStream_t stream)
                { tuning_benchmark->run(state, size, stream); },
                size,
                stream);
            benchmarks.emplace_back(benchmark);
        }
    }
};

// Inserts spaces at beginning of string if string shorter than specified length.
inline std::string pad_string(std::string str, const size_t len)
{
    if(len > str.size())
    {
        str.insert(str.begin(), len - str.size(), ' ');
    }

    return str;
}

struct bench_naming
{
public:
    enum format
    {
        json,
        human,
        txt
    };
    static format& get_format()
    {
        static format storage = human;
        return storage;
    }
    static void set_format(std::string argument)
    {
        format result = human;
        if(argument == "json")
        {
            result = json;
        }
        else if(argument == "txt")
        {
            result = txt;
        }
        get_format() = result;
    }

private:
    static std::string matches_as_json(std::sregex_iterator& matches)
    {
        std::stringstream result;
        int               brackets_count = 1;
        result << "{";
        bool insert_comma = false;
        for(std::sregex_iterator i = matches; i != std::sregex_iterator(); ++i)
        {
            std::smatch m = *i;
            if(insert_comma)
            {
                result << ",";
            }
            else
            {
                insert_comma = true;
            }
            result << "\"" << m[1].str() << "\":";
            if(m[2].length() > 0)
            {
                if(m[2].str().find_first_not_of("0123456789") == std::string::npos)
                {
                    result << m[2].str();
                }
                else
                {
                    result << "\"" << m[2].str() << "\"";
                }
                if(m[3].length() > 0 && brackets_count > 0)
                {
                    int n = std::min(brackets_count, static_cast<int>(m[3].length()));
                    brackets_count -= n;
                    for(int c = 0; c < n; c++)
                    {
                        result << "}";
                    }
                }
            }
            else
            {
                brackets_count++;
                result << "{";
                insert_comma = false;
            }
        }
        while(brackets_count > 0)
        {
            brackets_count--;
            result << "}";
        }
        return result.str();
    }

    static std::string matches_as_human(std::sregex_iterator& matches)
    {
        std::stringstream result;
        int               brackets_count = 0;
        bool              insert_comma   = false;
        for(std::sregex_iterator i = matches; i != std::sregex_iterator(); ++i)
        {
            std::smatch m = *i;
            if(insert_comma)
            {
                result << ",";
            }
            else
            {
                insert_comma = true;
            }
            if(m[2].length() > 0)
            {
                result << m[2].str();
                if(m[3].length() > 0 && brackets_count > 0)
                {
                    int n = std::min(brackets_count, static_cast<int>(m[3].length()));
                    brackets_count -= n;
                    for(int c = 0; c < n; c++)
                    {
                        result << ">";
                    }
                }
            }
            else
            {
                brackets_count++;
                result << "<";
                insert_comma = false;
            }
        }
        while(brackets_count > 0)
        {
            brackets_count--;
            result << ">";
        }
        return result.str();
    }

public:
    static std::string format_name(std::string string)
    {
        format     format = get_format();
std::regex r("([A-z0-9]*):\\s*((?:custom_type<[A-z0-9,]*>)|[A-z:\\(\\)\\.<>\\s0-9]*)(\\}*)");
        // First we perform some checks
        bool checks[4] = {false};
        for(std::sregex_iterator i = std::sregex_iterator(string.begin(), string.end(), r);
            i != std::sregex_iterator();
            ++i)
        {
            std::smatch m = *i;
            if(m[1].str() == "lvl")
            {
                checks[0] = true;
            }
            else if(m[1].str() == "algo")
            {
                checks[1] = true;
            }
            else if(m[1].str() == "cfg")
            {
                checks[2] = true;
            }
        }
        std::string string_substitute = std::regex_replace(string, r, "");
        checks[3] = string_substitute.find_first_not_of(" ,{}") == std::string::npos;
        for(bool check_name_format : checks)
        {
            if(!check_name_format)
            {
                std::cout << "Benchmark name \"" << string
                          << "\" not in the correct format (e.g. "
                             "{lvl:block,algo:reduce,cfg:default_config} )"
                          << std::endl;
                exit(1);
            }
        }

        // Now we generate the desired format
        std::sregex_iterator matches = std::sregex_iterator(string.begin(), string.end(), r);

        switch(format)
        {
            case format::json: return matches_as_json(matches);
            case format::human: return matches_as_human(matches);
            case format::txt: return string;
        }
        return string;
    }
};

template <typename T>
struct Traits
{
    //static inline method instead of static inline attribute because that's only supported from C++17 onwards
    static inline const char* name(){
        static_assert(sizeof(T) == 0, "Traits<T>::name() unknown");
        return "unknown";
    }
};

// Explicit definitions
template <>
inline const char* Traits<int>::name() { return "int"; }
template <>
inline const char* Traits<short>::name() { return "short"; }
template <>
inline const char* Traits<int8_t>::name() { return "int8_t"; }
template <>
inline const char* Traits<uint8_t>::name() { return "uint8_t"; }
template<>
inline const char* Traits<uint16_t>::name()
{
    return "uint16_t";
}
template<>
inline const char* Traits<uint32_t>::name()
{
    return "uint32_t";
}
template<>
inline const char* Traits<rocprim::half>::name()
{
    return "rocprim::half";
}
template<>
inline const char* Traits<long long>::name()
{
    return "int64_t";
}
// On MSVC `int64_t` and `long long` are the same, leading to multiple definition errors
#ifndef WIN32
template <>
inline const char* Traits<int64_t>::name() { return "int64_t"; }
#endif
template <>
inline const char* Traits<float>::name() { return "float"; }
template <>
inline const char* Traits<double>::name() { return "double"; }
template<>
inline const char* Traits<custom_type<int, int>>::name()
{
    return "custom_type<int,int>";
}
template<>
inline const char* Traits<custom_type<float, float>>::name()
{
    return "custom_type<float,float>";
}
template<>
inline const char* Traits<custom_type<double, double>>::name()
{
    return "custom_type<double,double>";
}
template<>
inline const char* Traits<custom_type<char, double>>::name()
{
    return "custom_type<char,double>";
}
template<>
inline const char* Traits<custom_type<long, double>>::name()
{
    return "custom_type<long,double>";
}
template<>
inline const char* Traits<custom_type<long long, double>>::name()
{
    return "custom_type<int64_t,double>";
}
template<>
inline const char* Traits<rocprim::empty_type>::name()
{
    return "empty_type";
}
template<>
inline const char* Traits<HIP_vector_type<float, 2>>::name()
{
    return "float2";
}
template<>
inline const char* Traits<HIP_vector_type<double, 2>>::name()
{
    return "double2";
}

inline void add_common_benchmark_info()
{
    hipDeviceProp_t   devProp;
    int               device_id = 0;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&devProp, device_id));

    auto str = [](const std::string& name, const std::string& val) {
        benchmark::AddCustomContext(name, val);
    };

    auto num = [](const std::string& name, const auto& value) {
        benchmark::AddCustomContext(name, std::to_string(value));
    };

    auto dim2 = [num](const std::string& name, const auto* values) {
        num(name + "_x", values[0]);
        num(name + "_y", values[1]);
    };

    auto dim3 = [num, dim2](const std::string& name, const auto* values) {
        dim2(name, values);
        num(name + "_z", values[2]);
    };

    str("hdp_name", devProp.name);
    num("hdp_total_global_mem", devProp.totalGlobalMem);
    num("hdp_shared_mem_per_block", devProp.sharedMemPerBlock);
    num("hdp_regs_per_block", devProp.regsPerBlock);
    num("hdp_warp_size", devProp.warpSize);
    num("hdp_max_threads_per_block", devProp.maxThreadsPerBlock);
    dim3("hdp_max_threads_dim", devProp.maxThreadsDim);
    dim3("hdp_max_grid_size", devProp.maxGridSize);
    num("hdp_clock_rate", devProp.clockRate);
    num("hdp_memory_clock_rate", devProp.memoryClockRate);
    num("hdp_memory_bus_width", devProp.memoryBusWidth);
    num("hdp_total_const_mem", devProp.totalConstMem);
    num("hdp_major", devProp.major);
    num("hdp_minor", devProp.minor);
    num("hdp_multi_processor_count", devProp.multiProcessorCount);
    num("hdp_l2_cache_size", devProp.l2CacheSize);
    num("hdp_max_threads_per_multiprocessor", devProp.maxThreadsPerMultiProcessor);
    num("hdp_compute_mode", devProp.computeMode);
    num("hdp_clock_instruction_rate", devProp.clockInstructionRate);
    num("hdp_concurrent_kernels", devProp.concurrentKernels);
    num("hdp_pci_domain_id", devProp.pciDomainID);
    num("hdp_pci_bus_id", devProp.pciBusID);
    num("hdp_pci_device_id", devProp.pciDeviceID);
    num("hdp_max_shared_memory_per_multi_processor", devProp.maxSharedMemoryPerMultiProcessor);
    num("hdp_is_multi_gpu_board", devProp.isMultiGpuBoard);
    num("hdp_can_map_host_memory", devProp.canMapHostMemory);
    str("hdp_gcn_arch_name", devProp.gcnArchName);
    num("hdp_integrated", devProp.integrated);
    num("hdp_cooperative_launch", devProp.cooperativeLaunch);
    num("hdp_cooperative_multi_device_launch", devProp.cooperativeMultiDeviceLaunch);
    num("hdp_max_texture_1d_linear", devProp.maxTexture1DLinear);
    num("hdp_max_texture_1d", devProp.maxTexture1D);
    dim2("hdp_max_texture_2d", devProp.maxTexture2D);
    dim3("hdp_max_texture_3d", devProp.maxTexture3D);
    num("hdp_mem_pitch", devProp.memPitch);
    num("hdp_texture_alignment", devProp.textureAlignment);
    num("hdp_texture_pitch_alignment", devProp.texturePitchAlignment);
    num("hdp_kernel_exec_timeout_enabled", devProp.kernelExecTimeoutEnabled);
    num("hdp_ecc_enabled", devProp.ECCEnabled);
    num("hdp_tcc_driver", devProp.tccDriver);
    num("hdp_cooperative_multi_device_unmatched_func", devProp.cooperativeMultiDeviceUnmatchedFunc);
    num("hdp_cooperative_multi_device_unmatched_grid_dim", devProp.cooperativeMultiDeviceUnmatchedGridDim);
    num("hdp_cooperative_multi_device_unmatched_block_dim", devProp.cooperativeMultiDeviceUnmatchedBlockDim);
    num("hdp_cooperative_multi_device_unmatched_shared_mem", devProp.cooperativeMultiDeviceUnmatchedSharedMem);
    num("hdp_is_large_bar", devProp.isLargeBar);
    num("hdp_asic_revision", devProp.asicRevision);
    num("hdp_managed_memory", devProp.managedMemory);
    num("hdp_direct_managed_mem_access_from_host", devProp.directManagedMemAccessFromHost);
    num("hdp_concurrent_managed_access", devProp.concurrentManagedAccess);
    num("hdp_pageable_memory_access", devProp.pageableMemoryAccess);
    num("hdp_pageable_memory_access_uses_host_page_tables", devProp.pageableMemoryAccessUsesHostPageTables);

    const auto arch = devProp.arch;
    num("hdp_arch_has_global_int32_atomics", arch.hasGlobalInt32Atomics);
    num("hdp_arch_has_global_float_atomic_exch", arch.hasGlobalFloatAtomicExch);
    num("hdp_arch_has_shared_int32_atomics", arch.hasSharedInt32Atomics);
    num("hdp_arch_has_shared_float_atomic_exch", arch.hasSharedFloatAtomicExch);
    num("hdp_arch_has_float_atomic_add", arch.hasFloatAtomicAdd);
    num("hdp_arch_has_global_int64_atomics", arch.hasGlobalInt64Atomics);
    num("hdp_arch_has_shared_int64_atomics", arch.hasSharedInt64Atomics);
    num("hdp_arch_has_doubles", arch.hasDoubles);
    num("hdp_arch_has_warp_vote", arch.hasWarpVote);
    num("hdp_arch_has_warp_ballot", arch.hasWarpBallot);
    num("hdp_arch_has_warp_shuffle", arch.hasWarpShuffle);
    num("hdp_arch_has_funnel_shift", arch.hasFunnelShift);
    num("hdp_arch_has_thread_fence_system", arch.hasThreadFenceSystem);
    num("hdp_arch_has_sync_threads_ext", arch.hasSyncThreadsExt);
    num("hdp_arch_has_surface_funcs", arch.hasSurfaceFuncs);
    num("hdp_arch_has_3d_grid", arch.has3dGrid);
    num("hdp_arch_has_dynamic_parallelism", arch.hasDynamicParallelism);
}

#endif // ROCPRIM_BENCHMARK_UTILS_HPP_
