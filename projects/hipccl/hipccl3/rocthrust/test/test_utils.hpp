/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019 Advanced Micro Devices, Inc. All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once

#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/limits.h>
#include <thrust/mr/allocator.h>
#include <thrust/detail/event_error.h>

#include "test_seed.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>
#include <iterator>

#define TEST_EVENT_WAIT(e) test_event_wait(e)

// for demangling the result of type_info.name()
// with msvc, type_info.name() is already demangled
#ifdef __GNUC__
#include <cxxabi.h>
#endif // __GNUC__

#include <string>
#include <cstdlib>

#ifdef __GNUC__
inline std::string demangle(const char* name)
{
  int status = 0;
  char* realname = abi::__cxa_demangle(name, 0, 0, &status);
  std::string result(realname);
  std::free(realname);

  return result;
}
#else
inline std::string demangle(const char* name)
{
  return name;
}
#endif

class UnitTestException
{
    public:
    std::string message;

    UnitTestException() {}
    UnitTestException(const std::string& msg) : message(msg) {}

    friend std::ostream& operator<<(std::ostream& os, const UnitTestException& e)
    {
        return os << e.message;
    }

    template <typename T>
    UnitTestException& operator<<(const T& t)
    {
        std::ostringstream oss;
        oss << t;
        message += oss.str();
        return *this;
    }
};


class UnitTestError   : public UnitTestException
{
    public:
    UnitTestError() {}
    UnitTestError(const std::string& msg) : UnitTestException(msg) {}
};

class UnitTestKnownFailure : public UnitTestException
{
    public:
    UnitTestKnownFailure() {}
    UnitTestKnownFailure(const std::string& msg) : UnitTestException(msg) {}
};



class UnitTestFailure : public UnitTestException
{
    public:
    UnitTestFailure() {}
    UnitTestFailure(const std::string& msg) : UnitTestException(msg) {}
};

template<typename T>
  std::string type_name(void)
{
  return demangle(typeid(T).name());
} // end type_name()

template <typename Event>
__host__
void test_event_wait(Event&& e)
{
  ASSERT_EQ(true, e.valid_stream());

  // Call at least once the hipDeviceSynchronize()
  // before the stream ready state check
  e.wait();
  while(!e.ready())
  {
      e.wait();
  }

  ASSERT_EQ(true, e.valid_stream());
  ASSERT_EQ(true, e.ready());
}

std::vector<size_t> get_sizes()
{
    std::vector<size_t> sizes = {
        0, 1, 2, 12, 63, 64, 211, 256, 344,
        1024, 2048, 5096, 34567, (1 << 17) - 1220, 1000000, (1 << 20) - 123
    };
    return sizes;
}

std::vector<seed_type> get_seeds()
{
    std::vector<seed_type> seeds;
    std::random_device rng;
    std::copy(prng_seeds.begin(), prng_seeds.end(), std::back_inserter(seeds));
    std::generate_n(std::back_inserter(seeds), rng_seed_count, [&](){ return rng(); });
    return seeds;
}

template <class T>
inline auto get_random_data(size_t size, T, T, int seed) ->
    typename std::enable_if<std::is_same<T, bool>::value, thrust::host_vector<T>>::type
{
    std::random_device          rd;
    std::default_random_engine  gen(rd());
    gen.seed(seed);
    std::bernoulli_distribution distribution(0.5);
    thrust::host_vector<T>      data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template <class T>
inline auto get_random_data(size_t size, T min, T max, int seed) ->
    typename std::enable_if<rocprim::is_integral<T>::value && !std::is_same<T, bool>::value,
                            thrust::host_vector<T>>::type
{
    std::random_device               rd;
    std::default_random_engine       gen(rd());
    gen.seed(seed);
    std::uniform_int_distribution<T> distribution(min, max);
    thrust::host_vector<T>           data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

template <class T>
inline auto get_random_data(size_t size, T min, T max, int seed) ->
    typename std::enable_if<rocprim::is_floating_point<T>::value, thrust::host_vector<T>>::type
{
    std::random_device                rd;
    std::default_random_engine        gen(rd());
    gen.seed(seed);
    std::uniform_real_distribution<T> distribution(min, max);
    thrust::host_vector<T>            data(size);
    std::generate(data.begin(), data.end(), [&]() { return distribution(gen); });
    return data;
}

#if defined(WIN32) && defined(__clang__)
template <>
inline thrust::host_vector<unsigned char> get_random_data(size_t size, unsigned char min, unsigned char max, int seed_value)
{
    std::random_device                 rd;
    std::default_random_engine         gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    thrust::host_vector<unsigned char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<unsigned char>(distribution(gen)); });
    return data;
}

template <>
inline thrust::host_vector<signed char> get_random_data(size_t size, signed char min, signed char max, int seed_value)
{
    std::random_device                 rd;
    std::default_random_engine         gen(rd());
    gen.seed(seed_value);
    std::uniform_int_distribution<int> distribution(static_cast<int>(min), static_cast<int>(max));
    thrust::host_vector<signed char> data(size);
    std::generate(data.begin(), data.end(), [&]() { return static_cast<signed char>(distribution(gen)); });
    return data;
}
#endif

template <class T>
struct custom_compare_less
{
    __host__ __device__ bool operator()(const T& lhs, const T& rhs) const
    {
        return lhs < rhs;
    }
}; // end less

struct user_swappable
{
    inline __host__ __device__ user_swappable(bool swapped = false)
        : was_swapped(swapped)
    {
    }

    bool was_swapped;
};

inline __host__ __device__ bool operator==(const user_swappable& x, const user_swappable& y)
{
    return x.was_swapped == y.was_swapped;
}

inline __host__ __device__ void swap(user_swappable& x, user_swappable& y)
{
    x.was_swapped = true;
    y.was_swapped = false;
}

class my_system : public thrust::device_execution_policy<my_system>
{
public:
    my_system(int)
        : correctly_dispatched(false)
        , num_copies(0)
    {
    }

    my_system(const my_system& other)
        : correctly_dispatched(false)
        , num_copies(other.num_copies + 1)
    {
    }

    void validate_dispatch()
    {
        correctly_dispatched = (num_copies == 0);
    }

    bool is_valid()
    {
        return correctly_dispatched;
    }

private:
    bool correctly_dispatched;

    // count the number of copies so that we can validate
    // that dispatch does not introduce any
    unsigned int num_copies;

    // disallow default construction
    my_system();
};

struct my_tag : thrust::device_execution_policy<my_tag>
{
};

template <typename T, unsigned int N>
struct FixedVector
{
    T data[N];

    __host__ __device__ FixedVector()
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
            data[i] = T();
    }

    __host__ __device__ FixedVector(T init)
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
            data[i] = init;
    }

    __host__ __device__ FixedVector operator+(const FixedVector& bs) const
    {
        FixedVector output;
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
            output.data[i] = data[i] + bs.data[i];
        return output;
    }

    __host__ __device__ bool operator<(const FixedVector& bs) const
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
        {
            if(data[i] < bs.data[i])
                return true;
            else if(bs.data[i] < data[i])
                return false;
        }
        return false;
    }

    __host__ __device__ bool operator==(const FixedVector& bs) const
    {
        #pragma nounroll
        for(unsigned int i = 0; i < N; i++)
        {
            if(!(data[i] == bs.data[i]))
                return false;
        }
        return true;
    }
};

template <typename Key, typename Value>
struct key_value
{
    typedef Key   key_type;
    typedef Value value_type;

    __host__ __device__ key_value(void)
        : key()
        , value()
    {
    }

    __host__ __device__ key_value(key_type k, value_type v)
        : key(k)
        , value(v)
    {
    }

    __host__ __device__ bool operator<(const key_value& rhs) const
    {
        return key < rhs.key;
    }

    __host__ __device__ bool operator>(const key_value& rhs) const
    {
        return key > rhs.key;
    }

    __host__ __device__ bool operator==(const key_value& rhs) const
    {
        return key == rhs.key && value == rhs.value;
    }

    __host__ __device__ bool operator!=(const key_value& rhs) const
    {
        return !operator==(rhs);
    }

    friend std::ostream& operator<<(std::ostream& os, const key_value& kv)
    {
        return os << "(" << kv.key << ", " << kv.value << ")";

    }

    key_type   key;
    value_type value;
};

inline unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

template<typename T, typename = void>
  struct generate_random_integer;

template<typename T>
  struct generate_random_integer<T,
    typename thrust::detail::disable_if<
      thrust::detail::is_non_bool_arithmetic<T>::value
    >::type
  >
{
  T operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));

      return static_cast<T>(rng());
  }
};

template<typename T>
  struct generate_random_integer<T,
    typename thrust::detail::enable_if<
      thrust::detail::is_non_bool_integral<T>::value
    >::type
  >
{
  T operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));
      thrust::uniform_int_distribution<T> dist;

      return static_cast<T>(dist(rng));
  }
};

template<typename T>
  struct generate_random_integer<T,
    typename thrust::detail::enable_if<
      thrust::detail::is_floating_point<T>::value
    >::type
  >
{
  T operator()(unsigned int i) const
  {
      T const min = std::numeric_limits<T>::min();
      T const max = std::numeric_limits<T>::max();

      thrust::default_random_engine rng(hash(i));
      thrust::uniform_real_distribution<T> dist(min, max);

      return static_cast<T>(dist(rng));
  }
};

template<>
  struct generate_random_integer<bool>
{
  bool operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));
      thrust::uniform_int_distribution<unsigned int> dist(0,1);

      return dist(rng) == 1;
  }
};


template<typename T>
  struct generate_random_sample
{
  T operator()(unsigned int i) const
  {
      thrust::default_random_engine rng(hash(i));
      thrust::uniform_int_distribution<unsigned int> dist(0,20);

      return static_cast<T>(dist(rng));
  }
};



template<typename T>
thrust::host_vector<T> random_integers(const size_t N)
{
    thrust::host_vector<T> vec(N);
    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(N),
                      vec.begin(),
                      generate_random_integer<T>());

    return vec;
}

template<typename T>
T random_integer()
{
    return generate_random_integer<T>()(0);
}

template<typename T>
thrust::host_vector<T> random_samples(const size_t N)
{
    thrust::host_vector<T> vec(N);
    thrust::transform(thrust::counting_iterator<size_t>(0),
                      thrust::counting_iterator<size_t>(N),
                      vec.begin(),
                      generate_random_sample<T>());

    return vec;
}

// Use this with counting_iterator to avoid generating a range larger than we
// can represent.
template <typename T>
typename thrust::detail::disable_if<
  thrust::detail::is_floating_point<T>::value
, T
>::type truncate_to_max_representable(std::size_t n)
{
  return thrust::min<std::size_t>(
    n, static_cast<std::size_t>(thrust::numeric_limits<T>::max())
  );
}

// TODO: This probably won't work for `half`.
template <typename T>
typename thrust::detail::enable_if<
  thrust::detail::is_floating_point<T>::value
, T
>::type truncate_to_max_representable(std::size_t n)
{
  return thrust::min<T>(
    n, thrust::numeric_limits<T>::max()
  );
}

enum threw_status
{
  did_not_throw
, threw_wrong_type
, threw_right_type_but_wrong_value
, threw_right_type
};

void check_assert_throws(
  threw_status s
, std::string const& exception_name
, std::string const& file_name = "unknown"
, int line_number = -1
)
{
  switch (s)
  {
    case did_not_throw:
    {
      UnitTestFailure f;
      f << "[" << file_name << ":" << line_number << "] did not throw anything";
      throw f;
    }
    case threw_wrong_type:
    {
      UnitTestFailure f;
      f << "[" << file_name << ":" << line_number << "] did not throw an "
        << "object of type " << exception_name;
      throw f;
    }
    case threw_right_type_but_wrong_value:
    {
      UnitTestFailure f;
      f << "[" << file_name << ":" << line_number << "] threw an object of the "
        << "correct type (" << exception_name << ") but wrong value";
      throw f;
    }
    case threw_right_type:
      break;
    default:
    {
      UnitTestFailure f;
      f << "[" << file_name << ":" << line_number << "] encountered an "
        << "unknown error";
      throw f;
    }
  }
}

template <typename Future>
__host__
void test_future_value_retrieval(Future&& f, decltype(f.extract()) &return_value)
{
  ASSERT_EQ(true, f.valid_stream());
  ASSERT_EQ(true, f.valid_content());

  auto const r0 = f.get();
  auto const r1 = f.get();

  ASSERT_EQ(true, f.ready());
  ASSERT_EQ(true, f.valid_stream());
  ASSERT_EQ(true, f.valid_content());
  ASSERT_EQ(r0, r1);

  auto const r2 = f.extract();

  ASSERT_THROW(
    auto x = f.extract();
    THRUST_UNUSED_VAR(x)
    , thrust::event_error
  );

  ASSERT_EQ(false, f.ready());
  ASSERT_EQ(false, f.valid_stream());
  ASSERT_EQ(false, f.valid_content());
  ASSERT_EQ(r2, r1);
  ASSERT_EQ(r2, r0);

  return_value = r2;
}

template<class T>
struct precision_threshold
{
    static constexpr float percentage = 0.01f;
};

template<>
struct precision_threshold<rocprim::half>
{
    static constexpr float percentage = 0.075f;
};
