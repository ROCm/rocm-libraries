#include <thrust/detail/config.h>

#if THRUST_CPP_DIALECT >= 2011

#include <thrust/async/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "test_header.hpp"

template <typename T>
struct custom_plus
{
  __host__ __device__
  T operator()(T lhs, T rhs) const
  {
    return lhs + rhs;
  }
};

#if 0
template <typename T>
struct test_async_reduce
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    auto r0 = thrust::reduce(
      h0_data.begin(), h0_data.end()
    );

    auto f0 = thrust::async::reduce(
      d0_data.begin(), d0_data.end()
    );

    auto r1 = std::move(f0).get();

    ASSERT_EQUAL(r0, r1);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_reduce
, NumericTypes
> test_async_reduce_instance;
#else
TESTS_DEFINE(AsyncReduceTests, FullTestsParams);
TYPED_TEST(AsyncReduceTests, TestAsyncReduce)
{
    using T = typename TestFixture::input_type;
    const std::vector<size_t> sizes = get_sizes();
    for(auto size : sizes)
    {
        /*
        thrust::host_vector<T> h_data = get_random_data<T>(
                size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
                */
        thrust::host_vector<T> h_data(size);
        thrust::device_vector<T> d_data = h_data;

        ASSERT_EQ(h_data, d_data);

        auto r0 = thrust::reduce(
                h_data.begin(), h_data.end()
                );

        auto f0 = thrust::async::reduce(
                d_data.begin(), d_data.end()
                );

        auto r1 = std::move(f0).get();

        ASSERT_EQ(r0, r1);
    }
}
#endif

#if 0
template <typename T>
struct test_async_reduce_with_policy
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    auto r0 = thrust::reduce(
      h0_data.begin(), h0_data.end()
    );

    auto f0 = thrust::async::reduce(
      thrust::device, d0_data.begin(), d0_data.end()
    );

    auto r1 = std::move(f0).get();

    ASSERT_EQUAL(r0, r1);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_reduce_with_policy
, NumericTypes
> test_async_reduce_with_policy_instance;

template <typename T>
struct test_async_reduce_with_init
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    T const init = unittest::random_integer<T>();

    auto r0 = thrust::reduce(
      h0_data.begin(), h0_data.end(), init
    );

    auto f0 = thrust::async::reduce(
      d0_data.begin(), d0_data.end(), init
    );

    auto r1 = std::move(f0).get();

    ASSERT_EQUAL(r0, r1);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_reduce_with_init
, NumericTypes
> test_async_reduce_with_init_instance;

template <typename T>
struct test_async_reduce_with_policy_init
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    T const init = unittest::random_integer<T>();

    auto r0 = thrust::reduce(
      h0_data.begin(), h0_data.end(), init
    );

    auto f0 = thrust::async::reduce(
      thrust::device, d0_data.begin(), d0_data.end(), init
    );

    auto r1 = std::move(f0).get();

    ASSERT_EQUAL(r0, r1);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_reduce_with_policy_init
, NumericTypes
> test_async_reduce_with_policy_init_instance;

template <typename T>
struct test_async_reduce_with_init_op
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    T const init = unittest::random_integer<T>();
    custom_plus<T> op{};

    auto r0 = thrust::reduce(
      h0_data.begin(), h0_data.end(), init, op
    );

    auto f0 = thrust::async::reduce(
      d0_data.begin(), d0_data.end(), init, op
    );

    auto r1 = std::move(f0).get();

    ASSERT_EQUAL(r0, r1);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_reduce_with_init_op
, NumericTypes
> test_async_reduce_with_init_op_instance;

template <typename T>
struct test_async_reduce_with_policy_init_op
{
  __host__
  void operator()(std::size_t n)
  {
    thrust::host_vector<T>   h0_data(unittest::random_integers<T>(n));
    thrust::device_vector<T> d0_data(h0_data);

    ASSERT_EQUAL(h0_data, d0_data);

    T const init = unittest::random_integer<T>();
    custom_plus<T> op{};

    auto r0 = thrust::reduce(
      h0_data.begin(), h0_data.end(), init, op
    );

    auto f0 = thrust::async::reduce(
      thrust::device, d0_data.begin(), d0_data.end(), init, op
    );

    auto r1 = std::move(f0).get();

    ASSERT_EQUAL(r0, r1);
  }
};
// TODO: Switch to `DECLARE_VARIABLE_UNITTEST` when we add `custom_numeric` to
// the list of types it covers.
VariableUnitTest<
  test_async_reduce_with_policy_init_op
, NumericTypes
> test_async_reduce_with_policy_init_op_instance;

// TODO: Async copy then reduce.

// TODO: Device-side reduction usage.

// TODO: Make random_integers more generic.

#endif // THRUST_CPP_DIALECT >= 2011

#endif
