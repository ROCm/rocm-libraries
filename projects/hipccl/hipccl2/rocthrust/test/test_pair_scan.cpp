// Google Test
#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "test_assertions.hpp"

#include <thrust/pair.h>
#include <thrust/transform.h>
#include <thrust/scan.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>

#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC


#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

template<class InputType> struct Params
{
    using input_type = InputType;
};

template<class Params> class PairScanVariablesTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<char>,
        Params<unsigned char>,
        Params<short>,
        Params<unsigned short>,
        Params<int>,
        Params<unsigned int>,
        Params<float>
> TestVariableParams;

TYPED_TEST_CASE(PairScanVariablesTests, TestVariableParams);

struct make_pair_functor
{
  template<typename T1, typename T2>
  __host__ __device__
    thrust::pair<T1,T2> operator()(const T1 &x, const T2 &y)
  {
    return thrust::make_pair(x,y);
  } // end operator()()
}; // end make_pair_functor


struct add_pairs
{
  template <typename Pair1, typename Pair2>
  __host__ __device__
    Pair1 operator()(const Pair1 &x, const Pair2 &y)
  {
    return thrust::make_pair(x.first + y.first, x.second + y.second);
  } // end operator()
}; // end add_pairs


TYPED_TEST(PairScanVariablesTests, TestPairScan)
{
  using T = typename TestFixture::input_type;

  const std::vector<size_t> sizes = get_sizes();
  for(auto size : sizes)
  {
    typedef thrust::pair<T,T> P;

    thrust::host_vector<T> h_p1 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());
    thrust::host_vector<T> h_p2 = get_random_data<T>(size,
                                                     std::numeric_limits<T>::min(),
                                                     std::numeric_limits<T>::max());
    thrust::host_vector<P>   h_pairs(size);
    thrust::host_vector<P>   h_output(size);

    // zip up pairs on the host
    thrust::transform(h_p1.begin(), h_p1.end(), h_p2.begin(), h_pairs.begin(), make_pair_functor());

    thrust::device_vector<T> d_p1 = h_p1;
    thrust::device_vector<T> d_p2 = h_p2;
    thrust::device_vector<P> d_pairs = h_pairs;
    thrust::device_vector<P> d_output(size);

    P init = thrust::make_pair(13,13);

    // scan with plus
    thrust::inclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), add_pairs());
    thrust::inclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), add_pairs());
    ASSERT_EQ_QUIET(h_output, d_output);

    // scan with maximum (thrust issue #69)
    thrust::inclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), thrust::maximum<P>());
    thrust::inclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), thrust::maximum<P>());
    ASSERT_EQ_QUIET(h_output, d_output);

    // scan with plus
    thrust::exclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), init, add_pairs());
    thrust::exclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), init, add_pairs());
    ASSERT_EQ_QUIET(h_output, d_output);
    
    // scan with maximum (thrust issue #69)
    thrust::exclusive_scan(h_pairs.begin(), h_pairs.end(), h_output.begin(), init, thrust::maximum<P>());
    thrust::exclusive_scan(d_pairs.begin(), d_pairs.end(), d_output.begin(), init, thrust::maximum<P>());
    ASSERT_EQ_QUIET(h_output, d_output);
  }
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
