// Google Test
#include <gtest/gtest.h>

#include <thrust/advance.h>
#include <thrust/sequence.h>

// HIP API
#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#define HIP_CHECK(condition) ASSERT_EQ(condition, hipSuccess)
#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

#include "test_utils.hpp"
#include "test_assertions.hpp"

template<class InputType>
struct Params
{
    using input_type = InputType;
};

template<class Params> class AdvanceVectorTests : public ::testing::Test
{
public:
    using input_type = typename Params::input_type;
};

typedef ::testing::Types<
        Params<thrust::device_vector<short>>,
        Params<thrust::device_vector<int>>,
        Params<thrust::host_vector<short>>,
        Params<thrust::host_vector<int>>
> VectorParams;

TYPED_TEST_CASE(AdvanceVectorTests, VectorParams);

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC

// TODO expand this with other iterator types (forward, bidirectional, etc.)

TYPED_TEST(AdvanceVectorTests, TestAdvance)
{
    using Vector = typename TestFixture::input_type;
    using T = typename Vector::value_type;

    typedef typename Vector::iterator Iterator;

    Vector v(100);
    thrust::sequence(v.begin(), v.end());

    Iterator i = v.begin();

    thrust::advance(i, 7);

    ASSERT_EQ(*i, T(7));
    
    thrust::advance(i, 13);

    ASSERT_EQ(*i, T(20));
    
    thrust::advance(i, -10);

    ASSERT_EQ(*i, T(10));
}

#endif // THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HCC