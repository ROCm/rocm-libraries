
// #include <thrust/unique_ptr.h>
#include "hip/hip_runtime.h"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

class A {
public:    
    __host__ __device__ A() {}
    __host__ __device__ A(int num): val_a(num) {}

    __host__ __device__ ~A() {}
 
    int val_a;
}; 

TESTS_DEFINE(UniquePtrAllocDeallocTests, NumericalTestsParams);

TYPED_TEST(UniquePtrAllocDeallocTests, TestMakeUnique)
{
    using T = typename TestFixture::input_type;
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    T initial_value = T(1);
    if(std::is_floating_point<T>::value)
    {
        initial_value = T(7.71);
    }

    thrust::unique_ptr<T> p1 = thrust::make_unique<T>(initial_value);
    ASSERT_EQ(*p1, initial_value);

    p1 = thrust::make_unique<T>();
    ASSERT_EQ(*p1, T {});
}

TEST(UniquePtrAllocDeallocTests, TestMakeUniqueUserType)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::unique_ptr<A> p = thrust::make_unique<A>(7);
    A host_p = *p;

    ASSERT_EQ(host_p.val_a, 7);
}

TEST(UniquePtrAllocDeallocTests, TestUniquePtrDltr)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    void *raw_addr = nullptr;
    size_t sz = 0;
    {
        thrust::unique_ptr<A> p = thrust::make_unique<A>();
        ASSERT_NE(p, nullptr);

        hipError_t st = hipMemPtrGetInfo(static_cast<void*>(p.get_raw()), &sz);
        ASSERT_EQ(st, hipSuccess);
        ASSERT_GE(sz, sizeof(A));

        raw_addr = static_cast<void*>(p.get_raw());
    }
    
    size_t dummy = 0;
    hipError_t st = hipMemPtrGetInfo(raw_addr, &dummy);
    ASSERT_EQ(st, hipErrorInvalidValue);
}
