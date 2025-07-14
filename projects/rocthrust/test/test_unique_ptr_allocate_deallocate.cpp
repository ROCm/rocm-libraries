
#include <thrust/unique_ptr.h>
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

TEST(UniquePtrAllocDeallocTests, TestUniquePtrCmp)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Pointers of same type
    {
        thrust::unique_ptr<int> p1 = thrust::make_unique<int>(1);
        thrust::unique_ptr<int> p2 = thrust::make_unique<int>(2);

        int* ptr1 = p1.get_raw();
        int* ptr2 = p2.get_raw();

        ASSERT_EQ((p1 == p2), (ptr1 == ptr2));
        ASSERT_EQ((p1 != p2), (ptr1 != ptr2));
        ASSERT_EQ((p1 < p2), (ptr1 < ptr2));
        ASSERT_EQ((p1 <= p2), (ptr1 <= ptr2));
        ASSERT_EQ((p1 > p2), (ptr1 > ptr2));
        ASSERT_EQ((p1 >= p2), (ptr1 >= ptr2));
    }

    // Pointers of different type
    {
        thrust::unique_ptr<int>   p1 = thrust::make_unique<int>(1);
        thrust::unique_ptr<float> p2 = thrust::make_unique<float>(2.2);

        int*   ptr1 = p1.get_raw();
        float* ptr2 = p2.get_raw();

        ASSERT_EQ((p1 == p2), (static_cast<void*>(ptr1) == static_cast<void*>(ptr2)));
        ASSERT_EQ((p1 != p2), (static_cast<void*>(ptr1) != static_cast<void*>(ptr2)));
        ASSERT_EQ((p1 < p2), (static_cast<void*>(ptr1) < static_cast<void*>(ptr2)));
        ASSERT_EQ((p1 <= p2), (static_cast<void*>(ptr1) <= static_cast<void*>(ptr2)));
        ASSERT_EQ((p1 > p2), (static_cast<void*>(ptr1) > static_cast<void*>(ptr2)));
        ASSERT_EQ((p1 >= p2), (static_cast<void*>(ptr1) >= static_cast<void*>(ptr2)));
    }

    // Default-constructed pointers of same type
    {
        const thrust::unique_ptr<int> p1;
        const thrust::unique_ptr<int> p2;

        ASSERT_EQ(p1, p2);
    }

    // Default-constructed pointers of different type
    {
        const thrust::unique_ptr<int>   p1;
        const thrust::unique_ptr<float> p2;

        ASSERT_EQ(p1, p2);
    }
}

TEST(UniquePtrAllocDeallocTests, TestUniquePtrNullptr)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Test with a non-null unqiue_ptr
    {
        const thrust::unique_ptr<int> p = thrust::make_unique<int>(1);

        ASSERT_NE(p, nullptr);
        ASSERT_LT(nullptr, p);
        ASSERT_LE(nullptr, p);
        ASSERT_GT(p, nullptr);
        ASSERT_GE(p, nullptr);
    }

    // Test with null unique_ptr
    {
        const thrust::unique_ptr<int> p;

        ASSERT_EQ(p, nullptr);
        ASSERT_LE(p, nullptr);
        ASSERT_LE(nullptr, p);
        ASSERT_GE(p, nullptr);
        ASSERT_GE(nullptr, p);
    }
}
