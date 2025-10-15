
#include <thrust/unique_ptr.h>
#include <thrust/device_delete.h>
#include "hip/hip_runtime.h"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"

class A {
public:    
    __host__ __device__ A() : finished(nullptr), val_a(0) {}
    __host__ __device__ A(int num): finished(nullptr), val_a(num) {}
    __host__ __device__ A(thrust::device_ptr<bool> dev_p, int num): finished(dev_p), val_a(num) {}

    __host__ __device__ ~A() {
    #if defined(__HIP_DEVICE_COMPILE__)
        if(finished)
            *finished = true; 
    #endif
    }
 
    thrust::device_ptr<bool> finished;
    int val_a;
}; 

class B : public A
{
public:
    __host__ __device__ B() : A() {}
    __host__ __device__ B(int num) : A(num) {}
};

TESTS_DEFINE(UniquePtrAllocDeallocTests, NumericalTestsParams);

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.create/make_unique.single.pass.cpp
TYPED_TEST(UniquePtrAllocDeallocTests, TestUniquePtrMakeUnique)
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

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.create/make_unique.single.pass.cpp
TEST(UniquePtrAllocDeallocTests, TestUniquePtrMakeUniqueUserType)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::unique_ptr<A> p = thrust::make_unique<A>(7);
    A host_p = *p;

    ASSERT_EQ(host_p.val_a, 7);
}

// Thrust-specific test for deleter behavior.
TEST(UniquePtrAllocDeallocTests, TestUniquePtrDltr)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    thrust::device_ptr<bool> finished = thrust::device_new<bool>(1);
    *finished = false;
    {
        thrust::unique_ptr<A> p = thrust::make_unique<A>(finished, 18);
        ASSERT_NE(p, nullptr);
    }
    ASSERT_EQ(*finished, true);
    thrust::device_delete(finished);
}

// Based on llvm-project/libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.special/cmp.pass.cpp
TEST(UniquePtrAllocDeallocTests, TestUniquePtrCmp)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Pointers of same type
    {
        thrust::unique_ptr<A> p1 = thrust::make_unique<A>(1);
        thrust::unique_ptr<A> p2 = thrust::make_unique<A>(2);

        A* ptr1 = p1.get_raw();
        A* ptr2 = p2.get_raw();

        ASSERT_FALSE(p1 == p2);
        ASSERT_TRUE(p1 != p2);
        ASSERT_EQ((p1 < p2), (ptr1 < ptr2));
        ASSERT_EQ((p1 <= p2), (ptr1 <= ptr2));
        ASSERT_EQ((p1 > p2), (ptr1 > ptr2));
        ASSERT_EQ((p1 >= p2), (ptr1 >= ptr2));
    }

    // Pointers of different type
    {
        thrust::unique_ptr<A> p1(thrust::device_new<A>(1));
        thrust::unique_ptr<B> p2(thrust::device_new<B>(2));

        A* ptr1 = p1.get_raw();
        B* ptr2 = p2.get_raw();

        ASSERT_FALSE(p1 == p2);
        ASSERT_TRUE(p1 != p2);
        ASSERT_EQ((p1 < p2), ((ptr1) < (ptr2)));
        ASSERT_EQ((p1 <= p2), ((ptr1) <= (ptr2)));
        ASSERT_EQ((p1 > p2), ((ptr1) > (ptr2)));
        ASSERT_EQ((p1 >= p2), ((ptr1) >= (ptr2)));
    }

    // Default-constructed pointers of same type
    {
        const thrust::unique_ptr<A> p1;
        const thrust::unique_ptr<A> p2;

        ASSERT_EQ(p1, p2);
    }

    // Default-constructed pointers of different type
    {
        const thrust::unique_ptr<A> p1;
        const thrust::unique_ptr<B> p2;

        ASSERT_EQ(p1, p2);
    }
}

// Based on libcxx/test/std/utilities/smartptr/unique.ptr/unique.ptr.special/cmp_nullptr.pass.cpp
TEST(UniquePtrAllocDeallocTests, TestUniquePtrCmpNullptr)
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
