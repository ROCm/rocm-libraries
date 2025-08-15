
// #include <thrust/unique_ptr.h>
#include "hip/hip_runtime.h"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>


TESTS_DEFINE(UniquePtrGeneralTests, NumericalTestsParams);

TEST(UniquePtrGeneralTests, TestUniquePtrSwap)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Test swap for single objects
    {
        thrust::unique_ptr<int> p1 = thrust::make_unique<int>(11);
        thrust::unique_ptr<int> p2 = thrust::make_unique<int>(22);

        int* raw_p1 = p1.get_raw();
        int* raw_p2 = p2.get_raw();

        p1.swap(p2);
        ASSERT_EQ(p1.get_raw(), raw_p2);
        ASSERT_EQ(p2.get_raw(), raw_p1);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrMoveAsgn)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Move assignment for single objects
    {
        void* raw_addr = nullptr;

        thrust::unique_ptr<int> p1 = thrust::make_unique<int>(1);
        int* raw_p1 = p1.get_raw();
        {
            thrust::unique_ptr<int> p2 = thrust::make_unique<int>(2);
            ASSERT_NE(p1.get_raw(), nullptr);
            ASSERT_NE(p2.get_raw(), nullptr);

            p2 = std::move(p1);

            ASSERT_EQ(p2.get_raw(), raw_p1);
            ASSERT_EQ(p1.get_raw(), nullptr);

            raw_addr = static_cast<void*>(p2.get_raw());
        }
        size_t dummy = 0;
        hipError_t st = hipMemPtrGetInfo(raw_addr, &dummy);
        ASSERT_EQ(st, hipErrorInvalidValue);
    }  
}

TEST(UniquePtrGeneralTests, TestUnqiuePtrSelfMoveAsgn)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Self-move for single object
    {
        thrust::unique_ptr<int> p = thrust::make_unique<int>(1);
        int* raw_p = p.get_raw();
        p = std::move(p);
        ASSERT_EQ(p.get_raw(), raw_p);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrNullAsgn)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // NULL assignment for single object
    void* raw_addr = nullptr;
    size_t sz = 0;
    {
        thrust::unique_ptr<int> p = thrust::make_unique<int>(1);
        ASSERT_NE(p, nullptr);

        hipError_t st = hipMemPtrGetInfo(static_cast<void*>(p.get_raw()), &sz);
        ASSERT_EQ(st, hipSuccess);
        ASSERT_GE(sz, sizeof(int));
        raw_addr = static_cast<void*>(p.get_raw());

        p = nullptr;

        ASSERT_EQ(p, nullptr);

        size_t dummy = 0;
        st = hipMemPtrGetInfo(raw_addr, &dummy);
        ASSERT_EQ(st, hipErrorInvalidValue);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrDefaultCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Default constructed unique_ptr
    {
        thrust::unique_ptr<int> p;
        ASSERT_EQ(p, nullptr);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrMoveCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Move constructor for single object
    {
        thrust::unique_ptr<int> p1 = thrust::make_unique<int>(42);
        int* raw_p1 = p1.get_raw();
        thrust::unique_ptr<int> p2(std::move(p1));
        ASSERT_EQ(p2.get_raw(), raw_p1);
        ASSERT_EQ(p1, nullptr);
    }   
}

TEST(UniquePtrGeneralTests, TestUniquePtrNullptrCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object
    {
        thrust::unique_ptr<int> p(nullptr);
        ASSERT_EQ(p, nullptr);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrPointerCtor)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());

    // Single object, default deleter
    {
        thrust::device_ptr<int> dev_p = thrust::device_malloc<int>(1);

        thrust::unique_ptr<int> s(dev_p);
        ASSERT_EQ(s.get(), dev_p);
    }
}

TEST(UniquePtrGeneralTests, TestUniquePtrDtorNullptr)
{
    SCOPED_TRACE(testing::Message() << "with device_id= " << test::set_device_from_ctest());
    
    // Single object
    {
        void* raw_addr = nullptr;
        {
            thrust::unique_ptr<int> p(nullptr);
            ASSERT_EQ(p, nullptr);

            raw_addr = static_cast<void*>(p.get_raw());
        }
        size_t dummy = 0;
        hipError_t st = hipMemPtrGetInfo(raw_addr, &dummy);
        ASSERT_EQ(st, hipErrorInvalidValue);
    }
}