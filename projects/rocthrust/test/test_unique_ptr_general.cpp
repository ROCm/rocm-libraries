
// #include <thrust/unique_ptr.h>
#include "hip/hip_runtime.h"
#include "test_param_fixtures.hpp"
#include "test_utils.hpp"


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