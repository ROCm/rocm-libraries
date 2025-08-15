
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