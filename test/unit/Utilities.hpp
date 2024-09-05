
#pragma once

#include <common/Utilities.hpp>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#ifdef TEST
#undef TEST // Rely on TEST_F and TEST_P instead
#endif

MATCHER_P(HasHipSuccess, p, "")
{
    auto result = arg;
    if(result != hipSuccess)
    {
        std::string msg = hipGetErrorString(result);
        *result_listener << msg;
    }
    return result == hipSuccess;
}
