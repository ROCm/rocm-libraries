#pragma once

#include "../driver.hpp"

template <template <class...> class Driver, typename TestCase, typename Check>
void invoke_with_params(Check&& check)
{
    for(const auto& test_value : TestCase::GetParam())
    {
        std::vector<std::string> tokens = get_args(test_value);
        std::vector<const char*> ptrs;
        ptrs.reserve(tokens.size() + 1);
        ptrs.emplace_back(TestCase::fp_args.data());

        std::transform(tokens.begin(), tokens.end(), std::back_inserter(ptrs), [](const auto& str) {
            return str.data();
        });

        testing::internal::CaptureStderr();
        test_drive<Driver>(ptrs.size(), ptrs.data(), "unnamed");
        check(testing::internal::GetCapturedStderr());
    }
}
