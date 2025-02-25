

#include <catch2/catch_test_case_info.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/interfaces/catch_interfaces_testcase.hpp>

#include <algorithm>
#include <vector>

namespace rocRollerTests
{

    std::vector<Catch::TestCaseInfo*> const& getSortedTestInfos()
    {
        static auto rv = []() {
            auto infos = Catch::getRegistryHub().getTestCaseRegistry().getAllInfos();

            // This function shouldn't return any nullptrs.
            for(auto const& info : infos)
                REQUIRE(info != nullptr);

            std::sort(infos.begin(), infos.end(), [](auto lhs, auto rhs) { return *lhs < *rhs; });

            return infos;
        }();

        return rv;
    }

    bool CurrentTestHasTag(std::string const& tagName)
    {
        Catch::Tag tag(tagName);

        auto testName = Catch::getResultCapture().getCurrentTestName();

        auto infos = getSortedTestInfos();

        auto first_match = std::lower_bound(
            infos.begin(),
            infos.end(),
            testName,
            [](Catch::TestCaseInfo* info, std::string const& name) { return info->name < name; });

        // If the test name is somehow not found in the list of tests.
        REQUIRE(first_match != infos.end());

        auto next_iter = first_match;
        ++next_iter;
        if(next_iter != infos.end())
        {
            // This assertion requires that the current test be the only one with that exact name.
            REQUIRE((*next_iter)->name != testName);
        }

        bool found = false;
        for(auto const& testTag : (*first_match)->tags)
        {
            if(testTag == tag)
            {
                return true;
            }
        }

        auto current_test_tags = *first_match;
        CAPTURE(current_test_tags);

        return false;
    }

}
