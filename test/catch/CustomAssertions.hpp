
#pragma once

#include <string>

#include <catch2/catch_test_macros.hpp>

namespace rocRollerTests
{
    bool CurrentTestHasTag(std::string const& tagName);
}

/**
 * Asserts that the currently running test has a tag with the provided name. Meant to be called
 * from test helper functions to enforce rules about tests that use certain functionality.
 *
 * - Do not include the square brackets in the name of the tag.
 * e.g. `REQUIRE_TEST_TAG("gpu");` will check that the current test has the "[gpu]" tag.
 * - Does not resolve any tag aliases, so you must provide the canonical name of the tag.
 */
#define REQUIRE_TEST_TAG(name)                            \
    do                                                    \
    {                                                     \
        CAPTURE(name);                                    \
        REQUIRE(rocRollerTests::CurrentTestHasTag(name)); \
    } while(0)
