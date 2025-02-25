
#include "CustomAssertions.hpp"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("CurrentTestHasTag fails test on duplicate test name", "[meta-test][!shouldfail]")
{
    // This would return true if it didn't fail an assertion due to duplicate test names.
    REQUIRE(rocRollerTests::CurrentTestHasTag("meta-test"));
}

TEST_CASE("CurrentTestHasTag fails test on duplicate test name", "[meta-test][asdf][!shouldfail]")
{

    // This would return true if it didn't fail an assertion due to duplicate test names.
    REQUIRE_FALSE(rocRollerTests::CurrentTestHasTag("fdsa"));
}

TEST_CASE("CurrentTestHasTag", "[meta-test][asdf]")
{
    REQUIRE(rocRollerTests::CurrentTestHasTag("meta-test"));
    REQUIRE(rocRollerTests::CurrentTestHasTag("asdf"));
    REQUIRE_FALSE(rocRollerTests::CurrentTestHasTag("fdsa"));
}

TEST_CASE("REQUIRE_TEST_TAG works", "[meta-test][asdf]")
{
    REQUIRE_TEST_TAG("asdf");
    REQUIRE_TEST_TAG("meta-test");
}

TEST_CASE("REQUIRE_TEST_TAG fails the test", "[meta-test][asdf][!shouldfail]")
{
    REQUIRE_TEST_TAG("asdf");
    REQUIRE_TEST_TAG("fdsa");
}
