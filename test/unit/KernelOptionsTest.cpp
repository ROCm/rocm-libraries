#include <rocRoller/Context.hpp>
#include <rocRoller/KernelOptions.hpp>

#include "GenericContextFixture.hpp"

using namespace rocRoller;

class KernelOptionsTest : public GenericContextFixture
{
};

TEST_F(KernelOptionsTest, ToString)
{
    KernelOptions Test;
    std::string   output = Test.toString();

    EXPECT_THAT(output, testing::HasSubstr("Kernel Options:"));
    EXPECT_THAT(output, testing::HasSubstr("  logLevel:"));
    EXPECT_THAT(output, testing::HasSubstr("  alwaysWaitAfterLoad:"));
    EXPECT_THAT(output, testing::HasSubstr("  alwaysWaitBeforeBranch:"));
}
