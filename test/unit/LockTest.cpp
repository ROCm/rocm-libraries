
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GenericContextFixture.hpp"
#include "SourceMatcher.hpp"

using namespace rocRoller;

namespace rocRollerTest
{
    class LockStateTest : public GenericContextFixture
    {
    };

    TEST(LockStateTest, Basic)
    {
        auto none_lock = Scheduling::LockState();
        auto scc_lock  = Scheduling::LockState(Scheduling::Dependency::SCC);
        auto vcc_lock  = Scheduling::LockState(Scheduling::Dependency::VCC);

        EXPECT_EQ(none_lock.isLocked(), false);
        EXPECT_EQ(scc_lock.isLocked(), true);
        EXPECT_EQ(vcc_lock.isLocked(), true);

        EXPECT_EQ(none_lock.getLockDepth(), 0);
        EXPECT_EQ(scc_lock.getLockDepth(), 1);
        EXPECT_EQ(vcc_lock.getLockDepth(), 1);

        EXPECT_EQ(none_lock.getDependency(), Scheduling::Dependency::None);
        EXPECT_EQ(scc_lock.getDependency(), Scheduling::Dependency::SCC);
        EXPECT_EQ(vcc_lock.getDependency(), Scheduling::Dependency::VCC);

        auto lock_inst = Instruction::Lock(Scheduling::Dependency::SCC, "Lock Instruction");
        none_lock.add(lock_inst);

        EXPECT_EQ(none_lock.isLocked(), true);
        EXPECT_EQ(none_lock.getLockDepth(), 1);

        auto unlock_inst   = Instruction::Unlock("Unlock Instruction");
        auto comment_isntr = Instruction::Comment("Comment Instruction");
        scc_lock.add(unlock_inst);
        vcc_lock.add(comment_isntr);

        EXPECT_EQ(scc_lock.isLocked(), false);
        EXPECT_EQ(vcc_lock.isLocked(), true);
        EXPECT_EQ(scc_lock.getDependency(), Scheduling::Dependency::None);
        EXPECT_EQ(scc_lock.getLockDepth(), 0);

        vcc_lock.add(unlock_inst);
        EXPECT_EQ(vcc_lock.isLocked(), false);

        EXPECT_THROW(scc_lock.add(unlock_inst), FatalError);
        EXPECT_THROW(vcc_lock.isValid(true), FatalError);
        EXPECT_NO_THROW(vcc_lock.isValid(false));

        EXPECT_THROW({ auto l = Scheduling::LockState(Scheduling::Dependency::Unlock); },
                     FatalError);
        EXPECT_THROW({ auto l = Scheduling::LockState(Scheduling::Dependency::Count); },
                     FatalError);
    }
}
