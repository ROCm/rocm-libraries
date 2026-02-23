// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

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

    TEST_F(LockStateTest, Basic)
    {
        auto lock_inst    = Instruction::Lock(Scheduling::Dependency::SCC, "Lock Instruction");
        auto unlock_inst  = Instruction::Unlock("Unlock Instruction");
        auto comment_inst = Instruction::Comment("Comment Instruction");

        auto lock_m0_inst   = Instruction::Lock(Scheduling::Dependency::M0, "Lock M0");
        auto unlock_m0_inst = Instruction::Unlock(Scheduling::Dependency::M0, "Lock M0");

        {
            auto none_lock = Scheduling::LockState(m_context);
            EXPECT_EQ(none_lock.isNonPreemptibleStream(0), false);
            EXPECT_EQ(none_lock.getLockDepth(0), 0);

            EXPECT_EQ(none_lock.getTopDependency(0), Scheduling::Dependency::None);

            // stream0:scc
            none_lock.add(lock_inst, 0);

            // trying to unlock the type of lock not held by stream0
            EXPECT_THROW(none_lock.add(unlock_m0_inst, 0), FatalError);
            // trying to lock out of order: m0 after scc
            EXPECT_THROW(none_lock.add(lock_m0_inst, 0), FatalError);

            // scc is a non-preemptible lock
            EXPECT_EQ(none_lock.isNonPreemptibleStream(0), true);
            EXPECT_EQ(none_lock.getLockDepth(0), 1);

            // stream0:
            none_lock.add(unlock_inst, 0);
            EXPECT_EQ(none_lock.isNonPreemptibleStream(0), false);
            EXPECT_EQ(none_lock.getLockDepth(0), 0);

            // stream0:m0
            none_lock.add(lock_m0_inst, 0);
            EXPECT_EQ(none_lock.getLockDepth(0), 1);

            // m0 is not a non-preemptible lock
            EXPECT_EQ(none_lock.isNonPreemptibleStream(0), false);

            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 0), true);
            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 1), true);

            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 0), true);
            // stream1 can't lock m0 as it is held by stream0
            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 1), false);

            // stream0:m0,m0
            none_lock.add(lock_m0_inst, 0);
            EXPECT_EQ(none_lock.getLockDepth(0), 2);

            EXPECT_EQ(none_lock.isNonPreemptibleStream(0), false);

            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 0), true);
            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 1), true);

            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 0), true);
            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 1), false);

            // stream0:m0
            none_lock.add(unlock_inst, 0);
            EXPECT_EQ(none_lock.getLockDepth(0), 1);

            EXPECT_EQ(none_lock.isNonPreemptibleStream(0), false);

            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 0), true);
            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 1), true);

            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 0), true);
            // stream1 can't acquire m0 as it is held by stream0
            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 1), false);

            //stream1:scc
            EXPECT_EQ(none_lock.isSchedulable(lock_inst, 1), true);
            none_lock.add(lock_inst, 1);

            EXPECT_THROW(none_lock.add(lock_inst, 0), FatalError);

            EXPECT_EQ(none_lock.getLockDepth(0), 1);
            EXPECT_EQ(none_lock.getLockDepth(1), 1);

            // scc is a non-preemptible lock
            EXPECT_EQ(none_lock.isNonPreemptibleStream(1), true);

            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 0), false);
            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 1), true);

            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 0), false);
            EXPECT_EQ(none_lock.isSchedulable(lock_inst, 0), false);
            // can't lock out of order: m0 after scc
            EXPECT_THROW(none_lock.isSchedulable(lock_m0_inst, 1), FatalError);

            // stream0:m0
            // can't add any instruction from stream0 until another stream holds a non-preemptible lock.
            EXPECT_THROW(none_lock.add(unlock_inst, 0), FatalError);
            EXPECT_EQ(none_lock.getLockDepth(0), 1);

            EXPECT_EQ(none_lock.isNonPreemptibleStream(1), true);

            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 0), false);
            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 1), true);

            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 0), false);
            EXPECT_EQ(none_lock.isSchedulable(lock_inst, 0), false);
            EXPECT_THROW(none_lock.isSchedulable(lock_m0_inst, 1), FatalError);

            none_lock.add(unlock_inst, 1);
            EXPECT_EQ(none_lock.getLockDepth(1), 0);

            EXPECT_EQ(none_lock.isNonPreemptibleStream(1), false);

            none_lock.add(unlock_inst, 0);
            EXPECT_EQ(none_lock.getLockDepth(0), 0);

            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 0), true);
            EXPECT_EQ(none_lock.isSchedulable(comment_inst, 1), true);

            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 0), true);
            EXPECT_EQ(none_lock.isSchedulable(lock_m0_inst, 1), true);
        }

        {
            auto scc_lock = Scheduling::LockState(m_context, Scheduling::Dependency::SCC);
            EXPECT_EQ(scc_lock.isNonPreemptibleStream(0), true);
            EXPECT_EQ(scc_lock.getLockDepth(0), 1);
            EXPECT_EQ(scc_lock.getTopDependency(0), Scheduling::Dependency::SCC);
            scc_lock.add(unlock_inst, 0);
            EXPECT_EQ(scc_lock.isNonPreemptibleStream(0), false);
            EXPECT_EQ(scc_lock.getTopDependency(0), Scheduling::Dependency::None);
            EXPECT_EQ(scc_lock.getLockDepth(0), 0);
            EXPECT_THROW(scc_lock.add(unlock_inst, 0), FatalError);
        }

        {
            auto vcc_lock = Scheduling::LockState(m_context, Scheduling::Dependency::VCC);
            EXPECT_EQ(vcc_lock.isNonPreemptibleStream(0), false);
            EXPECT_EQ(vcc_lock.getLockDepth(0), 1);
            EXPECT_EQ(vcc_lock.getTopDependency(0), Scheduling::Dependency::VCC);
            vcc_lock.add(comment_inst, 0);
            EXPECT_EQ(vcc_lock.isNonPreemptibleStream(0), false);
            vcc_lock.add(unlock_inst, 0);
            EXPECT_EQ(vcc_lock.isNonPreemptibleStream(0), false);
        }

        EXPECT_THROW({ auto l = Scheduling::LockState(m_context, Scheduling::Dependency::Count); },
                     FatalError);
    }
}
