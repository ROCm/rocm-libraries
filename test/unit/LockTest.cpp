/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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
        auto none_lock = Scheduling::LockState(m_context);
        auto scc_lock  = Scheduling::LockState(m_context, Scheduling::Dependency::SCC);
        auto vcc_lock  = Scheduling::LockState(m_context, Scheduling::Dependency::VCC);

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

        EXPECT_THROW({ auto l = Scheduling::LockState(m_context, Scheduling::Dependency::Unlock); },
                     FatalError);
        EXPECT_THROW({ auto l = Scheduling::LockState(m_context, Scheduling::Dependency::Count); },
                     FatalError);
    }
}
