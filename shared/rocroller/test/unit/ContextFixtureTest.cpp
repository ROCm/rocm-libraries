/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include "ContextFixture.hpp"
#include <common/Utilities.hpp>
#include <gtest/gtest.h>

class ResourceLockFixtureTest : public ContextFixture
{
    rocRoller::ContextPtr createContext()
    {
        return nullptr;
    }
};

TEST_F(ResourceLockFixtureTest, HasTagInMiddleOPENMP_MiddleYes)
{
    EXPECT_NO_THROW(normL2(std::vector<int>{1, 2, 3}));
}

TEST_F(ResourceLockFixtureTest, OPENMP_Prefix)
{
    EXPECT_NO_THROW(normL2(std::vector<int>{1, 2, 3}));
}

TEST_F(ResourceLockFixtureTest, GPU_OPENMP_Prefix)
{
    EXPECT_NO_THROW(normL2(std::vector<int>{1, 2, 3}));
}

TEST_F(ResourceLockFixtureTest, NoTag)
{
    EXPECT_THROW(normL2(std::vector<int>{1, 2, 3}), rocRoller::FatalError);
}