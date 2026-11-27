/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// Argument checks of the plan cache functions of a handle: hiptensorHandleResizePlanCache,
// hiptensorHandleWritePlanCacheToFile, and hiptensorHandleReadPlanCacheFromFile. No kernel is
// launched.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <random>
#include <sstream>
#include <string>

#include <gtest/gtest.h>
#include <hiptensor/hiptensor.h>

namespace
{
    // A file in the test temp directory, unique to this process and test, removed on
    // destruction. ctest registers this binary once per test category, so several copies can
    // run at the same time and must not share files.
    struct TempFile
    {
        std::string mPath;

        TempFile()
        {
            std::ostringstream name;
            name << ::testing::TempDir() << "hiptensor_"
                 << ::testing::UnitTest::GetInstance()->current_test_info()->name() << "_"
                 << std::hex << std::random_device{}() << std::random_device{}() << ".bin";
            mPath = name.str();
        }
        ~TempFile()
        {
            std::remove(mPath.c_str());
        }
        TempFile(TempFile const&)            = delete;
        TempFile& operator=(TempFile const&) = delete;

        char const* c_str() const
        {
            return mPath.c_str();
        }
    };

    // Sets an environment variable for the lifetime of the object and then restores it.
    class ScopedEnv
    {
    public:
        ScopedEnv(char const* name, char const* value)
            : mName(name)
        {
            if(char const* old = std::getenv(name))
            {
                mOld = old;
            }
            set(value);
        }
        ~ScopedEnv()
        {
            set(mOld ? mOld->c_str() : nullptr);
        }
        ScopedEnv(ScopedEnv const&)            = delete;
        ScopedEnv& operator=(ScopedEnv const&) = delete;

    private:
        void set(char const* value)
        {
#ifdef _WIN32
            _putenv_s(mName.c_str(), value ? value : "");
#else
            if(value)
            {
                setenv(mName.c_str(), value, 1);
            }
            else
            {
                unsetenv(mName.c_str());
            }
#endif
        }

        std::string                mName;
        std::optional<std::string> mOld;
    };

    class PlanCacheApiTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
        }

        void TearDown() override
        {
            if(mHandle != nullptr)
            {
                hiptensorDestroy(mHandle);
            }
        }

        hiptensorHandle_t mHandle = nullptr;
    };
} // namespace

TEST_F(PlanCacheApiTest, NullHandleReturnsNotInitialized)
{
    TempFile file;
    uint32_t numCachelinesRead = 0;
    EXPECT_EQ(hiptensorHandleResizePlanCache(nullptr, 8), HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(nullptr, file.c_str()),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(nullptr, file.c_str(), &numCachelinesRead),
              HIPTENSOR_STATUS_NOT_INITIALIZED);
}

TEST_F(PlanCacheApiTest, ZeroEntriesReturnsInvalidValue)
{
    EXPECT_EQ(hiptensorHandleResizePlanCache(mHandle, 0), HIPTENSOR_STATUS_INVALID_VALUE);
}

// The checks don't depend on whether the plan cache is enabled.
TEST_F(PlanCacheApiTest, NullFileNameReturnsInvalidValue)
{
    uint32_t numCachelinesRead = 0;
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(mHandle, nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(mHandle, nullptr, &numCachelinesRead),
              HIPTENSOR_STATUS_INVALID_VALUE);
}

// A null count is rejected even when the file can be read.
TEST_F(PlanCacheApiTest, NullNumCachelinesReadReturnsInvalidValue)
{
    TempFile file;
    ASSERT_EQ(hiptensorHandleWritePlanCacheToFile(mHandle, file.c_str()), HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(mHandle, file.c_str(), nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);
}

// With the plan cache disabled, nothing is read and no cache line is reported.
TEST(PlanCacheDisabledTest, ReadReportsNoCachelines)
{
    ScopedEnv         disable("HIPTENSOR_DISABLE_PLAN_CACHE", "1");
    hiptensorHandle_t handle = nullptr;
    ASSERT_EQ(hiptensorCreate(&handle), HIPTENSOR_STATUS_SUCCESS);

    // The file does not exist: with the plan cache enabled the read would fail.
    TempFile file;
    uint32_t numCachelinesRead = 12345u;
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, file.c_str(), &numCachelinesRead),
              HIPTENSOR_STATUS_SUCCESS);
    EXPECT_EQ(numCachelinesRead, 0u);

    EXPECT_EQ(hiptensorHandleResizePlanCache(handle, 0), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorHandleWritePlanCacheToFile(handle, nullptr), HIPTENSOR_STATUS_INVALID_VALUE);
    EXPECT_EQ(hiptensorHandleReadPlanCacheFromFile(handle, file.c_str(), nullptr),
              HIPTENSOR_STATUS_INVALID_VALUE);

    hiptensorDestroy(handle);
}
