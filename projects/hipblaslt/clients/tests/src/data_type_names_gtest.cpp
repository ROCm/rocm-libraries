// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// The API log names the data types that the library accepts.

#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

#include <cstdlib>
#include <optional>
#include <string>

namespace
{
    // Sets an environment variable, or unsets it for nullptr, until the end of the scope.
    class ScopedEnv
    {
    public:
        ScopedEnv(const char* name, const char* value)
            : m_name(name)
        {
            if(const char* old = std::getenv(name))
                m_saved = old;
            set(value);
        }

        ~ScopedEnv()
        {
            set(m_saved ? m_saved->c_str() : nullptr);
        }

    private:
        void set(const char* value)
        {
#ifdef _WIN32
            _putenv_s(m_name, value ? value : "");
#else
            if(value)
                setenv(m_name, value, 1);
            else
                unsetenv(m_name);
#endif
        }

        const char*                m_name;
        std::optional<std::string> m_saved;
    };

    void createLayoutAndExit(hipDataType type)
    {
        hipblasLtMatrixLayout_t layout = nullptr;
        if(hipblasLtMatrixLayoutCreate(&layout, type, 16, 16, 16) != HIPBLAS_STATUS_SUCCESS)
            std::exit(1);
        hipblasLtMatrixLayoutDestroy(layout);
        std::exit(0);
    }

    TEST(DataTypeNames, smoke_LayoutLogNamesInt32AndComplexTypes)
    {
        // The logger reads its environment once per process, so each case runs in a
        // newly started copy of this binary and the test matches the copy's stderr.
        GTEST_FLAG_SET(death_test_style, "threadsafe");
        ScopedEnv level("HIPBLASLT_LOG_LEVEL", "5");
        ScopedEnv file("HIPBLASLT_LOG_FILE", nullptr);

        EXPECT_EXIT(createLayoutAndExit(HIP_R_32I), ::testing::ExitedWithCode(0), "type=R_32I ");
        EXPECT_EXIT(createLayoutAndExit(HIP_C_32F), ::testing::ExitedWithCode(0), "type=C_32F ");
        EXPECT_EXIT(createLayoutAndExit(HIP_C_64F), ::testing::ExitedWithCode(0), "type=C_64F ");
    }
} // namespace
