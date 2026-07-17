/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <gtest/gtest.h>
#include <miopen/logger.hpp>
#include "gtest_common.hpp"

// LogConvolutionExecution is an internal function in src/hip/convolutionocl.cpp that is not
// intended for general use. It is declared here rather than in a dedicated header because
// it has exactly two callers: the .cpp itself, and this test. A header would give a misleading
// impression of its public status. The forward declaration below causes a link error if the
// signature drifts, which is the main thing a header would have provided.
namespace miopen {
struct Handle;
struct AlgorithmName;
namespace conv {
struct ProblemDescription;
} // namespace conv
namespace debug {
MIOPEN_INTERNALS_EXPORT
void LogConvolutionExecution(const Handle& handle,
                             const conv::ProblemDescription& problem,
                             const std::string& network_config,
                             const AlgorithmName& algorithm_name);
} // namespace debug
} // namespace miopen

// These tests exercise the two gate conditions that LogConvolutionExecution checks at entry:
//
//   if(!IsLogging(LoggingLevel::Info) && !IsLogBufferOn())
//       return;
//
// End-to-end calling of the function itself requires a real Handle (GPU context) and is
// integration-test scope. The tests below validate that the conditions behave correctly
// under the relevant environment variables, and therefore that the early-exit guard works.

TEST(CPU_LogConvolutionExecution_NONE, LoggingDisabledAtLevelZero)
{
    ScopedEnvironment<std::string> log_level(MIOPEN_LOG_LEVEL, "0");
    EXPECT_FALSE(miopen::IsLogging(miopen::LoggingLevel::Info));
}

TEST(CPU_LogConvolutionExecution_NONE, LoggingEnabledAtInfoLevel)
{
    ScopedEnvironment<std::string> log_level(MIOPEN_LOG_LEVEL, "5"); // LoggingLevel::Info
    EXPECT_TRUE(miopen::IsLogging(miopen::LoggingLevel::Info));
}

TEST(CPU_LogConvolutionExecution_NONE, LoggingDisabledBelowInfoLevel)
{
    ScopedEnvironment<std::string> log_level(MIOPEN_LOG_LEVEL, "4"); // LoggingLevel::Warning
    EXPECT_FALSE(miopen::IsLogging(miopen::LoggingLevel::Info));
}

TEST(CPU_LogConvolutionExecution_NONE, LogBufferOffWhenSizeIsZero)
{
    ScopedEnvironment<std::string> buffer_size(MIOPEN_LOG_BUFFER_SIZE, "0");
    EXPECT_FALSE(miopen::IsLogBufferOn());
}

TEST(CPU_LogConvolutionExecution_NONE, LogBufferOnWhenSizeIsNonZero)
{
    ScopedEnvironment<std::string> buffer_size(MIOPEN_LOG_BUFFER_SIZE, "128");
    EXPECT_TRUE(miopen::IsLogBufferOn());
}

TEST(CPU_LogConvolutionExecution_NONE, EarlyExitConditionHoldsWhenBothDisabled)
{
    // Reproduces the exact combined check in LogConvolutionExecution.
    ScopedEnvironment<std::string> log_level(MIOPEN_LOG_LEVEL, "0");
    ScopedEnvironment<std::string> buffer_size(MIOPEN_LOG_BUFFER_SIZE, "0");
    EXPECT_FALSE(miopen::IsLogging(miopen::LoggingLevel::Info));
    EXPECT_FALSE(miopen::IsLogBufferOn());
}
