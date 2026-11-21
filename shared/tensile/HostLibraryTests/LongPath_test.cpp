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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <gtest/gtest.h>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/LongPath.hpp>
#include <Tensile/Tensile.hpp>

#ifdef TENSILE_USE_HIP
#include <Tensile/hip/HipSolutionAdapter.hpp>
#endif

#include "TestData.hpp"

#include <chrono>
#include <filesystem>
#include <random>
#include <string>

using namespace Tensile;

namespace
{
    // A temporary directory nested deep enough that every file path in it is
    // at least 300 characters, past the Windows MAX_PATH limit of 260.
    class LongPathLoading : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Unique per run, so concurrent test processes never share a root
            auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
            m_root     = std::filesystem::temp_directory_path()
                     / ("tensile_long_path_" + std::to_string(std::random_device{}()) + "_"
                        + std::to_string(stamp));
            m_dir = m_root;
            for(int i = 0; i < 5; ++i)
                m_dir /= std::string(60, static_cast<char>('a' + i));

            std::filesystem::create_directories(longPathSafe(m_dir.string()));
            ASSERT_GE(m_dir.string().size(), 300u);
        }

        void TearDown() override
        {
            std::error_code ec;
#ifdef _WIN32
            // The root itself is short, so longPathSafe() would leave it as is,
            // and remove_all would then reach the deep entries under it through
            // ordinary paths. Remove it through its \\?\ form instead.
            std::wstring root = m_root.lexically_normal().native();
            if(root.rfind(L"\\\\", 0) == 0)
                root = L"\\\\?\\UNC\\" + root.substr(2);
            else
                root = L"\\\\?\\" + root;
            std::filesystem::remove_all(root, ec);
#else
            std::filesystem::remove_all(m_root, ec);
#endif
        }

        // Copies `source` into the deep directory and returns the copy's path
        // in its ordinary (not \\?\) form, as a library would build it.
        std::string copyIn(std::filesystem::path const& source)
        {
            auto target = m_dir / source.filename();
            std::filesystem::copy_file(source,
                                       longPathSafe(target.string()),
                                       std::filesystem::copy_options::overwrite_existing);
            return target.string();
        }

        std::filesystem::path m_root;
        std::filesystem::path m_dir;
    };
} // namespace

TEST(LongPathSafe, ShortPathUnchanged)
{
    // Absolute, so the result can't depend on the working directory's depth
#ifdef _WIN32
    std::string const path = "C:/rocm/library/TensileLibrary_gfx942.dat";
#else
    std::string const path = "/opt/rocm/library/TensileLibrary_gfx942.dat";
#endif
    EXPECT_EQ(longPathSafe(path), std::filesystem::path(path));
}

#ifdef _WIN32
TEST(LongPathSafe, LongPathGetsVerbatimPrefix)
{
    // Still past MAX_PATH once ".." is resolved
    std::string const path = "C:/x/../" + std::string(300, 'a') + "/b/TensileLibrary.dat";
    auto const        safe = longPathSafe(path).native();
    EXPECT_EQ(safe, L"\\\\?\\C:\\" + std::wstring(300, L'a') + L"\\b\\TensileLibrary.dat");
    EXPECT_TRUE(isVerbatimPath(safe));
}

TEST(LongPathSafe, LongUncPathGetsVerbatimUncPrefix)
{
    std::string const path = "\\\\server\\share\\" + std::string(300, 'a') + "\\TensileLibrary.dat";
    EXPECT_EQ(longPathSafe(path).native(),
              L"\\\\?\\UNC\\server\\share\\" + std::wstring(300, L'a') + L"\\TensileLibrary.dat");
}
#else
TEST(LongPathSafe, LongPathUnchangedOffWindows)
{
    std::string const path = "/" + std::string(300, 'a') + "/../b/TensileLibrary.dat";
    EXPECT_EQ(longPathSafe(path), std::filesystem::path(path));
}
#endif

TEST_F(LongPathLoading, LoadsMessagePackLibrary)
{
    auto source = TestData::Instance().file("SampleTensileKernels", "dat");
    if(!std::filesystem::is_regular_file(source))
        GTEST_SKIP() << "msgpack test data not built: " << source;

    auto library = LoadLibraryFile<ContractionProblem>(copyIn(source));
    ASSERT_NE(library, nullptr);
}

#ifdef TENSILE_USE_HIP
TEST_F(LongPathLoading, LoadsCodeObjectFile)
{
    int deviceCount = 0;
    if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        GTEST_SKIP() << "no GPU";

    hipDeviceProp_t props;
    ASSERT_EQ(hipGetDeviceProperties(&props, 0), hipSuccess);
    std::string arch = props.gcnArchName;
    arch             = arch.substr(0, arch.find(':'));

    // Only a code object built for this GPU loads, so pick one by its arch suffix
    std::filesystem::path codeObject;
    for(auto const& candidate : TestData::Instance().glob("test_kernels_lite/library/*.*co"))
    {
        auto name = candidate.filename().string();
        auto pos  = name.rfind(arch);
        if(pos != std::string::npos && pos + arch.size() < name.size()
           && (name[pos + arch.size()] == '.' || name[pos + arch.size()] == '-'))
        {
            codeObject = candidate;
            break;
        }
    }
    if(codeObject.empty())
        GTEST_SKIP() << "no test code object built for " << arch;

    hip::SolutionAdapter adapter(false, "long path");
    EXPECT_EQ(adapter.loadCodeObjectFile(copyIn(codeObject)), hipSuccess);
}
#endif
