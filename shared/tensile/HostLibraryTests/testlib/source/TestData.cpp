/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "TestData.hpp"

#include <glob.h>
#include <unistd.h>

#include <stdexcept>
#include <vector>

#include <Tensile/Utils.hpp>

#if defined(_WIN32)
#include <windows.h>
#endif

TestData::operator bool() const
{
    return std::filesystem::is_directory(dataDir());
}

TestData TestData::Invalid()
{
    return TestData(invalid_data());
}

TestData TestData::Env(std::string const& varName)
{
    char* var = getenv(varName.c_str());

    if(var == nullptr)
        return Invalid();

    return TestData(var);
}

std::filesystem::path TestData::dataDir() const
{
    return m_dataDir;
}

std::filesystem::path TestData::file(std::string const& filename) const
{
    auto simple = dataDir() / filename;
    if(std::filesystem::is_regular_file(simple))
        return simple;

    auto datFile = file(filename, "dat");
    if(std::filesystem::is_regular_file(datFile))
        return datFile;

    auto yamlFile = file(filename, "yaml");
    if(std::filesystem::is_regular_file(yamlFile))
        return yamlFile;

    return simple;
}

std::filesystem::path TestData::file(std::string const& filename,
                                     std::string const& extension) const
{
    return dataDir() / (filename + "." + extension);
}

std::vector<std::filesystem::path> TestData::glob(std::string const& pattern) const
{
    std::string wholePattern = (dataDir() / pattern).string();

    glob_t result;
    result.gl_pathc = 0;
    result.gl_pathv = nullptr;
    result.gl_offs  = 0;

    int err = ::glob(wholePattern.c_str(), 0, nullptr, &result);

    // This way globfree will be called regardless of if an exception is thrown.
    std::shared_ptr<glob_t> guard(&result, globfree);

    if(err == GLOB_NOSPACE || err == GLOB_ABORTED)
        throw std::runtime_error(Tensile::concatenate("Glob ", wholePattern, " failed."));

    std::vector<std::filesystem::path> rv(result.gl_pathc);

    for(size_t i = 0; i < result.gl_pathc; i++)
        rv[i] = result.gl_pathv[i];

    return rv;
}

std::filesystem::path TestData::ProgramLocation()
{
#if defined(__linux__)
    return std::filesystem::read_symlink("/proc/self/exe");
#elif defined(_WIN32)
    // Use buffer large enough for Windows long path (up to 32767 chars).
    constexpr DWORD      kMaxPathChars = 32768;
    std::vector<wchar_t> buf(kMaxPathChars);
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if(n == 0)
        throw std::runtime_error("GetModuleFileNameW failed");
    if(n >= buf.size())
        throw std::runtime_error("GetModuleFileNameW: path longer than maximum supported");
    return std::filesystem::path(buf.data());
#else
    throw std::runtime_error("ProgramLocation() not implemented for this platform");
#endif
}

TestData::TestData()
    : m_dataDir(ProgramLocation().parent_path() / "data")
{
    if(!std::filesystem::is_directory(m_dataDir))
    {
        auto newValue = ProgramLocation().parent_path().parent_path() / "data";
        if(std::filesystem::is_directory(newValue))
            m_dataDir = newValue;
    }
}

TestData::TestData(std::string const& dataDir)
    : m_dataDir(dataDir)
{
}

TestData::TestData(invalid_data) {}
