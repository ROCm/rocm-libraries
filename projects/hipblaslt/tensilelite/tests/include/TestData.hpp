/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include <Tensile/Singleton.hpp>

struct TestData : public TensileLite::LazySingleton<TestData>
{
    using Base = TensileLite::LazySingleton<TestData>;

    operator bool() const;

    static TestData Invalid();
    static TestData Env(std::string const& varName);

    std::filesystem::path dataDir() const;

    std::filesystem::path file(std::string const& filename) const;
    std::filesystem::path file(std::string const& filename, std::string const& extension) const;

    std::vector<std::filesystem::path> glob(std::string const& pattern) const;

private:
    friend Base;

    std::filesystem::path m_dataDir;

    struct invalid_data
    {
    };

    static std::filesystem::path ProgramLocation();

    TestData();
    TestData(std::string const& dataDir);
    TestData(invalid_data);
};
