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

#include <TestData.hpp>

#include <gtest/gtest.h>

TEST(TestData, Simple)
{
    auto data = TestData::Instance();

    EXPECT_TRUE(static_cast<bool>(data));

#if defined(TENSILE_MSGPACK) || defined(TENSILE_LLVM)
    auto is_regular_file
        = static_cast<bool (*)(boost::filesystem::path const&)>(boost::filesystem::is_regular_file);

    EXPECT_PRED1(is_regular_file, data.file("KernelsLite"));
    EXPECT_FALSE(is_regular_file(data.file("fjdlksljfjldskj")));

    auto datFiles  = data.glob(std::string("*.dat"));
    auto yamlFiles = data.glob(std::string("*.yaml"));

    EXPECT_GE(datFiles.size() + yamlFiles.size(), 6);
    for(auto file : datFiles)
    {
        std::cout << file << std::endl;
        EXPECT_PRED1(is_regular_file, file);
    }

    for(auto file : yamlFiles)
    {
        std::cout << file << std::endl;
        EXPECT_PRED1(is_regular_file, file);
    }
#endif

    if(TestData::Env("TENSILE_NEVER_SET_THIS_AKDJFLKDSJ"))
        FAIL() << "TestData object constructed with unset environment variable "
                  "should convert to false!";
}
