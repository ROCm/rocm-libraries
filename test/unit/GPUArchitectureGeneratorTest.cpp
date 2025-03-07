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

#include <GPUArchitectureGenerator/GPUArchitectureGenerator.hpp>
#include <GPUArchitectureGenerator/GPUArchitectureGenerator_defs.hpp>

#include "SimpleFixture.hpp"

class GPUArchitectureGeneratorTest : public SimpleFixture
{
protected:
    void SetUp() override
    {
        GPUArchitectureGenerator::FillArchitectures();
    }
};

TEST_F(GPUArchitectureGeneratorTest, BasicYAML)
{
    GPUArchitectureGenerator::GenerateFile("output.yaml", true);

    std::ifstream     generated_source_file("output.yaml");
    std::stringstream ss;
    ss << generated_source_file.rdbuf();
    std::string generated_source = ss.str();
    EXPECT_NE(generated_source, "");

    auto readback = rocRoller::GPUArchitecture::readYaml("output.yaml");
    EXPECT_EQ(readback.size(), rocRoller::SupportedArchitectures.size());
    for(auto& x : readback)
    {
        EXPECT_TRUE(x.second.HasCapability("SupportedISA"));
    }
    std::remove("output.yaml");
}

TEST_F(GPUArchitectureGeneratorTest, BasicMsgpack)
{
    GPUArchitectureGenerator::GenerateFile("output.msgpack");

    std::ifstream     generated_source_file("output.msgpack");
    std::stringstream ss;
    ss << generated_source_file.rdbuf();
    std::string generated_source = ss.str();
    EXPECT_NE(generated_source, "");

    auto readback = rocRoller::GPUArchitecture::readMsgpack("output.msgpack");
    EXPECT_EQ(readback.size(), rocRoller::SupportedArchitectures.size());
    for(auto& x : readback)
    {
        EXPECT_TRUE(x.second.HasCapability("SupportedISA"));
    }
    std::remove("output.msgpack");
}
