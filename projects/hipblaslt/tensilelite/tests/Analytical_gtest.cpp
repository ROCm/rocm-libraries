/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "Testing_analytical.hpp"
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

// Parse Analytical_gtest.yaml to get the test data
std::vector<MyTestData> parseYamlManually(const std::string& filename)
{

    std::string YamlfullPath = std::string(YAML_PATH) + "/testlib/source/" + filename;
    std::ifstream file(YamlfullPath);
    if(!file)
    {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return {};
    }

    std::string             line;
    std::vector<MyTestData> tests;
    MyTestData              current;
    enum class State
    {
        None,
        Inputs
    } state
        = State::None;
    int line_number = 0;

    while(std::getline(file, line))
    {
        line_number++;
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);

        if(line.empty() || line[0] == '#')
            continue;

        if(line.rfind("- name:", 0) == 0)
        {
            if(!current.name.empty())
                tests.push_back(current);
            current      = MyTestData{};
            current.name = line.substr(7);
            current.name.erase(0, current.name.find_first_not_of(" \t"));
            state = State::None;
        }
        else if(line.rfind("inputs:", 0) == 0)
        {
            state = State::Inputs;
        }
        else if(state == State::Inputs && line.rfind("- {", 0) == 0)
        {
            std::string inner = line.substr(3);
            if(inner.back() == '}')
                inner.pop_back();

            std::map<std::string, int> values;
            int                        expected = 0;
            std::stringstream          ss(inner);
            std::string                pair;
            while(std::getline(ss, pair, ','))
            {
                auto colon = pair.find(':');
                if(colon == std::string::npos)
                    continue;
                std::string key = pair.substr(0, colon);
                std::string val = pair.substr(colon + 1);
                key.erase(0, key.find_first_not_of(" \t"));
                key.erase(key.find_last_not_of(" \t") + 1);
                val.erase(0, val.find_first_not_of(" \t"));
                val.erase(val.find_last_not_of(" \t") + 1);

                try
                {
                    int num = std::stoi(val);
                    if(key == "expected")
                        expected = num;
                    else
                        values[key] = num;
                }
                catch(...)
                {
                    std::cerr << "Invalid number in line " << line_number << ": " << val
                              << std::endl;
                }
            }
            current.inputs.push_back(InputWithExpected{values, expected});
        }
    }

    if(!current.name.empty())
        tests.push_back(current);

    return tests;
}

TEST_P(AnalyticalGtest, DynamicDispatch)
{
    const MyTestData& test = GetParam();

    const std::string gpuArchNumber = std::to_string(test.inputs[0].values.at("gpu_arch"));
    auto gpuArchEnum = TensileLite::analytical::Hardware::archNameToEnum("gfx" + gpuArchNumber);

    //TODO: Hardcoding numbers for gfx942. Future archs could be added here with if else loop.
    auto gpuInfo = TensileLite::analytical::Hardware(
        gpuArchEnum, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, 1.0);

    if(test.name == "ComputeLoads")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeLoads(input_case.values.at("M"),
                         input_case.values.at("N"),
                         input_case.values.at("K"),
                         input_case.expected);
        }
    }
    else
    {
        FAIL() << "Unknown test name: " << test.name;
    }
}

// Instantiate tests using manual parser
INSTANTIATE_TEST_SUITE_P(AnalyticalYamlTests,
                         AnalyticalGtest,
                         ::testing::ValuesIn(parseYamlManually("Analytical_gtest.yaml")),
                         [](const ::testing::TestParamInfo<MyTestData>& info) {
                             std::string name = info.param.name;
                             for(auto& c : name)
                                 if(!std::isalnum(c))
                                     c = '_';
                             return name;
                         });
