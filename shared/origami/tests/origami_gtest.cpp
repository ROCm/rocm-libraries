// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "testing_origami.hpp"
#include <cctype>
#include <fstream>
#include <iostream>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#include <limits.h>
#else
#include <windows.h>
#endif

// Returns the directory path where the current executable resides.
// Adds a trailing slash ('/' on Linux, '\' on Windows) for easy file concatenation.
std::string getExecutableDir() {
#ifndef _WIN32
    // Linux branch

    char result[PATH_MAX];  // Buffer to store the path
    ssize_t count = readlink("/proc/self/exe", result, PATH_MAX);  
    // readlink reads the symbolic link /proc/self/exe, which points to the current executable

    if (count == -1) {
        // If readlink fails, return empty string
        return "";
    }

    result[count] = '\0'; // Null-terminate the buffer
    std::string fullPath(result); // Convert to std::string

    // Find the position of the last slash ('/') in the path
    // This separates the directory from the binary name
    size_t pos = fullPath.find_last_of('/');

    // Extract the directory portion
    std::string dir = (pos != std::string::npos) ? fullPath.substr(0, pos) : fullPath;

    // Ensure the directory string ends with a slash
    if (!dir.empty() && dir.back() != '/')
        dir += '/';

    return dir;

#else
    // Windows branch

    char path[MAX_PATH];  // Buffer to store the path
    DWORD length = GetModuleFileNameA(NULL, path, MAX_PATH);  
    // GetModuleFileNameA returns the full path of the current executable

    if (length == 0) {
        // Failed to get the executable path
        return "";
    }

    std::string fullPath(path, length);  // Convert to std::string

    // Find the position of the last backslash ('\') or forward slash ('/')
    size_t pos = fullPath.find_last_of("\\/");

    // Extract the directory portion
    std::string dir = (pos != std::string::npos) ? fullPath.substr(0, pos) : fullPath;

    // Ensure the directory string ends with a backslash
    if (!dir.empty() && dir.back() != '\\' && dir.back() != '/')
        dir += '\\';

    return dir;
#endif
}

// Parse origami_gtest.yaml to get the test data
std::vector<MyTestData> parseYamlManually(const std::string& filename)
{
    std::string   YamlfullPath = getExecutableDir() + filename;
    std::ifstream file(YamlfullPath);
    if(!file)
    {
        std::cerr << "Failed to open file: " << YamlfullPath << std::endl;
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
            if(!inner.empty() && inner.back() == '}')
                inner.pop_back();

            std::map<std::string, int> values;
            std::optional<int>         expected;
            std::optional<int>         expected_gt;
            std::optional<int>         expected_lt;

            std::stringstream ss(inner);
            std::string       pair;
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
                    else if(key == "expected_gt")
                        expected_gt = num;
                    else if(key == "expected_lt")
                        expected_lt = num;
                    else
                        values[key] = num;
                }
                catch(...)
                {
                    std::cerr << "Invalid number in line " << line_number << ": " << val
                              << std::endl;
                }
            }

            current.inputs.push_back(InputWithExpected{values, expected, expected_gt, expected_lt});
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
    auto gpuArchEnum = origami::hardware_t::arch_name_to_enum("gfx" + gpuArchNumber);

    //TODO: Hardcoding numbers for gfx942. Future archs could be added here with if else loop.
    auto gpuInfo = origami::hardware_t(
        gpuArchEnum, 304, 65536, 8, 1.0, 1.0, 1.0, 4000000, 1.0, 1, std::make_tuple(0, 0.015, 0));

    if(test.name == "ComputeLoads")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeLoads(input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.expected);
        }
    }
    else if(test.name == "EstimateL2Hit")
    {
        for(const auto& input_case : test.inputs)
        {
            EstimateL2Hit(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("MT_M"), input_case.values.at("MT_N"), input_case.values.at("MT_K"), input_case.values.at("element_size"),
                          input_case.values.at("splittingFactor"),input_case.expected_gt, input_case.expected_lt);
        }
    }
    else if(test.name == "ComputeNumMatrixInstructions")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeNumMatrixInstructions(gpuInfo, input_case.values.at("MT_M"), input_case.values.at("MT_N"), input_case.values.at("MT_K"),
                          input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"), input_case.expected);
        }
    }
    else if(test.name == "ComputeMTComputeLatency")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeMTComputeLatency(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"),
                          input_case.values.at("transA"), input_case.values.at("transB"),input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"), 
                          input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), input_case.expected, input_case.expected_gt);
        }
    }
    else if(test.name == "ComputeMemoryLatency")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeMemoryLatency(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), 
                          input_case.values.at("mx_block_size"), input_case.values.at("wgm"), input_case.values.at("numActiveCUs"), input_case.values.at("splittingFactor"));
        }
    }
    else if(test.name == "ComputeTileLatency")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeTileLatency(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"),
                          input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), input_case.values.at("element_size_out"), 
                          input_case.values.at("mx_block_size"), input_case.values.at("wgm"), input_case.values.at("numActiveCUs"), input_case.values.at("splittingFactor"));
        }
    }
    else if(test.name == "ComputeWaveLatency")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeWaveLatency(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"),
                          input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), input_case.values.at("element_size_out"), 
                          input_case.values.at("mx_block_size"), input_case.values.at("wgm"), input_case.values.at("numActiveCUs"), input_case.values.at("splittingFactor"));
        }
    }
    else if(test.name == "ComputeTotalLatency")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputeTotalLatency(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"),
                          input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), input_case.values.at("element_size_out"), 
                          input_case.values.at("mx_block_size"), input_case.values.at("wgm"), input_case.values.at("splittingFactor"),
                          input_case.values.at("max_cus"));
        }
    }
    else if(test.name == "ComputePerfGflops")
    {
        for(const auto& input_case : test.inputs)
        {
            ComputePerfGflops(input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"),
                          input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), input_case.values.at("element_size_out"), 
                          input_case.values.at("WGM"), input_case.values.at("max_cus"));
        }
    }
    else if(test.name == "EstimateMallHit")
    {
        for(const auto& input_case : test.inputs)
        {
            EstimateMallHit(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("MT_M"), input_case.values.at("MT_N"), input_case.values.at("MT_K"),input_case.values.at("element_size_A"),
                           input_case.values.at("numActiveCUs"), input_case.values.at("splittingFactor"), input_case.expected_gt);
        }
    }
    else if(test.name == "CheckLDSCapacity")
    {
        for(const auto& input_case : test.inputs)
        {
            CheckLDSCapacity(gpuInfo, input_case.values.at("MT_M"), input_case.values.at("MT_N"), input_case.values.at("MT_K"), input_case.values.at("element_size"));
        }
    }
    else if(test.name == "HardwareArchEnum")
    {
        for(const auto& input_case : test.inputs)
        {
            HardwareArchEnum(std::to_string(test.inputs[0].values.at("gpu_arch")));
        }
    }
    else if(test.name == "BestGridSize")
    {
        for(const auto& input_case : test.inputs)
        {
            BestGridSize(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("MT_M"), input_case.values.at("MT_N"), 
                          input_case.values.at("MT_K"), input_case.values.at("MI_M"), input_case.values.at("MI_N"), input_case.values.at("MI_K"),
                          input_case.values.at("element_size_A"), input_case.values.at("element_size_B"), input_case.values.at("element_size_out"), 
                          input_case.values.at("mx_block_size"), input_case.values.at("H_L2"), input_case.values.at("WGM"), 
                          input_case.values.at("biggest_allowable_split"), input_case.values.at("max_cus"), input_case.expected_gt);
        }
    }
    else if(test.name == "BestMacroTileSize")
    {
        for(const auto& input_case : test.inputs)
        {
            BestMacroTileSize(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("transA"), input_case.values.at("transB"), input_case.values.at("element_size_A"), 
                          input_case.values.at("element_size_B"), input_case.values.at("element_size_out"), input_case.values.at("mx_block_size"), 
                          input_case.values.at("H_L2"), input_case.values.at("WGM"), input_case.values.at("max_cus"));
        }
    }
    else if(test.name == "BestWGM")
    {
        for(const auto& input_case : test.inputs)
        {
            BestWGM(gpuInfo, input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), input_case.values.at("batch"),
                          input_case.values.at("MT_M"), input_case.values.at("MT_N"), input_case.values.at("MT_K"));
        }
    }
    else if(test.name == "UtilsTFlopsFromLatency")
    {
        for(const auto& input_case : test.inputs)
        {
            UtilsTFlopsFromLatency(input_case.values.at("M"), input_case.values.at("N"), input_case.values.at("K"), 
                          input_case.values.at("latency_cycles"), input_case.values.at("clock_GHz"));
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
                         ::testing::ValuesIn(parseYamlManually("origami_gtest.yaml")),
                         [](const ::testing::TestParamInfo<MyTestData>& info) {
                             std::string name = info.param.name;
                             for(auto& c : name)
                                 if(!std::isalnum(c))
                                     c = '_';
                             return name;
                         });

