/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************
 * hipBLASLt test: walk library logic YAMLs and validate each solution
 * parameter against TensileLite's consistency API. The YAMLs are "our" files;
 * TensileLite only provides validateSolutionParamsForSelection().
 * See docs/library-yaml-consistency-check-design.md.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <Tensile/SolutionValidation.hpp>

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

#ifndef HIPBLASLT_LIBRARY_LOGIC_ROOT
#define HIPBLASLT_LIBRARY_LOGIC_ROOT ""
#endif

namespace
{
    int getCuCountFromDirName(const fs::path& dir)
    {
        std::string name = dir.filename().string();
        std::regex  re(R"(_(\d+)cu$)");
        std::smatch m;
        if(std::regex_search(name, m, re))
            return std::stoi(m[1].str());
        return 0;
    }

    struct Violation
    {
        std::string library;
        std::string file;
        std::string rule;
        std::string detail;
    };

    void checkOneLibrary(const fs::path&              libDir,
                         int                          cuCount,
                         std::regex&                  reWgmXCC,
                         std::regex&                  reWorkGroup,
                         std::regex&                  reWavefront,
                         std::vector<Violation>&      violations)
    {
        std::string libName = libDir.filename().string();

        for(fs::recursive_directory_iterator it(libDir), end; it != end; ++it)
        {
            if(!it->is_regular_file() || it->path().extension() != ".yaml")
                continue;

            std::ifstream f(it->path(), std::ios::binary);
            if(!f)
                continue;

            std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            f.close();

            std::string filename = it->path().filename().string();

            // Validate each WorkGroupMappingXCC with (cuCount, xcc, default wg/wavefront)
            for(auto it_re = std::sregex_iterator(content.begin(), content.end(), reWgmXCC);
                it_re != std::sregex_iterator();
                ++it_re)
            {
                int xcc = std::stoi((*it_re)[1].str());
                TensileLite::SolutionParamsForValidation p;
                p.cuCount             = cuCount;
                p.workGroupMappingXCC = xcc;
                p.workGroup           = {8, 8, 4};
                p.wavefrontSize       = 64;
                std::string reason;
                if(!TensileLite::validateSolutionParamsForSelection(p, &reason))
                {
                    violations.push_back(
                        {libName, filename, "WorkGroupMappingXCC",
                         "value " + std::to_string(xcc) + " invalid for " + std::to_string(cuCount)
                             + " CUs: " + reason});
                }
            }

            // Validate each WorkGroup with (cuCount, default xcc, wg, default wavefront)
            for(auto it_re = std::sregex_iterator(content.begin(), content.end(), reWorkGroup);
                it_re != std::sregex_iterator();
                ++it_re)
            {
                int a = std::stoi((*it_re)[1].str());
                int b = std::stoi((*it_re)[2].str());
                int c = std::stoi((*it_re)[3].str());
                TensileLite::SolutionParamsForValidation p;
                p.cuCount             = cuCount;
                p.workGroupMappingXCC = 1;
                p.workGroup           = {a, b, c};
                p.wavefrontSize       = 64;
                std::string reason;
                if(!TensileLite::validateSolutionParamsForSelection(p, &reason))
                {
                    violations.push_back(
                        {libName, filename, "WorkGroup",
                         "[" + std::to_string(a) + "," + std::to_string(b) + "," + std::to_string(c)
                             + "]: " + reason});
                }
            }

            // Validate each WavefrontSize with (cuCount, default xcc, default wg, n)
            for(auto it_re = std::sregex_iterator(content.begin(), content.end(), reWavefront);
                it_re != std::sregex_iterator();
                ++it_re)
            {
                int n = std::stoi((*it_re)[1].str());
                TensileLite::SolutionParamsForValidation p;
                p.cuCount             = cuCount;
                p.workGroupMappingXCC = 1;
                p.workGroup           = {8, 8, 4};
                p.wavefrontSize       = n;
                std::string reason;
                if(!TensileLite::validateSolutionParamsForSelection(p, &reason))
                {
                    violations.push_back(
                        {libName, filename, "WavefrontSize",
                         "value " + std::to_string(n) + ": " + reason});
                }
            }
        }
    }
} // namespace

// Named LibraryYAML_quick so --gtest_filter=*quick* includes this test.
TEST(LibraryYAML_quick, SolutionLibrary_ConsistencyRules)
{
    const char* rootStr = HIPBLASLT_LIBRARY_LOGIC_ROOT;
    if(rootStr == nullptr || rootStr[0] == '\0')
        GTEST_SKIP() << "HIPBLASLT_LIBRARY_LOGIC_ROOT not set (path to library logic, e.g. .../Tensile/Logic/asm_full)";

    fs::path root(rootStr);
    if(!fs::is_directory(root))
        GTEST_SKIP() << "Library logic root not found: " << root.string();

    std::regex reWgmXCC(R"(WorkGroupMappingXCC\s*:\s*(-?\d+))");
    std::regex reWorkGroup(R"(WorkGroup\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*])");
    std::regex reWavefront(R"(WavefrontSize\s*:\s*(\d+))");

    std::vector<Violation> violations;

    for(fs::recursive_directory_iterator it(root), end; it != end; ++it)
    {
        if(!it->is_directory())
            continue;

        int cuCount = getCuCountFromDirName(it->path());
        if(cuCount <= 0)
            continue;

        checkOneLibrary(it->path(), cuCount, reWgmXCC, reWorkGroup, reWavefront, violations);
    }

    if(!violations.empty())
    {
        std::ostringstream out;
        out << violations.size() << " consistency violation(s) across CU-variant libraries:\n";
        for(const auto& v : violations)
            out << "  [" << v.library << "] [" << v.rule << "] " << v.file << ": " << v.detail << "\n";
        FAIL() << out.str();
    }
}
