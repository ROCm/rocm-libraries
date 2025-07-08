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

#include <fstream>
#include <map>
#include <vector>

#include <rocRoller/Utilities/Utils.hpp>
#include <rocRoller/Utilities/Version.hpp>

namespace rocRoller
{
    std::vector<std::set<int>> mergeSets(std::vector<std::set<int>> sets)
    {
        bool any = false;
        do
        {
            any = false;

            for(auto iterA = sets.begin(); iterA != sets.end(); iterA++)
            {
                for(auto iterB = iterA + 1; iterB != sets.end();)
                {
                    bool intersect = false;
                    for(auto el : *iterB)
                    {
                        if(iterA->contains(el))
                        {
                            intersect = true;
                            break;
                        }
                    }

                    if(intersect)
                    {
                        any = true;
                        iterA->insert(iterB->begin(), iterB->end());
                        iterB = sets.erase(iterB);
                    }
                    else
                    {
                        ++iterB;
                    }
                }
            }
        } while(any);

        return sets;
    }

    std::vector<char> readFile(std::string const& filename)
    {
        std::ifstream file(filename);

        AssertFatal(file.good(), "Could not read ", filename);

        std::array<char, 4096> buffer;

        std::vector<char> rv;

        file.read(buffer.data(), buffer.size());

        while(file.good() && !file.eof())
        {
            rv.insert(rv.end(), buffer.begin(), buffer.end());

            file.read(buffer.data(), buffer.size());
        }

        auto numRead = file.gcount();
        AssertFatal(numRead <= buffer.size());

        rv.insert(rv.end(), buffer.begin(), buffer.begin() + numRead);

        return rv;
    }

    std::string escapeSymbolName(std::string name)
    {
        char underscore = '_';
        char replaced   = '*';

        // Replace any character that isn't alphanumeric with underscore,
        // or one of a few special cases.

        std::map<char, char> specialCharacters = {{'+', 'p'}, {'-', 'm'}};
        for(auto& c : name)
        {
            if(!isalnum(c) && c != underscore)
            {
                auto iter = specialCharacters.find(c);
                if(iter != specialCharacters.end())
                    c = iter->second;
                else
                    c = replaced;
            }
        }

        // Delete any trailing '*'s.

        while(!name.empty() && name.back() == replaced)
            name.pop_back();

        std::string rv;
        rv.reserve(name.size());

        // Delete any leading '*'s, as well as any duplicate '*'s.

        for(auto const& c : name)
        {
            if(c != replaced || (!rv.empty() && rv.back() != replaced))
            {
                // cppcheck-suppress useStlAlgorithm
                rv += c;
            }
        }

        // If the name is now completely empty, return '_'.
        if(rv.empty())
            rv += replaced;

        // Replace any '*' chars with '_'.
        for(auto& c : rv)
        {
            if(c == replaced)
            {
                // cppcheck-suppress useStlAlgorithm
                c = underscore;
            }
        }

        return rv;
    }
}
