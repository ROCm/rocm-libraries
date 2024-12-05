
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
}
