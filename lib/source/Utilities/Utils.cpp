#include <fstream>
#include <vector>

#include <rocRoller/Utilities/Utils.hpp>

std::vector<char> rocRoller::readFile(std::string const& filename)
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
