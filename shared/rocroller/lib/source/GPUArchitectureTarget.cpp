// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <regex>
#include <string>

#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    GPUArchitectureTarget GPUArchitectureTarget::fromString(std::string const& archStr,
                                                            int                asicRevisionId)
    {
        GPUArchitectureTarget rv;

        int         start = 0;
        size_t      end   = archStr.find(":");
        std::string arch  = archStr.substr(start, end - start);

        std::regex  pattern(R"(rev(\d+))");
        std::smatch revIdMatch;
        if(std::regex_search(arch, revIdMatch, pattern))
        {
            rv.asicRevisionId = std::stoi(revIdMatch[1]);
            arch              = arch.substr(0, revIdMatch.position(0));
        }
        else if(arch == "gfx1250")
        {
            if(asicRevisionId == 2)
            {
                int newRevId = Settings::Get(Settings::GFX1250AsicRevisionId);
                Log::warn("Overriding current device asic revision id from {} to {}",
                          asicRevisionId,
                          newRevId);
                asicRevisionId = newRevId;
            }
            rv.asicRevisionId = asicRevisionId;
        }

        rv.gfx = rocRoller::fromString<GPUArchitectureGFX>(arch);

        while(end != std::string::npos)
        {
            start               = end + 1;
            end                 = archStr.find(":", start);
            std::string feature = archStr.substr(start, end - start);
            if(feature == "xnack+")
            {
                rv.features.xnack = true;
            }
            else if(feature == "sramecc+")
            {
                rv.features.sramecc = true;
            }
        }
        return rv;
    }
}
