
#pragma once

#include <array>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline std::string toString(GPUArchitectureGFX const& gfx)
    {
        switch(gfx)
        {
        case GPUArchitectureGFX::GFX803:
            return "gfx803";
        case GPUArchitectureGFX::GFX900:
            return "gfx900";
        case GPUArchitectureGFX::GFX906:
            return "gfx906";
        case GPUArchitectureGFX::GFX908:
            return "gfx908";
        case GPUArchitectureGFX::GFX90A:
            return "gfx90a";
        case GPUArchitectureGFX::GFX940:
            return "gfx940";
        case GPUArchitectureGFX::GFX941:
            return "gfx941";
        case GPUArchitectureGFX::GFX942:
            return "gfx942";
        case GPUArchitectureGFX::GFX1010:
            return "gfx1010";
        case GPUArchitectureGFX::GFX1011:
            return "gfx1011";
        case GPUArchitectureGFX::GFX1012:
            return "gfx1012";
        case GPUArchitectureGFX::GFX1030:
            return "gfx1030";
        default:
            return "gfxunknown";
        }
    }

    inline std::string GPUArchitectureFeatures::toString() const
    {
        std::string rv = "";
        if(sramecc)
        {
            rv = concatenate(rv, "sramecc+");
        }
        if(xnack)
        {
            rv = concatenate(rv, "xnack+");
        }
        return rv;
    }

    inline std::string GPUArchitectureFeatures::toLLVMString() const
    {
        std::string rv = "";
        if(xnack)
        {
            rv = concatenate(rv, "+xnack");
        }
        if(sramecc)
        {
            if(xnack)
                rv = concatenate(rv, ",");
            rv = concatenate(rv, "+sramecc");
        }
        return rv;
    }

    inline std::string GPUArchitectureTarget::toString() const
    {
        if(features.sramecc || features.xnack)
            return concatenate(gfx, ":", features.toString());
        else
            return rocRoller::toString(gfx);
    }

    inline GPUArchitectureTarget GPUArchitectureTarget::fromString(std::string const& archStr)
    {
        GPUArchitectureTarget rv;

        int         start = 0;
        size_t      end   = archStr.find(":");
        std::string arch  = archStr.substr(start, end - start);

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
