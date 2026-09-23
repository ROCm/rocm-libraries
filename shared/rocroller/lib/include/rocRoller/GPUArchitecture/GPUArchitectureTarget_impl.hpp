// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/Utilities/Utils.hpp>

namespace rocRoller
{
    inline std::string toString(GPUArchitectureGFX const& gfx)
    {
        switch(gfx)
        {
        case GPUArchitectureGFX::GFX908:
            return "gfx908";
        case GPUArchitectureGFX::GFX90A:
            return "gfx90a";
        case GPUArchitectureGFX::GFX942:
            return "gfx942";
        case GPUArchitectureGFX::GFX950:
            return "gfx950";
        case GPUArchitectureGFX::GFX1010:
            return "gfx1010";
        case GPUArchitectureGFX::GFX1011:
            return "gfx1011";
        case GPUArchitectureGFX::GFX1012:
            return "gfx1012";
        case GPUArchitectureGFX::GFX1030:
            return "gfx1030";
        case GPUArchitectureGFX::GFX1200:
            return "gfx1200";
        case GPUArchitectureGFX::GFX1201:
            return "gfx1201";
        case GPUArchitectureGFX::GFX1250:
            return "gfx1250";
        default:
            return "gfxunknown";
        }
    }

    inline std::ostream& operator<<(std::ostream& stream, GPUArchitectureGFX const& gfx)
    {
        return stream << toString(gfx);
    }

    inline std::string name(GPUArchitectureGFX const& gfx)
    {
        switch(gfx)
        {
        case GPUArchitectureGFX::GFX908:
            return "AMD CDNA 1";
        case GPUArchitectureGFX::GFX90A:
            return "AMD CDNA 2";
        case GPUArchitectureGFX::GFX942:
            return "AMD CDNA 3";
        case GPUArchitectureGFX::GFX950:
            return "AMD CDNA 4";
        case GPUArchitectureGFX::GFX1012:
            return "AMD RDNA 1";
        case GPUArchitectureGFX::GFX1030:
            return "AMD RDNA 2";
        case GPUArchitectureGFX::GFX1200:
        case GPUArchitectureGFX::GFX1201:
            return "AMD RDNA 4";
        case GPUArchitectureGFX::GFX1250:
            return "AMD CDNA 5";
        default:
            return "unknown";
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
            if(!rv.empty())
            {
                rv = concatenate(rv, ":");
            }
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
        std::string rv{rocRoller::toString(gfx)};
        if(asicRevisionId >= 0)
            rv = concatenate(rv, "rev", asicRevisionId);

        if(features.sramecc || features.xnack)
            rv = concatenate(rv, ":", features.toString());
        return rv;
    }

    inline std::string GPUArchitectureTarget::toAssemblerString() const
    {
        if(features.sramecc || features.xnack)
            return concatenate(gfx, ":", features.toString());
        else
            return rocRoller::toString(gfx);
    }

    inline std::string GPUArchitectureTarget::name() const
    {
        return rocRoller::name(gfx);
    }

}
