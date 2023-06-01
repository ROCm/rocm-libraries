
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

namespace rocRoller
{
    inline std::string GPUArchitectureTarget::createString(bool withFeatures) const
    {
        std::stringstream ss;
        ss << "gfx" << m_majorVersion << m_minorVersion << std::hex << m_pointVersion;
        if(withFeatures)
        {
            if(m_xnack)
            {
                ss << ":xnack+";
            }
            if(m_sramecc)
            {
                ss << ":sramecc+";
            }
        }
        return ss.str();
    }

    inline std::string GPUArchitectureTarget::createLLVMFeatureString() const
    {
        std::stringstream ss;
        if(m_xnack)
        {
            ss << "+xnack";
        }
        if(m_sramecc)
        {
            if(m_xnack)
                ss << ",";
            ss << "+sramecc";
        }

        return ss.str();
    }

    inline std::string GPUArchitectureTarget::toString() const
    {
        return m_stringRep;
    }

    inline void GPUArchitectureTarget::parseString(std::string const& input)
    {
        int         start = 3; //Skip gfx
        size_t      end   = input.find(":");
        std::string arch  = input.substr(start, end - start);

        if(arch.length() == 4)
        {
            m_majorVersion = std::stoi(arch.substr(0, 2));
        }
        else
        {
            m_majorVersion = std::stoi(arch.substr(0, 1));
        }

        m_minorVersion = std::stoi(arch.substr(arch.length() - 2, 1));
        m_pointVersion = std::stoi(arch.substr(arch.length() - 1, 1), nullptr, 16);

        while(end != std::string::npos)
        {
            start               = end + 1;
            end                 = input.find(":", start);
            std::string feature = input.substr(start, end - start);
            if(feature == "xnack+")
            {
                m_xnack = true;
            }
            else if(feature == "sramecc+")
            {
                m_sramecc = true;
            }
        }

        m_stringRep       = createString(true);
        m_versionRep      = createString(false);
        m_llvmFeaturesRep = createLLVMFeatureString();
    }

    inline std::string GPUArchitectureTarget::getLLVMFeatureString() const
    {
        return m_llvmFeaturesRep;
    }

    inline std::string GPUArchitectureTarget::getVersionString() const
    {
        return m_versionRep;
    }

    inline GPUArchitectureTarget::GPUArchitectureTarget(std::string const& input)
    {
        parseString(input);
    }

    inline int GPUArchitectureTarget::getMajorVersion() const
    {
        return m_majorVersion;
    }
}
