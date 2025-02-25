
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

#include <rocRoller/Serialization/Base_fwd.hpp>

namespace rocRoller
{
    class GPUArchitectureTarget
    {
    public:
        GPUArchitectureTarget() = default;
        explicit GPUArchitectureTarget(std::string const& input);

        GPUArchitectureTarget& operator=(std::string const& input);

        bool operator==(GPUArchitectureTarget a) const
        {
            return m_majorVersion == a.m_majorVersion && m_minorVersion == a.m_minorVersion
                   && m_pointVersion == a.m_pointVersion && m_sramecc == a.m_sramecc
                   && m_xnack == a.m_xnack;
        }
        bool operator!=(GPUArchitectureTarget a) const
        {
            return m_majorVersion != a.m_majorVersion || m_minorVersion != a.m_minorVersion
                   || m_pointVersion != a.m_pointVersion || m_sramecc != a.m_sramecc
                   || m_xnack != a.m_xnack;
        }
        bool operator<(GPUArchitectureTarget a) const
        {
            if(m_majorVersion < a.m_majorVersion)
            {
                return true;
            }
            else if(m_majorVersion > a.m_majorVersion)
            {
                return false;
            }
            else if(m_minorVersion < a.m_minorVersion)
            {
                return true;
            }
            else if(m_minorVersion > a.m_minorVersion)
            {
                return false;
            }
            else if(m_pointVersion < a.m_pointVersion)
            {
                return true;
            }
            else if(m_pointVersion > a.m_pointVersion)
            {
                return false;
            }
            else
            {
                return toString() < a.toString();
            }
        }

        std::string toString() const;

        void parseString(std::string const&);

        // Return a string of features that can be provided as input to the LLVM Assembler.
        // These should have the ON/OFF symbol in front of each feature, and be comma
        // delimmited.
        std::string getLLVMFeatureString() const;

        // Return the target string without the features list.
        std::string getVersionString() const;

        int getMajorVersion() const;

        struct Hash
        {
            std::size_t operator()(const GPUArchitectureTarget& input) const
            {
                return std::hash<std::string>()(input.toString());
            };
        };

        template <typename T1, typename T2, typename T3>
        friend struct rocRoller::Serialization::MappingTraits;

    private:
        int m_majorVersion = 0;
        int m_minorVersion = 0;
        int m_pointVersion = 0;

        bool m_sramecc = false;
        bool m_xnack   = false;

        std::string m_stringRep;
        std::string m_versionRep;
        std::string m_llvmFeaturesRep;

        std::string createString(bool) const;
        std::string createLLVMFeatureString() const;
    };
}

#include "GPUArchitectureTarget_impl.hpp"
