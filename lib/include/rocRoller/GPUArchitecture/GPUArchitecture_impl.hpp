#pragma once

#include "GPUArchitecture.hpp"

#include "Utilities/Utils.hpp"

#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>

#include <rocRoller/Utilities/Timer.hpp>

namespace rocRoller
{
    inline GPUArchitectureTarget const& GPUArchitecture::target() const
    {
        return m_isaVersion;
    }

    inline void GPUArchitecture::AddCapability(GPUCapability capability, int value)
    {
        m_capabilities[capability] = value;
    }

    inline void GPUArchitecture::AddInstructionInfo(GPUInstructionInfo info)
    {
        if(m_instruction_infos.find(info.getInstruction()) != m_instruction_infos.end())
            throw std::runtime_error(
                concatenate("Instruction info already exists for ", info.getInstruction()));
        m_instruction_infos[info.getInstruction()] = info;
    }

    inline bool GPUArchitecture::HasCapability(GPUCapability capability) const
    {
        return m_capabilities.find(capability) != m_capabilities.end();
    }

    inline bool GPUArchitecture::HasCapability(std::string capabilityString) const
    {
        return m_capabilities.find(GPUCapability(capabilityString)) != m_capabilities.end();
    }

    inline int GPUArchitecture::GetCapability(GPUCapability capability) const
    {
        auto iter = m_capabilities.find(capability);
        if(iter == m_capabilities.end())
            throw std::runtime_error(concatenate("Capability ", capability, " not found"));

        return iter->second;
    }

    inline int GPUArchitecture::GetCapability(std::string capabilityString) const
    {
        return GetCapability(GPUCapability(capabilityString));
    }

    inline bool GPUArchitecture::HasInstructionInfo(std::string instruction) const
    {
        auto iter = m_instruction_infos.find(instruction);
        return iter != m_instruction_infos.end();
    }

    inline rocRoller::GPUInstructionInfo
        GPUArchitecture::GetInstructionInfo(std::string instruction) const
    {
        auto iter = m_instruction_infos.find(instruction);
        if(iter != m_instruction_infos.end())
        {
            return iter->second;
        }
        else
        {
            return GPUInstructionInfo();
        }
    }

    inline GPUArchitecture::GPUArchitecture()
        : m_isaVersion(GPUArchitectureTarget())
    {
    }

    inline GPUArchitecture::GPUArchitecture(GPUArchitectureTarget isaVersion)
        : m_isaVersion(isaVersion)
    {
    }

    inline GPUArchitecture::GPUArchitecture(
        GPUArchitectureTarget const&                     isaVersion,
        std::map<GPUCapability, int> const&              capabilities,
        std::map<std::string, GPUInstructionInfo> const& instruction_infos)
        : m_isaVersion(isaVersion)
        , m_capabilities(capabilities)
        , m_instruction_infos(instruction_infos)
    {
    }

    inline std::ostream& operator<<(std::ostream& os, const GPUCapability& input)
    {
        os << input.ToString();
        return os;
    }

    inline std::istream& operator>>(std::istream& is, GPUCapability& input)
    {
        std::string recvd;
        is >> recvd;
        input = GPUCapability(recvd);
        return is;
    }

    inline std::ostream& operator<<(std::ostream& os, const GPUWaitQueueType& input)
    {
        os << input.ToString();
        return os;
    }

    inline std::ostream& operator<<(std::ostream& os, const GPUArchitectureTarget& input)
    {
        return (os << input.ToString());
    }

    inline std::istream& operator>>(std::istream& is, GPUArchitectureTarget& input)
    {
        std::string recvd;
        is >> recvd;
        input.parseString(recvd);
        return is;
    }

    template <std::integral T>
    requires(!std::same_as<bool, T>) bool GPUArchitecture::isSupportedConstantValue(T value) const
    {
        if constexpr(std::signed_integral<T>)
            return value >= -16 && value <= 64;
        else
            return value <= 64;
    }

    template <std::floating_point T>
    std::unordered_set<T> supportedConstantValues()
    {
        static_assert(std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, Half>,
                      "Unsupported floating point type");

        T one_over_two_pi;

        if constexpr(std::same_as<T, float>)
            one_over_two_pi = 0.15915494f;
        else if constexpr(std::same_as<T, double>)
            one_over_two_pi = 0.15915494309189532;
        else if constexpr(std::same_as<T, Half>)
            one_over_two_pi = 0.1592f;

        return {0.0, 0.5, 1.0, 2.0, 4.0, -0.5, -1.0, -2.0, -4.0, one_over_two_pi};
    };

    template <std::floating_point T>
    bool GPUArchitecture::isSupportedConstantValue(T value) const
    {
        static auto supportedValues = supportedConstantValues<T>();

        return supportedValues.find(value) != supportedValues.end();
    }

    inline std::map<GPUCapability, int> const& GPUArchitecture::getAllCapabilities() const
    {
        return m_capabilities;
    }

    inline std::map<std::string, GPUInstructionInfo> const&
        GPUArchitecture::getAllIntructionInfo() const
    {
        return m_instruction_infos;
    }
}
