#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <type_traits>

#include <rocRoller/Scheduling/Scheduler.hpp>
#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    template <typename Option>
    typename Option::Type Settings::get(Option const& opt)
    {
        // opt exists in m_values
        auto itr = m_values.find(opt.name);
        if(itr != m_values.end())
        {
            return std::any_cast<typename Option::Type>(itr->second);
        }
        // opt does not exist in m_values
        else
        {
            typename Option::Type val = getValue(opt);
            set(opt, val);
            return val;
        }
    }

    template <typename Option>
    typename Option::Type Settings::getValue(Option const& opt)
    {
        const char* var   = std::getenv(opt.name.c_str());
        std::string s_var = "";

        // If explicit flag is set
        if(var)
        {
            s_var = var;
            return getTypeValue<typename Option::Type>(s_var);
        }
        // If option is covered by bit field
        else if(opt.bit >= 0)
        {
            return extractBitValue(opt);
        }
        // Default to defaultValue
        else
        {
            return opt.defaultValue;
        }
    }

    template <typename T>
    inline T Settings::getTypeValue(std::string const& var) const
    {
        if constexpr(CCountedEnum<T>)
        {
            return fromString<T>(var);
        }
        else if constexpr(std::same_as<Settings::bitFieldType, T>)
        {
            return bitFieldType{var};
        }
        else if constexpr(std::same_as<bool, T>)
        {
            return var != "0";
        }
        else if constexpr(std::same_as<std::string, T>)
        {
            return var;
        }
        else
        {
            std::istringstream stream(var);
            T                  value;
            stream >> value;
            return value;
        }
    }

    template <typename Option>
    inline typename Option::Type Settings::extractBitValue(Option const& opt)
    {
        if constexpr(std::is_same_v<typename Option::Type, bool>)
        {
            return get(SettingsBitField).test(opt.bit);
        }
        else
        {
            return opt.defaultValue;
        }
    }

    template <typename Option>
    inline void Settings::set(Option const& opt, char const* val)
    {
        set(opt, std::string(val));
    }

    template <typename Option, typename T>
    inline void Settings::set(Option const& opt, T const& val)
    {
        if constexpr(std::is_same_v<typename Option::Type, T>)
        {
            if(!getenv(opt.name.c_str()))
                m_values[opt.name] = val;

            // Setting new bitfield changes values of those covered by bitfield
            if constexpr(std::is_same_v<typename Option::Type, bitFieldType>)
            {
                for(auto optName : setBitOptions)
                {
                    auto itr = m_values.find(optName);
                    if(itr != m_values.end())
                    {
                        m_values.erase(itr);
                    }
                }

                setBitOptions.clear();
            }

            // If corresponding env var is set, we will never get here since we grab from env
            else if(opt.bit >= 0)
            {
                setBitOptions.push_back(opt.name);
            }
        }
        else
        {
            Throw<FatalError>("Trying to set " + opt.name
                              + " with incorrect type. Not setting value.");
        }
    }

    inline std::string toString(LogLevel logLevel)
    {
        switch(logLevel)
        {
        case LogLevel::None:
            return "None";
        case LogLevel::Error:
            return "Error";
        case LogLevel::Warning:
            return "Warning";
        case LogLevel::Terse:
            return "Terse";
        case LogLevel::Verbose:
            return "Verbose";
        case LogLevel::Debug:
            return "Debug";
            break;
        case LogLevel::Count:
            return "Count";
        }

        Throw<FatalError>("Unsupported LogLevel.");
    }

    inline std::ostream& operator<<(std::ostream& os, const LogLevel& input)
    {
        return os << toString(input);
    }

    inline Settings::Settings() {}
}
