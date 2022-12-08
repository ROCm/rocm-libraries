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
        std::istringstream stream(var);
        T                  value;
        stream >> value;
        return value;
    }

    template <>
    inline Scheduling::SchedulerProcedure
        Settings::getTypeValue<Scheduling::SchedulerProcedure>(std::string const& proc) const
    {
        if(proc == Scheduling::toString(Scheduling::SchedulerProcedure::Sequential))
        {
            return Scheduling::SchedulerProcedure::Sequential;
        }
        else if(proc == Scheduling::toString(Scheduling::SchedulerProcedure::RoundRobin))
        {
            return Scheduling::SchedulerProcedure::RoundRobin;
        }
        else if(proc == Scheduling::toString(Scheduling::SchedulerProcedure::Random))
        {
            return Scheduling::SchedulerProcedure::Random;
        }
        else if(proc == Scheduling::toString(Scheduling::SchedulerProcedure::Cooperative))
        {
            return Scheduling::SchedulerProcedure::Cooperative;
        }
        else
        {
            Throw<FatalError>("Trying to get unsupported/invalid scheduling procedure.");
        }
    }

    template <>
    inline Settings::bitFieldType
        Settings::getTypeValue<Settings::bitFieldType>(std::string const& var) const
    {
        bitFieldType res{var};
        return res;
    }

    template <>
    inline int Settings::getTypeValue<int>(std::string const& var) const
    {
        return std::stoi(var, nullptr, 0);
    }

    template <>
    inline bool Settings::getTypeValue<bool>(std::string const& var) const
    {
        return var != "0";
    }

    template <>
    inline std::string Settings::getTypeValue<std::string>(std::string const& var) const
    {
        return var;
    }

    template <>
    inline LogLevel Settings::getTypeValue<LogLevel>(std::string const& var) const
    {
        if(var == "None")
        {
            return LogLevel::None;
        }
        else if(var == "Error")
        {
            return LogLevel::Error;
        }
        else if(var == "Warning")
        {
            return LogLevel::Warning;
        }
        else if(var == "Terse")
        {
            return LogLevel::Terse;
        }
        else if(var == "Verbose")
        {
            return LogLevel::Verbose;
        }
        else if(var == "Debug")
        {
            return LogLevel::Debug;
        }
        else
        {
            Throw<FatalError>("Trying to get invalid LogLevel.");
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

    template <typename T>
    inline std::string Settings::toString(T const& var) const
    {
        Throw<FatalError>("Unsupported type for Settings::toString()");
    }

    template <>
    inline std::string Settings::toString(LogLevel const& logLevel) const
    {
        std::string rv = "";

        switch(logLevel)
        {
        case LogLevel::None:
            rv = "None";
            break;
        case LogLevel::Error:
            rv = "Error";
            break;
        case LogLevel::Warning:
            rv = "Warning";
            break;
        case LogLevel::Terse:
            rv = "Terse";
            break;
        case LogLevel::Verbose:
            rv = "Verbose";
            break;
        case LogLevel::Debug:
            rv = "Debug";
            break;
        case LogLevel::Count:
            rv = "LogLevel Count (";
            rv += std::to_string(static_cast<int>(LogLevel::Count));
            rv += ")";
            break;
        default:
            Throw<FatalError>("Unsupported LogLevel for Settings::toString().");
        }

        return rv;
    }

    inline std::ostream& operator<<(std::ostream& os, const LogLevel& input)
    {
        os << Settings::getInstance()->toString(input);
        return os;
    }

    inline Settings::Settings() {}
}
