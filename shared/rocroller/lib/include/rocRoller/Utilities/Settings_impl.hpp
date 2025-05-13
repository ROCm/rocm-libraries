/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <type_traits>

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/RTTI.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace rocRoller
{
    template <typename Option>
    typename Option::Type Settings::get(Option const& opt)
    {
        {
            Settings::MapReaderLock readerLock{m_mapLock};
            // opt exists in m_values
            auto itr = m_values.find(opt.name);
            if(itr != m_values.end())
            {
                return std::any_cast<typename Option::Type>(itr->second);
            }
        }
        // opt does not exist in m_values
        typename Option::Type val = opt.getValue();
        set(opt, val);
        return val;
    }

    template <typename Option>
    typename Option::Type Settings::getDefault(Option const& opt)
    {
        return std::any_cast<typename Option::Type>(opt.getDefault());
    }

    template <typename Option>
    typename Option::Type Settings::Get(Option const& opt)
    {
        return getInstance()->get(opt);
    }

    template <typename T, bool LazyValue>
    inline T SettingsOption<T, LazyValue>::getValue() const
    {
        const char* var = std::getenv(name.c_str());

        // If explicit flag is set
        if(var)
        {
            std::string s_var = var;
            return getTypeValue(s_var);
        }
        // Default to defaultValue
        else
        {
            if constexpr(LazyValue)
            {
                return defaultValue();
            }
            else
            {
                return defaultValue;
            }
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
            Settings::MapWriterLock writerLock{m_mapLock};
            m_values[opt.name] = val;
        }
        else
        {
            Throw<FatalError>("Trying to set ",
                              opt.name,
                              " with incorrect type ",
                              typeName<T>(),
                              "(value: ",
                              val,
                              "). Not setting value.");
        }
    }

    inline std::string toString(LogLevel logLevel)
    {
        switch(logLevel)
        {
        case LogLevel::None:
            return "None";
        case LogLevel::Critical:
            return "Critical";
        case LogLevel::Error:
            return "Error";
        case LogLevel::Warning:
            return "Warning";
        case LogLevel::Terse:
            return "Terse";
        case LogLevel::Info:
            return "Info";
        case LogLevel::Verbose:
            return "Verbose";
        case LogLevel::Debug:
            return "Debug";
        case LogLevel::Trace:
            return "Trace";
        case LogLevel::Count:
            return "Count";
        }

        Throw<FatalError>("Unsupported LogLevel.");
    }

    inline std::ostream& operator<<(std::ostream& os, LogLevel const& input)
    {
        return os << toString(input);
    }

    inline std::string toString(F8Mode f8Mode)
    {

        switch(f8Mode)
        {
        case F8Mode::NaNoo:
            return "NaNoo";
        case F8Mode::OCP:
            return "OCP";
        case F8Mode::Count:
            return "Count";
        }

        Throw<FatalError>("Unexpected F8Mode.");
    }

    inline std::ostream& operator<<(std::ostream& os, const F8Mode& input)
    {
        return os << toString(input);
    }

    inline Settings::Settings()
    {
        const char* bitfieldChars = std::getenv(Settings::BitfieldName.c_str());

        if(bitfieldChars)
        {
            bitFieldType bitfield = bitFieldType{strtoull(bitfieldChars, NULL, 0)};

            auto settings = SettingsOptionBase::instances();

            for(auto const* setting : settings)
            {
                if(setting->getBitIndex() >= 0)
                {
                    if(auto envVar = setting->getFromEnv())
                    {
                        Settings::MapWriterLock writerLock{m_mapLock};
                        m_values[setting->name] = *envVar;
                    }
                    else
                    {
                        Settings::MapWriterLock writerLock{m_mapLock};
                        m_values[setting->name] = bitfield.test(setting->getBitIndex());
                    }
                }
            }
        }
    }

    inline SettingsOptionBase::SettingsOptionBase(std::string name, std::string description)
        : name(std::move(name))
        , description(std::move(description))
    {
        m_instances.push_back(this);
    }

    inline std::string SettingsOptionBase::help() const
    {
        return this->name + ": " + this->description;
    }

    inline std::vector<SettingsOptionBase const*> const& SettingsOptionBase::instances()
    {
        return m_instances;
    }

    template <typename T, bool LazyValue>
    inline SettingsOption<T, LazyValue>::SettingsOption(std::string      name,
                                                        std::string      description,
                                                        DefaultValueType defaultValue,
                                                        int              bit)
        : SettingsOptionBase(std::move(name), std::move(description))
        , defaultValue(std::move(defaultValue))
        , bit(bit)
    {
    }

    template <typename T, bool LazyValue>
    inline T SettingsOption<T, LazyValue>::getTypeValue(std::string const& var) const
    {
        if constexpr(CCountedEnum<T>)
        {
            return fromString<T>(var);
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

    template <typename T, bool LazyValue>
    inline std::optional<std::any> SettingsOption<T, LazyValue>::getFromEnv() const
    {
        const char* var = std::getenv(name.c_str());

        if(var)
        {
            std::string s_var = var;
            return getTypeValue(s_var);
        }
        else
        {
            return {};
        }
    }

    template <typename T, bool LazyValue>
    inline std::any SettingsOption<T, LazyValue>::getDefault() const
    {
        if constexpr(LazyValue)
        {
            return defaultValue();
        }
        else
        {
            return defaultValue;
        }
    }

    template <typename T, bool LazyValue>
    inline int SettingsOption<T, LazyValue>::getBitIndex() const
    {
        return bit;
    }

    template <typename T, bool LazyValue>
    inline std::string SettingsOption<T, LazyValue>::help() const
    {
        std::string output = SettingsOptionBase::help() + " ( default: ";
        if constexpr(std::is_same<decltype(this->defaultValue), std::bitset<32>>::value)
        {
            output += this->defaultValue.to_string();
        }
        else if constexpr(std::is_same<decltype(this->defaultValue), std::string>::value)
        {
            if(this->defaultValue.empty())
            {
                output += "\"\"";
            }
            else
            {
                output += this->defaultValue;
            }
        }
        else
        {
            if constexpr(LazyValue)
            {
                output += toString(this->defaultValue());
            }
            else
            {
                output += toString(this->defaultValue);
            }
        }
        if(this->bit >= 0)
        {
            output += ", bit " + std::to_string(this->bit);
        }
        output += " )";
        return output;
    }

    inline std::string Settings::help() const
    {
        auto const& options = SettingsOptionBase::instances();
        std::string prefix  = "Environment variables:\n";
        return std::accumulate(
            options.begin(), options.end(), std::move(prefix), [](auto accum, auto setting) {
                return std::move(accum) + setting->help() + "\n";
            });
    }
}
