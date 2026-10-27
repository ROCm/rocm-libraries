// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_tools/CliNumbers.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

/// @file CliArguments.hpp
/// @brief Reading one option's value, and why it was refused, as every tool here does it.
namespace hipdnn_tools
{

/// @brief The value of the option at @p index in @p args, read once per option.
///
/// A following flag is not a value: `--output --count 5` must not write to "--count". A number
/// is the whole value or nothing (CliNumbers.hpp), and one outside what the option takes, or
/// what its target can hold, is refused rather than clamped or wrapped.
class OptionReader
{
public:
    OptionReader(const std::vector<std::string>& args, size_t& index)
        : _args(args)
        , _index(index)
    {
    }

    /// The option's value, or empty (and @ref refused reports it) when there is none.
    std::string next()
    {
        if(_index + 1 >= _args.size() || _args[_index + 1].rfind("--", 0) == 0)
        {
            _missingValue = true;
            return {};
        }
        return _args[++_index];
    }

    /// Parses the value with @p parse into @p target, which must be at least @p atLeast and
    /// fit @p target's type.
    template <typename Target, typename Parse, typename Bound>
    void number(Target& target, const Parse& parse, Bound atLeast)
    {
        const auto text = next();
        if(_missingValue)
        {
            return;
        }
        const auto value = parse(text);
        if(!value)
        {
            _badNumber = text;
            return;
        }
        if(*value < atLeast)
        {
            _badNumber = text;
            _bound = " of at least " + render(atLeast);
            return;
        }
        if constexpr(std::is_integral_v<Target>)
        {
            constexpr auto MOST = std::numeric_limits<Target>::max();
            if constexpr(static_cast<uint64_t>(MOST)
                         < static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            {
                if(*value > static_cast<int64_t>(MOST))
                {
                    _badNumber = text;
                    _bound = " of at most " + render(MOST);
                    return;
                }
            }
        }
        target = static_cast<Target>(*value);
    }

    template <typename Target>
    void integer(Target& target, int64_t atLeast)
    {
        number(target, parseInteger, atLeast);
    }

    template <typename Target>
    void real(Target& target, double atLeast)
    {
        number(target, parseReal, atLeast);
    }

    /// Writes why @p flag's value was refused to @p err; true when it was.
    bool refused(const std::string& flag, std::ostream& err) const
    {
        if(_missingValue)
        {
            err << flag << " needs a value\n";
            return true;
        }
        if(_badNumber)
        {
            err << flag << " needs a number" << _bound << ", not '" << *_badNumber << "'\n";
            return true;
        }
        return false;
    }

private:
    template <typename Value>
    static std::string render(Value value)
    {
        std::ostringstream text;
        text << value;
        return text.str();
    }

    const std::vector<std::string>& _args;
    size_t& _index;
    bool _missingValue = false;
    std::optional<std::string> _badNumber;
    std::string _bound;
};

} // namespace hipdnn_tools
