// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Value.hpp - the language's runtime value.
//
// A small standalone variant (null, bool, int64, double, string, array) that
// deliberately does NOT depend on nlohmann/json: nlohmann expresses the rule
// being compiled, never the values an evaluation produces. Null means
// "unresolved", not a value -- see Operators.hpp for what that costs each
// operator.
//
// Full reference: docs/JsonExpression.md.

#include <hipdnn_data_sdk/utilities/Visitor.hpp>

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
// ===========================================================================
// Value - standalone runtime value (json-like, no nlohmann dependency)
// ===========================================================================
class Value
{
public:
    using Array = std::vector<Value>;

    Value()
        : _v(nullptr)
    {
    }
    Value(std::nullptr_t)
        : _v(nullptr)
    {
    }
    Value(bool b)
        : _v(b)
    {
    }
    Value(int i)
        : _v(static_cast<std::int64_t>(i))
    {
    }
    Value(std::int64_t i)
        : _v(i)
    {
    }
    Value(double d)
        : _v(d)
    {
    }
    Value(const char* s)
        : _v(std::string(s))
    {
    }
    Value(std::string s)
        : _v(std::move(s))
    {
    }
    Value(Array a)
        : _v(std::move(a))
    {
    }

    /// Build a numeric value, storing an integer when the double is exactly
    /// integral and representable, so integer inputs yield integer output.
    static Value number(double d)
    {
        if(std::isfinite(d))
        {
            const double t = std::trunc(d);
            if(t == d && d >= -9.007199254740992e15 && d <= 9.007199254740992e15)
            {
                return {static_cast<std::int64_t>(t)};
            }
        }
        return {d};
    }

    bool isNull() const
    {
        return std::holds_alternative<std::nullptr_t>(_v);
    }
    bool isBool() const
    {
        return std::holds_alternative<bool>(_v);
    }
    bool isInt() const
    {
        return std::holds_alternative<std::int64_t>(_v);
    }
    bool isDouble() const
    {
        return std::holds_alternative<double>(_v);
    }
    bool isNumber() const
    {
        return isInt() || isDouble();
    }
    bool isString() const
    {
        return std::holds_alternative<std::string>(_v);
    }
    bool isArray() const
    {
        return std::holds_alternative<Array>(_v);
    }

    bool asBool() const
    {
        return std::get<bool>(_v);
    }
    std::int64_t asInt() const
    {
        return std::get<std::int64_t>(_v);
    }
    double asDouble() const
    {
        return std::get<double>(_v);
    }
    const std::string& asString() const
    {
        return std::get<std::string>(_v);
    }
    const Array& asArray() const
    {
        return std::get<Array>(_v);
    }

    /// Truthiness: false, 0, "", null and the empty array are falsy.
    bool truthy() const
    {
        return std::visit(
            hipdnn_data_sdk::utilities::Visitor{[](std::nullptr_t) { return false; },
                                                [](bool b) { return b; },
                                                [](std::int64_t i) { return i != 0; },
                                                [](double d) { return d != 0.0; },
                                                [](const std::string& s) { return !s.empty(); },
                                                [](const Array& a) { return !a.empty(); }},
            _v);
    }

    /// JS Number() coercion. Non-numeric strings and multi-element arrays yield
    /// NaN; empty string / empty array / null yield 0.
    double toNumber() const
    {
        return std::visit(hipdnn_data_sdk::utilities::Visitor{
                              [](std::nullptr_t) { return 0.0; },
                              [](bool b) { return b ? 1.0 : 0.0; },
                              [](std::int64_t i) { return static_cast<double>(i); },
                              [](double d) { return d; },
                              [](const std::string& s) { return stringToNumber(s); },
                              [](const Array& a) {
                                  if(a.empty())
                                  {
                                      return 0.0;
                                  }
                                  if(a.size() == 1)
                                  {
                                      return a.front().toNumber();
                                  }
                                  return std::nan("");
                              }},
                          _v);
    }

    /// Structural equality (== / !=). Integers and doubles of equal value
    /// compare equal; differing kinds do not.
    bool operator==(const Value& o) const
    {
        if(isInt() && o.isInt())
        {
            // Exactly, not through double: above 2^53 a double cannot tell two
            // adjacent int64 apart, and this language gates dispatch on sizes,
            // strides and byte offsets, where that is a wrong decision rather
            // than a rounding error.
            return asInt() == o.asInt();
        }
        if(isNumber() && o.isNumber())
        {
            return toNumber() == o.toNumber();
        }
        // Otherwise this is exactly variant equality: differing alternatives are
        // unequal, and each alternative compares with its own operator==.
        return _v == o._v;
    }
    bool operator!=(const Value& o) const
    {
        return !(*this == o);
    }

    /// Ordering result of `compare`. UNORDERED is the NaN case, and makes every
    /// ordering test false.
    enum class Ordering
    {
        LESS,
        EQUAL,
        GREATER,
        UNORDERED
    };

    /// Three-way compare for ordering. Two strings compare lexically; anything
    /// else is compared as a number, and a NaN operand yields UNORDERED.
    static Ordering compare(const Value& a, const Value& b)
    {
        if(a.isString() && b.isString())
        {
            const auto& x = a.asString();
            const auto& y = b.asString();
            if(x < y)
            {
                return Ordering::LESS;
            }
            return x > y ? Ordering::GREATER : Ordering::EQUAL;
        }
        if(a.isInt() && b.isInt())
        {
            // Exactly, for the same reason operator== does: routing two int64
            // through double reports adjacent values above 2^53 as EQUAL, which
            // makes <= and >= both hold on a pair that is neither.
            const std::int64_t x = a.asInt();
            const std::int64_t y = b.asInt();
            if(x < y)
            {
                return Ordering::LESS;
            }
            return x > y ? Ordering::GREATER : Ordering::EQUAL;
        }
        const double x = a.toNumber();
        const double y = b.toNumber();
        if(std::isnan(x) || std::isnan(y))
        {
            return Ordering::UNORDERED;
        }
        if(x < y)
        {
            return Ordering::LESS;
        }
        return x > y ? Ordering::GREATER : Ordering::EQUAL;
    }

    /// Human-readable rendering, mainly for diagnostics and tests.
    std::string dump() const
    {
        return std::visit(hipdnn_data_sdk::utilities::Visitor{
                              [](std::nullptr_t) { return std::string("null"); },
                              [](bool b) { return std::string(b ? "true" : "false"); },
                              [](std::int64_t i) { return std::to_string(i); },
                              [](double d) { return std::to_string(d); },
                              [](const std::string& s) { return "\"" + s + "\""; },
                              [](const Array& a) {
                                  std::string s = "[";
                                  for(std::size_t i = 0; i < a.size(); ++i)
                                  {
                                      s += ((i != 0u) ? "," : "") + a[i].dump();
                                  }
                                  return s + "]";
                              }},
                          _v);
    }

    /// Stream rendering (used by GoogleTest value printing and diagnostics).
    friend std::ostream& operator<<(std::ostream& os, const Value& v)
    {
        return os << v.dump();
    }

private:
    static double stringToNumber(const std::string& s)
    {
        std::size_t b = 0;
        std::size_t e = s.size();
        while(b < e && (std::isspace(static_cast<unsigned char>(s[b])) != 0))
        {
            ++b;
        }
        while(e > b && (std::isspace(static_cast<unsigned char>(s[e - 1])) != 0))
        {
            --e;
        }
        if(b == e)
        {
            return 0.0; // JS Number("") == 0
        }
        const std::string t = s.substr(b, e - b);
        const char* first = t.c_str();
        char* last = nullptr;
        const double d = std::strtod(first, &last);
        if(last != first + t.size())
        {
            return std::nan(""); // trailing garbage -> NaN
        }
        return d;
    }

    // The alternatives are reached by type through std::visit, so their order
    // here carries no meaning.
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array> _v;
};
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
