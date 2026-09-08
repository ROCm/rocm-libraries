// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Value.hpp - the language's runtime value.
//
// A small standalone variant (null, bool, int64, double, string, array). It
// does not depend on nlohmann/json, because nlohmann represents the rule being
// compiled, not the values an evaluation produces. Null means "unresolved"
// rather than being a value of its own; Operators.hpp describes what that
// means for each operator.
//
// Full reference: docs/JsonExpression.md.

#include <hipdnn_data_sdk/utilities/Visitor.hpp>

#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor::jsonexpr
{
/// How deeply a Value handed back by a data source may nest.
///
/// A Value is a tree, and every consumer of one walks it recursively:
/// containsUnresolved, dump, toNumber over an array, variant equality, and the
/// destructor. None of those can bound their own depth after the fact, so the
/// bound belongs where a Value is built.
///
/// MAX_EXPRESSION_DEPTH bounds a *rule*; this bounds a *document*. The two are
/// separate limits on separate inputs that happen to share a magnitude, and a
/// document is no more trusted than a rule: both are read off disk. A data
/// source must not build a Value deeper than this. JsonDataSource stops at the
/// bound and yields null, which the language reads as unresolved, so the
/// enclosing predicate declines rather than the process overflowing its stack.
inline constexpr std::size_t MAX_VALUE_DEPTH = 256;

// ---- runtime value --------------------------------------------------------
// Json-like, with no nlohmann dependency.
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

    /// True when this value is null, or is an array holding a null anywhere
    /// inside it. Null is the language's unresolved marker, so an eager
    /// operator uses this to avoid answering from a partly resolved array.
    bool containsUnresolved() const
    {
        if(isNull())
        {
            return true;
        }
        if(!isArray())
        {
            return false;
        }
        for(const Value& item : asArray())
        {
            if(item.containsUnresolved())
            {
                return true;
            }
        }
        return false;
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

    /// Structural equality (== / !=). An integer equals a double only when the
    /// double is exactly that integer. Values of different, non-numeric kinds
    /// are never equal.
    bool operator==(const Value& o) const
    {
        if(isInt() && o.isInt())
        {
            // Compared as int64, not through double. Above 2^53 a double cannot
            // distinguish adjacent int64 values, and this language gates
            // dispatch on sizes, strides and byte offsets, so that would be a
            // wrong decision rather than a rounding error.
            return asInt() == o.asInt();
        }
        if(isInt() && o.isDouble())
        {
            return intEqualsDouble(asInt(), o.asDouble());
        }
        if(isDouble() && o.isInt())
        {
            return intEqualsDouble(o.asInt(), asDouble());
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

    /// Reports the outcome of `compare`. UNORDERED means at least one operand
    /// was not finite, which makes the ordering operators decline.
    enum class Ordering
    {
        LESS,
        EQUAL,
        GREATER,
        UNORDERED
    };

    /// Three-way comparison. Two strings compare lexically. An int64 and a
    /// double compare without rounding the integer through double. A string
    /// opposite a non-string is UNORDERED, and so is a non-finite operand.
    /// Anything else compares as a number.
    ///
    /// Ordering agrees with operator== on what a kind mismatch means. `==`
    /// answers false for "1" against 1, because the two are different kinds
    /// rather than different values; ordering must not then answer true for
    /// "1" <= 1 by quietly coercing the string. One expression reporting a
    /// pair as unequal *and* as ordered is a contradiction a rule author
    /// cannot reason about, and the coercing answer is the widening one: it
    /// lets a criterion pass on a comparison that was never really made.
    ///
    /// Declining instead is fail-closed and surfaces the real fault, which is
    /// a data source modelling a number as a string. Arithmetic still coerces,
    /// so {"+": ["2", "3"]} is 5; only the predicates that gate criteria are
    /// strict.
    static Ordering compare(const Value& a, const Value& b)
    {
        if(a.isString() != b.isString())
        {
            return Ordering::UNORDERED; // a kind mismatch is not an ordering
        }
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
            // Compared as int64 for the same reason operator== is: routing two
            // int64 through double reports adjacent values above 2^53 as EQUAL,
            // which makes <= and >= both hold on a pair that is neither.
            const std::int64_t x = a.asInt();
            const std::int64_t y = b.asInt();
            if(x < y)
            {
                return Ordering::LESS;
            }
            return x > y ? Ordering::GREATER : Ordering::EQUAL;
        }
        if(a.isInt() && b.isDouble())
        {
            return compareIntAndDouble(a.asInt(), b.asDouble());
        }
        if(a.isDouble() && b.isInt())
        {
            return reverse(compareIntAndDouble(b.asInt(), a.asDouble()));
        }
        const double x = a.toNumber();
        const double y = b.toNumber();
        if(!std::isfinite(x) || !std::isfinite(y))
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
    static constexpr double INT64_UPPER_EXCLUSIVE_AS_DOUBLE = 9223372036854775808.0;

    static bool doubleRepresentsInt64(double d)
    {
        return std::isfinite(d) && std::trunc(d) == d
               && d >= static_cast<double>(std::numeric_limits<std::int64_t>::min())
               && d < INT64_UPPER_EXCLUSIVE_AS_DOUBLE;
    }

    static bool intEqualsDouble(std::int64_t i, double d)
    {
        return doubleRepresentsInt64(d) && i == static_cast<std::int64_t>(d);
    }

    /// The build enables -Wswitch-default, so the `default:` is required even
    /// though every Ordering is already handled above it.
    static Ordering reverse(Ordering c)
    {
        switch(c)
        {
        case Ordering::LESS:
            return Ordering::GREATER;
        case Ordering::GREATER:
            return Ordering::LESS;
        case Ordering::EQUAL:
        case Ordering::UNORDERED:
        default:
            return c;
        }
    }

    static Ordering compareIntAndDouble(std::int64_t i, double d)
    {
        if(!std::isfinite(d))
        {
            return Ordering::UNORDERED;
        }
        if(d < static_cast<double>(std::numeric_limits<std::int64_t>::min()))
        {
            return Ordering::GREATER;
        }
        if(d >= INT64_UPPER_EXCLUSIVE_AS_DOUBLE)
        {
            return Ordering::LESS;
        }

        const auto whole = static_cast<std::int64_t>(d);
        if(i < whole)
        {
            return Ordering::LESS;
        }
        if(i > whole)
        {
            return Ordering::GREATER;
        }

        const auto wholeAsDouble = static_cast<double>(whole);
        if(wholeAsDouble < d)
        {
            return Ordering::LESS;
        }
        if(wholeAsDouble > d)
        {
            return Ordering::GREATER;
        }
        return Ordering::EQUAL;
    }
    /// Read a numeric string, the way JS Number() does for the spellings this
    /// language accepts. Surrounding whitespace is trimmed and an empty string
    /// reads as 0; anything else must be a number in full, or the result is
    /// NaN and every operator downstream declines.
    ///
    /// std::from_chars rather than std::strtod, for two reasons:
    ///
    ///   - strtod is locale-dependent. Under a comma-decimal LC_NUMERIC it
    ///     reads "1.5" as 1, and a library evaluating rules off disk does not
    ///     control the host process's locale. from_chars is locale-independent
    ///     by specification.
    ///   - strtod reports a range error only through errno, which this code
    ///     did not check, so "1e-999" underflowed to a clean 0 - a number the
    ///     language never represented, which then compares equal to a literal
    ///     0 and divides as a zero divisor. from_chars reports the same case
    ///     as result_out_of_range, and reports it *without* flagging a genuine
    ///     denormal such as "5e-324", which errno would have.
    ///
    /// The switch narrows two spellings, both deliberately. A hexadecimal
    /// float ("0x10") is no longer read as 16: in a descriptor that is a typo,
    /// not a value. A leading '+' is restored by hand below, because
    /// from_chars rejects it and "+5" is an ordinary way to write 5.
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
        if(s[b] == '+')
        {
            ++b; // from_chars rejects the sign strtod accepted
            if(b == e)
            {
                return std::nan(""); // a bare sign is not a number
            }
        }
        const char* first = s.data() + b;
        const char* last = s.data() + e;
        double d = 0.0;
        const std::from_chars_result r = std::from_chars(first, last, d);
        if(r.ec != std::errc() || r.ptr != last)
        {
            // Trailing garbage, an unreadable spelling, or a magnitude outside
            // the double range in either direction. NaN is unorderable, so the
            // enclosing predicate declines rather than answering from a value
            // that was never read.
            return std::nan("");
        }
        return d;
    }

    // std::visit selects an alternative by type, so the order here does not
    // matter.
    std::variant<std::nullptr_t, bool, std::int64_t, double, std::string, Array> _v;
};
} // namespace hipdnn_plugin_sdk::ingestor::jsonexpr

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
