// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace hipdnn_integration_tests::bundle
{

namespace detail
{

// `*` matches any run of characters (including none) and `?` matches exactly one.
// Nothing else is special, which is GTest's own rule.
inline bool globMatches(std::string_view pattern, std::string_view text)
{
    constexpr auto NO_STAR = std::string_view::npos;

    std::size_t patternPos = 0;
    std::size_t textPos = 0;
    std::size_t starPos = NO_STAR;
    std::size_t starText = 0;

    while(textPos < text.size())
    {
        if(patternPos < pattern.size() && pattern[patternPos] == '*')
        {
            starPos = patternPos++;
            starText = textPos;
        }
        else if(patternPos < pattern.size()
                && (pattern[patternPos] == '?' || pattern[patternPos] == text[textPos]))
        {
            ++patternPos;
            ++textPos;
        }
        else if(starPos != NO_STAR)
        {
            // Let the last `*` swallow one more character and retry from there.
            patternPos = starPos + 1;
            textPos = ++starText;
        }
        else
        {
            return false;
        }
    }

    while(patternPos < pattern.size() && pattern[patternPos] == '*')
    {
        ++patternPos;
    }
    return patternPos == pattern.size();
}

// True if `predicate` holds for any of the `:`-separated patterns. An empty list holds
// one empty pattern.
template <typename Predicate>
bool anyPattern(std::string_view patterns, const Predicate& predicate)
{
    std::size_t start = 0;
    while(true)
    {
        const auto end = patterns.find(':', start);
        const auto pattern = end == std::string_view::npos ? patterns.substr(start)
                                                           : patterns.substr(start, end - start);
        if(predicate(pattern))
        {
            return true;
        }
        if(end == std::string_view::npos)
        {
            return false;
        }
        start = end + 1;
    }
}

// True if any of the `:`-separated patterns matches. An empty pattern matches only an
// empty name, so "" selects nothing.
inline bool anyPatternMatches(std::string_view patterns, std::string_view name)
{
    return anyPattern(patterns,
                      [name](std::string_view pattern) { return globMatches(pattern, name); });
}

struct GTestFilterParts
{
    std::string_view positive;
    std::string_view negative;
};

// Split at the first `-`. An empty positive part before a `-` means `*`.
inline GTestFilterParts splitGTestFilter(std::string_view filter)
{
    const auto dash = filter.find('-');
    GTestFilterParts parts{filter.substr(0, dash), {}};
    if(dash != std::string_view::npos)
    {
        parts.negative = filter.substr(dash + 1);
        if(parts.positive.empty())
        {
            parts.positive = "*";
        }
    }
    return parts;
}

// The pattern positions that are live after `globMatches` consumes `prefix`. Position
// `pattern.size()` means the whole pattern is consumed. A `*` at a live position also
// makes the next position live, because the `*` can match no characters.
inline std::vector<bool> globLivePositionsAfter(std::string_view pattern, std::string_view prefix)
{
    const auto addWithStarClosure = [pattern](std::vector<bool>& live, std::size_t position) {
        while(position <= pattern.size() && !live[position])
        {
            live[position] = true;
            if(position == pattern.size() || pattern[position] != '*')
            {
                break;
            }
            ++position;
        }
    };

    std::vector<bool> live(pattern.size() + 1, false);
    addWithStarClosure(live, 0);
    for(const char c : prefix)
    {
        std::vector<bool> next(pattern.size() + 1, false);
        bool anyLive = false;
        for(std::size_t position = 0; position < pattern.size(); ++position)
        {
            if(!live[position])
            {
                continue;
            }
            if(pattern[position] == '*')
            {
                addWithStarClosure(next, position);
                anyLive = true;
            }
            else if(pattern[position] == '?' || pattern[position] == c)
            {
                addWithStarClosure(next, position + 1);
                anyLive = true;
            }
        }
        live = std::move(next);
        if(!anyLive)
        {
            break;
        }
    }
    return live;
}

// True if some text T makes `prefix + T` match `pattern`. Every pattern tail matches
// some text, so a live position after `prefix` is sufficient.
inline bool globMayMatchSomeExtension(std::string_view pattern, std::string_view prefix)
{
    const auto live = globLivePositionsAfter(pattern, prefix);
    return std::find(live.begin(), live.end(), true) != live.end();
}

// True if `prefix + T` matches `pattern` for every text T. This holds when a live
// position after `prefix` starts a tail of only `*`. The check can return false for
// some patterns that match every extension.
inline bool globMatchesEveryExtension(std::string_view pattern, std::string_view prefix)
{
    const auto live = globLivePositionsAfter(pattern, prefix);
    for(std::size_t position = 0; position < pattern.size(); ++position)
    {
        if(live[position] && pattern.find_first_not_of('*', position) == std::string_view::npos)
        {
            return true;
        }
    }
    return false;
}

} // namespace detail

/// Whether `--gtest_filter=<filter>` would run the test named `fullName`
/// ("Suite.Test"), by GTest's grammar: `POSITIVE[-NEGATIVE]`, each a `:`-separated
/// list of `*`/`?` globs, split at the first `-`. An empty positive part before a
/// `-` means everything.
///
/// Registration uses this to skip loading bundles the run is about to filter out.
/// GTest only applies the filter inside RUN_ALL_TESTS(), long after every bundle
/// would otherwise have been parsed, expanded and had its tensors read, so a run
/// that selects one test paid for all of them. GTest's own matcher is internal, so
/// this reimplements it; keeping the two in step is what GTestFilter's unit tests pin.
inline bool gtestFilterSelects(std::string_view filter, std::string_view fullName)
{
    const auto parts = detail::splitGTestFilter(filter);
    return detail::anyPatternMatches(parts.positive, fullName)
           && !detail::anyPatternMatches(parts.negative, fullName);
}

/// Whether `--gtest_filter=<filter>` can select some test of the suite `suiteName`.
/// False means that gtestFilterSelects() is false for every "suiteName.Test", so the
/// suite's bundles need no parse. True can also occur for a suite with no selected test.
inline bool gtestFilterMaySelectSuite(std::string_view filter, std::string_view suiteName)
{
    std::string prefix{suiteName};
    prefix += '.';

    const auto parts = detail::splitGTestFilter(filter);
    const auto mayMatch = [&prefix](std::string_view pattern) {
        return detail::globMayMatchSomeExtension(pattern, prefix);
    };
    const auto matchesAll = [&prefix](std::string_view pattern) {
        return detail::globMatchesEveryExtension(pattern, prefix);
    };
    return detail::anyPattern(parts.positive, mayMatch)
           && !detail::anyPattern(parts.negative, matchesAll);
}

} // namespace hipdnn_integration_tests::bundle
