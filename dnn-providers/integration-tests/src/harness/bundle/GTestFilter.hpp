// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

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

// True if any of the `:`-separated patterns matches. An empty pattern matches only an
// empty name, so "" selects nothing.
inline bool anyPatternMatches(std::string_view patterns, std::string_view name)
{
    std::size_t start = 0;
    while(true)
    {
        const auto end = patterns.find(':', start);
        const auto pattern = end == std::string_view::npos ? patterns.substr(start)
                                                           : patterns.substr(start, end - start);
        if(globMatches(pattern, name))
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
    const auto dash = filter.find('-');
    auto positive = filter.substr(0, dash);
    const auto negative
        = dash == std::string_view::npos ? std::string_view{} : filter.substr(dash + 1);

    if(dash != std::string_view::npos && positive.empty())
    {
        positive = "*";
    }

    return detail::anyPatternMatches(positive, fullName)
           && !detail::anyPatternMatches(negative, fullName);
}

} // namespace hipdnn_integration_tests::bundle
