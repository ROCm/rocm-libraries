#pragma once

#include <common/SourceMatcher.hpp>

MATCHER_P(MatchesSource, ref, "")
{
    auto normalizedRef = NormalizedSource(ref, false);
    auto normalizedArg = NormalizedSource(arg, false);

    return ExplainMatchResult(::testing::StrEq(normalizedRef), normalizedArg, result_listener);
}

MATCHER_P(MatchesSourceIncludingComments, ref, "")
{
    auto normalizedRef = NormalizedSource(ref, true);
    auto normalizedArg = NormalizedSource(arg, true);

    return ExplainMatchResult(::testing::StrEq(normalizedRef), normalizedArg, result_listener);
}
