// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "catalog/Selection.hpp"

#include <algorithm>
#include <cmath>
#include <optional>

namespace aot_catalog_engine::catalog
{

namespace
{

// Extract an integer from a shape value for the range/divisibility predicates.
// int64 is used directly; an integral double is narrowed; anything else (bool,
// string, non-integral double) is not an integer and yields nullopt.
std::optional<int64_t> asInteger(const ShapeValue& value)
{
    if(const auto* i = std::get_if<int64_t>(&value))
    {
        return *i;
    }
    if(const auto* d = std::get_if<double>(&value))
    {
        if(std::floor(*d) == *d)
        {
            return static_cast<int64_t>(*d);
        }
    }
    return std::nullopt;
}

bool ruleHolds(const ConstraintRule& rule, const ShapeValue& value)
{
    // Fail closed on a rule that constrains nothing.
    if(rule.empty())
    {
        return false;
    }

    if(rule.equals.has_value() && value != *rule.equals)
    {
        return false;
    }
    if(rule.notEquals.has_value() && value == *rule.notEquals)
    {
        return false;
    }
    if(!rule.oneOf.empty()
       && std::find(rule.oneOf.begin(), rule.oneOf.end(), value) == rule.oneOf.end())
    {
        return false;
    }

    if(rule.min.has_value() || rule.max.has_value() || rule.multipleOf.has_value())
    {
        const auto integer = asInteger(value);
        if(!integer.has_value())
        {
            return false; // numeric predicate against a non-integer value
        }
        if(rule.min.has_value() && *integer < *rule.min)
        {
            return false;
        }
        if(rule.max.has_value() && *integer > *rule.max)
        {
            return false;
        }
        if(rule.multipleOf.has_value())
        {
            if(*rule.multipleOf == 0 || *integer % *rule.multipleOf != 0)
            {
                return false;
            }
        }
    }

    return true;
}

} // namespace

bool satisfies(const Constraints& constraints, const ProblemShape& problem)
{
    for(const auto& [key, rule] : constraints)
    {
        auto it = problem.find(key);
        if(it == problem.end())
        {
            return false; // constrained key absent from the problem -> fail closed
        }
        if(!ruleHolds(rule, it->second))
        {
            return false;
        }
    }
    return true;
}

} // namespace aot_catalog_engine::catalog
