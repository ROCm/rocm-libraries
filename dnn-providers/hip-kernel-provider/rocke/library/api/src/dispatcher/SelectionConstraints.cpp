// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "dispatcher/SelectionConstraints.hpp"

#include <algorithm>

namespace rocke_client::dispatcher
{

bool attributesMatchConstraints(const std::map<std::string, AttrValue>& attributes,
                                const AttributeConstraints& constraints)
{
    for(const auto& [name, rule] : constraints)
    {
        const auto it = attributes.find(name);
        if(it == attributes.end())
        {
            // A constrained attribute the problem does not expose cannot match.
            return false;
        }
        const AttrValue& value = it->second;

        if(rule.empty())
        {
            // Malformed constraint (no operator). Never matches; the catalog parser
            // must reject these at parse time to mirror the Python producer.
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
        if(rule.oneOf.has_value())
        {
            const auto& options = *rule.oneOf;
            if(std::ranges::find(options, value) == options.end())
            {
                return false;
            }
        }
    }
    return true;
}

bool satisfies(const AotInstance& instance, const SdpaProblem& problem)
{
    return satisfies(instance, problem, problem.attributes());
}

bool satisfies(const AotInstance& instance,
               const SdpaProblem& problem,
               const std::map<std::string, AttrValue>& problemAttributes)
{
    const CompileSpec& spec = instance.compileSpec;

    // Exact shape match. block_size_{q,k} are kernel tiling, not selection keys.
    if(spec.dtype != problem.dtype)
    {
        return false;
    }
    if(spec.canonicalLayout != toString(problem.layout))
    {
        return false;
    }
    if(spec.seqlenQ != problem.seqlenQ || spec.seqlenK != problem.seqlenK)
    {
        return false;
    }
    if(spec.numQueryHeads != problem.numQueryHeads || spec.numKvHeads != problem.numKvHeads)
    {
        return false;
    }
    if(spec.headSize != problem.headSize)
    {
        return false;
    }
    if(spec.maskMode != problem.maskMode)
    {
        return false;
    }

    // Batch range (inclusive).
    if(problem.batch < instance.batch.min || problem.batch > instance.batch.max)
    {
        return false;
    }

    // Runtime attribute constraints.
    return attributesMatchConstraints(problemAttributes, instance.attributeConstraints);
}

} // namespace rocke_client::dispatcher
