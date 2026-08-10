// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Applicability matcher. Forked from PR #9207's SelectionConstraints and
// extended with the min/max/multiple_of predicates the plan calls for (the JSON
// schema already declared them; the runtime ignored them). Genericized over the
// op-agnostic ProblemShape rather than the SDPA-named problem.

#pragma once

#include <string>

#include "catalog/CatalogTypes.hpp"

namespace aot_catalog_engine::catalog
{

// True iff every constraint holds against `problem`. A key that is constrained
// but absent from the problem fails closed; an all-empty rule fails closed.
bool satisfies(const Constraints& constraints, const ProblemShape& problem);

// Human-readable reason the constraints do NOT hold against `problem`, or an
// empty string when they DO. Names the first failing key -- either absent from
// the problem shape (the "typo'd / unpublished constraint key silently never
// matches" trap) or the predicate it violates. For HIPDNN_AOT_DEBUG diagnostics
// only; not on the hot path.
std::string explainMismatch(const Constraints& constraints, const ProblemShape& problem);

// "key=value, key=value, ..." rendering of a decoded problem shape, for
// HIPDNN_AOT_DEBUG diagnostics.
std::string describeShape(const ProblemShape& problem);

} // namespace aot_catalog_engine::catalog
