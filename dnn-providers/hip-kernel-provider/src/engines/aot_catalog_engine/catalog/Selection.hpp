// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Applicability matcher. Forked from PR #9207's SelectionConstraints and
// extended with the min/max/multiple_of predicates the plan calls for (the JSON
// schema already declared them; the runtime ignored them). Genericized over the
// op-agnostic ProblemShape rather than the SDPA-named problem.

#pragma once

#include "catalog/CatalogTypes.hpp"

namespace aot_catalog_engine::catalog
{

// True iff every constraint holds against `problem`. A key that is constrained
// but absent from the problem fails closed; an all-empty rule fails closed.
bool satisfies(const Constraints& constraints, const ProblemShape& problem);

} // namespace aot_catalog_engine::catalog
