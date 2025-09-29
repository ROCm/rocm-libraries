// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @name Visit from call operator
 * @description Visit from call operator
 * @kind problem
 * @problem.severity error
 * @precision high
 * @id rr/visit-from-call-operator
 * @tags
 */

import cpp

from FunctionCall fc
// Calls to std::visit()
where
  (
    fc.getTarget().hasQualifiedName("std::__1", "visit") or
    fc.getTarget().hasQualifiedName("std", "visit")
  ) and
  // From functions named 'operator()'
  fc.getEnclosingFunction().hasName("operator()") and
  // Where the expression for argument 0 makes reference to 'this'
  fc.getArgument(0).getAChild() instanceof ThisExpr
select fc,
  "Call to " + fc.getTarget().getQualifiedName() + " from " +
    fc.getEnclosingFunction().getQualifiedName()
