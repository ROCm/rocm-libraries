// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**

/**
 * @name Variant access from get element
 * @description Variant access from get element
 * @kind problem
 * @problem.severity recommendation
 * @precision high
 * @id rr/variantaccess-from-getelement
 * @tags
 */

import cpp

from FunctionCall f, FunctionCall getElement
where
  getElement.getTarget().hasName("getElement") and
  (f.getTarget().hasName("get") or f.getTarget().hasName("holds_alternative")) and
  f.getArgument(0) = getElement and
  (f.getLocation().getFile().getBaseName() != "Hypergraph_impl.hpp" and f.getLocation().getFile().getBaseName() != "HypergraphTest.cpp")
select f, "Variant access from getElement; use graph.get<> instead."
