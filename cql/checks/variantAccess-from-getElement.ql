/**
 * @kind problem
 * @id rr/variantaccess-from-getelement
 * @problem.severity recommendation
 */

import cpp

from FunctionCall f, FunctionCall getElement
where
  getElement.getTarget().hasName("getElement")
  and (f.getTarget().hasName("get") or f.getTarget().hasName("holds_alternative"))
  and f.getArgument(0) = getElement
  and f.getLocation().getFile().getBaseName() != "Hypergraph_impl.hpp"
select f, "Variant access from getElement; use graph.get<> instead."
