/**
 * @kind problem
 * @id rr/visit-from-call-operator
 * @problem.severity recommendation
 */


import cpp

from FunctionCall fc
// Calls to std::visit()
where (fc.getTarget().hasQualifiedName("std::__1", "visit") 
        or fc.getTarget().hasQualifiedName("std", "visit"))
// From functions named 'operator()'
and fc.getEnclosingFunction().hasName("operator()")
// Where the expression for argument 0 makes reference to 'this'
and fc.getArgument(0).getAChild() instanceof ThisExpr
select fc, "Call to " + fc.getTarget().getQualifiedName() 
         + " from " + fc.getEnclosingFunction().getQualifiedName() 
