/**
 * @name Checks for aliases for pointer types
 * @description Finding declarations of 'std::shared_ptr<Type>' (and other std dynamic memory types) when an alias 'using TypePtr = std::shared_ptr<Type>' exists.
 * @kind problem
 * @problem.severity warning
 * @precision medium
 * @id rr/ptr-type-alias
 * @tags maintainability
 *       readability
 */

import cpp
import NamingUtils

string basicSharedPtrRegex() {
  result = "^shared_ptr<[^<>]+>$" // Notice the `[^<>]+` excluding more complex types
}

from UsingAliasTypedefType t, Variable v, Namespace n
where
  isInCheckedDirs(t) and
  isInCheckedDirs(v) and
  not v.declaredUsingAutoType() and
  t.getNamespace() = n and
  (v.getNamespace() = n or n.getAChildNamespace() = v.getNamespace()) and
  v.getUnderlyingType() = t.getUnderlyingType() and
  v.getType().toString().regexpMatch(basicSharedPtrRegex())
select v,
  "There exists an alias for `" + v.getUnderlyingType().toString() + "`, most likely being`" +
    v.getUnderlyingType().toString().regexpReplaceAll("^shared_ptr<", "").regexpReplaceAll(">$", "")
    + "Ptr`, but is currently declared as `" + v.getType().toString() + "`"
// Directly logging `t.getBaseType().toString()` adds weird strings to the query
