// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @name Enforce variable naming
 * @description Ensures the naming convention for code base is followed, refer to the repo README for more info.
 * @kind problem
 * @problem.severity error
 * @precision high
 * @id rr/variable-naming
 * @tags maintainability
 *       readability
 */

import cpp
import NamingUtils
import Cases

string pass() { result = "__PASS__" }

predicate isNotLambdaCapture(Declaration v) { exists(v.getNamespace()) }

string followNamingConvention(MemberVariable v) {
  (
    if v.isStatic()
    then result = pass() // TODO: care about static vars
    else (
      if v.isPublic() then result = followPublic(v) else result = followPrivateProtected(v)
    )
  )
}

string followPrivateProtected(MemberVariable v) {
  if
    v.toString().regexpMatch("^m_.*") and
    v.toString().substring(2, v.toString().length()).regexpMatch(camelCase())
  then result = pass()
  else result = "nonstatic private or protected variables should be in m_camelCase"
}

string followPublic(MemberVariable v) {
  if v.toString().regexpMatch(camelCase())
  then result = pass()
  else result = "nonstatic public variables should be in camelCase"
}

from MemberVariable v
where
  isInCheckedDirs(v) and
  isNotLambdaCapture(v) and
  not exists(v.(BitField)) and
  not followNamingConvention(v) = pass()
select v, "`" + v.toString() + "` " + followNamingConvention(v)
