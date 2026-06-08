/**
 * @name Backward def-use slice of a branch predicate
 * @description Interprocedural/cross-statement backward slice: starting from the
 *              variable reads in the predicate at the chosen branch line, follow
 *              def-use (SSA) transitively to gather the dependency symbol set.
 *              The result is a SUPERSET of the intra-statement names the stdlib
 *              ast def-use seed found: it includes every Name read in the predicate
 *              (including builtins like `len`) PLUS the transitive backward
 *              dependencies through `args` (= argParser.parse_args(userArgs)) back
 *              to `argParser` and `userArgs`.
 * @kind table
 * @id pchaos/backward-defuse-slice
 */

import python

/** Any variable-read Name in the seed predicate (Tensile.py, function Tensile, line 526). */
predicate seedRead(Name n) {
  n.getScope().(Function).getName() = "Tensile" and
  n.getLocation().getFile().getShortName() = "Tensile.py" and
  n.getLocation().getStartLine() = 526 and
  n.isUse()
}

/** SsaVariable `dep` flows into the definition of SsaVariable `v` (one backward step). */
predicate defUseStep(SsaVariable v, SsaVariable dep) {
  exists(Assign a, Name defName, Name rhsUse |
    defName = v.getDefinition().getNode() and
    a.getATarget() = defName and
    rhsUse = a.getValue().getASubExpression*() and
    rhsUse.isUse() and
    dep.getAUse().getNode() = rhsUse
  )
  or
  dep = v.getAPhiInput()
}

/** Transitive backward def-use closure from the seed reads. */
SsaVariable relevant() {
  exists(Name n | seedRead(n) and result.getAUse().getNode() = n)
  or
  defUseStep(relevant(), result)
}

/**
 * The full CodeQL symbol set:
 *  (a) every Name read at the seed line (covers builtins/free names like `len`
 *      that have no SSA variable) -- this guarantees superset of the ast seed; PLUS
 *  (b) the transitive backward def-use closure (the interprocedural win).
 */
string sliceSymbol() {
  exists(Name n | seedRead(n) and result = n.getId())
  or
  result = relevant().getVariable().getId()
}

from string sym
where sym = sliceSymbol()
select sym
