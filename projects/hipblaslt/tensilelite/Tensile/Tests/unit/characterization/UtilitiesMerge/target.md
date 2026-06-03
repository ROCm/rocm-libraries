# Utilities/merge.py — characterization target

Pins the library-logic merge tooling: path/file helpers, reindex/fixSize/
cmpHelper/addKernel/sanitize/removeUnusedKernels, loadData, compare* guards,
the Tag enums + strToScalarValueTag, getSolutionTag/findSolutionWithIndex,
tag-key add/remove, findFastestCompatibleSolution, mergeLogic (improve / reject
/ force / new-size / addSolutionTags), and the file-IO end-to-ends
mergePartialLogics + avoidRegressions (merge + copy-through).

Coverage: 314 stmts, 7 missed → 97.8% line (96.05% blended).

Residual misses are out of reach without low value:
- 169/174-176: removeUnusedKernels debug() prints (verbosity>=2 AND an unused
  solution simultaneously).
- 351/354: duplicate-size warnings (need same size with differing tags).
- 359->365/383->385/393: the `addSolutionTags` reject-via-findFastestCompatible
  branch — getSolutionTag emits only a 1-tuple MFMA tag, so the merge path feeds
  a scalar IntEnum into findFastestCompatibleSolution's `tags[1]`, which would
  raise. The function is tested standalone with its intended 4-tuple contract.
