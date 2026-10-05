# Refresh LDSTrInstA/LDSTrInstB results after rebasing onto develop

## Status

Accepted

## Context

This branch rebases the per-tensor `LDSTrInstA`/`LDSTrInstB` override feature
(see ADR 0029) onto a much newer `develop` tip. 45 `_codegen` golden `.ambr`
files had real merge conflicts (develop had independently regenerated the same
basenames via unrelated refactors); these were provisionally resolved to
develop's pre-rebase content during the rebase itself.

After the rebase completed and `rocisa` was rebuilt in-tree, running the full
characterization suite showed 82 failing nodes across 54 saved-result files:
the 45 conflicted `_codegen` files (stale basenames from the provisional
resolution) plus 3 previously-passing families (`LibraryIO`,
`SolutionClass`, `ValidParameters`) that pin the full derived-parameter-state
dict and valid-parameter-name roster, which had not yet been told about the
two new `LDSTrInstA`/`LDSTrInstB` keys this feature adds.

For every affected node, only `basename` text and/or the two new
`LDSTrInstA`/`LDSTrInstB` dict entries changed; kernel counts and `err`/return
codes are unchanged.

## Decision

Re-record exactly the 82 failing saved-result nodes (54 files) identified by
running the full characterization suite post-rebase, using scoped
`pytest <node_ids> --snapshot-update`. No other goldens were touched.

## Consequences

The saved results now reproduce against current `develop` combined with this
branch's `LDSTrInstA`/`LDSTrInstB` feature and a freshly built in-tree
`rocisa`. Future name or parameter-roster changes require the same count and
return-code audit; the saved basenames alone are not evidence that emitted
assembly is correct.
