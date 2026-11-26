# Lane: PR-diff map (`DIFF`)

**Trigger:** the target is a PR (or, in a later round, the delta since the last round). **Agent:** `scout`. **Output:**
`R/lanes/DIFF.md`: the map, then ideas.

Weight the fuzz towards what the change touched, not the engine as a whole.

## Method

1. `git -C $WT diff <base>..<pr> --stat` (the PR file list truncates at 100), then the diff of everything that reaches this engine:
   descriptors and their lists, kernel objects and generators, matcher rules, tuning rows, knobs, the engine TOML, docs that state
   new support or new declines.
2. For each change, what graphs it affects: newly accepted, newly declined, served by a new or changed kernel, given a new default.
3. For each, a graph that exercises it, and the graph that exercised that code before (on base, or on a shipped arch).
4. What the PR description claims (support, divergences, known issues) against what the diff does.

## Output

A table first:

| Change (path:line) | Effect | Graph that exercises it | Covered by a current space |
|---|---|---|---|

Then ideas for every change no space reaches, and for every claim in the description the diff does not back (or that a space
contradicts: a probe `expect` now stale, an axis missing the new values).
