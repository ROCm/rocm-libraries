# Lane: applicability against servability (`APPL`)

**Trigger:** always. **Agent:** `scout`. **Output:** `R/lanes/APPL.md`, ideas.

An engine should accept a graph only if it serves it, and decline the rest before plan build. Find the graphs where the matcher and
what the engine can actually serve disagree.

## Method

1. List the matcher's rules (brief section 7: matcher files, coverage doc rule ids). For each, the property it reads and its limit.
2. List what the engine can serve: descriptors, catalogs, kernel instances, tuning rows, per arch.
3. For each rule, find graphs at and just past its edges: **accepted but unservable** (no descriptor serves it, a kernel limit the
   matcher does not read, a property the matcher never reads) and **declined but servable** (a rule stricter than the kernels: lost
   coverage). Compare sibling rules across arches, dtypes and forward/backward.
4. Check paths that build or size a plan without the matcher: workspace sizing, cached plans, forced `engine_variant` knobs.
5. Check the probe families in the current spaces: each `expect` must follow a rule id of this engine. Flag any copied from another
   engine (an overflow probe copied from one engine expects `declined` where another engine's matcher accepts).

## Ideas

`Expect: ACCEPT-UNSERVABLE` or `DECLINE-SERVABLE` with the rule id in `Why:`; probe families with a corrected `expect` as
`Place: probe <space>.<family> expect <...>`. Boundary rows (below, at, above the limit) as separate ideas only when they are different graphs.
