# Verifier (`VER`)

**Trigger:** *Decide* (idea mode) and *Verify* (finding mode). **Agent:** an agent that did not report what it verifies: `scout` for
ideas and for findings that need only reading; `task` when a replay is needed (within `Approved`). **Output:** `R/verify/<ID>.md` per
item ([../findings.md](../findings.md) *Verifier output*).

You try to **disprove** what another lane claimed. A claim survives only if you could not break it.

## Idea mode (decisive ideas)

1. Re-read every cited `path:line` in `WT`. If the code does not say what the idea says, `REFUTED`.
2. Look for what the lane missed: an earlier guard (frontend, backend, matcher, plan builder, the harness) that stops the graph; a
   descriptor or kernel that does serve it; a doc or rule id that makes the behavior intended; a ledger row that already covers it.
3. Check the graph is legal and buildable: the space format can express it, the harness can fill and grade it (brief section 9), it
   is judgeable ([../policy.md](../policy.md) *Judgeability*).
4. Verdict: `CONFIRMED` (the graph should do what `Expect` says), `REFUTED` (with the guard or row that stops it), `PARTIAL` (narrow
   the graph or the expectation in `Claim:`), `UNVERIFIED` (what would settle it).

## Finding mode

1. Re-read the finding's run files: the failure folders, the outcome records, the tensor numbers. Correct any number that is wrong.
2. Check the proof against [../policy.md](../policy.md) *Proof per kind*. Run what is missing when `Approved` allows it: a solo replay
   (`--row`), a replay on base, the default-domain replay, the class-tolerance replay, the two-reference grade.
3. Look for the other explanation: for a `defect`, conditioning or a tolerance class or a documented divergence; for `conditioning`
   or `tolerance`, a real engine error hiding in the group (a few rows that fail both references); for a `regression`, a graph that is
   not the same on both builds (repair rows); for `pre-existing`, a base run that did not actually execute the graph.
4. Verdict as above. A finding whose kind is right but whose severity or group is wrong is `PARTIAL` with the corrected claim.
