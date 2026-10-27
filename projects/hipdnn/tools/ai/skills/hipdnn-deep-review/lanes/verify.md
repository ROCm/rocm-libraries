# Verifier (`VER`)

You try to disprove findings someone else reported. You receive the dispatch inputs (no context packet) and a batch of findings with their
lane reports.

Where this file and [common.md](common.md) conflict, this file wins: write
`$EVIDENCE/r<N>/verify/<ID>.md` per finding ([../findings.md](../findings.md) *Verifier
output*) and any defect you find yourself to `verify/VER<batch>.md` (`VER<batch>-p<n>`), work in `$SCRATCH/VER-<batch>/`, and judge a ledger `REFUTED` claim on its merits.

## Per finding

For `SUGGESTION` and `FUTURE WORK` do steps 1, 2 and 4 only (a `source-claim` re-read); to raise
one to `IMPORTANT` or above, also run step 3.

1. Re-read every cited `path:line` at `$HEAD` (`$BASE` for removed content). Read a
   `<sha>:<path>:<line>` anchor with `git -C "$CLONE" show <sha>:<path>`, a `commit <sha>`
   anchor with `git -C "$CLONE" log -1 --format='%an <%ae>%n%cn <%ce>%n%B' <sha>`, and `pr-*:`
   and `branch:` anchors in `$EVIDENCE/r<N>/pr/`. Correct wrong anchors. If the code does not say what the finding says, `REFUTED`.
2. Look for what the reporter missed: an upstream guard, a caller that never passes the
   input, a test elsewhere, a document that makes the behavior intentional. Start with the
   docs, RFCs and tests the area files list.
3. Run the proof [../findings.md](../findings.md) *Proof per kind* requires.
   - Reuse the build directory, configure line and binaries `RUN` recorded under **Checked, no finding**, and never mutate the
     shared `$SCRATCH/RUN/` tree. Every scratch build disables clang-tidy as `RUN` does
     ([build-validate.md](build-validate.md) step 3). With none, build
     the smallest target that holds the proof (a full checkout is `fullclone "$SCRATCH/VER-<batch>/src" "$HEAD"`), or return `UNVERIFIED` with `no build
     available`.
   - **Mutating a built tree.** A build tree is bound to its absolute path (the generator hashes command lines), so a copy at
     another path rebuilds everything. Use `Mutation:` from `dispatch-context.md` when it names a recipe. Otherwise, in order:
     (1) a private copy visible at the original absolute path, where the host provides one (mount namespace, container or VM);
     (2) the shared tree in place, serialized with a lock, reverted and baseline-checked after every mutant; (3) your own
     full build in `$SCRATCH/VER-<batch>/`. Record which you used.
   - **defect:** the smallest reproduction (test, probe or CLI call) that fails at `$HEAD`.
     Run it at `$BASE` when the finding says the PR introduced it.
   - **test-gap:** confirm the named tests pass unmutated; mutate the behavior the finding
     names, inside the changed code or what it calls. Rebuild, rerun, revert, and recheck
     the baseline before the next mutant. Discard a mutant that changes no observable
     behavior. When the reporter already mutated, rerun only the 3 most decisive mutants and state which were reused.
4. Set the verdict and the tier you can defend; `PARTIAL` states the narrower claim.

## Handoffs

A finding that duplicates or overlaps another is not a finding of yours. Write `ORCH: cluster <ID> with <ID>` under **Handoffs**
and give each its own verdict. The orchestrator clusters them ([../RUNBOOK.md](../RUNBOOK.md) *Merge and verify*).

Use the claim's revision and configuration. A proof awaiting approval stays `UNVERIFIED`;
put the request in the verify file.
