# Findings

Formats for lane reports and verifier files. Tiers and the M-rules are in
[policy.md](policy.md).

## Finding

```markdown
ID:        <LANE>-p<n>         provisional; the orchestrator assigns the final ID
Kind:      defect | test-gap | source-claim | process
Tier:      BLOCKING | IMPORTANT | SUGGESTION | FUTURE WORK
Where:     path:line read from $HEAD, every location involved; $BASE:path:line for removed content
Expected:  the contract and its source (header, doc, schema, test, PR description, policy rule)
Actual:    observed behavior and the input or path that reaches it
Proof:     what was run or read: command, exit code, log as $EVIDENCE/<relative path>
Fix:       smallest change that resolves it, or the decision a human must make
```

Write the block as plain lines, never inside a code fence. `Actual` and `Fix` are written for the PR author, who posts them
nearly verbatim: no first person, no lane IDs and no `$VAR`s (`path:line` anchors, quoted code and final IDs are fine).

`<LANE>` is the lane's name (`VER` for a verifier). `<n>` counts from 1 within the report; a
follow-up report continues after the lane's last `p<n>`. Read each anchor from `$HEAD`
(`$REPO`, or `git -C "$CLONE" show` outside the sparse set), not from a diff hunk or an
earlier head. Mark anything not observed directly `[INFERENCE]`. Never write absolute local
paths: cite `$EVIDENCE/...`, `$SCRATCH/...` or repository-relative paths.

File a finding only when you can name the input or situation that makes the code or text
produce a wrong result or decision, and the consequence. Style and preference are not
findings. At `IMPORTANT` or above, **Proof** holds the reproduction or the quoted
contradiction and the policy rule it relies on.

## Proof per kind

| Kind | Confirmed when | `REFUTED` when | `UNVERIFIED` when |
|---|---|---|---|
| defect | A reproduction fails at `$HEAD` for the stated reason, and passes or is absent at `$BASE` if the finding says the PR introduced it | It passes at `$HEAD`, or fails for another reason | It cannot run here (device, declined approval, dependency, no build) |
| test-gap | A plausible, non-equivalent mutation in the claimed behavior survives the named tests, which pass unmutated | The mutation is killed | The tests cannot run, or only skip |
| source-claim | Every cited `path:line` supports it and no other location contradicts it | A cited or uncited location contradicts it | - |
| process | The PR text, CI run or label the rule needs is absent or contradicts the claim; for skill or brief text, the cited text lacks or contradicts the requirement | The artifact exists and supports the claim, or the cited text states the requirement | The artifact is not readable |

A `SUGGESTION` or `FUTURE WORK` finding of any kind is verified as a `source-claim` re-read.
No fix is applied to confirm a defect. A compile failure, a skip or a failure for another
reason proves nothing. Record each mutation's location, diff, test filter and result in the
verify file's directory.

## Verdicts

A verifier returns `CONFIRMED`, `PARTIAL` (a narrower claim holds; restate it), `DOWNGRADED`
or `UPGRADED` (tier changed, with the reason), `REFUTED` or `UNVERIFIED` (what blocked it).
`CONFIRMED` and `REFUTED` are equal outcomes: never look for a reason to confirm, and
correct tier inflation as readily as errors.

A second verifier, who is neither the reporter nor the first verifier, rechecks a `REFUTED`
verdict on a `BLOCKING` or `IMPORTANT` finding and any verdict whose final tier is on the
other side of the `IMPORTANT`/`SUGGESTION` line from the reported tier; an upgrade needs the
proof its kind requires. If both agree the verdict stands; otherwise the second applies. At a
new head, state the evidence class: `rerun` (the reproduction or mutation ran again),
`text-read` (the text was compared with the claim) or `not-exercised`.

## Lane report

`$EVIDENCE/r<N>/lanes/<LANE>.md`; a follow-up writes `lanes/<LANE>-f<k>.md` with the same
sections. The first line is `# <LANE> (round <N>, <$HEAD_SHORT>)`. Sections, in order:

1. **Findings** in the format above (may be empty; **Checked, no finding** is then the
   content).
2. **Coverage:** one row per examined file: path, read depth (`full`, `diff-only`,
   `sampled`; nothing else in that cell), tests that reach it. `RUN` adds two literal non-file lines, and no other lane does: `Flag: <NAME>=<value> (<configure
   line or preset>)` for each option enabled, and `Build: none (<missing>)` when nothing built. Ledger and Coverage parsing ignore
   them (a row counts only when its second cell is a read depth); keep them free of `|`.
3. **Checked, no finding:** what was examined and found correct. A lane that owns an M-rule
   ([policy.md](policy.md) *M-rules*: `TESTQ` M1 and M3, `TESTB` M2, `AUTH` M4 and M5) puts
   one literal line per rule here: `M<n>: PASS|FAIL|WAIVED(<code>)|N/A|UNVERIFIED <evidence>`
   (with no PR, `N/A (no PR)` for M4 and M5).
   `ROCK` and `ROCK-f1` also write literal non-finding lines here. These lines contain no `|`, and Coverage and ledger parsing
   ignore them. `ROCK` writes:
   - `Timing-base: <entry> <family> <exposure> max-shard <s>s limit <s>s (<run ids>)`
   - `Flag-dep: <NAME> pin=<v> main=<v> (<FLAGS.cmake line>)`
   - `Win-path: <entry or file> <length> (<root source>)`

   `ROCK-f1` writes `Timing: <entry> <family> <exposure> base <s>s delta <s>s (<source>) bound <s>s/<N> shards limit <s>s`. The
   formats are defined in [lanes/therock.md](lanes/therock.md).
4. **Handoffs:** `<target lane>: <one line>` for issues outside the brief.
5. **Approval requests:** command, architecture, test selection, expected duration and
   where it runs, for anything needing a GPU, remote compute or another approval.

A lane may add **Needs human decision**: `<ID or ->: <question>. Options: <A>, <B>.
Recommendation: <A>, because <reason>.`

A lane whose brief does not apply returns only `N/A: <reason>`. A lane that cannot proceed
returns `STOP: <reason>` with evidence and at least one way forward. Any other report
conforms when it has the five sections in order, every finding uses the ID block and the
first line is present.

## Verifier output

`$EVIDENCE/r<N>/verify/<ID>.md`, or `<ID>-2.md` for a second verifier. `<ID>` is a final ID (`VER-<n>` for a verifier's own finding). It starts with this
block, one labelled line each; an approval request in the form above may follow:

```markdown
ID:        <ID>
Verdict:   CONFIRMED | PARTIAL | DOWNGRADED | UPGRADED | REFUTED | UNVERIFIED
Tier:      <final tier>
Claim:     <the narrowed claim, one sentence; PARTIAL only>
Reason:    <one line>
Proof:     <command, exit code, log as $EVIDENCE/<relative path>>
Class:     rerun | text-read | not-exercised   (at a new head)
Not established: <what the proof does not cover>
```

Keep mutation diffs and logs in
`verify/<ID>-evidence/`; evidence subdirectories are local-only (never shared) and the
verify file cites them by `$EVIDENCE/` path. Write defects found in the code under review while verifying to
`verify/VER<batch>.md` (*Lane report* format) as `VER<batch>-p<n>` candidates; `verify/VER-<n>.md` is the verdict file of the
final ID `VER-<n>`. Duplicate and cluster observations are Handoffs, never findings ([lanes/verify.md](lanes/verify.md)).
