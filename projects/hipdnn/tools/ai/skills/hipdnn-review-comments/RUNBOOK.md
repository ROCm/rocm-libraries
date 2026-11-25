# hipDNN Review Comments runbook

You are the orchestrator ([SKILL.md](SKILL.md) *Entry contract* lists the inputs). Names used below:
`<owner>/<repo>` and `<pr>` come from the PR; `$HEAD` and `$BASE` are its head and base SHAs;
`$OUT` is the run directory; `$EVIDENCE` and `<round>` are the deep-review evidence directory and
round in ledger mode.

```bash
RC="$SKILL/scripts/review_comments.py"
gh api repos/<owner>/<repo>/pulls/<pr> --jq '{head: .head.sha, base: .base.sha}'
```

**Head check:** re-run that command and compare `head`. If it moved, stop: start again from stage 2
in a new run directory with this run as input. Never present or post comments checked against an
older head.

## Formats

All files live in `$OUT`; [scripts/review_comments.py](scripts/review_comments.py) `check` enforces them.

| File | Format |
|---|---|
| `FINDINGS.md` | A free preamble (head, sources, decisions that bind every stage), then one `## <ID>. <claim>` per item. An ID is `F<n>` or `<LANE>-<n>`. Free text may follow the heading. Labelled lines: `Tier:` (`BLOCKING`, `IMPORTANT`, `SUGGESTION` or `FUTURE WORK`; optional), `Evidence:` (required, except in ledger mode when the ledger cell is empty), `Decision:` (optional). Any other `## ` heading is an error |
| `TRIAGE.md` | A table `ID \| Anchor (or -) \| Action \| Priority \| Reason`. Action: `KEEP`, `MERGE INTO <ID>`, `REVISE`, `UNANCHORED` or `DROP`. Priority: `P1` (must land before merge), `P2` (high value, low effort) or `P3` (not worth a comment now: listed for the user, never drafted). The outer pipes are optional; copy this header row: `\| ID \| Anchor (or -) \| Action \| Priority \| Reason \|` |
| `COMMENTS.md` | Lines `Head:` and `Base:` (7-40 hex digits), then `### <n>. <P1\|P2> <path>:<line>[-<line>] (<IDs>)` and the exact body per comment, then an optional `## Review body`. A comment's IDs are its `KEEP` or `REVISE` ID plus every ID that TRIAGE merges into it |
| `comments.json` | One object per comment, in order: `commit_id`, `path`, `line`, `side` (`RIGHT`), `body`, `finding_ids` (the heading's IDs, stripped before posting); a range adds `start_line` and `start_side` (`RIGHT`) |
| `POSTED.urls` | One line per posted item: `<n> <html_url>` for inline comment `n`, `review <html_url>` for the review body |
| `POSTED.md`, `posted.tsv` | Written by `posted`. `POSTED.md` lists every posted comment and the review body. In ledger mode `posted.tsv` lists each comment's web URL with its ledger IDs, merged ones included, in [hipdnn-deep-review](../hipdnn-deep-review/report.md) *posted.tsv* format; comments with no ledger ID and the review body stay out |

Ledger mode also keeps `ledger.json` (the evidence directory, round and universe) in `$OUT`.

## Agent briefs

Every stage 2-4 prompt names the PR, `$HEAD`, `$BASE`, the code location, `$OUT` and `EXISTING.md`, and carries:

```
Scope: read-only on source; write only inside $OUT. Build only when a fact check needs it,
within the host's build limits. Return a short summary and file paths.
STOP: if evidence contradicts a decision in FINDINGS.md, or a claim cannot be checked at the
head, return "STOP: <what> - <evidence> - <alternative>" instead of deciding yourself.
```

Stages 3 and 4 also carry this standard:

```
Comment standard - every PR comment and the review body:
- Human-readable, high-level, concise: 2-4 sentences. A reader who has not seen our review understands it.
- Lead with the problem and its consequence, then one concrete ask. Add suggested wording or a tiny code suggestion only when it is shorter than describing it.
- Prefix non-blocking items with "Optional:" or "Nit:". Blocking items have no prefix.
- Polite, direct, collegial. Ask a question when the intent is genuinely unclear.
- Never mention finding IDs, lanes, ledgers, rounds, agents, evidence paths or people not on the PR. Code quoted in backticks is exempt.
- No restating the diff, no code dumps, no stacked hedges.
- Every factual claim (lines, behavior, callers, test coverage) is true at the head.
```

A `STOP:` return names what stopped, the evidence and an alternative. Halt the pipeline, take it to the user, and re-run the stopped stage in a
fresh agent with the answer as a closed input.

## 1. Compile (orchestrator)

1. Fetch `$HEAD` and `$BASE` and write `$OUT/FINDINGS.md`.
   - **Ledger mode:** one item per universe ID, with its tier and `Evidence:` from the ledger. The command prints `out:` when `$OUT` was not given:

     ```bash
     python3 "$RC" findings "$EVIDENCE" <round> --head "$HEAD" [--out "$OUT"]
     ```

     It runs `$SKILL/../hipdnn-deep-review/scripts/comments.py universe` and stops with an install hint when that skill is absent. Add the user's decisions to the preamble or as `Decision:` lines; add items the discussion raised as `F<n>`.
   - **Free-form:** copy the given `FINDINGS.md`, or write one from the discussion, the research and the user's decisions: one item per issue, each with its claim and `Evidence:` (`path:line` at the head, plus the source).
2. Write `$OUT/EXISTING.md`: what is already said about the PR.

   ```bash
   gh api --paginate repos/<owner>/<repo>/pulls/<pr>/files > "$OUT/files.json"
   gh api graphql --paginate -f owner=<owner> -f repo=<repo> -F pr=<pr> -f query='
   query($owner:String!,$repo:String!,$pr:Int!,$endCursor:String){repository(owner:$owner,name:$repo){pullRequest(number:$pr){reviewThreads(first:50,after:$endCursor){pageInfo{hasNextPage endCursor} nodes{isResolved isOutdated path line comments(first:1){nodes{url author{login} body}}}}}}}'
   gh api --paginate repos/<owner>/<repo>/pulls/<pr>/reviews --jq 'map({user: .user.login, state, body})'
   gh api --paginate repos/<owner>/<repo>/issues/<pr>/comments --jq 'map({user: .user.login, url: .html_url, body})'
   ```

   Keep thread URL, author, location, resolved state and text; add the `COMMENTS.md` and `POSTED.md` of earlier runs beside `$OUT`. The first command saves the file patches that `check` reads when there is no `checkout:`.
3. With a `checkout:`, confirm it holds `$BASE` and `$HEAD`; otherwise agents read code through `gh api` at `$HEAD`.

## 2. Curate (fresh agent)

Input: `FINDINGS.md`, `EXISTING.md`. Write `$OUT/TRIAGE.md`.

1. Fact-check each item at the head: true, still present, not fixed. Read the code; do not trust the finding's wording.
2. Judge usefulness apart from severity: what breaks if it is not fixed and how much work the fix is. Assign `P1`, `P2` or `P3`.
3. Give every item one row with its own reason: the check that decided it, with evidence. Never write `nit`, an empty reason or a copied one.
4. Choose the action:
   - `KEEP`, or `REVISE` when the claim needs correcting (the reason states the corrected claim). The Anchor is `path:line`, an added or context line of the diff at the head.
   - `MERGE INTO <ID>` for items that share a root cause and one fix; the reason names the target, which is itself `KEEP` or `REVISE`.
   - `UNANCHORED` when the issue is real but no diff line causes or exposes it. It is surfaced to the user, not drafted, and not a drop.
   - `DROP` for items that are false, fixed, already raised or low value. List an item already raised in another reviewer's open thread by that thread's URL in its reason.
5. A user decision in `FINDINGS.md` binds. `DROP` of a `BLOCKING` or `IMPORTANT` item needs a `Decision:` line on the item; without one, return `STOP:` with the evidence.
6. Put issues outside `FINDINGS.md` under a *Not in scope - for the user* section after the table, never in it.

## 3. Draft (fresh agent)

Input: `TRIAGE.md`, `FINDINGS.md`, `EXISTING.md`. Write `$OUT/COMMENTS.md` and `$OUT/comments.json`.

1. One comment per `KEEP` or `REVISE` row at `P1` or `P2` (never `P3`; those are presented in stage 5), `P1` first, covering the rows merged into it; its heading lists every ID. Follow the comment standard.
2. Anchor on the row's `path:line` or a range inside one hunk. When the issue sits outside the diff, anchor on the diff line that causes or exposes it and open with "Outside the diff (`path:line`):".
3. Write a `## Review body` only when `FINDINGS.md` says one is wanted and which items it covers (for example `UNANCHORED` ones); never put an ID outside backticks in it.
4. Run `python3 "$RC" check "$OUT" [--head <head>] [--checkout <dir> | --files "$OUT/files.json"] [--terms <file>] [--evidence "$EVIDENCE"]` and fix what it prints. It never replaces stage 4.

## 4. Review (fresh agent)

Input: everything in `$OUT`. It does not edit drafts; it writes `$OUT/COMMENT-REVIEW.md` with a verdict per comment and for the review body:

- `PASS`;
- `FIX` with the exact replacement body or anchor and the reason;
- `REMOVE` with the evidence (a false or unverifiable claim).

Per comment it verifies that every claim holds at the head, that the anchor is the line the comment is about, that the comment meets the standard and its ask fits the priority. It also re-reads `TRIAGE.md`: reasons are true, drops are justified, merges share a fix. `REMOVE` of a `BLOCKING` or `IMPORTANT` item is a `STOP:` for the user. It runs `check` as in stage 3.

**Dry run:** with `dry-run:<head>`, run `check` against that head and stop. Report its result, the TRIAGE actions and the comment IDs; present and post nothing.

## 5. Present (orchestrator)

1. Apply every `FIX` and `REMOVE` to `COMMENTS.md` and `comments.json` together; renumber. A removed item's TRIAGE row becomes `DROP` (or `UNANCHORED`) with the review's evidence as its reason. Verify each `FIX`ed comment against the code yourself, plus one claim per priority present.
2. Run `python3 "$RC" check "$OUT" --head "$HEAD"` with `checkout:`/`--files`, `terms:` and `--evidence` as before, where `$HEAD` is the live head: it fails when the head moved.
3. Present, in order: the head; each comment with number, priority, `path:line`, the exact body and a one-line verification note; the review body when present; then dropped, `P3`, removed, `UNANCHORED` and not-in-scope items, one line each with the reason.
4. Discuss. Apply the user's edits to both files, run `check` again and re-present each changed comment verbatim. A later override of a `DROP` becomes `KEEP` in TRIAGE with a `Decision:` line on the item; re-run stages 3-4 in fresh agents for it. A review body the user asks for is likewise a `Decision:`, followed by stages 3-4.

## 6. Post (orchestrator, on explicit approval)

Posting is a GitHub write; never reply to a thread. Skip this stage for a dry run.

1. Ask the user to approve: the PR, the head, and every inline comment's `path:line` and exact body, plus the review body. Any later edit needs a new approval.
2. Re-run the head check, then `check --head "$HEAD"` as in stage 5.
3. Post each inline comment directly, one second apart, with `finding_ids` stripped, resuming from `POSTED.urls` and stopping at the first failure:

   ```bash
   set -o pipefail
   N=$(python3 "$RC" payload "$OUT/comments.json" --count)
   DONE=$(grep -c '^[0-9]' "$OUT/POSTED.urls" 2>/dev/null); DONE=${DONE:-0}
   for i in $(seq $((DONE+1)) "$N"); do
     url=$(python3 "$RC" payload "$OUT/comments.json" "$i" \
       | gh api repos/<owner>/<repo>/pulls/<pr>/comments --method POST --input - --jq .html_url) \
       || { echo "FAILED at comment $i of $N"; break; }
     echo "$i $url" >> "$OUT/POSTED.urls"
     sleep 1
   done
   ```

   After a failure, report the error and the comments already posted. Resume only on the user's word, after checking the PR for an anchor that posted although its call failed; add its line to `POSTED.urls` by hand.
4. When the review body was approved, post it once, as a `COMMENT` review without inline comments:

   ```bash
   url=$(python3 "$RC" review-payload "$OUT" --head "$HEAD" \
     | gh api repos/<owner>/<repo>/pulls/<pr>/reviews --method POST --input - --jq .html_url) \
     && echo "review $url" >> "$OUT/POSTED.urls"
   ```

5. Record the result and print the follow-up command:

   ```bash
   python3 "$RC" posted "$OUT"
   ```

   It writes `POSTED.md` and, in ledger mode, `posted.tsv`, then prints `ledger.py link <evidence> <round> --from <out>/posted.tsv`. Run it only if the user wants the comments linked now; the next deep-review round also finds `posted.tsv` in `$OUT` and links it.
