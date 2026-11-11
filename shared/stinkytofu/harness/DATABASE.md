# Database: what agents learn

Read this only if `$HARNESS_DB` is set, or when proposing an entry. The framework works without it.

The harness ships no database. Point `$HARNESS_DB` at your own.

Three kinds of knowledge, kept apart:

| Kind | Where | Who changes it |
| --- | --- | --- |
| Docs (what may be public) | the project `docs/` folder | humans decide what is exposed |
| Rules for agents | `AGENTS.md` files (for example the DAG scheduler rules) | humans |
| **What agents learn** | **the database** | **agents propose, humans review** |

This repo holds docs and rules only. The database is a git repo outside rocm-libraries, reached
through `$HARNESS_DB`. It holds what agents learn. Anything that identifies a specific machine,
network, or internal system also stays out of this repo.

```text
$HARNESS_DB/
  index.md       one line per entry. Read this first, then open only the entries you need.
  entries/       patterns, findings, and measurements (including noise per benchmark and machine).
                 One file each: source, status, confirmed_by, last_verified.
  memories/      what past runs learned about the code, the workflow, and the machines
  config/        host names, ssh aliases, leak-patterns. Humans write it. Agents may read it, and
                 do not change it.
```

## Entry format

One Markdown file per entry, with frontmatter:

```markdown
---
name: <short-slug>
kind: pattern | insight | finding | threshold | procedure | memory
status: candidate | confirmed     # not used by memories
certainty: verified | measured | hypothesis   # for patterns, findings, insights
source: <page URL, repo file path, PR or issue number, experiment id, or person; may be empty for a memory>
source_revision: <the page's revision number when the entry was written, if the source is a page>
confirmed_by: <name>              # set by a human, together with status
last_verified: <date>
scope: <target or kernel family this applies to>
---
> **Claim:** <one sentence a person can agree or disagree with>
> **Certainty:** <same as the field>. **Evidence:** <what, where, when, how many runs>
> **Not checked:** <the biggest gap>
> **Next test:** <one line, for measured and hypothesis>
> **To confirm:** read this card. If the claim, the evidence, and "not checked" are right, set
> `status: confirmed` and `confirmed_by: <name>`.

<details: how to detect it, why it matters, what was ruled out, how to apply>
```

**The review card.** The first lines of the body are a short card, so a person can decide in under a
minute whether the entry is worth noting down. Everything after it is detail for whoever uses the entry.

**Certainty** says how sure the claim is. It is set by whoever writes the entry, and a person can change it.
- `verified`: a fix, an independent source (the hardware docs, the code), or a repeated test confirms it.
- `measured`: seen in our runs, but the cause is not proven.
- `hypothesis`: an explanation or a rule that has not been tested yet. Treat it as a place to start.

Certainty is not status. `status` says whether a person reviewed the entry. `certainty` says how sure the
claim is. A reviewed entry can still be a hypothesis.

`index.md` has one line per entry: `- [<title>](<path>) — <one-line hook> | <kind> | <certainty> | <status>`.
Use `-` for the certainty of a procedure or a memory.
**Entries from a page.** When you are about to use an entry that has `source_revision`, check the
page's revision number (the count of saved edits, not a release number) with whatever access you have.
`source` gives the page address. Use the site in that address, not your default site. The check
applies when you follow an entry. When you only mention a `candidate`, it is optional.
- Same revision: use the entry as it is and say nothing.
- Higher revision: read the page now and compare it with the entry's facts. List what is changed, new,
  or removed. Tell the developer, and recommend either "a fact in the entry changed, update it" or "the
  edits are in other sections, no change needed". If the developer agrees, propose the updated entry as
  a candidate with the new revision and today's date in `last_verified`. If not, keep the old entry.
- Cannot reach the page: say so once and use the entry as it is.

A source can be a page, a file in a repo, a PR or issue, an experiment, or a person. Only a page needs
`source_revision`. If the source is a file in this repo, point to it and read it when needed, and do
not copy it into the entry. A memory (what a run learned) may have no source. Any other entry with no
`source` is not accepted in review. `last_verified` shows which entries are going
stale.

An entry's `status` is `candidate` (a hit warns the developer) or `confirmed` (a hit rejects
the candidate before it costs GPU time). Only a human sets `confirmed`. A `confirmed` entry with a
person's name in `confirmed_by` counts wherever the database lives. If the database is a git repo, its
pull request is the review.

- **Read:** use the index and load entries on demand. Never load the whole database.
- **Write:** agents never edit the main branch. When a task taught you something that may be worth
  keeping, tell the developer and ask whether to note it down, not after every run. If they agree, work on
  a branch and open a PR in the database repo with status `candidate`. A human reviews and merges it. If
  the database is a plain directory, write the entry file with `status: candidate` and add its line to
  `index.md`.
  - If the developer tells you to remember or note something ("remember this", "note this down", "save
    this", "add this to the database"), write the entry now, the same way, and tell them the file name.
    They have already agreed, so do not ask again.
  - If they tell you not to ("don't save this", "no need to remember"), write nothing, and do not ask
    again this session.
  - Put a fact about the code, a kernel, tuning, or the workflow in `entries/` (with the review card) or
    `memories/`. A preference about how you should work with this developer belongs in your own memory,
    not here. If you cannot tell which, ask.
  - If `$HARNESS_DB` is not set, or is `none`, say so and write nothing here.
- **Refine:** an entry whose `scope` matches the target can add to a public guideline on the same
  point, for example a fuller version of a procedure. Follow the entry where it adds detail, and tell
  the developer you are using it. Only a `confirmed` entry is followed. A `candidate` entry is
  mentioned to the developer as a suggestion. If an entry contradicts the guideline instead of adding
  to it, ask the developer which to follow. An entry never loosens a public rule on safety or
  confidentiality.
- **Promote:** a human may turn a confirmed entry into a doc or a rule. An agent does not edit docs or
  rules to record what it learned.
- **Any directory works:** `$HARNESS_DB` may point at any directory with this layout. The harness never
  creates one.
- **Sync:** engineers pull the database repo. One file per entry keeps merges conflict-free.
- **Not available or opted out:** the framework works without the database. It only loses the
  pattern check and the stored noise values, so measure the noise (SD) fresh from repeated
  runs, and record the pattern check as `skipped` in the result. If the database was read but
  nothing matched, record `no entries`. That is a different outcome from `skipped`.
  - `$HARNESS_DB` not set or not readable: say so once, then continue without it.
  - `$HARNESS_DB=none`, or the developer says not to use it: skip it silently. Do not ask again, and
    do not propose entries.
- **Missing value:** if a value is needed and not set, ask the developer. Do not guess, and do not
  write it into a tracked file.

The leak-pattern list lives in `config/`, not here, because a list of confidential patterns is itself
a leak.
