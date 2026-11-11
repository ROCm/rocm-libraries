# Database: memories and know-how

Read this only if `$HARNESS_DB` is set, or when proposing an entry. The framework works without it.

The harness ships no database. Point `$HARNESS_DB` at your own.

Three kinds of knowledge, kept apart:

| Kind | Where | Who changes it |
| --- | --- | --- |
| Docs (what may be public) | the project `docs/` folder | humans decide what is exposed |
| Rules for agents | `AGENTS.md` files (for example the DAG scheduler rules) | humans |
| **Memories and know-how** | **the database** | **agents propose, humans review** |

This repo holds docs and rules. It holds no know-how and no lists. The database is a git repo outside
rocm-libraries, reached through `$HARNESS_DB`. It holds only what agents learn: memories and
know-how. Anything that identifies a specific machine, network, or internal system also stays out of
this repo.

```text
$HARNESS_DB/
  index.md       one line per entry. Read this first, then open only the entries you need.
  know-how/      known-bad patterns, tuning insights, measured findings, noise (SD) per benchmark
                 and machine. One file each: source, status, confirmed_by, last_verified.
  memories/      what past runs learned about the code, the workflow, and the machines
  config/        host names, ssh aliases, leak-patterns. Humans only; agents do not change it.
```

## Entry format

One Markdown file per entry, with frontmatter:

```markdown
---
name: <short-slug>
kind: pattern | insight | finding | threshold | memory
status: candidate | confirmed     # know-how only; memories have no status
source: <PR or issue number, experiment id, or person>
confirmed_by: <name>              # set by a human, together with status
last_verified: <date>
scope: <target or kernel family this applies to>
---
What it is, how to detect it (for patterns), why it matters, and the evidence.
```

`index.md` has one line per entry: `- [<title>](<path>) — <one-line hook> | <kind> | <status>`.
An entry with no `source` is not accepted in review. `last_verified` shows which entries are going
stale.

A know-how entry's `status` is `candidate` (a hit warns the developer) or `confirmed` (a hit rejects
the candidate before it costs GPU time). Only a human sets `confirmed`.

- **Read:** use the index and load entries on demand. Never load the whole database.
- **Write:** agents never edit the main branch. To add or change a memory or know-how entry, work on a
  branch and open a PR in the database repo with status `candidate`. A human reviews and merges it.
  This is the one outward action that does not need the developer's go-ahead each time.
- **Promote:** a human may turn a confirmed entry into a doc or a rule. An agent does not edit docs or
  rules to record what it learned.
- **Local database:** `$HARNESS_DB` may point at any directory with this layout, for personal use. The
  harness never creates one. Entries in a directory that is not the reviewed shared repo stay
  `candidate`. Only an entry that went through the shared repo's review can be `confirmed`, so a
  local database never makes the pattern check reject anything.
- **Sync:** engineers pull the database repo. One file per entry keeps merges conflict-free.
- **Not available or opted out:** the framework works without the database. It only loses the
  known-bad pattern check and the stored noise values, so measure the noise (SD) fresh from repeated
  runs, and record the pattern check as `skipped` in the result.
  - `$HARNESS_DB` not set or not readable: say so once, then continue without it.
  - `$HARNESS_DB=none`, or the developer says not to use it: skip it silently. Do not ask again, and
    do not propose entries.
- **Missing value:** if a value is needed and not set, ask the developer. Do not guess, and do not
  write it into a tracked file.

The leak-pattern list lives in `config/`, not here, because a list of confidential patterns is itself
a leak.
