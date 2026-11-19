# State files

Read this when `progress.md` exists, or when you write results.

| File | Format | Written by |
| --- | --- | --- |
| the harness files | Markdown | humans |
| `results.json` (gitignored) | JSON, one entry per config; agents change only result fields | agents |
| `progress.md` (gitignored) | Markdown, append-only handoff log | agents |
| run logs (gitignored) | text, one file per run | scripts |

`results.json` and `progress.md` can contain confidential or private information. Never commit them
to this repo, and never paste them into a public PR, issue, or comment.

**`progress.md`** is how one session hands over to the next. A new session starts with no memory of the
last one, so it reads `progress.md` and `git log` to see where the work stands.
- Append one entry at the end of each step or session. Never rewrite earlier entries.
- Use this layout:
  ```text
  ## <date> <task>
  done: ...   next: ...   open questions: ...   baseline: <commit hash>
  ```
- It records the task in flight, so it is short-lived. What a task taught you that a later task could
  use goes to the database, after you ask the developer (see `DATABASE.md`).

Print little to the terminal and write full detail to the log file. Format an error as one line,
`ERROR: <reason>`, so it can be grepped.
