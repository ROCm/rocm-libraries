# Setup: load the harness at session start

Read this only when you set up a new machine or clone. Agents verifying a change never need it.

Claude Code loads a nested `CLAUDE.md` only after the agent reads a file in that folder. A session
started at the repo root therefore does not see the harness until it is already editing. A
`SessionStart` hook fixes that. It is optional, local to you, and not enforced: it puts the text in
front of the agent but does not make it comply.

## Setup

1. Add the hook to `~/.claude/settings.json` (merge with your existing `hooks`). Use the absolute path
   of your clone:

   ```json
   {
     "hooks": {
       "SessionStart": [
         { "hooks": [ { "type": "command",
                        "command": "<clone>/shared/stinkytofu/harness/tools/session-start.sh" } ] }
       ]
     }
   }
   ```

2. If you have access to the harness database, set `HARNESS_DB` to your clone of it in your shell
   profile. Set it to `none` to opt out. Unset, the script skips the database.

## What it does

- Prints the two `HARNESS.md` routers and, if set, `$HARNESS_DB/index.md`. About 1.5k tokens.
- Prints nothing outside this clone (worktrees of it count), so other projects are unaffected.

## Check

Start a fresh session in the repo root and ask the agent which harness rules it was given. `/hooks`
lists the configured hook.

## Limits

- The hook runs code from the repo. Read the script before you point your settings at it, and again
  after a pull that changes it.
- Do not put the hook in the repo's shared `.claude/settings.json`. It would run for everyone,
  including people without database access.
