<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Skill Prerequisites

Read this file FIRST before reading any other prerequisite files.

## Per-Skill Settings

Some skills require project-specific settings (JIRA keys, Confluence spaces, build
paths). These settings use a split system:

- **Schema**: Defined in each skill's SKILL.md — documents what settings are needed,
  how to auto-detect them, and examples. This is permanent reference documentation.
- **Values**: Stored in the monitor's `config.json` — a single file with all project
  settings as clean key-value tables, organized by section.

### Reading Settings

When a skill needs project-specific values:
1. Read the monitor's `config.json` (cache per conversation).
2. Find the skill's section (e.g., `## JIRA`, `## Confluence`, `## Build`).
3. If the section exists and has all required values → use them.
4. If values are missing → run the skill's setup flow (detect, confirm, verify, persist).
5. If `config.json` does not exist → create it and run setup.

### Codebase Knowledge

`config.json` also has a **Codebase Knowledge** section where skills record
project-specific patterns they discover during usage (e.g., where instance files
live, common directory conventions). Skills may read this section for context and
write new entries when they discover useful structural patterns about the codebase.

### Subagent Settings Handoff

When dispatching team members as subagents, the coordinator MUST read its settings
from `config.json`, resolve all values, and pass them as concrete values in
the subagent prompt. Team members should never read `config.json` or resolve
settings themselves — they receive actual values from the coordinator.

## Path Resolution

All paths in skill files are relative to the skill's base directory (given in the
system message as "Base directory for this skill: ..."). Shared resources live one
level up in `../shared/`. Project-level resources (team members, style guides) live
at the `.claude/` root.

**Examples:**
- `../shared/relationship_discovery.md` from `.claude/skills/code-review/` →
  `.claude/skills/shared/relationship_discovery.md`
- `../../team_members/dispatch_table.md` from `.claude/skills/code/` →
  `.claude/team_members/dispatch_table.md`

## Team Member Dispatch

The dispatch table at `.claude/team_members/dispatch_table.md` is the single source
of truth for resolving generic expert roles (e.g., "Code Expert", "GPU Expert") to
context-specific team members. **Every skill that dispatches experts MUST use it.**
Skills MUST NOT hardcode team member names or file paths. Always resolve through the
dispatch table algorithm. Skills request team members by their **generic role name**
(e.g., "Code Expert", "Wiki Expert"), never by their specific team member name
(e.g., "C++ Expert", "Confluence Expert"). The dispatch table resolves generic roles
to context-specific team members — this indirection is mandatory.

- Read the dispatch table once per conversation (caching rules apply).
- Use the Resolution Algorithm in the dispatch table to map generic roles to team
  member files based on detected context. A dispatch table the monitor's install (`rk provision`) made has no
  Resolution Algorithm: there a generic role resolves to the one row whose Generic Role names it.
- Team member files live at `.claude/team_members/<name>.md`.
- When a skill says "consult the dispatch table" or "resolve via the dispatch table",
  it means: use the Resolution Algorithm in `.claude/team_members/dispatch_table.md`.
- **Registry staleness check**: If a team member file exists in `.claude/team_members/`
  but has no corresponding row in the dispatch table registry, the registry is stale.
  Rebuild it before proceeding by following the rebuild procedure in CLAUDE.md
  (Registry Maintenance section). Similarly, if the dispatch table file is missing
  entirely, bootstrap it from the template in CLAUDE.md and then rebuild. A workspace with no
  CLAUDE.md (the monitor's, which `rk provision` installs into) gets both from that install, which
  puts the rows between the registry's sentinels.

## Reading Prerequisites

- This file (prerequisites.md) is always listed as prerequisite #1. Read it
  completely first.
- Then read all remaining prerequisite files in parallel using parallel Read
  tool calls in a single message.
- Do NOT proceed until all reads have completed successfully.

## Conversation-Level Caching

- **Do not re-read files you have already read in this conversation.** Shared
  documents (style guides, team members, this file, relationship discovery,
  temporary file policy) are stable within a session. If you read them for
  a previous skill invocation, skip them.
- When a skill lists prerequisites you have already read, go directly to the
  skill's workflow.
- **Subagents cannot cache.** Each subagent starts with a fresh context and
  must read its team member file and any documents referenced in its prompt.

## Fail-Fast

If ANY prerequisite file cannot be found or read, **STOP IMMEDIATELY.** Do NOT guess,
skip, or continue without it. Report which file(s) are missing and ask the user to fix
the paths before retrying.

## Subagent Rules

- When spawning subagents, use model `opus` by default. Never use haiku. A skill
  may override to `sonnet` when explicitly stated in its SKILL.md (e.g., the info
  skill uses `sonnet` for lightweight guided topics).
- Instruct every subagent to read and fully adopt their team member file
  (`../../team_members/<name>.md`) before proceeding. The subagent must NOT proceed
  until the team member file has been fully read.
- Launch all subagents in a single message with parallel tool calls. Never
  dispatch sequentially unless one subagent's output is needed to brief another.
- The coordinator MUST include full file contents (with line numbers) and
  the complete diff in every subagent prompt. Do NOT summarize, paraphrase,
  or describe the code — paste the actual content. Subagents need real code
  to cite specific lines in their findings.
- Instruct every subagent with the following file-reading rules:

  **DO NOT** re-read files, diffs, or commits already pasted in this prompt.
  The code below is your complete review context — trust it, cite it directly,
  and do not spend tool calls fetching it again.

  **DO** read files that were NOT provided in this prompt when your analysis
  requires it — for example, verifying that a header exists on a branch,
  checking callers, reading sibling implementations, or confirming codebase
  state. These are new reads, not redundant ones.

  The distinction: re-reading provided content wastes time and tokens;
  reading new files to verify facts is due diligence.
