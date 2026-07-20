---
name: rfc-backlog
description: Turn a technical RFC into an actionable backlog of JIRA stories and tasks, grounded in the codebase the RFC affects. Derives unified, independently-shippable work items (each with a user-story sentence, acceptance criteria, functional and non-functional requirements), presents them for review, then — unless run as a dry run — collects component/epic/label defaults and creates the tickets in JIRA. Use when you have an approved or near-approved RFC and need to plan the implementation as trackable work.
argument-hint: "[<rfc-path>] [--dry-run]"
allowed-tools: Read, Write, Edit, Grep, Glob, Bash, WebFetch, mcp__jira-atlassian-hub-local__jira_get_all_projects, mcp__jira-atlassian-hub-local__jira_get_project_components, mcp__jira-atlassian-hub-local__jira_search, mcp__jira-atlassian-hub-local__jira_create_issue, mcp__jira-atlassian-hub-local__jira_link_to_epic, mcp__jira-atlassian-hub-local__jira_get_issue
---

# RFC → Backlog Skill

Convert a technical RFC into a backlog of JIRA work items. The aim is a backlog an engineer could pick up and start on: each item is a unified, independently-compilable, value-delivering piece of work — not a generic "implement the RFC" placeholder.

This skill is the planning bridge between `/rfc-review` (judge the design) and implementation (do the work). It reads the RFC, grounds the work in the actual codebase, and — when not a dry run — writes the tickets.

## Usage

```
/rfc-backlog                              # Find the RFC on the current branch, derive a backlog, create tickets
/rfc-backlog docs/rfcs/my-rfc.md          # Use a specific RFC file
/rfc-backlog --dry-run                    # Derive and display the backlog only — never touches JIRA
/rfc-backlog docs/rfcs/my-rfc.md --dry-run
```

## When to invoke this skill

- The user asks to break an RFC down into stories/tasks, a backlog, a work plan, or JIRA tickets.
- An RFC is approved (or close) and the user wants to plan implementation as trackable work.

Do NOT invoke for reviewing an RFC's design — use `/rfc-review` for that. This skill assumes the design is settled enough to plan against; if the RFC is still contentious, suggest a review first.

## Issue classification rule

Apply this rule to every derived item — it decides the JIRA issue type:

- **Story** — the work results in a pull request (code, build, docs-as-code, tests). It compiles and delivers value, even if only scaffolding for future work.
- **Task** — the work produces no PR: investigation/spikes, "meet with team X", decisions, coordination, manual ops.

## Workflow (summary)

The detailed, authoritative steps live in `prompt.md`. In brief:

1. **Locate & ingest the RFC** — argument path, else branch/PR context, else ask.
2. **Map the affected codebase** — resolve the directories, libraries, and build targets the RFC names so requirements are concrete.
3. **Derive the backlog** — decompose into unified items; classify each as Story or Task; order by dependency.
4. **Present** the backlog inline for review (template in `assets/templates/backlog-item.md`).
5. **Revise** — incorporate the user's changes; iterate until they approve the list.
6. **Dry-run gate** — if `--dry-run`, stop here. No prompts, no JIRA writes.
7. **Gather metadata** — load `config.json` defaults, confirm/collect component, epic, labels, project key; save back to `config.json`.
8. **Create tickets** — confirm the count, then create each issue in JIRA, link to the epic, and report the created keys/URLs.

## Output rules

- **Every item is shippable.** If an item can't compile or deliver standalone value, split it or fold it into another. State the dependency order explicitly.
- **Ground requirements in the codebase.** Functional requirements name real targets ("produces `libhipdnn_xxx.so`", "adds `hipdnnFooBar()` to the public header") — not vague restatements of the RFC.
- **Stories are fully specified.** Each Story has the user-story sentence, acceptance criteria, functional requirements, and non-functional requirements. Do not pad — a Story with no meaningful non-functional requirement says "None" rather than inventing one.
- **Never write to JIRA without two gates:** explicit approval of the list (Step 5) and explicit confirmation of metadata (Step 7). `--dry-run` short-circuits before either.
- **Cite the RFC.** Each item references the RFC section it derives from, so the reader can trace it back.
