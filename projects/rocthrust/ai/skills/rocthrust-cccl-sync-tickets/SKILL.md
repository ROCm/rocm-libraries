---
name: rocthrust-cccl-sync-tickets
description: Turns a rocthrust-cccl-sync-investigate report into a JIRA Epic/Story/Task work breakdown. Use when asked to file, plan, or create tickets for a CCCL-into-rocThrust sync.
---

# CCCL → rocThrust Sync (JIRA planning bridge)

This skill is the planning bridge between `rocthrust-cccl-sync-investigate`
and the code-side skills (`rocthrust-cccl-sync`,
`rocthrust-cccl-sync-resolve`, `rocthrust-cccl-sync-finalize`). It consumes
an investigation report's §5 ("JIRA work breakdown") and turns it into real
JIRA tickets. It depends on `atlassian-jira-create-ticket` for the actual
ticket-creation mechanics — read that skill first if you haven't already.

This skill is independent of the code-side pipeline: it can be run before,
during, or not at all. Nothing downstream requires tickets to exist.

## Ask the human for project/component/epic up front

**Never assume a fixed JIRA project or component.** Unlike RCCL, which
hardcodes `AICOMRCCL`/`RCCL_PROD`, rocThrust has no single project it
syncs are always filed under — the real historical CCCL 3.0 work
([PR #10464](https://github.com/ROCm/rocm-libraries/pull/10464)) cites
**both** `ROCM-29174` and `EXSWSTRHPC-300` in the same PR body, proving
there is no one fixed answer. Ask, and store the answers as:

- `$JIRA_PROJECT` — the project key (e.g. `ROCM`, `EXSWSTRHPC`, or whatever
  the human names).
- `$JIRA_COMPONENT` — component name, if the project uses one for
  rocThrust/CCCL work.
- `$JIRA_EPIC` — an existing Epic to file under, or "create a new one."

Also confirm the Jira base URL to use when reporting ticket links back
(don't assume `amd-hub.atlassian.net` matches RCCL's default without
checking).

## Ticket structure

- **Epic** — `CCCL <TO_TAG> Sync`. Top of the hierarchy for this sync.
- **One "Code Port" Story** (note the rename from RCCL's "Code Merge" —
  there is no merge here), tracked by the `rocthrust-cccl-sync` /
  `rocthrust-cccl-sync-resolve` pipeline. Description = the report's §5
  headline scope roll-up + NVIDIA-only exclusions.
- **One Task per line item** in the report's §5 "Per-feature disposition →
  line-item tasks" table. Consume that table close to verbatim — the
  investigate skill's report template was explicitly built
  forward-compatible for this, so it should already be in the right shape.
- **One container Task** for "Upstream bug fixes needing test coverage"
  from §5, listing each fix and what test should exercise it.

### Disposition → summary verb prefix

| Disposition (from report §5) | Task summary prefix |
|-------------------------------|----------------------|
| port-validate | `Validate / port …` |
| enable | `Enable / port …` |
| track-N-A | `Track … — …` |

### Standard line-item description shape

A short paragraph (what the feature is, why it's relevant to rocThrust, per
the report), followed by a fixed boilerplate checklist:

- [ ] Port/adapt via `rocthrust-cccl-sync-resolve`
- [ ] Add or port test/benchmark coverage (see report §3)
- [ ] Note disposition in the sync's CHANGELOG entry

### Field values

Use the human-provided values from above, never hardcoded IDs:

| Field | Value |
|-------|-------|
| `projectKey` | `$JIRA_PROJECT` |
| `type` | `Epic` / `Story` / `Task` per the structure above |
| `parentIssueId` | the Epic's key, for Stories/Tasks |
| `components` | `[{"name": "$JIRA_COMPONENT"}]`, if applicable |
| `priority` | ask the human — do not assume RCCL's `P2: Medium` default |

**Warning**: Jira custom-field IDs (Epic Link field, priority ID, component
IDs) are project-schema-specific. `atlassian-jira-create-ticket` documents
how to discover them for a project you haven't used before
(`createmeta` API, or cloning a reference ticket in the target project) —
do this the first time this skill is used against a new project. Do not
assume they match RCCL/AICOMRCCL's values.

## Workflow

1. Confirm the investigation report to use, and the Epic/project to file
   under (asking per the section above if not already known).
2. Build the full work-breakdown table (Epic → Story → Tasks) from the
   report's §5, following the structure above.
3. Gate on any of the report's own "Open questions for the human" — resolve
   them one at a time with the human before finalizing ticket content built
   from the affected areas.
4. Present the full table for approval before creating anything.
5. Create tickets in order (Epic, then Story, then Tasks) via
   `acli jira workitem create --from-json`, per
   `atlassian-jira-create-ticket`'s workflow.
6. Report back every created ticket's key and URL.

## What a good report must contain

For this skill to work without re-deriving information, the input
`cccl-investigation-<tag>.md` report must have a complete §5 (headline
scope roll-up, NVIDIA-only exclusions, per-feature disposition table,
upstream-bug-fixes-needing-coverage list) and a stated set of open
questions for the human. If §5 is missing or thin, go back to
`rocthrust-cccl-sync-investigate` rather than improvising ticket content
here.
