# RFC → Backlog — Detailed Workflow

You are turning a technical RFC into a backlog of JIRA work items. Your job is to produce the backlog a thoughtful tech lead would: each item is a real, unified piece of work grounded in the actual codebase, not a generic decomposition. The user reviews and edits the list before anything is written to JIRA.

## Operating principles

- **Read the RFC end-to-end before decomposing.** The right slicing of work is rarely obvious from the intro.
- **Ground every item in the code.** Before claiming "add library X" or "modify component Y", confirm with Glob/Grep that X/Y exist (or that the path the RFC names is where the new thing goes).
- **Each item must stand alone.** When its PR merges, the tree still compiles and the item delivers value — even if that value is only scaffolding for later work. If you can't describe the value, the slice is wrong.
- **Order by dependency, not by RFC section order.** Foundational work (interfaces, build targets, stubs) comes before the features built on it.
- **Never touch JIRA before both gates pass:** the user approves the list (Step 5) and confirms the metadata (Step 7). A `--dry-run` invocation must never reach those steps.

---

## Step 1 — Locate and ingest the RFC

In order of preference:
1. Argument path, if provided.
2. PR/branch context: run `git log -1 --name-only` and look for added/modified `.md` files under `docs/`, `rfcs/`, `design/`, or paths matching `*rfc*`.
3. If still ambiguous, ask the user for the path. Do not guess across multiple candidates.

Read the full RFC. Note its title, status, the problem it solves, the goals/non-goals, and the design it proposes. Follow links to referenced specs or prior RFCs where they affect scope.

## Step 2 — Map the affected codebase

The RFC names things; verify they are real so the backlog is concrete.

- For every component, directory, library, header, or build target the RFC mentions, confirm it with Glob/Grep. Record the actual path.
- Identify where *new* artifacts would live (which `CMakeLists.txt`, which provider/plugin dir, which public header) by analogy to existing ones.
- Note the build/test entry points relevant to the work (so acceptance criteria can reference "builds under preset X" / "passes suite Y").

This step is what separates a useful backlog from a paraphrase of the RFC. Spend real effort here.

## Step 3 — Derive the backlog

Decompose the RFC into a set of work items. For each item, apply the classification rule:

- **Story** — results in a pull request. It compiles and delivers value (functionality or scaffolding for future work).
- **Task** — produces no PR: spikes/investigation, "meet with team X", decisions, coordination, manual operations.

Good slicing heuristics:
- One coherent capability or layer per Story (e.g. "build skeleton + empty library target", "implement op dispatch", "wire op into public API", "add tests").
- A Story that introduces an interface and a Story that implements it can be separate if each independently compiles and the first has value (callers can build against the stub).
- Prefer 4–12 items for a typical RFC. If you have 30, you're over-slicing; if you have 2, you're under-slicing.
- Surface non-code work (design sign-off meetings, dependency owner conversations, perf-target agreement) as **Tasks** rather than dropping it.

### Required content per item

**Story** (use `assets/templates/backlog-item.md`):
- **Title** — imperative, specific.
- **User story** — `As a [user | maintainer | developer | integrator], I want [goal] so that [reason].`
- **Acceptance Criteria** — bullet list; observable, checkable conditions for "done".
- **Functional Requirements** — bullet list; concrete, codebase-grounded (e.g. "produces `libhipdnn_xxx.so`", "exposes `hipdnnFooBar()` in `include/...`").
- **Non-functional Requirements** — bullet list; performance targets, maintainability, compatibility, etc. Write "None" if genuinely none — do not invent.
- **RFC reference** — the section(s) this derives from.
- **Depends on** — other items in this backlog that must land first (by title or number).

**Task**:
- **Title**, **Description** (what and why), **Definition of done** (bullet list), **RFC reference**, **Depends on**.

## Step 4 — Present the backlog

Print the full backlog inline, in dependency order, numbered. Use the template structure. Lead with a one-line summary: how many Stories, how many Tasks, and the suggested implementation order.

Keep it scannable — the user is going to edit it.

## Step 5 — Revise until approved

Ask the user for changes: add/remove/split/merge items, adjust acceptance criteria, re-order, re-classify Story↔Task. Apply edits and re-display the affected items (or the whole list if it changed substantially).

Iterate until the user explicitly approves (e.g. "looks good", "ship it", "create them"). Do not proceed on silence or on a question.

## Step 6 — Dry-run gate

If the invocation included `--dry-run`:
- Stop here. Do **not** prompt for metadata and do **not** call any JIRA tool.
- Tell the user this was a dry run and how to create the tickets for real (re-run without `--dry-run`).
- Optionally offer to save the backlog to a file next to the RFC if the user wants a durable artifact.

Otherwise continue.

## Step 7 — Gather JIRA metadata

1. **Load defaults.** Read `config.json` in the skill directory if it exists (it is gitignored; `config.example.json` shows the shape). Fields: `projectKey`, `defaultComponent`, `defaultEpic`, `defaultLabels`.
2. **First run / missing config.** If `config.json` is absent or a field is blank:
   - If `projectKey` is unknown, ask for it (optionally list candidates via `jira_get_all_projects`).
   - Offer the project's real components via `jira_get_project_components`, and find candidate epics via `jira_search` (e.g. `project = <KEY> AND issuetype = Epic AND statusCategory != Done`).
3. **Confirm with the user** by asking directly (state the defaults pre-filled from config so they can accept or override):
   - **Component** — which component the work belongs to.
   - **Epic** — which epic to link the items under (epic key).
   - **Labels** — labels to apply to every item.
   - **Project key** — only if not already set.
4. **Save** the confirmed answers back to `config.json` so they default next time. Preserve any fields the user didn't change.

## Step 8 — Create the tickets

1. **Confirm the write.** State exactly what will happen: "Creating N issues (X Stories, Y Tasks) in project `<KEY>`, component `<C>`, under epic `<EPIC>`, labels `[...]`." Proceed only on a clear go-ahead.
2. **Create each item** with `jira_create_issue`:
   - `project_key` = the project key.
   - `issue_type` = `Story` or `Task` per the classification.
   - `summary` = the item title.
   - `description` = the item body rendered from `assets/templates/jira-description.md` (Markdown; the tool accepts Markdown).
   - `components` = the chosen component.
   - `additional_fields` = labels, and the epic link (`{"epic_link": "<EPIC>"}` or `{"labels": [...], "epic_link": "<EPIC>"}`). If the epic link is rejected by the project's configuration, fall back to `jira_link_to_epic` after creation.
   - Create dependency-earlier items first where it's cheap to do so.
3. **Report.** Print a table of created issues: number, type, title, key, and URL. Note any failures explicitly with the error — never report a ticket as created if the call failed.

## Failure handling

- **MCP/JIRA unavailable.** If the JIRA tools error or the server isn't reachable, stop before Step 7's writes, report it, and offer to save the approved backlog to disk so no work is lost.
- **Epic or component not found.** Surface the mismatch and re-prompt rather than guessing a near-match.
- **Partial creation.** If creation fails midway, report which items were created (with keys) and which weren't, so a re-run can finish the rest without duplicating.
