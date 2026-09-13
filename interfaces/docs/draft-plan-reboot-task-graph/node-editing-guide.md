# Task graph node editing guide

This guide defines the repeatable review used for each delivery-graph node.

Terminology comes from [draft-plan-reboot.md](../draft-plan-reboot.md). Requirements come from the sibling `draft-plan-reboot-specs` directory.

## Required node structure

### Deliverable

- Use one sentence.
- Name the production result.
- State what exists when the task finishes.
- Exclude implementation steps, tests, schedules, and dependency prose.

### Implementation overview

- Use three to six bullets.
- Keep every bullet within 20 words.
- Name the component performing each action.
- State where new production logic lives.
- Describe the major control flow from inputs to outputs.
- Separate facade, broker, provider, adapter, and packaging responsibilities.
- Avoid detailed field lists when a task item can hold them.
- Avoid vague verbs unless the glossary defines their executed steps.

### Specification

- Link every governing specification by its short name.
- Use links relative to the generated graph.
- Display `TBD` when the exact requirement identifier is unsettled.
- Do not use machine-specific absolute paths.

### Task items

- Give implementation items stable IDs such as `i12-w01`.
- Give each item a component label showing where its logic lives.
- Attach clickable test-ID labels instead of repeating test descriptions.
- Keep each Description within 15 words.
- Begin each Description with a concrete implementation action.
- Keep each Outcome within 15 words.
- State the user, compatibility, safety, or diagnostic value in each Outcome.
- Do not write “make the test pass” inside the Description.

### Definition of done

- Keep every bullet within 20 words.
- State observable completion evidence.
- Reference test IDs when one test directly proves the condition.
- Avoid repeating the implementation overview.

### Pull request

- Give the PR a concrete title and brief result-oriented description.
- List only plausible files to add or modify.
- Keep test files under i08 unless later review explicitly changes ownership.

## Test ownership

- Define executable tests in i08.
- Assign every test a stable global ID such as `test-20`.
- Never renumber an ID after another document references it.
- Record the owning implementation task and initial expected state.
- Use expected failure only when the test reaches the intended unimplemented behavior.
- Keep test Descriptions and Outcomes within 15 words.
- Link each test to a specification and requirement identifier.
- Use `TBD` when the exact requirement identifier remains unsettled.
- Show test IDs as labels on associated implementation items.
- Keep full test descriptions in the Tests overlay, not implementation nodes.

## Terminology

- Use terms defined in the plan consistently.
- Add unclear architecture terms to the Definitions overlay.
- Define operational verbs by the actions they execute.
- Link only the first occurrence of each defined term within a node.
- Name the responsible component instead of using an implied subject.
- Distinguish production components from test fixtures.
- Distinguish public contexts from provider contexts.

## Editing workflow

1. Identify the production artifact and responsible component.
2. Rewrite the Deliverable as one result-oriented sentence.
3. Reduce the Implementation overview to the essential control flow.
4. Move detailed implementation actions into component-labeled task items.
5. Move executable test definitions into i08.
6. Assign or preserve stable test IDs.
7. Attach relevant test labels to each implementation item.
8. Add specification links and requirement identifiers.
9. Add missing glossary terms or operational definitions.
10. Remove duplicated scope, tests, and completion claims.
11. Check all word limits mechanically.
12. Rebuild the standalone HTML.

## Review questions

- What production component owns this work?
- Where will the logic live?
- What input does it receive?
- What output or retained state does it produce?
- Is the sentence implementation work, a test, or completion evidence?
- Which test proves the behavior?
- Which specification requires it?
- What regression or failure does the outcome expose?
- Does another item already cover the same behavior?
- Can a knowledgeable engineer understand the work without reopening the plan?

## Wording examples

Avoid:

> Apply the specification’s rules.

Prefer:

> The broker enforces trusted-path and module-loading rules from the linked specifications.

Avoid:

> Validate provider compatibility.

Prefer an implementation item:

> Implement broker checks for compatible interface versions and required provider-profile operations.

Then attach the relevant test ID and specification reference as labels.
