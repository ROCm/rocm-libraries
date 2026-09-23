# Backlog item templates

Use these structures when presenting derived items to the user (Step 4) and as the
basis for the JIRA description (see `jira-description.md`). Drop fields that don't
apply rather than padding them.

---

## Story template

### <N>. [Story] <imperative, specific title>

**User story:** As a <user | maintainer | developer | integrator>, I want <goal> so that <reason>.

**Acceptance Criteria**
- <observable, checkable condition>
- <...>

**Functional Requirements**
- <concrete, codebase-grounded — e.g. "produces `libhipdnn_xxx.so`">
- <e.g. "exposes `hipdnnFooBar()` in `include/hipdnn/...`">

**Non-functional Requirements**
- <performance target, maintainability, compatibility, ...>
- <or "None">

**RFC reference:** <section / heading this derives from>
**Depends on:** <item number(s) that must land first, or "Nothing">

---

## Task template

### <N>. [Task] <imperative, specific title>

**Description:** <what the work is and why it's needed — no PR results from it>

**Definition of done**
- <checkable outcome — e.g. "decision recorded in the RFC", "owner X agrees on the perf target">
- <...>

**RFC reference:** <section / heading this derives from>
**Depends on:** <item number(s) that must land first, or "Nothing">
