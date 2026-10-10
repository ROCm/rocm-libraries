# Claude Workspace

This workspace provides a reusable system of **skills**, **team members**, and **style guides** for Claude Code. Skills are slash commands (`/analyze`, `/build`, `/code`, etc.) that act as coordinators. They dispatch domain-specific **team members** as subagents to perform expert work. Style guides enforce project-specific coding standards.

## Directory Structure

```
.claude/
├── skills/                    # Slash command definitions (auto-discovered from */SKILL.md)
│   ├── <skill-name>/SKILL.md  # One directory per skill — skills are auto-discovered
│   └── shared/                # Shared policies read by all skills
│       ├── prerequisites.md         # Path resolution, dispatch rules, fail-fast
│       ├── relationship_discovery.md # Procedure for understanding code dependencies
│       └── temporary_file_policy.md  # Temp file creation and cleanup rules
├── team_members/              # Expert personas dispatched by skills
│   ├── dispatch_table.md      # Cached registry (auto-generated from team member frontmatter)
│   ├── cpp_expert.md          # Each file has YAML frontmatter: role, name, context, domain
│   ├── cpp_architect.md
│   ├── cpp_tester.md
│   ├── ... (one file per expert)
│   └── style_guide_expert.md
├── style_guides/              # Project-specific coding standards
│   ├── style_guide_registry.md # Cached registry (auto-generated from style guide frontmatter)
│   └── ck_cpp_style_guide.md   # Each file has YAML frontmatter: name, context
├── project_config.md          # Project-specific settings (values only)
└── settings.local.json        # Local permission overrides
```

## Where Things Go

The `.claude/` directory lives at the root of your project repository (alongside
`src/`, `CMakeLists.txt`, etc.). It contains all Claude Code workspace configuration
for the project.

| What you want to store | Where it goes | Commit? |
|------------------------|---------------|---------|
| A new slash command | `.claude/skills/<name>/SKILL.md` | Yes |
| A shared policy for all skills | `.claude/skills/shared/<name>.md` | Yes |
| A new domain expert | `.claude/team_members/<name>.md` | Yes |
| Your project's coding standards | `.claude/style_guides/<name>.md` | Yes |
| Project settings (JIRA keys, build paths, etc.) | `.claude/project_config.md` | No |
| Local permission overrides | `.claude/settings.local.json` | No |

**Version control**: Commit the `.claude/` directory so the entire team shares
skills, team members, and style guides. Exclude `project_config.md` and
`settings.local.json` — these contain machine-specific values and are generated
per-developer via each skill's guided setup flow (see [Per-Skill Settings](#per-skill-settings)).

### Codebase Knowledge

`project_config.md` also has a **Codebase Knowledge** section where skills record
project-specific structural patterns they discover during usage (e.g., where instance
files live, common directory conventions). Skills may read this section for context
and write new entries when they discover useful patterns. Because `project_config.md`
is not committed, Codebase Knowledge is local to each developer's environment.

---

## How It Works

1. The user invokes a skill: `/code fix the memory leak in foo.cpp`
2. The skill reads its prerequisites (`shared/prerequisites.md`, etc.)
3. The skill detects context from code, user request, and environment
4. The skill consults the **dispatch table** to resolve generic roles (e.g., "Code Expert") to context-specific team members (e.g., "C++ Expert")
5. The skill dispatches team members as **subagents** (using the Agent tool with model `opus`)
6. Each subagent reads and adopts its team member file before doing any work
7. The coordinator assembles subagent results into a final output

## Key Principles

- **Expert knowledge lives in team member files, not in skills.** Skills should dispatch team members as subagents when the task benefits from specialized review, analysis, or domain expertise. Skills may execute procedural work directly (running commands, assembling outputs) when dispatching a subagent would add overhead without improving quality.
- **Team member frontmatter is the source of truth** for role, context, and domain. The dispatch table registry is a generated cache — rebuilt from frontmatter when missing or stale. See [Registry Maintenance](#registry-maintenance).
- **Subagents use model `opus` by default.** Never haiku. A skill may override to `sonnet` when explicitly stated in its SKILL.md.
- **Fail-fast on missing files.** If a prerequisite or team member file can't be read, stop immediately and report it.
- **Never install packages on the bare machine.** Use venvs or containers.
- **Ask the user before creating files.** Follow the temporary file policy.
- **Tool boundaries matter.** Each skill owns specific tools. For example, the confluence skill uses the `atlassian` CLI for Confluence but dispatches the Project Management Expert for any JIRA data — it never accesses JIRA directly.

---

## Per-Skill Settings

Skills that need project-specific configuration (JIRA credentials, Confluence spaces,
build paths) use a split settings system:

- **Schema** (in each SKILL.md): The skill declares what settings it needs, with
  descriptions, auto-detection hints, and examples. This is permanent documentation
  that stays in the skill file.
- **Values** (in `.claude/project_config.md`): A single file where all project
  settings are stored as clean key-value pairs, organized by section (JIRA,
  Confluence, Build, etc.).

### How It Works

On first use, if a skill's settings are missing from `project_config.md`, the skill
runs a **guided setup**:

1. **Detect** — auto-detect what it can from the environment (MCP servers, CLI tools,
   filesystem paths) using hints from the Settings Schema
2. **Confirm** — present detected values to the user, ask them to confirm or adjust
3. **Verify** — test each setting works (MCP query, CLI command, path check)
4. **Persist** — save confirmed values to `project_config.md`

Users can always:
- **View/edit settings** by opening `.claude/project_config.md`
- **Re-run setup** for any skill with `/<skill> setup`

### Adding Settings to a New Skill

If your skill needs project-specific settings:

1. Add a `## Settings Schema` section to your SKILL.md with a table:
   `| Setting | Description | Auto-Detect | Example |`
2. Document which section name your skill uses in `project_config.md`
   (e.g., `## JIRA`, `## Confluence`)
3. Add a `### Operation: Setup` that implements the detect → confirm → verify →
   persist flow for your settings
4. At the start of your skill's workflow, read your section from
   `project_config.md`. If values are missing, trigger the setup flow.
5. When dispatching subagents, pass resolved settings as concrete values in the
   prompt — subagents should never read `project_config.md` directly.

---

## Adding a New Team Member

Team members are domain experts dispatched by skills as subagents. Each one lives in its own file at `.claude/team_members/`.

### Step 1: Create the team member file

Create `.claude/team_members/<name>.md`. The file **must** start with YAML frontmatter
that defines how the dispatch table finds this expert, followed by the team member body:

````markdown
---
role: Code Expert
name: Python Expert
context: Python
domain: type hints, virtual environments, import resolution, packaging, pytest, async/await, GIL, memory management, decorators, metaclasses
---

## Team Member: Python Expert (Correctness and Idiom)

**Role**:
- You are a [description of expertise and perspective].
- [Additional context about how this expert approaches problems.]

**Mandate**: [One-sentence summary of what this expert is responsible for.]

### What to Check

[Checklist or categories of things this expert reviews/does.]

### Output Format

```
## [Review/Analysis Type]

### [Category 1]
- [ ] **[file:line]** Finding description.
  **Impact**: What goes wrong if this isn't addressed.
  **Suggestion**: How to fix it.

### [Category 2]
...
```
````

#### Frontmatter fields

| Field | Required | Description |
|-------|----------|-------------|
| `role` | Yes | The generic role name skills use to request this expert (e.g., "Code Expert", "Debugger Expert", "Profiling Expert"). Use an existing generic role if your expert fills the same role in a different context. Create a new generic role only if no existing role fits. Check the current dispatch table registry to see existing roles. |
| `name` | Yes | Display name matching the `## Team Member:` heading (e.g., "Python Expert"). |
| `context` | Yes | When this expert applies. Must match a value from the Context Detection section of the dispatch table. Valid values include: `C++`, `C++/LLVM`, `HIP/AMD`, `Host`, `CMake`, `Git`, `JIRA`, `Confluence`, `Any`, `Python`, `Rust`, `JavaScript/TypeScript`, `Go`, `Java`, `C`, `Shell`. Use `Any` if the expert is language/context-agnostic. |
| `domain` | Yes | Comma-separated keywords that help disambiguate when multiple experts share the same generic role. These are matched (with fuzzy/semantic matching) against domain hints provided by skills. Be specific — describe what the expert *specifically* handles using concrete tool names, techniques, and failure modes. |

#### Writing good domain keywords

Domain keywords are how the dispatch algorithm tells apart multiple experts that share
the same generic role (e.g., three Profiling Experts). Good keywords are:

- **Specific tools and APIs**: `rocprofv3`, `gtest`, `ASan`, `Perfetto`, `Doxygen`
- **Concrete techniques**: `hardware counters`, `binary instrumentation`, `roofline analysis`
- **Failure modes and symptoms**: `crashes`, `hangs`, `memory leaks`, `linker errors`
- **Distinguishing concepts**: `interactive debugging` vs `passive crash triage`

Avoid vague keywords like "performance", "analysis", or "debugging" that don't
disambiguate — those are already captured by the generic role name.

#### Body consistency rules

- Start with `## Team Member:` heading
- Use `**Role**:` (not `**Background**:` or other variants)
- Include `**Mandate**:` line
- Include `### What to Check` section
- Include `### Output Format` section with a template
- Keep the file focused — one expert, one domain

### Step 2: Rebuild the dispatch table registry

After creating the file, the dispatch table registry must be regenerated so the new
expert appears. Follow the rebuild procedure in [Registry Maintenance](#registry-maintenance).

### Step 3: Update skills that should dispatch this expert

If your new team member fills an existing generic role (e.g., adding a Python Code Expert alongside the C++ Code Expert), existing skills will automatically resolve it via context detection — no skill changes needed.

If your team member introduces a **new generic role**, update the relevant skill(s) to dispatch that role. Look for dispatch sections in the skill's SKILL.md and add the new role where appropriate.

---

## Adding a New Style Guide

Style guides are enforced by the **Style Guide Expert** team member. They live in `.claude/style_guides/` and are matched by context.

### Step 1: Create the style guide file

Create `.claude/style_guides/<language_or_domain>_style_guide.md`:

```markdown
---
name: Display Name
context: ContextValue
---

# Project Style Guide Rules

[Rules organized by category: naming, formatting, includes, etc.]

## Cardinal Rules

[Rules that must NEVER be broken — list these first.]

## [Category]

[Specific rules with examples of correct and incorrect usage.]
```

**Important fields:**
- `context:` in the frontmatter must match a Context Detection value from the dispatch table (e.g., `C++`, `Python`, `CMake`). This is how the Style Guide Expert finds the right guide.
- `name:` is a human-readable display name.

### Step 2: Rebuild the style guide registry

After creating the file, the style guide registry must be regenerated so the new
guide appears. Follow the rebuild procedure in [Registry Maintenance](#registry-maintenance).

---

## Registry Maintenance

Both the **team member dispatch table** and the **style guide registry** are
auto-generated caches. The source of truth is the YAML frontmatter in each team
member and style guide file. The registries should be rebuilt whenever:

- The registry file does not exist (bootstrap from scratch)
- The registry exists but is empty (no content between sentinel comments)
- A new team member or style guide file is created
- A team member's or style guide's frontmatter is modified

### How to rebuild

**Team member registry** (`.claude/team_members/dispatch_table.md`):

1. Glob `.claude/team_members/*.md` (excluding `dispatch_table.md`)
2. Read the YAML frontmatter from each file (`role`, `name`, `context`, `domain`)
3. Build a markdown table row for each file: `| {role} | {name} | \`{filename}\` | {context} | {domain} |`
4. Sort rows by **Generic Role** so experts that share a role are grouped together
   (e.g., all Debugger Experts adjacent, all Profiling Experts adjacent)
5. Replace everything between the sentinel comments in `dispatch_table.md`:
   `<!-- BEGIN GENERATED REGISTRY -->` and `<!-- END GENERATED REGISTRY -->`
6. Context and Domain values must be copied **verbatim** from frontmatter — no rewording

**Style guide registry** (`.claude/style_guides/style_guide_registry.md`):

1. Glob `.claude/style_guides/*.md` (excluding `style_guide_registry.md`)
2. Read the YAML frontmatter from each file (`name`, `context`)
3. Build a markdown table row for each file: `| {name} | \`{filename}\` | {context} |`
4. Replace everything between the sentinel comments in `style_guide_registry.md`

### Bootstrap templates

If the registry file is missing entirely, create it from these templates before
running the rebuild procedure:

**`.claude/team_members/dispatch_table.md`:**

````markdown
# Team Member Dispatch Reference

## Context Detection

Skills detect context from three sources: the code/files, the user's request, and
the environment. **Multiple contexts can be active simultaneously** (e.g., a HIP
application produces both C++ and HIP/AMD context). Each registry row's Context is
checked independently — a row matches if its Context is present in any of the
detected contexts.

### Source 1: Code and Files

**Programming Language** (by file extension):
- `.cpp`, `.hpp`, `.h`, `.cc`, `.cxx` → C++
- `.py` → Python
- `.rs` → Rust
- `.js`, `.jsx`, `.ts`, `.tsx` → JavaScript/TypeScript
- `.go` → Go
- `.java` → Java
- `.c` → C
- `.sh`, `.bash` → Shell

**Compiler Toolchain** (by file content or extension):
- `.ll` (LLVM IR), `.mlir` (MLIR), `.td` (TableGen), `.bc` (LLVM bitcode) → C++/LLVM
- `compile_commands.json` containing `clang`, `amdclang`, or `hipcc` → C++/LLVM
- `CMakeLists.txt` setting `CMAKE_CXX_COMPILER` to `hipcc`, `amdclang++`, or `clang++` → C++/LLVM

**GPU Runtime** (by code content or build target):
- HIP API calls (`hip*`, `__global__`, device code) OR GPU targets (`gfx942`, `gfx90a`, `gfx908`, etc.) → HIP/AMD
- CUDA API calls (`cuda*`, `__global__`, device code) OR GPU targets (`sm_80`, `sm_86`, etc.) → NVIDIA GPU (CUDA)

**Execution Environment:**
- Code without GPU involvement (no HIP/CUDA detected) → Host
- HIP/CUDA codebases contain **both** host and device code. When a role has team
  members in both Host and HIP/AMD contexts, both match — the algorithm will ask
  the user which applies.

**Build System** (by file presence):
- `CMakeLists.txt` → CMake
- `Makefile` → Make
- `build.gradle` → Gradle
- `pyproject.toml`, `setup.py` → Python build

### Source 2: User's Request

The user's request may contain keywords that match a **Context** or **Domain** value
in the Team Member Registry. Treat any match as detected context. If the user
describes an activity without naming a specific tool and multiple team members
match, ASK which tool or system they are using.

### Source 3: Environment

Some contexts are always present based on the working environment:
- Inside a git repository → Git
- ROCm installed / AMD GPU present → HIP/AMD

---

**If context cannot be determined from any source**: ASK the user to clarify.

**If no matching team member exists**: STOP and tell the user. Ask if they want to
create a new team member for the missing area. Guide them through the creation process.

## Resolution Algorithm

Skills request team members using **generic role names** (the "Generic Role" column
below). The dispatch table resolves each generic role to a **context-specific team
member** based on detected context and domain matching.

**Dispatching multiple different generic roles is normal.** A single task often needs
several experts — e.g., Code Expert + GPU Expert + Compiler Expert. Dispatch them
all without asking. Each role is resolved independently through the algorithm below.

**Hard rule: Do not guess when multiple team members match the SAME generic role.**
If a single role resolves to more than one candidate and the domain does not clearly
identify one — e.g., which profiling tool, interactive or passive debugging, host or
device sanitizer — ASK the user to disambiguate before dispatching.

When a skill requests a generic role (with an optional domain hint):

1. Detect context using the Context Detection rules above.
2. Find all rows where **Generic Role** matches the requested role.
3. Filter to rows whose **Context** matches the detected context.
4. If a **domain hint** was provided, further filter to rows whose **Domain**
   keywords overlap with the hint. The domain hint is a short phrase describing
   what the skill needs (e.g., "hardware counters", "system-wide timeline",
   "interactive debugging", "crash triage"). Domain hint matching uses keyword
   and fuzzy matching with logical intuition — if the hint's intent aligns with
   a team member's domain keywords, it matches. Exact substring match is not
   required; semantic relevance is.
5. If exactly one match → dispatch that team member.
6. If multiple matches and the domain hint clearly identifies one → dispatch it.
7. If multiple matches and the domain is ambiguous or no hint was provided →
   ASK the user which team member to dispatch, showing the Domain descriptions
   to help them choose.
8. If zero context-specific matches but a row with Context "Any" exists →
   dispatch that team member.
9. If zero matches at all → STOP and tell the user. Ask if they want to create
   a new team member for this context.

### Domain Hint Examples

| Skill Request | Domain Hint | Result (HIP/AMD context) |
|---|---|---|
| Profiling Expert | "hardware counters" | rocProf Expert |
| Profiling Expert | "roofline analysis" | ROCm Compute Profiler Expert |
| Profiling Expert | "host-device timeline" | ROCm Systems Profiler Expert |
| Profiling Expert | (none) | Ask the user which profiling tool |
| Debugger Expert | "interactive debugging" | rocgdb Expert |
| Debugger Expert | "crash triage" | ROCr Debug Expert |
| Debugger Expert | (none) | Ask the user: interactive session or passive crash dump? |
| Sanitizer Expert | (any, Host context) | Host Sanitizer Expert |
| Sanitizer Expert | (any, HIP/AMD context) | GPU Sanitizer Expert |

## Team Member Registry

<!-- BEGIN GENERATED REGISTRY — do not edit manually -->
<!-- END GENERATED REGISTRY -->
````

**`.claude/style_guides/style_guide_registry.md`:**

````markdown
# Style Guide Registry

<!-- BEGIN GENERATED REGISTRY — do not edit manually -->
<!-- END GENERATED REGISTRY -->
````

After creating the file from the template, run the rebuild procedure above to
populate the registry from frontmatter.

---

## Adding a New Skill

Skills are slash commands defined by a `SKILL.md` file in `.claude/skills/<skill-name>/`.

### Step 1: Create the skill directory and file

Create `.claude/skills/<skill-name>/SKILL.md`:

```markdown
---
name: skill-name
description: One-line description of what this skill does and when to use it.
argument-hint: <required-arg> [optional-arg]
---

# Skill Name

You are a [coordinator role description].

## Prerequisites (Read First)

1. `../shared/prerequisites.md` — path resolution, fail-fast rules, and team member dispatch
2. `../shared/temporary_file_policy.md` — policy for managing temporary files

## [Workflow sections...]

## Output Format

[Define the standard output format for this skill.]
```

### Key expectations for skills

1. **List prerequisites.** Every skill should reference `../shared/prerequisites.md` as prerequisite #1. Add `../shared/temporary_file_policy.md` if the skill creates any files.

2. **Dispatch team members as subagents.** Skills should NOT do expert work themselves. They should:
   - Detect context from code/request/environment
   - Consult the dispatch table to resolve generic roles to team members
   - Dispatch subagents using the Agent tool with model `opus`
   - Assemble subagent results into the final output

3. **Respect tool boundaries.** Each skill should clearly document which tools it uses directly and which it delegates to subagents. For example:
   - The confluence skill uses the `atlassian` CLI directly but dispatches the Project Management Expert for JIRA data
   - The build skill uses container tools directly but dispatches the Build System Expert for CMake guidance

4. **Define an output format.** Every skill should produce structured, consistent output so users know what to expect.

5. **Follow the temporary file policy.** Ask the user where to put files before creating them. Track what you create and offer cleanup.

### Step 2: Skills are auto-discovered

Claude Code discovers skills by scanning `.claude/skills/*/SKILL.md`. Once the file exists, the skill appears as a slash command. No registration step is needed.

---

## Shared Policies

Files in `.claude/skills/shared/` are read by all skills as prerequisites. They
codify cross-cutting rules that every skill must follow.

### Existing policies

- **`prerequisites.md`** — Path resolution, dispatch table usage, fail-fast rules, subagent rules (always use `opus`, paste full code, don't re-read provided content)
- **`relationship_discovery.md`** — Procedure for discovering code relationships and dependencies
- **`temporary_file_policy.md`** — Rules for creating, tracking, and cleaning up temporary files. Includes session-level location (ask once, reuse), shared system etiquette, and stale file detection.

### Adding a new shared policy

Create a shared policy when a rule applies to **multiple skills** and would otherwise
be duplicated across their SKILL.md files. If a rule only applies to one skill, keep
it in that skill's SKILL.md instead.

1. Create `.claude/skills/shared/<policy-name>.md` with the policy content.
2. Add the policy to each relevant skill's `## Prerequisites (Read First)` section
   so skills know to read it before starting their workflow.
3. Update this section of CLAUDE.md to list the new policy with a brief description.
