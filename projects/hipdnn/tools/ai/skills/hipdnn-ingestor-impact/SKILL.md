---
name: hipdnn-ingestor-impact
description: "Check whether a rocm-libraries change (PR, branch, commit range or uncommitted work) breaks the hipDNN kernel ingestor at run time while the default build stays green: kernel argument lists (ABI) out of step with their C++ and test copies, also against the target branch's tip, dispatch and launch-surface edits, descriptor and generic-target changes, removed Python symbols still read, which shipped rocKE variants' emitted code changed, support predicates that accept specs their builders reject, and C++ restatements whose values or refusals differ from Python's. Reads git objects; the optional emitter runs rocKE on the CPU. Ends in a Runtime impact: BREAKS | NEEDS-DEVICE | NONE | N/A line. For a full multi-agent review, use hipdnn-deep-review, whose IMPACT lane runs the same scripts."
argument-hint: "[<PR number> | branch:<name> | <base>..<head> | commit:<sha> | worktree] [emit]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN Ingestor Impact

Use this skill when the user asks whether a change can break the kernel ingestor at run time, and
suggest it for any change under `dnn-providers/hip-kernel-provider/`, any plugin SDK header
(`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/`, `projects/hipdnn/plugin_sdk/cmake/GpuGenericTargets*`),
`projects/hipdnn/plugin_sdk/data/gpu_generic_targets.json` or `projects/hipdnn/tools/IngestorGenerator/`.
[reference.md](reference.md) holds the contracts, row statuses, emitter rules and known gaps; read
it before judging rows.

The answer is advisory. It never builds the product, runs its tests, dispatches CI or posts
anything; the only thing it compiles is the throwaway C++ table of step 6.

## Prerequisites

`git` 2.38 or later (`merge-tree --write-tree`), Python 3.10 or newer and PyYAML; the emitter also
needs Jinja2. Run from a rocm-libraries clone or worktree. On Windows, `python3` is often the
Microsoft Store stub: use `python` or `py -3`. The optional C++ tables (step 6) need a host C++
compiler; on Windows, the ROCm wheel's `clang++` with the MSVC `INCLUDE` and `LIB` set.

## Workflow

1. **Locate the scripts.** Skills are host-level, not tied to a repo checkout: default to the
   scripts bundled with the skill you were invoked from, `<skill-directory>/scripts/`, where
   `<skill-directory>` holds this file. Use the copy in a checkout only when developing this skill.

2. **Pin the revisions** as full SHAs, once. Never pass `FETCH_HEAD` to a later command: another
   fetch can rewrite it in between.
   - **Open PR `<n>`.** `gh pr view <n> -R ROCm/rocm-libraries --json baseRefName,headRefOid`, then
     `git fetch origin <baseRefName> pull/<n>/head`. `HEAD` is `headRefOid`, `TIP` is
     `git rev-parse origin/<baseRefName>`, and `BASE` is `git merge-base <TIP> <HEAD>`.
   - **Merged PR.** The repository squash-merges: take `mergeCommit` from `gh pr view`; `BASE` is
     its first parent (`<sha>^`), `HEAD` the merge commit, and there is no `TIP`.
   - **Branch `<name>`.** `TIP` is `origin/develop` (or the branch it targets), `HEAD` the branch
     tip, `BASE` their merge base.
   - **`<base>..<head>`** or **commit `<sha>`** (`<sha>^..<sha>`). No `TIP` unless the user names
     a target that moved.
   - **Uncommitted work.** The scripts read commits only. `git stash create` makes a commit of the
     tracked and staged changes without touching the worktree, the index or any ref: `HEAD` is its
     output and `BASE` is `HEAD` of the checkout. Stage a new file first, or it is left out. An
     empty output means there is nothing to analyse.

3. **Collect the static facts.** Drop `--target-tip <TIP>` here and below when there is no `TIP`:

   ```bash
   python3 <skill-directory>/scripts/impact.py report --repo . --base <BASE> --head <HEAD> --target-tip <TIP>
   ```

   Exit 1 means it could not run: report its message, never a verdict. The commands here write
   `python3`; on Windows use `python` or `py -3` (see *Prerequisites*).

4. **Run the emitter** when any rocKE or `hkp_pack` Python file changed in code, including rows the
   report calls `inert`, or when the user asks for `emit`. It builds every shipped variant at both
   sides on two CPU workers for a few minutes, then puts each builder's support predicate to a
   table ([reference.md](reference.md) *Predicate tables*); on a host that other people or agents use at the same time, run one emitter at a time. Put its output
   outside the repository and delete any earlier file first:

   ```bash
   rm -f <scratch-dir>/emit.json
   python3 <skill-directory>/scripts/impact_emit.py run --repo . --base <BASE> --head <HEAD> --out <scratch-dir>/emit.json
   python3 <skill-directory>/scripts/impact.py report --repo . --base <BASE> --head <HEAD> --target-tip <TIP> --emit <scratch-dir>/emit.json
   ```

   Here `<scratch-dir>` is any directory outside the repository. An emitter exit 1 leaves no proof:
   keep the static report and say the emitted code was not compared.

5. **Judge every row** whose status is not `inert` or `outside`, with
   [reference.md](reference.md) *Judging the rows*. Read both sides at `<HEAD>` (`git show
   <HEAD>:<path>`), and at the merged tree when the row says `merged-only`. Judge each predicate
   table finding not marked `at_base` with *Predicate tables*.

6. **Write a C++ table** for each `changed` `launch_surface` row whose C++ side is a header-only,
   HIP-free function, when a host compiler is available: [reference.md](reference.md) *C++
   differential tables*. Work in a scratch directory outside the repository and delete it after.
   A row whose C++ side is a matcher in `*Native.cpp` gets no table: read it by hand and say so in
   *C++ table detail*. When a row qualifies but no table can be written (no compiler, the Python
   side cannot be imported), say so on the report's `C++ tables:` line rather than leaving it out.

7. **Report** in the format below.

## Report

```markdown
Runtime impact: <rollup as printed>  (judged: <rollup after judgment, with why when it differs>)
Pack-time: <line as printed>  (judged: <how many findings hold, and why any do not>)
C++ tables: <k> of <n> tables disagree (<function>: <what disagrees>, <severity>; ...) | <n> written, all agree | none written (<why>)
Analysed: <BASE>..<HEAD>  target tip: <TIP or none>  emitter: <flavor, or not run and why>

| Row | Status | Judgment | Evidence (path:line at HEAD or merged tree) |

Device runs that would settle the changed rows
- <ctest entry> on <arch> with the ingestor, rocKE and SDPA flags ON

Predicate tables: <the section's count line, or "emitter not run">
| Builder | Check | Field | Values | State (new, first shipped, at base) | Judgment |

C++ table detail: <per table: the pair, rows, each disagreement and its judgment; or none>

Not judged here: <the note the script printed (it prints one only with NONE or N/A), or none>
Known gaps that apply: <from reference.md, or none>
```

`<k>` on the `C++ tables:` line counts every table with at least one disagreeing row, whatever the
judgment; the judgment and severity follow in the parenthesis. Every judged finding (a row, a
predicate-table finding, a C++ table disagreement) carries a severity from
[reference.md](reference.md) *Severities*: `BLOCKING`, `IMPORTANT` or `SUGGESTION`.

Group `inert` and `outside` rows into one count each, with each `inert` reason listed once. Write
`none` for an empty section; do not drop it.

## Never claim

- **Device behaviour.** `NONE` means no runtime impact was found statically. It never means the
  engine dispatches or computes correctly on a device.
- **CI green.** A build or test run with the ingestor flags OFF, or on runners with no GPU, is no
  evidence about this question. State the flag values a quoted run actually had.
- **Cross-architecture results.** Never carry a result from one architecture to another, and expand
  a generic target such as `gfx11-generic` to its members.
- **Safety of files not inspected.** An `outside` file was counted, not checked.

## Related skills

- [hipdnn-deep-review](../hipdnn-deep-review/SKILL.md): a full review whose `IMPACT` lane runs these
  scripts and turns the rows into findings.
- [hipdnn-superbuild](../hipdnn-superbuild/SKILL.md) and
  [hipdnn-superbuild-test](../hipdnn-superbuild-test/SKILL.md): the flags-ON build and the device
  runs this skill names.
- [hipdnn-ingestor-engine](../hipdnn-ingestor-engine/SKILL.md): building or extending an ingestor
  integration.
