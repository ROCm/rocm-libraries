# Concepts (Linearizer, Inventory, Planner, Builder read this file)

## Logical change

The unit of splitting: the smallest set of ledger entries that has **one purpose, one consumer and one risk surface**, and that is closed under the contract dependencies below. Removing any entry would break the build or tests, leave dead or unused code, or leave a producer without its consumer. Examples: a function signature change plus all updated call sites; a bug fix plus its test; an enum value, the code that handles it, and the metadata mapping and validation that produce it.

- **Consumer:** the code or user that reads what the change produces at runtime (a launch path, a loader, a generated artifact, a reviewer-visible API). Two entries with the same consumer and the same purpose are one logical change.
- **Risk surface:** what breaks, and for whom, if the change is wrong or reverted alone: existing behavior, serialized or generated output, a public API, or only new opt-in surface.

## Contract halves

A contract is a value that one piece of code produces and another consumes: an enum and the strings that name it, a serializer and its parser, a generator and its loader, a metadata table and the validator for it, a flag and the code that reads it. The two sides are **halves of one contract and stay in one group**, whatever their language, directory or test framework, unless **each half is independently useful and independently revertable** (it has its own consumer, or it changes something that works today, and landing it alone leaves nothing dead or unparseable).

Language, directory, component and test framework are never reasons to split. They are tie-breakers only after the consumer and risk-surface tests leave a real choice.

## Behavior change versus new surface

A hunk that changes what an already-working path produces (a different source for a value, an unconditional instead of a guarded computation, a bit that is now set, a serialized name that changes) is a **behavior change**. A hunk that only adds something no existing caller can reach is **new surface**. Keep behavior changes out of the group that adds new surface, using subtractive states where the file serves both (`analyze-and-plan.md`, Phase 3). If an intermediate state would need invented text, keep them together and say so in the proposal.

**A behavior change must be shown, not read off the diff.** A hunk that removes or loosens a guard, condition or early return (or adds one) is a behavior change only if the condition can actually be false (or true) for some reachable input. Before classifying it, read the definitions of every predicate in the condition and the places that construct its inputs (for example check whether `isA(x)` already implies `isB(x)`). Reading definitions needs no code execution and is always allowed. If you cannot establish reachability, classify the hunk as `behavior change (unverified)`; such a hunk never gets its own group and never makes its group a `fix`: it goes with the nearest group that owns the surrounding code, and the proposal lists it as unverified. A guard that can be shown never to fire is dead code removal: type `chore`, no behavior change.

## Hunk ledger

One record of the aggregate change `git diff B L`, in `ledger.tsv` (written once by the Inventory with `scripts/build_ledger.py`; never edited). Formats and columns: `files.md`. Each row is one of:
- `H<n>`: one `-U0` hunk inside a modified text file, with its line ranges in B and in L;
- `F<n>`: a whole-file change: add, delete, rename, type change (for example file to symlink), binary, submodule pointer, or a **mode-only** change (these produce no hunk);
- `G<n>`: one file of a generated set (see Generated files).

The planning files (`groups.tsv`, `plan_assign.tsv`, `transitions.tsv`) assign every row to groups. Every entry is assigned exactly once, except entangled entries, which have one step per group.

The ledger is generated with fixed `git diff` options recorded in its header; the scripts refuse a ledger made with different options. Adjacent changes with no unchanged line between them are one `-U0` hunk and cannot be separated by hunks; treat them as an entangled entry if they belong to different groups. A rename that also edits the file is one whole-file `F` row. The `\ No newline at end of file` state belongs to the hunk that touches the last line.

## Linear stack

The output is one linear stack: `01` is based on B, and each `NN` is based on `NN-1`. No siblings, no merges, one commit per branch. Dependencies decide which orders are valid; they never create branching. Each branch is a prefix of the next, so the top branch contains everything. This maps directly onto one `gh stack`.

## Generated files

Generated or bulk data files (for example rocRoller `.s` assembly, logic/tuning YAML) go into **standalone branches** by default, so each PR with hand-written code stays small and reviewable. One standalone branch per producing purpose (for example one per kernel set), not one branch mixing unrelated generated sets.

- The Inventory names the sets (`--generated-set NAME=GLOB`, and `--attr-generated` for `linguist-generated` / `-diff` attributes) and records them in `inventory.md`. Default globs for the size estimate before the Inventory runs: `*.s`, `*.o`, `*.bin`, `*.hsaco`, `*Logic*/**.yaml`, `*.lock`, files marked `linguist-generated`. Generated means *produced by a tool*; hand-authored bulk files are ordinary `F` rows. A set is assigned as a whole: a plan row with entry `@<set>` assigns every file of the set. A set is never split over groups.
- Exceptions, which stay with their code as ordinary `F` rows:
  - deletions, moves and renames (and type changes) of generated files: `build_ledger.py` makes these `F` rows automatically, because only added and modified paths join a set. Their `generated` column still names the set (size accounting only); they are assigned on their own, with the code that stops using or starts using the path;
  - generated files added as test files or fixtures, and any other change that must be atomic with its code (the code and the files cannot each build and pass without the other in either order): the Inventory excludes them with `--exclude-generated GLOB` (recorded in the ledger header) and says why in `inventory.md`.
- Generated files are never hand-edited or synthesized; they are always taken whole from L. If a generated file would need different states in two groups, keep both changes in one group.

## Formatting changes

The pre-commit hooks run on the copy before and after the update (0e), so the source arrives hook-clean. Consequences:
- Formatter changes inside lines the branch adds or changes are already part of the final text; the ledger sees only the aggregate diff and they travel with the group that owns that code.
- Formatter changes to existing lines the branch did not otherwise touch form their own `style` group at the bottom of the stack.
- A formatter change on lines that other code also changes is an entangled entry (below).
- The hook commits are identified by their SHAs in `update.md`, not by subject; they are committed with the subject `style: apply pre-commit hook fixes`.

## Entangled changes

When a fix, a refactor and a feature change the same code (for example one line carries a bug fix and a rename), the default is to split them. A **test file that covers several groups** (for example one new test file whose cases assert three different fixes, or one test body that exercises three APIs) is the same situation: it is an entangled whole-file entry, and the default is to split it by what each part asserts. Such an entry is not split by line range; it gets an ordered sequence of states from B's text to L's text, one step per group, in `plan_assign.tsv`. Keep the entry in one group (stating both purposes, and telling the user in Phase 6) only when an intermediate state would need text that exists nowhere in the final file.

- **Where a state comes from**, in order of preference:
  1. *History.* For an `H` entry run `scripts/extract_state.py --entry H<n> --commit <full sha> --out states/<file>` (it takes the hunk's region at that commit of L; use `git log -L <range>:<file> B..L` to find which commit) and reference it as `stored:<path>` with note `history: <sha>`; for an `F` entry use `commit:<sha>`. L is linear on B, so a commit's version is B plus a prefix of the work.
  2. *Subtraction.* The final text with blocks deleted (whole test cases, helpers, struct fields, calls that belong to later groups) and no line added: reference it as `subtractive:<path>`. `check_plan.py` proves it: its lines (exact bytes, including terminators) must be an ordered subsequence of the final text (for an `H` entry L's lines of the hunk, for an `F` entry the whole text file at L), and of the next step's text when that step is also `subtractive:`. It is allowed for text hunks and for added, renamed or type-changed regular text files. Subtractive states are expected and are the default way to stage tests.
  3. *Hand-written.* Anything else (for example a version that keeps base lines a later group deletes) is `stored:<path>` with note `synthesized`.
  Every state that is not `final` or `commit:` is a file under `$RUN/states/`.
- **What a stored or subtractive state is:** for an `H` entry, the exact bytes of the replacement for the hunk's lines, including line terminators (it must end with a newline, or it glues itself to the next line); for an `F` entry, the whole file. `commit:<sha>` is allowed only for whole-file entries and means the file's content at that commit. A submodule entry takes no `stored:` or `subtractive:` state. State paths must stay inside `$RUN`. (This is the hunk-level form; every branch that changes a file with entangled entries still gets that file's other hunks from the build script, so no full-file versions are written by hand.)
- **How it is used:** `build_stack.py` builds each branch from B plus the entries assigned to groups up to that branch, using the state of the latest step at or below it. The last step is always `final`, so `tree(top) == tree(L)` holds.
- **Visibility and trust:** Phase 6 shows every non-final state as a before/after diff under the branch that uses it (`render_plan.py --states`). A `subtractive:` state that passes `check_plan.py` is proven to add no text, so it is not an `--implement` stop; a `stored:` state marked `synthesized` is. Neither check proves that the intermediate branch builds: that is established only by the Phase 8 tests (if the user chose tests).
- Hand-written text lives in the plan, never in a built branch. A wrong stored state is fixed by correcting the stored text and rebuilding.

## Transition code

Temporary code added in branch X and removed in a higher branch Y is allowed only when it is purposeful: it exists for compatibility so that each branch between X and Y builds and passes on its own. It is never a repair of an earlier mistake.
- Recorded in `transitions.tsv` with the file, X (`added_in`), Y (`removed_in`), the stored full-file content that applies to branches X up to (not including) Y, and the reason. Phase 6 lists it under "Transition code" and in the descriptions of X and Y.
- The Planner may plan it in Phases 3–5. A Phase 8 test failure may only lead to a *proposal*: the Builder writes it under `## Blocked` (what, where added, where removed, why) and the orchestrator asks the user, also under `--implement`. A proposal is not a fix attempt. This is the only home of this rule.
- Windows on one path must not overlap, and the path must not be deleted, renamed or type-changed inside its window. A temporary file that exists at neither end is allowed. The `stored` file is a full file and must already include every other change to that path in the window; when a fix moves an entry of that path into or out of the window, regenerate the stored file (`check_plan.py` warns about entries inside a window).
- Code that is added in one branch and removed in a higher one without a record is forbidden. A failure caused by a bad split is fixed by changing the split.

## Types (closed list) and tie-break order

Types are the conventional-commit types. They are used in the Phase 6 `type` column, for ordering, and as the commit subject prefix `type(scope): subject`. A type never overrides a dependency; it only orders groups that are ready at the same point in the stack.

| Type | Description | Order |
|---|---|---|
| `fix` | bug fixes: small, urgent, uncontroversial. Requires a stated defect: the wrong output, crash or misbehavior the group repairs, and the evidence for it (a failing case, a reachable input). A change that only alters behavior, or whose effect is unverified, is not a `fix` | 1st |
| `chore` | removing dead code | 2nd |
| `style` | formatting only | 2nd |
| `test`, `ci`, `build`, `docs` | changes that are only coverage, CI, build configuration, or documentation. Tests, build lines and docs for code in this split travel inside that code's group (see Phase 4 in `analyze-and-plan.md`) and take its type (`fix`, `feat`, ...); a standalone `test` group is for tests of code that is not part of this split, or tests that cannot be split without invented text. If it depends on a group in this split it goes right after that group; if independent it is 3rd | 3rd |
| `refactor`, `perf` | restructuring with no behavior change; performance work | 4th |
| `feat` | new behavior, and plumbing that later features need | 5th |

Plumbing that a later feature uses is ordered before that feature by dependency, not by type.

## Edge cases

- **Binary files:** one `F` row, assigned to one group by purpose. If a binary serves several purposes (rare), assign it to the highest group and note the overlap.
- **Renames:** a rename plus its content edits is one whole-file `F` row (`status R`, `old_path`). If another group also changes the renamed file, that group goes above the renamer, using a stored state.
- **Submodule pointers:** whole-file `F` rows (`binary` column `sub`), assigned to the group whose feature needs them, or to a `feat` group if shared infrastructure.
- **Mode and type changes:** their own `F` rows; a file with content hunks and a mode change has both `H` rows and one mode `F` row.
- **Merge commits in the source:** classified in Phase 0c and handled in Phase 1; their residue is evidence for attribution. Never use `--first-parent` to enumerate work: it hides commits that arrived through merges.
