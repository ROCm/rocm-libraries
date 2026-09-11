# Choosing the next module to test

Choose the next mutation target from measurements that another developer can
review. The goal is not one universal score. The goal is a clear decision whose
inputs and reasoning remain understandable after the current campaign ends.

## 1. Keep one fixed list of candidate modules

List the source modules before comparing them. Give every module one status:

- `completed`: the mutation run and result review are certified for the
  recorded source and settings;
- `ready`: the selected tests and environment are ready for a mutation slice;
- `coverage-gap`: the tests do not execute the required percentage of the
  target file;
- `deferred`: a named dependency, cost, or design decision prevents work;
- `out-of-scope`: generated, third-party, data-only, or otherwise excluded,
  with a reason; or
- `pending-evidence`: a required measurement or environment record is
  missing.

Do not infer source modules from directory names used by tests that record
current behavior. One directory may test several modules, and one module may
need several test directories. Map every source module to an explicit test
selection.

Keep the candidate list unchanged for one comparison. Adding or removing a
module can change a scaled rank even when every raw measurement stays the same.

## 2. Prefer direct test and mutation measurements

Collect these measurements in order:

1. The percentage of the exact target file executed by tests, including branch
   coverage and the count of lines not executed.
2. Counts of prior not-killed mutants, separated by mutmut's original status.
3. Mutants that review confirmed still survive or have no test.
4. Timeouts, segmentation faults, and other unresolved results that require
   environment work.
5. Whether a repeatable covering-test selection exists.
6. Cost: unchanged-source test duration, mutant count, and estimated rerun
   time.

For every measured value, record the source commit ID, tool versions, settings,
test selection, command, and output path.

Do not compare mutation counts from different mutmut versions, target sets,
excluded lines, or covered-line settings.

## 3. Use secondary measurements only to break ties

Code complexity, lines of code, imports, and recent edits may help choose
between otherwise similar modules. They do not replace measured test coverage
or mutation results.

Define every secondary measurement:

- lines of code: physical lines, logical statements, or executable lines;
- complexity: the tool, tool version, and method used to combine values;
- connections to other modules: the parsed import graph and whether tests are
  included; and
- recent edits: the time window and Git history used.

Counting text matches in Python files does not measure how many modules import a
target. Do not label that count as an import measurement.

## 4. Do not calculate a score from missing data

When a required input is missing, mark the value `PENDING`. Choose the next
work through explicit review instead of inventing a number.

Before adopting a weighted score, require a committed decision record that
defines:

- why each measurement predicts useful mutation work;
- units and allowed ranges;
- the fixed module list;
- how values are scaled before combining them;
- weights and how small weight changes affect the order;
- treatment of missing or unresolved inputs; and
- when the formula must receive a new version or be retired.

Scores calculated from different module lists cannot be compared directly. Do
not copy final statuses from a private or missing plan and present them as
calculated output.

## 5. Use this default decision order

1. Do not rerun a `completed` module unless its source or settings changed, or
   a later report shows a possible regression.
2. For `coverage-gap` modules, first add a repeatable test selection that
   executes enough of the target file.
3. Resolve environment-caused timeouts and unresolved results before
   interpreting a mutation score.
4. Among `ready` modules, prefer modules with many confirmed survivors or
   no-test results, important product behavior, and manageable run time.
5. Use complexity, imports, code size, and recent edits only to break a tie
   supported by direct evidence.
6. Keep every deferred or out-of-scope reason visible.

Product importance is a human decision. Record why a module matters to shipped
behavior, validation, code generation, or frequently changed code.

## 6. Record the selection

Write:

```text
fixed module list and source commit ID
selected module
current status
covering-test record and target-file coverage
prior mutation and review results
estimated cost
secondary measurements used and their definitions
why this module ranks above the nearest alternatives
known blockers and required automated or device checks
reviewer or owner decision
```

Store the decision with the campaign handoff or tracked issue. Do not create a
history directory that is neither committed nor read by another tool.

## 7. Measure again when inputs change

Rebuild the comparison when:

- the source commit changes in a way that affects the modules;
- target modules or test selections change;
- mutmut or coverage versions or settings change;
- a module is completed;
- a new coverage gap or regression appears; or
- measured run time invalidates the plan.

Keep old run records as historical evidence. Do not present their counts or
decisions as current after inputs change.

## 8. Requirements for future automation

Automate the ranking only when a real recurring workflow will consume it and:

- every candidate maps to a source module;
- every required measurement has a validated file format and source record;
- missing values and invalid ranges stop the calculation;
- module connections use a parsed import graph;
- scaling remains stable across saved comparisons;
- formula and version changes are explicit; and
- history is committed or consumed by automated GitHub checks.

Tests must use a fixed example repository, not the live source tree. They must
prove that changing the candidate list cannot silently change the meaning of
past results.
