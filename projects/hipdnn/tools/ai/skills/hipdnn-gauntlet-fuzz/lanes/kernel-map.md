# Lane: kernel dispatch map (`KMAP`)

**Trigger:** always. **Agent:** `task` (it reads the plan, matcher and kernel). **Output:** `R/lanes/KMAP.md`: the map, then ideas.

Every kernel or variant the engine can launch must be hit by at least one fuzzed graph. Build the map that proves it.

## Method

1. From descriptors, catalogs, tuning tables, heuristics, knobs and fallbacks (brief section 7), list every kernel or variant per arch
   and dtype, including companion kernels (prep, reduce, tail-zero, head-pad copy, ...).
2. For each, the graph features that select it (head dims, dtype, mask, bias, dropout, sequence lengths against tile sizes, layout,
   GQA group, knobs) with `path:line`.
3. For each, **one concrete graph that hits it**, and whether a current space already draws such a graph.
4. Note how the engine reports which kernel ran (a `kernel: <engine> <kernel>` INFO line, which the harness records per case) or that
   it does not; then hits can only be shown by construction.

## Output

Before the ideas, a table:

| Kernel or variant | Arch, dtype | Selected when | Graph that hits it | Covered by |
|---|---|---|---|---|

Then one idea per kernel no current space hits (`Expect: KERNEL-UNREACHED`, `Place: kernel-rows <kernel>`), and per kernel only a
knob reaches (`Place: kernel-rows` with `--meta engine_variant={"id":...,"knobs":{...}}`). The `KEDGE` lane takes this table as input.
