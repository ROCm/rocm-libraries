# Comparing asm (tier 1)

Read this when you run tier 1. The flow is in `ab.md`. The rules below apply to any input. Phase 1 uses
whatever input the target needs. The perf set in "Phase 2" is for the regression check.

## What "identical" means (tier 1)

"Asm identical" is a content comparison, never a hash alone.

```text
- compare file by file with diff or cmp. Every file must match. A combined hash does not count.
- report how many files were compared and how many differ. Zero files compared is a failure.
- generate each side in its own working directory:
  `Tensile --cpu-only --gpu-targets <arch> <yaml> <working directory>`. `--cpu-only` requires
  `--gpu-targets`, and its perf numbers are synthetic, so it is only for generating asm. Put extra
  options such as `--global-parameters CpuThreads=4` after those arguments, because
  `--global-parameters` takes everything that follows it. The perf YAMLs set `KeepBuildTmp: True`,
  which keeps the asm. Find it with `find <working directory> -name '*.s'`. The folder names inside
  look the same on both sides, so the working directory is the only way to tell them apart. Print
  both paths and check they differ.
- know which layer you edited. `ROCISA_BACKEND` chooses how a kernel reaches the ASM IR:
  `ROCISA_BACKEND=stinkytofu` goes logical IR -> ASM IR, and `ROCISA_BACKEND=rocisa` goes rocisa -> ASM
  IR. Both then run the same ASM IR passes. A change in an ASM IR pass (for example the DAG scheduler)
  shows in both modes. A change in the logical IR shows only with `stinkytofu`, and a change in rocisa's
  own code shows only with `rocisa`. Run the mode that executes the code you edited, use the same mode
  on both sides, and record it. One mode is enough for a change in an ASM IR pass. With the wrong mode you will see no difference. With no edit, both modes
  give the same asm, and a difference is a bug. Print the loaded `rocisa` and `stinkytofu` paths on each
  side, and check that they are the worktree's own.
- build the baseline and the candidate in separate worktrees, each with its own venv, so the two
  rocisa bindings do not shadow each other (see the `worktree-venv` skill in `skills/`). Rebuild the
  candidate after the last edit (see "Tier 1 needs a fresh binding" in `ab.md`).
- the labels in the generated asm contain random hashes, so two runs of identical code differ in
  label names. Normalize the labels before comparing: in each file, rename every distinct label to a
  sequential name in order of first appearance. At the time of writing a label is the token
  `label_[A-Za-z0-9_]+`, but the naming may change, so check it. A raw diff that differs only in
  labels is not a real difference.
- list exactly what is normalized (labels, and anything else such as paths or timestamps), and show a
  diff proving nothing else was removed or changed.
- `tools/ab-compare.py` (next to this file) does the label-normalized compare:
  `python3 tools/ab-compare.py <baseline dir> <candidate dir>`. Exit 0 means identical after
  normalization, 1 means a difference or a file missing on one side, and 2 means an error. Use it, or
  do the same by hand. Run it on two runs of the same build for the baseline-against-itself check.
- prove that only labels changed: replace every label token with one fixed token in both versions of
  a file. The two texts must then be equal.
- as a check, compare the baseline against a second run of itself for one YAML. After normalizing, it
  must be identical. If it is not, the comparison is not trustworthy yet, and the label definition
  above may need updating.
- on any difference, show the diff. Do not call it "equivalent" unless the developer agrees.
```

## Phase 2: the regression set

The inputs are the perf YAMLs under `projects/hipblaslt/tensilelite/tensilelite/Utilities/tensile_generator/perf_yamls/<variant>/`.

```text
- the variant is a directory name there (a machine variant, not just an arch). Ask the developer
  which variant the change is for. Do not guess. If it has no directory, ask.
- run every YAML under that variant's maf/, mab/, and medium/ directories (skip a directory the
  variant does not have), including the finetune,
  k128k, and other variants. Ignore tuning/ subdirectories: perf YAMLs are for perf tests only.
  No sampling: the full set is already the minimum.
- one tier 1 result per YAML, not one for the whole set
```
