# sdpa_case_flow_poc

This package makes a list of test cases for SDPA forward. **SDPA** means
*scaled dot-product attention*, the "attention" step of a transformer model.
**Forward** means the step that computes the output (not the gradients).

The list of cases is one CSV file, `levels.csv`. A **CSV** file is a text table:
one line per row, with commas between the fields. Each row of `levels.csv` is
one test case. The row gives the case's tier, its sizes, its estimated time and
one value for every setting of the model.

The package is a **proof of concept (PoC)**: a first working version, made to
show that the method works. The design and the files are described in
[SDPA Forward Test Plan](../../docs/test-plans/sdpa-fwd.md).

## Words used in this document

| Word | Meaning |
|---|---|
| model | The written description of every SDPA setting and its possible values. It is in `model.yaml`. |
| tier | How large a test run is. There are four tiers: `quick`, `standard`, `comprehensive` and `full`. Each tier also runs the cases of the smaller tiers. |
| SHA-256 | A 64-digit hex number computed from the bytes of a file. If one byte of the file changes, the number changes. Two files with the same SHA-256 are the same file. |
| hex | Base-16 numbers, written with the digits `0`-`9` and `a`-`f`. |
| seed | The start number of the random choices. The same seed gives the same choices. This plan uses seed 2893. |
| graph | The hipDNN description of one operation: its tensors (multi-dimensional arrays of numbers), their sizes and the operation's settings. |
| bundle | A test that the integration suite runs, stored as files in `integration-test-bundles/`. |
| sweep | One bundle that holds many cases: one graph template (`graph.template.json`) and a list of case values (`sweep.json`). |
| topology | The set of tensors that a case uses. A sweep's template has the topology of its largest case; a case with fewer tensors names the missing ones in `absent_tensors`. |
| engine | One implementation of an operation, inside a provider plugin (a shared library that hipDNN loads). Example: `ASM_SDPA_ENGINE`. |
| reference | A simple, slow program that computes the expected output. hipDNN has a CPU reference and a GPU reference for SDPA. |
| arch | The GPU architecture name. This plan uses `gfx942` and `gfx950`. |
| CMake | The tool that configures and builds the C++ programs. |

## Inputs

All inputs are in [`test-plans/sdpa-fwd/`](../../test-plans/sdpa-fwd/):

| File | What it holds |
|---|---|
| `model.yaml` | The settings, their values and labels, the tier recipe and the cost model. |
| `rules.yaml` | The rules that remove combinations of values. |
| `injections.yaml` | Cases outside the generated sets. The generator checks that they name known values. |
| `timings-gfx942.csv`, `timings-gfx950.csv` | Measured seconds per case (`case_key,seconds,source`). A case that is not in a file gets its time from the cost model. |
| `levels.csv.sha256` | The SHA-256 of the `levels.csv` that these files give. |
| `levels.csv` | A copy of that `levels.csv`, so that you can compare your file with it. No tool reads this copy. |

The seed is 2893.

## Regenerate the CSV

Run the command from the folder `dnn-providers/integration-tests/scripts`. You
need Python 3 and the PyYAML package.

```bash
python3 -m sdpa_case_flow_poc.levels \
    --model-dir ../test-plans/sdpa-fwd \
    --out-dir <output folder> \
    --seed 2893 \
    --timings gfx942=../test-plans/sdpa-fwd/timings-gfx942.csv \
    --timings gfx950=../test-plans/sdpa-fwd/timings-gfx950.csv
```

The command writes `levels.csv` and `levels-summary.md` to the output folder.
It takes about 3 minutes. The same inputs always give the same `levels.csv`,
byte for byte.

## Check the CSV

`../test_sdpa_fwd_levels_regenerate.py` runs the command above. Then it
compares the SHA-256 of `levels.csv` with `levels.csv.sha256`. CTest (the test
runner of CMake) runs it as `hipdnn_sdpa_fwd_levels_regenerate_python_tests` in
the `comprehensive` tier.

If you change the model, the timing files or the generator, and the CSV changes
on purpose, then in the same change:

1. write the new SHA-256 to `levels.csv.sha256`;
2. copy the new `levels.csv` to `test-plans/sdpa-fwd/levels.csv`.

## The two C++ programs

The pipeline has two C++ programs (executables). Both are proof-of-concept
tools. Both read rows of a case CSV and turn each row into a hipDNN graph with
the same code: `src/sdpa-case-flow-poc/CaseGraph.cpp`. So one row always gives
the same graph in both programs.

| Program | What it does | Needs a GPU? | CMake option that builds it |
|---|---|---|---|
| `hipdnn_sdpa_case_flow_runner` | Builds the graph for each CSV row. With `--export-graphs` it writes each graph as a JSON file and runs nothing. Without `--export-graphs` it runs one engine on the graph, runs the CPU and GPU references, compares the outputs and measures the times. | No for `--export-graphs`. Yes for the other mode. | `HIPDNN_IT_ENABLE_SDPA_CASE_FLOW_RUNNER` |
| `hipdnn_sdpa_case_flow_discovery` | Builds the same graphs. It only asks one engine if it accepts each case, and writes the answer. It runs nothing. | Yes. hipDNN needs a GPU to ask an engine. | `HIPDNN_IT_ENABLE_SDPA_CASE_FLOW_DISCOVERY` |

Both options are `OFF` by default.

Which steps of the pipeline (below) need which program:

| Step | Program |
|---|---|
| 5. Export the graphs | runner, export mode (no GPU) |
| 8. Run the engines | runner, run mode (GPU); optional |
| 9. Reference check | runner, run mode (GPU) |

No step needs the discovery program. Use it to find out which cases an engine
accepts before you run them. Run `<program> --help` for its options.

**`generator_version`.** The runner writes a `generator_version` value into the
metadata of every graph it exports. The value is the first 12 hex digits of a
SHA-256 over three source files: the runner's `Main.cpp`, `CaseGraph.cpp` and
`CaseGraph.hpp`. CMake computes it when it builds the runner. So the value
tells which version of the source code made a graph. If one byte of these three
files changes, the value changes, even when the graphs stay the same.

## From the CSV to bundles

This section is the whole pipeline: model, then `levels.csv`, then graphs, then
bundles. With it you can make the `SdpaFwdPlan` bundle set again from the files
in the repository. The design behind each step is in
[SDPA Forward Test Plan](../../docs/test-plans/sdpa-fwd.md), sections 8 to 10.

The `SdpaFwdPlan` bundles that we committed came from exactly this process:

- the committed model files,
- seed 2893,
- a `levels.csv` with the SHA-256 in `levels.csv.sha256`,
- the default selection parameters of `select_rows.py`.

[Compare your bundles with ours](#10-compare-your-bundles-with-ours) tells how
to check your result.

### Where the tools are

| Folder | What it holds |
|---|---|
| `scripts/sdpa_case_flow_poc/` | The generator: model to `levels.csv` (`levels.py`), and the class-to-value file (`python3 -m sdpa_case_flow_poc`). |
| `scripts/sdpa_case_flow_poc/bundles/` | The bundle tools: `select_rows.py`, `topology.py`, `compare_exports.py`, `make_bundles.py`, `annotate_csv.py`, `compare_bundle_trees.py`, and for the GPU steps `analyse.py`, `build_skips.py`, `analyse_refcheck.py`. Run them as scripts. |
| `src/sdpa-case-flow-poc/` | `CaseGraph.cpp`, which makes the hipDNN graph of a row, and the discovery program (`hipdnn_sdpa_case_flow_discovery`). |
| `src/sdpa-case-flow-runner/` | The runner (`hipdnn_sdpa_case_flow_runner`). |
| `migration-scripts/import_graph.py` | The suite's import tool. Its opt-in grouped mode writes the sweeps. |

Every input and output of the bundle tools is a command-line argument. The
default values are paths inside the repository. Run `<tool> --help` for the
full list.

### Steps

Only the steps marked GPU need a GPU. Step 5 needs a build of the runner, but
no GPU. The times are what the steps took for the 9,883 bundle rows.

| Step | Tool | GPU | Output | Time |
|---|---|---|---|---|
| 1 | `sdpa_case_flow_poc.levels` | no | `levels.csv`, checked against `levels.csv.sha256` | about 3 min |
| 2 | `python3 -m sdpa_case_flow_poc` | no | `class-to-value.json` | seconds |
| 3 | `bundles/select_rows.py` | no | `selected.csv`, `exclusions.csv`, `summary.json` | under a minute |
| 4 | CMake | no | `hipdnn_sdpa_case_flow_runner` | depends on the computer |
| 5 | runner `--export-graphs`, `bundles/compare_exports.py` | no | one graph and one metadata file per row, per arch | about 1 min per arch |
| 6 | `bundles/make_bundles.py` | no | the `SdpaFwdPlan` sweeps, `placements.csv` | 10 to 60 min; one import process per graph |
| 7 | checks, `bundles/annotate_csv.py` | no | check results, `cases.csv` | minutes |
| 8 | `hipdnn_integration_tests`, `bundles/analyse.py`, `bundles/build_skips.py` | GPU | test results, `[[test_skips]]` entries | minutes per tier and arch |
| 9 | runner, `bundles/analyse_refcheck.py` | GPU | reference acceptance and CPU-vs-GPU agreement per row | about 6 GPU hours per arch |
| 10 | `bundles/compare_bundle_trees.py` | no | a match report against our bundles | seconds |

The commands below use these names. Set them first, with your own paths:

```bash
REPO=<repository root>
IT=$REPO/dnn-providers/integration-tests
TOOLS=$IT/scripts/sdpa_case_flow_poc/bundles
OUT=<an empty work folder>
BUILD=<CMake build folder>
```

### 1. The CSV

```bash
cd $IT/scripts
python3 -m sdpa_case_flow_poc.levels \
    --model-dir ../test-plans/sdpa-fwd --out-dir $OUT/levels --seed 2893 \
    --timings gfx942=../test-plans/sdpa-fwd/timings-gfx942.csv \
    --timings gfx950=../test-plans/sdpa-fwd/timings-gfx950.csv
sha256sum -c <(sed "s|  levels.csv|  $OUT/levels/levels.csv|" ../test-plans/sdpa-fwd/levels.csv.sha256)
```

The check must print `OK`. If it does not, then the model files, the timing
files or the generator changed, and the rest of the pipeline gives a different
set.

Then compare your `levels.csv` with the committed copy:

```bash
cmp $OUT/levels/levels.csv $IT/test-plans/sdpa-fwd/levels.csv
```

`cmp` prints nothing when the two files are the same. If they differ, it prints
the first byte and line that differ. To see the different lines, use
`diff $OUT/levels/levels.csv $IT/test-plans/sdpa-fwd/levels.csv`.

### 2. The class-to-value file

The model gives most sizes as a **class** (for example `batch_size` is `one`
or `many`). The class-to-value table in `model.yaml` gives the numbers for
each class. The selection step and the runner read this table as a JSON file.

```bash
cd $IT/scripts
python3 -m sdpa_case_flow_poc --model-dir ../test-plans/sdpa-fwd \
    --out-dir $OUT/discovery --modes "" --skip-second-strength
```

This writes `$OUT/discovery/class-to-value.json` and `manifest.yaml`. If you
leave `--modes` at its default, the command also writes the one-at-a-time and
pairwise sets that section 6 of the design document describes.

### 3. Select the rows that a bundle can express

Some rows cannot be bundles. For example, a bundle cannot hold exact tensor
contents. This step keeps the rows that a bundle can hold.

```bash
python3 $TOOLS/select_rows.py --input $OUT/levels/levels.csv \
    --class-values $OUT/discovery/class-to-value.json --out-dir $OUT/select
```

`selected.csv` holds the selected rows, each line the same as in `levels.csv`.
`exclusions.csv` gives every other row with its reasons (section 9.1 of the
design document).

The default values are:

- all four tiers;
- minimum sweep size 1;
- the zero-stride filter on;
- the descale filter (`--descale-non-fp8-filter`) off.

With these defaults, and an input whose SHA-256 is the one in
`levels.csv.sha256`, the script checks the counts that it records: 17,969 rows
in 832 sweeps.

`--no-zero-stride-filter` also selects the 870 zero-stride rows that a bundle can
otherwise express. The data-SDK change that they need (PR #14709) is merged. But
these rows are left out of this bundle set. Adding them is planned as a
follow-up change.

`topology.py --input … --class-values … --out topologies.csv` checks that the
topology names and the tensor signatures map one to one, and lists them. With
`--grouping-summary <json>` it also writes, per tier, the sweep counts with one
template per signature and with superset templates (see section 6).

### 4. Build the runner

The runner is part of the integration-test project. It needs hipDNN built with
SDPA support. The two options in
[The two C++ programs](#the-two-c-programs) add the two programs. For example,
with the repository superbuild (one CMake build of hipDNN and its providers):

```bash
cmake -S $REPO -B $BUILD -G Ninja -DCMAKE_BUILD_TYPE=Release \
    --toolchain $REPO/projects/hipdnn/cmake/ClangToolChain.cmake -DROCM_CMAKE_PATH=<ROCm root> \
    -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;hip-kernel-provider;hipdnn-integration-tests" \
    -DGPU_TARGETS="gfx942;gfx950" -DHIPDNN_ENABLE_SDPA=ON \
    -DHIPDNN_IT_ENABLE_SDPA_CASE_FLOW_RUNNER=ON -DHIPDNN_IT_ENABLE_SDPA_CASE_FLOW_DISCOVERY=ON
cmake --build $BUILD --target hipdnn_sdpa_case_flow_runner hipdnn_sdpa_case_flow_discovery
```

Step 5 needs only the runner. The steps that run engines (8 and 9) also need the engine plugin (`libhip_kernel_provider.so`) and
`hipdnn_integration_tests`. Build those targets too, or build everything.

### 5. Export the graphs

The export loads no plugin, creates no hipDNN handle and runs nothing on a
GPU. For each row it writes:

- `<case_key>/<case_key>.json`: the graph;
- `<case_key>/<case_key>.meta.json`: the bundle metadata;
- one JSON line with the row's outcome.

```bash
SHA=$(cut -d' ' -f1 $IT/test-plans/sdpa-fwd/levels.csv.sha256)
for ARCH in gfx942 gfx950; do
    $BUILD/bin/hipdnn_sdpa_case_flow_runner --input $OUT/select/selected.csv \
        --output $OUT/export-$ARCH.jsonl --class-values $OUT/discovery/class-to-value.json \
        --export-graphs $OUT/export-$ARCH --arch $ARCH --source-sha256 $SHA
done
python3 $TOOLS/compare_exports.py $OUT/export-gfx942 $OUT/export-gfx950 \
    --report $OUT/compare-exports.json --outcomes $OUT/export-gfx942.jsonl $OUT/export-gfx950.jsonl
```

Every selected row must have the outcome `exported`. If a row does not, run
step 3 again with `--export-outcomes <jsonl>` to exclude it. The two exports
must be equal, because no selected row depends on the arch (section 9.2).

### 6. Make the bundles

`make_bundles.py` runs `import_graph.py` in grouped mode once per graph, into
an empty work tree. Per tier, it puts each tensor signature into a superset
template (`topology.superset_map`): the signatures are taken largest first, and
each joins the smallest earlier superset that holds all of its tensors and has
the same scale form, dropout input form, page tables and ragged offsets, else
it starts a superset. The sweep of a superset is named after the superset's
signature. The script sorts the graphs by tier, superset, signature size
(largest first), signature and `case_key`, so a superset's own graphs create
its template. Each run uses `--op-dir SdpaFwdPlan --topology <name>
--template-attributes --template-alignment --skeleton-value-form --only-op-dir
--require-marked-sweep --superset-templates --case-id <dtype_q>_<case_key>
--strict`. A case with fewer tensors than its template lists them in
`absent_tensors`, and its metadata has `format_version` 2. `--exact-templates`
makes one sweep per signature instead (no `--superset-templates`).

```bash
python3 $TOOLS/make_bundles.py --selected $OUT/select/selected.csv \
    --class-values $OUT/discovery/class-to-value.json --export-dir $OUT/export-gfx942 \
    --work-dir $OUT/bundles --out-dir $OUT/make-bundles
```

The work tree gets `<tier>/SdpaFwdPlan/<Topology>/` with three files:
`graph.template.json`, `sweep.json` (compact layout) and an empty
`support.json`.

Useful options:

- `--install-into $IT/integration-test-bundles` then copies the four op
  folders into the suite's tree. It refuses to overwrite a folder that exists.
- `--sample N`, `--only-tier`, `--only-topology` and `--limit` import only some
  rows. Use them to try a change quickly.

`placements.csv` maps each `case_key` to its tier, topology, case id, bundle
path and GTest name (the name of the test in the suite). `report.json` has the
import times, the sweep sizes and, per tier, the sweep counts and one-case
sweep counts with one template per signature and with superset templates.

### 7. Verify

`make_bundles.py` checks the tree before it reports `OK`:

- each topology (superset) is one sweep per tier;
- no sweep holds two topologies;
- the op folder holds no other sweep;
- every case expands back to its exported graph: the absent tensors leave the
  template first, then the placeholders expand, then the tensor patches apply;
- no case names a tensor of its own graph in `absent_tensors`, and each case
  with absent tensors meets the bundle loader's rules for them.

`import_graph.py --strict` also checks each case when it writes it.

Then install the op folders into the suite's tree (if `--install-into` did not
do it), run the suite's checks, and write the annotated CSV:

```bash
for T in quick standard comprehensive full; do
    cp -r $OUT/bundles/$T/SdpaFwdPlan $IT/integration-test-bundles/$T/
done
python3 $IT/reference-data-scripts/verify_golden_bundles.py \
    $IT/integration-test-bundles/{quick,standard,comprehensive,full}/SdpaFwdPlan
python3 $IT/scripts/verify_support_claims.py
python3 $IT/scripts/support_features/feature_matrix.py --check
python3 $TOOLS/annotate_csv.py --input $OUT/levels/levels.csv \
    --class-values $OUT/discovery/class-to-value.json --selection-dir $OUT/select \
    --placements $OUT/make-bundles/placements.csv --out $OUT/cases.csv
```

`verify_support_claims.py` and `feature_matrix.py` read the suite's tree.
The committed `SUPPORT_MATRIX_OVERVIEW.md` on this branch already counts the
`SdpaFwdPlan` bundles. So `--check` fails until you install the set, and passes
after you install it. If your set differs, make the overview again with
`feature_matrix.py`. `cases.csv` is `levels.csv` with five columns added
(section 9.4 of the design document).

### 8. Run the engines (GPU)

Run `hipdnn_integration_tests` with the arguments of the ASM_SDPA_ENGINE lane
(a **lane** is one engine's run of the suite), one tier at a time. This command
prints the lane's exact arguments:
`ctest --test-dir $BUILD/dnn-providers/hip-kernel-provider -R '^hip-kernel-provider-asm-sdpa-external-integration_quick_suite$' -N -V`.
For example, on a gfx942 GPU:

```bash
RUNS=$OUT/results/validate/gfx942-job1
mkdir -p $RUNS
$BUILD/bin/hipdnn_integration_tests \
    --test-article $BUILD/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
    --test-engine ASM_SDPA_ENGINE \
    --test-config $REPO/dnn-providers/hip-kernel-provider/config/ASM_SDPA_ENGINE.toml \
    --golden-data-dir $OUT/bundles \
    --gtest_filter='quick_SdpaFwdPlan_*' --gtest_output=xml:$RUNS/quick.xml > $RUNS/quick.log 2>&1
```

Do the same for `standard`, `comprehensive` and `full`. On gfx950, write into
`validate/gfx950-job1`.

- To see the failures that the skips hide, run with a copy of the TOML file
  that ends before the line `# -- SdpaFwdPlan known engine defects --`.
- To see whether a failing case depends on the input data, run it again with
  `HIPDNN_TEST_HOST_INPUT_FILL=1`, into `<tier>-hostfill-<n>.xml`.

Then:

```bash
python3 $TOOLS/analyse.py --results $OUT/results --levels $OUT/levels/levels.csv \
    --placements $OUT/make-bundles/placements.csv --bundle-dir $OUT/bundles --part a
python3 $TOOLS/build_skips.py --validate $OUT/results/validate \
    --levels $OUT/levels/levels.csv --placements $OUT/make-bundles/placements.csv \
    --bundle-dir $OUT/bundles --out-dir $OUT/skips --dry-run
```

`analyse.py` counts the passed, failed, declined, skipped and unverifiable
cases per arch and tier. It maps each failure to a known defect tag. It reports
a failure with no tag as `UNMAPPED`; find the cause of such a failure before you
add a skip (section 10.2 of the design document).

`build_skips.py` without `--dry-run` writes the `[[test_skips]]` entries at the
end of the engine TOML file, after the marker line. It replaces an earlier
section that it wrote. `analyse.py --skips <TOML>` then checks two things:
every failure is skipped under the right reason, and no passing case is
skipped.

The runner can also run every row against one engine, with
`--plugin-path <plugin> --engine <name> --expect-arch <arch>`, into
`results/runner/<arch>-job<id>/<engine>/results.jsonl`. `analyse.py --part b`
summarises those runs.

### 9. Reference check (GPU)

The suite checks each output against the hipDNN GPU and CPU SDPA references.
The reference check runs both references on every selected row and compares
them. It pins an engine that declines every SDPA graph, so no engine runs:

```bash
$BUILD/bin/hipdnn_sdpa_case_flow_runner --input $OUT/select/selected.csv \
    --output $OUT/refcheck/gfx942/rows.jsonl --class-values $OUT/discovery/class-to-value.json \
    --plugin-path $BUILD/lib/hipdnn_plugins/engines/libhip_kernel_provider.so \
    --engine HIP_MLOPS_ENGINE --expect-arch gfx942 \
    --cpu-ref-max-mflop 1e12 --max-tensor-mb 20000 --warmup 0 --iterations 1
python3 $TOOLS/analyse_refcheck.py --rows $OUT/select/selected.csv \
    --placements $OUT/make-bundles/placements.csv --results "$OUT/refcheck/*/rows.jsonl"
```

The CPU reference makes this slow: about 6 hours per arch on one GPU node.
`--start N --limit M` split the rows into parts (shards) that can run at the
same time.

`analyse_refcheck.py` reports, per arch and tier:

- the rows with no result;
- each time a reference refused a row;
- each CPU-vs-GPU disagreement;
- the share of fp8 outputs that differ by one representable value.

For our set, both references accepted and agreed on every row of quick,
standard and comprehensive. In full, both declined the same 60 rows on both
arches.

### 10. Compare your bundles with ours

Our bundles are in commit `3fb4834159ed88c202e3c923ec832ea8358832dc` of
[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries), on the branch
`users/cmiserva/almiopen-2893/sdpa-fwd-integration-tests-plan-bundles`. You do
not need the whole branch. Fetch that one commit, then extract only the four
`SdpaFwdPlan` folders with `git archive`:

```bash
OURS=3fb4834159ed88c202e3c923ec832ea8358832dc
git -C $REPO fetch https://github.com/ROCm/rocm-libraries.git $OURS
mkdir -p $OUT/ours
git -C $REPO archive $OURS \
    dnn-providers/integration-tests/integration-test-bundles/{quick,standard,comprehensive,full}/SdpaFwdPlan \
    | tar -x -C $OUT/ours
python3 $TOOLS/compare_bundle_trees.py \
    $OUT/ours/dnn-providers/integration-tests/integration-test-bundles $OUT/bundles \
    --report $OUT/compare-bundles.txt
```

You can also use `git -C $REPO worktree add <folder> $OURS` to check out the
whole commit in a separate folder. Then give
`<folder>/dnn-providers/integration-tests/integration-test-bundles` as the first
argument.

`compare_bundle_trees.py` prints the number of files on each side, how many
are byte-identical, and how many are identical after it replaces every
`generator_version` value with a placeholder. It ends with `RESULT MATCH` (exit
code 0) or `RESULT MISMATCH` (exit code 1). `--report` also writes the names of
the files that differ.

**The one expected difference.** A match means `RESULT MATCH` with
`different=0`. The only expected difference is the `generator_version` value
in each case of each `sweep.json`:

- Our bundles carry `fadf8dc7c3ba`.
- A runner built from this branch writes `bd5cd80976f0`. If one of the three
  source files changes again, this value changes too.

The reason: `generator_version` is a fingerprint of the runner's source files
(see [The two C++ programs](#the-two-c-programs)), not of the graphs. After we
made our bundles, these source files changed:

- the runner got a later change to its reference comparison (element
  statistics for the reference check);
- the runner accepts any output folder;
- the three files were formatted with clang-format.

None of these changes the exported graphs. But they change the source files, so
they change `generator_version`. Every other byte of your set is expected to be
equal to ours.
