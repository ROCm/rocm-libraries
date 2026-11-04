# gtest conventions

Full test-naming rules live on the wiki: https://github.com/ROCm/MIOpen/wiki/GTest-development#naming
(`Smoke`/`Standard`/`Full`/`Perf`/`Unit` prefixes, `CPU`/`GPU` hardware token, datatype
suffix). `check_names.py` enforces that schema against `miopen_gtest --gtest_list_tests`.

## hipDNN shim surface

A test belongs to the hipDNN backend-swap surface if its full gtest name contains the
token `HipdnnShim`. Parameterized tests get it from the instantiation prefix, e.g.
`INSTANTIATE_TEST_SUITE_P(HipdnnShim, GPU_ConvFwdApi_FP32, ...)`. Non-parameterized tests
carry it directly in the suite name instead, e.g. `TEST_F(GPU_HipdnnShimConvFwdApi_FP32, ...)`,
since `check_names.py`'s prefix check only applies to the `Smoke`/`Standard`/`Full`/`Perf`/
`Unit` token used by parameterized instantiations.

The token is deliberately specific: it is a filter, so a generic word like `Forwarding`
would sooner or later pull in an unrelated test and silently double its runtime.

Select the surface with `--gtest_filter='*HipdnnShim*'`. That filter is what the
`forwarding_parity` ctest entries replay under `MIOPEN_HIPDNN_FORWARDING=disabled` and
`=enabled`; a test outside the surface is never replayed, and a test inside it that is not
reachable through public entry points makes the comparison meaningless.

Two consequences of that being the only entry path, both easy to "fix" in the wrong
direction:

- **Shim-surface tests stay out of `test_categories.yaml` on purpose.** The categorized
  smoke/standard/full entries turn those patterns into a `--gtest_filter`, so a matching
  pattern would register a third run of tests the parity entries already run twice. Absence
  from that file is the design, not a gap.
- **Excluding a shim-surface test means narrowing the parity filter,** not calling
  `add_gtest_negative_filter`. That function feeds the shard filter, which the parity entries
  do not use, so disabling a shim test through it looks like it worked and changes nothing.

To qualify, a shim-surface test must:
1. Reach compute only through a public `miopen.h` entry point — never through
   `miopen::solver::` or `ProblemDescription` directly.
2. Validate against an independent CPU or analytically-known reference, not a
   self-comparison.
3. Use a tolerance appropriate for cross-implementation comparison, not
   bit-reproducibility.

See `hipdnn_shim_conv.cpp` for worked examples.

### Tolerances

Every shim test that checks a floating-point result does it through `CheckWithinTolerance` in
`hipdnn_shim_helpers.hpp`, which fails when `miopen::rms_range(reference, result)` reaches
`Tolerance<T>()`. `rms_range` is the root-mean-square difference divided by the largest
magnitude in the reference, so the threshold is relative to the size of the output. `T` is
the tensor's element type, so the threshold depends on the data type, not on the operation:

| Type | Threshold |
|---|---|
| `float` | `80 * epsilon`, about 9.5e-6 |
| `half` | `80 * epsilon`, about 7.8e-2 |
| `bfloat16` | `4 * epsilon`, about 3.1e-2 |

These are the values `ConvFwdSolverTestBase::ThresholdChecks()` uses for MIOpen against the
CPU reference. They have not been re-derived for MIOpen against hipDNN. The tests do not rely
on the threshold being generous: inputs are whole numbers small enough that the exact result
is representable, so the threshold only has to cover rounding the kernel adds. Per operation
family:

- **Convolution forward, backward data and backward weights, `float`.** Inputs are whole numbers
  below 17 (below 13 for `dy`), and every sum is at most about 4e4, far below 2^24. A kernel
  that adds the products directly is therefore exact. Winograd and FFT algorithms round in
  their transforms, and `EveryFoundAlgorithmMatchesCpuReference` runs every algorithm Find
  returns, so that test is where the `float` threshold is exercised.
- **Convolution in `half` and `bfloat16`** (`hipdnn_shim_conv_translation.cpp`). Inputs are
  whole numbers below 5, so that a backward-weights sum cannot pass `half`'s maximum of 65504.
  Forward sums reach 2304. `half` holds every whole number to 2048, so only the top of that
  range is rounded. `bfloat16` holds whole numbers only to 256, so most outputs are rounded
  once when stored (at most about 2^-8 relative), which the `bfloat16` threshold absorbs.
- **Fused convolution, bias and ReLU, `half`.** Inputs are whole numbers in [-4, 4] and every
  product and partial sum is at most 2048, so the expected result is exact in `half`. A
  disagreement beyond the threshold is a real one, not accumulated rounding.
- **Grouped, transposed and rank-varied convolution, scaled convolution.** `float` data, whole
  numbers below 17, checked with the `float` threshold.
- **`int8` input.** There is no tolerance: the result is integer-valued and compared for
  equality.

If a new shim test needs a different threshold, add it next to the case that needs it with
the reason, rather than raising `Tolerance<T>()` for every test.

## Forwarding parity harness

With `MIOPEN_ENABLE_HIPDNN_WRAPPER=ON`, each `*_forwarding_parity` ctest entry runs
`script/run_forwarding_parity.py`. It replays the shim surface twice, once with
`MIOPEN_HIPDNN_FORWARDING=disabled` and once with `=enabled`, and requires the two runs to
agree test for test. The runs can differ only for entry points listed in
`kForwardingEntries` in `src/private/routing.cpp`, so a shim test that reaches an entry point
outside that list runs the same code twice and checks nothing.

"Agree" means the same set of tests with the same outcome in both runs. It does not mean
the two runs produced bit-identical output. Each test judges its own result against a CPU
reference within the tolerance above, so parity is "no regression under tolerance". Where
one run serves a call and the other declines it (or the reverse), the test records that with
a `parity_*` property, and `known_forwarding_divergences.txt` lists the gaps that are
accepted for now (see "Known divergences" below).

### Adding an entry point to the forwarding set

A change that adds an entry point to `kForwardingEntries` needs all of these, and reviewers
should check each one:

1. A hipDNN implementation of the entry point. Until it has one, its generated stub in
   `wrapper_passthrough.cpp` sends a forwarded call to `hipdnn::ForwardingNotImplemented`, which
   fails with `miopenStatusNotImplemented` instead of running MIOpen. To add the implementation,
   move the stub into `wrapper.cpp`, replace its dispatch line with a `MIOPEN_WRAPPER_FORWARD`
   call to the new function (as the convolution stubs do), and rerun the generator (see
   "Wrapper sources are generated" below).
2. A test in the shim surface (suite name contains `HipdnnShim`) that reaches the entry point
   through the public `miopen.h` API, checks against a CPU or analytically-known reference
   within tolerance, and checks that the call was actually served. See "hipDNN shim surface"
   above for what qualifies.
3. A tolerance note for the new operation family in "Tolerances" above, if its data or
   accumulation differs from the families listed there.
4. An update to `DefaultSetIsTheConvolutionFamily` in `wrapper_routing.cpp`. It makes the
   addition deliberate; it does not check that item 2 exists.
5. A `known_forwarding_divergences.txt` line for any case where forwarding serves a call that
   MIOpen declines, or the reverse.

Why it is set up this way:

- **Only the shim surface is replayed.** The rest of the suite reaches compute through
  internal headers that bypass the wrapper, so replaying it would cost two full runs to prove
  nothing, and any flaky test in it would look like a real divergence.
- **One ctest entry, not three tied together by a fixture.** A sharded ctest run hands out
  whole entries, so the members of a fixture can land in different shards and fail there as
  unsatisfied. The script runs the two replays and the comparison itself, and stops before the
  comparison if a replay fails.
- **The gtest shard variables are pinned to one shard.** A sharded CI run exports
  `GTEST_TOTAL_SHARDS`/`GTEST_SHARD_INDEX`, and a replay that honoured them would compare only
  one shard's slice of the shim surface, with no other entry covering the rest.
- **One bare entry plus one per `ex_gpu_*` label.** CI selects tests by a tier label combined
  with an architecture filter, either `-L ^ex_gpu_<arch>$` on an architecture that
  `test_categories.yaml` declares or `-LE ex_gpu` elsewhere. No single entry survives both, so
  exactly one of these is selected under each. `ForwardingParityGpuLabels.cmake` takes the
  labels from the shared parser's output, so an architecture gets a parity entry exactly when
  the parser registers an enabled test for it, and stops the configure if the parser fails.
  `forwarding_parity` goes on the bare entry only, so `ctest -L forwarding_parity` replays the
  surface once.
- **Tier labels are `quick`, `standard`, `comprehensive` and `full`,** so the harness runs in
  every tier lane. Not `ffm-quick`/`ffm-full`: those run a fixed list of patterns on a tight
  time budget.
- **In a discrete build, only the `test_hipdnn_shim_conv*` binaries get entries.** Any other binary would
  replay a filter that matches nothing, and the comparison rejects two empty runs.
- **No build-tree entry without a GPU.** The shim tests are all `GPU_` tests.

The packaged test list (`bin/MIOpen/CTestTestfile.cmake`) is generated from
`test_categories.yaml` and does not see `add_test()` calls, so the harness entries and
`wrapper_abi_check` are written into it separately. Without that, `ctest -L forwarding_parity`
on a packaged build would select nothing and pass. The packaged entries are not tied to
`MIOPEN_NO_GPU`, which describes the build machine; packages are often built without a GPU and
tested elsewhere. Only the single-binary build (`MIOPEN_TEST_DISCRETE=OFF`) ships them.

With the flag on, the packaged tests' scripts, data files, `forwarding_failure_probe` and the
fake hipDNN backend install under `bin/miopen_wrapper_tests/`, and the test list refers to
them as `../miopen_wrapper_tests/…`. `miopen_gtest` stays in `bin/`. The files share one
folder so that packagers can keep the fake backend, which has the real backend's file name,
out of runtime packages with a single pattern. TheRock's `ml-libs/artifact-miopen.toml` names
the folder; change both together.

`wrapper_abi_check` (`script/check_wrapper_abi.py`) checks the wrapper's exported ABI from the
two built libraries. It also checks the wrapper's list of required shared libraries
(`DT_NEEDED`): `libMIOpen_private` must be on it and `libhipdnn_backend` must not, so the
hipDNN backend is opened on first use rather than required at load time. The full list is
also committed as `test/public_abi/wrapper_needed.baseline`, but the ctest entry does not
compare against it, because it names ROCm soversions and the x86-64 loader and would fail on
every ROCm update. To compare it by hand, pass `--needed-baseline
test/public_abi/wrapper_needed.baseline` to `script/check_wrapper_abi.py`. It loads neither library, so it needs
no GPU and is registered whenever the flag is on. The packaged copy leaves out `--public-header`, which compares two source files that
the build-tree entry already checks.

`test_forwarding_parity_scripts` tests the harness scripts themselves. It needs no GPU or
build output, so it carries only the tier labels. It uses `unittest` rather than `pytest`,
because nothing installs `pytest` on a machine that builds MIOpen.

Two cache variables tune the harness:

- `MIOPEN_FORWARDING_PARITY_FILTER` (default `*HipdnnShim*`): the gtest filter replayed.
- `MIOPEN_FORWARDING_PARITY_TIMEOUT` (default `3600`): seconds per entry, covering both
  replays and the comparison.

### Saying more than pass or fail

A test that reaches an ending the gtest verdict does not distinguish should record it with
`RecordProperty("parity_<something>", ...)`. gtest writes the property into the `<testcase>`
element, and `compare_forwarding_runs.py` folds every `parity_*` property into the outcome it
compares. The fused convolution test uses this: it passes whether the library
computed a result or declined the problem, and without the property the two replays look like
they agreed when one of them did no work.

Record a property from within the test's own execution, keyed so that two cases in one test
cannot overwrite each other — gtest replaces a property recorded twice under the same key.

### Known divergences

`known_forwarding_divergences.txt` lists divergences that are accepted for now. Each line names
a test, the devices the gap is open on (`all`, or a comma-separated list of gfx names such as
`gfx90a,gfx942`), the outcome expected from each replay, and why the gap exists. A matching
divergence is printed and tolerated; anything else still fails.

The device a run used comes from the test binary itself: a shim test binary records MIOpen's
device name as the `forwarding_parity_device` property of its report. The comparison stops if the
two reports name different devices. It also stops if neither report names a device and the list
has entries. In that case no line can match the run.

Three rules make the list safe to have:

- A line applies only to runs on a device it lists. On any other device it tolerates nothing, so
  a gap that shows up on a new device fails until someone checks it there.
- A line that applies to the run but whose test no longer produces exactly those outcomes fails
  the comparison, which asks for it to be updated or deleted. The list therefore describes gaps
  that are open today rather than accumulating history. A line whose test is not in the run at
  all is ignored, because several test binaries share the list.
- A run that tolerates a line says so in its output and does not print the ordinary success
  line, so a known divergence never reads as a clean pass.

To add one, run the harness, copy the failing line's test name, the device it ran on, and both
outcomes into the file, and write down the reason. The list's own rules are covered by
`test_forwarding_parity_scripts`, or by hand with
`python3 -m unittest test_compare_forwarding_runs` from `script/`; neither needs a build or GPU.

## Wrapper sources are generated

`include/miopen/miopen.h` is the only place a public entry point is written by hand. Four files
copy its list of entry points, and `script/generate_wrapper_sources.py` writes them:
`src/private/miopen_private_rename.h`, `src/private/miopen_impl.h`,
`src/private/wrapper_passthrough.cpp`, and the provider's `MiopenApiPrivateRename.hpp`. They are
committed, so a change to the public API shows up in the PR diff. Do not edit them by hand.

After you change `miopen.h`, run this from the repository root and commit the files it changes:

```shell
python3 projects/miopen/script/generate_wrapper_sources.py
```

If the pinned `clang-format` release is not installed, run
`pre-commit run --hook-stage manual miopen-wrapper-sources-regenerate --all-files` instead. It
installs that release for you.

- `generate_wrapper_sources.py --check` (a pre-commit hook, and a step in MIOpen's component CI)
  fails when a generated file was edited by hand or is out of date.
- `check_public_abi.py check-headers` covers the parts that are still written by hand: the stubs in
  `wrapper.cpp`, the wrapper's own functions in `include_wrapper/miopen/hipdnn_forwarding.h`, and
  the provider's declarations of the three `miopenConvolution*GetWorkSpaceSizeRange` functions.
- The generator exits with an error naming the line instead of skipping a declaration it cannot
  read exactly, such as an entry point inside an `#if`, or an unnamed or array parameter.
- `wrapper.cpp` holds the stubs that do more than pass a call through. The generator skips every
  entry point defined there, and refuses one whose signature differs from `miopen.h`.

## Forwarding failure modes

`forwarding_failure_modes` checks what a flag-ON build does when forwarding is turned on but
hipDNN cannot be used: the status each call returns, the lines printed to stderr, and what
`miopenGetLastForwardedError` and `miopenGetErrorString` report. It also checks the routing trace
and that `MIOPEN_DISABLE_HIPDNN_FOR` keeps calls off a broken hipDNN. The switches involved are
read once per process, so each case runs a fresh process:

- `forwarding_failure_probe` links only the public `libMIOpen.so`, as an application would. It
  runs the steps named on its command line, such as `conv_forward` or `last_forwarded`, and prints
  one result line per step.
- A fake hipDNN backend stands in for a broken installation. It has the real backend's file name
  but no version, so hipDNN loads it and then refuses it. Every case points
  `HIPDNN_BACKEND_LIBRARY_PATH` at it, so no case can reach a real backend. It prints a marker
  line to stderr when it is loaded, which shows whether a case loaded it at all.
- `script/test_forwarding_failure_modes.py` is the `unittest` module that runs the probe under
  each environment and checks its output.

Run it from the build directory with `ctest -R forwarding_failure_modes`. It needs no GPU.

A missing backend cannot be tested directly. When `HIPDNN_BACKEND_LIBRARY_PATH` names a directory
without the backend, hipDNN moves on to its next search location and may find the real backend
there. The fake is the closest stand-in: hipDNN refuses it, which takes MIOpen down the same path
as a missing backend.

The fake is built as a CMake `MODULE` library, not `SHARED`. On Windows a `SHARED` library's DLL
is written next to `MIOpen.dll`, where hipDNN looks for its backend, so every other test would
load the fake. A `MODULE` library goes to its own directory on every platform.
