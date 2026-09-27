# Validating a packaged hipBLASLt build

This page is for someone who has installed a ROCm package and wants to check that hipBLASLt works
on their machine. Everything here runs from the installed files. You do not need to download source
code, build anything, or consult any other page.

hipBLASLt is a library that multiplies matrices on AMD graphics processors. A general matrix
multiply, usually shortened to GEMM, computes `D = alpha * A * B + beta * C`, optionally applying a
bias and an activation function to the result. Frameworks such as PyTorch call hipBLASLt for this
operation, so a hipBLASLt problem usually shows up as wrong numbers or poor performance in a much
larger program. Checking the library directly is faster than working backwards from there.

The package ships the same test program that AMD runs before releasing hipBLASLt, together with the
test definitions it reads. Running it compares hipBLASLt's results against the same calculation
performed on the host processor, so a failure means the two disagreed numerically rather than only
that something crashed.

If you are a hipBLASLt contributor looking for the testing strategy, which automated checks block a
merge, or where the coverage gaps are, read [TESTING.md](TESTING.md) instead.

## Contents

- [What your host needs](#what-your-host-needs)
- [Where the files are](#where-the-files-are)
- [Which command to run](#which-command-to-run)
- [How to tell whether it passed](#how-to-tell-whether-it-passed)
- [Reading the output when something is wrong](#reading-the-output-when-something-is-wrong)
- [Optional performance spot checks](#optional-performance-spot-checks)

## What your host needs

**An AMD graphics processor that this package carries kernels for.** hipBLASLt loads precompiled
kernels chosen for your specific processor architecture, and each package is built for a named set
of architectures. List the ones your copy contains:

```bash
ls <rocm>/lib/hipblaslt/library/
# gfx942
```

Throughout this page, `<rocm>` means the directory holding the `bin`, `lib` and `share`
subdirectories of your ROCm installation. If you installed from a package manager it is usually
`/opt/rocm`. If you unpacked an archive, it is the directory you unpacked into.

Compare that listing against your own processor, which `rocminfo` reports on the `Name` line of each
agent. If your architecture is absent, this package has no kernels for your hardware and the tests
cannot pass, regardless of anything else on this page.

**A working driver.** The tests open the graphics processor directly. If no device is available, the
tests report themselves as skipped rather than failed, which is easy to mistake for success. See
[How to tell whether it passed](#how-to-tell-whether-it-passed).

**CMake, for the recommended command.** The package registers its test levels with CTest, the test
driver that ships with CMake, but it does not include CMake itself. Either install CMake, or use the
alternative command in [Which command to run](#which-command-to-run), which needs nothing beyond the
package. The registration uses only long-standing CTest features, so any CMake your distribution
packages will run it.

**No environment variables, in the common case.** The programs locate their own libraries and their
own test data relative to where they are installed, so a plain invocation works from any working
directory. Two exceptions are worth knowing about before you need them:

- If the loader cannot find `libhipblaslt.so`, add the package's library directory:
  `export LD_LIBRARY_PATH=<rocm>/lib:$LD_LIBRARY_PATH`.
- `HIPBLASLT_TENSILE_LIBPATH` overrides where the library looks for its precompiled kernels. You
  only need it if the kernel directory has been moved out of the package. It is ignored when the
  process is running with elevated privileges, which is deliberate.

## Where the files are

All paths below are relative to `<rocm>`.

| Path | What it is |
| --- | --- |
| `bin/hipblaslt-test` | The test program. Runs GEMM problems and checks each result. |
| `bin/hipblaslt_gtest.data` | The test definitions the program reads. It finds this file next to itself. |
| `bin/hipblaslt/CTestTestfile.cmake` | Registers the named test levels so `ctest` can run them. |
| `bin/hipblaslt-bench` | Measures speed for one problem or a list of problems. |
| `bin/hipblaslt-perf` | Runs groups of measurements and writes a spreadsheet of results. |
| `lib/hipblaslt/library/<arch>/` | The precompiled kernels, one directory per processor architecture. |
| `libexec/hipblaslt-samples/` | Small standalone example programs. |
| `.info/version` | The ROCm version of this installation, for example `10.2.0`. |

The package also carries narrower programs whose names begin `bin/hipblaslt-bench-extop-`, which
measure individual operations such as layer normalization, and `bin/hipblaslt-api-overhead`, which
measures the cost of the library's own function calls rather than of the arithmetic. Neither is part
of checking that an installation works.

If `bin/hipblaslt-test` is absent, your installation does not include the test files. They are
distributed separately from the libraries: as a package named `amdrocm-blas-test`, and in archives
whose names contain `-tests-`. Install one of those and repeat.

### Confirming the tests match the library

Test programs and libraries installed from different builds can disagree in ways that look like
library defects. `bin/hipblaslt-test` guards against this by asking the library it actually loaded
which version it is, and printing the answer twice, once when it starts and once when it finishes.
For example:

```text
hipBLASLt version: 100500
hipBLASLt git version: 3f1c2ab
```

The integer encodes the version as `major * 100000 + minor * 100 + patch`, so `100500` is version
1.5.0. The second line is the source revision the library was built from.

Both lines describe the loaded library rather than the test program, so they tell you what was
actually exercised. If you installed everything from one archive or one package version, they agree
by construction. If you mixed sources, note that the `amdrocm-blas-test` package does not require a
matching version of the library package, so the package manager will not catch the mismatch for you.

## Which command to run

The package defines four test levels. They are cumulative: each one runs everything the level below
it runs, plus more.

| Level | What it adds | Expected duration | Time limit |
| --- | --- | --- | --- |
| `quick` | A small set of checks across the common data types | about 5 minutes | 600 seconds |
| `standard` | Wider coverage of data types, shapes and fused operations | about 30 minutes | 3600 seconds |
| `comprehensive` | The cases AMD runs nightly rather than per change | about 2 hours | 7200 seconds |
| `full` | Cases that require a host supporting managed memory, where the processor and the processor's host share one address space | up to 24 hours | 86400 seconds |

Start with `quick`. It is the level to use for "did this installation come up correctly." Use
`standard` when you want a result you would act on before deploying. The two longer levels exist for
release qualification and rarely earn their runtime otherwise.

Run a level from the directory holding the CTest file. The registered command uses a relative path
to reach the test program, so the working directory matters:

```bash
cd <rocm>/bin/hipblaslt
ctest -L quick --output-on-failure --output-junit hipblaslt-quick.xml
```

`-L` selects the level. `--output-on-failure` prints the failing case's own output, which is
otherwise hidden. `--output-junit` writes results in the JUnit XML format that most automation and
reporting tools already read.

A fifth level, `multi_gpu`, is registered but currently has no cases assigned to it. Running it
completes immediately and tells you nothing.

### Without CMake

The same work runs directly, using the same selection the `quick` level uses:

```bash
<rocm>/bin/hipblaslt-test --gtest_filter='*smoke*-*known_bug*' --gtest_output=xml:hipblaslt-quick.xml
```

The filter has two halves separated by `-`: patterns to include, then patterns to exclude. The other
levels use these include patterns, with the same `-*known_bug*` exclusion on each:

- `standard`: `*smoke*:*quick*:*pre_checkin*`
- `comprehensive`: `*smoke*:*quick*:*pre_checkin*:*nightly*`
- `full`: `*smoke*:*quick*:*pre_checkin*:*nightly*:*HMM*`

Expect a pause of a minute or two before the first result appears. The program parses its test
definitions first, and prints `info: parsing of test data may take a couple minutes before any test
output appears...` while it does.

## How to tell whether it passed

**The short answer.** `ctest` returns 0 and reports `100% tests passed`. Run directly, the test
program returns 0 and prints a `[  PASSED  ]` line with no `[  FAILED  ]` block above it.

**A failure is a numerical disagreement.** Each case computes a GEMM on the graphics processor and
the same GEMM on the host processor, then compares the two within a tolerance. A `[  FAILED  ]` line
therefore means the results differed by more than that tolerance, or the library rejected a call it
is supposed to accept. Either is worth reporting.

**Skipped cases are normal and are not failures.** The test program counts them and prints a tally
when it finishes, alongside a line for each one as it happens:

```text
[ SKIPPED  ] 12 tests.
```

Four reasons account for nearly all of them, and each prints its own line:

- `Skipped test due to limited memory environment.` The case asked for more memory than the device
  has. Expected on cards with less memory than the test author assumed.
- `Skipped test due to too few GPUs.` The case needs more devices than are visible.
- `Skipped known bug for current platform.` The case reproduces a defect already recorded against
  your architecture, so it is not run as a pass or fail signal.
- `No GPU available`, from cases that need a device when none was found. If you see this on every
  case, nothing was tested. Treat a run that skips everything as a failed check, not a pass.

**Cases recording known defects never run by default.** Every level excludes them, which is what
`-*known_bug*` does in the filters above. You will only meet them if you remove that exclusion.

**What to collect if something fails.** The failing case name exactly as printed, the level you ran,
your processor architecture from `lib/hipblaslt/library/`, the version lines the program printed, and
the XML file. The case name encodes the problem that failed, so it is the single most useful thing to
include.

## Reading the output when something is wrong

Two messages mean the installation is wrong rather than the library being broken. Both appear on
standard error, and both are worth recognizing because the failure that follows does not name its
cause.

**No kernels for this processor.**

```text
rocblaslt warning: No paths matched <path>. Make sure that HIPBLASLT_TENSILE_LIBPATH is set correctly.
```

The library looked for precompiled kernels and found none for your architecture. Check that
`lib/hipblaslt/library/` contains a directory matching your processor, as described in
[What your host needs](#what-your-host-needs).

**Kernels present but unloadable.**

```text
hipModuleLoad failed: <path>
```

The file exists and the driver refused it, usually because it was built for a different architecture
or the installation was rearranged after it was unpacked. The library has no fallback path here, so
this appears at the first GEMM rather than at startup.

**Everything failing at once** points the same direction. A genuine hipBLASLt defect usually affects
a recognizable family of cases: one data type, one shape, one fused operation. Total failure is
much more often a mismatched or incomplete installation.

**An empty command from `ctest`** means the test program is missing. If `ctest -V` shows a test
command with only arguments and no program, and mentions paths such as
`../RelWithDebInfo/hipblaslt-test`, then `bin/hipblaslt-test` is not where the CTest file expects it.
Confirm the file exists and that you are running from `<rocm>/bin/hipblaslt`.

Result files land in the directory you ran from: the JUnit XML at whatever path you passed, and
CTest's own logs under `Testing/Temporary/`, where `LastTest.log` holds the full output of the most
recent run.

## Optional performance spot checks

**A single measurement is not a pass or fail criterion.** Speed depends on clock behavior, cooling,
what else is using the device, and the specific problem shape. Treat these commands as a way to see
that performance is in a plausible range, and compare against your own earlier measurements on the
same machine rather than against a number from elsewhere.

One problem, with the result checked against the host processor:

```bash
<rocm>/bin/hipblaslt-bench --precision f32_r -v
```

The output is comma separated, one header line and one row per problem. Most columns describe the
problem. Four describe the outcome: `hipblaslt-Gflops` is billions of floating-point operations per
second, `hipblaslt-GB/s` is memory throughput, `us` is elapsed microseconds, and with `-v` the
`norm_error` column reports how far the result sat from the host processor's answer.

`bin/hipblaslt-perf` repeats measurements and aggregates them, which gives a more stable figure than
a single run. Running it from an installed package needs two adjustments, because its defaults assume
a source tree:

```bash
pip install GitPython
<rocm>/bin/hipblaslt-perf -e <rocm>/bin -w ./perf-output --suite example --samples 1
```

`-e` points it at the programs and `-w` at a writable output directory. The GitPython dependency is
imported unconditionally and is not installed with the package. Only the problem sets built into the
tool are available; the additional ones kept with the source are not part of the package.
