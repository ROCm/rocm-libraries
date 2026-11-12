---
name: therock-local-build
description: "Build and test a rocm-libraries artifact (hipDNN, a hipDNN provider, ...) locally against prebuilt ROCm by bootstrapping artifacts from a baseline TheRock CI run instead of building the whole stack. Use when a full TheRock build is too slow, you want to rebuild only hipDNN or hipkernelprovider, or you need a quick local build on top of CI's ROCm."
argument-hint: "[artifact(s) to rebuild]"
allowed-tools: Bash, Read, Glob
---

# Build part of ROCm locally on baseline CI artifacts

TheRock CI reuses artifacts from an earlier run (the **baseline run**) and compiles only
what changed. This skill does the same locally. The argument is the artifact list you pass
to `--rebuild` in step 1.

## Concepts

- **Subproject**: a CMake project built by TheRock. It installs into `<dir>/stage/`.
- **Artifact**: a named set of subproject outputs, published as one archive per
  **component** (`lib`, `run`, `dev`, `dbg`, `doc`, `test`) and GPU target.
- **Stage**: a group of artifacts built together (`math-libs` builds hipDNN and its
  providers). List them with `build_tools/artifact_manager.py list-stages`.
- **Inbound artifacts**: artifacts a stage needs that other stages produce.
  **Same-stage dependencies** are produced by the stage itself.
- **Bootstrap**: extract a baseline run's artifacts into the build directory and create
  a `<dir>/stage.prebuilt` marker beside each. CMake skips any subproject with that marker.
- `configure_stage.py --artifacts X` enables only the CMake features for `X`
  (`-DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_X=ON`). An enabled subproject without a
  marker is compiled.

So: bootstrap everything you reuse and enable only what you rebuild. `fetch --stage S`
bootstraps inbound artifacts only, so same-stage dependencies need the planner in step 1.

## Prerequisites

Linux x86-64 with `git`, `cmake`, `ninja`, `curl`, the GitHub CLI `gh` (logged in with
`gh auth login`), Python 3 (tested with 3.12), network access (artifact
bucket, third-party source tarballs, github.com) and local disk space (not a
network filesystem: configure runs `git` in the source checkouts and can stall). Budget
about 15 GB: 4 GB for the build directory and 11 GB for the rocm-libraries checkout.

Run the snippets in order in one shell session; later steps use the variables set earlier.

| Variable | Meaning |
|---|---|
| `RUN` | Baseline run id: a ROCm/TheRock `main` "Multi-Arch CI" run that built artifacts |
| `T` | TheRock checkout at the baseline run's head SHA |
| `RL_SRC`, `RL` | An existing rocm-libraries clone, and the checkout of it at the commit TheRock pins |
| `V` | Python venv for TheRock's scripts |
| `FAM`, `STAGE` | GPU family (`gfx950-dcgpu`) and the stage producing your artifact |
| `B` | Build directory |
| `CPUS`, `N` | CPU list the build may use (`8-15`) and its size (`8`) |
| `SKILL_DIR` | Directory containing this file |

1. **Baseline run.** List recent `main` runs and take the first whose bucket has your GPU
   target (`rand` is target-specific; replace `gfx950` in the check with your target).
   Runs whose setup skipped the build publish
   nothing; a failed or cancelled run is fine if the artifacts exist. Old runs expire.
   ```bash
   for R in $(gh run list -R ROCm/TheRock --branch main -L 20 --json databaseId,workflowName \
       -q '.[] | select(.workflowName=="Multi-Arch CI") | .databaseId'); do
     if curl -s "https://therock-ci-artifacts.s3.amazonaws.com/?prefix=$R-linux/rand_lib_gfx950&max-keys=1" \
         | grep -q '<Key>'; then RUN=$R; break; fi
   done; echo "RUN=${RUN:?no baseline run found}"
   ```
2. **TheRock and venv.**
   ```bash
   SHA=$(gh run view $RUN -R ROCm/TheRock --json headSha -q .headSha)  # or the API's head_sha
   git clone https://github.com/ROCm/TheRock.git $T && git -C $T checkout $SHA
   python3 -m venv $V && $V/bin/pip install -r $T/requirements.txt
   export PATH=$V/bin:$PATH      # CMake needs meson from the venv
   ```
3. **rocm-libraries at TheRock's pin.** The artifacts were built from this commit; newer
   sources can fail to compile against them (see Troubleshooting).
   ```bash
   PIN=$(git -C $T ls-tree HEAD rocm-libraries | awk '{print $3}'); echo "${PIN:?TheRock checkout missing}"
   git -C $RL_SRC worktree add --detach $RL $PIN
   ```
4. **`rocm-systems/shared/kpack`.** Target-specific artifacts such as `hipkernelprovider`
   are split by a tool in this directory. Check out only it (10 MB):
   ```bash
   SYS=$(git -C $T ls-tree HEAD rocm-systems | awk '{print $3}')
   git -C $T/rocm-systems init -q
   git -C $T/rocm-systems remote add origin https://github.com/ROCm/rocm-systems.git
   git -C $T/rocm-systems sparse-checkout set --cone shared/kpack
   git -C $T/rocm-systems fetch -q --depth=1 --filter=blob:none origin $SYS
   git -C $T/rocm-systems checkout -q FETCH_HEAD
   ```

## Build

Run `artifact_manager.py` and `configure_stage.py` from `$T`.

1. **Plan.** The planner lists what to bootstrap and which same-stage dependencies must be
   rebuilt too. It prints two `export` lines on stdout; commentary goes to stderr:
   ```bash
   eval "$(python3 $SKILL_DIR/scripts/plan_bootstrap.py --therock $T --stage $STAGE \
     --rebuild hipkernelprovider --run-id $RUN)"
   echo $REBUILD_ARTIFACTS    # hipdnn-integration-tests,hipkernelprovider
   ```
2. **Fetch and bootstrap** (about 3 GB). Archives are named per GPU target;
   `--expand-family-to-targets` maps `gfx950-dcgpu` to `gfx950`. Quote several families
   (`'gfx94X-dcgpu;gfx950-dcgpu'`).
   ```bash
   cd $T
   $V/bin/python build_tools/artifact_manager.py fetch --run-id $RUN \
     --run-github-repo ROCm/TheRock --stage all --amdgpu-families $FAM \
     --expand-family-to-targets --output-dir $B --bootstrap \
     --exclude-components dbg,doc,test --exclude-artifacts "$EXCLUDE_ARTIFACTS" \
     --download-concurrency 4 --extract-concurrency 4
   ```
3. **Configure.** Print the feature flags, then pass them to CMake:
   ```bash
   $V/bin/python build_tools/configure_stage.py --artifacts "$REBUILD_ARTIFACTS" \
     --amdgpu-families $FAM --dist-amdgpu-families $FAM --oneline
   cmake -S $T -B $B -GNinja -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=$RL \
     -DPython3_EXECUTABLE=$V/bin/python \
     -DTHEROCK_AMDGPU_FAMILIES=$FAM -DTHEROCK_DIST_AMDGPU_FAMILIES=$FAM \
     -DTHEROCK_ENABLE_ALL=OFF \
     -DTHEROCK_ENABLE_HIPKERNELPROVIDER=ON -DTHEROCK_ENABLE_HIPDNN_INTEGRATION_TESTS=ON
   ```
   The last line holds the `-DTHEROCK_ENABLE_*=ON` flags the first command prints for this
   example; use the ones it prints for your artifacts. `-DPython3_EXECUTABLE` makes the
   artifact splitter use the venv (it imports `msgpack`).
4. **Build**, limited to `N` CPUs (see Limit threads):
   ```bash
   CMAKE_BUILD_PARALLEL_LEVEL=$N nice -n 10 taskset -c $CPUS \
     ninja -C $B -j $N artifact-hipkernelprovider
   ```
   Use `artifact-<name>` for one artifact, or `stage-$STAGE therock-artifacts` for every
   enabled one. The installed tree is `$B/dist/rocm`; packaged artifacts are in
   `$B/artifacts` (for `hipkernelprovider`: `hipkernelprovider_{dbg,lib,test}_generic`). To rebuild a subproject from scratch run `ninja -C $B <Subproject>+expunge`,
   then `cmake -S $T -B $B`, then build.

   `dist/rocm` holds the rebuilt subprojects and what they pull in, but it can lack
   bootstrapped libraries that only show up at run time (observed when rebuilding
   `miopen`: no `librocblas`, `libhipblaslt` or `librocm_sysdeps_*` in `dist/rocm`, so
   every binary failed to load). Check the loader closure (Verify, step 2) before
   running any test. Every bootstrapped subproject also installs to its own
   `<subproject>/stage` directory, at any depth (sysdeps are at
   `third-party/sysdeps/linux/<lib>/build/stage`), so a complete tree can be assembled
   by copying all `stage` directories, rebuilt ones first, without overwriting:
   ```bash
   S=$PWD/rocm-merged; mkdir -p $S
   for st in $B/math-libs/composable_kernel/stage $B/ml-libs/*/stage \
             $(find $B -type d -name stage -not -path '*/stage/*' | sort); do
     [ -d "$st" ] && cp -an "$st"/. "$S"/ 2>/dev/null
   done
   ```
   Put the subprojects you rebuilt in the first list so their files win.

## Limit threads

`ninja -j` limits only the outer build. Each subproject runs its own `cmake --build` on
every visible CPU. Set both `taskset -c $CPUS` (ninja's default `-j` follows CPU affinity)
and `CMAKE_BUILD_PARALLEL_LEVEL=$N`. Capping makes the build slower; size `N` to the machine.

## Verify

1. After step 3, list what will compile. `.../build` means the subproject compiles;
   `.../stage` alone means the prebuilt copy is used.
   ```bash
   ninja -C $B -t commands artifact-hipkernelprovider | \
     grep -oE '(ml-libs|math-libs)/[A-Za-z0-9_+-]+/(stage|build)' | sort -u
   ```
   Expected: `build` for `hipdnn_integration_tests` and `hipkernelprovider` only.
2. After step 4, check that every shipped library and executable resolves its
   dependencies. The planner trusts `BUILD_TOPOLOGY.toml`, which misses run-time links
   (`libhipblas` needs `librocsolver`, which needs `librocsparse`; `blas` lists neither),
   so excluded artifacts can leave unresolved libraries. Run from the tree you test
   (`$B/dist/rocm` or the merged tree above):
   ```bash
   R=$B/dist/rocm   # or $S
   for f in $(find $R/lib $R/bin -maxdepth 2 -type f \( -name '*.so*' -o -perm -u+x \)); do
     LD_LIBRARY_PATH=$R/lib:$R/lib/llvm/lib ldd $f 2>/dev/null | grep 'not found'
   done | sort | uniq -c    # expect no output
   ```
   For each `not found` library, find the artifact that ships it, drop that name from
   `EXCLUDE_ARTIFACTS`, and re-run step 2. Checked on gfx1201 for a `miopen` rebuild: with
   `solver` and `sparse` removed from the planner's exclude list, the bootstrapped base
   (all `stage` directories merged, before any rebuild) had no unresolved libraries; with
   them excluded, `libhipblas` and `librocsolver` did.
3. Run the tests from `$B/dist/rocm` (or the merged tree). Setting the two `*_VISIBLE_DEVICES`
   variables hides GPUs; GPU test cases then skip, and the run must report 0 failures.
   ```bash
   cd $B/dist/rocm && export LD_LIBRARY_PATH=$PWD/lib
   export HIP_VISIBLE_DEVICES=-1 ROCR_VISIBLE_DEVICES=-1
   nice -n 10 taskset -c $CPUS ./bin/hip_kernel_provider_tests   # hipkernelprovider
   ```
   For a hipDNN rebuild run `./bin/hipdnn_backend_tests` and `./bin/hipdnn_data_sdk_tests`.
   Run the same binaries on a GPU machine for GPU coverage. CI runs tests through
   `build_tools/github_actions/test_executable_scripts/test_runner.py`.

Verified on Ubuntu 24.04 (x86-64), gfx950, with baseline runs on TheRock `main`: rebuilt
hipDNN and hipkernelprovider (with hipdnn-integration-tests). Their unit tests passed with
GPUs hidden: 0 failures, GPU cases skipped for lack of a device.

## Troubleshooting

- **A dependency compiles that should be prebuilt.** It is a same-stage dependency that
  `fetch --stage S` does not bootstrap. Use the planner (step 1).
- **Every test fails at once with `error while loading shared libraries`.** The tree
  lacks a bootstrapped library: see Build step 4 and Verify step 2.
- **An incremental rebuild does nothing after you edit a header in another subproject**
  (for example a `composablekernel` header consumed by MIOpen). A subproject's
  `stamp/build.stamp` depends only on its configure stamp and its own sources, so ninja
  reports `no work to do` (observed with a one-line CK header edit). Untested remedy:
  delete `<subproject>/stamp/build.stamp` and `stage.stamp` for the edited subproject
  and its consumers, then run ninja again so their inner builds re-run. Hash the output
  library before and after; identical hashes mean the edit did not reach it.
- **`find_package(hipdnn_integration_tests)`: "config file not found".**
  `hipdnn-integration-tests` publishes only `test` and `run` components, so its CMake
  package is in no archive. Rebuild it with its consumer; the planner does this. If you
  already bootstrapped it, delete `$B/ml-libs/hipdnn_integration_tests/stage.prebuilt` and
  re-run `cmake -S $T -B $B`.
- **Compile errors in rebuilt code against prebuilt artifacts** (a missing helper in
  hipDNN's test SDK, a changed function signature). The sources are newer than the
  baseline's. Build from the pinned `$RL`, or rebuild the artifacts your change depends on.
  Prebuilt consumers of a rebuilt artifact are not rebuilt: they are absent from
  `dist/rocm` unless you list them in `--rebuild`.
- **ninja: "needed by artifacts/..._dbg_generic" or a missing `split_artifacts.py`.** The
  kpack checkout (Prerequisites step 4) is missing.
- **`ModuleNotFoundError: msgpack` while splitting artifacts.** The splitter ran under the
  system Python. Re-run configure with `-DPython3_EXECUTABLE=$V/bin/python` (step 3), then build.
- **Third-party libraries compile** (spdlog, fmt, flatbuffers, nlohmann-json, googletest)
  when you rebuild hipDNN core. They download as source tarballs and are small. The planner
  notes inbound artifacts that have no archives in the baseline.
- **The CI result differs.** rocm-libraries CI also sets
  `THEROCK_FLAG_HIPKERNELPROVIDER_ENABLE_ROCKE=ON`; the default is OFF and the ON build
  was not tested here. Check `.github/workflows/therock-multi-arch-ci.yml` for current flags.

## Not covered

Windows, GPU-backed test runs, the CI test path, stages other than `math-libs`, and
artifacts that depend on a rebuilt artifact (for example the providers when you rebuild
hipDNN).
