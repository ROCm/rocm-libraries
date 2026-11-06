# TensileLite common tests: further speed-up avenues (gfx942)

Collected from the gfx942 timing work of 2026-11-03..05 on branch `users/alvasile/new_test_client`
(host `sharkmi300x-1`, MI300X, 25 pinned cores, `OMP_NUM_THREADS=25`). For each avenue: why that part
is slow today, the evidence, the options, an estimated saving in CI terms, and the risk.

## How to read the numbers

- **CI-like stages** are what matter: the whole build stage in one pytest session
  (`--build-only --gpu-targets gfx942 -n 4 --dist=worksteal`, cache cleared at the start), then the
  whole run stage (`--use-cache -n 4`). Current wall times at `9c5e825e`/`97c59828`: build ≈ 441 s,
  run ≈ 742 s (run measured at `c1345541`; nothing since changes the run stage).
- **Build stage:** timers are summed over the 4 xdist workers (≈1,640 s of instrumented time).
  With `worksteal` the workers are balanced, so a summed saving of X s is roughly X/4 s of wall.
- **Run stage:** every client run on a 1-GPU runner takes the same per-GPU lock
  (`Tests/conftest.py:92-163`, `ClientWriter.py:314`), so clients run one at a time and client time
  is 88-95% of the run wall. A second saved inside the client is about a second of run wall.
- Per-launch costs below come from the instrumented per-config run pass
  (`full_runs_gfx942/post_run/AGGREGATE_analysis.txt`, 441 client launches for 123 configs); CI
  makes the same 441 launches.
- "Estimated" means not measured end to end.

## Already done (for context)

| Change | Commit | Effect (CI-like) |
|---|---|---|
| Deselect configs that opt out of every target arch (563 of 709 on gfx942) | `e2bfff82` | build 612 → 564 s with `load` |
| `--dist=worksteal` for `hw-common` | `056fa353` | build 564 → 475 s (one run); run stage unchanged |
| Artifact gzip level 9 → 4 | `545d5d23` | packing 137.7 → 25.7 s summed (replay), +11.5% size |
| Invalid-kernel check in the parent | `97c59828` | validate 124 → 28 s summed, but the pool resize moved to assembly; wall unchanged (see B1) |
| Timers for artifact packing/extraction; analyzer fixes | `e04a42eb`, `094b04a4` | measurement only |

---

## A. Run stage

### A1. Hardware-monitor setup on every client launch

- **Why it is slow:** `HardwareMonitorListener` (`client/src/HardwareMonitorListener.cpp:38-55`)
  creates a `HardwareMonitor` whenever `--hardware-monitor` is true, which calls
  `amdsmi_init(AMDSMI_INIT_AMD_GPUS)` (`client/src/HardwareMonitor.cpp:139-158`) and starts a
  sampling thread. `HardwareMonitor` defaults to `True` (`Common/GlobalParameters.py:61`); only
  gfx950 and gfx1250 force it off (`Tensile.py:837-839`). The bundled rocm_smi lists `/tmp` twice
  per GPU to find partition files (16 full scans with 8 GPUs); this host's `/tmp` has ~37k entries.
- **Evidence:** `listener_setup` = 179.3 s over 441 launches (407 ms each), 22% of the 829 s
  instrumented run total. Measured with the real client (3 runs each, cores 48-72, GPU 6):

  | Variant | `listener_setup` (ms) | `hip_initialization` (ms) |
  |---|---:|---:|
  | as in the pass (`ROCR_VISIBLE_DEVICES=6`, monitor on) | 404-411 | 378-409 |
  | `hardware-monitor=False` | **0.02-0.03** | 386-430 |
  | `HIP_VISIBLE_DEVICES=6` instead of ROCR (monitor on / off) | 406-407 / 0.02-0.03 | 384-391 |
  | monitor on, private empty `/tmp` (mount namespace) | **102-103** | 380-407 |

- **Why `amdsmi_init` costs ~400 ms here:** the bundled rocm_smi looks up
  `rocmsmi_boot_{compute,memory}_partition_<i>.*` by listing `/tmp`, twice per GPU (16 full
  directory scans with 8 GPUs). One `ls -f /tmp` of the 37,120 entries costs 25-45 ms. The path is
  hard-coded: `TMPDIR` does not change it. The remaining ~100 ms is amd-smi's sysfs walk of all 8
  GPUs (KFD `io_links`, PCI, DRM, hwmon), which does not honour `ROCR_VISIBLE_DEVICES`. So on a CI
  runner with a small `/tmp` the cost is probably ~100 ms per launch (not measured on a runner).
- **Wasted:** the monitor only samples around timed enqueues (`preEnqueues`/`postEnqueues`), and only
  11 of 123 configs time anything (A2). The other ~410 launches initialize it for nothing.
- **Options:**
  1. Start the monitor only when something is timed: `m_active = hardware-monitor &&
     num-syncs-per-benchmark > 0` (or create `m_monitor` lazily in `preEnqueues`).
  2. Pass `HardwareMonitor: False` for test runs (as `Tensile.py` already does for gfx950).
- **Estimated saving:** ~165 s of run wall on this host; ~40 s on a runner with a small `/tmp`.
- **Risk:** low; option 1 keeps the clock/temperature reports for the 11 timing configs.

### A2. Why 11 YAMLs use the hardware monitor at all

- **Why:** these configs keep the default `SyncsPerBenchmark: 1` (the other ~110 set 0), so the
  client runs the timed benchmark loop and the monitor samples it. Each is an explicit opt-out in
  `Tests/unit/test_common_yaml_syncs_per_benchmark.py` with a reason:

  | Config | Stated reason for the benchmark loop |
  |---|---|
  | `benchmark_runs/fp32_tn` | benchmark timing coverage |
  | `client/rotate_mode0`, `client/rotate_mode1` | RotatingMode test requires the benchmark loop |
  | `gemm/icache_flush` | flush_icache only runs inside the sync loop |
  | `gemm/fp8nfp16mix_hfp8ns`, `gemm/xfp32`, `groupedgemm/grouped_gemm`, `groupedgemm/grouped_gemm_ck_gfx942`, `sparse/gfx94x/libray_logic` | LibraryLogic requires benchmark data |
  | `streamk/sk_hybrid`, `streamk/sk_queue_selfclean` | back-to-back launches validate output and synchronizer |

- **Open question:** none of these reasons needs clock, temperature or fan readings; they need the
  timed loop. If LibraryLogic and the CSV consumers do not read the monitor's columns
  (`ClockRateSys`, `TempEdge`, ...), the monitor can be off for every test run, timed or not. To
  check: which `ResultKey`s the LibraryLogic path reads.
- **Saving:** the remaining ~31 launches × 0.1-0.4 s once A1 is done (a few seconds).

### A3. HIP initialization on every client launch

- **Where:** `client/main.cpp:883-897`: `GetHardware` (`hipGetDeviceCount`, `hipSetDevice`,
  `hip::GetCurrentDevice`, `main.cpp:449-462`) and one `DeviceContext` (stream creation).
- **Why it is slow:** a minimal probe (`hipGetDeviceCount` → `hipSetDevice` → `hipStreamCreate` →
  `hipMalloc`) reproduces the 381 ms: `hipGetDeviceCount` 334-361 ms, `hipSetDevice` 0.0 ms, first
  `hipStreamCreate` 44-45 ms, `hipMalloc` 0.1 ms.
  - `hipGetDeviceCount` (= ROCr `hsa_init`): the thunk reads the topology of every KFD node before
    `ROCR_VISIBLE_DEVICES` filtering. `strace` counts 16,266 `openat` and 26,481 `read`, including
    10,016 opens of `/sys/devices/virtual/kfd/kfd/topology/nodes/<n>/caches/<c>/properties`
    (626 cache entries per MI300X node, 8 GPU nodes, each read twice) and ~5,000 per-CPU cache
    files (192 CPUs). The ~350 ms between the first KFD ioctl and the first queue ioctl contains
    no ioctls: pure sysfs parsing. Reading the 5,008 GPU cache files once with `cat` costs 188 ms,
    so two passes explain ~340 ms.
  - First stream: each hardware queue gets a 186,019,840-byte anonymous mapping registered with
    `AMDKFD_IOC_SVM` (~15 ms each, probably the per-queue CWSR save area) before
    `AMDKFD_IOC_CREATE_QUEUE`; three are made at start-up.
  - No variable helps: `ROCR_VISIBLE_DEVICES` vs `HIP_VISIBLE_DEVICES` gave 378-430 vs 384-391 ms,
    and no `HSA_*` / `HSAKMT_*` variable in `libhsa-runtime64.so.1.18.70204` skips the
    cache-topology parsing (checked the full list of environment strings).
  - Concurrency: 4 simultaneous probes took 472-491 ms (`amdsmi_init`) and 439-453 ms
    (`hipGetDeviceCount`) instead of 397 / 334 ms, so start-up mostly overlaps across processes
    (relevant to A6b).
- **Evidence:** `hip_initialization` 167.9 s / 441 launches (381 ms each). gfx950's ~150 ms is
  probably a smaller topology (inferred).
- **Options:** fewer launches (A4); `GPU_MAX_HW_QUEUES=1` saves ~15 ms setup + ~15 ms exit per launch.
- **Saving:** `GPU_MAX_HW_QUEUES=1` ≈ 13 s; the rest only via A4. **Risk:** low-medium.

### A4. Too many client launches (441 for 123 configs)

- **Why:** `BenchmarkProblems._benchmarkProblemType` runs one client process per benchmark step
  (`BenchmarkProblems.py:1175-1176` → `runClient`, `:1116`). A config with several problem types or
  problem-size groups launches the client several times, each paying start-up, HIP init, monitor,
  library/code-object loading and exit.
- **Evidence:** fixed per-launch costs ≈ 468 s (Client launch group: `listener_setup` 179,
  `hip_initialization` 168, `process_exit` 55, `library_loading` 17, `data_init_setup` 15,
  `code_object_loading` 13, `main_teardown_locals` 11, `process_startup` 9) = 56% of the run total.
  `DEFERRED.md` #5 also records a first-call cost per process in `cpu_data_init`.
- **Can the client be reused today? No.** One client process serves exactly one job: one solution
  library (`--library-file`, a single value), its code objects and one problem set. `main()` reads
  its arguments once and builds the problem factory, data initialization, listeners and reporters
  for that job (`client/main.cpp:812`, `:864`, `:967`, `:997`). Several `--config-file` arguments are
  merged into one option set, not run as separate jobs (`main.cpp:643-653`), and Tensile's run script
  starts the client once per config file (`ClientWriter.py:441-442`).
- **What reuse would take:**
  1. A batch mode in the client: a list of jobs per process, with `main()`'s body moved into a
     per-job function and per-job state reset between jobs (code-object modules, caches, listeners,
     results file).
  2. Tensile launching once per config with all of its steps. In the run stage (`--use-cache`) every
     step's library already exists, so this works if a config's steps do not depend on earlier
     steps' results (believed true in the test flow; not verified).
  3. Or, larger: a persistent client per xdist worker receiving jobs over a pipe.
- **What reuse saves, and what it does not** (per launch / summed over 441 launches):

  | Cost | Per launch | Summed | Removed by reuse? |
  |---|---:|---:|---|
  | `hip_initialization` | 381 ms | 168 s | yes |
  | `process_exit` | 124 ms | 55 s | yes |
  | `process_startup` | 19 ms | 9 s | yes |
  | `main_teardown_locals` | 26 ms | 11 s | partly |
  | `listener_setup` (hardware monitor) | 407 ms | 179 s | yes, but A1 removes it more simply |
  | `library_loading` | 39 ms | 17 s | no (per job) |
  | `code_object_loading` | 30 ms | 13 s | no (per job) |
  | `data_init_setup` | 34 ms | 15 s | no (per job) |

  After A1, reuse removes ~0.5 s per launch, ~230 s summed, mostly HIP initialization.
- **What reuse costs:** isolation. A GPU memory fault or hung kernel in one step would take down
  every later step in that process (today each step is its own process), and state left over by one
  job (buffers, cached pointer arrays, loaded modules) can affect the next.
- **Interaction with the GPU lock (A6b):** both target the same fixed per-launch cost.
  Shrinking the lock keeps the start-up work but lets the 4 workers overlap it with another
  client's GPU phase (saves wall, not CPU) and can also overlap the CPU reference and comparison.
  Reuse removes the start-up work (saves CPU and wall) but does nothing for the CPU phases inside
  the lock. Reuse also needs the narrower lock: a client living across several steps cannot be
  wrapped by today's process-level lock without holding the GPU for the whole config, so the lock
  has to move into the client first. Once the lock is narrowed, the start-up cost is already
  overlapped, and reuse's remaining CI gain is at most ~230 s / 4 ≈ 60 s of run wall; it matters
  more for serial (per-config) and multi-GPU runs.
- **Estimated saving (CI):** ~210 s of run wall if done instead of A6b (after A1); ~60 s at most
  if done after A6b. **Risk:** medium-high.
- **Recommendation:** do A1, then A6b option 1 (lock inside the client, after start-up), measure;
  consider reuse only if the remaining HIP initialization cost still matters.

### A5. Client process exit

- **Why:** `process_exit` 124 ms per launch: mostly unmapping the five 186 MB per-queue regions
  (~15 ms each) plus kernel-side teardown. An early `_exit` saves only ~14 ms.
- **Saving:** ~6 s via fast exit, ~8 s via `GPU_MAX_HW_QUEUES=1`. Mostly folded into A3/A4.

### A6. Clients run one at a time under `-n 4`

- **Why:** with one visible GPU every xdist worker gets device id "0" and the same lock file; the
  lock is held around the whole client run. Run wall ≈ sum of client times; the GPU is busy only a
  fraction of the time (≈13% estimated on gfx950).
- **Evidence:** `worksteal` balanced the run-stage workers but the wall stayed 747/736 s vs
  751/746 s; summed test time rose (lock waiting).
- **Options:** a shared lock for configs that tolerate a shared GPU, exclusive for Stream-K etc.;
  split `OMP_NUM_THREADS` among concurrent clients.
- **Estimated saving:** 30-50% of the run wall (not measured). **Risk:** high: a Stream-K config
  (`sk_sgemm_hybrid_pap`) failed validation once while sharing a GPU on gfx950.

### A6b. Shrink the region the GPU lock covers (investigate)

- **What the lock covers today:** `runClient` takes `ClientExecutionLock` around the whole client
  subprocess (`ClientWriter.py:314-322`: `Popen` → `communicate()`). The lock file is per physical
  GPU (`Tests/conftest.py:92-163`, `worker_lock_path`), passed to Tensile as `--client-lock`
  (`Tensile.py:261, :295`). So everything the client does is serialized, not only the GPU work.
- **What inside it actually needs the GPU to itself** (per-config run totals,
  `post_run/AGGREGATE_analysis.txt`, summed over 441 launches, instrumented):

  | Part of the client run | Time | Needs exclusive GPU? |
  |---|---:|---|
  | Process start, HIP init, hardware monitor, library/code-object load, teardown, exit (A1-A5) | ~468 s | no (HIP init touches the device but does not compete with another client's kernels) |
  | CPU reference GEMM (`solve_cpu_fast`) | ~38 s | no (CPU) |
  | Result comparison on the CPU (`validate_element_comparison`) | ~32 s | no (CPU) |
  | Input preparation (host init, uploads/resets) | ~54 s | partly (copies) |
  | GPU readback, Synchronizer reduce | ~30 s | partly |
  | Kernel launches and execution (incl. 51 s of instrumentation-only `warmup_first_sync`) | ~68 s | yes |

  On this split most of the ~720 s of client time does not need the GPU exclusively, yet with one
  GPU it all runs one client at a time (A6).
- **Why the lock exists (to confirm):** keep benchmark timings free of interference, avoid device
  memory pressure from 4 clients, and avoid the shared-GPU failure seen with a Stream-K config
  (`sk_sgemm_hybrid_pap`, gfx950). A narrower lock that still covers all kernel execution keeps the
  last two properties; timed configs (A2) could keep holding it for their timed loop.
- **Options, smallest first:**
  1. Take the lock inside the client instead of around the process: pass the lock path to the
     client (`--gpu-lock`), acquire it (`flock`) after start-up/HIP init/library load and release it
     before teardown and exit. This alone moves the ~1 s per launch of start-up and exit (A1, A3,
     A5) out of the serialized region.
  2. Hold it only around the solution loop's GPU phases (uploads, launches, readback), not around
     CPU reference and CPU comparison.
  3. Per-problem: compute the CPU reference before taking the lock.
- **Constraints:** 4 clients each run `OMP_NUM_THREADS=25` CPU work on the same 25 cores, so
  overlapping CPU phases will contend; split the thread count among clients or keep the CPU-heavy
  phases short. Any client crash must release the lock (`flock` is released on process exit).
- **Device memory:** with the lock taken after start-up, up to 4 clients hold their allocated
  buffers on the GPU at once. Fine for MI300X; check configs with large buffers.
- **Relation to client reuse (A4):** the narrower lock is the smaller first step and a prerequisite
  for reuse (see A4).
- **Estimated saving:** not measured. Option 1 alone removes ~1 s per launch from the serialized
  region (~440 s of the ~720 s), which could cut the run wall substantially once overlapped across 4
  workers; the realistic gain depends on CPU contention. **Risk:** medium (client and harness
  change; timing-sensitive configs must keep exclusivity). **Before building:** confirm why the
  lock was added (benchmark timing interference, device memory, Stream-K sharing) and whether a
  config's benchmark steps are independent in the run stage; then prototype option 1 on one
  multi-step config (e.g. `gemm/fp16_tn`) with `-n 4` and compare run-stage wall.

### A7. Result validation

- **Why:** `validate_element_comparison` (32.4 s) compares on one thread; `validate_gpu_readback`
  (24.9 s, 11-17 µs per call, dominated by call count in `fp16_tn`, `bufferLoadStore0`); the pinned
  host buffer is reallocated whenever the size changes rather than when it grows
  (`ReferenceValidator.cpp:729`, 3.3 s).
- **Options:** multi-threaded comparison; grow-only buffer; later a GPU-side compare.
- **Estimated saving:** 15-20 s (CPU changes), 40-50 s with a GPU compare. **Risk:** low / medium
  (must keep tolerance, NaN and bounds-check semantics).

### A8. Smaller run-side items

- `gpu_input_reset_batched_inputs` 19.7 s: the GuardPageAll pointer arrays are re-uploaded; noted
  in `after_run.md`, not filed anywhere.
- CPU reference `solve_cpu_fast` 37.4 s on gfx942 (the large FP8 MX cases are gfx950-only).
- Artifact extraction: < 1 s per config, outside the lock. Not worth pursuing.
- `warmup_first_sync` exists only under instrumentation; not a CI cost.

---

## B. Build stage

CI-like instrumented build stage, summed over the 4 workers (mean of 2 runs at `97c59828`):
`python_kernel_codegen` 520.5 s, `python_kernel_write_assemble` 388.9 s,
`python_solgen_forked_solutions` 316.6 s, `python_kernel_bench_postprocess` 132.5 s,
`python_kernel_validate` 27.7 s, `python_artifact_compress` 28.3 s; TOTAL 1,637 s.

### B1. Process-pool resizes and restarts

- **Why it is slow:** `ParallelMap2` (`Common/Parallel.py`, ~line 272) asks joblib for
  `min(budget, number of tasks)` workers, so every small step asks for a different pool size.
  joblib's loky backend passes thread-limit variables computed as `cpu_count() // n_jobs` for any
  of them not set in the parent (`MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, ...;
  `joblib/_parallel_backends.py:221-245`), so a different size changes the executor's arguments and
  loky shuts the whole pool down and starts a new one (`loky/reusable_executor.py:189-215`). A plain
  shrink also waits for the extra workers to exit (`_resize`).
- **Evidence:** on `gemm/gsuasb.yaml`, each size change cost ~0.5 s (515 / 500 / 580 / 472 ms) vs
  16 ms when the size did not change. The per-config pass attributed ~190 s summed to restarts,
  shrinks and grows, plus ~136 s to cold starts. Moving the invalid-kernel check out of the pool
  only moved the resize to assembly (`97c59828`).
- **What shrinking buys:** nothing measurable: idle workers use no CPU, the threads-per-worker
  change is irrelevant (workers run Python and launch clang; `OMP_NUM_THREADS` is fixed), and the
  workers being shut down already exist.
- **Options:** keep the worker count constant within a build when `procs` is given (no task-count
  cap); keep the thread-limit variables constant (`parallel_config(inner_max_num_threads=...)` or set
  them once in the parent); keep running one-task calls in the parent. In CI the pool lives in each
  xdist worker process and would also stay warm across configs.
- **Estimated saving:** 100-200 s summed → ~25-50 s of build wall. **Risk:** low-medium
  (`ParallelMap2` is shared with TensileCreateLibrary; can be limited to calls with explicit `procs`).

### B2. Candidate solutions that are built and then thrown away

- **Why it is slow:** forking expands every parameter combination, and each candidate is fully
  constructed before validity is known: `deepcopy(problemType.state)` (`BenchmarkProblems.py:286`),
  `Solution(...)` (`Solution.py:910-1047`), then rejection inside
  `Solution.assignDerivedParameters`. Each pool task also pickles `globalParameters`,
  `isaInfoMap`, `rocIsaData` and the assembler (`Parallel.py:291`, `BenchmarkProblems.py:342-353`).
- **Evidence:** across the build, 843,891 candidates → 105,160 valid (87.5% rejected). `dtv`:
  534,736 → 18,620 (96.5% rejected), 28.4 s in the per-config pass; `dtl`: 16,688 → 2,078 (87.5%);
  `cgemm` 55,296 → 20,520; `bufferLoadStore0` 70,656 → 12,106. Real work ≈ 0.8 ms CPU per candidate
  (~675 CPU-s, ~590 of it on rejected ones); the rest of `python_solgen_forked_solutions` is pool
  overhead (B1).
- **Options:** reject before building the full `Solution` (cheap pre-checks on the fork parameters
  for the rules that reject most); ship constants once per worker instead of per task; batch forks
  across steps.
- **Estimated saving:** 20-40 s of per-config time (mostly `dtv`, `cgemm`, `bufferLoadStore0`,
  `zgemm`) → ~10-20 s of build wall. **Risk:** medium: the early check must reject exactly what
  `Solution` rejects. **First step:** profile `dtv`'s generation to find which rules reject most.

### B3. Kernels generated and then rejected (codegen)

- **Why:** some valid solutions overflow resources only during code generation.
- **Evidence:** 105,160 valid solutions → 97,220 kernels kept (7,940 rejected by KernelWriter).
  The branch already stops codegen early once the static VGPR allocation exceeds `MaxVgpr` or
  `overflowedResources` is set; `DEFERRED.md` #10 records the rejected kernels' cost falling from
  374.5 to ~10.8 CPU-s on the probed configs (gfx950).
- **Option:** reject over-budget kernels in solution validation, before codegen (#10).
- **Saving:** small (~10 CPU-s). **Status:** deferred.

### B4. Kernel-name and compile-key work repeated per solution

- **Why:** `getKernelCompileKey` → `getKernelNameMin` is computed per solution and per kernel three
  times: in `removeInvalidSolutionsAndKernels`, in `validKeys` (`Run.py:1253-1257`) and in
  `passPostKernelInfoToSolution` (`Run.py:727+`); `python_kernel_bench_postprocess` (132.5 s) also
  spends time in naming and library construction (1.26 ms per solution, 74% in six configs).
- **Option:** compute each name/key once and cache it on the solution.
- **Estimated saving:** 30-60 s summed (~29 s of it in the validate path). **Risk:** low.

### B5. Assembly path

- **Why:** codegen results go back to the parent and then into a second pool per step; each `.s` is
  written, then re-read and regex-scanned (`_retargetAssemblySource`, `Toolchain/Component.py:301`),
  and assembled with one clang process per kernel (~56 ms per kernel).
- **Option:** assemble inside the codegen workers and drop the re-read.
- **Estimated saving:** 40-70 s summed. **Risk:** low-medium.

### B6. Code generation itself

- **Why:** ~50 ms of worker CPU per kernel (`dtv` 12,406 kernels; large tn configs ~200 ms per
  kernel).
- **Options:** `DEFERRED.md` #8 (longest-first dispatch within a codegen step) and #9 (cheaper
  generator: `hasAnyDependency`, epilogue). **Risk:** medium-high (changes generated code).

### B7. Scheduling tail and core sharing

- **Why:** `dtv` alone takes ~230-270 s of a ~441-475 s build, and with collection order it starts
  mid-stage. Each xdist worker's pool is sized for 4 busy workers (2 × 25 / 4 = 12 processes), so
  when workers go idle at the end, the last one cannot use the freed cores; when all 4 are busy
  there are up to 48 processes on 25 cores (test time rose 8% under `worksteal`).
- **Options:** start the longest configs first (cost-ordered collection; pairs with
  `load --maxschedchunk=1/2`, not with `worksteal`); split `dtv`; a shared core budget across xdist
  workers (jobserver-style token pipe) so the last worker gets all cores.
- **Estimated saving:** up to ~30-40 s (the stage's lower bound is ~435 s of work / 4 workers).
  **Risk:** medium for the shared budget (tokens must be released if a worker crashes).

### B8. Artifact packing in the combined `hw-common` job

- **Why:** `hw-common` runs the combined mode (`Tests/common/test_config.py`): each config's build
  output is packed into a `.tar.gz` in the test's tmpdir and unpacked immediately. Nothing is
  transferred, so packing and unpacking are pure overhead there.
- **Evidence:** level 4 packing ≈ 25.7 s summed for the gfx942 set (replay); level 0 5.4 s.
- **Options:** skip the archive in combined mode (pass the output directory to the run step), or
  pack at level 0/1 there. Keep level 4 for a split CI that uploads artifacts.
- **Estimated saving:** ~20 s summed (≈5 s of wall) plus extraction. **Risk:** low, but the
  round trip currently also checks the artifact whitelist; keep that check if it matters.

### B9. CMS configs fork `UseCustomMainLoopSchedule: [0, 1]`

- **Where:** `gemm/gfx950/custom_mainloop_scheduling.yaml` (14 forks with `[0, 1]`),
  `gemm/gfx950/agntab_coverage_gfx950.yaml` (1), `gemm/gfx12/custom_schedule.yaml` (1). All of them
  skip gfx942, so this costs time on gfx950 and gfx12, not in the gfx942 numbers above.
- **Why it is slow:** the fork builds every tile configuration twice: with the custom main-loop
  schedule (1) and with the regular scheduler (0). The default `-1` already resolves to 1 where CMS
  is supported and 0 otherwise (`SolutionStructs/Solution.py:3950-3963`). The `0` kernels exercise the
  regular scheduler, which the general configs already cover, so in a CMS test they mostly double the
  codegen, assembly and benchmark work of those steps.
- **Evidence (gfx950):** `custom_mainloop_scheduling` has 66 benchmark steps (so 66 client launches,
  see A4) and 79 kernels; it was the second-slowest test of the gfx950 split run phase (159 s under
  `-n 4`) and 30 s of per-config build after the build changes (`after_run.md`, `after_build.md`,
  `python_kernel_codegen_investigation.md`). How much of that is the `0` half was not measured.
- **Option:** fork `[1]` only (or `[-1]`) in the CMS configs, unless a step specifically compares
  the two; keep one explicit `0` case if the "CMS off" path for CMS-capable tiles needs coverage.
- **Estimated saving:** up to half of the kernels in the 14 forked steps. **Risk:** low (test
  coverage decision for the CMS owners).

---

## C. Per-process start-up (Tensile and pytest)

CI pays these once per xdist worker per stage (~4 s of wall per stage), not per config; the
per-config timing harness pays them per config (~4 s × 123).

### C1. Collection loads every YAML

- **Why:** `configMarks` parses all 744 test YAMLs to compute marks even when one test is selected
  (`Tests/conftest.py:82` → `Tests/common/config_helpers.py:125, :155`), ~2.25 s per pytest process
  (already `CSafeLoader`). The new arch deselection reuses these marks and adds nothing.
- **Option:** cache the marks per YAML keyed on mtime/size. **Saving:** ~2 s per stage in CI.

### C2. ISA capability probing

- **Why:** `makeIsaInfoMap` (`Tensile.py:841` → `Capabilities.py:166`) runs 114 `amdclang++`
  probes serially (~12 ms each, `hardware_caps.hpp:33-75`, `:97+`) for the one ISA: 1.4-1.9 s per
  process.
- **Options:** cache on disk keyed on assembler identity, ISA and probe-table version; or run the
  probes in parallel (measured 1.3 → 0.35 s); or probe with `llvm-mc` (0.42 s serial).
- **Saving:** ~1-1.5 s per stage in CI. **Risk:** low-medium (stale cache if the compiler changes).

### C3. `rocm_agent_enumerator` on every rocisa import

- **Why:** with `ROCISA_BACKEND` unset, `import rocisa` runs `rocm_agent_enumerator` (45 ms,
  `rocisa/__init__.py:124-171`) in every process, including every new loky worker (32 times for
  `gemm/ag` in one build). `--gpu-targets` does not skip it.
- **Option:** pass the parent's backend choice to workers. **Saving:** estimated 10-15 s summed in
  CI (more while B1 restarts pools). **Risk:** low.

### C4. Process exit

- **Why:** shutting the reusable loky executor down at exit takes 0.2-0.7 s; interpreter teardown
  up to 0.45 s (~816k live objects after `Tensile()` returns). Once per worker in CI.
- **Option:** `os._exit` after flushing. **Saving:** ~0 in CI. **Risk:** medium.

---

## D. Measurement environment and CI configuration (not product code)

- **Editable rocisa rebuilds on every import.** The venv's scikit-build-core editable install runs
  `cmake --build/--install` under a machine-wide lock on each `import rocisa` (1-9 s; 0.11 s with
  `SKBUILD_EDITABLE_SKIP=<build dir>`). It inflated a whole first measurement 2-5× and coupled the
  two measurement sides. The measurement scripts should set `SKBUILD_EDITABLE_SKIP`. CI installs a
  wheel and is unaffected.
- **Shared-host `/tmp`** with ~37k entries adds ~300 ms to every `amdsmi_init` (A1).
- **CI marker `not skip-gfx94X` matches nothing** (TheRock's `fetch_test_configurations.py`): configs
  carry exact-arch marks. The in-repo deselection (`e2bfff82`) makes it moot for speed; the dead
  expression could still be cleaned up in TheRock.
- **`comm/gfx950/fused_a2a.yaml` fails the gfx942 build-only stage** ("0 valid solutions"): a
  gfx950-only config without a `skip-gfx942` mark (`CI_SPLIT_TIMING.md` finding 3). Correctness, not
  speed.
- **Per-config timing harness overhead:** the serial scripts start one pytest per config (~5 s each
  for collection, probing, start-up); this is harness cost, not CI cost.

---

## E. Test-suite hygiene (not speed)

### E1. Unit tests insert into `sys.path` at import time

- **What:** 24 files under `Tests/` change `sys.path`, most at module import (collection) time, e.g.
  `Tests/unit/test_subtile_gfx1250_codegen.py:19-21` puts the TensileLite root at index 0 so it can
  `import gpu_test_helpers`; `gpu_test_helpers.py:39`, `test_mxs_tilespan.py`,
  `test_emitMfmaInstruction.py`, `test_storeD_roundtrip.py`, ... do the same; several insert the
  characterization `_codegen` dir; `test_analyze_timing.py` and `test_aggregate_timing.py` insert
  `scripts/`, and `test_artifact_timing.py` (added in this work, `e04a42eb`) inserts `Tests/common`.
  `Tests/conftest.py:39` already appends the same root at the end of the path.
- **Why it matters:** the change is process-wide and permanent, so every later test in the same
  pytest/xdist worker process sees it, and what a test imports depends on which other modules were
  collected first. Inserting at index 0 makes the source tree shadow an installed package (CI runs
  against the installed `share/hipblaslt/tensilelite` tree), and lets bare module names
  (`gpu_test_helpers`, `artifact_helpers`, `analyze_timing`) resolve only by accident of order. The
  order-dependence bug fixed in `9c5e825e` is the same class of problem (import-time side effects).
- **Options:** declare the import roots once (`pythonpath =` in `pytest.ini`, or the root
  `conftest.py`), import helpers as package modules, and load scripts with `importlib` from an
  explicit path in the tests that need them; then remove the per-file insertions, starting with the
  ones that insert at index 0.
- **Speed:** negligible (a longer `sys.path` only adds lookups). **Risk:** low, but touches many
  files; run the unit suite in random order / per file to confirm nothing relied on the side effect.

---

## Suggested order

| # | Avenue | Stage | Estimated CI wall saving | Risk |
|---|---|---|---:|---|
| 1 | A1 hardware monitor only when timed | run | ~40 s (CI runner) / ~165 s (this host) | low |
| 2 | B1 constant pool size and thread-limit env | build | ~25-50 s | low-medium |
| 3 | A6b lock inside the client (after start-up), then narrow to GPU phases | run | potentially large; not measured | medium |
| 4 | B2 early candidate rejection (profile `dtv` first) | build | ~10-20 s | medium |
| 5 | B4 / B5 name caching, assemble in codegen workers | build | ~20-30 s | low-medium |
| 6 | A7 validation | run | ~15-50 s | low-medium |
| 7 | B7 longest-first, shared core budget | build | ~30-40 s | medium |
| 8 | B8 skip packing in combined mode | build | ~5 s | low |
| 9 | A6 concurrent clients | run | 30-50% of run (unmeasured) | high |
| 9b | A4 client reuse (batch mode), only if HIP init still matters after A6b | run | ≤ ~60 s after A6b (~210 s without it) | medium-high |
| 10 | C1-C3 start-up caching | both | a few s per stage | low |
| 11 | B9 CMS `[0, 1]` forks (gfx950/gfx12) | build + run | not measured; up to half of those steps | low |
| — | E1 `sys.path` insertions | hygiene | none | low |
