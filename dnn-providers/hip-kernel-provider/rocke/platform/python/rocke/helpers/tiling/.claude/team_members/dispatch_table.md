# Team Member Dispatch Reference

## Context Detection

Skills detect context from three sources: the code/files, the user's request, and
the environment. **Multiple contexts can be active simultaneously** (e.g., a HIP
application produces both C++ and HIP/AMD context). Each registry row's Context is
checked independently — a row matches if its Context is present in any of the
detected contexts.

### Source 1: Code and Files

**Programming Language** (by file extension):
- `.cpp`, `.hpp`, `.h`, `.cc`, `.cxx` → C++
- `.py` → Python
- `.rs` → Rust
- `.js`, `.jsx`, `.ts`, `.tsx` → JavaScript/TypeScript
- `.go` → Go
- `.java` → Java
- `.c` → C
- `.sh`, `.bash` → Shell

**Compiler Toolchain** (by file content or extension):
- `.ll` (LLVM IR), `.mlir` (MLIR), `.td` (TableGen), `.bc` (LLVM bitcode) → C++/LLVM
- `compile_commands.json` containing `clang`, `amdclang`, or `hipcc` → C++/LLVM
- `CMakeLists.txt` setting `CMAKE_CXX_COMPILER` to `hipcc`, `amdclang++`, or `clang++` → C++/LLVM

**GPU Runtime** (by code content or build target):
- HIP API calls (`hip*`, `__global__`, device code) OR GPU targets (`gfx942`, `gfx90a`, `gfx908`, etc.) → HIP/AMD
- CUDA API calls (`cuda*`, `__global__`, device code) OR GPU targets (`sm_80`, `sm_86`, etc.) → NVIDIA GPU (CUDA)

**Execution Environment:**
- Code without GPU involvement (no HIP/CUDA detected) → Host
- HIP/CUDA codebases contain **both** host and device code. When a role has team
  members in both Host and HIP/AMD contexts, both match — the algorithm will ask
  the user which applies.

> **Note**: This interactive prompt happens more often than you might expect in GPU
> codebases, because most code has both host and device paths. For example, asking
> for a "Sanitizer Expert" in a HIP codebase will ask: *Host (ASan/UBSan) or GPU
> (AddressSanitizer/XNACK)?* — since both experts match. This is by design: the
> system prefers to ask rather than guess.

**Build System** (by file presence):
- `CMakeLists.txt` → CMake
- `Makefile` → Make
- `build.gradle` → Gradle
- `pyproject.toml`, `setup.py` → Python build

### Source 2: User's Request

The user's request may contain keywords that match a **Context** or **Domain** value
in the Team Member Registry. Treat any match as detected context. If the user
describes an activity without naming a specific tool and multiple team members
match, ASK which tool or system they are using.

### Source 3: Environment

Some contexts are always present based on the working environment:
- Inside a git repository → Git
- ROCm installed / AMD GPU present → HIP/AMD

---

**If context cannot be determined from any source**: ASK the user to clarify.

**If no matching team member exists**: STOP and tell the user. Ask if they want to
create a new team member for the missing area. Guide them through the creation process.

## Resolution Algorithm

Skills request team members using **generic role names** (the "Generic Role" column
below). The dispatch table resolves each generic role to a **context-specific team
member** based on detected context and domain matching.

**Dispatching multiple different generic roles is normal.** A single task often needs
several experts — e.g., Code Expert + GPU Expert + Compiler Expert. Dispatch them
all without asking. Each role is resolved independently through the algorithm below.

**Hard rule: Do not guess when multiple team members match the SAME generic role.**
If a single role resolves to more than one candidate and the domain does not clearly
identify one — e.g., which profiling tool, interactive or passive debugging, host or
device sanitizer — ASK the user to disambiguate before dispatching.

When a skill requests a generic role (with an optional domain hint):

1. Detect context using the Context Detection rules above.
2. Find all rows where **Generic Role** matches the requested role.
3. Filter to rows whose **Context** matches the detected context.
4. If a **domain hint** was provided, further filter to rows whose **Domain**
   keywords overlap with the hint. The domain hint is a short phrase describing
   what the skill needs (e.g., "hardware counters", "system-wide timeline",
   "interactive debugging", "crash triage"). Domain hint matching uses keyword
   and fuzzy matching with logical intuition — if the hint's intent aligns with
   a team member's domain keywords, it matches. Exact substring match is not
   required; semantic relevance is.
5. If exactly one match → dispatch that team member.
6. If multiple matches and the domain hint clearly identifies one → dispatch it.
7. If multiple matches and the domain is ambiguous or no hint was provided →
   ASK the user which team member to dispatch, showing the Domain descriptions
   to help them choose.
8. If zero context-specific matches but a row with Context "Any" exists →
   dispatch that team member.
9. If zero matches at all → STOP and tell the user. Ask if they want to create
   a new team member for this context.

### Domain Hint Examples

| Skill Request | Domain Hint | Result (HIP/AMD context) |
|---|---|---|
| Profiling Expert | "hardware counters" | rocProf Expert |
| Profiling Expert | "roofline analysis" | ROCm Compute Profiler Expert |
| Profiling Expert | "host-device timeline" | ROCm Systems Profiler Expert |
| Profiling Expert | (none) | Ask the user which profiling tool |
| Debugger Expert | "interactive debugging" | rocgdb Expert |
| Debugger Expert | "crash triage" | ROCr Debug Expert |
| Debugger Expert | (none) | Ask the user: interactive session or passive crash dump? |
| Sanitizer Expert | (any, Host context) | Host Sanitizer Expert |
| Sanitizer Expert | (any, HIP/AMD context) | GPU Sanitizer Expert |

## Team Member Registry

<!-- BEGIN GENERATED REGISTRY — do not edit manually -->
<!-- Rebuilt by reading YAML frontmatter from all team member files in this directory. -->
<!-- To rebuild: glob .claude/team_members/*.md (excluding dispatch_table.md), read frontmatter, regenerate this table. -->

| Generic Role | Team Member | File | Context | Domain |
|---|---|---|---|---|
| Code Expert | C++ Expert | `cpp_expert.md` | C++ | RAII, memory leaks, lifetime, dangling pointers, boundary conditions, off-by-one, data races, deadlocks, undefined behavior, aliasing, type safety, exception safety, const correctness, implicit conversions, move semantics, iterator invalidation, lambda captures, smart pointers |
| Architecture Expert | C++ Architect | `cpp_architect.md` | C++ | Design patterns, API design, interface composition, code duplication, inheritance hierarchies, virtual functions, single responsibility, namespace organization, visibility, template hygiene, compile-time error messages, structural debt, abstraction boundaries |
| Testing Expert | C++ Tester | `cpp_tester.md` | C++ | Test coverage, edge cases, boundary testing, floating-point tolerance, precision, determinism, flaky tests, test infrastructure, GPU test dispatch sizes, architecture guards, test compile cost, gtest, parameterized tests |
| Realist | Realist | `realist.md` | Any | Assumption validation, complexity vs. problem fit, trade-off analysis, cost-benefit, failure scenarios, evidence-based decisions, completeness assessment, simpler alternatives, over-engineering detection |
| Documentation Expert | Documentation Expert | `documentation_expert.md` | Any | Doxygen, public API docs, TODO/FIXME audits, documentation gaps, consistency, deprecated code, spelling, grammar, audience fit, staleness, accuracy, completeness |
| Onboarding Expert | Onboarding | `onboarding.md` | Any | First-read comprehension, information gaps, naming clarity, navigation, cross-references, usage examples, newcomer perspective, workspace system, skills, team members, dispatch, settings, onboarding, teaching, explanation, discoverability |
| Compiler Expert | Clang Expert | `clang_expert.md` | C++/LLVM | Clang, building, linking, object code, assembly generation, build errors, compile time, template instantiation depth, IR function count, constexpr evaluation, SFINAE, concepts, compiler flags, optimization levels, -ftime-trace, include cost, header organization, compile-time loops, codegen, explicit instantiation, linker errors, undefined symbols, ABI compatibility |
| Revision Control Expert | Git Expert | `git_expert.md` | Git | Commits, commit messages, commit hygiene, single responsibility, PR structure, branch naming, context creep, changelog, release notes, rebase, merge, squash, separability |
| Repository Platform Expert | GitHub Expert | `github_expert.md` | Git | GitHub, pull requests, PR comments, PR reviews, reviewer feedback, CI status checks, GitHub Actions, gh CLI, draft PRs, PR merging, branch protection, issue tracking, labels, milestones, releases, repository settings, code owners, review requests |
| GPU Expert | AMD GPU Expert | `amdgpu_expert.md` | HIP/AMD | GPU kernel correctness, race conditions, synchronization, cache coherency, memory model, LDS, shared memory, GPU memory management, floating-point precision, lambda captures in kernels, execution model, thread mapping, wavefront intrinsics, __restrict__, inline assembly, ISA analysis, architecture portability, crashes, hangs |
| MMA Expert | MMA Kernel Expert | `mma_expert.md` | HIP/AMD | MFMA, WMMA, matrix cores, MMA atom mapping, canonical machinery, interleaved layouts, wave-tile, macro-tile, thread-tile, static tile distribution, WarpDistributionEncoding, Rs Hs Ps Ys, replication_lengths, hierarchical_lengths, lane distribution, register distribution, RegisterMapper, matrix_coordinates, bijection encoding, K-vectorization, coalesced loads, LDS layout selection, double-buffered prefetch, software pipelining, register transpose, in-register reorder, dtype-graded reorder cost, free-relabel symmetry, A/B/C layout selection, derived C accumulator, MMA soundness, K-distribution, layout cost model, GEMM tiling, rocWMMA, gfx90a, gfx942, CDNA |
| LDS Expert | LDS Memory Expert | `lds_expert.md` | HIP/AMD | LDS, local data store, shared memory, groupshared, bank conflicts, bank count, bank width, address-to-bank mapping, ds_read, ds_write, ds_read_b32/b64/b128, half-wave serialization, per-phase arbitration, replay cycles, SQ_LDS_BANK_CONFLICT, SQ_LDS_ADDR_CONFLICT, SQ_LDS_IDX_ACTIVE, LDS padding, swizzle, XOR swizzle, K-stride aliasing, broadcast read, LDS occupancy, LDS allocation, double-buffering LDS, gfx90a, gfx942, gfx908, CDNA, RDNA, gfx11, gfx12, MI200, MI300 |
| Tiling Expert | Tiling Kernel Architect | `tiling_expert.md` | HIP/AMD | tiling api, kernel authoring, IRBuilder, end-to-end kernel design, pipeline design, macro tile, wave tile, thread tile, tensor descriptor, make_tensor_desc, make_tile_desc, make_window, load_fragment, store_fragment, make_fragment, TileMma, double-buffered prefetch, software pipelining, K-loop, CShuffle epilogue, LDS double buffer swap, global load, local store, wave read, C store, GEMM pipeline, algorithm to kernel, api gap proposal, novel algorithm, gfx90a, gfx942, CDNA |
| Debugger Expert | rocgdb Expert | `rocgdb_expert.md` | HIP/AMD | Interactive debugging, breakpoints, stack inspection, wavefront tracing, memory inspection, watchpoints, data breakpoints, assembly-level debugging, AMD_LOG_LEVEL, HIP_LAUNCH_BLOCKING, signal analysis, core dumps, printf debugging, binary analysis, bisection, multi-process debugging, container debugging, crashes, hangs, segfaults, memory aperture violations, out of memory, stack trace, TUI modes, disassembly |
| Debugger Expert | ROCr Debug Expert | `rocr_debug_expert.md` | HIP/AMD | Passive crash triage, GPU exception trapping, wavefront state dumps, fault diagnosis, assert traps, memory violations, code object analysis, debug info, GPU core dumps, librocm-debug-agent, no interactive debugger needed |
| Sanitizer Expert | Host Sanitizer Expert | `host_sanitizer_expert.md` | Host | ASan, UBSan, TSan, MSan, LSan, memory errors, heap overflow, stack overflow, use-after-free, uninitialized memory, data races, memory leaks, CPU-side sanitization, sanitizer compatibility |
| Sanitizer Expert | GPU Sanitizer Expert | `sanitizer_expert.md` | HIP/AMD | GPU AddressSanitizer, XNACK, device memory errors, heap buffer overflow, use-after-free on GPU, global memory, HSA_XNACK, instrumented ROCm libraries, GPU-side sanitization, xnack+ architectures |
| Profiling Expert | rocProf Expert | `rocprof_expert.md` | HIP/AMD | Hardware counters, rocprofv3, SQ/TCC/TCP/TA/TD/GRBM/SPI blocks, instruction mix, utilization metrics, cache hit/miss, memory throughput, API tracing, activity tracing, HIP/HSA tracing, rocTX markers, Perfetto, CSV, multi-pass collection, compute-bound vs memory-bound vs latency-bound, bottleneck classification |
| Profiling Expert | ROCm Compute Profiler Expert | `rocm_compute_profiler_expert.md` | HIP/AMD | rocprof-compute, Speed-of-Light, SOL analysis, roofline plots, VALU/MFMA/HBM/L2 utilization, per-hardware-block analysis, Grafana GUI, incremental profiling, kernel-level analysis, application replay, automated bottleneck identification |
| Profiling Expert | ROCm Systems Profiler Expert | `rocm_systems_profiler_expert.md` | HIP/AMD | rocprof-sys, system-wide profiling, call-stack sampling, binary instrumentation, host-device serialization, kernel overlap, transfer overlap, thread activity, load balance, MPI profiling, Perfetto timeline, causal profiling, CPU metrics, system-wide timeline |
| Build System Expert | CMake Expert | `cmake_expert.md` | CMake | CMakeLists.txt, targets, dependencies, compiler flags, build order, circular dependencies, library organization, external dependencies, install rules, packaging, feature toggles, conditional compilation, CI/CD integration, file placement, ROCm package integration, CMake deployment, find_package, config-file packages, export sets |
| Statistics Expert | Statistics Expert | `statistics_expert.md` | Any | Hypothesis testing, Wilcoxon signed-rank, TOST, non-inferiority, sample size, signal-to-noise, baseline stability, serial correlation, effect size, p-value, warmup policy, outlier policy, deterministic vs stochastic, experimental design |
| Project Management Expert | JIRA Expert | `jira_expert.md` | JIRA | JIRA, tickets, epics, stories, tasks, sub-tasks, story points, estimation, issue linking, workflow states, transitions, backlog hygiene, sprint retrospectives, JQL queries, dashboards, risk identification, release planning |
| Wiki Expert | Confluence Expert | `confluence_expert.md` | Confluence | Confluence storage format, XHTML macros, code blocks, syntax highlighting, tables, page layout, multi-column, cross-linking, JIRA integration, document conversion, content fidelity, page formatting, publishing |
| Style Expert | Style Guide Expert | `style_guide_expert.md` | Any | Code writing, code editing, code modification, code refactoring, code review, naming conventions, formatting, indentation, file structure, import style, include style, comment style, documentation style, code organization, cardinal rules, code style enforcement |

<!-- END GENERATED REGISTRY -->
