# gfx1250 MIOpen-facing convolutions: follow-on performance scope

**Status after F01–F09 (2026-09-26):** Sections 1–6 below are the original, pre-experiment scope, not current implementation status. The completed outcomes are in [implementation decisions, Follow-on F01–F09](gfx1250_miopen_convolution_implementation_decisions.md#follow-on-f01-f09-measured-decisions) and the [shape-strength audit](gfx1250_miopen_convolution_follow_on_shape_theories.md). In particular, F01 timing and F02 packing integration are now implemented. See [section 7](#7-review-of-the-completed-follow-on-results) for review inputs and [section 8](#8-next-scope-gfx1250-wmma-in-the-actual-convolution-winners) for the next WMMA-specific scope. Do not restart the old task list from its original priority table.

**Targeted workload update:** [section 9](#9-targeted-instances-for-tracelense_workload_conv_onlytxt) scopes the exact commands in `tracelense_workload_conv_only.txt`, including their default NCHW layouts and observed internal CK eligibility. The later [WM1/WM2 outcomes](gfx1250_miopen_convolution_implementation_decisions.md#wm1-scheduling-and-wm2-reuse-feasibility-on-the-actual-forward-winner) supersede section 8's experiment status: WM1a rejected, WM1b deferred, WM2 blocked on its runtime contract. Neither is restarted here.

**Earlier high-value pass:** [section 10](#10-high-value-depthwise-targets-from-reported-default-driver-latencies) records the nine-row analysis using older depthwise results. TC1/TC2 remain implemented. The newer hipconv results supersede that broad queue; use section11 for current priorities.

**Latest focus — supplement hipconv:** [section 11](#11-focused-residual-11x11-wrw-alongside-hipconv) records the BF16 G3/11x11 WRW reduction; [section 12](#12-measured-g192-stride-two-depthwise-wrw) records the G192/3x3/stride2 result. [Section 13](#13-family-wide-ck-performance-phase-for-the-nchw-workload) starts a latency-weighted, reusable CK family phase. The superseded broad FWD/BWD queue stays closed; hipconv's winning paths are untouched.

## Recommendation

Do not restart the rejected v3, N16/N32, merge16, or direct-store sweeps. First make prospective cross-family timings comparable. Then prioritize **backward-data residue-window tightening**, **proof-based removal of general-convolution clears**, and **a genuinely direct depthwise WRW reduction**. Treat the positive forward-packing result as a separate integration project with an explicit workspace contract.

This follows [implementation decisions and trial evidence](gfx1250_miopen_convolution_implementation_decisions.md), not the earlier unimplemented wish list. Reported results and retained changes are accepted as the baseline; none were rerun to confirm them. This investigation changed no kernels and ran no GPU benchmarks, builds, or project tests. New evidence consists of current-source inspection, arithmetic experiments, workload-shape analysis, and extraction of resource fields from existing traces.

**Every future latency benefit is [INFERENCE; unmeasured].** Reduced modeled work is not a speedup. Existing reported timings remain outputs of their original measurement paths; the timing observation below qualifies prospective comparisons, not the fact those results were observed.

Scope remains convolution-only, gfx1250, and MIOpen-required CK layouts: 2D NHWGC and 3D NDHWGC, FP16/BF16/FP32 with existing numerical contracts. Keep **CK-profiler-first** execution. MIOpen plugin, ABI, selection and layout-conversion work is identified separately, not silently folded into a CK-only task.

Paths beginning `include/`, `library/`, `profiler/`, `test/`, or `example/` are relative to `projects/composablekernel` unless stated otherwise. `MI` means `projects/miopen`. The compact source index in section 4 defines the short keys used in work packages. Line numbers refer to the inspected working tree; containing HEAD is `e9a612e0b9e826af2e7469ff9f9b872b533a84f1`, which does not identify uncommitted implementation changes.

## 1. Baseline: what not to propose again

| Existing result | Follow-on interpretation |
|---|---|
| W01 safe backward-data/WRW clear elision retained, including untimed preprocessing fix | Preserve these guards/tests. General-convolution coverage needs a new mathematical proof, not wider conditionals. |
| W02 normalized single-target BF16 compile flags retained | Finished build-parity work; do not add another architecture guard. |
| W05 split-1 scalar WRW retained | Next question is candidate-local auto-split admission, not another scalar row. |
| W11 bounded 2D split2/4 backward-data retained | Preserve paired boundaries, no-D/plain operation, architecture and determinism restrictions. No general split-K unlock. |
| W03 forward v3 lost after the 36-block 3x3 check | Long reduction was tested. Another v3 row needs a measured resource/scheduling mechanism absent from that trial. |
| W04 narrow tile portfolio and W08 direct-store port lost | Do not treat untried neighboring dimensions as sufficient justification to restart the same sweep. |
| W06 FP16 merge16 lost; BF16 merge remains | Direct group-local reduction is a different algorithm; copying another dense merge row is not. |
| W07 factor-4 packing won its reported CK-only example | Integration of an existing win. Workspace sizing and pointer installation are both missing in MIOpen forward. |
| W09 single-buffer async lost | It did not overlap producer and consumer through ping-pong buffers. A redesigned trial still needs a bottleneck and resource gate. |
| W10 standalone LDS transpose mapping passed, no convolution result | Neither rejected for convolution performance nor ready to use. An actual operand-layout bottleneck/map is missing. |

The new `CK_PROFILER_MIOPEN_LAYOUTS_ONLY=ON` subset is the correct laboratory, with `MIOPEN_REQ_LIBS_ONLY=OFF`; the latter disables the profiler. Reuse `profiler/README.md:22-42`. Do not build an unrelated full-operation profiler or a MIOpen plugin merely to enumerate these CK candidates.

## 2. Priority and ownership

| ID | Priority | Classification | Deliverable / first stop gate |
|---|---|---|---|
| F01 | P0 prerequisite | NEW measurement finding | Common raw invocation metric and unique candidate identity; no cross-family promotion before this. |
| F02 | P1 integration | EXISTING WIN | Safe forward packing workspace contract; never register it alone. |
| F03 | P1 CK | NEW | Exact per-residue backward-data windows; preserve every valid coordinate/contribution. |
| F04 | P1 CK | CONTINUATION WITH NEW PROOF | Complete-output-write predicate beyond 1x1; reject any uncovered logical output. |
| F05 | P2 CK | NEW | Group-vectorized depthwise WRW reduction without dense off-diagonal products. |
| F06 | P2 CK | CONTINUATION WITH NEW POLICY | Auto selection respects a split-1-only candidate's legal domain. |
| F07 | P2 CK | NEW | Convolution-aware CTA ordering at unchanged tile/ISA; require cache/traffic evidence. |
| F08 | P2 CK | NEW | Incremental convolution addressing in a current winner; require emitted address-work evidence. |
| F09 | P3 CK | NEW, codegen-gated | Simplify slice lookup/metadata only if compilation has not already removed the overhead. |

F03 and F04 share transform/coverage reasoning and should have one integration owner, but must be measured independently. F05 owns a distinct WRW path. F06 is a host/candidate-policy change. F07 changes only block mapping; F08 changes transfer/address progression. Do not combine map, tile, pipeline and epilogue changes into one experiment. F02 is separate MIOpen ownership and can be scoped without touching CK arithmetic.

### F01 — Make future convolution comparisons use a common metric

**Source finding.** The convolution profilers pass `StreamConfig{nullptr, time_kernel, 0, 5, 50, time_kernel}`: timing also enables the cache-flush path. Different device families implement that path differently:

- `K-LAUNCH:197-261` times preprocessing plus the kernel normally. Its cache-flush variant at `:279-372` subtracts a separately measured icache-flush interval.
- Direct WMMA WRW uses that variant (`D-W:899-940`).
- Two-stage WMMA WRW uses `ck::utility::launch_and_time_kernel_with_preprocess<false>` (`D-W2:950-1003`). `F-CACHE:452-518` records the start event **before** preprocessing, then subtracts a fixed `0.005` ms per iteration for 80 multiprocessors, otherwise `0.01` ms. The template's `false` argument does not move the active start event after preprocessing. This is an allowance, not a measurement of this invocation's cache overhead.
- Two-stage WRW then sums separately timed GEMM and conversion stages (`D-W2:1200-1224`); this is not one event interval enclosing the whole invocation.

Thus W05's reported direct-versus-two-stage numbers used different corrections. They remain the observed outputs. Do not add/subtract constants to the historical CSV to manufacture a corrected result; its raw elapsed/cache-component data are absent.

**Additional identity finding.** `P-F:467-517` invokes the first factory op before a loop that starts again at index zero. If supported, that op is numbered/executed twice. Current `--instance` and `--list-instances` agree with this numbering, but the list is not a count of unique physical candidates. Fix or explicitly deduplicate this before training a selector or summarizing candidate coverage. Preserve warm-up behavior and clear distinction between registered, supported, uniquely supported and executed counts.

**Smallest future change.** In the convolution profiler, separate cache policy from the timing flag. Add a common **raw same-stream event interval around complete untimed `Invoker::Run` calls**, with internal timing/cache instrumentation disabled, after compilation warm-up. This includes required clears, packing and cast kernels. Report hot-reuse and any deliberately rotated/cold policy separately; place artificial flushes outside the raw interval and apply the same policy to both candidates. Keep existing numbers labeled as legacy metrics while comparing the new metric, rather than silently redefining historical output. If this affects public profiler output, document the contract and migrate consumers deliberately.

**Acceptance.** A two-stage trace has all required kernels inside the raw interval; a split-atomic invocation still clears on every measured iteration; no fixed-offset subtraction in the raw metric; one record per exact candidate and requested/effective split. No changes to production kernel math, no generic timing-system rewrite. Test the observable timing/reporting contract and candidate enumeration, not source spelling. Only after this gate should a new cross-family result change priorities.

### F02 — Integrate the existing forward-packing win safely

**Why still worth doing.** The decisions report records factor-4 FP16 packing at G32/N64, per-group C=K=4, 3x3/s1/p1, 28x28: pack+GEMM median 0.04527 ms versus 0.05858 ms for the best eligible unpacked factory candidate. This is an existing CK-only result, not a new measurement or an end-to-end MIOpen claim. Use F01 for prospective comparisons.

**Current contract, not just one missing callback.** In MI:

1. `src/ck_impl/ck_grouped_conv_fwd_impl.cpp:305-319` returns zero from `ck_impl_fwd_get_workspace_size`.
2. `src/solver/conv/conv_hip_implicit_gemm_grouped_fwd_xdlops.cpp:445-460` uses only layout-transform workspace at problem level.
3. `src/ck_impl/implicitgemm_ck_util.hpp:1122-1292` installs CK workspace for WRW, not the forward NHWC branch. Its NCHW path at `:949-1100` similarly reserves/sets a fourth CK region only for WRW.
4. `src/include/miopen/solver/implicitgemm_ck_util_common.hpp:371-426,492-520` manages the three transformed tensors and optional fourth region; channels-last non-WRW currently returns zero.
5. Forward dry support queries use null data pointers (`src/ck_impl/ck_grouped_conv_fwd_impl.cpp:161-165`). CK's packing path can pass eligibility without a workspace pointer; the null-workspace check in `D-F:2115-2127` is specific to internal transpose. Dry eligibility must remain possible, but an actual nonzero-scratch launch must not accept missing/undersized scratch.

**Bounded implementation contract.** Keep the existing 2D, Default, no-D, packed-weight, divisible-G packing restrictions (`D-F:365-376,957-964,1603-1645,1864-1875`). The problem-level callback has no kernel ID, so return a conservative supported-candidate maximum; query the **selected** argument's actual workspace for invocation. In NHWC, set the selected CK pointer and validate capacity. In NCHW, place packed-weight scratch in a disjoint aligned fourth region after conversion buffers, keep it alive until pack+GEMM completes, and do not alias transformed weights. Preserve the argument's owned narrowed-array lifetime.

For the measured shape, source weights are `32*4*9*4*2 = 9216` bytes; factor-4 packed storage is **36,864 bytes**, before any allocation alignment. Check this against the concrete argument rather than hardcoding it. Source expression `D-F:957-964` sizes the expanded group/channel geometry. A problem-level allocation must cover the selected solution's scratch plus any external transforms.

Follow the versioning rule in MI `src/include/miopen/solver/ck_impl_interface.hpp:25-31,104-108`; coordinate the callback semantics and matching plugin/loader. `src/solver/ck_impl_lib_loader.cpp:281-289,318-409,522-547` validates/binds that contract. Do not ship a registry change independently of this cutover. Test selected-candidate execution with the real allocation path before enabling production enumeration.

**Acceptance.** No-pack controls still request zero native CK scratch; packed requests have sufficient aligned storage; repeated calls with changing weights repack correctly; poisoned scratch and grouped outputs verify; NHWC and externally transformed NCHW are separate cases. No 3D/fusion registration, no cached mutable weights. MIOpen end-to-end validation is a separately scheduled dependency after CK validation.

**Important non-bug.** Retained W11 NHWGC split2/4 backward-data writes atomically into E and reports zero CK scratch (`D-B:335-351,1095-1144,2284-2291`). The missing non-WRW workspace installation is **not** a demonstrated W11 null-scratch bug. Do not block selection of that retained path on F02.

### F03 — Tighten backward-data windows per frozen residue

**New mechanism.** `B-TRANS:704-779,1272-1348` calculates one conservative H/W tilde window for every residue, although residue `t` writes input coordinate

`i = t*dilation + h*stride - left_pad`.

Intersect the current `[lo,hi)` window with the exact valid-input range:

```text
lo_t = max(lo, ceil_div_signed(left_pad - t*dilation, stride))
hi_t = min(hi, ceil_div_signed(input_length + left_pad - t*dilation, stride))
```

Clamp empty intersections consistently. Use signed-safe division; C++ truncation toward zero is not mathematical ceiling/floor for all inputs. Apply the same shifted window to A, C and applicable D descriptors, preserving the existing per-residue reduction slices and all custom/transpose branches. The initial prototype should be plain 2D, with independent verification before affecting other users of the shared transform.

**Executed arithmetic evidence.** A host-side model of the current formulas and their intersection checked 27,240 legal axis configurations and 49,359 residues: input lengths 1–24, filters 1–7, stride 1–4, dilation 1–3, independent left/right pads 0–3, positive output extent. Every tightened range preserved the model's current valid input-coordinate set and retained every input position with at least one brute-force output/filter contribution. This did not compare compiled C++ descriptor offsets or individual floating-point summands, and is **not** proof of a changed GPU kernel.

| Fixture, per-group C tile fits one GEMM N tile | Current M per residue | Tight M per residue | Modeled total M64 CTAs across four slices |
|---|---:|---:|---:|
| N4, input8x8, 2x2/s2/p0 | 100 | 64 | 8 → 4 |
| N4, input8x8, 3x3/s2/p1 | 100 | 64 | 8 → 4 |
| N42, input240x320, 3x3/s2/p1 | 818,202 | 806,400 | 51,140 → 50,400 |

The last case is a real command in `model_e_miopen_convolution_commands_nhwc.json:562-565`: BF16, G1/C24/K96, backward-data. Its reduction in modeled CTAs is only about 1.45%, not the tiny fixture's 2x ratio and not a latency prediction. Four slices can share **one grouped launch**; do not call them four mandatory launches.

**First implementation experiment.** Emit old/new descriptor lengths and valid address/contribution tuples for the small cases, odd extents and gcd/stride/dilation controls. Then one already-supported Default candidate, unchanged tile/pipeline, on the real case. Compare all relevant old-WMMA/XDL-ported competitors after F01.

**Acceptance/stop.** No missing or duplicated contribution, identical logical result, correct D/fused behavior for any shared users affected, unchanged split policy. Require a reduction in padded work/CTA count or emitted address/mask work. Stop if padding absorbs the change or the actual kernel gains nothing. No ISA addition is needed.

### F04 — Prove full Set coverage for general backward-data convolutions

**Different from relaxing W01.** `D-B:726-817` retains clear elision only for its proven 1x1 regime. Some Default split-1 problems have complete coverage by disjoint residue slices. The 2x2/s2/p0 even-extent example and the 3x3/s2/p1 real shape in F03 are candidates; stride-2 1x1 remains a holes counterexample.

**Proof obligation.** Derive an inexpensive host predicate from the descriptor mathematics, not a table of lucky shapes. For every logical E element, exactly one launched Set slice must write it, including valid zero-result writes; every needed residue must have an admitted slice, all boundary windows must be covered, and empty-dot or gcd-induced holes must retain zeroing. Limit the first version to homogeneous FP16/BF16, packed disjoint E, positive matching extents, no D/alias, split1, and plain layout. F03's exact ranges help reasoning but are not themselves a proof that every logical output is visited.

The arithmetic model found full axis-product coverage for those two fixture classes. For real N42/C24/240x320 BF16, the logical dX allocation is **154,828,800 bytes**. That is the write volume currently subject to initialization, not measured HBM traffic or time saved.

**Acceptance.** Poisoned output on repeated CK calls, CPU comparison, partial tiles and odd extents; negative controls for stride>filter, dilation/gcd holes, empty slices, D aliasing, irregular strides, mismatched extents and split2/4. Trace the retained/removable clear only after the new predicate is implemented. Preserve one grouped launch where it already batches slices; do not change scheduling simultaneously. Extend to 3D only after three-axis coverage proof and actual 3D correctness.

### F05 — True group-local depthwise WRW, not another dense merge row

**New algorithmic direction.** With C/K per group both 1, depthwise 3x3 has nine weights per group. The existing merged-16 geometry processes a dense M16/N144 space of 2,304 positions for 144 logical weights; masking off-diagonal stores does not remove dense matrix products (`W-TRANS:175-221,527-555`). The rejected FP16 merge16 trial is accepted as a failure of that row. This task removes the off-diagonal arithmetic rather than trying to tune it away.

**Real first regimes.** Use BF16 corpus commands M06/M14/M15/M25 from `miopen_wrw_shapes.txt`: G192/256/512, N42, per-group C=K=1, 3x3, reduction R=201,600 or 50,400. FP16 is a secondary precision check, not the only target. G3/11x11 M19 is a low-group, large-filter negative/later regime, not an assumed beneficiary.

**Concrete mapping experiment.** Start with one group chunk and one/few filter coordinates per CTA, FP32 accumulation, split1 final low-precision Set. Partition lanes into `(group_lane,reduction_lane)`: neighboring groups are contiguous in packed NHWGC input/dY, whereas a naive one-group CTA has strided group-sized accesses. Explore a small 8/16-group lane chunk with the remaining block dimension covering R; reduce separately for each group. Do not put all groups and all nine filters into so few CTAs that the GPU is starved. Model

`CTAs = ceil(G/group_chunk) * ceil(filter_volume/filter_tile)`

before choosing the mapping. For group_chunk16/filter_tile1, G192/256/512 gives 108/144/288 CTAs at split1; actual occupancy/latency still needs measurement. Output GKYXC is group-major: group-vectorized input loads imply strided final group stores. A one-time scalar store can be preferable to an elaborate output transpose; measure before adding one.

**Reuse:** convolution indexing/group strides in `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp:822-898,960-1010`, and `PartitionedBlockwiseReduction_v2` in `include/ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp:99-154`. Existing DL is still a padded GEMM implementation, not this proposed direct reduction. Physical NHWGC order is in `include/ck/library/utility/convolution_host_tensor_descriptor_helper.hpp:186-202`.

**Acceptance/stop.** Unique weight ownership, no off-diagonal compute, group-tail safety, pads/stride1/2, FP32 accumulation with one final conversion, no accidental atomic nondeterminism in split1. Validate transaction/address mapping and actual register/spill pressure, then compare complete calls against retained scalar direct, existing BF16 merged and all other eligible kernels. Stop if rereading input/dY, strided stores or long serial R defeats the arithmetic savings. Any later R partitioning needs an explicit reduction/workspace contract; do not quietly turn this into low-precision atomics or combine it with the deterministic project below.

### F06 — Auto split must respect the legal domain of the retained scalar candidate

**Source mismatch.** The retained `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3_Split1` (`I-W:43-82`) rejects an effective `k_batch_ != 1`. Its inherited argument still runs the generic auto-split calculation first (`D-W:539-570`). Auto can therefore remove a valid split1-only candidate from an auto-only request; a fixed1 request still works.

For the already-reported W05 geometry G3/N2/K5/C3, 3x3/5x6/s1/p1, the scalar tile is M16/N16/K32:

```text
R = 2*5*6 = 60
grid = G*ceil(K/16)*ceil(C*3*3/16) = 6
reduction cap = ceil(60/32) = 2
```

When `max_active_blocks_per_CU * CU_count >= 12`, the existing occupancy formula reaches effective split2 and the split1 wrapper rejects it. This conditional deduction does not claim occupancy was measured or that default MIOpen selection misses the candidate.

**First change.** Make an auto request on this constrained candidate mean its best **legal** split, namely 1, while still rejecting explicitly requested split2/4. Implement before argument construction/occupancy selection, not by mutating `k_batch_` after descriptors are formed. Reuse candidate-specific legality rather than changing `split_k_utils.hpp` globally. Preserve the existing ordinary direct cap128, two-stage occupancy-squared cap, and their competitors.

**MIOpen nuance.** MI `src/include/miopen/solver/implicitgemm_ck_util_common.hpp:53-75` cycles `1,2,...,128,-1,1`. Both 2D/3D WRW generic searches use it even though the AI heuristic config says `supports_split_k_autodeduce=false`. Defaults and deterministic search remain split1. Thus this is auto-mode consistency, **not proof of a missing fixed1 solution in exhaustive search**.

**Acceptance/stop.** Requested versus effective split is explicit in records; fixed1 and auto produce equivalent legal arguments; explicit split>1 still rejects; no scalar atomic instantiation. Require an actual auto-mode user/selection benefit before changing production policy. Stop if the surrounding search already enumerates fixed1 and the change adds no useful behavior, or if long-R competitors remain better. Do not retest the reported W05 result simply to confirm it.

### F07 — Change convolution CTA order while leaving tile geometry unchanged

**New knob.** Existing old-WMMA grid code selects `BlockToCTileMap_M00_N0_M01Adapt` (`MAP:115-246`; `G-OLD:905-947`). Its default M01=8 visits several spatial-M tiles for a filter/output-channel tile before moving along N. M01=1 instead visits output-channel tiles for the same spatial-M tile consecutively. This exchanges weight reuse against receptive-field/input reuse without another N16/N32 portfolio.

**Executed arithmetic probe.** For M0=23/N0=4, M01=1/4/8/16 all produced bijections over 92 tile coordinates, including partial final M groups. First eight coordinates:

```text
M01=8: (0,0),(1,0),(2,0),(3,0),(4,0),(5,0),(6,0),(7,0)
M01=1: (0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)
```

This validates the scalar mapping model, not GPU execution order. Block IDs do not guarantee co-residency or a particular scheduler order.

**First experiment.** Choose a current supported winner with multiple M and N tiles on a high-spatial BF16 corpus case; trace the selected old-WMMA or XDL-ported family's actual map, not just its name. Vary one map parameter at identical tile/dtype/pipeline. Compute per-group weights `K*C*filter_volume*sizeof(T)`, per-CTA input demand before spatial reuse, and cache working sets. N0=1 or a resident weight/input set may leave no opportunity. Capture relevant cache/traffic counters where available; otherwise use carefully controlled raw timing and label the mechanism unconfirmed.

**Acceptance/stop.** Every tile/group exactly once, partial tiles preserved, no cross-group addresses, no numerical change. Stop without a reproducible raw-latency gain over best eligible family or if changed order only improves the artificial cache policy from F01. No cache-hint, tile-size or cluster-launch changes in this experiment.

### F08 — Reduce address work or recover spatial reuse in the actual winning forward path

**New source-level opportunity, not an observed instruction bottleneck.** `F-TRANS:955-987,1516-1526` composes pad/embed/merge transforms for forward Default: GEMM M=(batch,Ho,Wo), reduction=(filterY,filterX,C). The compiler may already hoist or simplify those transforms. Source complexity alone is not evidence of costly divides or modulo instructions.

**Entry gate.** Identify the real winner for an existing BF16 high-spatial 3x3 command, then inspect that code object for address generation, quotient/remainder, boundary predicates and input traffic relative to WMMA. Keep exact tile, specialization, backend, split, compiler and code-object identity. Do not use the slower removed v3 tile as the only baseline.

**Bounded change if the gate passes.** Specialize incremental C/filter/spatial coordinate progression in the existing transfer path. Preserve vector-contiguous segments and use a correct carry/reset when crossing C, X, Y, output-row and batch boundaries. Consider interior versus boundary address handling only if it does not add launch overhead or register state that overwhelms the saving. Initially 2D packed 3x3/s1, then prove stride/dilation and edges before extending eligibility.

Adjacent output columns share `Y*max(0,X-strideW)*C` logical input elements away from edges. This is potential reuse, not the number of saved physical transactions. The first task is address-work reduction; explicit shared-memory spatial staging is a separate algorithm if traffic analysis later warrants it.

**Acceptance/stop.** CPU-correct padded/odd/tail/group cases; reduced emitted address/mask work or measured traffic at unchanged arithmetic; no new spills. Stop if compiler already hoists the work, if additional coordinate state increases VGPR pressure, or if the whole invocation loses. This is distinct from W03 pipeline staging and W04 narrow tile registration.

### F09 — Specialize backward-data slice lookup only after inspecting codegen

**Source fact.** `D-B:64-111` performs a binary search of host-passed `GemmArgs` ranges; `:555-596,849-1045,1305-1396` stores/passes descriptors in sets of up to 32. The XDL-v1 sibling already bypasses lookup for `gemms_count==1` (`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp:139-163`). Compiler specialization may already make some native-v3 cases cheap.

**First probe, no new framework.** Inspect kernarg size, scalar descriptor loads, branch instructions, SGPR/VGPR/scratch and host argument construction for a one-slice 1x1 and four-slice stride2 Default instance. Only then add a one-slice fast path or a compact piece of lookup metadata using the existing convention. Do not move all descriptors to global memory without accounting for additional loads/lifetime.

Four 3x3/s2/p1 residues have different reduction lengths: at K96, 12/6/6/3 K32 blocks. Equal-size division or one universal main-loop decision is not a safe replacement for range lookup. Preserve `BlockStart_`/`BlockEnd_`, empty slices, mixed main-loop state and split indexing. Typical four/eight-slice 2D/3D cases already fit one set; do not invent a many-launch problem.

**Existing resource evidence, not causality.** Recorded final traces contain 72 VGPRs for the 1x1 kernel and 96 for the stride2 kernel; both report 8,704 LDS bytes and zero scratch. They are different generated kernels. This does not prove lookup caused the difference or that lowering VGPRs changes occupancy. Stop if compiled lookup/metadata cost is negligible or already optimized away.

## 3. Conditional ISA/reduction queue — not immediate implementation tasks

### A. Overlapped async only after a bottleneck/resource gate

The supplied CDNA5 ISA section 10.8, lines 5849-5911, permits async global/LDS movement but tracks it with ASYNCcnt and allows LDS access reordering. Existing wrappers remain available. The prior single-buffer prototype is not an overlapped producer/consumer design.

Before another prototype, show synchronous staging/load waits in the **winner**, not just in removed native-v3 code. For a same-precision tile with both operands in LDS, model input storage as `(Mtile+Ntile)*Ktile*element_bytes`; ping-pong doubles it. For M128/N64/K32 FP16/BF16 this is 12,288 → 24,576 bytes before other concurrent storage/alignment. CShuffle reuse may avoid summing input and epilogue storage, but do not assume lifetimes overlap safely. Check actual occupancy and registers before implementing.

Then one vector-safe im2col candidate can overlap next-tile async production with current WMMA computation. Require emitted async instructions, ASYNCcnt completion before read, and consumer DS/barrier completion before buffer reuse, including zero-filled invalid lanes. Long-K 3x3 becomes meaningful only after its address mapping is implemented. Stop without a proved staging/load bottleneck or if ping-pong reduces occupancy enough to lose. No new instruction wrappers and no repeated single-buffer trial.

### B. LDS transpose needs a physical operand map

CDNA5 section 11.2.4, lines 6627-6644: LDS transpose is wave32 and ignores EXEC. Existing tagged-matrix proof is accepted, but the old-family B LDS layout `[K0,N,K1]` with K1=8 (`G-OLD:409-430`) does not automatically match a contiguous 16-column N-by-K tile for that operation.

Require an actual permutation/bank-cost hotspot, then derive the exact lane/address/fragment map and all inactive-lane addresses. A candidate `[K][N]` physical arrangement, byte address `2*(k*Nstride+n)`, is a hypothesis to validate, not permission to replace the descriptor. Include cost of producing the new LDS layout. Stop if no software permutation or LDS inefficiency is removed. W10 remains open but is below the concrete non-ISA work above.

### C. Deterministic parallel WRW is a separate contract-sized project

Existing two-stage FP32 atomics are not deterministic. If a separately requested deterministic workload shows long-R/small-grid split1 underutilization, consider independent FP32 partials `[split,group,K,filter...,C]` plus a fixed-order reduction and final cast. Scratch is approximately `4*S*G*K*C*filter_volume` bytes, versus one weight-sized FP32 buffer today, plus alignment. M18 has R=3,225,600 and only 648 weights: a concrete small-output candidate, not evidence of speedup.

Preserve an explicit numerical/reproducibility contract: fixed reduction order can be repeatable without being bitwise identical to serial split1. Require agreed tolerances, workspace limits, CPU correctness, repeatability and full two-pass latency. MIOpen deterministic admission/search would be a separate integration change. Stop without a demonstrated deterministic bottleneck. Do not solve it by allowing FP16/BF16 atomic splits under deterministic mode.

Cluster multicast/TDM remain deferred: neither missing wrappers nor a proven next convolution bottleneck. F07/F08 should establish whether duplicated traffic survives ordinary cache/address improvements before taking on cluster participation, launch, tensor-descriptor and barrier-lifetime contracts.

## 4. Exact source starting points

All entries below are relative to CK. Read cited sections first; follow only the selected candidate's implementation path.

| Key | File |
|---|---|
| D-F | `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp` |
| D-B | `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp` |
| D-W | `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_wmma_cshuffle_v3.hpp` |
| D-W2 | `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_wmma_cshuffle_v3.hpp` |
| I-W | `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp` |
| B-TRANS | `include/ck/tensor_operation/operator_transform/transform_conv_bwd_data_to_gemm_v1.hpp` |
| F-TRANS | `include/ck/tensor_operation/operator_transform/transform_conv_fwd_to_gemm.hpp` |
| W-TRANS | `include/ck/tensor_operation/operator_transform/transform_conv_bwd_weight_to_gemm_v2.hpp` |
| MAP | `include/ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp` |
| G-OLD | `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp` |
| K-LAUNCH | `include/ck/host_utility/kernel_launch.hpp` |
| F-CACHE | `include/ck/host_utility/flush_cache.hpp` |
| P-F | `profiler/include/profiler/profile_grouped_conv_fwd_impl.hpp` |

Other old-family entry points: `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp:145-240,375-385,510-570`; `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_instance.hpp:47-83`; `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp:63-98,132-165`. Their existence does not identify which one wins a particular command.

Local ISA source: `/home/sgundabo/MISA/amd-instinct-cdna5-instruction-set-architecture.md`. Also relevant: section 5.7 dependency counters, lines 2377-2428; matrix-scheduling caution at 2500-2507; WMMA table at 4194-4250; floating atomics at 6648-6689. Use targeted sections, not the entire document. No new ISA assembly/runtime claim is made in this follow-on investigation.

## 5. Workload selection and executable handoff

### Existing shapes, not historical timings

New metadata analysis of the monorepo-root `model_e_miopen_convolution_commands_nhwc.json` found 112 entries, of which **86** are convolution commands: 29 forward, 28 backward-data, 29 WRW, all BF16. There are 72 G1 commands, 12 with G192/256/512, and two G3 commands. Non-convolutions were excluded. Counts are commands, not invocation frequencies or current runtime weights. Historical statuses/timings in that file and CK's `miopen_baseline_results.csv` are not a new baseline.

`projects/composablekernel/miopen_wrw_shapes.txt` already supplies 29 `name|ckProfiler arguments` records. Use its numeric per-group C/K arguments; descriptive labels may show total channels. Selected derived geometry:

| IDs | G | Per-group C/K | Filter / output extent | R=N*Ho*Wo | Use |
|---|---:|---|---|---:|---|
| M06 | 192 | 1/1 | 3x3, 60x80, N42 | 201,600 | F05 stride2 depthwise |
| M14 | 256 | 1/1 | 3x3, 60x80, N42 | 201,600 | F05 stride1 depthwise |
| M15 / M25 | 256 / 512 | 1/1 | 3x3, 30x40, N42 | 50,400 | F05 group/stride controls |
| M18 | 1 | 3/24 | 3x3, 240x320, N42 | 3,225,600 | Odd-channel long-R negative for F06; conditional reduction project |
| M19 | 3 | 1/1 | 11x11, 480x640, N42 | 12,902,400 | Low-group/large-filter negative for F05 |

Do not start with the largest memory case blindly: calculate input/output/workspace bytes and check capacity before executing. Keep synthetic fixtures for proof and use real commands for priority. Hold out entire shape families when changing ranking; do not tune and report on the same synthetic handful.

### Existing environment and prospective commands

The current filtered build was located at `/tmp/ck-gfx1250-rocm102-profiler/bin/ckProfiler`; its cache records ROCm 10.2, gfx1250, FP16/BF16/FP32, the layout-only profiler option ON, and MIOpen-only packaging OFF. It was not rebuilt or executed in this investigation. Temporary builds can disappear; use the documented profiler recipe to recreate one in a fresh directory when needed.

Example **future enumeration**, not executed here, for the real F03/F04 backward-data shape:

```sh
/tmp/ck-gfx1250-rocm102-profiler/bin/ckProfiler grouped_conv_bwd_data \
  2 1 1 2 0 0 2 1 42 96 24 3 3 240 320 2 2 1 1 1 1 1 1 1 \
  --list-instances
```

Here dtype2=BF16, layout1=NHWGC, split1; no forward index-type argument. For WRW use dtype5 for homogeneous BF16 and layout2. The existing 29-row WRW file uses `-1` auto. Do not translate it to dtype2, which means BF16/FP32/BF16 there. Forward alone has the extra indexing argument. Record requested and effective split.

After enumeration, select the intended type and use its current supported ID. IDs are shape/build dependent and F01 may remove a duplicate; never reuse a saved numeric ID across builds. `--list-instances` still constructs arguments and checks eligibility, so it is not a substitute for a pure source/arithmetic probe. New performance runs must use the F01 common metric rather than treating a legacy `time=1` number as raw full-call latency.

### Evidence record required for every next candidate

- Revision plus working-tree patch, compiler/toolchain, build options, GPU identity and clock/power/cache policy.
- Exact direction, dtype, layout, per-group channels, dimensions/pads/stride/dilation, requested/effective split, type string and code-object identity.
- Unique supported count and actually executed candidate; explicit CPU correctness. Reuse existing gfx1250 CPU-reference convention.
- Raw complete invocation intervals and separately labeled stage/cache-adjusted numbers; all required clear/pack/cast work included. Alternate candidates after compile warm-up; median and spread over independent runs.
- Actual LDS/register/scratch metadata; kernarg size/disassembly where addressing or lookup is the proposed mechanism. Hardware counters only when available, with names and tool versions; metadata is not an occupancy or cache-stall counter.
- A matched-instance comparison isolates mechanism; best eligible family comparison establishes utility. Neither alone proves MIOpen selection or end-to-end gain.
- Conclude **keep / reject / prerequisite missing**. A negative result is a completed experiment; do not leave broad opt-in scaffolding or register a slower candidate to claim progress.

For retained W05/W11 production selection, separately trace MI `DefaultKernelFromList`, `SetNextValue`, `IsValid` and exact type+split IDs before changing rankings. W11's C64/K512 BF16 counterexample in the decisions report forbids universal split4 preference. Deterministic policy must select split1. Do not force new FP32, TF32, sparse or quantized numerical contracts into this work.

## 6. Evidence produced in this investigation

1. Read implementation decisions and accepted reported retained/retired outcomes without rerunning their tests or benchmarks.
2. Independently inspected current convolution, transfer, timing and MIOpen workspace/selection code; found the fixed timing allowance, duplicate forward enumeration, exact residue-window opportunity and candidate-auto legality mismatch.
3. Ran the bounded residue arithmetic enumeration and the three M/CTA/coverage calculations recorded in F03/F04. It checks equations only; it is not a permanent test or a kernel result.
4. Ran the CTA-map permutation model for M0=23/N0=4 and M01=1/4/8/16; all 92 coordinates were unique/covered. It does not establish hardware cache locality.
5. Parsed existing workload commands to derive the corpus counts and WRW geometry above, without treating historical timing/status fields as current.
6. Extracted these **existing trace fields**, without launching anything:

| Recorded trace | LDS bytes | Scratch | VGPR count | Accum-VGPR count | SGPR count | Workgroup X |
|---|---:|---:|---:|---:|---:|---:|
| Final backward-data 1x1 split1 | 8,704 | 0 | 72 | 0 | 128 | 64 |
| Final backward-data stride2 split1 | 8,704 | 0 | 96 | 0 | 128 | 64 |

Sources: `/tmp/ck-gfx1250-rocm102-traces/final_bwd_1x1_split1_kernel_trace.csv` and `/tmp/ck-gfx1250-rocm102-traces/final_bwd_stride2_split1_kernel_trace.csv`. These are different generated kernels, not matched before/after variants; no cause, occupancy, speed, clock state or cache-counter conclusion follows. No standalone code object with proved provenance for the current winner was established. That remains the explicit gate for F08/F09 and the conditional ISA work.

## 7. Review of the completed follow-on results

### Decision

**Keep F01, F02 and the bounded BF16 F05 row; accept the reported rejected trials. The results do not establish a gfx1250 WMMA performance ceiling.** This review accepts the reported runs without repeating their tests or benchmarks. New evidence below is source/disassembly inspection, host arithmetic and isolated compiler/assembler probes, not a new convolution timing result.

| Result | Input on interpretation / next action |
|---|---|
| F01 complete-call timing and unique enumeration | Correct foundation. Keep `--raw-invocation`; do not revise historical `Perf:` numbers or introduce another competing default metric. The implementation synchronizes each of 50 intervals, so this is an isolated complete-invocation measurement, **not saturated WMMA throughput**. Add a separately labelled kernel timeline/queued diagnostic when testing instruction scheduling. |
| F02 packing | The new raw medians, 0.052956 versus 0.049173 ms, mean **7.14% lower latency**, or 1.077x speed ratio, on that FP16 shape. Do not compare this percentage directly with the earlier cache-adjusted packing ratio. Matching local MIOpen NHWC/NCHW selected-path correctness now exists; automatic selection and conversion-inclusive performance remain separate gates. |
| F05 short-R BF16 depthwise WRW | Keep its bounded admission and existing competitors. The stronger R256/stride2 wins do not make G512/R512 stride1 a reliable universal win: the longer series overlaps and the shorter series reversed. All 86 convolution commands in the inspected corpus are BF16, but **none is in the retained high-G/short-R class**. This is a useful CK coverage win, not evidence of a model-level gain or improved WMMA utilization; the new kernel is a direct reduction. |
| F03/F04 | The trials failed their best-family utility gate; removing them was justified. F04's 0.240788 to 0.275671 ms regression after removing a clear is worth one *causal diagnostic* only if this path becomes important: hash-match the math kernel and separate clear, kernel and enqueue-gap timing, then examine available traffic/cache counters. Cache warming is a hypothesis, not an established reason to keep redundant initialization forever. Do not restore the general predicate speculatively. |
| F06/F07 | These stopped at applicability/actual-winner gates; neither is a measured rejection of the unimplemented mechanism. F06 needs no adapter for the traced production search. F07 needs a real winning map with multiple M/N tiles before any cache-order experiment. |
| F08/F09 | These tested meaningful existing winners and lost. F08 regressed about **16.3%** despite reported VGPR128 to120; F09 shrank code without changing the reported resources and still lost. Do not optimize register or instruction counts as proxies for latency. Require evidence that the changed work is on the critical path. |

Two important qualifications to further performance claims:

1. **A family called XDL can already execute native gfx1250 WMMA.** The real BF16 forward winner uses an XDL-v3 device/grid, an **Interwave-v1 block pipeline**, and `v_wmma_f32_16x16x32_bf16`. A native-WMMA-v3 registration is not automatically a more native arithmetic path. The old “v3 versus v1” experiments are not a comparison of all possible gfx1250 WMMA schedules.
2. **Reconcile resource provenance before occupancy claims.** The saved F08 trace reports VGPR128, while its matching-symbol disassembly uses registers through at least `v243`; F09 likewise uses register indices above its reported64. The trace-count units/runtime-code-object association have not been resolved. This is not proof the measured timings are wrong. It means those fields cannot yet be treated as per-lane allocated registers. The same symbol can exist in differently compiled binaries. Record hashes and decode the target's resource metadata before claiming an occupancy threshold was crossed.

The current evidence supports “specific implementations were not useful,” not “the GPU lacks headroom,” nor a numerical claim about percentage of peak. No clock-matched sustained dense-BF16 capability, dynamic WMMA issue rate or full-model workload weighting was established by these reports.

## 8. Next scope: gfx1250 WMMA in the actual convolution winners

### 8.1 Boundary and starting evidence

Continue with legacy CK, 2D NHWGC and 3D NDHWGC, MIOpen-facing convolution factories only. BF16 real workloads lead; FP16 is a precision/coverage extension after the same mechanism is demonstrated. Preserve FP32 accumulation, existing strict FP32 semantics, output conversion, split policy and deterministic restrictions. Do not substitute BF16 accumulators, TF32, FP8, sparsity, generic GEMM kernels or a `ck_tile` migration. A shared primitive change requires other-architecture regression checks before it is made general.

**Primary witness S1:** forward BF16 G1/N42/C=K128, 3x3/s1/p1, input30x40, from `model_e_miopen_convolution_commands_nhwc.json` at the monorepo root. Its GEMM is M=50,400, N=128, R=1,152. The recovered public tuple is `device_grouped_conv_fwd_xdl_bf16_comp_instances`, row with block256/M128/N128/K64, AK1=BK1=8, public XDL32x32/repeats2x2, Interwave/v1 (`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_comp_instance.hpp:84-102`, specifically row98). The device's wave32 remap makes the emitted geometry 16x16/repeats2x4; do not confuse the public and emitted tuples.

Source chain: `device_grouped_conv_fwd_multiple_abd_xdl_cshuffle_v3.hpp:412-427,594-625` -> `gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp:880-905` -> `blockwise_gemm_pipeline_xdlops_selector.hpp:139-162` -> `blockwise_gemm_pipeline_xdlops_v1.hpp:383-785` -> `warp/xdlops_gemm.hpp` -> `utility/amd_wmma.hpp:517-567`. Device/grid/block paths are under `include/ck/tensor_operation/gpu/`; utility paths are under `include/ck/`. The registrar lives in required-layout `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/nhwgc/xdl/comp/device_grouped_conv2d_fwd_xdl_nhwgc_gkyxc_nhwgk_bf16_comp_instance.cpp`.

Recovered evidence: `/tmp/ck-gfx1250-followon-f08-trace/winner_isa.txt:639-803`, `winner_kernel_trace.csv`, and `incremental_kernel_trace.csv` in the same directory. The disassembly header names the extracted baseline object; its symbol matches the dispatch symbol and the geometry gives **394 CTAs, eight wave32 waves per CTA**. This is symbol/path/geometry evidence, not a hash-attested association between runtime bytes and extracted bytes. Temporary paths can disappear or be rebuilt.

**Executed static analysis of that saved code:** the burst at lines730-752 contains 16 native BF16 WMMA instructions and eight distinct eight-float accumulator destinations. Twelve of the 15 adjacent WMMA pairs name the same A register range; none names the same B range. Lines683-709 have 24 `ds_load_b128` operand loads. Partial DScnt waits and `s_setprio 1/0` already exist. The inspected file has wide global/LDS transfers and no operand `v_perm`/`ds_bpermute`/`ds_permute`, LDS-transpose or global-async instructions. These are static observations, not dynamic stall or bank-conflict measurements.

| Witness | Purpose and current gate |
|---|---|
| S1 above | First schedule/reuse/async subject: actual real-workload winner, 18 K64 blocks. Nominal dense work 14.864 GFLOP; one-read/write logical input+weight+output volume 26,099,712 bytes. These are geometry, not executed padded work or measured DRAM bytes. |
| S2 forward BF16 G1/N42/C48/K192, 3x3/s1/p1, input120x160 | Real transfer/tail/hold-out regime: M806,400/N192/R432, nominal 133.772 GFLOP and logical volume387,237,888 bytes. Its exact winner is **not established here**; enumerate and identify it before selecting a mechanism. The logical volumes imply about569 and345 FLOP/byte for S1/S2 respectively, not a measured roofline. |
| S3 backward-data BF16 G1/N42/C24/K96, 3x3/s2/p1, input240x320 | Real multi-residue control. The existing best XDL family is reported near0.220 ms, but its exact type/ISA has not been recovered here. Do not use a slower native row as its replacement baseline. |
| S4 backward-data BF16 G=N=1/C16/K1024, 1x1, input8x8, split4 | Saved F09 identity: native block64/M32/N32/K128, extra A/B padding1, repeats2/1, Intrawave/v1; eight CTAs plus mandatory clear. Useful small-grid/launch control, **not** the unpadded K32/64 bank-layout witness below. |

For WRW/3D extension, first identify an actual winning typed row and its block pipeline. No new WRW/3D WMMA winner or speedup was established in this review. Keep long-R depthwise scalar redesign and MIOpen ranking as separate scopes.

### 8.2 Gate G0 — distinguish issue, operand supply and launch limits

Before performance implementation, bind each selected type/split to compiler/options, binary and runtime code-object hashes, exact symbol, decoded VGPR/LDS/scratch allocation and dispatch geometry. Keep the F01 complete-call result; add a separately labelled kernel/API timeline and, where supported, queued/captured repeated complete calls as a throughput diagnostic. Every queued split-atomic call still clears, every packed call still repacks, and no producer/consumer may race shared scratch. Do not silently replace the isolated-call metric or treat graph replay as ordinary MIOpen latency.

Inventory counters actually supported by the installed gfx1250 toolchain before requesting them. Desired observations: active/eligible waves, matrix issue/idle intervals, load/LDS waits, LDS service/conflicts, cache/DRAM traffic and effective clocks. Record names, units, tool version and missing counters; never fabricate architecture-generic counter names. If counters are unavailable, use matched one-factor ablations plus timelines and explicitly leave the causal explanation unconfirmed. Static opcode counts and API occupancy estimates alone are not achieved utilization.

Keep comparisons against both the unchanged winning row and all eligible competing families, with requested/effective split fixed or explicitly enumerated. Use alternating independent processes and spreads; retained borderline changes need held-out confirmation. No re-run is needed merely to confirm the already reported failed trials. These diagnostics are prerequisites for **new** interventions, not grounds to relabel old outcomes.

### 8.3 Prioritized experiment queue

All latency benefits below are **[INFERENCE; unmeasured]**. A closed feasibility gate is a completed investigation, not an instruction to register a slower substitute.

| ID | Priority | Mechanism | Entry/first stop gate |
|---|---|---|---|
| WM1 | P1 | Actual-winner interwave priority and K32 fragment staging | S1 kernel-stage scheduling/operand liveness is material; preserve its tile and arithmetic. Two independent ablations, not one combined patch. |
| WM2 | P1 feasibility, then conditional performance | gfx1250 matrix-A reuse hints | Compiler encoding now verified; hardware reuse direction/adjacency/lifetime must be clarified and proven before any convolution enablement. |
| WM3 | P2 conditional | Native thread-transfer 64-bank layout specialization | Find a winning **unpadded same-major K32/64** row. S1 and S4 do not qualify. |
| WM4 | P2 after supply/resource gate | Same-winner XOR-preserving async ping-pong | Demonstrate exposed staging cost and viable two-buffer residency; preserve the S1 physical map and im2col semantics. |

#### WM1 — interwave policy and bounded operand live ranges, not another v3 row

`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp:470-474,549-648,655-778` is the actual S1 block pipeline. It loads fragments, clusters MACs, raises priority after the first matrix instruction and restores priority after the cluster. The late `block_sync_lds()` explicitly protects data as well as affecting scheduling; it is **not an optional optimization barrier**.

- **WM1a, priority only:** compare one gfx1250 S1 variant without the `s_setprio 1/0` writes against unchanged S1. Retain compiler scheduling barriers, every LDS/cluster barrier, load placement, accumulator order and epilogue; handle main-loop and tail identically. This tests whether the inherited priority policy helps this target, not whether interwave synchronization can be removed. Expect zero change in arithmetic and declared accumulator count; inspect actual codegen for secondary scheduling changes.
- **WM1b, separately, fragment staging:** the existing `NumMacClusters` mechanism computes `KPerInnerLoop=max(KPerThread/NumMacClusters,KPack)`. S1 has KPerThread32/KPack16; clusters1 ->2 changes inner length32 ->16 and repeat1 ->2, while keeping blockK64. Operand data used per cluster becomes `(MRepeat2+NRepeat4)*inner_length` BF16 elements: **96 ->48 dwords per lane**. This is a bound on a cluster's operand demand, not a promised register saving: declared thread descriptors still contain the full product `KRepeat*KPerInnerLoop=32`, so only compiler live-range shortening can help. Priority bursts and placement of cluster synchronization also change; call this a cluster-staging experiment, not a pure occupancy measurement. Preserve the final correctness barrier.

`CK_EXPERIMENTAL_INTER_WAVE_SCHEDULING_MAC_CLUSTERS` is globally and unconditionally defined as1 in `include/ck/ck.hpp:289`. Do not globally flip it, assume `-D` overrides it, or register a portfolio. Isolate the trial to the one row/build; only design a candidate-specific production parameter if the result earns retention. WM1a and WM1b each start from the unchanged baseline, not from each other.

Check: unchanged per-output K order and WMMA count; repeated signed/padded/group/tail CPU-reference forward results; main-loop and tail transitions; no new scratch. For WM1b require emitted earlier consumption/shorter live ranges or a measured scheduling effect; reconcile physical register accounting before claiming residency. Stop if no kernel-stage improvement beyond noise, if any complete-call benefit fails against the best family, or if changed priorities hurt held-out shapes. Never add a second accumulator bank to manufacture independence: S1 already has eight chains/64 FP32 accumulator values per lane; doubling it adds storage and merge arithmetic and changes reduction order.

#### WM2 — matrix-A reuse: supported encoding, unresolved runtime contract

`include/ck/utility/amd_wmma.hpp:517-567` passes `false,false` to gfx1250 FP16/BF16 FP32-accumulating WMMA builtins. The saved S1 sequence reuses each A fragment across four consecutive N fragments. The opportunity is reduced operand-collection cost, **not fewer WMMA instructions, fewer mathematical operands, or lower accumulator precision**.

**Executed toolchain probe:** isolated `--offload-device-only --offload-arch=gfx1250 -nogpuinc -nogpulib -O2 -S` compilation with `/home/sgundabo/rocm-10.2/llvm/bin/clang++` (clang24, build `bc1e171b6a5333d498ad60fa4894549aa112db93`) produced these modifiers for both FP16 and BF16:

| Final builtin booleans | Emitted modifier |
|---|---|
| `false,false` | none |
| `true,false` | `matrix_a_reuse` |
| `false,true` | `matrix_b_reuse` |

The same installation's `llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1250 --show-encoding` accepted BF16 `matrix_a_reuse` and `matrix_b_reuse`; for the fixed probe registers the second encoding byte changed0x00 ->0x20/0x40. Generic `op_sel`/`op_sel_hi` spelling was rejected. These compile-only snippets were **not executed** and are not valid reuse-correctness tests; they establish syntax and lowering only.

The supplied ISA §7.12, lines4287-4309, calls RA/RB caching hints for reuse in the *next* instruction, but also requires the current instruction to be the same as the *previous* instruction or results are undefined. Clarify the precise opcode/operand identity, cache lifetime, first/last-instruction and interposed-instruction rules from authoritative hardware/compiler documentation before implementation. The text does not justify blanket `true,true`.

After clarification, first prove a tagged two-/multi-WMMA sequence with exactly known A/B/C contents and reuse boundaries; include changed-A, changed-opcode and loop/tail transitions according to the allowed contract. Then enable A-only hints on one S1 fragment run without changing each output's K accumulation order. Verify emitted physical operand ranges and modifiers after allocation/scheduling; equal source-level values do not guarantee legal physical reuse. Keep hints off outside the proven run. No B-reuse loop reordering in this experiment.

Stop if semantics remain unresolved, compiler scheduling cannot maintain the contract, correctness fails, additional copies/spills erase the benefit or full-call timing loses. This is the clearest unexploited **gfx1250 instruction-field** opportunity found, not a demonstrated performance gap.

#### WM3 — native 64-bank map audit with a strict affected-path gate

The XDL winner already uses gfx125-specific `get_n_lds_banks(gfx125_t{})` and layered XOR maps in `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_common.hpp:337-450,701-816`. Do not propose adding that feature to S1.

The native `ABTransferThreadTiles::GetBlockDescriptor` has a different branch in `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp:149-198`: when **not padded** and `ABMajorLayout == ABLayout`, its layer expression uses `32*4 / KPerBlock / sizeof(LDSTypeAB) / ABPackedSize`. The target's LDS has64 banks (`include/ck/utility/amd_arch.hpp:55-85`). With two-byte unpacked operands, a64-bank expression changes layers **2 ->4 at K32, 1 ->2 at K64, and 1 ->1 at K128**. A32-bank-oriented map on64-bank hardware is not inherently incorrect or slow.

First find an actual winning native forward/backward/WRW row on a real shape that reaches this branch with K32/64 and meaningful LDS cost. The known F09 row has extra A/B padding and K128: it fails both gates. An unknown high-spatial1x1 winner is a lead, not evidence. Stop rather than retile an unrelated slow row merely to activate the branch.

If the gate passes, compare a single gfx125-aware map for one operand at identical CTA geometry, cluster, vector width, arithmetic and pipeline. Reuse the existing architecture-bank query rather than creating a second constants convention. Do not simultaneously change the opposite-major kfold/mpair branch's128-byte assumptions. Prove producer/consumer descriptor composition, bijection, alignment and all-lane addresses with tagged data; count bank service using the real b128 instruction service groups, distinguishing broadcasts from conflicts. Check both LDS writes and reads: improving one can damage the other. Stop on unchanged emitted map, no bank/service benefit, worse address work/spills or no best-family latency gain. Extend to3D only after its descriptor and convolution checks.

#### WM4 — actual-winner async im2col ping-pong, preserving its map

S1 already overlaps global prefetch with current compute using VGPR staging, but still emits `buffer_load_b128 -> ds_store_b128 -> ds_load_b128` into one input LDS buffer. The new mechanism is to remove valid-path producer staging/stores and produce the **next** K64 tile into a second LDS buffer while the current tile computes. This is different from the failed single-buffer1x1 W09 trial and the removed native-v3 W03 row. First demonstrate exposed producer/operand-supply cost in S1; synchronous opcode presence alone is insufficient.

Keep S1's im2col/global descriptors, vector8 contiguous C segments, XOR map, tile, WMMA order, arithmetic scheduler and epilogue. Reuse existing `AsyncCopyToLds` and b128 wrappers; do not add another ISA wrapper. **Do not just set `DirectLoad=true`:** the existing path also selects padded layouts and different pipeline machinery, so that would not isolate the intended mechanism.

Concrete address feasibility for S1, derived from the existing gfx125 map: for logical operand row `u in [0,128)` and reduction `k in [0,64)`, define

```text
x = 4*floor(u/8) + (u mod 4)
z = 8*(floor(u/4) mod 2) + floor(k/8)
F(u,k) = 128*x + 8*(z XOR (x mod 16)) + (k mod 8)  # element offset
byte address = operand_base + 2*F(u,k)
```

A occupies16,384 bytes and B follows it. For the exact `(8,32,1)` producer cluster, origin row `u0<32` advances as `u0+32*j`, `j=0..3`. The equation gives `F(u0+32*j,k)=F(u0,k)+2048*j`, so four vector8 accesses have additive LDS byte offsets0/4096/8192/12288. **Executed host model:** all8,192 element offsets were unique and covered0..8191; all1,024 vector accesses were contiguous/aligned16B and satisfied that additivity. This checked the source-derived equation, not compiled CK descriptors or GPU addressing. Compare the real descriptor's offsets before using it in a transfer adapter; other cluster/XOR maps need not be additive.

Existing direct-load code uses an origin plus constexpr access offset, making that check necessary (`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_direct_load.hpp:224-266`). The legacy wrapper supports1/4/8/16-byte async copies and handles invalid inputs with **ordinary DS zero stores** (`include/ck/utility/amd_buffer_addressing_builtins.hpp:952-1075,1218-1246`; `include/ck/utility/dynamic_buffer.hpp:257-278`). Preserve invalid-vector zeroing and verify all elements of each source vector share validity; no out-of-bounds im2col load may be substituted for a masked vector.

**Separate global-address gate:** the default gfx125 inline-assembly wrapper compensates the instruction's shared source/destination immediate by encoding `uint32_t((src_offset-static_dst_offset)*sizeof(T))` (`amd_buffer_addressing_builtins.hpp:1047-1067`). LDS additivity alone does not prove this source address is safe. A host-derived S1 example at flattened output-spatial index64 (h1/w24), first filter/C vector, has valid input offset2944 BF16 elements and static LDS offset4096 elements: the compensated vector offset is negative. Unsigned-vector-plus-immediate arithmetic would shift the source by4GiB; signed arithmetic would not. ISA async lines5869-5871 say unsigned, whereas the general GVS table at5658-5659 says I32; this review does **not** resolve the runtime interpretation or report a demonstrated kernel fault. Require exact emitted-source-address validation. A conservative first adapter can reuse the existing wrapper with `static_dst_offset=0` and the full per-access LDS destination, avoiding that compensation; count the extra address work. Do not cast away or ignore negative-offset cases merely because the LDS map is bijective.

Input-storage model: `(128+128)*64*2 = 32,768` bytes ->**65,536 bytes** for ping-pong, before any additional state/alignment or non-overlapping epilogue storage. Removing staged input represents32 dwords/thread of logical producer data at block256, not a guaranteed VGPR saving. Neither the target's LDS capacity nor the trace's resource fields predicts achieved residency. Compare actual compiled allocation and active waves. Account for CShuffle reuse only after every producer and consumer has finished using the buffers.

Synchronization contract: finish ASYNCcnt loads **and ordinary DS zero stores** before publishing a new buffer to all consumer waves; finish all consumer LDS reads before overwriting an old buffer; drain outstanding transfers before epilogue reuse. Existing `include/ck/utility/synchronization.hpp:16-73` contains distinct direct/async synchronization helpers; selecting one by name is not proof that every dependency is covered. Start with ordinary workgroup barriers, not cluster/TDM/split-barrier infrastructure. ISA §10.8, lines5849-5911, permits out-of-order LDS access; LOADcnt/DScnt are not interchangeable with ASYNCcnt.

Check: real compiled-descriptor/tagged-fragment equality; CPU-reference signed, poisoned, grouped, padded/tail cases; one/two/odd/even/many block-loop counts; unchanged per-output reduction order; support rejection for non-vector-safe inputs. Require emitted async b128, fewer valid-path producer DS stores and genuine next-buffer overlap without scratch. Stop if preserving map/schedule is impossible, zero-fill/reuse races remain, occupancy or address overhead erases savings, or the best-family complete-call comparison fails. Initial BF162D, then FP16/3D only with equivalent transfer proof.

### 8.4 Explicitly deferred work

- **LDS-transpose W10:** not performance-rejected, but the recovered forward winner has no software operand permutes to remove and already reads b128. Two transpose loads do not inherently move fewer bytes/registers than two ordinary b128 loads. Reopen only for a net producer+consumer bank/layout advantage, including the cost of making an operand transpose-friendly. Preserve the exact native fragment map. LDS-transpose is wave32 and ignores EXEC (ISA §11.2.4, lines6627-6644): every inactive lane must still address valid, initialized storage. The old tagged standalone test is not convolution proof.
- **Hardware `SCHED_MODE[2]` issue control:** distinct from `s_setprio` and compiler barriers. ISA §5.7.2.1, lines2499-2507, says disabling the multicycle-XDL arbiter stall may benefit a single resident wave/SIMD but can block other-wave coexecution. No such case is established for S1. Do not force lower occupancy to manufacture it. Only after a real single-wave issue bottleneck, consider a narrowly scoped save/set/restore experiment with all hazards preserved.
- **Blanket hazard/NOP changes:** the FP16/BF16 five-NOP rule in ISA §7.12.1 concerns prior D becoming next A/B/index, not every D-to-C accumulator recurrence. Do not add or remove waits based on that misreading. Preserve compiler/hardware RAW/WAR/WAW rules at operand and epilogue boundaries.
- **Cluster multicast, TDM, FP8/sparse/low-precision accumulation:** not justified by the measured winners or existing MIOpen numerical contracts. No generic ISA survey or wrapper-building project.
- **Further tile, direct-store, clear-elision, carry-check or lookup sweeps:** the completed trials remain closed unless a new measured mechanism changes their entry gate. WM1–WM4 are not aliases for restarting them.

### 8.5 Execution, ownership and acceptance

1. **One evidence owner** establishes G0 and stable S1–S4 identities. Reuse the filtered profiler recipe in `profiler/README.md:22-65`: `MIOPEN_REQ_LIBS_ONLY=OFF`, `CK_PROFILER_MIOPEN_LAYOUTS_ONLY=ON`. Use `--list-instances` on the current build, then select the exact type/split with `--raw-invocation`; never carry numeric IDs across builds. Reuse shapes from the corpus, not its historical timings.
2. WM1 and WM4 share the S1 pipeline: separate experimental variants and serialize that mutation boundary. WM2 initially owns only primitive semantics/codegen, WM3 only an affected native descriptor. Independent read-only analysis can proceed together; GPU measurements/builds remain isolated from competing work. Do not combine retained ablations until each has an independent matched result.
3. Each experiment ends **keep / reject / prerequisite missing**, with code-object evidence, correctness, complete-call median/spread and a best-family comparison. Keep tests only for a retained observable contract; remove rejected kernel registrations/scaffolding. New retained arithmetic/transfer code needs existing 2D/3D and other-architecture checks for every shared caller actually affected, not just one passing microcase.
4. CK performance retention does not establish MIOpen benefit. Prove matching plugin enumeration, selected execution, workspace/lifetime and numerical/deterministic admission; measure actual Find/default choice and external layout/alpha-beta costs separately. F02/F05's selected-path correctness is already reported and need not be repeated just to validate this review.

**Evidence produced for this update:** read the completed reports and scoped source chains; extracted existing F08/F09 instruction/geometry evidence; parsed86 BF16 convolution commands; calculated S1/S2 work/bytes and reported ratios; counted the saved16-WMMA/24-LDS-load burst; checked the8,192-element XOR model; compiled isolated FP16/BF16 hint snippets and assembled BF16 reuse modifiers. No CK/MIOpen build, GPU convolution, counter collection or prior reported test/benchmark was rerun. No measured WMMA speedup or hardware-peak percentage is claimed.

## 9. Targeted instances for tracelense_workload_conv_only.txt

### 9.1 Decision and layout boundary

**Start with two independent, concrete slices:**

1. **CK instance experiment:** line2, BF16 backward-data for C128/K10, 1x1, N42/input120x160. Only four split-1 configurations are supported in the probed binary, all with scalar A/dY access. Try vector2 A access on one existing tile, not another tile portfolio.
2. **MIOpen integration experiment:** line16's BF16 G1/N42/C16/K256/H=W=1 pointwise forward, with lines17–18 and the reverse/larger channel pairs as controls. Existing CK instances already support these shapes. Remove physically unnecessary NCHW staging using the existing transpose-skippability convention, keeping the selected CK kernel unchanged.

The second is deliberately **not advertised as a WMMA kernel improvement**. For the commands supplied, it is a more concrete first opportunity than adding a nominally native row. No latency gain has been measured in this investigation.

The input is the monorepo-root `tracelense_workload_conv_only.txt`: **86 commands, 29 base shapes, all BF16/N42/2D**, with29 forward,28 backward-data and29 WRW commands. It supplies no timings, invocation frequencies or default-selected solver IDs; the order is not a performance ranking. Line58–59's G3/11x11 shape has no backward-data command; none is invented here.

**The supplied commands are NCHW, not NHWC.** In `projects/miopen/driver/conv_driver.hpp:668-705`, omitted layout flags resolve to NCHW for all three 2D tensors. The grouped CK solvers admit packed default layouts and internally stage into NHWGC/GKYXC/NHWGK. The filtered CK profiler below checks that internal operation, not the original command's end-to-end path. Adding `--in_layout NHWC --fil_layout NHWC --out_layout NHWC` creates a useful diagnostic workload but changes the user's layout contract; it must not silently replace these commands.

### 9.2 Complete workload inventory

`C/K` below are **per group**, unlike the driver's total `-c/-k`. All spatial dimensions are input dimensions. Unless shown otherwise,1x1 has stride1/pad0 and3x3 has pad1; dilation is1. Line numbers are stable keys for this input snapshot, not CK instance IDs.

| Shape | Input lines | G | Per-group C/K | Input HxW | Filter / stride | Triage |
|---|---|---:|---|---|---|---|
| T01 | 1–3 | 1 | 128/10 | 120x160 | 1x1/s1 | Primary CK A-vector2 experiment in BWD; FWD/WRW controls |
| T02 | 4–6 | 1 | 128/128 | 120x160 | 3x3/s1 | Large dense control; no new winner claim |
| T03 | 7–9 | 1 | 128/128 | 30x40 | 1x1/s1 | Aligned pointwise control |
| T04 | 10–12 | 1 | 128/128 | 30x40 | 3x3/s1 | Prior WM1 S1 forward; preserve recorded decision |
| T05 | 13–15 | 1 | 128/512 | 30x40 | 1x1/s1 | Wide-output control |
| T06 | 16–18 | 1 | 16/256 | 1x1 | 1x1/s1 | Primary no-copy integration witness |
| T07 | 19–21 | 192 | 1/1 | 120x160 | 3x3/s2 | Real depthwise, long-R WRW; not short-R F05 |
| T08 | 22–24 | 1 | 192/48 | 120x160 | 1x1/s1 | Existing internal candidates observed |
| T09 | 25–27 | 1 | 192/64 | 60x80 | 1x1/s1 | Existing internal candidates observed |
| T10 | 28–30 | 1 | 24/128 | 240x320 | 1x1/s1 | Large streaming control; C24 is already vector-compatible |
| T11 | 31–33 | 1 | 24/24 | 240x320 | 3x3/s1 | Narrow-channel dense control; no old tile sweep |
| T12 | 34–36 | 1 | 24/96 | 240x320 | 3x3/s2 | Prior F03/F04 BWD control; preserve rejected trials |
| T13 | 37–39 | 1 | 256/16 | 1x1 | 1x1/s1 | Reverse-channel no-copy control |
| T14 | 40–42 | 1 | 256/128 | 30x40 | 1x1/s1 | Aligned pointwise control |
| T15 | 43–45 | 256 | 1/1 | 60x80 | 3x3/s1 | Real depthwise, long-R WRW |
| T16 | 46–48 | 256 | 1/1 | 60x80 | 3x3/s2 | Real depthwise, long-R WRW |
| T17 | 49–51 | 1 | 256/64 | 60x80 | 1x1/s1 | Aligned pointwise control |
| T18 | 52–54 | 1 | 32/512 | 1x1 | 1x1/s1 | Larger-channel no-copy control |
| T19 | 55–57 | 1 | 3/24 | 480x640 | 3x3/s2 | Odd-channel, large-spatial negative/extension regime |
| T20 | 58–59 | 3 | 1/1 | 480x640 | 11x11/s1/p5 | Low-group large-filter depthwise; FWD/WRW only |
| T21 | 60–62 | 1 | 48/128 | 120x160 | 1x1/s1 | Existing-portfolio selection control |
| T22 | 63–65 | 1 | 48/192 | 120x160 | 1x1/s1 | Existing-portfolio selection control |
| T23 | 66–68 | 1 | 48/192 | 120x160 | 3x3/s1 | Prior WM1 S2 forward; faster other family already recorded |
| T24 | 69–71 | 1 | 512/32 | 1x1 | 1x1/s1 | Reverse larger-channel no-copy control |
| T25 | 72–74 | 1 | 512/128 | 30x40 | 1x1/s1 | Aligned pointwise control |
| T26 | 75–77 | 512 | 1/1 | 30x40 | 3x3/s1 | Real depthwise, long-R WRW |
| T27 | 78–80 | 1 | 64/128 | 60x80 | 1x1/s1 | Aligned pointwise control |
| T28 | 81–83 | 1 | 64/256 | 60x80 | 1x1/s1 | Aligned pointwise control |
| T29 | 84–86 | 1 | 96/48 | 120x160 | 1x1/s1 | Non-power-of-two channel control |

This is full input accounting, not a claim that every direction of every row was executed or that every proposed control needs a new kernel.

### 9.3 Executed internal-layout support census

Used `/tmp/ck-gfx1250-rocm102-profiler/bin/ckProfiler`, SHA-256 `66b2203d7596f327a2660039fb229d2635fa4acf802e35f2ac10fbb70339c10b`, the existing ROCm10.2/gfx1250 filtered binary. **30 list-only probes covered24 distinct command lines**, with verification, initialization and timing disabled. They allocate shape-sized device buffers and query factory/support predicates, but do not invoke convolution kernels. The largest logical input+weight+output allocation in the entire input inventory is980,588,544 bytes. No build, new convolution benchmark, CPU-reference run or MIOpen driver invocation was performed.

| Base shape | FWD supported configurations | BWD at split1 | WRW at split1 | Additional split enumeration |
|---|---:|---:|---:|---|
| T01, lines1–3 | 119 | 4 | 35 | BWD split0 lists8 configurations across split1/2/4; WRW `all` lists275 |
| T06, lines16–18 | 277 | 68 | 36 | WRW `all`:212 |
| T08, lines22–24 | 341 | 68 | 83 | Not additionally probed |
| T09, lines25–27 | 341 | 68 | 85 | Not additionally probed |
| T10, lines28–30 | 277 | 68 | 71 | Not additionally probed |
| T13, lines37–39 | 353 | 68 | 36 | WRW `all`:212 |
| T18, lines52–54 | 279 | 68 | 36 | WRW `all`:212 |
| T24, lines69–71 | 361 | 68 | 36 | WRW `all`:212 |

All probes returned success. **Supported is not correct, fastest, selected by MIOpen, or suitable for deterministic execution.** Counts include separate convolution specializations/split configurations, not necessarily unique code bodies. Some type strings omit template fields and can collide; preserve the source tuple, binary, list ID and eventually loaded symbol/HSACO rather than naming a row by a short string alone.

The exact translated commands, complete candidate lists, rejection output, statuses and86-command inventory are saved at `/tmp/ck-tracelense-scope-3BBdNz/support.json`. This temporary evidence may disappear. Re-enumerate after any rebuild; numeric IDs below describe only this SHA and request.

Reproducible **list-only internal-layout** examples, run from CK's directory with `CKP` pointing to that binary:

```sh
# Input line2: BF16 BWD, internally NHWGK/GKYXC/NHWGC, split1
"$CKP" grouped_conv_bwd_data 2 1 0 0 0 0 \
  2 1 42 10 128 1 1 120 160 1 1 1 1 0 0 0 0 1 --list-instances
# Input line16: BF16 forward; extra 0 after layout selects 32-bit indexing
"$CKP" grouped_conv_fwd 2 1 0 0 0 0 0 \
  2 1 42 256 16 1 1 1 1 1 1 1 1 0 0 0 0 --list-instances
# Input line18: homogeneous BF16 WRW uses dtype5/layout2, not dtype2
"$CKP" grouped_conv_bwd_weight 5 2 0 0 0 0 \
  2 1 42 256 16 1 1 1 1 1 1 1 1 0 0 0 0 1 --list-instances
```

These do not modify the user driver commands. A later timing run must deliberately set the desired initialization/verification mode and select the current exact candidate before adding `--raw-invocation`; do not time the uninitialized list-only recipe.

### 9.4 TC1 — C128/K10 backward-data: one missing A-vector2 instance

**Exact request:** input line2, `-n42 -c128 -H120 -W160 -k10 -y1 -x1 -g1 -F2`, BF16, stride1/dilation1/pad0. GEMM is **M806,400/N128/R10**: N is the dX channel width C128, while the reduction is convolution output K10. Native K32 computes a padded reduction with31.25% useful positions; changing load width does not eliminate that padding.

The four observed split1 rows are two geometries, each with Default and `Filter1x1Stride1Pad0`:

| Fixed-split1 list ID | Existing device / geometry | Actual source access widths A/B/E |
|---|---|---|
| 0/1 | `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3`, block64/M64/N64/K32, AK1/BK1=8, WMMA16x16, repeats4/2 | 1/1/1 |
| 2/3 | Same class, block128/M128/N128/K32, AK1/BK1=8, WMMA16x16, repeats8/2 | **1/4/8** |

The specialized parent ID3 string is `DeviceGroupedConvBwdDataMultipleD_Wmma_CShuffleV3<128, 128, 128, 32, 8, 8, Filter1x1Stride1Pad0, 16, 16, 8, 2, 1, 4, 1, 1>`. Its last two1s are **CShuffle repeat counts, not E access width**; `GetTypeString` omits E width. Source authority is `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_instances.hpp:107,129-144`, especially the large-tiles row143. Its registrar is `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/nhwgc/wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_bf16_large_tiles_instance.cpp`.

**Why this gap is real:** K10 rejects A widths4/8/16, but allows2. B4 and E8 divide C128. The other supported A width is1; there is no eligible vector2 alternative in this binary. XDL-v3 vector2 rows visible in source are `LargeTensors` variants hard-vetoed on gfx1250 (`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v3.hpp:994-1000`), not competitors to unlock by removing a safety guard. There is no BF16 old non-v3 WMMA branch in this BWD factory. Current scalar support exists; this is an **access-efficiency gap, not missing convolution functionality**.

**Minimal experiment:** instantiate the row143 parent with only `ABlockTransferSrcScalarPerVector:1 ->2`, keeping A vector dimension2, A/B clusters `S<4,32,1>`, A/B LDS destination width8, B source4, E store8, block128/M128/N128/K32, FP32 accumulation, final BF16 conversion, padded descriptors, existing scheduler and split1. Start with the2D NHWGC `Filter1x1Stride1Pad0` specialization. Do not automatically extend its generic tuple to3D, all specializations or other architectures. Keep the scalar fallback for odd K and preserve existing support predicates.

The mathematical access conditions are straightforward: each dY row has10 BF16 elements (20 bytes), so five4-byte pairs fit without crossing a logical row; padding pairs beyond channel9 must remain masked/zero. The native class checks K modulo A vector width and C modulo B/E widths. This proposal does **not** change AK1/BK1 to2, invent a K16 BF16 WMMA instruction, add N16/N32 tiles, change lookup/scheduling, or revisit unsafe atomics. It is separate from WM1/WM2/WM4.

**Entry measurement:** establish the best complete-call candidate among the existing legal rows before declaring ID3 the winner. Its geometry has6,300 CTAs and one K32 block versus25,200 CTAs for the64x64 geometry, but counts alone do not establish speed. The specialized split1 native path already has W01's proven clear elision; Default can still pay a206,438,400-byte dX clear. Separate load-width benefit from specialization/initialization cost. Existing split2/4 configurations remain competitors where legally allowed, not a reason to split this R10 experiment; deterministic execution must retain split1.

The ideal single-read/write logical volume is222,568,960 bytes:206,438,400 dX,16,128,000 dY and2,560 weights, for2,064,384,000 nominal FLOPs (~9.28 FLOP/byte). A/dY is only about7.25% of that volume. Vector2 can reduce issued loads/packing overhead without halving total bytes; output traffic or the driver's layout conversions may dominate. These are analytical volumes, not measured cache/DRAM traffic.

**Acceptance if implemented:** unchanged eligible problem and reduction order; signed/cancelling CPU-reference checks with contributions **only in reduction channels8/9**; changed inputs and poisoned dX on repeated calls; grouped per-group K10 control; K11 rejected by vector2 while scalar remains; K8/K16 and partial-spatial controls. No vector crosses a logical K row. Inspect emitted load width, VGPR/LDS/scratch and full-call distributions against both unchanged parent and best eligible row. Require an actual instruction/transaction or latency difference; stop if the compiler already combines scalar loads, if layout/output traffic absorbs the saving, or if only a weaker comparator loses. Register one bounded row only after correctness and best-family utility, not a speculative portfolio.

For the exact default-NCHW driver path, follow with matched-plugin selected execution and conversion-inclusive latency; no MIOpen ranking change is implied by CK-only retention.

### 9.5 TC2 — tiny pointwise NCHW: keep the instance, remove no-op staging

**Primary request:** line16, BF16 N42/C16/K256, G1/H=W=Y=X=1, stride/dilation1/pad0, forward. Controls are lines17–18 and T13/T18/T24's other directions. The pointwise forward GEMMs are `(M,N,R)=(42,256,16)`, `(42,16,256)`, `(42,512,32)`, `(42,32,512)`. BWD swaps output-channel width and reduction roles. WRW is `(M,N,R)=(K,C,42)`, not depthwise reduction.

These cases already have hundreds of eligible forward configurations,68 BWD split1 configurations and36 WRW split1 configurations. Examples already available:

- For T06 WRW line18, observed split1 IDs10/11 are `DeviceGroupedConvBwdWeight_Wmma_CShuffleV3` block64/M32/N32/K32, A/B/store2, and the retained `_Split1` block32/M16/N16/K32, source1/1/store1. Both use FP32 accumulation and direct final BF16 Set. Definitions are in `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_weight/device_grouped_conv_bwd_weight_v3_wmma_instance.hpp`.
- The same request also lists XDL-ported and explicit-GEMM alternatives, including ID34 `DeviceGroupedConvBwdWeight_Explicit_Xdl<DeviceBatchedGemmMultipleD_Wmma_CShuffleV3<MNKPadding, CRR>>`, block64/64x64x64, source vectors4x4. It is a support example, not an asserted fastest row.
- T01 WRW likewise has35 split1/275 all-split configurations, including explicit-GEMM odd-M paths. It does not lack a generic pointwise implementation merely because some native vector4/8 rows reject K10.

Do not implement “avoid two-stage WRW” as though direct Set support were absent. First identify the best existing selected kernel, then **hold that kernel fixed** for TC2.

**Source-grounded gap:** MIOpen's grouped NCHW invoker unconditionally constructs and runs two source conversions, destination-initialization conversion and destination conversion-back (`projects/miopen/src/ck_impl/implicitgemm_ck_util.hpp:1004-1068,1128-1129`). The underlying transpose invoker executes a kernel even when the transpose is skippable (`src/hip/batched_transpose_sol.cpp:409-435`). `BatchedTransposeSolution::IsSkippable()` already defines `height==1 || width==1`, and `src/solver/conv/conv_hipconv.cpp:113-217` already applies it per operand, borrowing caller pointers and sharing the layout plan with workspace sizing. The grouped CK path lacks that optimization. Reuse that convention; do not weaken global tensor-descriptor equality or introduce a new solver/API.

For these G1/unit-spatial problems, physical addresses are exactly `x[n,c]=n*C+c`, `y[n,k]=n*K+k`, `w[k,c]=k*C+c` in both layouts. Singleton-dimension strides may differ while all reachable offsets agree. **Executed host arithmetic:**109,504 input/output/weight offset comparisons across the four channel pairs matched. This is not a compiled MIOpen alias-path correctness test.

| Channel pair C/K | Logical x / w / y bytes | Current layout-only workspace, each region rounded256 bytes |
|---|---|---:|
| 16/256 | 1,344 / 8,192 / 21,504 | 31,232 |
| 256/16 | 21,504 / 8,192 / 1,344 | 31,232 |
| 32/512 | 2,688 / 32,768 / 43,008 | 78,592 |
| 512/32 | 43,008 / 32,768 / 2,688 | 78,592 |

These are derived layout-staging bytes, **not total solution workspace or measured savings**. Native split/packing/WRW scratch and problem-level maximum requirements must remain. The first experiment avoids four conversion invocations for qualifying, demonstrably disjoint buffers while **retaining the existing conservative layout-workspace reservation** for staged fallback. Do not promise removal of the three regions yet: a descriptor-only workspace query cannot know whether runtime source/destination pointers overlap. Layout-workspace reduction requires a separately established existing non-alias API guarantee or other sound problem-level guarantee; no such guarantee was established here.

**Minimal integration:** build a per-operand staged-or-borrowed plan using the existing `IsSkippable` convention under packed-default-layout constraints. Keep problem/selected workspace sufficient for both branches; use runtime borrowing only when live sources, destination and scratch are demonstrably disjoint, otherwise retain the existing staged invocation. Start with G1, BF162D, unit input/output spatial and1x1 filter, stride/dilation1/pad0, existing default alpha1/beta0 behavior. Borrow current invocation pointers, never ownership; preserve direction tags (`x,w ->y`, `dy,w ->dx`, `x,dy ->dw`) and backward descriptor swaps. Do not cache data by pointer or shape. Do not add hidden allocations or new overlap-rejection semantics to compensate for an undersized workspace query. Only after proving a sufficient pre-existing alias contract may the shared query/solution plan omit layout regions.

Preserve every selected kernel's native initialization, scratch, alignment and stream lifetime. Removing the output-initialization *copy* is safe here because the borrowed destination already holds the same prior contents, not because beta0 alone proves every kernel overwrites every output. Keep forward's existing deterministic exclusion and BWD/WRW deterministic split1 checks. Do not generalize this slice to bilinear/scaled/fused semantics or arbitrary strided tensors without their own proof. Copy elision does not imply workspace elision; native CK scratch remains a separate requirement.

**Acceptance if implemented:** exact original NCHW commands, same selected CK type/split before/after, reference results, changed contents and changed allocations on repeated calls, poisoned destinations, workspace size/ownership checks and overlap cases retaining the existing staged behavior. Observe conversions absent only when both physical-layout and runtime borrowing conditions hold, and retained on a nondegenerate H/W control; prove no borrowed pointer is freed or treated as a workspace subbuffer. Compare ordinary complete MIOpen invocations, separately from internal CK timing, then report default/Find selection. A source-derived four-launch reduction is not itself a latency claim.

**Important negative control:** T01's HxW=19,200 is not a whole-operation layout no-op. Its weights alone have1x1 spatial extent, but activations differ physically between NCHW and NHWC. Do not replace the T01 transpose path with reinterpret-casts. Per-operand weight borrowing is a possible later extension, not a reason to claim the roughly222.6MB activation-plus-weight staging disappears.

### 9.6 What remains outside these two first experiments

- **T10 C24 pointwise:** C24 is divisible by8, not an intrinsic vector-access failure. Its BWD `(3225600,24,128)` already admits native M64/N32/K32 and M32/N32/K128, plus XDL-ported N32 candidates. Current internal BWD support count is68. Its logical operand footprint is980,588,544 bytes; establish real memory/layout/selected-kernel costs before more narrow tiles.
- **Real depthwise WRW:** T07/T15/T16/T26 have R201,600/201,600/50,400/50,400; T20 has R12,902,400 and only3 groups. None fits the retained F05 short-R condition. Do not restore the rejected serial group-local reduction. Any R-partitioned partial-reduction algorithm would be a separate workspace/reproducibility project with its own winning comparator.
- **T04/T12/T23:** retain existing WM1 and residue/clear experiment outcomes. Their occurrence in this file does not justify repeating those trials or changing WM2's unresolved reuse contract.
- **Other inventory rows:** not labelled unsupported or unimportant. This pass identifies a couple of concrete mechanisms rather than making86 new performance claims. Use workload frequency and end-to-end timings to prioritize further investigations; none were supplied here.

**Handoff and retention:** TC1 owns CK's one2D BF16 BWD instance and support/codegen checks. TC2 owns MIOpen's existing grouped transpose/workspace orchestration, reusing its established per-operand convention. They can be investigated independently; builds and performance runs must not compete for the GPU. Preserve the supplied driver layouts, exact current candidate identity and all prior numerical/safety guards. Keep only changes that pass correctness and improve their intended complete invocation against real eligible competitors. This update changed documentation only; support enumeration and host arithmetic are the new evidence, not a measured speedup.

## 10. High-value depthwise targets from reported default-driver latencies

**Historical scope, superseded by section11.** Keep these source/mathematical findings as references, but do not start the broad DW-B/DW-F implementation tracks: the newer benchmark shows hipconv handling those cases. Only the focused residual WRW plan in section11 is active.

### 10.1 Decision: optimize the slow algorithms and their consumers

**The earlier changes produced real but small absolute gains on the wrong latency tier for this next phase.** Keep TC1/TC2; do not spend the next effort extracting another fraction of a microsecond from their selected instances. All nine newly supplied slow rows have `total C = total K = G`, hence **one input and one output channel per group**. The right question is now whether a direct depthwise algorithm can replace padded dense-matrix work or a general naïve fallback on the **original NCHW driver command**.

Do not make WMMA use an acceptance condition. Fixed-filter FP32 vector arithmetic, contiguous spatial accesses, explicit reuse and bounded reductions can be more appropriate here. Native gfx1250 wave32 ownership, register/LDS use and vectorized loads/stores still matter. No low-precision accumulation, TF32, sparsity, separable-filter assumption, or unrelated backend migration is proposed.

The reported numbers below are accepted as the user's baseline observations, in milliseconds. They were **not rerun**. Nine values sum to **19.662 ms if each executes once**; no invocation frequencies were supplied, so this is not a model-iteration total or a frequency-weighted ranking. The two11x11 rows account for8.502 ms (43.24% of that unweighted sum). WRW accounts for9.209 ms, BWD5.204 ms and FWD5.249 ms. These are opportunity weights, not predicted savings.

| Key | Original workload line | Direction / reported selected solver | G | Input / output spatial | Filter, stride, pad | Reported ms |
|---|---:|---|---:|---|---|---:|
| H1 | 59 | WRW,156 `ConvHipImplicitGemmGroupWrwXdlops` | 3 | 480x640 / 480x640 | 11x11,s1,p5 | 4.271 |
| H2 | 58 | FWD,137 `ConvHipImplicitGemmGroupFwdXdlops` | 3 | 480x640 / 480x640 | 11x11,s1,p5 | 4.231 |
| H3 | 20 | BWD,86 `ConvDirectNaiveConvBwd` | 192 | 120x160 / 60x80 | 3x3,s2,p1 | 2.864 |
| H4 | 21 | WRW,156 grouped CK | 192 | 120x160 / 60x80 | 3x3,s2,p1 | 1.836 |
| H5 | 45 | WRW,156 grouped CK | 256 | 60x80 / 60x80 | 3x3,s1,p1 | 1.810 |
| H6 | 44 | BWD,86 naïve | 256 | 60x80 / 60x80 | 3x3,s1,p1 | 1.334 |
| H7 | 77 | WRW,156 grouped CK | 512 | 30x40 / 30x40 | 3x3,s1,p1 | 1.292 |
| H8 | 43 | FWD,85 `ConvDirectNaiveConvFwd` | 256 | 60x80 / 60x80 | 3x3,s1,p1 | 1.018 |
| H9 | 47 | BWD,86 naïve | 256 | 60x80 / 30x40 | 3x3,s2,p1 | 1.006 |

Line keys refer to `tracelense_workload_conv_only.txt`. Every row is BF16,N42,dilation1,default NCHW, with driver `-c/-k` both equal to G. Do not accidentally turn H1/H2 into dense C3/K3/G1 problems. H1 has **363 weights**, not1,089.

### 10.2 Priority and first deliverables

Run three independent direction tracks, sharing only the measurement/integration requirements. If sequencing strictly, start the straightforward H3 backward gather while developing the11x11 kernels; those are better-value subjects than more small pointwise rows.

| Track | Targets and covered reported time | First bounded deliverable | Why different from the previous trials |
|---|---|---|---|
| **DW-B: direct backward-data** | H3/H6/H9,5.204 ms; H3 first | Native-NCHW3x3 stride1/2, wave32-safe output gather, visible through the existing depthwise solver | Replaces general naïve indexing/precision and fixes a genuinely missing dedicated path; no descriptor-window or lookup tweak |
| **DW-W: partitioned WRW** | H1/H4/H5/H7,9.209 ms; H1 is the largest row,3x3 is the simpler first implementation control | Bounded spatial-reduction chunks -> disjoint FP32 partials -> fixed-order final reduction/BF16 store | Removes the unbounded per-lane R loop that defeated F05, and avoids dense off-diagonal work and whole-tensor layout conversions |
| **DW-F: direct forward** | H2/H8,5.249 ms; H2 first for absolute value | Spatial-CTA11x11 halo kernel;3x3 direct reuse/streaming control | Changes the algorithm/grid and dedicated solver coverage, not a dense WMMA tile or priority flag |

For each track the value criterion is **measured absolute reduction of the corresponding original default-driver row**, with correctness and supported-neighbor coverage. No speedup multiplier is predicted. A forced-kernel win can justify further work, but does not finish a track whose normal solver still chooses the old path. Avoid a combinatorial tuning portfolio: begin with the mappings below and change one mechanism only when its measured bottleneck warrants it.

### 10.3 What the source says is missing

**Dedicated forward already exists but cannot serve these two shapes.** MIOpen's `ConvDepthwiseFwd2D` (current solver185) and `ck_depthwise_fwd_impl.cpp` have native-NCHW BF16/wave32 support. Their private CK-style factory specializes whole square images7/14/28/56/112, filters3/5/7 and batch tiles32/8/2. Exact-image and batch-divisibility checks exclude N42/60x80, and there is no11x11 member. This is not a missing general gfx1250 whitelist. Source anchors: `projects/miopen/src/ck_impl/ck_depthwise_fwd_impl.cpp:386-733,810-821` and `src/include/miopen/conv/device_grouped_conv_fwd.hpp:733-855`.

**Dedicated backward is blocked more fundamentally.** `ConvDepthwiseBwdData2D` (current solver186) requires wave64 (`projects/miopen/src/solver/conv/conv_ck_grouped_conv_fwd.cpp:265-291`). Its plugin hardcodes block64, NBatch8/32, square7/14/56 and stride1 forward-kernel reuse (`src/ck_impl/ck_depthwise_bwd_data_impl.cpp:39-177`). N42, rectangular sizes and stride2 each need real support. The old kernel uses hardware lane IDs and wave-local assumptions; exposing its64-thread tuples to two wave32s by deleting a guard would be unsafe. Adding a safe wave32 candidate and changing host eligibility are one integration change, not independent patches.

**The naïve baseline is already a direct gather, not an atomic scatter.** MIOpen's BF16 naïve forward/backward instantiations use `ushort,double,ushort` (`projects/miopen/src/kernels/gpu_reference_kernel/naive_conv.cpp:2234-2256`). They have general runtime loops/indexing and FP64 accumulation. Thus the proposed gain is specialization, standard BF16-convolution FP32 accumulation, spatial ownership/reuse and reachability—not removal of imaginary atomics or upscaling buffers. Keep the high-precision reference unchanged. Optimized FP32 results must pass existing BF16 tolerances; bitwise identity with the FP64 reference is not promised.

**General grouped CK is not categorically unsupported for C/K1.** Scalar BF16 matrix entries exist. Their small matrix dimensions, padding and default-NCHW staging remain different from a native direct depthwise path. A reported naïve selection does not prove every grouped candidate rejects the problem. Likewise, the selected solver156 name does not identify its exact CK instance: capture that identity during implementation, not by guessing from the solver label.

**Existing direct implementations are starting points, not finished answers.** The MIOpen-private CK-style direct forward kernel and CK Tile's direct depthwise pipeline both use LDS. The latter can loop over rectangular spatial tiles, but its grid is group/batch-based and scans tiles inside a CTA; for G3/N42 that is only126 CTAs at NBatch1. Merely adding an11x11 tuple there does not implement spatial-grid parallelism. The generated CK Tile factory is also not the factory MIOpen consumes. Use their dtype, direct-indexing and reuse conventions, but keep one implementation owner per new kernel rather than duplicating production algorithms in three stacks.

### 10.4 DW-B — specialized native-NCHW backward gather

For original, unrotated weight `W[g,r,t]`, compute

```text
dx[n,g,h,w] = sum float(dy[n,g,ho,wo]) * float(W[g,r,t])
ho = (h + 1 - r) / stride
wo = (w + 1 - t) / stride
include only divisible numerators and in-bounds dy coordinates; r,t in [0,3)
```

Each output has one owner and is written once after FP32 accumulation. No channel mixing, matrix packing, atomics, explicit zero-upsample or whole-tensor transpose. The formula uses the weights as stored; do not also apply the old forward-reuse `FlipFilter=true` convention.

For stride1, unroll nine taps with boundary checks. For stride2, specialize parity: even h permits only r1; odd h permits r0/r2, with the equivalent column rule. Interior output parity classes have1/2/2/4 products, not nine. At the last odd row/column the upper dy coordinate can be out of range and must be skipped. This is not the same boundary rule as stride1.

**Initial mapping:**128 threads/four wave32s own one `(n,g)` and a4x64 dX tile. Wave q owns row `4*hTile+q`; lane l owns adjacent columns `64*wTile+2*l` and `+1`. Two FP32 accumulators per lane, followed by one safe BF16 pair store where both outputs are valid; use predicated scalar tails when needed. Adjacent lanes access adjacent spatial data. For stride2 the column parity is fixed for the pair and row parity is wave-uniform; share dy values between the pair. Use `threadIdx.x`-based ownership, not a64-lane assumption. No LDS is needed in this first mapping. The tile is a starting point, not an optimum; larger spatial work per CTA is a bounded follow-up only if launch/CTA overhead is observed.

**Executed coordinate proof:** compared the residue-gather contribution set to an independent forward-edge scatter enumeration on783 small cases/25,758 dX cells, including odd sizes, stride4 holes and11x11 controls. Every set matched. Exact3x3 target histograms per `(n,g)` plane are:

| Target | dX cells with1 /2 /4 /6 /9 contributions | Total products per plane |
|---|---|---:|
| H3,120x160/s2 | 4,941 /9,598 /4,661 /0 /0 | 42,781 |
| H6,60x80/s1 | 0 /0 /4 /272 /4,524 | 42,364 |
| H9,60x80/s2 | 1,271 /2,398 /1,131 /0 /0 | 10,591 |

These three shapes have no holes, but complete Set ownership must still write zero for an empty sum if later eligibility permits holes. The proof validates integer coordinate sets, **not FP32/BF16 GPU numerics or latency**.

**Source/integration owner:** extend the existing MIOpen-consumed direct depthwise path with a dedicated gather primitive/sibling, not direction branches spread through the forward kernel. Use `src/ck_impl/ck_depthwise_bwd_data_impl.cpp` for typed candidate enumeration, native NCHW args, support and invoker; reuse `ConvDepthwiseBwdData2D`'s existing solver registry and plugin callback family. A CK grouped-convolution depthwise primitive can be instantiated directly by this plugin; no Tile dispatcher or new public operation is needed. Preserve old CDNA forward-reuse tuples and stale-ID rejection. Lift the solver's wave64-only gate only together with candidate-specific safe gfx1250/wave32 selection; keep legacy64 tuples unavailable there.

Admission must check BF16,2D,packed default layout, `G=totalC=totalK`,3x3,p1,dilation1,stride1/2, separate dy/dX spatial dimensions and checked offsets. N42 must not be rounded down to a batch tile. Preserve existing alpha/beta handling rather than silently ignoring invocation scalars. Nine weights per group are not naturally aligned as five BF16 pairs: scalar/guarded weight reads must never consume a tenth coefficient.

**Acceptance:** exact H3/H6/H9 with reference verification; asymmetric weight patterns to catch double flips; all parity classes and final rows/columns; changed input and poisoned dX; cancellation/large finite values; tail and existing-architecture controls. Trace a complete Set gather with no hidden layout conversions. Full, unforced Direct Find must enumerate/select it where it wins, replacing the reported86 fallback—not merely pass a forced186 test. Stop if benefit exists only in a different layout, requires unsafe old wave ownership, fails BF16 accuracy or cannot improve the original driver operation.

### 10.5 DW-W — bounded partial reductions, not the failed long-R scalar kernel

Let `R = N*Ho*Wo`, `F = filterH*filterW`. Each logical weight is

```text
dw[g,f] = sum_r dy[n,g,ho,wo] * x[n,g,ho*stride+fy-pad,wo*stride+fx-pad]
r = (n*Ho + ho)*Wo + wo; f = fy*filterW + fx
```

Compute `hi=ho*stride+fy-pad` and `wi=wo*stride+fx-pad` in signed arithmetic. Include a term only when `0<=hi<Hi` and `0<=wi<Wi`; test before forming/loading its x address. Padding contributes zero by skipping that product, not by loading out of bounds and masking afterward. Preserve this rule for special values too: multiplying a fabricated zero by NaN/Inf is not the same as skipping an invalid tap. Empty partial sums still store zero to their owned P slots.

**Stage1:** one CTA owns `(split,g,filter_tile)` and `r in [split*Q,min(R,(split+1)*Q))`. Use256 threads with lanes traversing consecutive output-spatial positions in NCHW; each thread handles r values at increments256 and maintains a small set of FP32 filter accumulators. Reuse one dy value across the filter tile and stream the corresponding x values rather than materializing a full patch; WRW computes weights and does not read them as an input operand. Reduce with a fixed wave/block tree, then uniquely write FP32 `P[split,g,f]`. Write zero partials too; every logical P slot must be defined without a preliminary clear. No off-diagonal group products and no FP16/BF16/FP32 atomic accumulation.

**Stage2:** fixed-order reduction across the private split dimension, one final BF16 conversion per weight. Keep f contiguous in `P[S,G,F]`; a small weight tile across lanes lets each split's reads coalesce. Do not implement one enormously strided scalar load stream per weight by default. The reduction tree/order and configuration must be fixed for repeated-run reproducibility; this does not promise equality with a serial FP32 sum, the previous atomic algorithm, or across different compilers/configurations.

Start with **Q2048/filter_tile9 for3x3**, and **Q8192/filter_tile11 (one filter row) for11x11**. Initial calculated resources:

| Target | R | Private splits S | Filter tile | Stage1 CTAs | Logical FP32 partial bytes |
|---|---:|---:|---:|---:|---:|
| H1,11x11/G3 | 12,902,400 | 1,575 | 11 | 51,975 | 2,286,900 |
| H4,3x3/G192/s2 | 201,600 | 99 | 9 | 19,008 | 684,288 |
| H5,3x3/G256/s1 | 201,600 | 99 | 9 | 25,344 | 912,384 |
| H7,3x3/G512/s1 | 50,400 | 25 | 9 | 12,800 | 460,800 |

Formulas: `S=ceil(R/Q)`, `CTAs=S*G*ceil(F/filter_tile)`, `partial_bytes=4*S*G*F`. Alignment, block-reduction LDS and any required compatibility workspace are additional. H1's partial allocation rounds to2,287,104 bytes at256-byte alignment. The public solver workspace must cover every path it admits, not blindly return this analytical P size.

The bounded alternative Q4096 halves3x3 split pressure approximately: S50/50/13 and partial345,600/460,800/239,616 bytes. For11x11 Q16384 gives S788 and1,144,176 bytes; filter_tile22 reduces CTA count but doubles filter-accumulator state relative to11. Treat these as **at most one measured chunk/filter-reuse follow-up**, not an autotuning cross-product. Smaller Q increases CTA/partial overhead; larger Q increases serial work and can reduce parallelism. Filter fusion saves dy rereads but can increase registers/LDS reduction cost.

The distinction from failed F05 is substantial. That NHWGC kernel assigns group8/reduction32 and loops over the entire R, giving1,575–6,300 serial terms per reduction lane on the real3x3 cases, and even more on11x11. Here each lane initially handles at most8 terms for3x3 or32 for11x11, with NCHW-coalesced spatial lanes and many independent partial CTAs. Do not extend F05's short-R admission or merely increase its group tile.

**Executed integer model:** six signed-data cases, including3x3/s1/s2,11x11, batch-crossing chunks, partial chunks and filter_tile11/22, wrote all5,810 partial slots exactly once and matched independent serial gradients. This proves ownership and contribution accounting in that model only. GPU FP32 accumulation/reduction accuracy, LDS reuse barriers and numerical repeatability still need tests.

**CK and MIOpen integration:** implement one native-NCHW depthwise WRW device-op family, reusing CK's existing block-reduction and device/invoker conventions. Starting points are `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_bf16.hpp` for the existing direct implementation and `include/ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp` for fixed reductions. Do not retain its NHWGC group-lane addressing in the new NCHW path.

Route native IDs through the **existing `GrpConvWrw` plugin callbacks**, using a second, separately typed native-layout factory/argument/invoker branch in `projects/miopen/src/ck_impl/ck_grouped_conv_wrw_impl.cpp`. The current `DeviceOpGWrw` fixes NHWGC/GKYXC/NHWGK; native candidates cannot be appended to that vector or receive its synthesized channel-last strides. Reuse dimension extraction/checking, but build args from actual packed NCHW descriptors. Existing candidate IDs continue to use the current transformed route. Ensure `fill_valid_kernels`, applicability, per-ID support, workspace and solution agree on the native ID. Include only the new consumed native sources in the MIOpen-only CK build; do not enable all excluded layout libraries.

A native internal layout does not by itself require a new public operation, thirteenth plugin family or ABI bump: the current five WRW callbacks can express the same mathematical operation. Preserve their established semantics, error behavior and old-family support; change versioning only if an exported contract actually changes. A broad rewrite of `implicitgemm_ck_util.hpp` or a new generic dispatcher is not required.

**Private partition policy:** encode Q/filter-tile identity in the native candidate ID/configuration. Do not feed S1575 into the existing public split-K1..128 search or remove its caps. The native implementation is a whole two-stage algorithm with private S; expose it through its explicit legal public configuration (initially suffix `+1`) and reject unsupported public split requests. Deterministic admission requires demonstrated fixed-tree repeatability and all existing semantic checks; suffix1 alone is not proof.

The outer problem-level workspace may remain conservative for still-enumerated transformed fallbacks. The native invocation must actually use NCHW pointers and avoid transforms, not wrap the new kernel inside the old NHWGC invoker. Do not promise reduced selected-solution workspace while preserving an alias-dependent fallback that needs more memory: either establish the relevant API non-alias contract or reserve enough for every allowed runtime path. No hidden allocation or new overlap rejection can substitute for a correct query. This is the same workspace discipline learned from TC2.

**Acceptance:** exact H1/H4/H5/H7, repeated changed inputs, poisoned P/dw, all batch/group/filter edges, Q remainder and cross-image chunks, FP32 partials and one final BF16 conversion, cancellation/high-dynamic-range numerical controls and repeatability when advertised. Trace exactly the required two compute/reduction stages plus any genuinely required epilogue; no layout transforms, whole-P clears or atomic accumulation on the fast path. Compare the whole operation against the actually selected156 CK row and other eligible families, then establish ordinary Find/default selection. Existing explicit-GEMM WRW is1x1/s1/p0-only and is **not** a legal3x3/11x11 comparator. Stop if partial overhead, redundant x/dy reads, register spills or accuracy defeats default-driver benefit; do not return to unpartitioned F05.

### 10.6 DW-F — spatially tiled11x11, with a small3x3 direct control

**H2 first:** one CTA owns `(n,g,hTile,wTile)`, cooperatively loads an input halo from the original NCHW plane, computes a bounded output tile with FP32 accumulators and directly stores BF16. Preserve arbitrary11x11 weights: no separable approximation. The grid must include spatial tiles; a group/batch grid that scans an entire480x640 image inside each CTA is not the intended algorithm.

For four adjacent output columns per thread, stream one filter row's14 input values and reuse them across four accumulators. Do not hold the entire11x11 patch or121 converted weights per thread. Optionally place121 FP32 weights in484 bytes of shared storage, loaded alongside the halo, and compare against uniform cached weight access only if needed. All threads must reach the cooperative-load barrier, including out-of-image/tail lanes.

| Output tile | Threads at4 outputs/thread | Raw BF16 halo | Halo bytes +121 FP32 weights | H2 spatial CTAs |
|---|---:|---|---:|---:|
| 16x16 | 64 | 26x26 | 1,836 | 151,200 |
| 8x32 | 64 | 18x42 | 1,996 | 151,200 |
| 16x32 | 128 | 26x42 | 2,668 | 75,600 |

Start16x16 as the bounded correctness baseline;8x32 tests wider coalesced rows,16x32 tests halo/CTA amortization. This is a three-point **algorithmic mapping** comparison, not a broad WMMA tile sweep. Exact480x640 divides all three output tiles. Count the true LDS pitch, alignment and any zero-fill padding before claiming the table's allocation. A FP32-promoted halo doubles its input portion but removes repeated BF16 conversion; consider it only with codegen/bank evidence, not simultaneously with the initial mapping.

Ideal halo elements per output are2.640625/2.953125/2.1328125, versus121 independent input-tap references in an unreused scalar expression. This is **not a40–57x traffic or speedup claim**: the baseline's cache and register reuse, inter-CTA duplication, transaction width and compute throughput matter. H2's distinct x+y+weight storage is154,829,526 bytes and nominal arithmetic9.367 GFLOP, not a measured roofline. Halo starts at `w0-5`, often unaligned for wide global loads; use safe coalesced/scalar loads or a proven alignment peel, not vector-pointer reinterpretation across boundaries.

Use correlation filter order for forward. Out-of-image halo entries must be initialized, but `0*NaN/Inf` is not automatically equivalent to skipping an invalid tap; retain the existing reference's boundary/special-value semantics where required, with predicated accumulation if necessary. Every valid output has one owner and one final conversion/store.

**Executed finite-integer model:**16x16,8x32 and16x32 halo tiles on three small/ragged image sizes, nine cases/5,643 outputs, matched an independent untiled11x11 correlation and wrote each output once. This is not a GPU LDS, BF16 rounding or special-value proof.

**H8 control/companion:** first assess one bounded rectangular60x80,NBatch2 reuse candidate in the existing wave32 direct factory—N42 is divisible by2. Existing image/batch checks must remain valid. That reuses an LDS kernel, not a no-LDS implementation, and may lose from large per-lane work. The distinct no-LDS alternative is fixed3x3, four adjacent W outputs/thread, streaming six input values per source row (18 values for four outputs rather than36 independent references), four FP32 accumulators, one final output store. With128 threads and width80,1,200 four-output vectors per plane need10 CTAs; H8 has107,520 CTAs. Guard final lanes and row boundaries. Its mechanisms are FP32 instead of reference FP64, fixed indexing and register reuse, not a claim that the existing G256 grid lacks all parallelism. Stop after the better of these bounded direct starting points is established; do not expand whole-image specializations indiscriminately.

**Integration:** reuse `ConvDepthwiseFwd2D`/solver185 and `src/ck_impl/ck_depthwise_fwd_impl.cpp`'s native-NCHW callback/invoker path. The currently MIOpen-consumed CK-style direct implementation is under `projects/miopen/src/include/miopen/conv/device_grouped_conv_fwd.hpp`; changing only a standalone CK Tile registrar will not reach it. A spatial-CTA device/kernel variant is necessary for11x11, but a new public solver frontend is not. Keep kernel ownership explicit and avoid a duplicate public/Tile/private implementation campaign. Support/IDs must explicitly admit2D BF16, depthwise equality, filter/stride/dilation/padding and exact output formula, rectangular spatial tiles and N42 without dropped batches. Preserve old candidates and other architectures.

**Acceptance:** exact H2/H8 reference verification, all halo corners/edges, last group/batch, changed inputs, poisoned output, any admitted ragged tile sizes, signed index and vector alignment checks, allocated registers/LDS/scratch and no hidden layout work. Then full Direct Find/default-driver selection and repeatable complete-operation improvement versus reported137/85 paths. A forced185 or stand-alone CK Tile win remains an intermediate result. No new high-value claim without the original driver benefiting.

### 10.7 Shared integration, comparison and release gates

1. **Retain the original requests and reported baselines.** No NHWC flag substitution, synthetic smaller batch, filter approximation or reduced precision. First implementation runs should preserve the nine keys and collect the exact selected solver, kernel ID and loaded code object. The supplied timings are accepted; this investigation does not re-run them for confirmation.
2. **Keep kernel ownership and runtime scope honest.** Some necessary code is in MIOpen's CK plugin/private depthwise implementation, not only the guarded CK archive. New public CK-native rows must have an actual MIOpen consumer. Runtime gfx1250 checks are required where build-wide `CK_USE_GFX1250` would otherwise expose a row in mixed-target builds. Wave32 safety is an implementation requirement, not a CMake spelling.
3. **Measure absolute driver benefit.** Record full convolution-operation latency, all required launches/workspace stages, matching verification settings and observed clocks/state. Keep kernel-only/ATT evidence separate; the prior measurement-state work showed instrumentation and execution modes can change timing. Compare paired independent runs after implementation, not unrelated historical medians. Report delta-ms per original row first; report model totals only when real invocation counts are available.
4. **Selection is part of delivery.** First prove candidate correctness with a pinned ID, then fresh **full** Find in an isolated test database and normal cached/default operation. Do not delete user databases, use a restricted two-solver search as default-selection proof, or force a global ranking merely to make the new kernel appear. Reject stale IDs safely. Update the existing direction solver fixtures and architecture masks only for implemented support; missing required plugins in the integration suite are failures, not passing skips.
5. **Preserve numerical, alias and workspace contracts.** Standard BF16 inputs/output with FP32 optimized accumulation, high-precision reference verification, no intermediate BF16 reductions, checked size/offset products, current pointers on every call, no cached mutable tensors. Explicitly resolve alpha/beta and supported alias behavior in the existing host path. Public queries must cover all runtime branches; keep conservative compatibility reservations until a smaller contract is actually proven. No missing-scratch workaround, silently ignored scale, or weakened guard.
6. **Retain only useful implementations.** No new ISA wrapper, dispatcher, broad tuple sweep or unmeasured global split policy. Complete the default-driver proof for each supported target. Failed algorithm trials are documented and removed; partial correctness or a compiled registrar is not completion. Keep TC1/TC2 improvements, but do not present their small selected-path savings as progress on these millisecond hotspots.

### 10.8 Evidence and source handoff

The nine source/solver investigations and host models above are complete **scope evidence**, not kernel implementations or GPU measurements. New execution consisted of arithmetic inventory, bounded gather contribution enumeration, integer WRW partition/ownership checks and integer halo-tiling checks. No GPU benchmark, build, prior correctness test or reported timing was rerun.

| Concern | Primary source anchors |
|---|---|
| Dedicated FWD/BWD solvers and admission | `projects/miopen/src/solver/conv/conv_ck_grouped_conv_fwd.cpp` |
| Actual native depthwise plugin factories/invokers | `projects/miopen/src/ck_impl/ck_depthwise_fwd_impl.cpp`; `projects/miopen/src/ck_impl/ck_depthwise_bwd_data_impl.cpp` |
| Existing MIOpen-consumed CK-style direct math/ownership | `projects/miopen/src/include/miopen/conv/device_grouped_conv_fwd.hpp` |
| CK Tile direct depthwise conventions, but different consumer/grid | `include/ck_tile/ops/grouped_convolution/pipeline/grouped_convolution_forward_depthwise_pipeline.hpp`; `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp` |
| Existing short-R direct WRW and fixed reductions | `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_bf16.hpp`; `include/ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp` |
| Current typed WRW factory, argument construction and transposes | `projects/miopen/src/ck_impl/ck_grouped_conv_wrw_impl.cpp`; `projects/miopen/src/ck_impl/ck_grouped_conv_impl_helpers.hpp`; `projects/miopen/src/ck_impl/implicitgemm_ck_util.hpp` |
| Workspace/selection integration | `projects/miopen/src/include/miopen/solver/implicitgemm_ck_util_common.hpp`; `projects/miopen/src/solver/conv/conv_hip_implicit_gemm_grouped_wrw_xdlops.cpp`; `projects/miopen/src/mlo_dir_conv.cpp` |
| Preserve naïve reference / understand actual baseline | `projects/miopen/src/solver/conv/conv_direct_naive_conv.cpp`; `projects/miopen/src/kernels/gpu_reference_kernel/naive_conv.cpp` |

`include/` paths in this table are relative to CK; `projects/miopen/` paths are monorepo-relative. Read the current containing symbols rather than trusting old line numbers after implementation work.

## 11. Focused residual 11x11 WRW alongside hipconv

### 11.1 Updated evidence and objective

**Decision: one CK WRW family, complementary to hipconv.** The latest [benchmark artifact](../../miopen/miopen_benchmarker/tracelense_workload_gfx1250_miopen_shapes_ck_improvements.json) supersedes section10's earlier multi-direction priorities. Its112 records include86 convolution commands. The exact target is the slowest selected convolution row:

**Integration decision: retain the existing channels-last path first.** The CK reduction will consume NHWGC/GKYXC/NHWGK through the current WRW device interface and MIOpen transposes. Native NCHW was an optional additional optimization, not a prerequisite; its separate typed factory, argument and invoker branch are deferred. The original driver command stays NCHW. This first experiment isolates the reduction algorithm instead of changing both algorithm and layout integration.

```text
MIOpenDriver convbfp16 -n 42 -c 3 -H 480 -W 640 -k 3 -y 11 -x 11 -p 5 -q 5 -u 1 -v 1 -l 1 -j 1 -m conv -g 3 -F 4 -t 1
```

The artifact records solver156 `ConvHipImplicitGemmGroupWrwXdlops`, **4.764284 ms**, count1, and successful GPU-reference verification. Its forward counterpart now selects hipconv/solver220 at **0.380066 ms**. G256/60x80/3x3 stride1 FWD/BWD select hipconv at0.138272/0.139811 ms; the earlier naïve-solver figures must not remain their current baselines. These are reported outcomes accepted without rerunning them, not matched old/new experiments performed here.

This is the **dominant residual** in the earlier artifact, not literally the only depthwise WRW left to CK: the artifact also selects grouped CK for the other WRW rows. Its next-largest selected convolution row was G192/3x3/stride2 WRW at 1.135313 ms. [Section 12](#12-measured-g192-stride-two-depthwise-wrw) records the subsequent focused implementation; neither result restarts the broad FWD/BWD queue.

Using the user's **25.9 ms aggregate** as the stated reference, this one count1 row represents about18.4%. The JSON also contains non-convolution records; the following is a what-if against that stated aggregate, not a reconstruction of model wall time:

| Hypothetical new target latency | Saved time | Aggregate if everything else stays unchanged | Reduction of25.9 ms |
|---|---:|---:|---:|
| 2.382142 ms,2x target improvement | 2.382142 ms | 23.517858 ms | 9.20% |
| 1.0 ms | 3.764284 ms | 22.135716 ms | 14.53% |
| 0.5 ms | 4.264284 ms | 21.635716 ms | 16.46% |

These are **targets/scenarios, not predicted performance**. A robust smaller gain remains useful; the point is to optimize absolute milliseconds in this residual rather than revisit the earlier sub-microsecond scheduling interventions.

### 11.2 Why hipconv is genuinely inapplicable here

The gap is established from current source, not inferred merely from the selected solver:

- `projects/miopen/src/hipconv/src/arch/cdna5/depthwise/depthwise_1d_toeplitz/kernel.h:1120-1143` accepts Fprop/Dgrad only, explicitly excluding Wgrad. Its odd-filter set includes11, explaining the working11x11 forward case without implying WRW coverage.
- `projects/miopen/src/hipconv/src/arch/cdna5/grouped/grouped_multi_g_wgrad/kernel.hpp:482-519` requires3x3 and channels per group4/8/16/32, with additional layout/stride/dtype constraints. This request has11x11 and channels per group1; it fails both independent gates.
- The direct CDNA5 family also rejects Wgrad (`projects/miopen/src/hipconv/src/arch/cdna5/direct/kernel.hpp:842-860`). MIOpen's hipconv wrapper can convert NCHW to NHWC, so the external NCHW label alone is not the cause of rejection.

Do not remove these predicates: the registered kernels do not implement this operation. Do not change hipconv, disable solver220, or force CK globally. The new candidate's eligibility should depend on its own mathematical/layout contract, not on calling a competitor's applicability function. Future hipconv support should compete normally through Find.

### 11.3 Exact incumbent recovered from the tuning record

The accompanying `projects/miopen/miopen_benchmarker/tracelense_workload_gfx1250_miopen_shapes_ck_improvements_tuning.log:617199-617236` records GenericSearch's selected CK configuration and the exact NCHW BF16 WRW performance-db key. The inserted configuration is:

```text
DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32, 16, 16, 32, Default, 8, 1, 1, 1, 4, 1, 4, 1, 1, 1, BlkGemmPipelineScheduler: Intrawave, BlkGemmPipelineVersion: v1, 1>+-1
```

This is a block32/M16/N16/K32, two-stage WMMA instance with **NumGroupsToMerge1**. The `+-1` suffix requests **auto split**; it does not report effective `k_batch`. Do not describe the incumbent as split1 or split128 without capturing its resolved argument. The log also shows the NCHW conversion invocations. Tuning-record identity is not a hash-attested loaded-code-object capture; collect the effective split and loaded symbol when preparing the new implementation's matched comparison, without repeating the already reported timing just for confirmation.

Compare against that recorded **auto-split incumbent**, not a forced split1 version of it. The new algorithm's public `+1` label describes its own private partition policy and is not a reason to force the old algorithm to `+1` for the performance comparison.

Per group, the logical weight GEMM is M1/N121/R12,902,400. The incumbent's output tiles cover M16/N128: **2,048 matrix positions for121 logical weights**, or16.93x in this output-tile work model, before split-dependent reduction padding. This is not a16.93x speedup forecast; WMMA and scalar FP32 have different throughput, and memory/synchronization costs remain. It does make a genuinely depthwise reduction worth testing. Existing explicit-GEMM WRW only admits1x1/s1/p0, and retained F05 only admits short-R3x3/high-G, so neither is a hidden applicable shortcut for this case.

### 11.4 One first prototype: row-aligned NHWGC partial reductions

**[INFERENCE; unmeasured]** Replace padded dense-matrix work with direct FP32 reductions over the existing transformed NHWGC x and NHWGK dY buffers. Keep the MIOpen transposes and two CK stages, but change what those stages compute: disjoint weight-gradient partials followed by a final sum/cast into GKYXC dW. The incumbent is already two-stage; merely saying “two kernels” is not the optimization.

Geometry is fixed for the first performance witness: N42,G3,Ho=Hi480,Wo=Wi640,filter11x11,pad5,stride/dilation1. There are363 output weights and12,902,400 reduction positions per group. Start with **12 output rows per image strip**, one filter row per CTA. This refines section10.5's linear-Q8192 partition; it is not another unrelated algorithm candidate.

Stage1 grid owns `(n, strip, g, fy)`, with40 strips/image and11 filter rows. Each CTA has256 threads and11 FP32 accumulators per thread, one for each fx. Map threads as follows:

```text
row_lane = threadIdx.x / 128       # 0 or1; constant bit decomposition
col_lane = threadIdx.x % 128       # 0..127
ho = strip*12 + row_lane + 2*j     # j=0..5
wo = col_lane + 128*k             # k=0..4 for width640
hi = ho + fy - 5
wi = wo + fx - 5                  # fx=0..10
```

Each thread processes **exactly30 dY positions** on the target. Neighboring lanes access neighboring logical columns, but a fixed group in NHWGC has spatial stride G: at G3 their addresses differ by3 BF16 elements/6 bytes, not the contiguous2-byte step of NCHW. Start with correct scalar BF16 reads and ordinary caching; do not claim ideal contiguous transactions or reinterpret group triplets as aligned vector2/4 data. The reduced coalescing is an explicit tradeoff for avoiding new MIOpen layout machinery. Batch/row/column division is still removed from the inner traversal by construction. Reuse each dY value across11 fx accumulators and stream x values, without a121-element patch. Shared staging or cross-lane shuffles are not prerequisites for the first proof.

Compute hi/wi in signed arithmetic and check their bounds **before any x address/load**. Invalid padded taps are skipped, including under special-value semantics; do not issue an invalid read and mask it afterward. Tail rows/columns in later family tests predicate work, not participation in the block reduction. All threads reach required reduction barriers even when their local sum is zero.

For packed channels-last tensors with per-group C=K=1, the valid-coordinate offsets in BF16 elements are:

```text
x_offset  = ((n*Hi + hi)*Wi + wi)*G + g
dy_offset = ((n*Ho + ho)*Wo + wo)*G + g
dw_offset = g*121 + fy*11 + fx
```

Use the supplied CK lengths/strides or explicitly validate these packed-layout predicates. Do not reuse NCHW address formulas on transposed buffers. Also validate ho/wo before any dY load on partial strip/column fixtures. The mathematical partial ownership and P layout below are unchanged by the input layout.

Reduce each of the11 accumulators through a fixed wave/block tree, using the existing CK reduction conventions. Write exactly one FP32 partial for each `(n,strip,g,fy,fx)`, including zero partials. No atomics and no preliminary whole-partial-buffer clear. A suitable layout is `P[s,g,f]`, with `strips_per_image=ceil(Ho/12)`, `s=n*strips_per_image+strip`, `f=fy*11+fx`. The target has40 strips/image; small/tail fixtures must use their actual strip count, not a hardcoded40 in workspace indexing.

| Quantity | Exact prototype |
|---|---:|
| Private partial count S per group/weight | 42*40 =1,680 |
| Stage1 CTAs | 42*40*3*11 =55,440 |
| FP32 partial values | 1,680*3*121 =609,840 |
| Logical partial bytes | 2,439,360 |
| Partial bytes rounded to256-byte alignment | 2,439,424 |
| Maximum per-thread reduction positions | 30, versus an unbounded full-R traversal |

Stage2 reduces the1,680 partials for each weight in a fixed FP32 order, then converts once to BF16. Keep adjacent f values coalesced when reading P; tile both weight and reduction dimensions rather than defaulting to a wholly strided serial stream. Its output is only363 weights, so measure this stage's own launch/underfill cost, but include it in every complete-operation comparison.

The logical P strides are121 floats/group and363 floats/split; neither guarantees16-byte alignment for every vector-four load. Start with correctly aligned scalar FP32 accesses coalesced across lanes, or explicitly pad and resize P before using wider per-thread loads. Mask the final363-weight tile and any partial split tile before loading; do not copy a float4 reduction helper's alignment assumptions unchanged.

The row-aligned design has6.67% more CTAs/partials than linear-Q8192 (1,575 splits,51,975 CTAs,2,286,900 bytes), trading that cost for simple traversal without image-crossing chunks. No claim is made that this trade is already faster. **At most one reuse follow-up:** if the first kernel is limited by repeated dY reads/CTA overhead rather than register pressure, try two adjacent filter rows per CTA. Eleven filter rows require six tiles, so that variant has **30,240 CTAs, not an exact halving**, unchanged logical P size and a guarded final single-row tile. It doubles filter-accumulator state; do not combine it with a new chunk policy, LDS layout and scheduler sweep.

This is not the rejected F05 long-R design. Both use channels-last data, but F05's group8/reduction32 lanes traversed all R; here R is split into bounded image strips with independent partial owners and a separate final reduction. Explicit reduction parallelism—not a layout change—is the substantive difference. It is also not a group-merging WMMA experiment, a reuse-hint experiment, a filter approximation or a new FFT/im2col workspace path.

### 11.5 CK implementation using the existing MIOpen WRW integration

Implement the partitioned reduction as one more **NHWGC/GKYXC/NHWGK** BF16 operation in the existing `DeviceGroupedConvBwdWeight` interface and factory consumed by `DeviceOpGWrwPtrs`. No separate native-NCHW factory, `CKArgs` type, callback dispatch branch or transpose bypass is part of this first experiment.

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_bf16.hpp` is the existing direct-WRW/interface pattern. Keep its retained short-R candidate unchanged; add the bounded11x11 implementation rather than relaxing its admission.
- `include/ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp` supplies fixed-reduction precedents; preserve synchronization/lifetime requirements when reducing multiple filter accumulators.
- Register the new BF16 row through the existing grouped2D WRW NHWGC instance library/factory. Implement standard `MakeArgumentPointer`, `IsSupportedArgument`, `GetWorkSpaceSize`, `SetWorkSpacePointer`, invoker and unique type-ID behavior. Initial performance target remains the exact G3/11x11 request; broaden only after validation. Runtime gfx1250 eligibility must remain safe in mixed-target builds.

**The existing workspace path already supports this.** `projects/miopen/src/ck_impl/ck_grouped_conv_wrw_impl.cpp:23-25,38-94,218-238` uses the current typed factory, standard arguments and NCHW/NHWC invokers. `GetCKSplitkMaxWorkspaceSize` in `src/ck_impl/implicitgemm_ck_util.hpp:674-703` queries supported instances; the NCHW invoker obtains selected CK scratch and adds it after layout staging (`:1039-1043`), creates its subbuffer and calls `SetWorkSpacePointer` (`:1208-1220`). Therefore the new partial buffer can use normal CK workspace reporting. Rebuild matching CK/plugin artifacts and test the actual path; no new solver, public ABI or MIOpen layout-routing code is expected.

Dry support/workspace queries use null tensor pointers and a synthetic non-null scratch pointer (`projects/miopen/src/ck_impl/ck_grouped_conv_impl_helpers.hpp:223-249`). Compute size/support from shape and policy without dereferencing those pointers. Validate real pointers, scratch bounds/alignment and complete writes at live invocation. Report the checked P allocation through `GetWorkSpaceSize`; do not hide allocations in the kernel or assume scratch was pre-cleared.

Private S1,680 is **not public split-K1,680**. Encode strip/filter-tile policy in the candidate identity, admit its public `+1` configuration initially and reject unsupported public split requests. Do not change the existing1..128/auto search. Fixed-tree repeatability must be established before deterministic admission; suffix1 alone is not proof.

Keep all existing transposes, output-initialization/alias behavior, alpha/beta handling and conservative workspace accounting. The new P allocation is additional CK scratch, not a replacement for the layout buffers. No copy-elision or workspace-reservation reduction belongs in this prototype. Performance must include those unchanged integration costs; no global solver-ranking change is part of registration.

**Isolation from hipconv:** the added row is BF16,2D,depthwise11x11,s1/d1/p5, initially focused on the exact target through the existing internal layout. It does not change FWD/BWD or hipconv applicability. Original NCHW driver inputs still traverse MIOpen's established transform path; any direct NHWC measurements are diagnostic, not substitutes for the requested result.

### 11.6 Proof and acceptance

Host ownership evidence remains valid: all307,200 positions in a480x640 plane are owned once, with30 positions per thread/strip. After this layout revision, three signed-integer models explicitly converted NCHW x/dY arrays into NHWGC storage and used the channels-last offsets above. They wrote all3,025 partial slots once and matched independent NCHW serial gradients, including G3, H13/W131 and H25/W17 strip/column/group/batch boundaries. These validate layout arithmetic and ownership, **not real GPU transposes, barriers, FP32 error, BF16 conversion or speed**.

Implementation acceptance is deliberately narrow:

1. Full target GPU-reference verification plus small independent numerical checks; all42 images,3 groups and121 coefficients; asymmetric/signed/cancelling and large-dynamic-range data; changed inputs, poisoned P/dW and repeated calls. Test strip, column and filter-row tails. Invalid taps must not be loaded; empty partials must still be written. FP32 optimized accumulation uses established BF16 accuracy tolerances, not a claim of bitwise identity with a different reduction tree.
2. Record the incumbent's **effective auto split**, new private partition geometry, exact loaded symbols and allocation/spill metadata. These diagnostics accompany implementation; unavailable stall counters must not become another indefinite prerequisite-only project.
3. Trace the two new CK stages with no whole-P clear or atomic reduction, **retaining the expected MIOpen layout conversions**; then measure uninstrumented complete NCHW MIOpen invocations with transposes, partial reduction and final reduction/cast all included. Record internal CK time separately to identify where gains or remaining costs occur. Compare against the recorded auto-split WMMA incumbent and other applicable solvers. Do not time uninitialized data or claim a kernel-only improvement as the original driver gain.
4. Demonstrate ordinary full Find and subsequent cached/default selection for the original command in an isolated test DB, preserving stale-ID safety. Solver156 may stay the visible winner while its CK instance changes. A forced new CK ID alone is intermediate evidence, not the requested aggregate improvement.
5. Keep the latest hipconv-winning FWD/BWD rows unchanged and check non-target WRW behavior. Remeasure the complete benchmark aggregate under the same methodology before publishing a set-level speedup; the what-if table is not that measurement. If no meaningful full-operation win exists, stop this one prototype rather than reopening the old broad queues.

**Deferred native-layout option:** reconsider NCHW only after the channels-last algorithm is correct and measured, and only if conversion time remains a material limit on the target. A native path could remove copies or improve spatial access, but would be a separate, justified MIOpen integration change with its own pointer/workspace/alias proof. It is not a condition for implementing or retaining this reduction.

Sections 11.1–11.6 describe the pre-implementation experiment; the measured outcome and final candidate are recorded below.

### 11.7 Measured row-strip implementation

Implementation: `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_row_strip_bf16.hpp`; existing BF16 NHWGC WRW registrar and `grouped_convolution_backward_weight.hpp` factory; focused GPU test `test/grouped_convnd_bwd_weight/test_grouped_convnd_bwd_weight_row_strip_bf16.cpp`.

**Keep the bounded CK candidate.** `DeviceGroupedConvBwdWeightDepthwiseRowStripBf16<12, 128, 11, 16, Fy2, Split1>` is registered for gfx1250 BF16 2D packed NHWGC/GKYXC/NHWGK with G3, C=K=1 per group, 11x11/s1/d1/p5 and public split1/auto only. It uses the existing MIOpen NCHW transposes and WRW workspace path; no hipconv or MIOpen routing source was changed. Stage1 pairs adjacent filter rows in one 256-thread CTA, reusing each dY load across their 22 filter accumulators; the odd eleventh row is guarded. Stage2 reads the 2,439,360-byte FP32 `P[s,g,f]` and converts once to BF16. The target allocation is 2,439,424 bytes aligned to 256; 42 images × 40 strips × 3 groups × 6 filter-row tiles produce 30,240 CTAs. This replaces the first, one-filter-row prototype's 55,440 CTAs; no atomic or whole-P clear is used.

The focused gfx1250 GPU test compares every weight and FP32 partial against an independent NCHW serial reference for G3 N2/H13/W131, N1/H25/W17 and N1/H1/W1. It passes twice with changed signed/cancelling inputs, poisoned scratch and dW, invalid/empty taps and strip/column/filter-row tails. Dry null-pointer support, both integer descriptor widths, exact packed strides, public split policy and G2 rejection pass. The original NCHW N42/480x640 driver also reports GPU-reference verification OK with all 121 filter coefficients and G3 via the existing MIOpen transform path.

On the same rebuilt CK profiler and hot-reuse raw full-`Invoker::Run` metric (50 repetitions), the incumbent two-stage WMMA auto resolves to **split341, 4.57751 ms**; row-strip resolves to **split1, 3.31870 ms**. Requested/effective split is not conflated with the candidate's private 1,680-strip partition. The prior reported default NCHW MIOpen row was **4.764284 ms**; an isolated full Find using a matching locally linked plugin selects the paired-row candidate at `+1` and reports **3.477980 ms**, a 1.286304-ms / 27.0% difference to that historical observation, *not* an interleaved paired MIOpen trial. A subsequent cached/default driver invocation selects solver156 at 3.552505 ms and verifies on GPU. Its performance DB records the exact `Fy2` candidate type and `+1` request. The first one-filter-row prototype also won a full Find at 3.978266 ms, but was rejected in favor of the paired-row result. External NCHW staging remained in all these full-call measurements.

Uninstrumented timings above are distinct from instrumentation: a final `rocprofv3` cached-driver trace has ten dispatches of each CK stage, twenty input transposes and ten each of the remaining output transposes. Stage1 averages 3.233 ms and stage2 0.055 ms *under tracing*; no P clear appears per invocation. The analogous first-prototype trace measured 3.631/0.055 ms, supporting the bounded adjacent-filter-row reuse. The focused test executable's emitted code-object metadata reports stage1 48 VGPRs, 22,528 LDS bytes, no scratch/spills, and stage2 12 VGPRs, 1,024 LDS bytes, no scratch/spills; both take 88-byte kernel arguments. The MIOpen trace identifies both loaded stage symbols in code object 37 with the same LDS and zero private bytes. These metadata are not occupancy counters. With an old, trial-only `Fy1` performance-DB ID under the new plugin, MIOpen warns that the ID is invalid and runs a correct but slow fallback (~540 ms); the new candidate has a distinct ID, so databases containing that obsolete *experimental* ID should be retrained, not silently mapped to the new algorithm.

Non-target checks retained hipconv/solver220 for G3/11x11 forward and G256/3x3 forward/backward-data; G256/3x3 WRW remained grouped CK/solver156 and verified. The G256 WRW check used an untuned isolated database; its latency is not a meaningful before/after comparison.

The complete `tracelense_workload_gfx1250_miopen_shapes.txt` benchmark was rerun with the same script's `--tuning` pass followed by its ordinary default measurement pass, an isolated MIOpen database and the matching locally linked plugin. **All 112 commands verified OK in both passes, with zero validation mismatches.** Sum of the 112 count-weighted measurement rows is **23.985120 ms** versus **25.498539 ms** in `projects/miopen/miopen_benchmarker/tracelense_workload_gfx1250_miopen_shapes_ck_improvements.json`: an observed **1.513419 ms / 5.94%** lower command-time sum. The target row measured **3.539252 ms** versus its old 4.764284 ms, accounting for **1.225032 ms** of the observed aggregate difference. The remaining 0.288387 ms is not attributed to this one candidate: three non-target commands changed solver IDs during tuning, and thermal/timing variation is not isolated by this single aggregate pass. The earlier user-stated 25.9-ms reference is not the same quantity as the old JSON's 25.498539-ms row sum; do not subtract these as if they were paired baselines. Reproduction artifacts from this run are `/tmp/ck-wrw-rowstrip-aggregate.json`, `.xlsx`, and `_tuning.log` (temporary local files, not source-tree fixtures).

## 12. Measured G192 stride-two depthwise WRW

**Target:** `MIOpenDriver convbfp16 -n 42 -c 192 -H 120 -W 160 -k 192 -y 3 -x 3 -p 1 -q 1 -u 2 -v 2 -l 1 -j 1 -m conv -g 192 -F 4 -t 1`. The preceding complete 112-command benchmark recorded solver156 at **1.137520 ms**. Its tuned CK type was `DeviceGroupedConvBwdWeightTwoStage_Wmma_CShuffleV3<32, 16, 144, 32, Default, ..., 16>+128`, not split1. A cached-driver kernel trace measured the GEMM stage at about 0.990 ms and retained NCHW transposes at about 0.124 ms per call. Its traced kernel reported 764-byte kernargs, 10,240 LDS bytes and 48 private bytes; these are metadata, not a diagnosis of the latency.

**Retained candidate:** `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_depthwise_grouped_row_strip_bf16.hpp`, type `DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16<16, 8, 9, Split1>`, registered through the existing BF16 NHWGC grouped WRW factory. It keeps MIOpen's NCHW-to-NHWGC staging. A 256-thread CTA owns eight output rows and sixteen adjacent groups, with sixteen reduction lanes streaming the at-most-80 output columns in five bounded waves. One dY value feeds nine guarded x taps and separate FP32 accumulators; a block tree writes every `P[s,g,f]`, including zeros. Stage2 coalesces sixteen neighboring weights across sixteen split lanes and performs one BF16 Set. The exact G192 target has `S=42*ceil(60/8)=336`, `336*ceil(192/16)=4,032` stage1 CTAs and **2,322,432 bytes** of checked, 256-byte-aligned FP32 partial scratch. No atomics or P clear. The first measured group8/32-lane mapping (8,064 CTAs) was superseded by the faster group16/16-lane mapping, not left as another registered row.

Admission is gfx1250, BF16, 2D packed NHWGC/GKYXC/NHWGK, C=K=1 per group, `192<=G<=256`, 3x3/s2/d1/p1, `N<=42`, `Ho<=60`, `Wo<=80`, `513<=N*Ho*Wo<=201600`, and public split1 or auto (both effective1); explicit larger splits reject. Checked input/output/weight strides, extents and workspace are required. MIOpen removes redundant right padding on even input extents: the device op accepts right pad 0 or 1 **only when the supplied output extent agrees with that padding**. That admission fix made the original NCHW command reachable without changing MIOpen source or broadening the retained short-R F05 kernel.

**Correctness and measurement:** the focused GPU test `test/grouped_convnd_bwd_weight/test_grouped_convnd_bwd_weight_grouped_row_strip_bf16.cpp` passes three G193 odd-width, partial-row, group-tail and empty-tap fixtures against an independent serial NCHW reference for every BF16 weight and FP32 partial. It also checks poisoned scratch/dW on changed-input repeats, dry null-pointer queries, both descriptor widths, splits, invalid layouts and adjusted right padding. The original N42/G192 command passes MIOpen's GPU reference in full Find and subsequent cached/default execution. On the same CK profiler hot-reuse raw full-Invoker interval, the prior merged16 fixed128 reports **1.007240 ms** and the group16 candidate **0.452101 ms** (55.1% lower). The complete NCHW MIOpen command reports **0.570090 ms** after full Find versus the earlier benchmark's **1.137520 ms** (49.9% lower across separate runs); cached/default selection reports 0.570467 ms and records the new candidate at `+1`. The complete call includes the unchanged transposes; these are not CK-only numbers. A traced final run has ten dispatches of each CK stage, with stage1 averaging 0.431 ms and stage2 0.015 ms under instrumentation; transposes remain and no P clear occurs per invocation. The focused test code object reports stage1 62 VGPRs, 9,216 LDS bytes, no scratch/spills; stage2 12 VGPRs, 1,024 LDS bytes, no scratch/spills. Neither trace timing nor resource metadata establishes cache/occupancy causality.

One adjacent admitted G256/3x3/s2/N42 command also passed GPU verification and full Find selected this row at **0.243071 ms** against its previous benchmark observation of **0.431235 ms**; stride1/FWD/BWD and G3/11x11 remain outside its admission. The previous 112-command aggregate in section11.7 has **not** been rerun with this candidate: no additional set-level speedup or model wall-time claim follows from these two target measurements. The local performance tests used a matching relinked CK plugin; installed CK/plugin artifacts require rebuilding before deployment.

These section12 admission and aggregate-status statements describe the G192 stride-two commit. The later, separately measured stride-one/G512 extension and its current admission are in section13.1.

## 13. Family-wide CK performance phase for the NCHW workload

**Objective and measurement boundary.** Improve latency across CK-winning *families*, not merely enumerate more supported instances. In the complete post-G3, pre-G192 112-command run from section11.7, 52 of the 86 BF16 convolutions selected grouped CK (15.971378 ms of the 23.985120-ms row sum): 29 WRW/11.345959 ms, 11 BWD-data/2.930754 ms and 12 FWD/1.694665 ms. The earlier baseline had 51 CK winners; three non-target solver IDs changed between those two tuning passes, so counts and timing provenance must travel together. The G192 and G256 stride-two wins in section12 were measured subsequently and have **not** been folded into that 112-row sum. Dense G1 arithmetic and depthwise singleton-C/K reduction are different algorithms; no single scalar WRW candidate can replace dense WMMA FWD/BWD/WRW.

| Representative CK winner, original NCHW driver | Complete driver in this local trace | CK arithmetic stage | NCHW staging stages | Interpretation/limit |
|---|---:|---:|---:|---|
| G1 C24/K128, 1x1 WRW, N42/240x320 | 0.524796 ms | 0.174 ms | 0.238+0.086+0.004+0.004 ms | Pointwise high-spatial WRW is staging-heavy; this is one of six G1 pointwise WRW rows with at least 10,000 spatial cells, together 1.526896 ms in the 112-row run. TC2's unit-spatial borrow proof does not cover them. |
| G1 C3/K24, 3x3/s2 BWD-data, N42/480x640 | 0.700494 ms | 0.456 ms | 0.086+0.077+0.046+0.003 ms | Compute and conversion both matter. A per-call clear remains in this multi-residue BWD trace; the 1x1 Set-coverage shortcut is not a blanket exemption. |
| G1 C24/K24, 3x3/s1 FWD, N42/240x320 | 0.486231 ms | 0.178 ms | 0.114+0.085+0.085+0.007 ms | Staging-heavy even though the selected packed CK core is WMMA-v3. |
| G256, C=K=1, 3x3/s1 WRW, N42/60x80 | 0.821916 ms | 0.708 ms | 0.034+0.034+0.005+0.005 ms | At this trace's baseline arithmetic was the first CK-only target; the then-current section12 row rejected stride1. Section13.1 records the later extension. |
| G512, C=K=1, 3x3/s1 WRW, N42/30x40 | 0.619520 ms | 0.500 ms | 0.037+0.037+0.005+0.005 ms | Second family/geometry check, not assumed equivalent performance. |

These `rocprofv3` stage figures are **instrumented single-run diagnostics**, not uninstrumented complete-call subtractions, measured HBM traffic or a reason to assume all members of a family have identical shares. The two stride-one depthwise WRW rows contribute 1.412689 ms in the preceding 112-command measurement; the selected MIOpen CK configuration is two-stage merged16 WMMA at `+128` for G256 and auto for G512. G1 pointwise WRW contributes 2.320941 ms over eighteen rows, but its four unit-spatial rows sum only 0.035827 ms; optimize high-spatial members, not the easiest tiny fixture.

**B1 — first CK-only family experiment.** Reuse section12's group16/8-row FP32 partial ownership and final reduction for packed depthwise 3x3/s1/p1 at G256/N42/60x80 and G512/N42/30x40. Keep the existing stride-two **type/ID and numerical contract** stable: make the stage-one stride a compile-time 1-or-2 choice within the existing candidate, with mathematically checked output extent and right pad and no new MIOpen ABI. This recompiles the stage-one code object; its stride-two performance must be rechecked rather than assumed identical. Bound the initial admission to the proven group, output and reduction range; exclude F05's short-R domain. CPU and GPU controls must cover both strides, asymmetric signed/cancelling values, odd row/column and group tails, empty taps, poisoned P/dW and changing inputs; preserve every partial's unique ownership and the original G192/stride2 MIOpen result. Compare raw complete CK calls against each shape's actual best eligible family and full NCHW Find/cached calls, including transposes. Keep stride1 admission only if *both* stride-one corpus rows benefit without a meaningful G192/G256 stride-two regression. A loss removes the trial, rather than adding an unused instance.

**B2 — cross-direction staging feasibility, separately owned.** High-spatial G1 FWD and pointwise WRW spend substantial traced time outside packed CK math. NCHW-to-NHWGC conversion is presently in MIOpen's grouped CK integration, not one shared CK grid function. A broader gain requires either a proven native-NCHW CK access map or a fused conversion/compute design **plus** selected-solution pointer, alias and workspace contracts in MIOpen. Do not globally skip transposes: TC2 proved borrowing only for packed, disjoint, unit-spatial pointwise tensors. First compare actual selected FWD/BWD/WRW types and full-call staging traffic on multiple G1 high-spatial shapes; derive the native operand coalescing/WMMA fragment map and capacity before changing plugin routing or ABI. Stop if saved conversion is outweighed by slower CK compute or larger scratch. This is not silently part of B1.

**B3 — CK WMMA grid/transfer only after an actual-winner gate.** Selected high-spatial FWD and BWD-data rows above use WMMA-v3, while the dense pointwise WRW winner uses a different XDL-ported path; touching `gridwise_gemm_wmma_cshuffle_v3.hpp` will not reach all three. Inspect their loaded code objects and address instruction/traffic or stall evidence at identical tile, split and compiler before changing common group/tile progression. Revalidate row/column masks, BWD residue slices, group offsets, split clears and epilogues separately. F03's reduced padded work and F08's emitted-address simplification both failed to improve their best eligible families; do not repeat those changes based on source-level arithmetic or VGPR counts alone.

**Promotion gate for this phase.** Use F01's hot-reuse raw complete CK invocation metric to isolate a mechanism and MIOpen's original NCHW full-call Find to establish utility. Record exact type, requested/effective split, loaded code-object identity, workspace, transposes/clears/casts and GPU-reference result. Hold out whole filter/group/stride families, compare to every eligible family and check cached/default selection. Remeasure all 112 commands with identical methodology before claiming an aggregate or external workload gain; optimizing a few synthetic cases alone is not a set-level result.

### 13.1 B1 measured: two stride specializations, one checked family

The first CK-only experiment is retained in the working tree. `device_grouped_conv_bwd_weight_depthwise_grouped_row_strip_bf16.hpp` now compiles its existing group16/8-row stage1 separately for stride1 and stride2, dispatching from the validated argument; the stage2 FP32 partial reduction, BF16 Set, public split policy and `DeviceGroupedConvBwdWeightDepthwiseGroupedRowStripBf16<16, 8, 9, Split1>` ID remain unchanged. Its bounded 3x3/d1/p1 admission now covers `192<=G<=512`, matched stride1/2 in both axes, `N<=42`, `Ho<=60`, `Wo<=80` and `513<=N*Ho*Wo<=201600`. Right padding is checked against the supplied extent (stride1 requires pad1 here; stride2 may accept MIOpen's redundant-pad0 on even extents). This is a reusable **depthwise WRW family**, not a claim that dense G1 or FWD/BWD should use scalar reductions. F05's short-R admission remains separate.

| Exact BF16 original NCHW WRW command | Prior selected CK raw complete call | B1 CK raw complete call | Prior benchmark full call | B1 full Find / cached default |
|---|---:|---:|---:|---:|
| G256/N42/60x80, 3x3/s1/p1 | 0.730117 ms (merged16, fixed128) | 0.470587 ms (private split1) | 0.811210 ms | 0.557212 / 0.554323 ms |
| G512/N42/30x40, 3x3/s1/p1 | 0.511984 ms (merged16, auto effective96) | 0.281850 ms (private split1) | 0.601479 ms | 0.364675 / 0.364034 ms |

Both full Finds selected the existing CK solver156 with the grouped row-strip instance and GPU-reference verification passed. Their observed full-call decreases versus the preceding benchmark rows are 31.3% and 39.4% across separate runs; they are not matched alternating MIOpen processes. Both shapes request **3,096,576 bytes** of private FP32 P, plus unchanged external layout staging. The original G192/3x3/s2 raw CK result remained 0.452101 → 0.452086 ms; its cached full NCHW call measured 0.567920 ms versus the earlier 0.570090 ms. The adjacent G256/3x3/s2 cached call measured 0.241375 ms versus 0.243071 ms. These tiny stride-two differences are not claimed as gains. The same type string permits existing tuned IDs to resolve the appropriate stride-specialized code object without silently changing their public split configuration.

The focused GPU suite passed both existing stride2 and new stride1 G257, G512 and H1 odd/tail/group cases: every private FP32 P slot and BF16 weight matched an independent NCHW serial reference within established tolerances on changed signed inputs with poisoned scratch/dW. Dry long/int arguments, null pointers, workspace sizing, invalid stride3/right padding and both valid split request forms were checked. Test-executable code-object metadata reports stage1 stride1/stride2 at 68/62 VGPRs, 9,216 LDS bytes and zero private scratch/spills; stage2 uses 12 VGPRs and 1,024 LDS bytes. These resource values and the prior instrumented stage shares do not by themselves establish why the new calls win. The full-set measurement below uses the same matching plugin and documented benchmark method.

**Complete 112-command remeasurement:** the benchmark script ran a full fresh tuning pass and a subsequent default measurement pass with an isolated MIOpen database and the matching CK plugin. All 112 commands verified OK in both passes, with zero tuning/measurement validation mismatches. The count-weighted measurement sum is **22.685773 ms**, versus **23.985120 ms** in the prior post-G3, pre-G192/stride1 full run: **1.299347 ms / 5.42% lower**. Against the earlier artifact's 25.498539-ms row sum the observed difference is 2.812766 ms / 11.03% across multiple successive changes and runs; this is not a paired causal estimate. The four directly admitted 3x3 depthwise WRW rows (G192/s2, G256/s2, G256/s1, G512/s1) account for **1.248706 ms** of the latest aggregate difference; the remaining 0.050641 ms is unassigned. Three non-target commands changed solver IDs during tuning, and clocks/cache behavior were not independently controlled. This is a sum of driver-reported per-command GPU times, **not** measured model wall time or proof that all CK-winning shapes improved. Local reproduction artifacts are `/tmp/ck-broad-family-aggregate.json`, `.xlsx`, and `_tuning.log`.

### 13.2 B2 historical native-NCHW WRW trial and guarded forward Set

**Status (2026-09-27): native WRW reverted.** Commit `aa5ea26d955` records the complete direct-NCHW pointwise WRW experiment; `e7d26a58d7a` reverts it. Current CK grouped WRW remains NHWGC/GKYXC/NHWGK and uses MIOpen's existing NCHW layout conversions. The native instance, its `+-1` selection, and the 22.086015-ms aggregate below are **historical measurements, not current-branch performance or a retained kernel**; the earlier 22.685773-ms full-set run is the last measured all-NHWGC baseline, not a measurement of the exact post-revert worktree. The independent guarded forward output-initialization change still invokes NHWGC CK and was not part of these two commits.

**Native WRW experiment (reverted):** `device_grouped_conv_bwd_weight_nchw_pointwise_bf16_wmma.hpp` and its gfx1250 BF16 NGCHW/GKCYX/NGKHW registrar add `DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma<8,32,64,64>` for packed G1 1x1/s1/p0 with `1<=N<=42`, `8<=C<=24` divisible by 8, `16<=K<=128` divisible by 16, and `1024<=H*W<=76800` divisible by 512. Public split `1`, `0`, and auto `-1` all execute the same private eight spatial partitions: stage1 writes disjoint FP32 `P[split,image,k,c]`, and stage2 reduces images and splits in a fixed order before one BF16 Set. The selected MIOpen performance-DB ID is `DeviceGroupedConvBwdWeightNchwPointwiseBf16Wmma<8,32,64,64>+-1`. The native invoker consumes the caller's physical NCHW x/dY and writes dW directly; packed strides, alpha=1/beta=0, disjoint tensors, gfx1250, and a disjoint/aligned caller workspace are checked. An output/input alias uses the original staged NHWGC WRW invoker instead, with no hidden allocation. The reported full-target workspace remains the conservative staged-fallback 980,588,544 bytes even though native FP32 partials need only 4,128,768 bytes. An older installed CK without the matching header/registrar still builds and uses staged WRW; a matching CK archive and plugin rebuild are required to get this candidate.

**Correctness and mechanism:** the focused CK GPU test checks every FP32 partial and BF16 weight against an independent serial NCHW reference on changed signed/cancelling inputs, poisoned P/dW, 8/16 minimum channel tiles, 24/112 partial tiles, both integer descriptor widths, split1/auto, invalid geometry and pointer aliases. The selected MIOpen plugin GPU test separately exercises repeated disjoint calls, x==dW and dY==dW staged fallbacks, unchanged inputs, invalid/null/undersized/overlapping workspace, and an independent CPU weight reference. The original N42/C24/K128/240x320 NCHW driver passed GPU-reference verification in fresh full Find and cached/default execution. In cached `rocprofv3` traces, the staged predecessor has ten CK dispatches at ~0.174 ms each, input conversions at ~0.238/~0.086 ms and two ~0.004-ms conversions, plus a per-call clear. Native WRW has ten ~0.329-ms partial and ten ~0.026-ms finalizer dispatches, with **no per-call NCHW conversion or P clear**. The native partial/finalizer code-object metadata reports 64/8 VGPRs, 12,288/512 LDS bytes, and 36/0 private bytes; these are not occupancy or stall counters. Native arithmetic is slower than the old CK stage in this trace; eliminating the staging makes the complete call faster.

The original pointwise command's fresh Find chose solver156 and the native auto ID at **0.354872 ms**; the same isolated DB's cached/default measurement is **0.359526 ms**, versus **0.503987 ms** in the preceding 112-command run (0.144461 ms / 28.7% lower across separate full-set runs). Four alternating old-staged/new-native original-command processes, with separate cached DBs and GPU-reference verification each time, measured staged `[0.522705, 0.525209, 0.505701, 0.536025]` ms and native `[0.362192, 0.361577, 0.362174, 0.360879]` ms: medians **0.523957 vs 0.361876 ms**, 30.9% lower for the new complete call. The final source-linked plugin again selected solver156 and verified at **0.361106 ms**. CK-only time and instrumented stage times are not substituted for these full-operation measurements.

**Forward staging:** a persistent-grid transpose experiment was slower in matched FWD and WRW calls and was restored. The retained forward change instead skips only the output-initialization transpose when a selected plain 2D NCHW BF16 WMMA-v3 or XDL-ported CK Set-epilogue provably overwrites every packed output element: default alpha/beta, supported high-spatial dimensions, matching actual descriptors, disjoint user buffers and sufficient nonaliasing workspace remain required. The input and weight conversions remain. A focused selected-plugin test covers WMMA-v3 and XDL-ported IDs, independently poisoned output and workspace, changed calls, partial N/K tiles and H/W tails, plus alias and nondefault-scalar fallback. The G1 N42/C24/K24 3x3/s1/p1/240x320 original FWD row measured **0.491121 → 0.348260 ms** across the two complete runs; it remains solver137 and passed GPU verification.

**Complete workload:** `/tmp/ck-native-wrw-complete-112.json` is a fresh `--tuning` pass followed by cached/default measurements for exactly the 112 keys in `/tmp/ck-broad-family-aggregate.json`, using the matching locally linked CK plugin and an isolated MIOpen DB. **All 112 tuned and measured commands verified OK, with zero validation mismatches.** The count-weighted sum is **22.086015 ms vs 22.685773 ms**, an observed **0.599758 ms / 2.64% lower** sum. The two directly compared G1 target rows account for **0.287322 ms** of that difference. FWD contributed 0.504140 ms and WRW 0.170445 ms, while BWD-data and batchnorm moved the other way by 0.069320 and 0.005507 ms. Twelve rows changed visible solver selections across independent tuning passes; the remaining 0.312436 ms is **not** attributed to these two code changes. The original G3/11x11 WRW stayed grouped CK/solver156 (**3.556848 → 3.537982 ms**), G3/11x11 forward stayed hipconv/solver220 (**0.378737 → 0.380988 ms**), and G192/G256/G512 3x3 WRW held out of the native pointwise admission and verified with their existing grouped CK family. These command-time sums are not end-to-end model speedups or a paired 112-command causal estimate.

**Stale-cache downgrade control:** copying the new `+-1` performance-DB entry to a separately built plugin backed by the older installed CK produces MIOpen's explicit `Invalid config loaded from Perf Db` warning. That default invocation still verifies but falls back to an **untuned 96.903755-ms** WRW call; correctness is not performance safety. A fresh `MIOPEN_FIND_ENFORCE=4` run with the old CK verifies at **0.508715 ms**, and its subsequent cached/default call verifies at **0.497615 ms**. Rebuild the matching CK/plugin for native WRW; after any downgrade, retune or replace the stale performance DB rather than accepting its automatic fallback latency.
