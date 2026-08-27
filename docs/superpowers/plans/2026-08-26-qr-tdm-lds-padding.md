# gfx1250 `qr_tdm` LDS Padding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production descriptor-coupled 256-byte LDS padding for gfx1250 BF16/FP16 `qr_tdm`, qualify Q/K/V independently on the prefill and decode paths, and retain only the measured production configurations.

**Architecture:** Keep all padding machinery private to the FMHA TDM policy. A byte-based `LdsPaddingConfig` is the single source for TDM raw encoding, padded row-major reader descriptors, actual-issue geometry proofs, and descriptor-derived arena sizing. Only the `qr_tdm` wrapper branch changes from separate shared arrays to one 256-byte-aligned arena; other FMHA pipelines remain byte-for-byte structurally unchanged.

**Tech Stack:** C++17/HIP, CK Tile tensor descriptors and tile distributions, gfx1250 TDM, GoogleTest, Python FMHA code generation, CMake/Ninja, LLVM code-object metadata tools.

**Spec:** `docs/superpowers/specs/2026-08-26-qr-tdm-lds-padding-design.md`

## Global Constraints

- Work locally for reading and editing; build, test, benchmark, and profile only on the configured remote GPU host.
- Before every remote build or run after a source change, source `.remote.env` locally and synchronize with `rsync -az --delete $RSYNC_EXCLUDES ./ "$REMOTE_HOST:$REMOTE_PATH/"`.
- Tasks 1--6 and 8--9 are fully pre-authorized for all remote/GPU actions and retries, including
  `rsync`, `ssh`, `docker exec`, configure/build, correctness execution, and metadata extraction.
  Treat those actions like local development and do not stop for permission. Task 7 is the only
  authorization boundary: request authorization when entering its qualification runs, and stop again
  after reporting the ablation table so the user can select the production configuration.
- Never print or hard-code `.remote.env` values, never commit remotely, and copy remote-only results back before another destructive sync.
- Preserve the user's existing `.gitignore` modification and any unrelated untracked files.
- Use `apply_patch` for source and document edits. Do not use destructive checkout or reset commands to restore temporary experiment edits.
- Use `superpowers:test-driven-development` before implementation, `superpowers:systematic-debugging` for any failure, and `superpowers:verification-before-completion` before a success claim or commit.
- Production scope is gfx1250, BF16/FP16, `d=128`, double-buffer prefill (`kM0 > 64`) and single-buffer decode (`kM0 <= 64`).
- `numeric_traits<DataType>::PackedSize` must equal one whenever padding is enabled.
- Padding helpers remain FMHA-local and must not include, call, or refactor GEMM policy code.
- The production arena requires every Q/K/V region base to be 256-byte aligned. The temporary legacy-phase layout is diagnostic-only and cannot become a fallback.
- The padded ATT trace is a separate milestone and is not part of this implementation plan.
- Every FMHA development build follows the fast single-instance loop from `AGENTS.md`: resolve one
  exact blob name locally, filter codegen to that blob, configure only API `fwd`, build only
  `tile_example_fmha_fwd`, and reuse the fixed `projects/composablekernel/build/` directory. After
  changing a tracked policy/codegen file, rerun only `ninja tile_example_fmha_fwd -j64`; do not clean
  or rebuild the full instance matrix.

## File Structure

- Modify `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
  - Owns byte configs, raw encoder, padded descriptors, issue/access proofs, production selection, and arena layouts.
- Modify `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp`
  - Consumes encoded configs and policy pointers; preserves producer/consumer ordering.
- Modify `projects/composablekernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`
  - Replaces the four prefill shared arrays with a single aligned arena only for `qr_tdm` and passes the arena base to both paths.
- Create `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`
  - Contains compile-time config/descriptor/layout checks and gfx125 TDM-to-reader round-trip tests.
- Modify `projects/composablekernel/test/ck_tile/fmha/CMakeLists.txt`
  - Adds the focused gfx125-only padding test target without changing existing FMHA test targets.
- Modify `projects/composablekernel/test/ck_tile/fmha/test_fmha_fwd.cpp`
  - Adds focused non-symmetric end-to-end BF16/FP16 coverage for both paths and high-risk traits.
- Temporarily modify `projects/composablekernel/example/ck_tile/01_fmha/CMakeLists.txt`
  - Adds the `FMHA_FWD_QUICK_FILTER` hook prescribed by `AGENTS.md`; remove the hunk before any production commit.
- Potentially modify `projects/composablekernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`
  - Only after measurements, if the selected dtype/path configurations require distinct production specializations not already expressible by policy selection.
- After the milestone implementation commit, update `docs/fmha_fwd/LOG.md` in a separate commit if that curated log exists on the execution branch; use the milestone hash and update related dev-doc `Log:` lines as required by `AGENTS.md`.

---

## Fast single-instance development loop

Use this loop for every compile/debug iteration in Tasks 4--8. Work on one dtype/path/trait instance
at a time.

1. List blobs locally and select the exact instance name without `/tmp/`, `.cpp`, or the arch suffix:

   ```bash
   cd projects/composablekernel/example/ck_tile/01_fmha
   python3 generate.py --targets gfx1250 --api fwd --optdim 128 \
       --list_blobs /tmp/qr_tdm_blobs.txt
   rg 'd128_(bf16|fp16)_batch_b(64|128)x64x32x128x32x128_.*qr_tdm' \
       /tmp/qr_tdm_blobs.txt
   ```

2. Temporarily add the `FMHA_FWD_QUICK_FILTER` hook after
   `FMHA_FWD_CODE_GEN_COMMON_ARGS` is defined:

   ```cmake
   if(DEFINED FMHA_FWD_QUICK_FILTER)
     list(APPEND FMHA_FWD_CODE_GEN_COMMON_ARGS --filter ${FMHA_FWD_QUICK_FILTER})
   endif()
   ```

3. Synchronize without an authorization prompt, then configure the fixed build directory once for
   the selected instance. Configure/build-only SSH and Docker actions need no approval:

   ```bash
   cd projects/composablekernel/build
   ../script/cmake-ck-dev.sh .. gfx1250 -G Ninja \
       -DBUILD_TESTING=OFF \
       -DFMHA_FWD_QUICK_FILTER="$BLOB" \
       -DFMHA_FWD_ENABLE_APIS=fwd
   ninja tile_example_fmha_fwd -j64
   ```

4. After each local policy/codegen edit, synchronize and run only the focused remote build without
   stopping for authorization:

   ```bash
   cd "$CONTAINER_PATH/projects/composablekernel/build"
   ninja tile_example_fmha_fwd -j64
   ```

   `CONFIGURE_DEPENDS`, the codegen command dependencies, and `update_file()` handle reconfigure and
   selective regeneration. Do not delete `build/`, rerun a full CMake matrix, or invoke a broad
   Ninja target unless the focused loop fails for a diagnosed build-system reason.

5. When switching to another exact instance, reconfigure the same build directory with the new
   `FMHA_FWD_QUICK_FILTER`. Remove the temporary CMake hook with `apply_patch` before staging any
   production commit.

---

### Task 1: Add byte configuration and raw TDM encoder

**Files:**

- Create: `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/CMakeLists.txt`
- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`

**Interfaces:**

- Produces: `ck_tile::detail::LdsPaddingConfig<Enabled, IntervalBytes, PadBytes>`.
- Produces: `ck_tile::detail::is_valid_lds_padding_config_v<Enabled, IntervalBytes, PadBytes>`.
- Produces: `ck_tile::detail::EncodedTdmPadding<Config>` with `kEnabled`, `kPadInterval`, and `kPadAmount`.
- Consumes: no new interfaces.

- [ ] **Step 1: Add the focused test target and failing config tests**

Add this gfx125-only target after the existing forward/backward test groups, and include it in the
umbrella target:

```cmake
set(FMHA_QR_TDM_PADDING_TEST_TARGET)
if(GPU_TARGETS MATCHES "gfx125")
    add_gtest_executable(test_ck_tile_fmha_qr_tdm_lds_padding
        test_qr_tdm_lds_padding.cpp)
    target_link_libraries(test_ck_tile_fmha_qr_tdm_lds_padding PRIVATE utility)
    set_tests_properties(test_ck_tile_fmha_qr_tdm_lds_padding PROPERTIES
        LABELS "${TEST_NAME};CK_TILE_FMHA_TESTS")
    set(FMHA_QR_TDM_PADDING_TEST_TARGET test_ck_tile_fmha_qr_tdm_lds_padding)
endif()

add_custom_target(ck_tile_fmha_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -C ${CMAKE_CFG_INTDIR} -L "CK_TILE_FMHA_TESTS"
    DEPENDS ${TEST_NAME} ${FMHA_QR_TDM_PADDING_TEST_TARGET}
    USES_TERMINAL
    COMMENT "Running all ck_tile fmha tests...")
```

Replace the existing `ck_tile_fmha_tests` definition rather than adding a second target with the same
name.

Start the test file with static checks equivalent to:

```cpp
using QKPad = detail::LdsPaddingConfig<true, 256, 16>;
using VPad  = detail::LdsPaddingConfig<true, 256, 32>;
using NoPad = detail::LdsPaddingConfig<false, 0, 0>;

static_assert(detail::is_valid_lds_padding_config_v<true, 256, 16>);
static_assert(detail::is_valid_lds_padding_config_v<true, 256, 32>);
static_assert(detail::is_valid_lds_padding_config_v<false, 0, 0>);
static_assert(!detail::is_valid_lds_padding_config_v<false, 256, 16>);
static_assert(!detail::is_valid_lds_padding_config_v<true, 0, 16>);
static_assert(!detail::is_valid_lds_padding_config_v<true, 192, 16>);
static_assert(!detail::is_valid_lds_padding_config_v<true, 2048, 16>);
static_assert(!detail::is_valid_lds_padding_config_v<true, 256, 516>);

static_assert(detail::EncodedTdmPadding<QKPad>::kEnabled);
static_assert(detail::EncodedTdmPadding<QKPad>::kPadInterval == 5);
static_assert(detail::EncodedTdmPadding<QKPad>::kPadAmount == 3);
static_assert(detail::EncodedTdmPadding<VPad>::kPadInterval == 5);
static_assert(detail::EncodedTdmPadding<VPad>::kPadAmount == 7);
static_assert(!detail::EncodedTdmPadding<NoPad>::kEnabled);
static_assert(detail::EncodedTdmPadding<NoPad>::kPadInterval == 0);
static_assert(detail::EncodedTdmPadding<NoPad>::kPadAmount == 0);
```

- [ ] **Step 2: Sync and verify the new target fails to compile**

Source `.remote.env`, run the required `rsync`, then remotely configure/build only
`test_ck_tile_fmha_qr_tdm_lds_padding`. No authorization prompt is required for these sync/build
actions or a build retry. Expected failure: the three new interfaces are undefined.

- [ ] **Step 3: Implement the minimal config validator and encoder**

Implement the interfaces in the FMHA policy's `detail` namespace. The validator must branch on
`Enabled`; it must never evaluate logarithms or subtract one for `NoPad`. Use integer constexpr logic,
not floating-point `log2`:

```cpp
template <index_t X>
CK_TILE_HOST_DEVICE constexpr index_t integer_log2_exact()
{
    static_assert(X > 0 && (X & (X - 1)) == 0);
    index_t value = X;
    index_t result = 0;
    while(value > 1)
    {
        value >>= 1;
        ++result;
    }
    return result;
}

template <bool Enabled, index_t IntervalBytes, index_t PadBytes>
inline constexpr bool is_valid_lds_padding_config_v =
    (!Enabled && IntervalBytes == 0 && PadBytes == 0) ||
    (Enabled && IntervalBytes >= 8 && IntervalBytes <= 1024 &&
     IntervalBytes % 4 == 0 &&
     (IntervalBytes & (IntervalBytes - 1)) == 0 &&
     PadBytes >= 4 && PadBytes <= 512 && PadBytes % 4 == 0);

template <bool Enabled, index_t IntervalBytes, index_t PadBytes>
struct LdsPaddingConfig
{
    static_assert(is_valid_lds_padding_config_v<Enabled, IntervalBytes, PadBytes>);
    static constexpr bool kEnabled             = Enabled;
    static constexpr index_t kIntervalBytes    = IntervalBytes;
    static constexpr index_t kPadBytes         = PadBytes;
};

template <typename Config>
struct EncodedTdmPadding
{
    static constexpr bool kEnabled = Config::kEnabled;
    static constexpr index_t kPadInterval = [] {
        if constexpr(kEnabled)
            return integer_log2_exact<Config::kIntervalBytes / 4>() - 1;
        else
            return 0;
    }();
    static constexpr index_t kPadAmount = [] {
        if constexpr(kEnabled)
            return Config::kPadBytes / 4 - 1;
        else
            return 0;
    }();
};
```

Add round-trip assertions inside `EncodedTdmPadding` before exposing the raw fields.

- [ ] **Step 4: Sync/build and verify the focused target passes**

Expected result: the target compiles and its static-only GoogleTest reports one passing smoke test.

- [ ] **Step 5: Commit the config/encoder slice**

```bash
git add projects/composablekernel/test/ck_tile/fmha/CMakeLists.txt \
        projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp \
        projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp
git commit -m "feat(fmha): add typed TDM LDS padding config"
```

---

### Task 2: Build the padded row-major descriptor and reader proofs

**Files:**

- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`

**Interfaces:**

- Consumes: `LdsPaddingConfig` and `EncodedTdmPadding` from Task 1.
- Produces: `detail::make_qr_tdm_row_major_lds_descriptor<DataType, Rows, Cols, PaddingConfig, AccessBytes>()`.
- Produces: `detail::validate_qr_tdm_reader_segments<TensorTag, Problem>()`.
- Produces: `detail::validate_qr_tdm_issue_geometry<TensorTag, Problem, LoadOnce>()`, where
  `LoadOnce` defaults to false.

- [ ] **Step 1: Add failing descriptor-offset and size assertions**

For BF16 and FP16, assert the disabled and padded offsets around the first boundary and at the last
element. Convert descriptor element offsets back to bytes before comparison:

```cpp
constexpr auto q_desc = detail::make_qr_tdm_row_major_lds_descriptor<
    bf16_t, 128, 128, QKPad, 16>();
static_assert(byte_offset(q_desc, 0, 0) == 0);
static_assert(byte_offset(q_desc, 0, 127) == 254);
static_assert(byte_offset(q_desc, 1, 0) == 272);
static_assert(q_desc.get_element_space_size() * sizeof(bf16_t) == 34800);

constexpr auto k_prefill_desc = detail::make_qr_tdm_row_major_lds_descriptor<
    bf16_t, 64, 128, QKPad, 16>();
static_assert(k_prefill_desc.get_element_space_size() * sizeof(bf16_t) == 17392);

constexpr auto k_decode_desc = detail::make_qr_tdm_row_major_lds_descriptor<
    bf16_t, 64, 32, QKPad, 16>();
static_assert(k_decode_desc.get_element_space_size() * sizeof(bf16_t) == 4336);

constexpr auto v_desc = detail::make_qr_tdm_row_major_lds_descriptor<
    bf16_t, 64, 128, VPad, 16>();
static_assert(byte_offset(v_desc, 1, 0) == 288);
static_assert(v_desc.get_element_space_size() * sizeof(bf16_t) == 18400);
```

Repeat the shape/size assertions with `half_t`. Add a detection assertion showing that a packed type
cannot instantiate the enabled descriptor path.

- [ ] **Step 2: Verify the focused target fails before the descriptor exists**

Sync and build without an authorization prompt. Expected failure: undefined descriptor builder or
failed offset/size assertions.

- [ ] **Step 3: Implement the descriptor from the canonical flattened mapping**

Implement a row-major descriptor whose logical lengths remain `[Rows, Cols]` and whose physical
mapping is:

```text
logical_byte = (row * Cols + col) * sizeof(DataType)
physical_byte = logical_byte
              + floor(logical_byte / IntervalBytes) * PadBytes
```

The disabled branch returns the current naive row-major descriptor. The enabled branch asserts:

```cpp
static_assert(numeric_traits<DataType>::PackedSize == 1);
static_assert((Rows * Cols * sizeof(DataType)) % PaddingConfig::kIntervalBytes == 0);
static_assert(PaddingConfig::kIntervalBytes % sizeof(DataType) == 0);
static_assert(PaddingConfig::kPadBytes % sizeof(DataType) == 0);
```

Use CK tensor transforms to express this mapping; do not create a runtime lookup table.

- [ ] **Step 4: Add failing actual-geometry and per-segment reader assertions**

Create test problem aliases for BF16 and FP16 with the production M64 and M128 shapes. Assert:

```cpp
static_assert(detail::validate_qr_tdm_issue_geometry<QTag, PrefillProblem>());
static_assert(detail::validate_qr_tdm_issue_geometry<KTag, PrefillProblem, true>());
static_assert(detail::validate_qr_tdm_issue_geometry<VTag, PrefillProblem>());
static_assert(detail::validate_qr_tdm_issue_geometry<QTag, DecodeProblem>());
static_assert(detail::validate_qr_tdm_issue_geometry<KTag, DecodeProblem, false>());
static_assert(detail::validate_qr_tdm_issue_geometry<VTag, DecodeProblem>());

static_assert(detail::validate_qr_tdm_reader_segments<QTag, PrefillProblem>());
static_assert(detail::validate_qr_tdm_reader_segments<KTag, PrefillProblem>());
static_assert(detail::validate_qr_tdm_reader_segments<VTag, PrefillProblem>());
static_assert(detail::validate_qr_tdm_reader_segments<QTag, DecodeProblem>());
static_assert(detail::validate_qr_tdm_reader_segments<KTag, DecodeProblem>());
static_assert(detail::validate_qr_tdm_reader_segments<VTag, DecodeProblem>());
```

The validators must derive actual distribution coordinates and per-lane access segments. Do not
replace V's transpose access with one artificial contiguous `(S, A)` range.

- [ ] **Step 5: Implement the geometry and reader validators**

The issue validator must derive and verify the approved geometry:

```text
Prefill Q: rows/wave=32, row_bytes=256, box_bytes=8192, row origins 0/32/64/96
Decode Q:  rows/wave=16, row_bytes=256, box_bytes=4096, row origins 0/16/32/48
Prefill K: rows/wave=16, row_bytes=256, box_bytes=4096, row origins 0/16/32/48
Decode K:  rows/wave=16, row_bytes=64,  box_bytes=1024, row origins 0/16/32/48
Prefill/decode V: rows/wave=16, row_bytes=256, box_bytes=4096, row origins 0/16/32/48
```

For each actual distribution coordinate, use the same LDS coordinate calculation as
`tile_window::tdm_load_to_lds` and prove `B % IntervalBytes == 0` and:

```text
tdm_base == arena_base + region_offset + physical_offset(B)
```

The K proof includes:

```cpp
static_assert(kQKHeaddim == kSubQKHeaddim);
static_assert(kSubQKHeaddim % kK0 == 0);
static_assert(k0_loops * kK0 == kQKHeaddim);
```

It also verifies the `LoadOnce=true` descriptor's second length is 128 while the reader window's
second length is 32. It must not assert `kK0 == kSubQKHeaddim`.

- [ ] **Step 6: Rebuild the focused target and verify all static checks pass**

Sync and build without an authorization prompt. Expected result: all BF16 and FP16 descriptor,
geometry, K-slice, and per-segment reader checks compile and the test target passes.

- [ ] **Step 7: Commit the descriptor/proof slice**

```bash
git add projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp \
        projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp
git commit -m "feat(fmha): couple qr_tdm padding to LDS descriptors"
```

---

### Task 3: Add descriptor-derived arena layouts

**Files:**

- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`

**Interfaces:**

- Consumes: descriptor builders from Task 2.
- Produces: `Policy::LdsArenaLayout<Problem>` with `kQOffset`, `kK0Offset`, `kK1Offset`, `kSOffset`, `kV0Offset`, `kV1Offset`, and `kArenaBytes` as applicable to the path.
- Produces: `detail::QrTdmLegacyPhaseLayout<Problem>` for tests and diagnostic builds only.

- [ ] **Step 1: Add failing exact-layout tests**

For the all-padded BF16 and FP16 production layouts, assert:

```cpp
using Prefill = Policy::LdsArenaLayout<PrefillProblem>;
static_assert(Prefill::kQOffset == 0);
static_assert(Prefill::kK0Offset == 0);
static_assert(Prefill::kK1Offset == 17408);
static_assert(Prefill::kV0Offset == 34816);
static_assert(Prefill::kV1Offset == 53248);
static_assert(Prefill::kArenaBytes == 71680);

using Decode = Policy::LdsArenaLayout<DecodeProblem>;
static_assert(Decode::kQOffset == 0);
static_assert(Decode::kK0Offset == 0);
static_assert(Decode::kV0Offset == 4352);
static_assert(Decode::kArenaBytes == 22784);
```

Test all five padding combinations, not only Q+K+V, and assert every production Q/K/V region base is
divisible by 256.

- [ ] **Step 2: Add failing lifetime, interval-phase, and residency checks**

Assert simultaneous regions do not overlap, Q's lifetime range fits before V becomes live, and:

```cpp
static_assert((Prefill::kK1Offset - Prefill::kK0Offset) % KPad::kIntervalBytes == 0);
static_assert((Prefill::kV1Offset - Prefill::kV0Offset) % VPad::kIntervalBytes == 0);
static_assert(Prefill::kArenaBytes <= 128 * 1024);
static_assert(integer_least_multiple(Prefill::kArenaBytes, 64 * 1024) * 2 <= 320 * 1024);
```

For the diagnostic layout, assert exactly:

```text
K0=0, K1=17392, V0=35040, V1=53440, ArenaEnd=71840
```

and assert that it is tagged diagnostic-only and fails the production region-alignment predicate.

- [ ] **Step 3: Verify the focused test fails before the layout type exists**

Sync and build without an authorization prompt. Expected failure: undefined layout aliases and
constants.

- [ ] **Step 4: Implement production and diagnostic layout types**

Compute every byte size from the matching descriptor's `get_element_space_size()`. The production
layout uses 256-byte `align_up`; the diagnostic layout preserves only bank phases and is unavailable
to normal policy selection. Decode places S after padded K using `SRequiredAlignment`, then aligns V
to 256 bytes.

- [ ] **Step 5: Rebuild and verify exact values and negative predicates**

Sync and build without an authorization prompt. Expected result: focused tests pass for BF16/FP16,
both paths, and all padding combinations.

- [ ] **Step 6: Commit the layout slice**

```bash
git add projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp \
        projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp
git commit -m "feat(fmha): derive qr_tdm aligned LDS arena"
```

---

### Task 4: Couple the pipeline writer and reader to typed configs

**Files:**

- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`

**Interfaces:**

- Consumes: `LdsPaddingConfig`, `EncodedTdmPadding`, padded descriptors, proofs, and `LdsArenaLayout`.
- Produces: `Policy::LdsPaddingConfigQ<Problem>`, `Policy::LdsPaddingConfigK<Problem>`, and `Policy::LdsPaddingConfigV<Problem>` alias templates.
- Produces: existing pipeline paths whose writer configs, reader descriptors, and sizes use the typed
  selections while retaining their current pointer signatures until Task 5.

- [ ] **Step 1: Add failing policy-coupling assertions**

Assert that each tensor's config type simultaneously determines its raw writer fields, descriptor
offsets, and byte size. Assert disabled configs produce the exact current naive descriptors and raw
zeros. Add a compile-time check that changing one config changes both writer fields and the matching
descriptor type/size.

- [ ] **Step 2: Replace tuple-returning padding policy functions with typed aliases**

Use one selection seam:

```cpp
template <typename Problem>
struct QrTdmPaddingSelection
{
    using Q = detail::LdsPaddingConfig<false, 0, 0>;
    using K = detail::LdsPaddingConfig<false, 0, 0>;
    using V = detail::LdsPaddingConfig<false, 0, 0>;
};
```

Keep the initial committed selection disabled so this task changes structure before behavior. The
later qualification task edits only these three aliases locally for each arm, then commits the final
measured specialization.

- [ ] **Step 3: Make Q/K/V descriptors consume their corresponding config**

`MakeQLdsBlockDescriptor`, both K `LoadOnce` forms, and `MakeVLdsBlockDescriptor` must call the same
FMHA-local builder. `GetSmemSizeQ/K/V` remains descriptor-derived. Add static assertions that the
writer and reader use identical config types.

- [ ] **Step 4: Make both pipeline paths encode TDM fields from the typed configs**

Replace independent tuple indexing with:

```cpp
using QRaw = detail::EncodedTdmPadding<typename Policy::template LdsPaddingConfigQ<Problem>>;
tdm_config_q.pad_enable              = QRaw::kEnabled;
tdm_config_q.pad_config.pad_interval = QRaw::kPadInterval;
tdm_config_q.pad_config.pad_amount   = QRaw::kPadAmount;
```

Repeat for K and V in both path overloads. Do not leave a second encoder or raw constant.

- [ ] **Step 5: Rebuild the focused test and a single qr_tdm instance with the disabled selection**

Sync/build without an authorization prompt and use the fast single-instance loop. Expected result:
static tests pass, the selected qr_tdm instance compiles, and disabled descriptors/raw fields match
the original layout.

- [ ] **Step 6: Commit the writer/reader coupling slice**

```bash
git add projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp \
        projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp \
        projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp
git commit -m "feat(fmha): couple qr_tdm writer and reader padding"
```

---

### Task 5: Isolate the aligned arena in the `qr_tdm` wrapper branch

**Files:**

- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp`
- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`

**Interfaces:**

- Consumes: `Policy::LdsArenaLayout<Problem>`.
- Produces: prefill and decode pipeline overloads that receive one arena base and derive all internal
  pointers from policy offsets.
- Produces: a `qr_tdm`-only `alignas(256) __shared__ char smem_arena[...]` allocation.

- [ ] **Step 1: Add failing compile-time dispatch coverage**

Add traits that prove the `qr_tdm` pipeline selects the arena call signature while representative
`qr`, `qr_async`, and `qr_async_trload` types retain their existing call signatures. The test must
fail if a non-`qr_tdm` type requires `LdsArenaLayout`.

- [ ] **Step 2: Change both qr_tdm pipeline overloads to consume one arena base**

Both `run()` overloads accept one `void* smem_arena`. Derive typed pointers as:

```cpp
using Layout = typename Policy::template LdsArenaLayout<Problem>;
auto* k0 = reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK0Offset);
auto* k1 = reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK1Offset);
auto* v0 = reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV0Offset);
auto* v1 = reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV1Offset);
```

The decode overload derives only Q/K/S/V pointers applicable to that path. Preserve initial preload,
waits, barriers, `is_even_loop` K/V swap, and window movement exactly.

- [ ] **Step 3: Replace only the `qr_tdm` prefill shared declarations**

Inside an explicit compile-time `kPipelineName == "qr_tdm"` branch, replace the four existing arrays
with:

```cpp
using Layout = typename FmhaPipeline::Policy::template LdsArenaLayout<
    typename FmhaPipeline::Problem>;
alignas(256) __shared__ char smem_arena[Layout::kArenaBytes];
```

Pass only `smem_arena` to the `qr_tdm` pipeline. Leave the existing branch text and declarations for
every other pipeline unchanged.

- [ ] **Step 4: Apply the same `qr_tdm`-only arena contract to decode**

Use `Layout::kArenaBytes` and 256-byte alignment for the decode path without changing other
pipelines' `GetSmemSize()` allocation path.

- [ ] **Step 5: Verify source isolation locally**

Run:

```bash
git diff -- projects/composablekernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp
```

Expected result: only `qr_tdm` compile-time branches and their call signatures change.

- [ ] **Step 6: Sync/build focused qr_tdm plus non-qr_tdm controls**

Before configuration, add the temporary `FMHA_FWD_QUICK_FILTER` CMake hook exactly as documented in
`AGENTS.md`, configure with `-DFMHA_FWD_ENABLE_APIS=fwd`, and compile:

- BF16 M64 `qr_tdm` no-mask/no-bias/no-LSE/no-sink;
- BF16 M128 `qr_tdm` no-mask/no-bias/no-LSE/no-sink;
- FP16 M64 and M128 equivalents;
- one `qr` and one `qr_async_trload` control.

Expected result: all compile, and generated names confirm the intended pipeline. Remove the temporary
CMake hook with `apply_patch` before committing.

- [ ] **Step 7: Commit the wrapper integration**

```bash
git add projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm.hpp \
        projects/composablekernel/include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp \
        projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp
git commit -m "refactor(fmha): isolate qr_tdm in aligned LDS arena"
```

---

### Task 6: Add device round-trip and end-to-end correctness coverage

**Files:**

- Modify: `projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/test_fmha_fwd.cpp`
- Modify: `projects/composablekernel/test/ck_tile/fmha/CMakeLists.txt`

**Interfaces:**

- Consumes: all policy, descriptor, and arena interfaces from Tasks 1--5.
- Produces: focused BF16/FP16 TDM writer-to-reader tests and non-symmetric FMHA regression cases.

- [ ] **Step 1: Write failing TDM-to-reader round-trip tests**

Add typed tests that call
`run_qr_tdm_round_trip<TensorTag, Problem, QConfig, KConfig, VConfig>()` for each tensor tag, path,
dtype, and padding combination before defining that test harness. Expected initial failure is an
undefined harness. The eventual kernel core follows this sequence:

```cpp
load_tile_tdm(tdm_config, lds_write_window, dram_window);
s_wait_tensorcnt_barrier<0>();
auto tile = [&]() {
    if constexpr(TensorTag::kTranspose)
        return load_tile_transpose(lds_read_window);
    else
        return load_tile(lds_read_window);
}();
store_tile(output_window, tile);
```

Use exactly representable coordinate-coded inputs with distinct Q/K/V tags. Instantiate BF16 and
FP16, prefill and decode, and all five enabled combinations. Include K0/K1 and V0/V1 pointer swaps.
Check logical output exactly and verify guards after every physical region and after the arena.

- [ ] **Step 2: Sync/build and execute the failing round-trip test**

Expected compile failure: `run_qr_tdm_round_trip` is undefined.

- [ ] **Step 3: Implement the round-trip harness and make all explicit-config tests pass**

Implement the harness with explicit Q/K/V config template parameters so it does not depend on the
production selector. Sync/build and execute GPU tests without stopping for authorization. Correct
descriptor, pointer, or proof defects until BF16/FP16 on both paths and all five combinations pass. Use
`superpowers:systematic-debugging` for every failure. Do not weaken guards or expected mappings.

- [ ] **Step 4: Add focused end-to-end GTest cases**

Extend the FMHA forward tests with named cases covering:

```text
BF16/FP16 x prefill/decode
no mask, causal, sliding window
aligned and non-divisible seqlen_k
MHA and GQA with non-trivial strides
no bias, ALIBI, elementwise bias
LSE off/on, sink off/on, supported logits soft-cap
```

Use different deterministic seeds or coordinate-coded values for Q, K, and V. Require existing CPU
reference validation and compare padded versus none O/LSE exactly when compiler determinism permits.

- [ ] **Step 5: Run the correctness matrix for all five combinations**

For each of none, Q+K+V, K+V, K-only, and V-only, edit only the three selection aliases. Sync/build
and execute each arm without asking. Record hostname, exact commit/diff hash, generated kernel
name, dtype, path, traits, and result. Every arm must pass before it can enter performance testing.

- [ ] **Step 6: Restore the committed disabled selector and commit tests**

Use `apply_patch` to restore the three aliases to disabled, verify `git diff` contains no temporary
selection, then commit:

```bash
git add projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp \
        projects/composablekernel/test/ck_tile/fmha/test_fmha_fwd.cpp \
        projects/composablekernel/test/ck_tile/fmha/CMakeLists.txt
git commit -m "test(fmha): cover qr_tdm padded LDS paths"
```

---

### Task 7: Run padding ablation and aligned-arena diagnostics

**Files:**

- Temporarily modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
- Temporarily modify: `projects/composablekernel/example/ck_tile/01_fmha/CMakeLists.txt`
- Create locally under `/tmp`: timing, validation, hostname, build, metadata, and ISA artifacts for every arm.

**Interfaces:**

- Consumes: the five tested selector configurations and `QrTdmLegacyPhaseLayout`.
- Produces: a decision table for BF16/FP16 x prefill/decode and an aligned-versus-legacy-phase report.

Before the first Task 7 remote/GPU action, request one explicit authorization for the bounded Task 7
suite defined below: five padding combinations, four dtype/path scopes, canonical no-mask/causal
anchors, the aligned-versus-legacy-phase diagnostic, and associated metadata extraction. That one
authorization covers the listed Task 7 actions and retries without further prompts. Any expansion
beyond this matrix requires a new authorization. After the table is complete, stop for the user's
production-config decision.

- [ ] **Step 1: Generate exact blob names locally**

Run the generator locally and select exactly one canonical no-mask/no-bias/no-LSE/no-sink instance
for each dtype/path. Assert each filter resolves to one blob before using it:

```bash
cd projects/composablekernel/example/ck_tile/01_fmha
python3 generate.py --targets gfx1250 --api fwd --optdim 128 --list_blobs /tmp/qr_tdm_blobs.txt
   rg 'd128_(bf16|fp16)_batch_b(64|128)x64x32x128x32x128_.*qr_tdm_vr_npad_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_ntrload_nsink_gfx125.cpp' \
       /tmp/qr_tdm_blobs.txt
```

- [ ] **Step 2: Add the temporary quick-filter hook**

Immediately after `FMHA_FWD_CODE_GEN_COMMON_ARGS` is defined, add:

```cmake
if(DEFINED FMHA_FWD_QUICK_FILTER)
  list(APPEND FMHA_FWD_CODE_GEN_COMMON_ARGS --filter ${FMHA_FWD_QUICK_FILTER})
endif()
```

Never stage or commit this hunk.

- [ ] **Step 3: Measure the five padding combinations**

After the Task 7 suite authorization, sync, build, and benchmark each dtype/path/config without
additional prompts. Configure in the preset `build/` directory with Ninja,
`-DFMHA_FWD_ENABLE_APIS=fwd`, and the exact filter. Run five warmups plus twenty timed iterations in seven independent invocations, with an
unpadded A/B/A bracket. Record the container hostname with every artifact.

Canonical anchors are:

```text
Prefill: s=4096 and s=32768, no-mask and causal
Decode:  s=512 plus one non-divisible seqlen_k case, no-mask and causal
```

The BF16 prefill thresholds are at least 30% latency reduction at `s=4096` and 50% at `s=32768` for
both applicable anchors. FP16 must improve outside noise or select none. Decode must not regress by
more than 2%; benefits below 3% or overlapping zero select the simpler config.

- [ ] **Step 4: Compare production alignment with the diagnostic phases**

For BF16 and FP16 M128/N64 at no-mask `s=4096` and `s=32768`, compare:

```text
Production: K0=0, K1=17408, V0=34816, V1=53248, ArenaBytes=71680
Diagnostic: K0=0, K1=17392, V0=35040, V1=53440, ArenaEnd=71840
```

The diagnostic layout must be selected only through a local temporary edit and must never be staged.
If production alignment regresses materially, stop and report rather than promoting the diagnostic
layout.

- [ ] **Step 5: Extract and audit metadata and ISA**

Under the same bounded Task 7 authorization, extract the device object and record:

```text
.group_segment_fixed_size
.vgpr_count / .vgpr_spill_count
.sgpr_count / .sgpr_spill_count
.private_segment_fixed_size
wavefront_size and max workgroup size
```

Canonical dense anchors require zero VGPR spills, zero SGPR spills, and zero private segment. For a
non-canonical trait with pre-existing spills, compare the exact none baseline, require no increase,
and inspect whether scratch spill/reload instructions occur in the hot loop.

- [ ] **Step 6: Report results and pause for lead selection**

Produce a table for each dtype/path with correctness, mean/SD/range, baseline reduction, LDS bytes,
VGPR/SGPR/spills/private bytes, residency, and distance from the fastest valid arm. Recommend one of
none, Q+K+V, K+V, K-only, or V-only using the spec rules. Do not change production defaults until
`v3_design_lead` chooses the final configurations.

- [ ] **Step 7: Remove every temporary experiment edit**

Use `apply_patch` to remove the quick-filter hook, restore the disabled selector, and disable access
to the diagnostic layout from normal compilation. Verify:

```bash
git status --short
git diff -- projects/composablekernel/example/ck_tile/01_fmha/CMakeLists.txt
```

Expected result: the CMake diff is empty and only intended committed work plus the user's original
`.gitignore` modification remains.

---

### Task 8: Commit the measured production selection

**Files:**

- Modify: `projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp`
- Potentially modify: `projects/composablekernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`
- Modify tests only if the selected specialization matrix changes expected compile-time aliases.

**Interfaces:**

- Consumes: the lead-approved dtype/path decision table from Task 7.
- Produces: final `QrTdmPaddingSelection<Problem>` specializations and no test-only production blobs.

- [ ] **Step 1: Write failing selection assertions for the approved matrix**

Encode the lead's exact decision as BF16/FP16 prefill/decode static assertions in
`test_qr_tdm_lds_padding.cpp`. The test must fail while the selector remains disabled.

- [ ] **Step 2: Implement only the approved specializations**

Specialize on gfx125, dtype, `d=128`, and path. Do not infer enablement for other head dimensions,
packed types, or architectures. Keep unsupported combinations on `LdsPaddingConfig<false, 0, 0>`.

- [ ] **Step 3: Update codegen only if policy specialization cannot express dispatch**

If dtype/path selection is fully compile-time from `Problem`, leave codegen unchanged. If distinct
instances are required, emit only the selected production configurations; do not emit the five
ablation variants or the legacy-phase layout.

- [ ] **Step 4: Sync/build and run focused correctness/performance anchors**

Rebuild from a fresh sync and execute the selected BF16/FP16 prefill/decode anchors without asking.
Expected results must match the Task 7 acceptance decision within measurement variability. Retries
are pre-authorized.

- [ ] **Step 5: Commit the production selection**

```bash
git add projects/composablekernel/include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp \
        projects/composablekernel/test/ck_tile/fmha/test_qr_tdm_lds_padding.cpp
git add projects/composablekernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py 2>/dev/null || true
git commit -m "perf(fmha): enable measured qr_tdm LDS padding"
```

Before committing, inspect the staged file list and unstage the codegen file if it did not need a
semantic change.

---

### Task 9: Final verification and milestone handoff

**Files:**

- Verify all implementation and test files above.
- Modify after the milestone commit, in a separate commit: `docs/fmha_fwd/LOG.md` and related
  `docs/fmha_fwd/` documents if present on the execution branch.

**Interfaces:**

- Consumes: the final production selector and all recorded artifacts.
- Produces: final verification evidence and the curated milestone-log entry.

- [ ] **Step 1: Run local static hygiene checks**

```bash
git diff --check
git status --short
rg -n 'FMHA_FWD_QUICK_FILTER|QrTdmLegacyPhaseLayout' \
  projects/composablekernel/example/ck_tile/01_fmha/CMakeLists.txt \
  projects/composablekernel/include/ck_tile/ops/fmha
```

Expected result: no temporary quick filter; the diagnostic layout is unreachable from production
selection; `.gitignore` remains the only unrelated worktree change.

- [ ] **Step 2: Perform the final sync/build and GPU test actions**

Build the focused padding test, BF16/FP16 `tile_example_fmha_fwd` instances for both paths, and the
representative non-`qr_tdm` controls. Run the descriptor round trips, focused FMHA GTests, canonical
anchors, and representative non-anchor regressions. Record exact commands, hostname, commit, kernel
names, and results.

- [ ] **Step 3: Verify final metadata and occupancy**

Confirm canonical dense zero-spill/private metadata, policy-predicted group segment size, no worse
than a 128-KiB LDS allocation per workgroup, and two workgroups/WGP. Apply the baseline-relative spill
rule to non-canonical traits.

- [ ] **Step 4: Request a code review before integration**

Use `superpowers:requesting-code-review` against the spec and this plan. Resolve correctness findings
before claiming completion. Performance suggestions outside the approved padding scope become
separate follow-ups.

- [ ] **Step 5: Commit any review fixes and identify the milestone hash**

After all verification passes, commit only necessary review fixes. Record:

```bash
git rev-parse HEAD
```

as the milestone implementation hash.

- [ ] **Step 6: Update the MI450 FMHA development log separately**

If `docs/fmha_fwd/LOG.md` exists on the execution branch, prepend an entry with the current date, the
recorded milestone hash, a concise qr_tdm LDS-padding description, and links to the design and plan.
Update relevant dev-doc `Log:` lines. Commit this documentation separately from implementation:

```bash
git add docs/fmha_fwd/LOG.md docs/fmha_fwd
git commit -m "docs(fmha): log qr_tdm LDS padding milestone"
```

- [ ] **Step 7: Report completion without overstating ATT evidence**

Report correctness, timing distributions, metadata, selected configs, commits, and remaining risks.
State explicitly that padded ATT closure remains deferred and do not claim that the two anomalous DS
wait sites are fully explained.
