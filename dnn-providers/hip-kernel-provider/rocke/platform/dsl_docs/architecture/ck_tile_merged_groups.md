# CK Tile "merged groups" (`NumGroupsToMerge`) — how it works

Analysis of the merged-groups path in CK Tile's **grouped convolution forward**
kernel, written as the basis for porting the same capability to ROCKE.

Paths are relative to `projects/composablekernel/include/ck_tile/`, except those
beginning `dispatcher/` or `example/`, which are relative to
`projects/composablekernel/`. Everything below was read in the tree at `develop`
(commit `8f77b587ff4`); line numbers are from that state.

> **Naming.** There is no `MergeGroups` / `kMergeGroups` identifier anywhere.
> The feature is spelled `NumGroupsToMerge` everywhere in code, abbreviated
> `Gm` in this doc. The string `"MergedGroups"` appears only in `GetName()`
> output.

---

## 1. TL;DR

For a **depthwise** convolution (`C_per_group == 1`), every operand of the
per-group GEMM degenerates to width 1, so every load and every store is a
scalar. Merged groups folds `Gm` consecutive conv groups into a single GEMM
tile by making the **group index the fastest-varying factor of both GemmM and
GemmN**. Because NHWGC stores `G` contiguously when `C == 1`, that turns the
scalar accesses into `Gm`-wide vector accesses.

The tile then computes a `Gm × Gm` block of (M-side group, N-side group) pairs,
of which only the **diagonal** is real work. The off-diagonal results are
computed in registers and then **thrown away by the C descriptor**, which routes
them to an invalid coordinate so the buffer store is predicated off.

Weights are **dense and real** — nothing is zero-padded, nothing is
block-diagonal, and there is no scratch buffer.

---

## 2. Where it lives

| Concern | File | Lines |
|---|---|---|
| Trait plumbing, `AsLayoutFwd` | `ops/grouped_convolution/utils/grouped_convolution_utils.hpp` | 142, 161–165, 173, 186–190 |
| A / B / C descriptors | `ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp` | 699–946, 1259–1320, 1389–1450 |
| `xor_t` transform | `core/algorithm/coordinate_transform.hpp` | 1322–1412 |
| Kargs, grid, windows, validation | `ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp` | 134–375, 590–616, 713–765, 897–1136, 1138–1196, 1475–1502 |

Forward only. The backward kernels carry six separate
`// TODO Add support for NumGroupsToMerge > 1` comments and hardcode `Gm = 1`.

---

## 3. The mapping

Per-group forward conv as a GEMM, with `C_`/`K_` meaning *per-group* channel
counts (`C_ = C/G`, `K_ = K/G`):

```
GemmM = N · Do · Ho · Wo          GemmN = K_          GemmK = Z · Y · X · C_
```

With merging (`Gm > 1`, which forces `C_ == 1`):

```
GemmM     = N · Do · Ho · Wo · Gm
GemmN     = K_ · Gm
GemmK     = Z · Y · X                    (C_ == 1)
GemmBatch = ceil(G / Gm)
```

`GemmBatch` and the per-batch strides are set identically in the 1D/2D/3D kargs
constructors (`grouped_convolution_forward_kernel.hpp:134-141`, `:234-241`,
`:341-348`, and `:159`/`:259`/`:366`):

```cpp
group_stride_a = args.C_ * NumGroupsToMerge;
group_stride_b = args.K_ * args.C_ * NumGroupsToMerge * prod(filter_spatial_lengths);
group_stride_c = args.K_ * NumGroupsToMerge;
GemmBatch      = integer_divide_ceil(args.G_, NumGroupsToMerge);
```

`gridDim` is `(TilePartitioner::GridSize(GemmM, GemmN), GemmBatch, n_splits)`
(`:760-765`), so `blockIdx.y` indexes a *bundle* of `Gm` groups rather than a
single group.

---

## 4. Why depthwise specifically

NHWGC memory order is `N, H, W, G, C`, i.e. `C` is fastest, then `G`. The
address of one input element is

```
n·NStride + h·HStride + w·WStride + g·(C_) + c
```

When `C_ == 1` the `g` stride is exactly **1** — consecutive groups are
adjacent in memory. So:

- **Unmerged:** the GEMM-K axis runs over `(Z,Y,X,C_)` with `C_ = 1`, so nothing
  contiguous sits under K; GemmN is `K_ = 1`, one element wide. Every A load, B
  load and C store is a single element.
- **Merged:** making `g` the fastest factor of GemmM and GemmN exposes a run of
  `Gm` contiguous channels on both sides. Loads and stores become `Gm`-wide.

This is the entire economic point. It does **not** apply to non-depthwise
grouped conv: with `C_ > 1` the group stride is `C_`, the channel run is already
`C_` wide, and `IsSupportedArgument` rejects `Gm > 1` outright
(`grouped_convolution_forward_kernel.hpp:1008-1031`, `:1108-1131`).

---

## 5. A descriptor — input

`MakeADescriptor_M_K<NHWGC>`, 2D Default spec, merged branch
(`transform_conv_fwd_to_gemm.hpp:905-939`). The comment at `:907` states the
premise directly: *"IsSupported ensures C == 1 to allow reading on G dimension"*.

```
naive( (N_, Hi_, Wi_, Gm),
       strides = (NStrideTensorA_, HiStride_, WiStride_, GStrideTensorA_) )
  -> pad H, pad W                          (left/right conv padding)
  -> embed (Y, Ho) over H, (X, Wo) over W  (dilation / stride)
  -> merge (N_, Ho_, Wo_, Gm)   -> M
     merge (Y_, X_)             -> K
```

Note the descriptor has **no `C_` dimension at all** — `C_ == 1`, so the group
axis takes its place as the innermost one, and `Gm` lands as the last (fastest)
factor of the M merge.

Because A is now merged, `AsLayoutFwd` flips the A operand to **ColumnMajor**
(`grouped_convolution_utils.hpp:186-190`):

```cpp
using AsLayoutFwd = std::conditional_t<NumGroupsToMerge == 1,
                                       /* RowMajor    */ ...,
                                       /* ColumnMajor */ ...>;
```

and `MakeABlockWindow` correspondingly **transposes** the descriptor (swapping
the M and K dim ids) and creates the tile window as `(KPerBlock, MPerBlock)` at
`{0, block_idx_m}` (`grouped_convolution_forward_kernel.hpp:1138-1196`).

---

## 6. B descriptor — weights

`MakeBDescriptor_N_K<GKYXC>`, merged branch (`:1297-1317`):

```
naive( (K_, Gm, ZYX_ * C_),
       strides = (KStrideTensorB_, GStrideTensorB_, CStrideTensorB_) )
  -> merge (K_, Gm) -> N
     pass_through    -> K
```

**The weights are the real, dense weights.** The `Gm` groups' filters are simply
read side by side. There is no zeroing, no block-diagonal construction, and no
extra memory. This is the part that is easy to guess wrong: the redundancy is
*not* removed on the operand side, it is removed on the **output** side.

---

## 7. C descriptor — the correctness mechanism

This is the heart of the feature. `MakeCDescriptor_M_N<NHWGK>`, merged branch
(`:1405-1449`), verbatim:

```cpp
const auto nhwo_groups_k_1_desc =
    make_naive_tensor_descriptor(make_tuple(N_, Ho_, Wo_, NumGroupsToMerge, K_, 1),
                                 make_tuple(NStrideTensorC_, HoStride_, WoStride_,
                                            GStrideTensorC_, KStrideTensorC_,
                                            GStrideTensorC_),          // <-- note
                                 number<VectorSizeC>{}, I1);

// Padd 1 to NumGroupsToMerge
const auto padded_desc = transform_tensor_descriptor(
    nhwo_groups_k_1_desc,
    make_tuple(make_merge_transform(make_tuple(N_, Ho_, Wo_)),
               make_pass_through_transform(NumGroupsToMerge),
               make_pass_through_transform(K_),
               make_pad_transform(1, 0, NumGroupsToMerge - 1)),
    ...);

// We need only matrices from diagonal. X_or returns 0 for the same
// values. So if matrices is not on diagonal then it will be stored in padding.
// To avoid use of modulo after xor we assume that NumBatch to merge is power of 2.
static_assert(NumGroupsToMerge == 1 || ... || NumGroupsToMerge == 64);

const auto unmerged_padded_desc = transform_tensor_descriptor(
    padded_desc,
    make_tuple(make_pass_through_transform(NDoHoWo),
               make_xor_transform<decltype(make_tuple(Gm, Gm)), /*ApplyModulo=*/false>(
                   make_tuple(Gm, Gm)),
               make_pass_through_transform(K_)),
    make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}),
    make_tuple(sequence<0>{}, sequence<1, 3>{}, sequence<2>{}));

// Merge To M, N
return transform_tensor_descriptor(
    unmerged_padded_desc,
    make_tuple(make_merge_transform(make_tuple(NDoHoWo, NumGroupsToMerge)),
               make_merge_transform(make_tuple(K_, NumGroupsToMerge))),
    make_tuple(sequence<0, 1>{}, sequence<2, 3>{}),
    make_tuple(sequence<0>{}, sequence<1>{}));
```

### 7.1 The two group axes

After the final merge, an accumulator coordinate `(m, n)` decomposes into:

- from `merge(NDoHoWo, Gm)` → a spatial part and an **M-side group** `g_m`
- from `merge(K_, Gm)` → a channel part and an **N-side group** `g_n`

In `unmerged_padded_desc` these are upper dims **1** (`g_m`) and **3** (`g_n`) —
that is what `sequence<1, 3>` wires into the xor.

### 7.2 What the xor does

`xor_t` (`core/algorithm/coordinate_transform.hpp:1322-1412`) with
`ApplyModulo = false`:

```cpp
idx_low(number<0>{}) = idx_up[number<0>{}];                         // g_m, untouched
idx_low(number<1>{}) = idx_up[number<1>{}] ^ idx_up[number<0>{}];   // g_n ^ g_m
```

So:

- `idx_low[0] = g_m` → feeds the **real group axis** (stride `GStrideTensorC_`),
- `idx_low[1] = g_n ^ g_m` → feeds the **padded axis**, whose underlying length
  is **1** and which was padded out to `Gm` by `make_pad_transform(1, 0, Gm-1)`.

The padded axis is in bounds only when `g_n ^ g_m == 0`, i.e. **`g_n == g_m`**.
Every off-diagonal `(g_m, g_n)` maps to a padded (invalid) coordinate, and
`pad_transform` reports it invalid, so the buffer store is predicated off. The
diagonal elements pass through untouched and land at the correct address.

`ApplyModulo = false` is what forces `Gm` to be a power of two ≤ 64 (the
`static_assert`): without the modulo, `g_n ^ g_m` only stays inside `[0, Gm)` if
`Gm` is a power of two.

### 7.3 Why the padded axis carries stride `GStrideTensorC_`, not 0

Easy to miss, and it matters. The 6th dim of `nhwo_groups_k_1_desc` has length 1
but stride `GStrideTensorC_`. If the invalidity check were somehow dropped, an
off-diagonal element would compute a **real, in-range, wrong** address — it
would corrupt a neighbouring group's output rather than hit a harmless slot.
The correctness therefore rests entirely on `pad_transform`'s validity flag
propagating into the store predicate. Note that `xor_t` itself always reports
valid (`is_valid_upper_index_always_mapped_to_valid_lower_index()` returns
`true`), so it contributes nothing to masking — it only *steers* the index.

**Consequence: no scratch buffer is needed and nothing extra is written.**

---

## 8. Cost model

> **Not load-bearing.** This section does not change the implementation — it is
> kept only as background. Skip to §11 for the part that does.

The usual framing is "merged groups costs `Gm×` more math". That is true against
*useful* FLOPs but it is the wrong comparison, because the unmerged depthwise
GEMM is itself massively padded: `GemmN = K_ = 1` gets rounded up to `NPerBlock`.

Per output tile, with block tile `(MB, NB)` and `GemmK = K`:

| | MACs executed |
|---|---|
| Unmerged | `G · M · NB · K` (N padded 1 → `NB`) |
| Merged, `Gm ≤ NB` | `(G/Gm) · (M·Gm/MB) · MB · NB · K` = **`G · M · NB · K`** |
| Merged, `Gm > NB` | `G · M · Gm · K` |

So **while `Gm ≤ NPerBlock`, merged groups is FLOP-neutral** — it executes
exactly the same number of MACs as the unmerged padded GEMM, but issues vector
loads instead of scalar ones. Useful-FLOP efficiency is

```
K_ · Gm / NPerBlock
```

which is why merging is worth doing at all, and why `Gm` should not exceed
`NPerBlock`. This matches the host-side config restriction in §10, where the
only shipped fwd merged tiles have `NPerBlock ∈ {16, 32}` while `Gm` goes to
8/16/32.

*(This derivation is mine, not something stated in the CK source — flagging it
as the part most worth a sanity check.)*

---

## 9. Constraints

From `IsSupportedArgument` (`grouped_convolution_forward_kernel.hpp:897-1136`),
the traits, and the `static_assert`s:

| Constraint | Where |
|---|---|
| `Gm ∈ {1,2,4,8,16,32,64}` (power of two) | `transform_conv_fwd_to_gemm.hpp:1443-1446` static_assert |
| `ConvC == 1` (per-group), i.e. strictly depthwise | `:1008-1031`, `:1108-1131` |
| `ConvG % Gm == 0` | `:1022-1027`, `:1123-1128` |
| `VectorSizeB` must be 1 (B-side requires `ConvC % VectorSizeB == 0`, `ConvC == 1`) | `:1043-1063` |
| C-side may use `ConvG % VectorSizeC == 0` **instead of** `ConvK % VectorSizeC == 0` ("Try to read over G") | `:1066-1106` |
| A operand becomes ColumnMajor | `grouped_convolution_utils.hpp:186-190` |
| `kPadM = kPadN = kPadK = true` (hardcoded) | `grouped_convolution_utils.hpp:161` ff. |
| `FixedVectorSize = true` | `grouped_convolution_utils.hpp:165` |
| No split-K on forward | `grouped_convolution_forward_kernel.hpp:590` |
| Layouts: NHWGC / GKYXC / NHWGK only (NGCHW, GKCYX, NGKHW unsupported in ck_tile fwd) | `transform_conv_fwd_to_gemm.hpp:140` |
| Backward: not supported | six `// TODO Add support for NumGroupsToMerge > 1` |

There are also `static_assert`s gating the merged path at
`grouped_convolution_forward_kernel.hpp:75-77` and inside `CheckGemmAsserts()`
`:598-616`.

---

## 10. How a merged config actually gets picked

- `example/ck_tile/20_grouped_convolution/conv_configs.hpp:231-254` defines
  `ConvConfigComputeV3_merged_groups` (`Gm = 2`, tile 16×32×32) but it is
  **dead code** — never instantiated.
- There is **no merged-groups test coverage** under `test/ck_tile/`; every test
  hardcodes `NumGroupsToMerge = 1`.
- Real selection happens in the dispatcher:
  `dispatcher/codegen/grouped_conv/grouped_config_rules_full.py:1199-1211`, with

  ```python
  _FWD_GM_TILES = [(64, 16, 16), (128, 32, 32)]
  FeatureSpec(num_groups_to_merge=8 / 16 / 32)
  ```

  Candidates are filtered by the generated `is_supported()` predicates, then
  picked. Note the tiles have `NPerBlock` of 16 and 32 — consistent with §8.

So the feature is shipped through codegen, not through the examples or tests,
and its only in-tree exercise is the dispatcher path.

---

## 11. Where the vector actually comes from: the **A load**, not the C store

This is the part most likely to be misread, so it is worth stating flatly.

### 11.1 The A-side win is the mechanism

In the merged A descriptor (§5) the `Gm` axis carries stride
`GStrideTensorA_ = C_ = 1` — i.e. **contiguous**. It is the innermost factor of
`merge(N_, Ho_, Wo_, Gm) → M`, and the merged A operand is **ColumnMajor**
(`grouped_convolution_utils.hpp:186-190`), so `M` is the tile window's fast
dimension (`MakeABlockWindow`, `grouped_convolution_forward_kernel.hpp:1138-1196`,
window shape `(KPerBlock, MPerBlock)`).

Result: the input load runs `Gm` contiguous channels wide instead of one. That
is the entire point of the feature. Unmerged depthwise gives `VectorSizeA = 1`
because `C_ == 1`; merged gives `VectorSizeA = min(Gm, cap)`.

The two merged forward instances actually shipped in-tree confirm this
(`experimental/grouped_convolution_tile_instances/configs/forward/tests/nhwgc_fp16.conf`):

| `Gm` | A src-scalar-per-vector | B src-scalar-per-vector | CShuffle store vector |
|---|---|---|---|
| 32 | **4** | 1 | **1** |
| 8 | **8** | 8 | **4** |

### 11.2 The C store is a tuning choice, not the mechanism

Note the last column: one merged instance stores **1 element at a time**, the
other 4. If the wide output store were the point of merged groups, the `Gm = 32`
instance would be pointless — yet it is the one with the larger merge degree.

The geometry explains why the store cannot simply ride along. Within a merged
tile, `m = nhwo·Gm + g_m` and `n = k·Gm + g_n`, valid iff `g_n == g_m`. So for a
fixed `m` exactly **one** `n` is valid. The valid outputs *are* contiguous in
memory — with `K_ == 1` the address is `nhwo·G + g_m`, stride 1 across `g_m` —
but they lie on the **tile diagonal**, not along a row or a column. Harvesting
them as one wide store requires a diagonal gather, which a generic 2D output
distribution does not perform.

The `ConvG % VectorSizeC == 0` fallback at `:1066-1106` ("Try to read over G",
guarded on `NumGroupsToMerge > 1`) is what makes a `VectorSizeC > 1` *legal* for
depthwise at all — without it, `ConvK == 1` would reject every vector width. It
permits a wide store; it does not demonstrate that one is achieved.

### 11.3 What is still unsettled

Whether the `Gm = 8, VectorSizeC = 4` instance issues a genuinely 4-wide *valid*
store, or a 4-wide store with three lanes masked off, is not decidable by reading
the descriptors. Settle it by building that one instance and counting
`global_store_dwordx2/x4` versus `global_store_short` in the ISA. **This does not
block the ROCKE port** — §12 shows the port can take the A-side win with the
store left scalar, exactly as ROCKE's wgrad path already does.

---

## 12. ROCKE comparison — epilogue, and the existing `group_merge`

### 12.1 Is the CK Tile epilogue the same as the ROCKE epilogue?

**Structurally yes.** Both are LDS-staged "cshuffle" epilogues: scatter the MFMA
accumulator into an LDS tile at its native lane layout, barrier, then read back
with a flat row-major thread distribution and store. That re-permutation is what
decouples the store pattern from the MFMA output layout.

| | CK Tile | ROCKE |
|---|---|---|
| Class / function | `CShuffleEpilogue`, `ops/epilogue/cshuffle_epilogue.hpp:82` | `_emit_cshuffle_epilogue`, `library/kernels/common/conv_implicit_gemm.py:1796` |
| Selected for grouped conv fwd | **unconditionally** — `example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp:94`, `:119` (`TdmEpilogue` only for TDM pipelines) | opt-in `epilogue="cshuffle"`; **mandatory** whenever `vec_c > 1` (`conv_implicit_gemm.py:439-452`) |
| Conditioned on `NumGroupsToMerge` / `group_merge` | **No** | No |
| Store width source | `GetVectorSizeC()` `:192-213`; `FixedVectorSize = true` (`grouped_convolution_utils.hpp:165`) returns the host-supplied `VectorSizeC` verbatim | `store_vec` from `CShuffleEpilogue.from_grid()`, capped 16 B |
| **Where the diagonal mask lives** | in the **C descriptor** — `xor` + `pad(1,0,Gm-1)` makes off-diagonal coords invalid (§7) | in the **`addr_fn` `valid` predicate** the epilogue already ANDs into its store guard |

So the user-visible behaviour is the same (a masked store); the difference is
*where* the mask is expressed. CK encodes it in the coordinate transform chain;
ROCKE folds an extra `cmp_eq(xor(gm_m, gm_n), 0)` term into a predicate the
epilogue computes anyway. The rationale comment at
`conv_implicit_gemm_wgrad.py:402-423` argues ROCKE's form is strictly cheaper —
the descriptor form costs an extra magic-division scan and buys nothing.

One consequence worth flagging: **CShuffle is not the merged-groups mechanism in
CK either.** It is the baseline epilogue for *all* grouped convolution — forward,
backward-data, backward-weight, and backward-weight-two-stage invokers all
instantiate it, at `Gm == 1` as well. It is not something merged groups turns on.

### 12.2 Relationship to ROCKE's existing `group_merge`

ROCKE already implements this idea for **wgrad** implicit GEMM
(`dnn-providers/hip-kernel-provider/rocke/library/kernels/common/conv_implicit_gemm_wgrad.py`):

- `group_merge: int = 1` field with a rationale comment (`:402-423`) that names
  CK's `NumGroupsToMerge` and the xor-onto-padded-dim trick explicitly;
- the `merged_problem` property (`:507-518`) — divide `groups` by `Gm` on a copy
  of the frozen `ConvProblem`; since `cpg`/`kpg` are *derived*, every downstream
  consumer automatically sees `Gm`-wide channel runs, and nothing else moves;
- masking done as an explicit **diagonal store predicate** in the epilogue
  (`_gm_dw_addr_fn`, `:935-967`) rather than CK's descriptor xor:

  ```python
  diag = b_.cmp_eq(b_.xor(gm_m, gm_n), c_zero)
  return off, (b_.land(valid, diag) if valid is not None else diag)
  ```

  The comment argues this is strictly cheaper, since these epilogues already
  guard the store with a predicate that an extra term folds into, whereas the
  descriptor form costs an extra magic-division scan.
- `_GROUP_MERGE_DEGREES = (2, 4, 8, 16, 32, 64)` (`:897`) — same power-of-two
  set as CK, for the same reason.

**What store width wgrad's merged path actually achieves: 1.** The gate
(`wgrad_group_merge_available`, `:993-998`) requires `cpg == kpg == 1`, so
`default_vector_sizes(1, 1, dtype)` returns `vec_c = 1`. The comment at
`_emit_wgrad_cshuffle_epilogue` `:2800-2806` says so outright — *"the store stays
scalar — cpg == 1 under the group-merge gate, so the vector width derived below
is 1"*. Whether the config routes through `_emit_wgrad_direct_epilogue`,
`_emit_wgrad_cshuffle_epilogue`, or the two-stage `_emit_wgrad_workspace_store_epilogue`
(which has no LDS stage at all), a merged wgrad kernel stores **one element per
valid lane**. ROCKE's existing merged path takes the load-side win and leaves the
store scalar — which, per §11, is exactly what CK's `Gm = 32` instance does too.

**The one-line lever for the fwd port.** On the forward path,
`ImplicitGemmConvSpec.default_vector_sizes(C, K, dtype)` (`conv_implicit_gemm.py:351-366`)
returns `vec_a = vec_b = _vec(cpg)` and `vec_c = _vec(kpg)`. For depthwise
`cpg == kpg == 1`, so all three are 1. Feeding it `merged_problem` instead of
`problem` makes `cpg == Gm`, so `vec_a = vec_b = _vec(Gm)` — up to 8. That is
precisely CK's A-side win, obtained through the same `merged_problem`
substitution wgrad already uses, with `vec_c` pinned to 1 and the diagonal test
folded into the epilogue predicate.

The forward port reuses this machinery on `conv_implicit_gemm.py`, but **it did
not end up a transplant**: fwd's stride-1 axis already sits inside the reduction
axis, so it puts `Gm` on GemmN and GemmK rather than CK/wgrad's GemmM and GemmN,
and spends the diagonal on the **B load** rather than the C store. The store is
therefore dense and `vec_c` is free to widen — see
`examples/gfx950/conv_fwd/fwd_merged_groups_case_study.md`.

Engine coverage differs by direction. **Forward** is implemented in both the
Python and the C++ engine (`group_merge` is a field on
`rocke_implicit_gemm_conv_spec`) and the fwd parity emitters build merged
configs, so byte-identity gates merged emission directly. **wgrad** is
Python-engine only and its knob is additive at its default of 1 — which is how
it keeps the gate green without merged coverage. That wgrad gap predates the
forward port.

---

## 13. Summary of claims to check

1. Merged groups = fold `Gm` groups into one tile; group index becomes the
   fastest factor of **both** GemmM and GemmN.
2. Weights are dense and real; redundancy is discarded on the **output** side.
3. Correctness is the `pad(1, 0, Gm-1)` + `xor<ApplyModulo=false>` pair, which
   sends off-diagonal results to an invalid coordinate so the store is
   predicated off. No scratch buffer; nothing extra is written.
4. `ApplyModulo = false` ⇒ `Gm` must be a power of two ≤ 64.
5. The padded axis has stride `GStrideTensorC_` (not 0), so the pad validity
   flag is load-bearing.
6. It only pays off for depthwise because `C_ == 1` makes the group stride 1 in
   NHWGC.
7. Against the *padded* unmerged GEMM, merging is FLOP-neutral for
   `Gm ≤ NPerBlock`; useful-FLOP efficiency is `K_·Gm/NPerBlock`. **(my
   derivation — not load-bearing, see §8)**
8. Forward only; no split-K; shipped via dispatcher codegen; untested in-tree.
9. The payoff is the **A-side load vector** (`VectorSizeA = min(Gm, cap)`),
   because the merged `Gm` axis has stride 1 and A becomes ColumnMajor. The C
   store is a per-instance tuning choice — the two shipped merged fwd instances
   use `VectorSizeC` of 1 and 4 respectively. (§11)
10. CK Tile grouped conv fwd uses `CShuffleEpilogue` **unconditionally**, at
    `Gm == 1` too. It is the baseline epilogue for all grouped conv, not
    something merged groups enables. (§12.1)
11. ROCKE's `_emit_cshuffle_epilogue` is the structural equivalent; the only
    real difference is that CK puts the diagonal mask in the C descriptor while
    ROCKE puts it in the `addr_fn` validity predicate. (§12.1)
12. ROCKE's merged wgrad path stores **one element per valid lane** on every
    route, and takes its win entirely on the load side. (§12.2)
