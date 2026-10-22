# MMA catalog (`data/mma_traits.json`)

One row per AMD matrix (MMA) instruction variant: one clang builtin plus the operand choices that
pick the variant (for example the A and B formats of an `f8f6f4` instruction). The tiling layer reads
it through `load_mma_traits()` (`mma_traits.py`); `TileMma` selects its atom from it and the encoder
(`mma/warp_encoding.py`) lays out A, B and C from a row's layout parameters.

The catalog is maintained by hand. The loader is strict: any data error fails the load with a
message naming the row and the field, so an edit cannot load half-right.

## Schema (`schema_version: 2`)

```json
{"_meta": {"schema_version": 2},
 "operations": [
  {"op_id": "mfma_f32_16x16x16f16",
   "llvm_builtin": "__builtin_amdgcn_mfma_f32_16x16x16f16",
   "family": "dense", "wave_size": 64,
   "supported_targets": ["gfx908", "gfx90a", "gfx942", "gfx950"],
   "a_dtype": "fp16", "b_dtype": "fp16", "c_dtype": "fp32", "d_dtype": "fp32",
   "auto_select": true,
   "dims": {"M": 16, "N": 16, "K": 16, "B": 1, "R": 1, "S": 1},
   "layout_params": {"ABK": 4, "AKN": 1, "AR": 1, "BKN": 1, "BR": 1, "CM": 4, "CMN": 1, "AMB": 1},
   "layout_source": "calculator",
   "a_layout": "K{4} L{K1M} V{K0}", "b_layout": "same", "c_d_layout": "M{4} L{M1N} V{M0}",
   "scale": null, "index_layout": null}]}
```

A row carries exactly these fields (plus `operand_signedness` on integer rows):

| Field | Rule |
|---|---|
| `op_id` | Unique. The builtin name without `__builtin_amdgcn_`. A builtin whose A and B formats are picked by an operand has one row per format pair, named `<stem>_<a_dtype>_<b_dtype>` (e.g. `wmma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1`). |
| `llvm_builtin` | The clang builtin. |
| `family` | `dense`, `sparse` or `scaled`. |
| `wave_size` | 32 or 64. |
| `supported_targets` | The gfx targets the row is listed on. |
| `a_dtype`, `b_dtype` | A and B operand dtypes. |
| `c_dtype`, `d_dtype` | The accumulator taken in (C) and the result returned (D). They differ only where the instruction returns another dtype than it accumulates in. |
| `operand_signedness` | Integer rows only (`i8` / `i4` operands), and required there: `selectable` when each operand has its own signedness flag (integer WMMA), so `u8` / `u4` also match, independently for A and B; `signed` for signed-only operands (integer MFMA). |
| `auto_select` | `false` marks a pin-only row (see Selection). |
| `dims` | Positive integers. `M`, `N`, `K`: the shape of one block. `B`: independent blocks per instruction. `R`: operand replication (equals `AR` and `BR`). `S`: as named by the layout strings, else 1. |
| `layout_params` | Positive integers, in logical elements per lane (glossary below). |
| `layout_source` | The evidence for the layout values (see Layout-exactness rule): `calculator` (the public AMD Matrix Instruction Calculator), `isa_doc` (the public CDNA4 ISA tables), `rocke_core` (rocke core's lane maps), `ck` (Composable Kernel's parameters) or `legacy_table` (the values carried over from the earlier generated table). |
| `a_layout`, `b_layout`, `c_d_layout` | The layouts in hardware notation (`K{4} L{K1M} V{K0}`, see Layout strings). `b_layout: "same"` means A's string over N. They state the same layout as `layout_params`, element for element. |
| `scale` | `null` unless `family` is `scaled`; then `{"block_k": 32, "a_dtypes": ["e8m0"], "b_dtypes": ["e8m0"]}`: the K elements sharing one scale (16 or 32), and the scale formats the instruction accepts for A and B (`core.arch` `MmaScaleDType` values). A scale format changes how a scale byte is decoded, never a register layout, so the accepted formats are a set on one row. |
| `index_layout` | Sparse rows: the sparsity index operand's layout string, else `null`. |

### Dtype names

Every dtype is a canonical `core.dtypes` name: `normalize_dtype(x) == x` (`fp16`, `bf16`, `fp32`,
`tf32`, `fp64`, `fp8e4m3`, `bf8e5m2`, `fp6e2m3`, `fp6e3m2`, `fp4e2m1`, `i8`, `i4`, `i32`). `fp8e4m3`
and `bf8e5m2` mean the target's native fp8 dialect; that fact lives in
`dsl_docs/reference/dtype_support.md` section 4E. An operand's dtypes are read from the instruction
name (`mfma_f32_16x16x32_fp8_bf8` is `fp8e4m3` x `bf8e5m2`).

### Layout parameters

CK's seven trait parameters plus one rocke parameter. They describe logical elements per lane, the
map `IRBuilder.mma` takes; how a value is held in registers below that (sub-byte packing, the
even-half placement of gfx11 16-bit results) is core's business.

| Code | Field | CK trait | Meaning |
|---|---|---|---|
| `ABK` | `k_ab_per_lane` | `kABKPerLane` | K elements of A (and B) one lane holds |
| `AKN` | `a_k_num_access` | `kAKNumAccess` | separate K runs those A elements form |
| `AR` | `a_repeat` | `kARepeat` | A replication across lanes |
| `BKN` | `b_k_num_access` | `kBKNumAccess` | separate K runs of B |
| `BR` | `b_repeat` | `kBRepeat` | B replication across lanes |
| `CM` | `c_m_per_lane` | `kCMPerLane` | M elements of C one lane holds |
| `CMN` | `c_m_num_access` | `kCMNumAccess` | separate M runs those C elements form |
| `AMB` | `a_m_register_blocks` | (rocke) | M blocks one lane holds in its A registers |

`AMB` is 1 except on the three `32x16x128_f4` rows, whose A operand holds its 32 rows as two
blocks of 16: the lanes span 16 rows and the second block sits in the most significant registers.
The encoder lays out every row with one block (`B == 1`); for a multi-block row it raises
`NotImplementedError` naming the missing feature (`mma/warp_encoding.encoder_supports_layout`).
`InterleavedStyle` does not model `AMB > 1` and refuses such a row by name; the canonical style
lays it out.

### Layout strings

A layout string says which element each (lane, register slot) of one wave holds, over logical
elements:

- `X{s0}` or `X{s1, s0}` splits the dim `X` (`M`, `N`, `K`, or `KP` for a sparse A operand's
  compressed K) into sub-dims: `X0` of size `s0` (fastest), `X1` of size `s1`, and a last sub-dim
  holding the rest of `X`. The element's coordinate is `X = X0 + s0 * (X1 + s1 * X2)`.
- `L{...}` lists the dims and sub-dims that form the lane index, most significant first.
- `V{...}` lists those that form the register slot index, most significant first.
- A bare name is a whole dim. `B` is the block index of a multi-block instruction. `R` is a
  replication index: every replica holds the same element. A dim of size 1 that the string does not
  name is at coordinate 0.

Example: `K{2, 4} L{K1M} V{K2K0}` (`wmma_f32_16x16x16_f16_w32_gfx12`, `K = 16`) splits K into `K0`
(4 values), `K1` (2) and `K2` (2). Lane `16 * K1 + M` holds slot `4 * K2 + K0`, so lane 0 holds K
0-3 and 8-11 of row 0, and lane 16 holds K 4-7 and 12-15. `index_layout` uses the same notation (`S`
there indexes the sparsity selector).

## Layout-exactness rule

The layout parameters and strings reproduce the hardware register layout exactly wherever an
independent arbiter establishes it: the public AMD Matrix Instruction Calculator, the public CDNA4
ISA tables, or rocke core's lane maps (including its gfx1250 scaled operand layout). Where none
exists, they take Composable Kernel's (CK develop) parameters, else the earlier table's strings. No
row stores another K order than that, not even one shared by A and B, and the strings equal the
encoder's layout element for element (`tests/helpers/tiling/traits/test_mma_catalog_layouts.py`).

## Differences from CK

- gfx12 wave32 16-bit rows (`wmma_{f32,f16}_16x16x16_f16_w32_gfx12`, their `bf16` twins, and the
  four `swmmac_*_16x16x32_{f16,bf16}_w32` rows): `AKN = BKN = 2`, the calculator's two K runs per
  lane. CK's 1 is a K order shared by A and B.
- Equal-width `f8f6f4` rows (both operands 8-bit, or both 6- or 4-bit): per-format K runs on each
  operand, as on the mixed-width rows (gfx950: 8-bit 2, 6/4-bit 1, from the CDNA4 ISA tables;
  gfx1250: 8-bit 4, 6/4-bit 2, from rocke core's scaled layout). CK uses 1 and 1 there, which holds
  only while no scales are used.
- gfx950 16-bit and 8-bit sparse rows (`smfmac_*` on gfx950): `AKN = 2`, the two K runs per lane
  that B also holds, so A and B pair the same K. CK's `AKN = 1` does not match its own `BKN = 2`.

## Unverified layouts

These rows have no independent arbiter (`layout_source` `ck` or `legacy_table`). Their layout does
not decide whether rocke issues them: a row the backend issues is issued. They count as
compile-verified only, until a hardware lane-map run confirms them.

- The earlier table's strings, sparse gfx950 (A and its index operand in two K runs per lane, as
  B): `smfmac_f32_16x16x64_f16`, `smfmac_f32_32x32x32_f16`,
  `smfmac_f32_16x16x64_bf16`, `smfmac_f32_32x32x32_bf16`, `smfmac_i32_16x16x128_i8`,
  `smfmac_i32_32x32x64_i8`, `smfmac_f32_16x16x128_bf8_bf8`, `smfmac_f32_16x16x128_bf8_fp8`,
  `smfmac_f32_16x16x128_fp8_bf8`, `smfmac_f32_16x16x128_fp8_fp8`, `smfmac_f32_32x32x64_bf8_bf8`,
  `smfmac_f32_32x32x64_bf8_fp8`, `smfmac_f32_32x32x64_fp8_bf8`, `smfmac_f32_32x32x64_fp8_fp8`.
- CK's parameters, gfx1250: `wmma_f32_16x16x4_f32` (on CK's lane map) and `wmma_bf16_16x16x32_bf16`,
  `wmma_bf16f32_16x16x32_bf16`, `wmma_f32_16x16x64_fp8_fp8`, `wmma_f32_16x16x64_fp8_bf8`,
  `wmma_f32_16x16x64_bf8_fp8`, `wmma_f32_16x16x64_bf8_bf8`, `wmma_f16_16x16x64_fp8_fp8`,
  `wmma_f16_16x16x64_fp8_bf8`, `wmma_f16_16x16x64_bf8_fp8`, `wmma_f16_16x16x64_bf8_bf8`,
  `wmma_i32_16x16x64_iu8`, `wmma_f16_16x16x128_fp8_fp8`, `wmma_f16_16x16x128_fp8_bf8`,
  `wmma_f16_16x16x128_bf8_fp8`, `wmma_f16_16x16x128_bf8_bf8`, `wmma_f32_16x16x128_fp8_fp8`,
  `wmma_f32_16x16x128_fp8_bf8`, `wmma_f32_16x16x128_bf8_fp8`, `wmma_f32_16x16x128_bf8_bf8`,
  `wmma_f16_16x16x32_f16`. CK lists no `wmma_bf16f32_16x16x32_bf16`; it takes the parameters of its
  bf16 x bf16 16x16x32 siblings.
- The earlier table's strings, sparse gfx1250 (CK has no such rows): `swmmac_f32_16x16x64_bf16`,
  `swmmac_bf16_16x16x64_bf16`, `swmmac_bf16f32_16x16x64_bf16`, `swmmac_f32_16x16x128_fp8_fp8`,
  `swmmac_f32_16x16x128_fp8_bf8`, `swmmac_f32_16x16x128_bf8_fp8`, `swmmac_f32_16x16x128_bf8_bf8`,
  `swmmac_f16_16x16x128_fp8_fp8`, `swmmac_f16_16x16x128_fp8_bf8`, `swmmac_f16_16x16x128_bf8_fp8`,
  `swmmac_f16_16x16x128_bf8_bf8`, `swmmac_i32_16x16x128_iu8`, `swmmac_f32_16x16x64_f16`,
  `swmmac_f16_16x16x64_f16`.
- The earlier table's strings, two M blocks of A per lane: `wmma_f32_32x16x128_f4`,
  `wmma_scale_f32_32x16x128_f4`, `wmma_scale16_f32_32x16x128_f4`.

## Selection

`load_mma_traits().select(target=, a=, b=, c=, m=, n=, k=, d=None, family="dense",
scale_block_k=None)` returns the one row for an intent. Inputs are canonical dtype names. `d=None`
means the result has the accumulator's dtype. `scale_block_k` is required for family `scaled` and
refused otherwise: block-16 and block-32 scaled instructions decode scales differently, so a kernel
names the one it holds.

A row is selectable on a target when it is listed on the target, is of the asked family, has one
block (`B == 1`) and has `auto_select: true`. The selection key is (target, family, A, B, C, D, M,
N, K, and the scale block size for `scaled`). The loader refuses two selectable rows with the same
key. When two instructions really answer the same key (a `_tied` variant, an `f8f6f4` 8-bit x 8-bit
variant beside the dedicated fp8 instruction), keep the dedicated one selectable and set
`auto_select: false` on the other: it stays reachable by name (`atom_override`), never by selection.
Multi-block rows are never selected; they are reachable by name.

`TileMma` drives only rows whose result dtype is the accumulator dtype (`d_dtype == c_dtype`): its
driver keeps the result in the accumulator. It refuses any other row on every path, by intent, by
name and by `atom_override` (today only `wmma_bf16f32_16x16x32_bf16`). `select(d=...)` still returns
such a row to a caller that asks for it.

A pin-only row lays out A, B and C exactly as its selectable twin does, with one exception: the
gfx1250 `f8f6f4` 8-bit x 8-bit rows are another instruction than the dedicated fp8 rows beside
them. Their K order is rocke core's scaled layout, the arbiter for the `f8f6f4` instruction; the
dedicated instructions have no arbiter and keep CK's order (Unverified layouts). The two orders
differ by one K permutation shared by A and B, so the same products are formed, and C is laid out
identically.

## Editing the catalog

1. Edit the row by hand; keep every field the schema lists, and set `layout_source` to where the new
   values come from.
2. Keep the layout strings and `layout_params` in agreement, following the layout-exactness rule; a
   row whose layout has no independent arbiter goes into Unverified layouts.
3. Run the tiling tests: `python -m pytest python/rocke/tests/helpers/tiling` (from `platform/`).
   `traits/test_mma_catalog_integrity.py` checks the invariants above,
   `traits/test_mma_catalog_layouts.py` checks every row's strings against the encoder and the
   arbiters, and the loader names any data error.

The catalog has no C++ counterpart: the tiling layer is Python only, and the C++ engine does not
read this file.
