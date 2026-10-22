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
   "layout_source": "legacy_table",
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
| `layout_source` | Where the layout values come from: `calculator`, `isa_doc`, `rocke_core`, `ck` or `legacy_table` (the values carried over from the earlier generated table). |
| `a_layout`, `b_layout`, `c_d_layout` | The layouts in hardware notation (`K{4} L{K1M} V{K0}`). `b_layout: "same"` means A's string over N. |
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

The encoder lays out a row only if it has one block (`B == 1`) and one A M register block
(`AMB == 1`); for any other row it raises `NotImplementedError` naming the missing feature
(`mma/warp_encoding.encoder_supports_layout`).

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

## Editing the catalog

1. Edit the row by hand; keep every field the schema lists, and set `layout_source` to where the new
   values come from.
2. Run the tiling tests: `python -m pytest python/rocke/tests/helpers/tiling` (from `platform/`).
   `traits/test_mma_catalog_integrity.py` checks the invariants above; the loader names any data error.

The catalog has no C++ counterpart: the tiling layer is Python only, and the C++ engine does not
read this file.
