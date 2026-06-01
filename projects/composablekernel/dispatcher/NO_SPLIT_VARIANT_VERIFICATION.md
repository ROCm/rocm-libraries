# Grouped Convolution - No "Split" Variant Verification

## Summary

✅ **VERIFIED:** There is **NO "split" variant** in grouped convolution kernels.

**Date:** 2026-04-26

## Supported Variants

### Only 3 Variants Exist

**Location:** `unified_grouped_conv_codegen.py` lines 70-75

```python
class GroupedConvVariant(Enum):
    """Grouped convolution kernel variants"""

    FORWARD = "forward"
    BACKWARD_DATA = "bwd_data"
    BACKWARD_WEIGHT = "bwd_weight"
```

**These are the ONLY variants:**
1. `forward` - Forward convolution
2. `bwd_data` - Backward data (gradient w.r.t. input)
3. `bwd_weight` - Backward weight (gradient w.r.t. weights)

**No "split" variant exists.**

## What "Split" Means in Grouped Conv

### split_image - Configuration Option (NOT a variant)

**Location:** `unified_grouped_conv_codegen.py` lines 100-118

```python
@dataclass
class GroupedConvTraitConfig(TraitConfigBase):
    """
    Conv-specific extensions beyond TraitConfigBase. These map to
    GroupedConvTraits template parameters in grouped_convolution_utils.hpp:
    - double_smem_buffer: ping-pong LDS for compute V4+ pipelines
    - num_groups_to_merge: fuse multiple groups into one tile (NumGroupsToMerge)
    - split_image: split spatial dims for large tensors (EnableSplitImage)
    - explicit_gemm: use explicit GEMM path (ExplicitGemm)
    - two_stage: two-stage bwd_weight with fp32 workspace + elementwise convert

    Note: CK Tile already uses long_index_t (64-bit) for group strides and
    batch offsets, so there is no separate "large_tensor" flag. For large
    spatial dimensions, use split_image=True instead.
    """

    double_smem_buffer: bool = False
    num_groups_to_merge: int = 1
    split_image: bool = False          # ← This is a CONFIG option, NOT a variant
    explicit_gemm: bool = False
    two_stage: bool = False
```

### Purpose of split_image

**What it does:**
- **Configuration flag** for handling large spatial dimensions
- Splits spatial dimensions into smaller chunks when enabled
- Used for very large tensors that might exceed memory or grid limits
- Applied to the C++ template as `EnableSplitImage` parameter

**What it is NOT:**
- ❌ NOT a separate kernel variant (like forward/bwd_data/bwd_weight)
- ❌ NOT a different algorithm or operation type
- ❌ NOT selectable via `--variant` parameter

### Usage

**Correct usage:**
```python
# split_image is a trait configuration option
config = GroupedConvKernelConfig(
    variant="forward",              # ← This is the variant
    trait=GroupedConvTraitConfig(
        split_image=True,          # ← This is a config option
        pipeline="compv4",
    ),
    # ...
)
```

**NOT a variant:**
```python
# ❌ WRONG - "split" is not a valid variant
config = GroupedConvKernelConfig(
    variant="split",  # ← ERROR: Not a valid variant
)
```

## Comparison with GEMM

### GEMM Has More Variants

For reference, GEMM operations have additional variants like:
- `gemm`
- `gemm_splitk` - Split-K GEMM variant
- `batched_gemm`
- `grouped_gemm`

**But grouped convolution does NOT have equivalent split variants.**

## split_k in Generated Code

**Location:** Code generation snippets in `unified_grouped_conv_codegen.py`

```cpp
// Lines ~570-575 (in generated code)
const index_t K_split = (gemm_k + k_grain - 1) / k_grain * Config::K_Tile;
const index_t num_loop = TilePartitioner::GetLoopNum(K_split);
```

**This is:**
- ✅ Internal implementation detail for K-dimension tiling
- ✅ Part of the GEMM transformation (convolution → GEMM)
- ❌ NOT exposed as a separate variant
- ❌ NOT user-configurable

## Verification Summary

| Concept | Type | Available? |
|---------|------|-----------|
| `forward` | Variant | ✅ YES |
| `bwd_data` | Variant | ✅ YES |
| `bwd_weight` | Variant | ✅ YES |
| `split` | Variant | ❌ NO - Does not exist |
| `split_image` | Config option | ✅ YES - For large tensors |
| `split_k` | Internal detail | N/A - Implementation only |

## Configuration Files

### Current VARIANT_PIPELINES (grouped_config_rules.py)

```python
VARIANT_PIPELINES: Dict[str, List[str]] = {
    "forward": [...],      # ✅ Valid
    "bwd_data": [...],     # ✅ Valid
    "bwd_weight": [...],   # ✅ Valid
    # "split": [...],      # ❌ Does not exist
}
```

### Valid Variant Names

**Accepted strings:**
- `"forward"` → `GroupedConvVariant.FORWARD`
- `"bwd_data"` → `GroupedConvVariant.BACKWARD_DATA`
- `"bwd_weight"` → `GroupedConvVariant.BACKWARD_WEIGHT`

**NOT accepted:**
- ❌ `"split"` - Does not map to any GroupedConvVariant
- ❌ `"split_image"` - This is a trait config, not a variant
- ❌ `"splitk"` or `"split_k"` - Not a grouped conv variant

## Conclusion

✅ **CONFIRMED: NO "split" variant in grouped convolution**

**Summary:**
1. Only 3 variants exist: `forward`, `bwd_data`, `bwd_weight`
2. `split_image` is a **configuration option**, not a variant
3. `split_k` is an **internal implementation detail**, not exposed
4. No plans to add "split" variant (convolution doesn't need it like GEMM does)

**If you need large tensor support:**
- Use `split_image=True` in trait configuration
- This works with all 3 existing variants
- No separate "split" variant is needed

## Code References

- **Variant enum:** `unified_grouped_conv_codegen.py` lines 70-75
- **split_image config:** `unified_grouped_conv_codegen.py` lines 100-118
- **Variant mapping:** `unified_grouped_conv_codegen.py` lines 1696-1698
- **Variant validation:** Uses enum, invalid values rejected by Python type system

## Audit Trail

- **File:** `dispatcher/codegen/unified_grouped_conv_codegen.py`
- **Lines reviewed:** 70-75, 100-118, 1696-1698
- **Date:** 2026-04-26
- **Reviewer:** Claude (Sonnet 4.5)
- **Conclusion:** ✅ NO "split" variant - Only forward/bwd_data/bwd_weight exist
