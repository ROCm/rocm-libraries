# Grouped Convolution - No "two_stage" Variant Verification

## Summary

✅ **VERIFIED:** There is **NO "two_stage" variant** in grouped convolution kernels.

`two_stage` is a **configuration option** for the `bwd_weight` variant, NOT a separate variant.

**Date:** 2026-04-26

## Supported Variants (Only 3)

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

**No "two_stage" variant exists.**

## What is two_stage?

### Configuration Option (NOT a Variant)

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
    split_image: bool = False
    explicit_gemm: bool = False
    two_stage: bool = False          # ← Config option, NOT a variant
```

### Purpose of two_stage

**What it does:**
- **Only applies to `bwd_weight` variant**
- Splits backward weight computation into two stages:
  1. **Stage 1:** GEMM computation with fp32 accumulation → fp32 workspace
  2. **Stage 2:** ElementWise conversion from fp32 workspace → output dtype (bf16/fp16)
- Improves numerical accuracy for weight gradients
- Uses temporary workspace (higher memory cost, better precision)

**What it is NOT:**
- ❌ NOT a separate kernel variant (like forward/bwd_data/bwd_weight)
- ❌ NOT applicable to forward or bwd_data
- ❌ NOT selectable via `--variant` parameter

## How two_stage is Used

### 1. Kernel Generation Logic

**Location:** Lines 1090-1107

```python
if variant == GroupedConvVariant.BACKWARD_WEIGHT:
    tile_configs = bwd_weight_tiles
    pipelines = [("compv3", "cshuffle"), ("mem", "default")]
    # Also generate two-stage variants (fp32 workspace + elementwise convert)
    two_stage_flags = [False, True]  # ← Generate BOTH regular and two-stage
elif variant == GroupedConvVariant.BACKWARD_DATA:
    tile_configs = fwd_bwd_data_tiles
    pipelines = [("compv3", "cshuffle"), ("mem", "default")]
    two_stage_flags = [False]  # ← bwd_data: NO two-stage
else:  # FORWARD
    tile_configs = fwd_bwd_data_tiles
    pipelines = [("compv3", "cshuffle"), ("compv4", "cshuffle")]
    two_stage_flags = [False]  # ← forward: NO two-stage
```

**Result:** For each `bwd_weight` kernel configuration, TWO kernels are generated:
- One with `two_stage=False` (regular GEMM)
- One with `two_stage=True` (two-stage with fp32 workspace)

### 2. Kernel Implementation Selection

**Location:** Lines 474-475

```python
if self.variant == GroupedConvVariant.BACKWARD_WEIGHT and tr.two_stage:
    return self._kernel_instance_two_stage(config, kernel_name)
```

**Behavior:**
- If variant is `bwd_weight` AND `two_stage=True` → Use two-stage implementation
- Otherwise → Use standard single-stage implementation

### 3. Two-Stage Implementation

**Location:** Lines 713-762 (`_kernel_instance_two_stage`)

```cpp
// Generated two-stage kernel structure:

struct {kernel_name}_Launcher {
    using WorkspaceDataType = float;  // ← fp32 workspace

    // Two-stage forces VectorSizeC = 1 for workspace writes
    static constexpr index_t VectorSizeC_TwoStage = 1;

    // Stage 1: GEMM into fp32 workspace
    using GemmKernel = GroupedConvolutionBackwardWeightKernel<...>;

    // Stage 2: ElementWise fp32 → output dtype
    using ConvertKernel = ElementWiseKernel<...>;

    // Launch both stages
    // ...
};
```

**Implementation details:**
- Uses fp32 workspace for accumulation (higher precision)
- VectorSizeC forced to 1 for workspace writes
- Two kernel launches per invocation (GEMM + Convert)
- Higher memory usage but better numerical accuracy

## Usage Examples

### Correct Usage (Configuration Option)

```python
from grouped_conv_utils import GroupedConvKernelConfig, GroupedConvTraitConfig

# Regular bwd_weight kernel
config_regular = GroupedConvKernelConfig(
    variant="bwd_weight",              # ← This is the variant
    trait=GroupedConvTraitConfig(
        pipeline="compv3",
        two_stage=False,              # ← Regular single-stage
    ),
    # ...
)

# Two-stage bwd_weight kernel
config_two_stage = GroupedConvKernelConfig(
    variant="bwd_weight",              # ← Same variant
    trait=GroupedConvTraitConfig(
        pipeline="compv3",
        two_stage=True,               # ← Two-stage with fp32 workspace
    ),
    # ...
)
```

### Invalid Usage (Trying to Use as Variant)

```python
# ❌ WRONG - "two_stage" is not a valid variant
config = GroupedConvKernelConfig(
    variant="two_stage",  # ← ERROR: Not a valid variant
)

# ❌ WRONG - two_stage on forward (not supported)
config = GroupedConvKernelConfig(
    variant="forward",
    trait=GroupedConvTraitConfig(
        two_stage=True,   # ← Ignored! Only works with bwd_weight
    ),
)

# ❌ WRONG - two_stage on bwd_data (not supported)
config = GroupedConvKernelConfig(
    variant="bwd_data",
    trait=GroupedConvTraitConfig(
        two_stage=True,   # ← Ignored! Only works with bwd_weight
    ),
)
```

## Kernel Naming Convention

Two-stage kernels are named differently to distinguish them:

**Regular bwd_weight:**
```
grouped_conv_bwd_weight_bf16_2d_64x64x64_compv3
```

**Two-stage bwd_weight:**
```
grouped_conv_bwd_weight_bf16_2d_64x64x64_compv3_two_stage
```

The `_two_stage` suffix indicates it's a two-stage implementation.

## When to Use two_stage?

### Advantages
✅ **Better numerical accuracy** - fp32 accumulation reduces rounding errors
✅ **Suitable for training** - Weight gradients benefit from higher precision
✅ **Already generated** - Both variants available for bwd_weight

### Disadvantages
❌ **Higher memory usage** - Requires fp32 workspace allocation
❌ **Two kernel launches** - Extra overhead from second kernel call
❌ **Slightly slower** - Due to workspace writes and conversion

### Recommendation
- **Use two_stage=True:** When numerical accuracy is critical (training)
- **Use two_stage=False:** When speed is critical and precision is acceptable (inference)

## Comparison with Other Config Options

| Name | Type | Applies To | Purpose |
|------|------|------------|---------|
| `forward` | **Variant** | N/A | Forward convolution |
| `bwd_data` | **Variant** | N/A | Backward data gradient |
| `bwd_weight` | **Variant** | N/A | Backward weight gradient |
| `two_stage` | Config option | `bwd_weight` only | Two-stage with fp32 workspace |
| `split_image` | Config option | All variants | Split spatial dims for large tensors |
| `double_smem_buffer` | Config option | All variants | Ping-pong LDS buffering |
| `num_groups_to_merge` | Config option | All variants | Fuse groups into tiles |
| `explicit_gemm` | Config option | All variants | Use explicit GEMM path |

## Verification Summary

| Aspect | Status |
|--------|--------|
| Is "two_stage" a variant? | ❌ **NO** |
| Is "two_stage" a config option? | ✅ **YES** |
| Works with forward? | ❌ NO |
| Works with bwd_data? | ❌ NO |
| Works with bwd_weight? | ✅ **YES** |
| Generates separate kernels? | ✅ YES (both regular and two-stage) |
| User-selectable? | ✅ YES (via trait config) |

## Code Flow Summary

```
User Request: bwd_weight kernel
                    ↓
        Codegen generates TWO kernels:
                    ↓
    ┌───────────────┴────────────────┐
    ↓                                ↓
two_stage=False              two_stage=True
(Regular GEMM)              (Two-stage fp32)
    ↓                                ↓
Single kernel launch         Two kernel launches:
GEMM → output               1. GEMM → fp32 workspace
                            2. Convert → output
```

## VARIANT_PIPELINES Compatibility

**Current configuration (grouped_config_rules.py):**

```python
VARIANT_PIPELINES: Dict[str, List[str]] = {
    "forward": [...],      # ✅ two_stage NOT applicable
    "bwd_data": [...],     # ✅ two_stage NOT applicable
    "bwd_weight": [...],   # ✅ two_stage=True/False both generated
}
```

**For bwd_weight:** Both regular and two-stage kernels are generated automatically when building. User doesn't need to specify anything special - both are available.

## Conclusion

✅ **CONFIRMED: NO "two_stage" variant**

**Summary:**
1. Only 3 variants exist: `forward`, `bwd_data`, `bwd_weight`
2. `two_stage` is a **boolean configuration option** for `bwd_weight` only
3. Enables two-stage computation: GEMM (fp32) → Convert (output dtype)
4. Improves numerical accuracy at the cost of memory and performance
5. Both `two_stage=True` and `two_stage=False` kernels are generated for `bwd_weight`
6. NOT applicable to `forward` or `bwd_data` variants

**If you need two-stage bwd_weight:**
- It's already generated! Just select the kernel with `_two_stage` suffix
- Or use `trait=GroupedConvTraitConfig(two_stage=True)` when creating configs
- No need for a separate "two_stage" variant

## Code References

- **Variant enum:** `unified_grouped_conv_codegen.py` lines 70-75
- **two_stage config:** `unified_grouped_conv_codegen.py` line 118
- **Generation logic:** `unified_grouped_conv_codegen.py` lines 1090-1107
- **Implementation selector:** `unified_grouped_conv_codegen.py` lines 474-475
- **Two-stage implementation:** `unified_grouped_conv_codegen.py` lines 713-762

## Audit Trail

- **File:** `dispatcher/codegen/unified_grouped_conv_codegen.py`
- **Lines reviewed:** 70-75, 100-118, 474-475, 713-762, 1090-1107
- **Date:** 2026-04-26
- **Reviewer:** Claude (Sonnet 4.5)
- **Conclusion:** ✅ NO "two_stage" variant - Only a config option for bwd_weight
