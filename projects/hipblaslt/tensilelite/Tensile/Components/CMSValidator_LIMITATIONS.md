# CMSValidator Known Limitations

## Unsupported Configurations

### DirectToLds = 0
- **Status**: Not supported
- **Error**: `assert kernel["DirectToLds"]` in Timeline constructor
- **Impact**: gfx1151 (RDNA 3.5) schedules currently disable all timeline passes

### LocalSplitU > 1
- **Status**: Not supported
- **Error**: `assert kernel.get("LocalSplitU", 1) == 1` in Timeline constructor

### UseF32XEmulation = True with UseDirect32XEmulation = False
- **Status**: Not supported
- **Error**: Raises `ValueError` in `_hook_up_packs_f32` and `hook_up_packs`
- **Workaround**: Set `UseDirect32XEmulation = True` when using TF32 emulation

### ForceUnrollSubIter with Register Reuse
- **Status**: Partially supported
- **Issue**: Does not fully account for VGPR reuse patterns in LR0/LR1/LR3 scheduling
- **Location**: `CMSValidator.py` line 611

## Validation Gaps

### False Negatives (schedule appears valid but may fail)
1. **Quad-cycle estimation is conservative**: `estimate_quad_cycles` does not model all pipeline stalls (only MFMA execution latency and type-switch penalties). Real cycle counts may be higher, making some schedules pass validation that would violate timing in practice.
2. **SBarrier timing assumes instant synchronization**: The validator checks that an `s_barrier` exists between the required instructions but does not model barrier latency.
3. **SWaitCnt counter values are not verified**: The validator trusts that `dscnt`/`vlcnt`/`vscnt` values in `SWaitCnt` instructions are correct — it does not count actual outstanding memory operations to verify.

### False Positives (schedule appears invalid but works)
1. None currently known.

## Architecture-Specific Behavior

### CDNA 4 (gfx950)
- Full timeline validation with quad-cycle timing from ISA section 7.6
- 4x4 MFMA TF32 emulation path validated
- All 7 validator passes enabled

### RDNA 3.5 (gfx1151)
- Only `VERIFY_ASCENDING_ORDER` is enabled
- All timeline passes disabled via `_disable_cdna4_only_passes_for_gfx1151`
- `VERIFY_CORRECT_NUMBER_OF_INSTRUCTIONS` disabled: CMS instruction counts are calibrated for wave64, don't match wave32 kernel expansion
- `VERIFY_SCC_OVERLAP` disabled: SCC constraint shape not audited for RDNA 3.5 wave32 scalar semantics
- **Silence means "unchecked," not "correct"**

### Older CDNA (gfx940, gfx941, gfx942, gfx90a)
- No CMS schedules registered for these ISAs
- `hasCustomSchedule` returns False

## idMap Limitations

- `idMap` is only used for instruction count verification (`verify_correct_number_of_instructions`)
- When `idMap` is not provided (unit tests), instruction count check is silently skipped
- The validator does not inspect actual rocisa instruction types or register operands from `idMap`

## Ascending Order Exception

- Pack instructions (`PackA0`, `PackB0`, etc.) are excluded from the ascending-order check because pack ordering has its own validation in `hook_up_packs`
