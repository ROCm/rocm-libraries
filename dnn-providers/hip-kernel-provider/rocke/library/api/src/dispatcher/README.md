# Dispatcher Hardware Profiles

## Overview

Hardware profiles provide architecture-specific constants used for FMHA feature extraction during kernel selection. The system uses a **hybrid approach**:

- **HIP-queried values** (highest confidence): `num_cus`, `max_clock_mhz`, `wavefront_size`, `lds_capacity`
- **Generated supplements** (for fields HIP doesn't expose): `shader_engines`, `num_xcd`, cache sizes, etc.

## File Structure

```
dispatcher/
├── HardwareProfile.hpp                    # Struct definition + query methods
├── HardwareProfileSupplements.hpp         # GENERATED - constexpr tables
└── RockeClientDispatcher.cpp              # Uses fromDeviceWithSupplement()
```

## Adding a New Architecture

1. **Update the Python source:**
   ```bash
   # Edit: rocke/platform/python/rocke/heuristics/gen_sweep_data.py
   HW_PROFILES["gfx1200"] = {
       "hw_num_cus": 64,
       "hw_simds_per_cu": 2,
       "hw_shader_engines": 8,
       # ... (11 fields total)
   }
   ```

2. **Regenerate the C++ supplement:**
   ```bash
   cd build
   cmake --build . --target rocke_regenerate_hw_profiles
   ```

3. **Verify the output:**
   ```bash
   # Check that HardwareProfileSupplements.hpp now includes kGfx1200Supplement
   grep -A2 "kGfx1200Supplement" library/api/src/dispatcher/HardwareProfileSupplements.hpp
   ```

4. **Commit both files:**
   ```bash
   git add rocke/platform/python/rocke/heuristics/gen_sweep_data.py
   git add rocke/library/api/src/dispatcher/HardwareProfileSupplements.hpp
   git commit -m "feat(rocke): add gfx1200 hardware profile"
   ```

## Field Reference

### HIP-Queryable Fields
These are populated from `hipGetDeviceProperties()` at runtime:

| Field | Source | Notes |
|-------|--------|-------|
| `num_cus` | `props.multiProcessorCount` | Actual CU count from device |
| `max_clock_mhz` | `props.clockRate / 1000` | Actual clock (kHz → MHz) |
| `wavefront_size` | `props.warpSize` | 64 for CDNA, 32 for RDNA |
| `lds_capacity` | `props.sharedMemPerBlock` | LDS size in bytes |

### Supplement-Only Fields
These are NOT available from HIP and come from generated tables:

| Field | Example | Notes |
|-------|---------|-------|
| `shader_engines` | 32 (gfx950) | Not exposed by HIP |
| `num_xcd` | 8 (gfx950) | Not exposed by HIP |
| `simds_per_cu` | 4 (CDNA), 2 (RDNA) | Microarch constant |
| `max_waves_per_cu` | 32 (CDNA), 16 (RDNA) | Microarch constant |
| `l1_cache_kb` | 32 | Not exposed by HIP |
| `l2_cache_kb` | 4096 | Not exposed by HIP |
| `l3_cache_kb` | 262144 (gfx950) | Not exposed by HIP |

## Usage in Dispatcher

```cpp
// Old way (incomplete - missing 7 fields)
problem->hw = HardwareProfile::fromDevice(device);

// New way (complete - HIP + supplement)
problem->hw = HardwareProfile::fromDeviceWithSupplement(device, arch);
```

## Data Flow

```
Python HW_PROFILES (gen_sweep_data.py)
    ↓
gen_hw_profiles.py (code generator)
    ↓
HardwareProfileSupplements.hpp (constexpr tables)
    ↓
HardwareProfile::fromDeviceWithSupplement()
    ├─ Query HIP for 4 fields (authoritative)
    └─ Look up supplement for 7 fields (generated)
    ↓
Complete 11-field HardwareProfile
    ↓
FMHA featurizer (group-C features)
    ↓
Model scoring
```

## Troubleshooting

### "Unknown arch" at runtime
- Symptom: Supplement fields are 0, but HIP-queried fields are populated
- Cause: `getSupplement(arch)` returned `nullptr` for unknown arch string
- Fix: Add the arch to `HW_PROFILES` and regenerate

### Drift between Python and C++
- Symptom: Training uses different hardware values than dispatcher
- Cause: Forgot to regenerate after updating `HW_PROFILES`
- Fix: Always regenerate and commit both files together

### Build error: "HardwareProfileSupplements.hpp not found"
- Cause: Generated file missing (clean build or forgot to commit)
- Fix: `cmake --build . --target rocke_regenerate_hw_profiles`
