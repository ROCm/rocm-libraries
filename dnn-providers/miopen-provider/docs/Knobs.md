# MIOpen Plugin Knobs

Configuration knobs for the MIOpen legacy plugin.

## Available Knobs

| Knob Name | Type | Default | Description |
|-----------|------|---------|-------------|
| `global.benchmarking` | int64 | 0 | Enable MIOpen solver tuning. Set to 1 to find optimal solver. |

## Usage

```cpp
#include <hipdnn_frontend/Knob.hpp>

// Create execution plan with benchmarking enabled
std::vector<hipdnn_frontend::KnobSetting> settings;
settings.emplace_back("global.benchmarking", 1LL);
graph.create_execution_plan_ext(engineId, settings);
```

## Notes

- **First run with benchmarking**: Solver search may take seconds to minutes
- **Subsequent runs**: Results cached in MIOpen performance database (`~/.config/miopen/`)
- **Recommendation**: Enable during warm-up/tuning, disable for production
