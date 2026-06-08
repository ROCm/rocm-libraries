# Bug: MIOpen TunaNet silently uses gfx942 model for gfx950 hardware

## Summary

MIOpen AI kernel tuning silently substitutes the gfx942 model when running on gfx950
(MI350X) hardware. There is no warning, log message, or error at the fallback site. Any
solver using this path on gfx950 receives predictions from a model selected by gfx942
architecture data, regardless of architectural differences.

## Location

`projects/miopen/src/conv/heuristics/ai_heuristics.cpp:1134`

```cpp
std::shared_ptr<Model> GetModel(const std::string& arch, const std::string& solver)
{
    static std::map<std::string, std::shared_ptr<Model>> models;
    auto it = models.find(solver);

    auto model_arch = arch;
    if(model_arch == "gfx950")
        model_arch = "gfx942"; // use gfx942 model for gfx950 until we have a gfx950 model

    if(it == models.end())
    {
        std::shared_ptr<Model> model = std::make_shared<Model>(model_arch, solver);
        models[solver]               = model;
        return model;
    }
```

## Reproduction

Run an AI kernel tuning path on a gfx950 node with a solver that calls `GetModel(arch,
solver)`. The function receives `arch == "gfx950"`, rewrites it to `"gfx942"`, and loads
that model without reporting the fallback.

## Expected Behavior

- If a gfx950-specific model exists: load it.
- If no gfx950 model exists: log a clear warning, for example: `TunaNet: no model for
  gfx950, falling back to gfx942; predictions may be inaccurate`.
- Never fall back silently.

## Actual Behavior

The architecture string is overwritten before model selection with no logging. Users and
instrumentation cannot tell that gfx950 predictions came from a gfx942 model unless they
inspect source code.

## Impact

- gfx950/MI350X AI kernel tuning can use MI300X-trained predictions.
- gfx950 differs from gfx942 in CU count and memory/cache characteristics.
- Suboptimal kernel selection can occur from day one of gfx950 deployment.
- The silent fallback makes the issue hard to detect from logs or benchmark output.

## Proposed Fix

```cpp
if(model_arch == "gfx950")
{
    MIOPEN_LOG_W("TunaNet: no model for gfx950, falling back to gfx942; "
                 "predictions may be inaccurate for MI350X hardware");
    model_arch = "gfx942";
}
```

Longer term, add a gfx950-specific model and remove the fallback.

## Discovered During

hipDNN heuristic plugin prototype work (branch: `users/ysoliman/heuristic-plugin-prototype`).
MIOpen TunaNet source inspection, 2026-06.
