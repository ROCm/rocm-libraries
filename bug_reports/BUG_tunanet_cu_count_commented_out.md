# Bug: MIOpen AI conv heuristics ignore hardware CU count in feature extraction

## Summary

`GetFeaturesND()` accepts `max_cu` and `arch`, and callers pass the current device maximum
compute-unit count and architecture string. The implementation comments both parameters out
in the signature and never adds either value to the feature map. As a result, candidate
selection features are not differentiated by CU count or architecture at this layer.

## Location

`projects/miopen/src/conv/heuristics/ai_conv_nd_kernel_tuning_utils.cpp:67`

```cpp
// Helper: Extract 3D convolution features
std::map<std::string, float>
GetFeaturesND(const ProblemDescription& problem, int /*max_cu*/, const std::string& /*arch*/)
{
    std::map<std::string, float> features;

    const bool is3d = problem.Is3d();
    // 1: spatial_dim
    features["spatial_dim"] = is3d ? 3.0f : 2.0f;

    // 2-5: in_channels, in_d, in_h, in_w
    features["in_channels"] = static_cast<float>(ProblemInterpreter::GetInputChannelC(problem));
```

The caller passes the hardware values:

`projects/miopen/src/include/miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp:249`

```cpp
std::map<std::string, float> features =
    GetFeaturesND(problem, ctx.GetStream().GetMaxComputeUnits(), arch);
```

and similarly at `projects/miopen/src/include/miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp:441`.

## Expected Behavior

`GetFeaturesND()` should either:

- include `max_cu` and `arch` in the feature map when the model expects them, or
- remove the parameters from the function and callers if they are intentionally unused.

If enabled, `max_cu` should come from the current device and `arch` should identify the
architecture used for prediction.

## Actual Behavior

`max_cu` and `arch` are passed by callers but explicitly unnamed/commented out in the
callee. They are not included in the feature map. Device configurations with different CU
counts can produce identical feature vectors for otherwise identical convolution shapes.

## Impact

- Predictions are not directly differentiated by hardware capability in this feature path.
- A 228-CU MI300X and a 304-CU MI350X can receive the same feature vector for the same conv
  shape if all tensor/problem fields match.
- This interacts poorly with the gfx950->gfx942 model fallback: both model selection and
  feature extraction can lose hardware specificity.

## Proposed Fix

```cpp
std::map<std::string, float>
GetFeaturesND(const ProblemDescription& problem, int max_cu, const std::string& arch)
{
    std::map<std::string, float> features;
    features["max_cu"] = static_cast<float>(max_cu);
    features["arch"] = EncodeArch(arch);
    // existing features...
}
```

This may require retraining or validating the model metadata so `max_cu` and `arch` are
known input features. Coordinate with the model training owner before enabling.

## Discovered During

hipDNN heuristic plugin prototype work (branch: `users/ysoliman/heuristic-plugin-prototype`).
MIOpen AI heuristic source inspection, 2026-06.
