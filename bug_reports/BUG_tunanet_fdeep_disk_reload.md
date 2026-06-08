# Bug: MIOpen TunaNetND reloads fdeep model from disk on every Forward() call

## Summary

`TunaNetNDModel::Forward()` calls `fdeep::load_model()` on every inference call. There is no
model member or lazy-load-once cache in the ND model path. Every new prediction reloads and
deserializes the model file from disk.

The legacy 2D `Model` class caches its fdeep model as a member, and the candidate-selection
path has a `GetFdeepModel()` cache helper. The ND model path should use the same pattern.

## Location

`projects/miopen/src/conv/heuristics/ai_heuristics.cpp:805`

```cpp
std::vector<float> Forward(const conv::ProblemDescription& problem) const override
{
    std::vector<float> features = ToFeatures(problem);
    MIOPEN_LOG_I2("TunaNetNDModel: Extracted " << features.size() << " features");

    // Use fdeep to run TunaNetND inference
    const int dim                = problem.Is3d() ? 3 : 2;
    const std::string model_path = ModelNDPath(device_name, dim);
    const auto model             = fdeep::load_model(model_path, true, fdeep::dev_null_logger);
    MIOPEN_LOG_I2("TunaNetNDModel: Loaded fdeep model from " << model_path << ".");

    // Convert features to fdeep tensor
    const auto input_tensor = fdeep::tensor(fdeep::tensor_shape(features.size()), features);
    const auto result       = model.predict({input_tensor});
```

Contrast with the cached helper in the same file:

`projects/miopen/src/conv/heuristics/ai_heuristics.cpp:1259`

```cpp
// Helper to load and cache fdeep models
const fdeep::model& GetFdeepModel(const std::string& path, const std::string& key)
{
    static std::map<std::string, std::unique_ptr<fdeep::model>> models;
    auto it = models.find(key);
    if(it == models.end())
    {
        if(!fs::exists(path))
            MIOPEN_THROW(miopenStatusInternalError, "Unable to load model file: " + path);
        auto model =
            std::make_unique<fdeep::model>(fdeep::load_model(path, true, fdeep::dev_null_logger));
```

## Expected Behavior

The fdeep model should be loaded once per `(device_name, dim)` and reused for subsequent
`Forward()` calls.

## Actual Behavior

Each `Forward()` call constructs `model_path` and calls `fdeep::load_model()` again. On
network-mounted filesystems, this turns prediction into repeated disk I/O and JSON/model
deserialization.

## Impact

- Repeated cold-path penalty for every TunaNetND prediction.
- Particularly severe on NFS-mounted home or system DB paths common on HPC clusters.
- Prediction latency becomes filesystem-dependent.
- The issue compounds with the gfx950 fallback: the fallback model can be reloaded from disk
  repeatedly.

## Proposed Fix

Cache the model as a member or use a static keyed cache:

```cpp
class TunaNetNDModel : public ModelND
{
    mutable std::optional<fdeep::model> model_;

    std::vector<float> Forward(const conv::ProblemDescription& problem) const override
    {
        if(!model_)
            model_.emplace(fdeep::load_model(ModelNDPath(device_name, dim), true,
                                             fdeep::dev_null_logger));
        return model_->predict(...);
    }
};
```

Alternatively, reuse the existing `GetFdeepModel(path, key)` cache pattern for ND models.

## Discovered During

hipDNN heuristic plugin prototype work (branch: `users/ysoliman/heuristic-plugin-prototype`).
MIOpen TunaNetND source inspection, 2026-06.
