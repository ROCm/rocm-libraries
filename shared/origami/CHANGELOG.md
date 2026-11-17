# Changelog for Origami

## Origami 1.1.0

### Added

* `origami::argsort_configs()` and `origami::argsort_result_t`. They rank a candidate list the same way as `rank_configs()` (same tie-breaking and fallback behavior), but return the position of each config in the input vector (best first) with its predicted latency instead of copies of the configs. Use them to avoid copying `config_t` objects. The returned indices refer to the exact `configs` vector that was passed in; they are not `config_t::index`.
* A `streamk::select_grid_size()` overload that takes the reduction strategy explicitly (`reduction_t reduction_strategy`). Callers that choose the reduction themselves with `streamk::select_reduction()` no longer need to copy and modify the `config_t`.

### Changed

* `rank_configs()`, `select_config()` and `select_topk_configs()` are now built on `argsort_configs()`. Ranking results are unchanged, but only the returned configs are copied rather than every candidate.
* `datatype_to_bits()` is now an `inline constexpr noexcept` function defined in `origami/types.hpp`, and `data_type_to_bytes()` is now `inline constexpr noexcept`. `datatype_to_bits()` is no longer exported from the Origami library (it no longer has `ORIGAMI_EXPORT`), so it is not an exported symbol of `liborigami`. Code that includes `origami/types.hpp` is unaffected. The Python binding is unchanged.
* The documentation of `config_t::index` now states that it is an opaque user tag that Origami does not use. To get the position of a config in the candidate vector, use the indices returned by `argsort_configs()`.

### Deprecated

* `config_t::reduction_strategy`. The reduction strategy depends on the problem as well as the config. Origami's launch model no longer reads it; it is still used by the 4-argument `streamk::select_grid_size()` and by `config_t::operator==` and `config_t::hash()`.
* The 4-argument `streamk::select_grid_size()`. Use the overload that takes `reduction_t` instead. The Python `select_grid_size` binding still exposes the 4-argument form, so the Python API is unchanged.
