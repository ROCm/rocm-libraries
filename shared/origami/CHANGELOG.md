# Changelog for Origami

## Origami 1.1.0

### Added

* `origami::argsort_configs()` and `origami::argsort_result_t`. They rank a candidate list the same way as `rank_configs()` (same tie-breaking and fallback behavior), but return the position of each config in the input vector (best first) with its predicted latency instead of copies of the configs. Use them to avoid copying `config_t` objects. The returned indices refer to the exact `configs` vector that was passed in; they are not `config_t::index`.

### Changed

* `rank_configs()`, `select_config()` and `select_topk_configs()` are now built on `argsort_configs()`. Ranking results are unchanged, but only the returned configs are copied rather than every candidate.
* `streamk::select_grid_size()` with `grid_selection_t::k_split_aware` now picks the reduction strategy itself, the same way `streamk::select_reduction()` does, when `config.reduction_strategy` is `reduction_t::none` (the default). A `config_t` that sets `reduction_strategy` is used as before. Behavior change: for a default `config_t`, `none` used to behave like "not parallel"; large-K, low-tile problems may now get the parallel-reduction grid.
* `datatype_to_bits()` is now an `inline constexpr noexcept` function defined in `origami/types.hpp`, and `data_type_to_bytes()` is now `inline constexpr noexcept`. `datatype_to_bits()` is no longer exported from the Origami library (it no longer has `ORIGAMI_EXPORT`), so it is not an exported symbol of `liborigami`. Code that includes `origami/types.hpp` is unaffected. The Python binding is unchanged.
