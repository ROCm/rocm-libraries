# Changelog for Origami

## Origami 1.1.0

### Added

* `origami::argsort_configs()` and `origami::argsort_result_t`. They rank a candidate list the same way as `rank_configs()` (same tie-breaking and fallback behavior), but return the position of each config in the input vector (best first) with its predicted latency instead of copies of the configs. Use them to avoid copying `config_t` objects. The returned indices refer to the exact `configs` vector that was passed in; they are not `config_t::index`.

### Changed

* `datatype_to_bits()` is now an `inline constexpr noexcept` function defined in `origami/types.hpp`, and `data_type_to_bytes()` is now `inline constexpr noexcept`. `datatype_to_bits()` is still exported from `liborigami`.
