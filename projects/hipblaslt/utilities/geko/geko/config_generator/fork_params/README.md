# `fork_params`

Hardware-specific **optimization parameter** builders (non-MI fork params and Tensile `Groups` dimensions) and **post-processors** that adjust MI groups and fork params after the main generation step. Selection is by `config['ARCH']` and `config['search_space']` in [`__init__.py`](__init__.py).

For YAML keys, CLI, and the full driver pipeline, see the [parent `config_generator` README](../README.md).

---

## Where this runs in the pipeline

[`generate_fork_params`](../fork_param_generator.py) does, for each `[M, N, B, K]`:

1. `MIDesign.generate_for_size` → MI groups  
2. `OptimizationParams.generate_for_size` → fork param dict + extra group dimensions  
3. Optional `post_processor.apply` → may change both  
4. Set `Groups` to `[mi_groups] + opt_groups` and compute kernel count  

---

## Adding a new `ARCH`

1. **`geko/constants.py`** and **`geko/config_generator/constants.py`** — Add a gfx-style id to `SUPPORTED_ARCH` in `geko/constants.py`, then add matching `_ARCH_SPECS[...]` / `HARDWARE_MAP[...]` in `config_generator/constants.py` with `CUs`, `XCC`, `ONLY_INCLUDE_MIs` (per-`DataType` MI allowlists), and Tensile `LibraryLogic` fields (fourth tuple component; reuse `_LIBRARY_LOGIC_FIELDS_GFX950` / `_LIBRARY_LOGIC_FIELDS_GFX942` or extend with new `_LIBRARY_LOGIC_FIELDS_*` as needed). If optional-field defaults differ per `ARCH`, extend [`CONFIG_DEFAULTS_BY_ARCH`](../constants.py) in [`constants.py`](../constants.py).

2. **`hw_profiles/<id>/optimization_param.py`** — Subclass [`BaseOptimizationParams`](optimization_param.py): decorate methods with `@param` and `@group`; discovery is automatic via `generate_for_size`. Add a **generic** search-space variant the same way (e.g. `GFX942GAParams` subclasses `BaseOptimizationParams` in the existing profiles).

3. **`hw_profiles/<id>/post_processor.py`** — Subclass [`BasePostProcessor`](post_processor.py): use `@post_process` for ordered steps. `MT_DU` handling lives on the base class. Provide heuristic and generic subclasses if both search spaces need different behavior.

4. **`__init__.py`** — Register all four: `_HEURISTIC_PROFILES`, `_GENERIC_PROFILES`, `_HEURISTIC_POST_PROCESSORS`, `_GENERIC_POST_PROCESSORS`.

5. **Tests** — Extend or add cases under [`tests/config_generator/`](../../../tests/config_generator/).

Until step 4 is done, `get_optimization_params` will raise `KeyError` for the new `ARCH` even if `HARDWARE_MAP` already lists it.

---

## Design choices

- **`BaseParamBuilder`** ([`optimization_param.py`](optimization_param.py)) loads Tensile metadata once via [`param_meta.py`](param_meta.py) so `_make_param` can attach default/range comments to [`ForkParameter`](../shared_utils.py) without hard-coding strings everywhere.

- **Split of concerns:** `MIDesign` owns MI discovery and filtering style; optimization profiles own enumerations of other fork axes and non-MI group dimensions; post-processors apply cross-cutting edits (e.g. tightening lists) without reimplementing MI logic.

- **Search space selection:** Parallel class sets and registries for the same `ARCH` string. `get_optimization_params` / `get_post_processor` switch on `config["search_space"]` and index the right map.

For extension patterns, read `generate_for_size` on [`BaseOptimizationParams`](optimization_param.py) and `apply` on [`BasePostProcessor`](post_processor.py).
