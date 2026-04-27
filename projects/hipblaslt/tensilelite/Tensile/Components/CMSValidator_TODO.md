# CMS Validator — Stage 09 Remaining Work

## Wholesale `make_mock_id_map` deletion

Phase 1 leaves `make_mock_id_map` in place. The wholesale migration
requires:

- Per-class kernel rewrite (~15 `setup_method` methods affected)
- Registered-schedule audit: every test class's tile shape must map to a
  registered CMS schedule. Some don't (e.g. there is no 128×64×32 TF32 TN
  schedule for `TestValidatePackTF32MultipleGroups`'s native MT0×MT1).
- 27 `_mock_dtype` call sites in `test_CustomSchedule.py` that swap to real
  `DataType` objects, which may change schedule predicate dispatch
- All `assert self.num_vmfma == N` test assertions audited if
  `create_base_kernel` baseline changes
- `hasCustomSchedule` mutates the kernel dict (e.g.
  `kernel["UseMFMAF32XEmulation"] = True`); `kernel_to_solution_config`
  must thread these mutations through, or validator and kernel writer
  disagree on what the kernel says

## `create_base_kernel` valid-Solution baseline

Same dependency chain as wholesale deletion. Must coordinate with that
migration.

## `_frozen_config_key` cache key collisions

The lazy-caching approach uses `_frozen_config_key(config)` to key
`_idmap_cache`. Verified to produce distinct keys for Phase 1's twin
configs, but the JSON serialization with `sort_keys=True` is not stress-
tested across all dict variations:

- Nested dicts with non-deterministic key insertion order
- `IsaVersion` inside list values (not just top-level)
- Mock objects that don't serialize cleanly

## Delete dead positional code

Once Known Issue 4 (below) is fixed:
- `_hook_up_packs_bf16`
- `_hook_up_packs_f32` (keep `_hook_up_middle_16_pairs`)
- `_hook_up_packs_f32_mfma`
- Hardcoded `pack_dependencies` dicts

## Known Issue 4

- `set_lr_needed_by_from_mfma_operands` doesn't propagate through pack
  chains (LR→Pack→MFMA)

Required before deleting `set_lr_needed_by_for_VMFMA`.

## `@applies_only_once` partial state on error

`hook_up_packs` is decorated with `@applies_only_once`. If
`derive_pack_must_start_after` raises (e.g., missing `rocisa_inst`), the
decorator may still mark the function as applied, preventing retry. Verify
the decorator does not record the function as applied on raise.

## Dual-path comparison re-add

When the primary path was wired (commit `23d8627a22`), the dual-path
comparison block (`derive_pack_must_start_after` vs positional
`_hook_up_packs_*`) was removed from `hook_up_packs`. We now have no way to
detect subtle bugs in the new path. Re-add as a debug-only mode (e.g.
`CMS_VALIDATE_DUAL_PATH=1` env var) before deleting the positional
functions.

## `_get_lrs_for_pack` still required

The original stage-09 plan was to delete `_get_lrs_for_pack`, but it's
still needed to tell `derive_pack_must_start_after` WHICH local reads to
consider for a given pack (the function inspects pack source registers but
needs to know the relevant LR scope). Either keep it or replace with a
register-based helper that walks the timeline.

## `MIArchVgpr` test verification

`MIArchVgpr` is copied through `kernel_to_solution_config` but the test
that exercises it (`test_schedule_128x128x32_TF32_MIArchVgpr`) calls the
mock-path `_make_context`. `MIArchVgpr` affects register allocation, so
the real idMap may differ in subtle ways from the test's hardcoded
schedule. Verify if/when that test is migrated to the real path.


## Investigate if build a full register chain is a good idea
Instead of having diferent functions recreating a traversal fo the registers, build the chain onces, and then have the different functions simply walk the already build graph.