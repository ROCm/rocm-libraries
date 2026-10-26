# HostNumerics design decisions

## Enum counts

Enums do not require a terminal `Count` or `N` member. Use `Count` when an actual consumer needs it for array sizing, iteration, numeric range validation, or seed allocation. Policy and status enums whose members only describe meaningful choices need no count. An invalid switch case that merely rejects `Count` is insufficient reason to add it.

A terminal count is a common C++ idiom for enums with consecutive values starting at zero. Adding a member before `Count` updates the count automatically. This interpretation depends on keeping those values consecutive; gaps, aliases, and bit flags require a different approach. `Count` is outside the ordinary choices and must be excluded from iteration over them.

The retained counts have concrete uses:

- `ScalarType::Count` sizes scalar metadata and caches, bounds type identifiers, and supports exhaustive iteration.
- `ScalarCategory::Count` lets the independent metadata test check that category identifiers are in range.
- The hipBLASLt adapter's `OperandSequence::Count` reserves one seed slot per operand role in each batch or grouped problem. Preserve those role values and the seed stride when changing initialization code.

Replacing `Count` with an arbitrary ordinary member and a "keep this last" comment carries the same ordering obligation while giving that member a second responsibility. Do not use that replacement as a general convention. Deriving a count from an existing collection is reasonable when it simplifies the code, but enum-to-table correspondence still needs validation. Avoid adding generic enum traits, reflection, or extra collections solely to eliminate a useful count.

Apply this decision to core code, consumer adapters, samples, and tests. Remove unnecessary counts and their dummy switch cases while preserving ordinary enum values, numerical behavior, and required invalid-value handling. Removing a named sentinel does not prevent casts from producing unnamed enum values; checked boundaries must retain their existing validation.
