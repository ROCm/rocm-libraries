# fixSizeInconsistencies Generator-Key Dedup Investigation

## Verdict

Fixed for the current `Tensile.TensileMergeLibrary` target source.

## Current source references

- `Tensile/TensileMergeLibrary.py:56-68`: `fixSizeInconsistencies` trims long size tuples and deduplicates via `sizesDict[tuple(value for value in size)] = [size, index]`.
- `Tensile/TensileMergeLibrary.py:235-236`: `mergeLogic` calls this function for both base and incremental size maps before merging.
- `Tensile/bin/TensileMergeLibrary:29-44`: the `TensileMergeLibrary` command imports and runs `Tensile.TensileMergeLibrary.main()`, so this is the active command path for this item.
- `Tensile/Tests/unit/characterization/TensileMergeLibrary/test_tensile_merge_library_char.py:59-74`: current checkout contains the characterization that two long-format sizes trimming to `[1, 2, 3, 4]` collapse to one entry.
- `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/TensileMergeLibrary/test_tensile_merge_library_char.py:59-74`: referenced investigation characterization matches the current checkout for this behavior.

## Static evidence

The old bug described in the characterization was a dict key like `(value for value in size)`. That creates a new generator object for every row, so two equal logical sizes never compare equal as dict keys.

The current implementation materializes the generator into a tuple at `Tensile/TensileMergeLibrary.py:61`:

```python
sizesDict[tuple(value for value in size)] = [size, index]
```

For the pinned duplicate case:

```python
sizes = [
    [[1, 2, 3, 4, 9, 9, 9, 9], [0, 0.5]],
    [[1, 2, 3, 4, 8, 8, 8, 8], [1, 0.6]],
]
```

`Tensile/TensileMergeLibrary.py:60` trims both size lists to `[1, 2, 3, 4]`. `Tensile/TensileMergeLibrary.py:61` then maps both rows to the same tuple key `(1, 2, 3, 4)`, so the dict contains one final entry and `len(newSizes)` returns `1` at `Tensile/TensileMergeLibrary.py:68`.

I did not run pytest. A direct import-based smoke check in this shell is blocked before reaching the helper because importing `Tensile.TensileMergeLibrary` currently pulls in `rocisa`, and this environment reports `ImportError: cannot import name 'rocIsa' from 'rocisa'`. The static source evidence is enough to verify the generator-key bug itself is not present in the active target implementation.

## Deprecated duplicate path

There is another function with the same name in `Tensile/Utilities/merge.py:56-72`. That file is a deprecated merge utility; its CLI path warns at `Tensile/Utilities/merge.py:534-535` to use `TensileMergeLibrary`. It does not contain the old dict generator-key pattern. It uses a separate in-place trim and membership approach, and its current characterization expects trimmed duplicates to be removed at `Tensile/Tests/unit/characterization/UtilitiesMerge/test_utilities_merge_char.py:90-98`.

## Impact

No open impact for `Tensile.TensileMergeLibrary`: duplicate long-format size rows that trim to the same 4-tuple are deduplicated in the current source. The previously described failure mode would have left duplicate size rows in merged logic files, but the current tuple key prevents that.

Residual risk is low. Because `dict` assignment keeps the last row for a duplicate key, the selected duplicate's index payload is last-writer-wins. That is existing behavior under the fixed implementation and should be intentional or covered by tests if payload choice matters.

## Recommended fix/test

No source fix is needed for `Tensile.TensileMergeLibrary`.

Keep the current characterization/unit coverage that asserts two long-format sizes trimming to the same prefix return one size. If strengthening coverage, add an assertion for the duplicate payload policy, e.g. whether the dedup should keep the first or last `[solutionIndex, efficiency]` entry when two rows trim to the same tuple.
