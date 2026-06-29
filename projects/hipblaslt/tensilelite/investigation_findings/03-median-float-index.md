# Item 3: median() Python 3 float-index crash

## Verdict

Open.

## Current Source References

- `Tensile/TensileBenchmarkLibraryClient.py:79` defines `PrintStats`.
- `Tensile/TensileBenchmarkLibraryClient.py:82` calls `median(gflopList)`.
- `Tensile/TensileBenchmarkLibraryClient.py:83` calls `median(msList)`.
- `Tensile/TensileBenchmarkLibraryClient.py:161` defines `median(lst)`.
- `Tensile/TensileBenchmarkLibraryClient.py:162` sorts the input into `sortedList`.
- `Tensile/TensileBenchmarkLibraryClient.py:163` returns `sortedList[len(sortedList)/2]`.
- `Tensile/Tests/unit/characterization/TensileBenchmarkLibraryClient/test_stats_char.py:26` defines `test_median_is_broken_in_py3`.
- `Tensile/Tests/unit/characterization/TensileBenchmarkLibraryClient/test_stats_char.py:29` pins `median([3, 1, 2])` as raising `TypeError`.

The referenced characterization file under `/home/alvasile/repos/rocm-libraries-investigation/projects/hipblaslt/tensilelite/Tensile/Tests/unit/characterization/TensileBenchmarkLibraryClient/test_stats_char.py` is available and matches the same pinned failure: `median()` uses `/`, yielding a float index on Python 3.

## Reproduction / Static Evidence

Static evidence in the target source:

```python
def median(lst):
  sortedList = sorted(lst)
  return sortedList[len(sortedList)/2]
```

On Python 3, `/` always produces a `float`, so `len(sortedList)/2` is `1.5` for a three-element list. List indices must be integers or slices.

Executed from `/home/alvasile/repos/rocm-libraries/projects/hipblaslt/tensilelite`:

```bash
python3 -c 'from Tensile.TensileBenchmarkLibraryClient import median; print(median([3, 1, 2]))'
```

Observed result:

```text
TypeError: list indices must be integers or slices, not float
```

The traceback points to `Tensile/TensileBenchmarkLibraryClient.py:163`.

## Impact

Any Python 3 execution path that calls `median()` fails before returning a statistic. `PrintStats` calls `median()` for both `gflopList` and `msList`, so `TensileBenchmarkLibraryClient` can complete benchmark subprocess collection but then crash when formatting summary statistics. The local characterization suite also documents this as a pinned latent bug.

## Recommended Fix / Test

Use integer indexing for the existing behavior:

```python
def median(lst):
  sortedList = sorted(lst)
  return sortedList[len(sortedList) // 2]
```

This preserves the Python 2 behavior of selecting the upper middle element for even-length lists. If a mathematically conventional median is intended, use `statistics.median` or explicitly average the two middle values for even-length lists, but that would be a behavior change.

Recommended tests:

- Update the pinned characterization so `median([3, 1, 2]) == 2`.
- Add an even-length case documenting the intended legacy behavior, for example `median([4, 1, 2, 3]) == 3` if preserving the current Python 2 semantics.
- Add a small `PrintStats` smoke test with non-empty `gflopList` and `msList` to prove the caller no longer crashes.
