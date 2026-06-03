# TensileBenchmarkLibraryClient.py — characterization target (stats)

Pins the pure stats helpers mean / stddev, and the median() py3 bug
(sortedList[len/2] uses float division -> TypeError on list index).

Resistance: BenchmarkProblemSize (subprocess), PrintStats, and the
TensileBenchmarkLibraryClient() argv/fs driver.
