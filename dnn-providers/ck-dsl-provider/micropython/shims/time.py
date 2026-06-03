# Minimal `time` shim — the embed port has no builtin `time`. ck_dsl uses it only
# for diagnostic timings (perf_counter / monotonic), so a constant clock is fine;
# this is what makes those timings read as zero in the embed build.
def perf_counter():
    return 0.0


def monotonic():
    return 0.0


def time():
    return 0.0
