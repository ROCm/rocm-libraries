# Minimal `time` shim — the embed port has no builtin `time`. ck_dsl uses it only
# for diagnostic timings (perf_counter, already zeroed by a build transform); a
# constant clock is fine.
def perf_counter():
    return 0.0


def monotonic():
    return 0.0


def time():
    return 0.0
