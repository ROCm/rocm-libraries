// wrapper_bench.c — measure per-call MIOpen API overhead.
//
//   wrapper_bench {getversion|createdestroy|noop} <iters>
//
// Prints a single line of TSV-ish output containing wall/cpu times, per-call
// nanoseconds, and getrusage counters (RSS, page faults, ctx switches).
// Mode "noop" exercises only the harness itself — it's the baseline used to
// subtract harness cost from the API-call measurements.
//
// The first call to each API is performed once outside the timed region so
// the steady-state number is not skewed by the one-shot PLT-resolution cost.
// For cold-vs-warm process startup, just compare separate invocations of
// this binary; the OS-load cost is what changes between them.

#include <miopen/miopen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static double now_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

static double cpu_sec(void)
{
    struct timespec t;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec / 1e9;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr, "usage: %s {getversion|createdestroy|noop} <iters>\n", argv[0]);
        return 2;
    }
    const char* mode = argv[1];
    long iters = strtol(argv[2], NULL, 10);
    if (iters <= 0) {
        fprintf(stderr, "iters must be > 0\n");
        return 2;
    }

    // Warm up the symbol resolution path so the timed loop sees steady state.
    size_t maj = 0, min = 0, pat = 0;
    miopenHandle_t handle = NULL;

    if (strcmp(mode, "getversion") == 0) {
        miopenGetVersion(&maj, &min, &pat);
    } else if (strcmp(mode, "createdestroy") == 0) {
        miopenCreate(&handle);
        miopenDestroy(handle);
    } else if (strcmp(mode, "noop") == 0) {
        /* nothing */
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        return 2;
    }

    double wall0 = now_sec();
    double cpu0  = cpu_sec();

    if (strcmp(mode, "getversion") == 0) {
        for (long i = 0; i < iters; ++i)
            miopenGetVersion(&maj, &min, &pat);
    } else if (strcmp(mode, "createdestroy") == 0) {
        for (long i = 0; i < iters; ++i) {
            miopenCreate(&handle);
            miopenDestroy(handle);
        }
    } else { /* noop */
        volatile long acc = 0;
        for (long i = 0; i < iters; ++i)
            acc += i;
        (void)acc;
    }

    double wall = now_sec() - wall0;
    double cpu  = cpu_sec() - cpu0;

    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);

    printf("mode=%s iters=%ld wall_s=%.6f cpu_s=%.6f per_call_ns=%.2f "
           "max_rss_kb=%ld minflt=%ld majflt=%ld nvcsw=%ld nivcsw=%ld "
           "v_maj=%zu v_min=%zu v_pat=%zu\n",
           mode, iters, wall, cpu, wall * 1e9 / iters,
           ru.ru_maxrss, ru.ru_minflt, ru.ru_majflt, ru.ru_nvcsw, ru.ru_nivcsw,
           maj, min, pat);
    return 0;
}
