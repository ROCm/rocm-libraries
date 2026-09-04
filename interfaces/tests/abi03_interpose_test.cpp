#define _GNU_SOURCE
#include <dlfcn.h>

#include <cstdio>
#include <cstring>

typedef int (*fn)(void);

#ifndef PROV_A_PATH
#define PROV_A_PATH "libprovA.so.6"
#endif
#ifndef PROV_B_PATH
#define PROV_B_PATH "libprovB.so.7"
#endif
#ifndef ANON_A_PATH
#define ANON_A_PATH "libanonA.so"
#endif
#ifndef ANON_B_PATH
#define ANON_B_PATH "libanonB.so"
#endif

static int run_positive() {
    void* A = dlopen(PROV_A_PATH, RTLD_NOW | RTLD_GLOBAL);
    void* B = dlopen(PROV_B_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!A || !B) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    int ok = 1;

    fn a6 = (fn)dlvsym(A, "rocblas_sgemm", "ROCBLAS_ABI_6");
    fn b7 = (fn)dlvsym(B, "rocblas_sgemm", "ROCBLAS_ABI_7");
    int va = a6 ? a6() : -1;
    int vb = b7 ? b7() : -1;
    printf(
        "[positive] dlvsym(A,rocblas_sgemm,ROCBLAS_ABI_6)=%d  "
        "dlvsym(B,rocblas_sgemm,ROCBLAS_ABI_7)=%d  (each resolves its OWN)\n",
        va, vb);
    ok &= (va == 6 && vb == 7);

    void* a_wrong = dlvsym(A, "rocblas_sgemm", "ROCBLAS_ABI_7");
    void* b_wrong = dlvsym(B, "rocblas_sgemm", "ROCBLAS_ABI_6");
    printf(
        "[positive] dlvsym(A,...,ROCBLAS_ABI_7)=%p  dlvsym(B,...,ROCBLAS_ABI_6)=%p  "
        "(expect nil/nil)\n",
        a_wrong, b_wrong);
    ok &= (a_wrong == nullptr && b_wrong == nullptr);

    fn a_plain = (fn)dlsym(A, "rocblas_sgemm");
    fn b_plain = (fn)dlsym(B, "rocblas_sgemm");
    int pa = a_plain ? a_plain() : -1;
    int pb = b_plain ? b_plain() : -1;
    printf(
        "[positive] dlsym(A,rocblas_sgemm)=%d  dlsym(B,rocblas_sgemm)=%d  "
        "(handle-scoped, each OWN)\n",
        pa, pb);
    ok &= (pa == 6 && pb == 7);

    printf("[positive] verdict: %s\n", ok ? "PASS (versioning defeats interposition)" : "FAIL");
    return ok ? 0 : 1;
}

static int run_negative() {
    void* A = dlopen(ANON_A_PATH, RTLD_NOW | RTLD_GLOBAL);
    void* B = dlopen(ANON_B_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!A || !B) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    fn g = (fn)dlsym(RTLD_DEFAULT, "rocblas_sgemm");
    int vg = g ? g() : -1;

    fn bown = (fn)dlsym(B, "rocblas_sgemm");
    int vb = bown ? bown() : -1;

    printf(
        "[negative] bare dlsym(RTLD_DEFAULT,rocblas_sgemm)=%d  "
        "(B's own handle-scoped value=%d)\n",
        vg, vb);

    int hazard_reproduced = (vg == 6 && vb == 7);
    printf(
        "[negative] interposition %s: global bare lookup resolved to the "
        "first-loaded loader (%d), shadowing the second (%d)\n",
        hazard_reproduced ? "REPRODUCED" : "NOT reproduced", vg, vb);
    return hazard_reproduced ? 0 : 1;
}

static int run_versioned_bare() {
    // Both providers carry genuine version nodes (A: ROCBLAS_ABI_6 default,
    // B: ROCBLAS_ABI_7 default), unlike the anonymous negative control. A is
    // loaded first, so it is earlier in the global scope.
    void* A = dlopen(PROV_A_PATH, RTLD_NOW | RTLD_GLOBAL);
    void* B = dlopen(PROV_B_PATH, RTLD_NOW | RTLD_GLOBAL);
    if (!A || !B) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 2;
    }

    // A bare, unversioned RTLD_DEFAULT lookup does not consult the version nodes:
    // it takes the first-loaded default definition (ABI_6), not the newer major.
    fn g = (fn)dlsym(RTLD_DEFAULT, "rocblas_sgemm");
    int vg = g ? g() : -1;

    // Same two noded DSOs, version-aware lookup: dlvsym still reaches B's node,
    // proving the nodes are present and functional - only the lookup form differs.
    fn v7 = (fn)dlvsym(B, "rocblas_sgemm", "ROCBLAS_ABI_7");
    int v = v7 ? v7() : -1;

    printf(
        "[versioned-bare] bare dlsym(RTLD_DEFAULT,rocblas_sgemm)=%d  "
        "dlvsym(B,...,ROCBLAS_ABI_7)=%d\n",
        vg, v);
    int ok = (vg == 6 && v == 7);
    printf("[versioned-bare] verdict: %s\n",
           ok ? "PASS (nodes present - dlvsym reaches ABI_7 - yet the bare lookup "
                "still takes first-loaded ABI_6; the node defense does not cover "
                "unversioned RTLD_DEFAULT)"
              : "FAIL (bare lookup or version-aware lookup did not behave as stated)");
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* mode = (argc > 1) ? argv[1] : "positive";
    if (!strcmp(mode, "positive")) return run_positive();
    if (!strcmp(mode, "negative")) return run_negative();
    if (!strcmp(mode, "versioned_bare")) return run_versioned_bare();
    fprintf(stderr, "usage: %s positive|negative|versioned_bare\n", argv[0]);
    return 2;
}
