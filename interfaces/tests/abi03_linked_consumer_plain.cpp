#include <cstdio>

extern "C" int rocblas_sgemm(void);

int main(void) {
    int v = rocblas_sgemm();
    printf(
        "[linked-plain] unversioned rocblas_sgemm -> %d  "
        "(same link line as the versioned consumer, provA ABI_6 first)\n",
        v);
    int ok = (v == 6);
    printf("[linked-plain] verdict: %s\n",
           ok ? "PASS (plain relocation interposed by the first-NEEDED ABI_6 "
                "provider - the hazard the version pin defeats)"
              : "FAIL (control did not reproduce the interposition)");
    return ok ? 0 : 1;
}
