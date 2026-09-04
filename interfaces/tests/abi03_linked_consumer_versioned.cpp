#include <cstdio>

extern "C" int rocblas_sgemm(void);
__asm__(".symver rocblas_sgemm, rocblas_sgemm@ROCBLAS_ABI_7");

int main(void) {
    int v = rocblas_sgemm();
    printf(
        "[linked-versioned] rocblas_sgemm@ROCBLAS_ABI_7 -> %d  "
        "(provA ABI_6 also NEEDED and earlier in scope)\n",
        v);
    int ok = (v == 7);
    printf("[linked-versioned] verdict: %s\n",
           ok ? "PASS (versioned relocation bound to ABI_7 despite an earlier "
                "ABI_6 interposer)"
              : "FAIL (bound to the wrong major)");
    return ok ? 0 : 1;
}
