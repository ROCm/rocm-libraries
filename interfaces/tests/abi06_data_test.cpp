#define _GNU_SOURCE
#include <dlfcn.h>

#include <cstdio>

typedef int (*fn)(void);

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <librocblas.so.6>\n", argv[0]);
        return 2;
    }
    const char* path = argv[1];
    void* h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        fprintf(stderr, "dlopen %s: %s\n", path, dlerror());
        return 2;
    }

    int bad = 0;

    const unsigned int* d = (const unsigned int*)dlsym(h, "rocrand_h_sobol32_direction_vectors");
    if (!d) {
        fprintf(stderr, "dlsym rocrand_h_sobol32_direction_vectors: %s\n", dlerror());
        bad = 1;
    } else {
        printf("data rocrand_h_sobol32_direction_vectors[0]=0x%08x\n", d[0]);
        bad |= (d[0] != 0x80000000u);
    }

    void* dv = dlvsym(h, "rocrand_h_sobol32_direction_vectors", "ROCBLAS_ABI_6");
    printf("dlvsym rocrand_h_sobol32_direction_vectors@ROCBLAS_ABI_6 -> %p\n", dv);
    bad |= (dv == nullptr);

    void* dw = dlvsym(h, "rocrand_h_sobol32_direction_vectors", "ROCBLAS_ABI_7");
    printf("dlvsym rocrand_h_sobol32_direction_vectors@ROCBLAS_ABI_7 (wrong) -> %p (expect nil)\n",
           dw);
    bad |= (dw != nullptr);

    fn g = (fn)dlvsym(h, "rocblas_sgemm", "ROCBLAS_ABI_6");
    int v = g ? g() : -1;
    printf("dlvsym rocblas_sgemm@ROCBLAS_ABI_6 -> %d\n", v);
    bad |= (v != 6);

    printf("[abi06_data] verdict: %s\n",
           bad ? "FAIL"
               : "PASS (real sobol32 data object versioned @ROCBLAS_ABI_6, "
                 "wrong-node nil, co-versioned rocblas_sgemm resolves)");
    dlclose(h);
    return bad ? 1 : 0;
}
