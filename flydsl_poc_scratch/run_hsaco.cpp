// Phase A proof: load a build-time flyDSL->HSACO code object and launch it via
// the HIP module API — exactly what hipDNN's `hsaco` UKD escape hatch does at
// runtime. Zero Python, zero flyDSL at runtime; just hipModuleLoadData + launch.
//
// build: hipcc run_hsaco.cpp -o run_hsaco
// run:   ./run_hsaco vadd_flydsl.hsaco vadd_0
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                  \
    do {                                                                                       \
        hipError_t e = (x);                                                                    \
        if (e != hipSuccess) {                                                                 \
            printf("HIP err %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "vadd_flydsl.hsaco";
    const char* sym = argc > 2 ? argv[2] : "vadd_0";

    // read the build-time code object into memory
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("cannot open %s\n", path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> blob(n);
    if (fread(blob.data(), 1, n, f) != (size_t)n) {
        printf("short read\n");
        return 1;
    }
    fclose(f);

    const int N = 256;
    std::vector<float> ha(N), hb(N), hout(N);
    for (int i = 0; i < N; i++) {
        ha[i] = (float)i;
        hb[i] = 1.0f;
    }

    float *da, *db, *dout;
    CK(hipMalloc(&da, N * sizeof(float)));
    CK(hipMalloc(&db, N * sizeof(float)));
    CK(hipMalloc(&dout, N * sizeof(float)));
    CK(hipMemcpy(da, ha.data(), N * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(db, hb.data(), N * sizeof(float), hipMemcpyHostToDevice));

    // === the escape-hatch core ===
    hipModule_t mod;
    CK(hipModuleLoadData(&mod, blob.data()));
    hipFunction_t fn;
    CK(hipModuleGetFunction(&fn, mod, sym));

    // kernarg layout from the .amdgpu_metadata: out@0, a@8, b@16 (24 bytes)
    struct {
        void* out;
        void* a;
        void* b;
    } args{dout, da, db};
    size_t argsz = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsz,
                      HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, /*gx*/ 1, 1, 1, /*bx*/ 256, 1, 1,
                             /*shmem*/ 0, /*stream*/ 0, nullptr, config));
    CK(hipDeviceSynchronize());
    // === end escape-hatch core ===

    CK(hipMemcpy(hout.data(), dout, N * sizeof(float), hipMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < N; i++) {
        float e = ha[i] + hb[i];
        if (std::fabs(hout[i] - e) > 1e-5f) bad++;
    }
    printf("%s | out[:5]=%.1f %.1f %.1f %.1f %.1f | mismatches=%d\n", bad ? "MISMATCH" : "OK",
           hout[0], hout[1], hout[2], hout[3], hout[4], bad);
    hipModuleUnload(mod);
    hipFree(da);
    hipFree(db);
    hipFree(dout);
    return bad ? 1 : 0;
}
