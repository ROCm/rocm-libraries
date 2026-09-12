// Phase A proof (RMSNorm): load a build-time flyDSL->HSACO code object and launch
// it via the HIP module API — exactly what hipDNN's `hsaco` UKD escape hatch does
// at runtime. Zero Python, zero flyDSL at runtime; just hipModuleLoadData + launch.
//
// Kernel: one thread per row, N baked per HSACO. grid=(rows,1,1) block=(1,1,1).
// flyDSL 0.3.x kernarg ABI: each Tensor arg = (global_buffer ptr 8B, by_value i32
// size 4B). Three tensors (out,x,w) => 44 bytes:
//   out_ptr@0 out_N@8  x_ptr@16 x_N@24  w_ptr@32 w_N@40
// out/x sizes are rows*N (flat views); w size is N. symbol rmsnorm_0.
//
// build: hipcc run_rmsnorm.cpp -o run_rmsnorm
// run:   ./run_rmsnorm rmsnorm_toy_n64_gfx950.hsaco rmsnorm_0 64 8
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                  \
    do {                                                                                       \
        hipError_t e = (x);                                                                    \
        if (e != hipSuccess) {                                                                 \
            printf("HIP err %d (%s) at %s:%d\n", e, hipGetErrorString(e), __FILE__, __LINE__); \
            return 1;                                                                          \
        }                                                                                      \
    } while (0)

static const float EPS = 1e-6f;

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "rmsnorm.hsaco";
    const char* sym = argc > 2 ? argv[2] : "rmsnorm_0";
    const int D = argc > 3 ? atoi(argv[3]) : 64;
    const int ROWS = argc > 4 ? atoi(argv[4]) : 8;

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

    const int N = ROWS * D;
    std::vector<float> hx(N), hw(D), hout(N);
    std::mt19937 rng(0);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int i = 0; i < N; i++) hx[i] = nd(rng);
    for (int i = 0; i < D; i++) hw[i] = nd(rng);

    float *dx, *dw, *dout;
    CK(hipMalloc(&dx, N * sizeof(float)));
    CK(hipMalloc(&dw, D * sizeof(float)));
    CK(hipMalloc(&dout, N * sizeof(float)));
    CK(hipMemcpy(dx, hx.data(), N * sizeof(float), hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, hw.data(), D * sizeof(float), hipMemcpyHostToDevice));

    // === the escape-hatch core ===
    hipModule_t mod;
    CK(hipModuleLoadData(&mod, blob.data()));
    hipFunction_t fn;
    CK(hipModuleGetFunction(&fn, mod, sym));

    // flyDSL 0.3.x ABI: (ptr, i32 size) per Tensor, 44 bytes interleaved.
    unsigned char args[44] = {};
    void* out_ptr = dout;
    void* x_ptr = dx;
    void* w_ptr = dw;
    const int32_t xn = N;  // rows*D, numel of the flat out/x views
    const int32_t wn = D;
    std::memcpy(args + 0, &out_ptr, sizeof(void*));
    std::memcpy(args + 8, &xn, sizeof(int32_t));
    std::memcpy(args + 16, &x_ptr, sizeof(void*));
    std::memcpy(args + 24, &xn, sizeof(int32_t));
    std::memcpy(args + 32, &w_ptr, sizeof(void*));
    std::memcpy(args + 40, &wn, sizeof(int32_t));
    size_t argsz = sizeof(args);
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, args, HIP_LAUNCH_PARAM_BUFFER_SIZE, &argsz,
                      HIP_LAUNCH_PARAM_END};
    CK(hipModuleLaunchKernel(fn, /*gx*/ ROWS, 1, 1, /*bx*/ 1, 1, 1,
                             /*shmem*/ 0, /*stream*/ 0, nullptr, config));
    CK(hipDeviceSynchronize());
    // === end escape-hatch core ===

    CK(hipMemcpy(hout.data(), dout, N * sizeof(float), hipMemcpyDeviceToHost));

    // CPU reference RMSNorm: out[r,i] = x[r,i] * rsqrt(mean_i(x^2)+eps) * w[i]
    double max_err = 0.0;
    for (int r = 0; r < ROWS; r++) {
        double ss = 0.0;
        for (int i = 0; i < D; i++) {
            float v = hx[r * D + i];
            ss += (double)v * v;
        }
        float inv = 1.0f / std::sqrt((float)(ss / D) + EPS);
        for (int i = 0; i < D; i++) {
            float ref = hx[r * D + i] * inv * hw[i];
            double e = std::fabs((double)hout[r * D + i] - ref);
            if (e > max_err) max_err = e;
        }
    }
    bool ok = max_err < 1e-3;
    printf("%s | max_abs_err=%.3e | out[0,:4]=%.4f %.4f %.4f %.4f\n", ok ? "OK" : "MISMATCH",
           max_err, hout[0], hout[1], hout[2], hout[3]);
    hipModuleUnload(mod);
    hipFree(dx);
    hipFree(dw);
    hipFree(dout);
    return ok ? 0 : 1;
}
