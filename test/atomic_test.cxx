#include "gpu/atomic.h"
#include <iostream>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

struct A {
    int b = 0;
};

__global__ void gmain() {
    gpu::atomic<int> gx(0);
    ++gx;
    assert(gx == 1);
    int tst_val = 4;
    int new_val = 5;
    assert(!gx.compare_exchange_strong(tst_val, new_val));
    assert(tst_val == 1);
    assert(gx == 1);
    assert(gx.compare_exchange_strong(tst_val, new_val));
    assert(tst_val == 1);
    assert(gx == 5);

    gpu::atomic<A> gy;
    gy.store(A{.b = 6});
    assert(static_cast<A>(gy).b == 6);

    A tst_val2{.b = 7};
    A new_val2{.b = 8};
    assert(!gy.compare_exchange_strong(tst_val2, new_val2));
    assert(tst_val2.b == 6);
    assert(static_cast<A>(gy).b == 6);
    assert(gy.compare_exchange_strong(tst_val2, new_val2));
    assert(tst_val2.b == 6);
    assert(static_cast<A>(gy).b == 8);

}

int main() {
    gpu::atomic<int> x(0);
    ++x;

    gpu::atomic<A> y;
    y.store(A{.b = 6});

    std::cout << x << std::endl;
    std::cout << static_cast<A>(y).b << std::endl;

    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return 0;
}
