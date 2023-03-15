#include "gpu/memory"
#include "hip/hip_runtime.h"
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
    __device__ ~A() {
        printf("Destroying A: %d\n", b);
    }
    int b = 0;
};

__global__ void gmain() {
    {
        gpu::unique_ptr<int[]> x(new int[32]);
    }
    {   
        gpu::unique_ptr<A> y(new A{5});
        printf("Before destruction\n");
    }
    A *ptr;
    {   
        gpu::unique_ptr<A> z(new A{7});
        printf("Inside braces\n");
        ptr = z.release();
        printf("After release\n");
    }
    printf("Outside braces\n");
    delete ptr;
    printf("After delete\n");
}

int main() {
    /*
    gpu::unique_ptr<int> x(0);
    ++x;

    gpu::atomic<A> y;
    y.store(A{.b = 6});

    std::cout << x << std::endl;
    std::cout << static_cast<A>(y).b << std::endl;
    //*/

    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return 0;
}
