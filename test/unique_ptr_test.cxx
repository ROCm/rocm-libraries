#include "gpu/memory"
#include "hip/hip_runtime.h"
#include <iostream>
#include <cassert>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

struct PrintingDeleter {
    __device__ ~PrintingDeleter() {
        printf("Destroying PrintingDeleter: %d\n", b);
    }
    int b = 0;
};

struct Trivial {
    char c;
    int i;
};

// Trivially copyable, but not trivial
struct TriviallyCopyable {
    char c = 5;
    int i = 7;
};

// Trivially Moveable, but not copy constructible (even non-trivially)
struct TriviallyMoveable {
    char c = 11;
    int i = 13;
    TriviallyMoveable(TriviallyMoveable &&) = default;
};

// Trivially Moveable, but no public constructor and not copy constructible (even non-trivially)
struct NotConstructible {
    char c = 23;
    int i = 29;
  private:
    NotConstructible() {};
  public:
    static NotConstructible make() { return NotConstructible(); }
    NotConstructible(NotConstructible &&) = default;
};

__global__ void gmain() {
    {
        gpu::unique_ptr<int[]> x(new int[32]);
    }
    {   
        gpu::unique_ptr<PrintingDeleter> y(new PrintingDeleter{5});
        printf("Before destruction\n");
    }
    PrintingDeleter *ptr;
    {   
        gpu::unique_ptr<PrintingDeleter> z(new PrintingDeleter{7});
        printf("Inside braces\n");
        ptr = z.release();
        printf("After release\n");
    }
    printf("Outside braces\n");
    delete ptr;
    printf("After delete\n");
}

__global__ void test_value(int *ptr, int expected_value, int new_value) {
    assert(*ptr == expected_value);
    *ptr = new_value;
}

int main() {
    {
        Trivial tt{};
        TriviallyCopyable tc{};
        TriviallyMoveable tm{};
        NotConstructible nc = NotConstructible::make();
        auto a = gpu::make_unique<int>();
        auto b = gpu::make_unique<float[]>(3);
        auto c = gpu::make_unique(tt);
        auto d = gpu::make_unique<Trivial>();
        auto e = gpu::make_unique<Trivial[]>(5);
        auto f = gpu::make_unique(tc);
        auto g = gpu::make_unique(TriviallyCopyable{});
        auto h = gpu::make_unique(std::move(tm));
        auto i = gpu::make_unique(TriviallyMoveable{});
        auto j = gpu::make_unique(std::move(nc));
        auto k = gpu::make_unique(NotConstructible::make());
    }

    {
        int x_h = 3;
        auto x_d = gpu::make_unique(x_h);
        hipLaunchKernelGGL(test_value, dim3(1), dim3(1), 0, nullptr, x_d.get(), 3, 5);
        CHECK(hipMemcpy(&x_h, x_d.get(), sizeof(x_h), hipMemcpyDeviceToHost));
        assert(x_h == 5);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }

    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return 0;
}
