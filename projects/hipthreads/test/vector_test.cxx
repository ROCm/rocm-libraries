#include "gpu/vector"
#include "hip/hip_runtime.h"
#include <iostream>
#include <cassert>
#include <vector>

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

struct PrintingObj {
    __host__ __device__ PrintingObj() {
        printf("Default Constructing\n");
    }
    __host__ __device__ PrintingObj(int _b) : b(_b) {
        printf("Constructing: %d\n", b);
    }
    __host__ __device__ PrintingObj(const PrintingObj &other) : b(other.b) {
        printf("Copy Constructing: %d\n", b);
    }
    __host__ __device__ PrintingObj(PrintingObj &&other) : b(other.b) {
        other.b = -1;
        printf("Move Constructing: %d\n", b);
    }
    __host__ __device__ PrintingObj& operator=(const PrintingObj &other) {
        printf("Copy Assignment: prev = %d, new = %d\n", b, other.b);
        b = other.b;
        return *this;
    }
    __host__ __device__ PrintingObj& operator=(PrintingObj &&other) {
        printf("Move Assignment: prev = %d, new = %d\n", b, other.b);
        if (this == &other)
            return *this;
        b = other.b;
        other.b = -2;
        return *this;
    }
    __host__ __device__ ~PrintingObj() {
        printf("Destroying: %d\n", b);
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
        gpu::vector<int> a(10);
        gpu::vector<int> b(12, 2);
        gpu::vector<int> c{3};
        gpu::vector<int> d{5, 7};
        gpu::vector<int> e{11, 13};
    }
    {
        gpu::vector<PrintingObj> a(2);
        printf("Mark1\n");
        gpu::vector<PrintingObj> b(2, 3);
        printf("Mark2\n");
        gpu::vector<PrintingObj> c{5};
        printf("Mark3\n");
        gpu::vector<PrintingObj> d{7, 11};
        printf("Mark4\n");
        gpu::vector<PrintingObj> e{13, 17, 19};
    }
    {
        printf("Mark5\n");
        PrintingObj arr[2] = {23, 29};
        PrintingObj x;
        printf("Mark6\n");
        {
            gpu::vector<PrintingObj> a(gpu::begin(arr), gpu::end(arr));
            x = std::move(a[0]);
        }
    }
}

// __global__ void test_value(int *ptr, int expected_value, int new_value) {
//     assert(*ptr == expected_value);
//     *ptr = new_value;
// }

// template <class T>
// gpu::vector_h<T> test_auto_conversion(gpu::vector_h<T> ptr) {
//     return ptr;
// }
// template <class T>
// std::unique_ptr<T> test_auto_conversion2(std::unique_ptr<T> ptr) {
//     return ptr;
// }

int main() {
    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return 0;
}
