#include "gpu/iterator"
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
    __host__ __device__ PrintingObj() noexcept {
        printf("Default Constructing\n");
    }
    __host__ __device__ PrintingObj(int _value) noexcept : value(_value) {
        printf("Constructing: %d\n", value);
    }
    __host__ __device__ PrintingObj(const PrintingObj &other) noexcept : value(other.value) {
        printf("Copy Constructing: %d\n", value);
    }
    __host__ __device__ PrintingObj(PrintingObj &&other) noexcept : value(other.value) {
        other.value = -1;
        printf("Move Constructing: %d\n", value);
    }
    __host__ __device__ PrintingObj& operator=(const PrintingObj &other) noexcept {
        printf("Copy Assignment: prev = %d, new = %d\n", value, other.value);
        value = other.value;
        return *this;
    }
    __host__ __device__ PrintingObj& operator=(PrintingObj &&other) noexcept {
        printf("Move Assignment: prev = %d, new = %d\n", value, other.value);
        if (this == &other)
            return *this;
        value = other.value;
        other.value = -2;
        return *this;
    }
    __host__ __device__ ~PrintingObj() {
        printf("Destroying: %d\n", value);
        value = -3;
    }
    int value = 0;
};

// Trivially Moveable, but no public constructor and not copy constructible (even non-trivially)
struct NotConstructible {
    char c = 29;
    int i;
  private:
    __device__ NotConstructible(int _i) : i(_i) {};
  public:
    static __device__ NotConstructible make(int _i) { return NotConstructible(_i); }
    __device__ NotConstructible(NotConstructible &&) = default;
    __device__ NotConstructible& operator=(NotConstructible &&) = default;
};

struct EqualityComparable {
    int i;
    __device__ EqualityComparable(int _i) : i(_i) {}
    __device__ bool operator==(const EqualityComparable &other) const { return i == other.i; }
};

struct LessThanComparable {
    int i;
    __device__ LessThanComparable(int _i) : i(_i) {}
    __device__ bool operator<(const LessThanComparable &other) const { return i < other.i; }
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
        assert(a.size() == 2);
        assert(a[0].value == 0);
        assert(a[1].value == 0);
        printf("Mark1\n");
        gpu::vector<PrintingObj> b(2, 3);
        assert(b.size() == 2);
        assert(b[0].value == 3);
        assert(b[1].value == 3);
        printf("Mark2\n");
        gpu::vector<PrintingObj> c{5};
        assert(c.size() == 1);
        assert(c[0].value == 5);
        printf("Mark3\n");
        gpu::vector<PrintingObj> d{7, 11};
        assert(d.size() == 2);
        assert(d[0].value == 7);
        assert(d[1].value == 11);
        printf("Mark4\n");
        gpu::vector<PrintingObj> e{13, 17, 19};
        assert(e.size() == 3);
        assert(e[0].value == 13);
        assert(e[1].value == 17);
        assert(e[2].value == 19);
        gpu::vector f = e;
        assert(e.size() == 3);
        assert(e[0].value == 13);
        assert(e[1].value == 17);
        assert(e[2].value == 19);
        assert(f.size() == 3);
        assert(f[0].value == 13);
        assert(f[1].value == 17);
        assert(f[2].value == 19);
        gpu::vector g = std::move(e);
        assert(e.size() == 0);
        assert(g.size() == 3);
        assert(g[0].value == 13);
        assert(g[1].value == 17);
        assert(g[2].value == 19);
    }
    {
        printf("Mark5\n");
        PrintingObj arr[2] = {23, 29};
        PrintingObj x;
        printf("Mark6\n");
        {
            gpu::vector a(gpu::begin(arr), gpu::end(arr));
            assert(a.size() == 2);
            assert(a[0].value == 23);
            assert(a[1].value == 29);
            x = std::move(a[0]);
            assert(a.size() == 2);
            assert(a[0].value == -2);
        }
    }
    {
        gpu::vector<PrintingObj> a{3, 5, 7};
        assert(a.size() == 3);
        assert(a[0].value == 3);
        assert(a[1].value == 5);
        assert(a[2].value == 7);
        printf("capacity = %zu\n", a.capacity());
        auto ibegin = a.begin();
        auto iend = a.end();
        gpu::vector b(a.begin(), a.end());
        assert(b.size() == 3);
        assert(b[0].value == 3);
        assert(b[1].value == 5);
        assert(b[2].value == 7);
        gpu::vector c(a.rbegin(), a.rend()-1);
        assert(c.size() == 2);
        assert(c[0].value == 7);
        assert(c[1].value == 5);
        gpu::vector d(ibegin+1, iend);
        assert(d.size() == 2);
        assert(d[0].value == 5);
        assert(d[1].value == 7);
        printf("Mark7\n");
        a.emplace_back(11);
        printf("Mark8\n");
        a.emplace(a.begin(), 13);
        printf("Mark9\n");
        assert(a.size() == 5);
        assert(a[0].value == 13);
        assert(a[1].value == 3);
        assert(a[2].value == 5);
        assert(a[3].value == 7);
        assert(a[4].value == 11);
        gpu::vector e(gpu::move_iterator(a.begin()), gpu::move_iterator(a.end()-1));
        printf("Mark10\n");
        assert(a.size() == 5);
        assert(a[0].value == -1);
        assert(a[1].value == -1);
        assert(a[2].value == -1);
        assert(a[3].value == -1);
        assert(a[4].value == 11);
        assert(e.size() == 4);
        assert(e[0].value == 13);
        assert(e[1].value == 3);
        assert(e[2].value == 5);
        assert(e[3].value == 7);
        e.resize(2);
        a.resize(6);
        a.resize(7, 17);
        assert(a.size() == 7);
        assert(a[0].value == -1);
        assert(a[1].value == -1);
        assert(a[2].value == -1);
        assert(a[3].value == -1);
        assert(a[4].value == 11);
        assert(a[5].value == 0);
        assert(a[6].value == 17);
        assert(e.size() == 2);
        assert(e[0].value == 13);
        assert(e[1].value == 3);
    }
    {
        gpu::vector<PrintingObj> a{3, 5, 7};
        assert(a.data()[0].value == 3);
        assert(a.data()[1].value == 5);
        assert(a.data()[2].value == 7);
        a.data()[1] = 11;
        assert(a[0].value == 3);
        assert(a[1].value == 11);
        assert(a[2].value == 7);
        a.front() = 13;
        a.back() = 17;
        assert(a[0].value == 13);
        assert(a[1].value == 11);
        assert(a[2].value == 17);
    }
    {
        gpu::vector<NotConstructible> a;
        a.reserve(3);
        assert(a.size() == 0);
        assert(a.empty());
        assert(a.capacity() == 3);
        a.push_back(NotConstructible::make(3));
        assert(a.size() == 1);
        assert(!a.empty());
        assert(a.capacity() == 3);
        a.push_back(NotConstructible::make(5));
        a.push_back(NotConstructible::make(7));
        assert(a.size() == 3);
        assert(a.capacity() == 3);
        a.push_back(NotConstructible::make(11));
        assert(a.size() == 4);
        assert(a.capacity() > 4);
        a.shrink_to_fit();
        assert(a.size() == 4);
        assert(a.capacity() == 4);
        assert(a[0].i == 3);
        assert(a[1].i == 5);
        assert(a[2].i == 7);
        assert(a[3].i == 11);
        a.erase(a.begin()+2);
        assert(a.size() == 3);
        assert(a.capacity() == 4);
        assert(a[0].i == 3);
        assert(a[1].i == 5);
        assert(a[2].i == 11);
        a.pop_back();
        assert(a.size() == 2);
        assert(a.capacity() == 4);
        assert(a[0].i == 3);
        assert(a[1].i == 5);
    }
    {
        gpu::vector<EqualityComparable> a{3, 5, 7};
        gpu::vector<EqualityComparable> b{3, 5, 7};
        gpu::vector<EqualityComparable> c{3, 6, 7};
        gpu::vector<EqualityComparable> d{3, 5, 7, 11};
        assert(a == a);
        assert(a == b);
        assert(a != c);
        assert(a != d);
        assert(d != a);
    }
    {
        gpu::vector<LessThanComparable> a{3, 5, 7};
        gpu::vector<LessThanComparable> b{3, 5, 7};
        gpu::vector<LessThanComparable> c{3, 6, 7};
        gpu::vector<LessThanComparable> d{3, 5, 7, 11};
        assert(!(a < a));
        assert(!(a > a));
        assert(a <= a);
        assert(a >= a);
        assert(!(a < b));
        assert(!(a > b));
        assert(a <= b);
        assert(a >= b);
        assert(a < c);
        assert(!(a > c));
        assert(a <= c);
        assert(!(a >= c));
        assert(a < d);
        assert(!(a > d));
        assert(a <= d);
        assert(!(a >= d));
        assert(!(d < a));
        assert(d > a);
        assert(!(d <= a));
        assert(d >= a);
    }
}

int main() {
    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return 0;
}
