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

static_assert(std::is_same<gpu::unique_ptr_h<int>,                  gpu::unique_ptr<int,               gpu::host_delete<int>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<Trivial>,              gpu::unique_ptr<Trivial,           gpu::host_delete<Trivial>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<TriviallyCopyable>,    gpu::unique_ptr<TriviallyCopyable, gpu::host_delete<TriviallyCopyable>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<TriviallyMoveable>,    gpu::unique_ptr<TriviallyMoveable, gpu::host_delete<TriviallyMoveable>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<NotConstructible>,     gpu::unique_ptr<NotConstructible,  gpu::host_delete<NotConstructible>>>::value);

static_assert(std::is_same<gpu::unique_ptr_h<int[]>,                gpu::unique_ptr<int[],               gpu::host_delete<int[]>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<Trivial[]>,            gpu::unique_ptr<Trivial[],           gpu::host_delete<Trivial[]>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<TriviallyCopyable[]>,  gpu::unique_ptr<TriviallyCopyable[], gpu::host_delete<TriviallyCopyable[]>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<TriviallyMoveable[]>,  gpu::unique_ptr<TriviallyMoveable[], gpu::host_delete<TriviallyMoveable[]>>>::value);
static_assert(std::is_same<gpu::unique_ptr_h<NotConstructible[]>,   gpu::unique_ptr<NotConstructible[],  gpu::host_delete<NotConstructible[]>>>::value);

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

template <class T>
gpu::unique_ptr_h<T> test_auto_conversion(gpu::unique_ptr_h<T> ptr) {
    return ptr;
}
template <class T>
std::unique_ptr<T> test_auto_conversion2(std::unique_ptr<T> ptr) {
    return ptr;
}

int main() {
    {
        Trivial tt{};
        TriviallyCopyable tc{};
        TriviallyMoveable tm{};
        NotConstructible nc = NotConstructible::make();
        auto a = gpu::make_unique<int>();
        auto b = gpu::make_unique<float[]>(3);
        auto c = gpu::make_unique<Trivial>(tt);
        auto d = gpu::make_unique<Trivial>();
        auto e = gpu::make_unique<Trivial[]>(5);
        auto f = gpu::make_unique<TriviallyCopyable>(tc);
        auto g = gpu::make_unique<TriviallyCopyable>(TriviallyCopyable{});
        auto h = gpu::make_unique<TriviallyMoveable>(std::move(tm));
        auto i = gpu::make_unique<TriviallyMoveable>(TriviallyMoveable{});
        auto j = gpu::make_unique<NotConstructible>(std::move(nc));
        auto k = gpu::make_unique<NotConstructible>(NotConstructible::make());
        auto l = gpu::make_unique<const NotConstructible>(NotConstructible::make());
    }

    {
        int x_h = 3;
        auto x_d = gpu::make_unique<int>(x_h);
        hipLaunchKernelGGL(test_value, dim3(1), dim3(1), 0, nullptr, x_d.get(), 3, 5);
        CHECK(hipMemcpy(&x_h, x_d.get(), sizeof(x_h), hipMemcpyDeviceToHost));
        assert(x_h == 5);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }

    {
        auto x_h = std::make_unique<int>(17);
        gpu::unique_ptr_h<int> x_d = std::move(x_h);
        hipLaunchKernelGGL(test_value, dim3(1), dim3(1), 0, nullptr, x_d.get(), 17, 19);
        int result;
        CHECK(hipMemcpy(&result, x_d.get(), sizeof(result), hipMemcpyDeviceToHost));
        assert(result == 19);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
    }

    {
        // convert from std::unique_ptr to gpu::unique_ptr before passing to test_auto_conversion
        // then convert the returned gpu::unique_ptr back to std::unique_ptr
        auto w_h = std::make_unique<int>(17);
        std::unique_ptr<int> w2_h = test_auto_conversion<int>(std::move(w_h));
        assert(*w2_h == 17);

        // convert from gpu::unique_ptr to std::unique_ptr before passing to test_auto_conversion
        // then convert the returned std::unique_ptr back to gpu::unique_ptr
        auto x_d = gpu::make_unique<int>(5);
        gpu::unique_ptr_h<int> x2_d = test_auto_conversion2<int>(std::move(x_d));

        // convert from std::unique_ptr to gpu::unique_ptr before passing to test_auto_conversion
        // then convert the returned gpu::unique_ptr back to std::unique_ptr
        auto y_h = std::make_unique<NotConstructible>(NotConstructible::make());
        std::unique_ptr<NotConstructible> y2_h = test_auto_conversion<NotConstructible>(std::move(y_h));

        // convert from gpu::unique_ptr to std::unique_ptr before passing to test_auto_conversion
        // then convert the returned std::unique_ptr back to gpu::unique_ptr
        auto z_d = gpu::make_unique<NotConstructible>(NotConstructible::make());
        gpu::unique_ptr_h<NotConstructible> z2_d = test_auto_conversion2<NotConstructible>(std::move(z_d));
    }

    {
        auto x_h = std::make_unique<int[]>(2);
        x_h[0] = 3;
        x_h[1] = 7;
        std::unique_ptr<int[]> x2_h = test_auto_conversion<int[]>({std::move(x_h), 2}).move_to_host(2);
        assert(x2_h[0] == 3);
        assert(x2_h[1] == 7);

        auto y_h = std::make_unique<int[]>(2);
        y_h[0] = 11;
        y_h[1] = 13;
        gpu::unique_ptr_h<int[]> y_d(std::move(y_h), 2);
        hipLaunchKernelGGL(test_value, dim3(1), dim3(1), 0, nullptr, y_d.get(),     11, 23);
        hipLaunchKernelGGL(test_value, dim3(1), dim3(1), 0, nullptr, y_d.get() + 1, 13, 29);
        CHECK(hipGetLastError());
        CHECK(hipDeviceSynchronize());
        std::unique_ptr<int[]> y2_h = std::move(y_d).move_to_host(2);
        assert(y2_h[0] == 23);
        assert(y2_h[1] == 29);

        auto z_h = std::unique_ptr<NotConstructible[]>(new NotConstructible[2]{NotConstructible::make(), NotConstructible::make()});
        z_h[0].i = 17;
        z_h[1].i = 19;
        gpu::unique_ptr_h<NotConstructible[]> z_d(std::move(z_h), 2);
        std::unique_ptr<NotConstructible[]> z2_h = std::move(z_d).move_to_host(2);
        assert(z2_h[0].i == 17);
        assert(z2_h[1].i == 19);
    }

    hipLaunchKernelGGL(gmain, dim3(1), dim3(1), 0, nullptr);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    return 0;
}
