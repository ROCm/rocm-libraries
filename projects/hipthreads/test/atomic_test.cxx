#include "gpu/atomic.h"
#include <iostream>

struct A {
    int b = 0;
};

int main() {
    gpu::atomic<int> x(0);
    ++x;

    gpu::atomic<A> y;
    y.store(A{.b = 6});

    std::cout << x << std::endl;
    std::cout << static_cast<A>(y).b << std::endl;
    return 0;
}
