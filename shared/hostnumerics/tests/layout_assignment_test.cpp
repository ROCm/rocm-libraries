// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstddef>
#include <cstdlib>
#include <new>
#include <roc/hostnumerics/layout.hpp>

namespace {
int allocationBudget = -1;
}

void* operator new(std::size_t size) {
    if (allocationBudget == 0) throw std::bad_alloc();
    if (allocationBudget > 0) --allocationBudget;
    if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
    throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
    std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
    std::free(memory);
}

int main() {
    using namespace roc::hostnumerics;
    const Layout source(Shape{2, 3}, {3, 1}, 0);
    Layout target;
    bool copyFailed = false;
    // Allow the shape copy, then fail the stride allocation.
    allocationBudget = 1;
    try {
        target = source;
    } catch (const std::bad_alloc&) {
        copyFailed = true;
    }
    allocationBudget = -1;
    return target == (copyFailed ? Layout{} : source) ? 0 : 1;
}
