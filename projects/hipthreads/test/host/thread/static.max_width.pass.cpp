// UNSUPPORTED: no-threads
//
// <thread>
//
// static constexpr unsigned max_width();
// Tests host-side max_width().

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    static_assert(hip::thread::max_width() > 0, "max_width must be positive");
    assert(hip::thread::max_width() > 0);

    return 0;
}
