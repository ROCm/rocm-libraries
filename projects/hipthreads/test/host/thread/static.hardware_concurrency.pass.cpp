// UNSUPPORTED: no-threads
//
// <thread>
//
// static unsigned hardware_concurrency();
// Tests host-side hardware_concurrency().

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    assert(hip::thread::hardware_concurrency() > 0);

    return 0;
}
