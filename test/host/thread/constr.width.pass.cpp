// UNSUPPORTED: no-threads
//
// <thread>
//
// template <class F, class ...Args> thread(unsigned width, F&& f, Args&&... args);
// Tests host-side thread construction with an explicit width parameter.

#include <hip/thread>
#include <cassert>

__managed__ bool op_run = false;

int main(int, char**) {
    op_run = false;

    hip::thread t(4, [] __device__() { op_run = true; });
    assert(t.joinable());
    t.join();
    assert(!t.joinable());
    assert(op_run);

    return 0;
}
