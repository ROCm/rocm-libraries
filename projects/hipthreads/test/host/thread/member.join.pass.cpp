// UNSUPPORTED: no-threads
//
// <thread>
//
// void join();
// Tests host-side join of hip::thread.

#include <hip/thread>
#include <cassert>

__managed__ bool op_run = false;

int main(int, char**) {
    op_run = false;

    hip::thread t([] __device__() { op_run = true; });
    assert(t.joinable());
    t.join();
    assert(!t.joinable());
    assert(t.get_id() == hip::thread::id());
    assert(op_run);

    return 0;
}
