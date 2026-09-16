// UNSUPPORTED: no-threads
//
// <thread>
//
// void swap(thread& t);
// Tests host-side swap of hip::thread.

#include <hip/thread>
#include <cassert>

__managed__ bool op_run = false;

int main(int, char**) {
    op_run = false;

    hip::thread t0([] __device__() { op_run = true; });
    hip::thread::id id0 = t0.get_id();
    hip::thread t1;
    hip::thread::id id1 = t1.get_id();

    t0.swap(t1);
    assert(t0.get_id() == id1);
    assert(t1.get_id() == id0);

    assert(!t0.joinable());
    assert(t1.joinable());
    t1.join();
    assert(op_run);

    return 0;
}
