// UNSUPPORTED: no-threads
//
// <thread>
//
// thread& operator=(thread&& t);
// Tests host-side move assignment of hip::thread.

#include <hip/thread>
#include <cassert>
#include <utility>

__managed__ bool op_run = false;

int main(int, char**) {
    op_run = false;

    hip::thread t0([] __device__() { op_run = true; });
    hip::thread::id id0 = t0.get_id();

    hip::thread t1;
    t1 = ::std::move(t0);
    assert(t1.get_id() == id0);
    assert(t1.joinable());
    assert(!t0.joinable());
    assert(t0.get_id() == hip::thread::id());

    t1.join();
    assert(op_run);

    return 0;
}
