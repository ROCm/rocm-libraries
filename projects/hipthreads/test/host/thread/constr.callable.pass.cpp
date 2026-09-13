// UNSUPPORTED: no-threads
//
// <thread>
//
// template <class F, class ...Args> thread(F&& f, Args&&... args);
// Tests host-side thread construction with a device lambda.

#include <hip/thread>
#include <cassert>

__managed__ bool op_run = false;

int main(int, char**) {
    op_run = false;

    hip::thread t([] __device__() { op_run = true; });
    assert(t.joinable());
    assert(t.get_id() != hip::thread::id());
    t.join();
    assert(!t.joinable());
    assert(op_run);

    return 0;
}
