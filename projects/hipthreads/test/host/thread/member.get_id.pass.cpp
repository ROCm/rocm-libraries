// UNSUPPORTED: no-threads
//
// <thread>
//
// id get_id() const;
// Tests host-side get_id() on hip::thread.

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    hip::thread t0;
    assert(t0.get_id() == hip::thread::id());

    hip::thread t1([] __device__() {});
    hip::thread::id id1 = t1.get_id();
    assert(id1 != hip::thread::id());

    t1.join();
    assert(t1.get_id() == hip::thread::id());

    return 0;
}
