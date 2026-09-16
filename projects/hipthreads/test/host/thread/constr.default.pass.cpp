// UNSUPPORTED: no-threads
//
// <thread>
//
// thread();
// Tests host-side default construction of hip::thread.

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    hip::thread t;
    assert(!t.joinable());
    assert(t.get_id() == hip::thread::id());

    return 0;
}
