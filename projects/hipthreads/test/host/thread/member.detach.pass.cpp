// UNSUPPORTED: no-threads
//
// <thread>
//
// void detach();
// Tests host-side detach of hip::thread.

#include <hip/thread>
#include <cassert>

__managed__ volatile bool done = false;

int main(int, char**) {
    done = false;

    hip::thread t([] __device__() { done = true; });
    assert(t.joinable());
    t.detach();
    assert(!t.joinable());
    assert(t.get_id() == hip::thread::id());

    while (!done) {}
    assert(done);

    return 0;
}
