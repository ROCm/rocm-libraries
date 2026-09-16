// UNSUPPORTED: no-threads
//
// <thread>
//
// class thread::id
//
// id(const id&) = default;
// Tests host-side copy of thread::id.

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    hip::thread::id id0;
    hip::thread::id id1 = id0;
    assert(id0 == id1);

    return 0;
}
