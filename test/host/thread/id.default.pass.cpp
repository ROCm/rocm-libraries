// UNSUPPORTED: no-threads
//
// <thread>
//
// class thread::id
//
// id();
// Tests host-side default construction of thread::id.

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    hip::thread::id id0;
    hip::thread::id id1;
    assert(id0 == id1);
    assert(id0 == hip::thread::id());

    return 0;
}
