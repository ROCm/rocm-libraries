// UNSUPPORTED: no-threads
// UNSUPPORTED: windows
//
// <thread>
//
// ~thread();
// Tests host-side destruction of hip::thread.
// Destroying a joinable thread calls terminate().

#include <hip/thread>
#include <cassert>
#include <cstdlib>
#include <exception>

void exit_success() {
    ::std::_Exit(0);
}

int main(int, char**) {
    // Destroying a non-joinable thread is safe.
    {
        hip::thread t;
        assert(!t.joinable());
    }
    {
        hip::thread t([] __device__() {});
        t.join();
        assert(!t.joinable());
    }

    // Destroying a joinable thread must call terminate().
    ::std::set_terminate(exit_success);
    {
        hip::thread t([] __device__() {});
        assert(t.joinable());
    }
    // If we reach here, terminate() was not called.
    assert(false);

    return 0;
}
