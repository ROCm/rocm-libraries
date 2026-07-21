// UNSUPPORTED: no-threads
// UNSUPPORTED: windows
//
// <thread>
//
// thread& operator=(thread&& t);
// Tests that move-assigning to a joinable thread calls terminate().

#include <hip/thread>
#include <cassert>
#include <cstdlib>
#include <exception>
#include <utility>

void exit_success() {
    ::std::_Exit(0);
}

int main(int, char**) {
    // Normal case: move-assign to non-joinable works
    {
        hip::thread t0([] __device__() {});
        hip::thread t1;
        t1 = ::std::move(t0);
        assert(t1.joinable());
        assert(!t0.joinable());
        t1.join();
    }

    // Terminate case: move-assign to joinable calls terminate
    ::std::set_terminate(exit_success);
    {
        hip::thread t0([] __device__() {});
        hip::thread t1([] __device__() {});
        t0 = ::std::move(t1);
        assert(false);
    }

    return 0;
}
