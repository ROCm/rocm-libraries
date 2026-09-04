// UNSUPPORTED: no-threads
//
// <thread>
//
// bool joinable() const;
// Tests host-side joinable() on hip::thread in various states.

#include <hip/thread>
#include <cassert>

__managed__ bool op_run = false;

int main(int, char**) {
    {
        hip::thread t;
        assert(!t.joinable());
    }
    {
        op_run = false;
        hip::thread t([] __device__() { op_run = true; });
        assert(t.joinable());
        t.join();
        assert(!t.joinable());
        assert(op_run);
    }

    return 0;
}
