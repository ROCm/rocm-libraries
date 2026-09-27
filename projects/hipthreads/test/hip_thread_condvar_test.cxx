#include "hip/thread"
#include "hip/mutex"
#include "hip/pseudo_mutex" // From hip::wthread library
#include "hip/pseudo_condition_variable"
#include "hip/hip_runtime.h"
#include <cassert>

__device__ void gmain() {
    static __device__ hip::pseudo_mutex m1, m2;
    static __device__ hip::pseudo_condition_variable cv;
    auto other_thread = hip::wthread([&] __device__() {
        hip::unique_lock guard1(m1);
        hip::unique_lock guard2(m2);
        assert(guard1.owns_lock());
        assert(guard2.owns_lock());
        cv.wait(guard1);
        assert(guard1.owns_lock());
        assert(guard2.owns_lock());
    });

    // Keep testing m2 until it's acquired by other_thread, so we know other thread has gotten past the point where it
    // locks m1 for the first time.
    // Technically the C++ standard doesn't enforce any synchronization between lock and try_lock, but whatever, this isn't a rigorous test anyways
    while (m2.try_lock()) {
        m2.unlock();
    }

    hip::unique_lock guard(m1); // wait for other_thread to call cv.wait(guard1)
    guard.unlock(); // make sure we unlock m1 so other_thread doesn't spin forever trying to acquire it again when returning from cv.wait
    cv.notify_one();
    other_thread.join();
}

int main() {
    hip::wthread([] __device__(){gmain();}).join();
    return 0;
}
