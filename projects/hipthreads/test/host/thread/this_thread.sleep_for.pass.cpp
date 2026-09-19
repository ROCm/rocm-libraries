// UNSUPPORTED: no-threads
//
// ALLOW_RETRIES: 3
//
// <thread>
//
// template <class Rep, class Period>
//   void sleep_for(const chrono::duration<Rep, Period>& rel_time);
// Tests host-side this_thread::sleep_for().

#include <hip/thread>
#include <cassert>
#include <hip/std/chrono>

int main(int, char**) {
    typedef cuda::std::chrono::system_clock Clock;
    cuda::std::chrono::milliseconds ms(250);
    Clock::time_point t0 = Clock::now();
    hip::this_thread::sleep_for(ms);
    Clock::time_point t1 = Clock::now();
    assert(t1 - t0 >= ms);

    return 0;
}
