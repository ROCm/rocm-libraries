// UNSUPPORTED: no-threads
//
// <thread>
//
// class thread::id
//
// bool operator==(thread::id, thread::id);
// bool operator!=(thread::id, thread::id);
// bool operator< (thread::id, thread::id);
// bool operator<=(thread::id, thread::id);
// bool operator> (thread::id, thread::id);
// bool operator>=(thread::id, thread::id);
// Tests host-side thread::id comparison operators.

#include <hip/thread>
#include <cassert>

int main(int, char**) {
    hip::thread::id def;

    assert(def == def);
    assert(!(def != def));
    assert(!(def < def));
    assert(def <= def);
    assert(!(def > def));
    assert(def >= def);

    hip::thread t([] __device__() {});
    hip::thread::id active = t.get_id();

    assert(active != def);
    assert(!(active == def));

    bool less = def < active;
    assert(less == !(active <= def) || active == def);
    assert((def < active) == (active > def));
    assert((def <= active) == (active >= def));

    t.join();

    return 0;
}
