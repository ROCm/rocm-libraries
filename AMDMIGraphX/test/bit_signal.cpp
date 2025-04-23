#include <migraphx/bit_signal.hpp>
#include <test.hpp>

TEST_CASE(triggered)
{
    migraphx::bit_signal<64> signals;
    EXPECT(signals.nslots() == 0);
    auto slot = signals.subscribe();
    EXPECT(signals.nslots() == 1);
    EXPECT(slot.valid());
    EXPECT(not slot.triggered());
    signals.notify();
    EXPECT(slot.triggered());
}

TEST_CASE(default_slot)
{
    auto slot = migraphx::bit_signal<64>::slot();
    EXPECT(not slot.valid());
}

TEST_CASE(copy_slot)
{
    migraphx::bit_signal<64> signals;
    EXPECT(signals.nslots() == 0);
    auto slot1 = signals.subscribe();
    auto slot2 = slot1; // NOLINT
    EXPECT(signals.nslots() == 2);
    EXPECT(slot1.i != slot2.i);
    EXPECT(slot1.valid());
    EXPECT(slot2.valid());
    EXPECT(not slot1.triggered());
    EXPECT(not slot2.triggered());
    signals.notify();
    EXPECT(slot1.triggered());
    EXPECT(slot2.triggered());
}

TEST_CASE(move_slot)
{
    migraphx::bit_signal<64> signals;
    EXPECT(signals.nslots() == 0);
    auto slot1 = signals.subscribe();
    auto slot2 = std::move(slot1);
    EXPECT(signals.nslots() == 1);
    EXPECT(not slot1.valid()); // cppcheck-suppress accessMoved
    EXPECT(slot2.valid());
    EXPECT(not slot2.triggered());
    signals.notify();
    EXPECT(slot2.triggered());
}

TEST_CASE(over_subscribe)
{
    migraphx::bit_signal<1> signals;
    EXPECT(signals.nslots() == 0);
    auto slot = signals.subscribe();
    EXPECT(signals.nslots() == 1);
    EXPECT(slot.valid());
    EXPECT(not slot.triggered());
    EXPECT(test::throws([&] { signals.subscribe(); }));
    EXPECT(test::throws([&] { auto slot2 = slot; }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
