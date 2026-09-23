#pragma once

// Use static_for only when the loop index must be a constexpr in the body：
//   - array index that has to resolve to a fixed VGPR slot,
//   - `if constexpr`,
//   - template arg,
//   - or a derived constexpr feeding such a use.
// Otherwise prefer a plain for with a constexpr bound -- HIPCC -O3 unrolls it to the same
// SASS, with shorter source and faster compiles.
//
// Invoke f<Start>(), f<Start+Step>(), ... in sequence at compile time over the
// half-open range [Start, End). Step must be non-zero; the sign of (End - Start)
// must match Step (or both be zero, which yields zero iterations).
//
// Backwards-compatible with the original single-argument form: static_for<N>(f)
// resolves to End=N, Start=0, Step=1, i.e. iterates I = 0, 1, ..., N-1.
template <int End, int Start = 0, int Step = 1, typename F>
__device__ __forceinline__ void static_for(F f)
{
    static_assert(Step != 0, "static_for: Step must be non-zero");
    constexpr int diff = End - Start;
    static_assert((diff == 0) || ((diff > 0) == (Step > 0)),
                  "static_for: sign of (End - Start) must match Step");

    constexpr int abs_diff = (diff < 0) ? -diff : diff;
    constexpr int abs_step = (Step < 0) ? -Step : Step;
    constexpr int Count    = (abs_diff + abs_step - 1) / abs_step;

    [&]<int... Is>(std::integer_sequence<int, Is...>) {
        (f.template operator()<Start + Is * Step>(), ...);
    }(std::make_integer_sequence<int, Count>{});
}

// Call f<I>() for the unique I that matches the runtime idx.
template <int N, typename F>
__device__ __forceinline__ void dispatch(int idx, F f)
{
    static_for<N>([&]<int I>() {
        if(idx == I)
            f.template operator()<I>();
    });
}
