#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/DataTypes/DataTypes_FP4.hpp>
#include <rocRoller/Utilities/Error.hpp>

namespace rocRoller
{
    static uint8_t getlow(uint8_t twoFp4)
    {
        uint32_t ret = twoFp4 & 0xf;
        uint8_t  fp4 = ret;
        return fp4;
    }

    static uint8_t gethigh(uint8_t twoFp4)
    {
        uint32_t ret = (twoFp4 >> 4) & 0xf;
        uint8_t  fp4 = ret;
        return fp4;
    }

    static uint8_t getFp4(uint8_t twoFp4, int high)
    {
        if(high == 1)
        {
            return gethigh(twoFp4);
        }
        else
        {
            return getlow(twoFp4);
        }
    }

    static void setlow(uint8_t* twoFp4, uint8_t fp4)
    {
        uint32_t value = fp4;
        *twoFp4        = *twoFp4 & 0xf0;
        value          = value & 0xf;
        *twoFp4        = *twoFp4 | value;
    }

    static void sethigh(uint8_t* twoFp4, uint8_t fp4)
    {
        uint32_t value = fp4;
        *twoFp4        = *twoFp4 & 0x0f;
        value          = value & 0xf;
        value          = value << 4;
        *twoFp4        = *twoFp4 | value;
    }

    static void setFp4(uint8_t* twoFp4, uint8_t value, int high)
    {
        if(high == 1)
        {
            sethigh(twoFp4, value);
        }
        else
        {
            setlow(twoFp4, value);
        }
    }

    // Library function that gets fp4 from matrix
    static uint8_t getFp4(
        const uint8_t* const buffer, int m, int n, int i, int j, int rowmajor, bool debug = false)
    {
        int index = 0;
        if(rowmajor == 1)
        {
            index = i * n + j;
        }
        else
        {
            index = j * m + i;
        }
        int high = index % 2;

        uint8_t twoFp4 = buffer[index / 2];
        uint8_t ret    = getFp4(twoFp4, high);
        if(debug)
        {
            printf("m:%d, n:%d, i:%d, j:%d, index:%d, rowmajor:%d, ret:%01x\n",
                   m,
                   n,
                   i,
                   j,
                   index,
                   rowmajor,
                   ret);
        }
        return ret;
    }

    // Library function that sets fp4 to matrix
    static void setFp4(uint8_t* buffer, uint8_t value, int m, int n, int i, int j, int rowmajor)
    {
        int index = 0;
        if(rowmajor == 1)
        {
            index = i * n + j;
        }
        else
        {
            index = j * m + i;
        }
        int high = index % 2;

        setFp4(buffer + index / 2, value, high);
    }

    template <typename T>
    std::vector<T> unpackFP4x8(uint32_t const* x, size_t n)
    {
        auto rv = std::vector<T>(n * 8);

        for(int i = 0; i < n * 8; ++i)
        {
            uint8_t value = getFp4(reinterpret_cast<uint8_t const*>(x), 0, 0, i, 0, 0);
            if constexpr(std::is_same_v<T, uint8_t>)
                rv[i] = value;
            else if constexpr(std::is_same_v<T, float>)
            {
                uint4_t in;
                in.val    = value;
                float f32 = fp4_to_f32<float>(in);
                rv[i]     = f32;
            }
            else
                Throw<FatalError>("Unable to unpack FP4x8: unhandled data type.");
        }
        return rv;
    }

    std::vector<float> unpackFP4x8(std::vector<FP4x8> const& f4x8)
    {
        return unpackFP4x8<float>(reinterpret_cast<uint32_t const*>(f4x8.data()), f4x8.size());
    }

    std::vector<uint8_t> unpackFP4x8(std::vector<uint32_t> const& f4x8regs)
    {
        return unpackFP4x8<uint8_t>(f4x8regs.data(), f4x8regs.size());
    }

    void packFP4x8(uint32_t* out, uint8_t const* data, int n)
    {
        for(int i = 0; i < n; ++i)
            setFp4(reinterpret_cast<uint8_t*>(out), data[i], 0, 0, i, 0, 0);
        return;
    }

    std::vector<uint32_t> packFP4x8(std::vector<uint8_t> const& f4bytes)
    {
        std::vector<uint32_t> f4x8regs(f4bytes.size() / 8);
        packFP4x8(f4x8regs.data(), f4bytes.data(), f4bytes.size());
        return f4x8regs;
    }

    std::vector<uint32_t> f32_to_fp4x8(std::vector<float> f32)
    {
        AssertFatal(f32.size() % 8 == 0, "Invalid FP32 size");
        std::vector<uint8_t> data;
        for(auto const& value : f32)
        {
            FP4 fp4 = FP4(value);
            data.push_back(reinterpret_cast<uint8_t&>(fp4));
        }
        return packFP4x8(data);
    }

    std::vector<float> fp4x8_to_f32(std::vector<uint32_t> in)
    {
        return unpackFP4x8<float>(reinterpret_cast<uint32_t const*>(in.data()), in.size());
    }

};
