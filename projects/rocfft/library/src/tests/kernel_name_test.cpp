// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "compute_scheme.h"
#include "enum_printer.h"
#include "load_store_ops.h"
#include "rtc_chirp_gen.h"
#include "rtc_transpose_gen.h"
#include "rtc_twiddle_gen.h"

#include <gtest/gtest.h>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Kernels built at runtime are cached on disk under their name, so two
// different kernels sharing a name would make the cache hand back the wrong
// kernel. These tests vary every input that goes into a name and stop at the
// first repeated name, printing it.

// last ComputeScheme value; update if a scheme is added after it
static constexpr int lastScheme = CS_KERNEL_3D_SINGLE;

TEST(rocfft_internal, transpose_kernel_names_are_unique)
{
    // the tiles the library uses (64x16 for single, 32x32 otherwise), plus
    // 16x64 to check that X and Y are not mixed up
    const std::vector<std::pair<unsigned int, unsigned int>> tiles = {{64, 16}, {32, 32}, {16, 64}};
    // higher dims all use one generic kernel, so they don't get their own name
    const std::vector<size_t>           dims   = {2, 3};
    const std::vector<KIntType>         itypes = {KIntType::U32, KIntType::U64};
    const std::vector<rocfft_precision> precisions
        = {rocfft_precision_half, rocfft_precision_single, rocfft_precision_double};
    // hermitian types generate the same code as complex ones, so they
    // deliberately share a name and are left out
    const std::vector<rocfft_array_type> arrayTypes = {rocfft_array_type_complex_interleaved,
                                                       rocfft_array_type_complex_planar,
                                                       rocfft_array_type_real};
    // {steps, direction}: the library uses 0-5 large twiddle steps, and the
    // direction only matters (and only appears in the name) when steps > 0
    std::vector<std::pair<size_t, int>> largeTwds = {{0, -1}};
    for(size_t steps = 1; steps <= 5; ++steps)
        for(int direction : {-1, 1})
            largeTwds.push_back({steps, direction});
    const std::vector<CallbackType> callbacks = {CallbackType::NONE,
                                                 CallbackType::USER_LOAD_STORE,
                                                 CallbackType::USER_LOAD_STORE_R2C,
                                                 CallbackType::USER_LOAD_STORE_C2R};

    std::set<std::string> names;
    for(auto [tileX, tileY] : tiles)
        for(auto dim : dims)
            for(auto itype : itypes)
                for(auto precision : precisions)
                    for(auto inType : arrayTypes)
                        for(auto outType : arrayTypes)
                            for(auto [steps, direction] : largeTwds)
                                for(auto cb : callbacks)
                                    for(bool diagonal : {false, true})
                                        for(bool tileAligned : {false, true})
                                            for(bool grid3D : {false, true})
                                            {
                                                TransposeSpecs specs{};
                                                specs.itype             = itype;
                                                specs.tileX             = tileX;
                                                specs.tileY             = tileY;
                                                specs.dim               = dim;
                                                specs.precision         = precision;
                                                specs.inArrayType       = inType;
                                                specs.outArrayType      = outType;
                                                specs.largeTwdSteps     = steps;
                                                specs.largeTwdDirection = direction;
                                                specs.diagonal          = diagonal;
                                                specs.tileAligned       = tileAligned;
                                                specs.cbtype            = cb;
                                                specs.grid3D            = grid3D;

                                                const auto name = transpose_rtc_kernel_name(specs);
                                                ASSERT_TRUE(names.insert(name).second)
                                                    << "duplicate kernel name " << name;
                                            }
}

TEST(rocfft_internal, twiddle_and_chirp_kernel_names_are_unique)
{
    // every table type, precision and index type
    const std::vector<TwiddleTableType> types = {TwiddleTableType::RADICES,
                                                 TwiddleTableType::LENGTH_N,
                                                 TwiddleTableType::HALF_N,
                                                 TwiddleTableType::LARGE,
                                                 TwiddleTableType::PARTIAL_PASS_N};
    const std::vector<rocfft_precision> precisions
        = {rocfft_precision_half, rocfft_precision_single, rocfft_precision_double};
    const std::vector<KIntType> itypes = {KIntType::U32, KIntType::U64};

    // twiddle and chirp kernels share one cache, so check them together
    std::set<std::string> names;
    for(auto precision : precisions)
        for(auto itype : itypes)
        {
            for(auto type : types)
            {
                const auto name = twiddle_rtc_kernel_name(type, precision, itype);
                ASSERT_TRUE(names.insert(name).second) << "duplicate kernel name " << name;
            }
            const auto name = chirp_rtc_kernel_name(precision, itype);
            ASSERT_TRUE(names.insert(name).second) << "duplicate kernel name " << name;
        }
}

TEST(rocfft_internal, load_store_ops_change_the_name)
{
    // a kernel that scales its output is different code from one that does
    // not, so it needs a different name; a kernel with no ops keeps the plain
    // name so existing cache entries stay valid
    LoadOps  plainLoad;
    StoreOps plainStore;
    StoreOps scalingStore;
    scalingStore.scale_factor = 2.0;

    EXPECT_TRUE(load_store_name_suffix(std::nullopt, std::nullopt).empty());
    EXPECT_TRUE(load_store_name_suffix(plainLoad, plainStore).empty());

    EXPECT_FALSE(scalingStore.name_suffix().empty());
    EXPECT_NE(load_store_name_suffix(plainLoad, scalingStore),
              load_store_name_suffix(plainLoad, plainStore));
}

TEST(rocfft_internal, kernel_scheme_abbreviations_are_unique)
{
    // the abbreviation goes into stockham kernel names, so two schemes that
    // generate different code must not share one
    std::set<std::string> abbreviations;
    size_t                handled = 0;

    for(int i = CS_NONE; i <= lastScheme; ++i)
    {
        try
        {
            abbreviations.insert(PrintKernelSchemeAbbr(static_cast<ComputeScheme>(i)));
            ++handled;
        }
        catch(const std::runtime_error&)
        {
            // schemes that describe a decomposition rather than a kernel
            // have no abbreviation
        }
    }

    // stockham and its partial-pass variants deliberately share a name; every
    // other kernel scheme must have a distinct abbreviation
    EXPECT_EQ(PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM_PP),
              PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM));
    EXPECT_EQ(PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM_PP_BLOCK_CC),
              PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM_BLOCK_CC));
    EXPECT_EQ(abbreviations.size() + 2, handled);
}

// Tuning results are saved to disk with the scheme, precision and other
// settings written as text, then read back with the StrTo* functions. If a
// value does not come back as itself, a saved solution would load the wrong
// kernel or fail to load.
TEST(rocfft_internal, every_scheme_round_trips_through_its_name)
{
    for(int i = CS_NONE; i <= lastScheme; ++i)
    {
        const auto scheme = static_cast<ComputeScheme>(i);
        const auto name   = PrintScheme(scheme);
        EXPECT_FALSE(name.empty()) << "scheme " << i << " has no printed name";
        EXPECT_EQ(StrToComputeScheme(name), scheme) << "scheme " << i << " printed as " << name;
    }
}

TEST(rocfft_internal, enum_printers_round_trip)
{
    for(auto precision : {rocfft_precision_half, rocfft_precision_single, rocfft_precision_double})
        EXPECT_EQ(StrToPrecision(PrintPrecision(precision)), precision);

    for(auto arrayType : {rocfft_array_type_complex_interleaved,
                          rocfft_array_type_complex_planar,
                          rocfft_array_type_real,
                          rocfft_array_type_hermitian_interleaved,
                          rocfft_array_type_hermitian_planar})
        EXPECT_EQ(StrToArrayType(PrintArrayType(arrayType)), arrayType);

    for(auto ebtype : {EmbeddedType::NONE, EmbeddedType::Real2C_POST, EmbeddedType::C2Real_PRE})
        EXPECT_EQ(StrToEBType(PrintEBType(ebtype)), ebtype);

    for(auto sbrc : {NONE, TILE_ALIGNED, TILE_UNALIGNED, DIAGONAL})
        EXPECT_EQ(StrToSBRCTransType(PrintSBRCTransposeType(sbrc)), sbrc);
}
