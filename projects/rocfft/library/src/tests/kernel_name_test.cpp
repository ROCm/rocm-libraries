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
#include <vector>

TEST(rocfft_internal, transpose_kernel_names_are_unique)
{
    const std::vector<unsigned int>     tiles = {16, 32, 64};
    const std::vector<size_t>           dims  = {2, 3};
    const std::vector<rocfft_precision> precisions
        = {rocfft_precision_half, rocfft_precision_single, rocfft_precision_double};
    const std::vector<rocfft_array_type> arrayTypes = {rocfft_array_type_complex_interleaved,
                                                       rocfft_array_type_complex_planar,
                                                       rocfft_array_type_real};
    const std::vector<CallbackType> callbacks = {CallbackType::NONE, CallbackType::USER_LOAD_STORE};

    std::set<std::string> names;
    size_t                combinations = 0;

    for(auto tile : tiles)
        for(auto dim : dims)
            for(auto precision : precisions)
                for(auto inType : arrayTypes)
                    for(auto outType : arrayTypes)
                        for(auto cb : callbacks)
                            for(bool diagonal : {false, true})
                                for(bool tileAligned : {false, true})
                                {
                                    TransposeSpecs specs{};
                                    specs.tileX        = tile;
                                    specs.tileY        = tile;
                                    specs.dim          = dim;
                                    specs.precision    = precision;
                                    specs.inArrayType  = inType;
                                    specs.outArrayType = outType;
                                    specs.diagonal     = diagonal;
                                    specs.tileAligned  = tileAligned;
                                    specs.cbtype       = cb;

                                    names.insert(transpose_rtc_kernel_name(specs));
                                    ++combinations;
                                }

    EXPECT_EQ(names.size(), combinations);
}

TEST(rocfft_internal, twiddle_and_chirp_kernel_names_are_unique)
{
    const std::vector<TwiddleTableType> types = {TwiddleTableType::RADICES,
                                                 TwiddleTableType::LENGTH_N,
                                                 TwiddleTableType::HALF_N,
                                                 TwiddleTableType::LARGE,
                                                 TwiddleTableType::PARTIAL_PASS_N};
    const std::vector<rocfft_precision> precisions
        = {rocfft_precision_half, rocfft_precision_single, rocfft_precision_double};

    std::set<std::string> names;
    size_t                combinations = 0;

    for(auto type : types)
        for(auto precision : precisions)
        {
            names.insert(twiddle_rtc_kernel_name(type, precision));
            ++combinations;
        }

    for(auto precision : precisions)
    {
        names.insert(chirp_rtc_kernel_name(precision));
        ++combinations;
    }

    EXPECT_EQ(names.size(), combinations);
}

TEST(rocfft_internal, load_store_ops_change_the_name)
{
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

TEST(rocfft_internal, kernel_scheme_abbreviations)
{
    std::set<std::string> abbreviations;
    size_t                handled = 0;

    for(int i = CS_NONE; i <= CS_KERNEL_3D_SINGLE; ++i)
    {
        const auto scheme = static_cast<ComputeScheme>(i);
        try
        {
            abbreviations.insert(PrintKernelSchemeAbbr(scheme));
            ++handled;
        }
        catch(const std::runtime_error&)
        {
            // scheme has no kernel abbreviation, which is expected for the
            // schemes that describe a decomposition rather than a kernel
        }
    }

    // the schemes that do have an abbreviation must actually be reachable
    EXPECT_GT(handled, 0u);
    // stockham and its partial-pass variant deliberately share one name, so
    // there are fewer distinct names than handled schemes
    EXPECT_GT(abbreviations.size(), 0u);
    EXPECT_LE(abbreviations.size(), handled);

    // spot-check a couple so a wholesale rename does not slip through
    EXPECT_EQ(PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM), "sbrr");
    EXPECT_EQ(PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM_BLOCK_CC), "sbcc");
    EXPECT_EQ(PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM_BLOCK_CR), "sbcr");
    EXPECT_EQ(PrintKernelSchemeAbbr(CS_KERNEL_STOCKHAM_BLOCK_RC), "sbrc");
}

TEST(rocfft_internal, every_scheme_round_trips_through_its_name)
{
    for(int i = CS_NONE; i <= CS_KERNEL_3D_SINGLE; ++i)
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
