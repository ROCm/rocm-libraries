// Copyright (C) 2021 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "node_factory.h"
#include "../../shared/arithmetic.h"
#include "../../shared/precision_type.h"
#include "function_pool.h"
#include "fuse_shim.h"
#include "hip/hip_runtime_api.h"
#include "logging.h"
#include "tree_node_1D.h"
#include "tree_node_2D.h"
#include "tree_node_3D.h"
#include "tree_node_bluestein.h"
#include "tree_node_real.h"

#include <functional>
#include <set>
#include <vector>

// TODO:
//   - better data structure, and more elements for non pow of 2
//   - validate corresponding functions existing in function pool or not
//   - SBRC should support un-aligned dim with BWD such as 10752 = 84 x 128(bwd=8)
NodeFactory::Map1DLength const NodeFactory::map1DLengthSingle = {
    // ----------------------------------------------------------
    // pow2 lengths
    // ----------------------------------------------------------
    {8192, 64}, //              CC (64cc + 128rc)
    {16384, 64}, //             CC (64cc + 256rc) // 128x128 no faster
    {32768, 128}, //            CC (128cc + 256rc)
    {65536, 256}, //            CC (256cc + 256rc)
    {131072, 256}, //           CC (256cc + 512rc)
    {262144, 512}, //           CC (512cc + 512rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (4096, 8192)
    // ----------------------------------------------------------
    {4704, 96}, //              CC (96cc + 49rc)
    {4913, 289}, //             CC (289cc + 17rc)
    {5488, 112}, //             CC (112cc + 49rc)
    {6144, 96}, //              CC (96cc + 64rc)
    {6561, 81}, //              CC (81cc + 81rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (8192, 16384)
    // ----------------------------------------------------------
    {9216, 72}, //              CC (72cc + 128rc)
    {10000, 100}, //            CC (100cc + 100rc)
    {10240, 160}, //            CC (160cc + 64rc)
    {10752, 96}, //             CC (96cc + 112rc)
    {11200, 224}, //            CC (224cc + 50rc)
    {12288, 192}, //            CC (192cc + 64rc)
    {15625, 125}, //            CC (125cc + 125rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (16384, 32768)
    // ----------------------------------------------------------
    {16807, 343}, //            CC (343cc + 49rc)
    {17576, 104}, //            CC (104cc + 169rc)
    {18816, 168}, //            CC (168cc + 112rc)
    {19200, 192}, //            CC (192cc + 100rc)
    {19683, 243}, //            CC (243cc + 81rc)
    {20480, 160}, //            CC (160cc + 128rc)
    {21504, 168}, //            CC (168cc + 128rc)
    {21952, 343}, //            CC (343cc + 64rc)
    {23232, 192}, //            CC (192cc + 121rc)
    {24576, 192}, //            CC (192cc + 128rc)
    {26000, 208}, //            CC (208cc + 125rc)
    {28672, 256}, //            CC (256cc + 112rc)
    {32256, 168}, //            CC (168cc + 192rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (32768, 65536)
    // ----------------------------------------------------------
    {34969, 289}, //            CC (289cc + 121rc)
    {36864, 192}, //            CC (192cc + 192rc)
    {38880, 160}, //            CC (160cc + 243rc)
    {40000, 200}, //            CC (200cc + 200rc)
    {40960, 160}, //            CC (160cc + 256rc)
    {43008, 168}, //            CC (168cc + 256rc) // CC (224cc + 192rc)
    {46080, 240}, //            CC (240cc + 192rc)
    {48000, 240}, //            CC (240cc + 200rc)
    {49152, 256}, //            CC (256cc + 192rc)
    {51200, 512}, //            CC (512cc + 100rc)
    {53248, 208}, //            CC (208cc + 256rc)
    {57344, 512}, //            CC (512cc + 112rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (65536, 131072)
    // ----------------------------------------------------------
    {68600, 343}, //            CC (343cc + 200rc)
    {71344, 208}, //            CC (208cc + 343rc)
    {73984, 289}, //            CC (289cc + 256rc)
    {76832, 224}, //            CC (224cc + 343rc)
    {79860, 60}, //             CC (60cc + 1331rc)
    {81920, 160}, //            CC (160cc + 512rc)
    {83521, 289}, //            CC (289cc + 289rc)
    {87808, 343}, //            CC (343cc + 256rc)
    {95832, 72}, //             CC (72cc + 1331rc)
    {98304, 512}, //            CC (512cc + 192rc)
    {102400, 512}, //           CC (512cc + 200rc)
    {106496, 208}, //           CC (208cc + 512rc)
    {110592, 216}, //           CC (216cc + 512rc)
    {114688, 224}, //           CC (224cc + 512rc)
};

NodeFactory::Map1DLength const NodeFactory::map1DLengthDouble = {
    // ----------------------------------------------------------
    // pow2 lengths
    // ----------------------------------------------------------
    {4096, 64}, //              CC (64cc + 64rc)
    {8192, 64}, //              CC (64cc + 128rc)
    {16384, 64}, //             CC (64cc + 256rc) // 128x128 ?
    {32768, 128}, //            CC (128cc + 256rc)
    {65536, 256}, //            CC (256cc + 256rc) // {65536, 64}
    {131072, 256}, //           CC (256cc + 512rc)
    {262144, 512}, //           CC (512cc + 512rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (4096, 8192)
    // ----------------------------------------------------------
    {4704, 96}, //              CC (96cc + 49rc)
    {4913, 289}, //             CC (289cc + 17rc)
    {5488, 112}, //             CC (112cc + 49rc)
    {6144, 96}, //              CC (96cc + 64rc)
    {6561, 81}, //              CC (81cc + 81rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (8192, 16384)
    // ----------------------------------------------------------
    {9216, 72}, //              CC (72cc + 128rc)
    {10000, 100}, //            CC (100cc + 100rc)
    {10240, 160}, //            CC (160cc + 64rc)
    {10752, 96}, //             CC (96cc + 112rc)
    {11200, 224}, //            CC (224cc + 50rc)
    {12288, 192}, //            CC (192cc + 64rc)
    {15625, 125}, //            CC (125cc + 125rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (16384, 32768)
    // ----------------------------------------------------------
    {16807, 343}, //            CC (343cc + 49rc)
    {17576, 104}, //            CC (104cc + 169rc)
    {18816, 168}, //            CC (168cc + 112rc)
    {19200, 192}, //            CC (192cc + 100rc)
    {19683, 243}, //            CC (243cc + 81rc)
    {20480, 160}, //            CC (160cc + 128rc)
    {21504, 168}, //            CC (168cc + 128rc)
    {21952, 343}, //            CC (343cc + 64rc)
    {23232, 192}, //            CC (192cc + 121rc)
    {24576, 192}, //            CC (192cc + 128rc)
    {26000, 208}, //            CC (208cc + 125rc)
    {28672, 256}, //            CC (256cc + 112rc)
    {32256, 168}, //            CC (168cc + 192rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (32768, 65536)
    // ----------------------------------------------------------
    {34969, 289}, //            CC (289cc + 121rc)
    {36864, 192}, //            CC (192cc + 192rc)
    {38880, 160}, //            CC (160cc + 243rc)
    {40000, 200}, //            CC (200cc + 200rc)
    {40960, 160}, //            CC (160cc + 256rc)
    {43008, 168}, //            CC (168cc + 256rc) // or (224cc + 192rc)
    {46080, 240}, //            CC (240cc + 192rc)
    {48000, 240}, //            CC (240cc + 200rc)
    {49152, 256}, //            CC (256cc + 192rc)
    {51200, 512}, //            CC (512cc + 100rc)
    {53248, 208}, //            CC (208cc + 256rc)
    {57344, 512}, //            CC (512cc + 112rc)

    // ----------------------------------------------------------
    // non-pow2 lengths in (65536, 131072)
    // ----------------------------------------------------------
    {68600, 343}, //            CC (343cc + 200rc)
    {71344, 208}, //            CC (208cc + 343rc)
    {76832, 224}, //            CC (224cc + 343rc)
    {78125, 125}, //            CC (125cc + 625rc)
    {79860, 60}, //             CC (60cc + 1331rc)
    {81920, 160}, //            CC (160cc + 512rc)
    {83521, 289}, //            CC (289cc + 289rc)
    {87808, 343}, //            CC (343cc + 256rc)
    {95832, 72}, //             CC (72cc + 1331rc)
    {98304, 512}, //            CC (512cc + 192rc)
    {102400, 512}, //           CC (512cc + 200rc)
    {106496, 208}, //           CC (208cc + 512rc)
    {110592, 216}, //           CC (216cc + 512rc)
    {114688, 224}, //           CC (224cc + 512rc)
};

NodeFactory::Map1DLength const NodeFactory::map1DLengthTRTRT = {
    // 3^18 performs better when decomposing with 3^7 kernel even when
    // 3^8 is available
    {387420489, 177147},
};

//
// Factorisation helpers
//

// Return true if the order of factors in a decomposition should be
// reversed.  This improves performance for some lengths.
inline bool reverse_factors(size_t length)
{
    std::set<size_t> reverse_factors_lengths = {32256, 43008};
    return reverse_factors_lengths.count(length) == 1;
}

// Search function pool for length where is_supported_factor(length) returns true.
inline size_t search_pool(const function_pool&               pool,
                          rocfft_precision                   precision,
                          size_t                             length,
                          const std::function<bool(size_t)>& is_supported_factor)
{
    // query supported lengths from function pool, largest to smallest
    auto supported  = pool.get_lengths(precision, CS_KERNEL_STOCKHAM);
    auto comparison = std::greater<size_t>();
    std::sort(supported.begin(), supported.end(), comparison);

    if(supported.empty())
        return 0;

    // start search slightly smaller than sqrt(length)
    auto v     = (size_t)sqrt(length);
    auto lower = std::lower_bound(supported.cbegin(), supported.cend(), v, comparison);
    if(*lower < sqrt(length) && lower != supported.cbegin())
        lower--;

    auto upper = supported.cend();

    // search!
    auto itr = std::find_if(lower, upper, is_supported_factor);
    if(itr != supported.cend())
        return *itr;

    return 0;
}

// Return largest factor that has BOTH functions in the pool.
inline size_t get_explicitly_supported_factor(const function_pool& pool,
                                              rocfft_precision     precision,
                                              size_t               length)
{
    auto supported_factor = [length, precision, &pool](size_t factor) -> bool {
        bool is_factor        = length % factor == 0;
        bool has_other_kernel = pool.has_function(FMKey(length / factor, precision));
        return is_factor && has_other_kernel;
    };
    auto factor = search_pool(pool, precision, length, supported_factor);
    if(factor > 0 && reverse_factors(length))
        return length / factor;
    return factor;
}

// Return largest factor that has a function in the pool.
inline size_t get_largest_supported_factor(const function_pool& pool,
                                           rocfft_precision     precision,
                                           size_t               length)
{
    auto supported_factor = [length](size_t factor) -> bool {
        bool is_factor = length % factor == 0;
        return is_factor;
    };
    return search_pool(pool, precision, length, supported_factor);
}

bool NodeFactory::Large1DLengthsValid(const function_pool&            pool,
                                      const NodeFactory::Map1DLength& map1DLength,
                                      rocfft_precision                precision)
{
    for(const auto& pair : map1DLength)
    {
        if(pair.first % pair.second != 0)
            return false;

        if(!pool.has_SBCC_kernel(pair.second, precision))
            return false;

        if(!pool.has_SBRC_kernel(pair.first / pair.second, precision))
            return false;
    }

    return true;
}

// helper to check 1d length maps at most once per process
bool NodeFactory::CheckLarge1DMaps(const function_pool& pool)
{
    static bool singleValid = NodeFactory::Large1DLengthsValid(
        pool, NodeFactory::map1DLengthSingle, rocfft_precision_single);
    static bool doubleValid = NodeFactory::Large1DLengthsValid(
        pool, NodeFactory::map1DLengthDouble, rocfft_precision_double);
    return singleValid && doubleValid;
}

// Checks whether the non-pow2 length input is supported for a Bluestein compute scheme
bool NodeFactory::NonPow2LengthSupported(const function_pool& pool,
                                         rocfft_precision     precision,
                                         size_t               length)
{
    // assume half precision behaves the same as single
    if(precision == rocfft_precision_half)
        precision = rocfft_precision_single;

    // Exceptions which have been found to perform poorly when compared to the next pow2 length
    static const std::map<rocfft_precision, std::set<size_t>> length_exceptions
        = {{rocfft_precision_single,
            {224, 2160, 2430, 2880, 3456, 21504, 21952, 23232, 79860, 95832, 110592}},
           {rocfft_precision_double,
            {104,  108,  180,  224,  225,  432,  450,  810,  2401,  2430,  2700,  2880,   3125,
             3200, 3240, 3375, 3456, 3600, 3645, 4913, 6561, 11200, 53248, 57344, 106496, 114688}}};

    if(length_excepted(length_exceptions, precision, length))
        return false;

    // Look for regular Stockham kernels support
    if(pool.has_function(FMKey(length, precision)))
        return true;

    assert(CheckLarge1DMaps(pool));

    // and for supported block CC + RC Stockham decompositions
    if(precision == rocfft_precision_single
       && (map1DLengthSingle.find(length) != map1DLengthSingle.end()))
        return true;
    if(precision == rocfft_precision_double
       && (map1DLengthDouble.find(length) != map1DLengthDouble.end()))
        return true;

    return false;
}

size_t NodeFactory::GetBluesteinLength(const function_pool& pool,
                                       rocfft_precision     precision,
                                       size_t               len)
{
    return BluesteinNode::FindBlue(
        pool, len, precision, BluesteinSingleNode::SizeFits(pool, len, precision));
}

bool NodeFactory::SupportedLength(const function_pool& pool, rocfft_precision precision, size_t len)
{
    // do we have an explicit kernel?
    if(pool.has_function(FMKey(len, precision)))
        return true;

    // can we factor with using only base radix?
    size_t p = len;
    while(!(p % 2))
        p /= 2;
    while(!(p % 3))
        p /= 3;
    while(!(p % 5))
        p /= 5;
    while(!(p % 7))
        p /= 7;
    while(!(p % 11))
        p /= 11;
    while(!(p % 13))
        p /= 13;
    while(!(p % 17))
        p /= 17;

    if(p == 1)
        return true;

    // do we have an explicit kernel for the remainder?
    if(pool.has_function(FMKey(p, precision)))
        return true;

    // finally, can we factor this length with combinations of existing kernels?
    if(get_explicitly_supported_factor(pool, precision, len) > 0)
        return true;

    return false;
}

inline void PrintFailInfo(rocfft_precision precision,
                          size_t           length,
                          ComputeScheme    scheme,
                          size_t           kernelLength = 0,
                          ComputeScheme    kernelScheme = CS_NONE)
{
    rocfft_cerr << "Failed on Node: length " << length << " (" << precision << "): "
                << "when attempting Scheme: " << PrintScheme(scheme) << std::endl;
    if(kernelScheme != CS_NONE)
        rocfft_cerr << "\tCouldn't find the kernel of length " << kernelLength << ", with type "
                    << PrintScheme(kernelScheme) << std::endl;
}

std::unique_ptr<TreeNode> NodeFactory::CreateNodeFromScheme(ComputeScheme s, TreeNode* parent)
{
    switch(s)
    {
    // Internal Node
    case CS_REAL_TRANSFORM_USING_CMPLX:
        return std::unique_ptr<RealTransCmplxNode>(new RealTransCmplxNode(parent));
    case CS_REAL_TRANSFORM_EVEN:
        return std::unique_ptr<RealTransEvenNode>(new RealTransEvenNode(parent));
    case CS_REAL_2D_EVEN:
        return std::unique_ptr<Real2DEvenNode>(new Real2DEvenNode(parent));
    case CS_REAL_3D_EVEN:
        return std::unique_ptr<Real3DEvenNode>(new Real3DEvenNode(parent));
    case CS_BLUESTEIN:
        return std::unique_ptr<BluesteinNode>(new BluesteinNode(parent));
    case CS_L1D_TRTRT:
        return std::unique_ptr<TRTRT1DNode>(new TRTRT1DNode(parent));
    case CS_L1D_CC:
        return std::unique_ptr<CC1DNode>(new CC1DNode(parent));
    case CS_L1D_CRT:
        return std::unique_ptr<CRT1DNode>(new CRT1DNode(parent));
    case CS_2D_RTRT:
        return std::unique_ptr<RTRT2DNode>(new RTRT2DNode(parent));
    case CS_2D_RC:
        return std::unique_ptr<RC2DNode>(new RC2DNode(parent));
    case CS_3D_RTRT:
        return std::unique_ptr<RTRT3DNode>(new RTRT3DNode(parent));
    case CS_3D_TRTRTR:
        return std::unique_ptr<TRTRTR3DNode>(new TRTRTR3DNode(parent));
    case CS_3D_BLOCK_RC:
        return std::unique_ptr<BLOCKRC3DNode>(new BLOCKRC3DNode(parent));
    case CS_3D_BLOCK_CR:
        return std::unique_ptr<BLOCKCR3DNode>(new BLOCKCR3DNode(parent));
    case CS_3D_RC:
        return std::unique_ptr<RC3DNode>(new RC3DNode(parent));
    case CS_3D_PP:
        return std::unique_ptr<PP3DNode>(new PP3DNode(parent));

    // Leaf Node that need to check external kernel file
    case CS_KERNEL_STOCKHAM:
        return std::unique_ptr<Stockham1DNode>(new Stockham1DNode(parent, s));
    case CS_KERNEL_STOCKHAM_PP:
        return std::unique_ptr<StockhamPP1DNode>(new StockhamPP1DNode(parent, s));
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
        return std::unique_ptr<SBCCNode>(new SBCCNode(parent, s));
    case CS_KERNEL_STOCKHAM_PP_BLOCK_CC:
        return std::unique_ptr<SBCCPPNode>(new SBCCPPNode(parent, s));
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
        return std::unique_ptr<SBRCNode>(new SBRCNode(parent, s));
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
        return std::unique_ptr<SBCRNode>(new SBCRNode(parent, s));
    case CS_KERNEL_2D_SINGLE:
        return std::unique_ptr<Single2DNode>(new Single2DNode(parent, s));
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
        return std::unique_ptr<SBRCTransXY_ZNode>(new SBRCTransXY_ZNode(parent, s));
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
        return std::unique_ptr<SBRCTransZ_XYNode>(new SBRCTransZ_XYNode(parent, s));
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
        return std::unique_ptr<RealCmplxTransZ_XYNode>(new RealCmplxTransZ_XYNode(parent, s));

    // Leaf Node that doesn't need to check external kernel file
    case CS_KERNEL_R_TO_CMPLX:
    case CS_KERNEL_R_TO_CMPLX_TRANSPOSE:
    case CS_KERNEL_CMPLX_TO_R:
    case CS_KERNEL_TRANSPOSE_CMPLX_TO_R:
        return std::unique_ptr<PrePostKernelNode>(new PrePostKernelNode(parent, s));
    case CS_KERNEL_TRANSPOSE:
    case CS_KERNEL_TRANSPOSE_XY_Z:
    case CS_KERNEL_TRANSPOSE_Z_XY:
        return std::unique_ptr<TransposeNode>(new TransposeNode(parent, s));
    case CS_KERNEL_COPY_R_TO_CMPLX:
    case CS_KERNEL_COPY_HERM_TO_CMPLX:
    case CS_KERNEL_COPY_CMPLX_TO_HERM:
    case CS_KERNEL_COPY_CMPLX_TO_R:
        return std::unique_ptr<RealTransDataCopyNode>(new RealTransDataCopyNode(parent, s));
    case CS_KERNEL_CHIRP:
    case CS_KERNEL_PAD_MUL:
    case CS_KERNEL_FFT_MUL:
    case CS_KERNEL_RES_MUL:
        return std::unique_ptr<BluesteinComponentNode>(new BluesteinComponentNode(parent, s));
    case CS_KERNEL_BLUESTEIN_SINGLE:
        return std::unique_ptr<BluesteinSingleNode>(new BluesteinSingleNode(parent, s));
    default:
        throw std::runtime_error("Scheme assertion failed, node not implemented:" + PrintScheme(s));
        return nullptr;
    }
}

std::unique_ptr<TreeNode> NodeFactory::CreateExplicitNode(NodeMetaData& nodeData,
                                                          TreeNode*     parent,
                                                          ComputeScheme determined_scheme)
{
    function_pool pool{nodeData.deviceProp};

    // when creating tree from solution map, scheme is L1D but not root
    // NB:
    //   Why we need this:
    //   Ideally, decide-scheme functions don't assign/change any lengths data,
    //   it should be assigned before. But that is not the case for L1D, when
    //   deciding L1D scheme, it appends an extra temporary length indicating
    //   how to "factorize" the large 1D, and then pop in later in "build-tree."
    //   But when we are creating tree from solution map, we already have the
    //   determined_scheme; however, if we don't do the decide-node-scheme,
    //   we lose that factor-length, and the later "pop" casues error. So
    //   we should still call the decide-node-scheme here as long as we know
    //   it's a L1D. (But not for the root-node, root-node calls the decide
    //   function anyway, before we try looking up solutions)
    if((determined_scheme == CS_L1D_TRTRT || determined_scheme == CS_L1D_CC
        || determined_scheme == CS_L1D_CRT)
       && (parent != nullptr))
    {
        auto s = DecideNodeScheme(pool, nodeData, parent);
        if(determined_scheme != s)
            throw std::runtime_error("solution map error for L1D sub-problem");
    }

    // createing tree without solution map, must call DecideNodeScheme
    if(determined_scheme == CS_NONE)
        determined_scheme = DecideNodeScheme(pool, nodeData, parent);

    // check if successfully created
    if(determined_scheme == CS_NONE)
        throw std::runtime_error("DecideNodeScheme Failed!: CS_NONE");

    auto node = CreateNodeFromScheme(determined_scheme, parent);
    node->CopyNodeData(nodeData);
    return node;
}

// FuseShim Creator
std::unique_ptr<FuseShim> NodeFactory::CreateFuseShim(FuseType                      type,
                                                      const std::vector<TreeNode*>& components)
{
    switch(type)
    {
    case FT_TRANS_WITH_STOCKHAM:
        return std::unique_ptr<TRFuseShim>(new TRFuseShim(components, type));
    case FT_STOCKHAM_WITH_TRANS:
        return std::unique_ptr<RTFuseShim>(new RTFuseShim(components, type));
    case FT_STOCKHAM_WITH_TRANS_Z_XY:
        return std::unique_ptr<RT_ZXY_FuseShim>(new RT_ZXY_FuseShim(components, type));
    case FT_STOCKHAM_WITH_TRANS_XY_Z:
        return std::unique_ptr<RT_XYZ_FuseShim>(new RT_XYZ_FuseShim(components, type));
    case FT_R2C_TRANSPOSE:
        return std::unique_ptr<R2CTrans_FuseShim>(new R2CTrans_FuseShim(components, type));
    case FT_TRANSPOSE_C2R:
        return std::unique_ptr<TransC2R_FuseShim>(new TransC2R_FuseShim(components, type));
    case FT_STOCKHAM_R2C_TRANSPOSE:
        return std::unique_ptr<STK_R2CTrans_FuseShim>(new STK_R2CTrans_FuseShim(components, type));
    default:
        throw std::runtime_error("FuseType assertion failed, type not implemented");
        return nullptr;
    }
}

ComputeScheme NodeFactory::DecideNodeScheme(const function_pool& pool,
                                            NodeMetaData&        nodeData,
                                            TreeNode*            parent)
{
    if((parent == nullptr)
       && ((nodeData.inArrayType == rocfft_array_type_real)
           || (nodeData.outArrayType == rocfft_array_type_real)))
    {
        return DecideRealScheme(pool, nodeData);
    }

    switch(nodeData.dimension)
    {
    case 1:
        return Decide1DScheme(pool, nodeData, parent);
    case 2:
        return Decide2DScheme(pool, nodeData);
    case 3:
        return Decide3DScheme(pool, nodeData);
    default:
        throw std::runtime_error("Invalid dimension");
    }

    return CS_NONE;
}

ComputeScheme NodeFactory::DecideRealScheme(const function_pool& pool, NodeMetaData& nodeData)
{
    // use size in real units to decide what scheme to use
    const auto& realLength = nodeData.direction == -1 ? nodeData.length : nodeData.outputLength;

    if(realLength[0] % 2 == 0 && nodeData.inStride[0] == 1 && nodeData.outStride[0] == 1)
    {
        switch(nodeData.dimension)
        {
        case 1:
            return CS_REAL_TRANSFORM_EVEN;
        case 2:
            return CS_REAL_2D_EVEN;
        case 3:
            return CS_REAL_3D_EVEN;
        default:
            throw std::runtime_error("Invalid dimension");
        }
    }
    // Fallback method
    return CS_REAL_TRANSFORM_USING_CMPLX;
}

ComputeScheme
    NodeFactory::Decide1DScheme(const function_pool& pool, NodeMetaData& nodeData, TreeNode* parent)
{
    ComputeScheme scheme = CS_NONE;

    // Build a node for a 1D FFT
    if(!SupportedLength(pool, nodeData.precision, nodeData.length[0]))
        return CS_BLUESTEIN;

    if(pool.has_function(FMKey(nodeData.length[0], nodeData.precision)))
    {
        // 2-kernel plans for lengths > 4k can still do better at
        // smaller batch size due to using more CUs.  So prefer
        // single-kernel only if we launch enough workgroups for all
        // CUs
        if(nodeData.length[0] > 4096)
        {
            auto kernel = pool.get_kernel(FMKey(nodeData.length[0], nodeData.precision));

            // higher dimensions and batch is the effective batch for
            // the kernel
            const auto totalBatch
                = product(nodeData.length.begin() + 1, nodeData.length.end()) * nodeData.batch;

            // Bluestein would have chosen this kernel for
            // single-kernel Bluestein, so continue using it for
            // chirp setup
            if((parent && parent->scheme == CS_BLUESTEIN)
               || totalBatch / kernel.transforms_per_block
                      >= static_cast<size_t>(pool.deviceProp->multiProcessorCount))
                return CS_KERNEL_STOCKHAM;
            // otherwise, fall through to multi-kernel plan
        }
        else
        {
            return CS_KERNEL_STOCKHAM;
        }
    }

    size_t divLength1 = 1;
    bool   failed     = false;

    if(IsPo2(nodeData.length[0])) // multiple kernels involving transpose
    {
        // TODO: wrap the below into a function and check with LDS size
        size_t block_threshold = 262144;
        if(nodeData.length[0] <= block_threshold)
        {
            // Enable block compute under these conditions
            if(nodeData.precision == rocfft_precision_single
               || nodeData.precision == rocfft_precision_half)
            {
                if(map1DLengthSingle.find(nodeData.length[0]) != map1DLengthSingle.end())
                {
                    divLength1 = map1DLengthSingle.at(nodeData.length[0]);
                }
                else
                {
                    failed = true;
                }
            }
            else
            {
                if(map1DLengthDouble.find(nodeData.length[0]) != map1DLengthDouble.end())
                {
                    divLength1 = map1DLengthDouble.at(nodeData.length[0]);
                }
                else
                {
                    failed = true;
                }
            }
            // for gfx906, 512 CC/RC isn't as fast, so use CRT
            // with a nicer length
            if(is_device_gcn_arch(nodeData.deviceProp, "gfx906") && nodeData.length[0] == 262144)
            {
                divLength1 = 64;
                scheme     = CS_L1D_CRT;
            }
            else
            {
                scheme = CS_L1D_CC;
            }
        }
        else
        {
            // get largest pow2 1D length
            auto largest = pool.get_largest_pow2_length(nodeData.precision);

            // need to ignore len 1, or we're going into a infinity decompostion loop
            // basically not gonna happen unless someone builds only a len1 kernel...
            if(largest <= 1)
            {
                failed = true;
            }
            else if(nodeData.length[0] > largest * largest)
            {
                divLength1 = nodeData.length[0] / largest;
            }
            else
            {
                size_t in_x = 0;
                size_t len  = nodeData.length[0];
                while(len != 1)
                {
                    len >>= 1;
                    in_x++;
                }
                in_x /= 2;
                divLength1 = (size_t)1 << in_x;
            }
            scheme = CS_L1D_TRTRT;
        }
    }
    else // if not Pow2
    {
        if(nodeData.precision == rocfft_precision_single
           || nodeData.precision == rocfft_precision_half)
        {
            if(map1DLengthSingle.find(nodeData.length[0]) != map1DLengthSingle.end())
            {
                divLength1 = map1DLengthSingle.at(nodeData.length[0]);
                scheme     = CS_L1D_CC;
            }
            else
            {
                failed = true;
            }
        }
        else if(nodeData.precision == rocfft_precision_double)
        {
            if(map1DLengthDouble.find(nodeData.length[0]) != map1DLengthDouble.end())
            {
                divLength1 = map1DLengthDouble.at(nodeData.length[0]);
                scheme     = CS_L1D_CC;

                // hack for special case of 43008. On gfx90a, 224 is better
                if(nodeData.length[0] == 43008 && is_device_gcn_arch(nodeData.deviceProp, "gfx90a"))
                {
                    divLength1 = 224;
                }
            }
            else
            {
                failed = true;
            }
        }

        if(failed)
        {
            scheme = CS_L1D_TRTRT;

            auto it = map1DLengthTRTRT.find(nodeData.length[0]);
            if(it != map1DLengthTRTRT.end())
            {
                divLength1 = it->second;
                failed     = false;
            }
            else
            {
                divLength1
                    = get_explicitly_supported_factor(pool, nodeData.precision, nodeData.length[0]);
                if(divLength1 == 0)
                {
                    // We need to recurse.  Note, for CS_L1D_TRTRT,
                    // divLength0 has to be explictly supported
                    auto divLength0 = get_largest_supported_factor(
                        pool, nodeData.precision, nodeData.length[0]);

                    // should ignore factor 1 or we're going into a infinity decompostion loop,
                    // (an example is to run len-81 when we build only pow2 kernels, we'll be here)
                    divLength1 = (divLength0 <= 1) ? 0 : nodeData.length[0] / divLength0;
                }
                failed = divLength1 == 0;
            }
        }
    }

    if(failed)
    {
        // can't find the length in map1DLengthSingle/Double.
        PrintFailInfo(nodeData.precision, nodeData.length[0], scheme);
        return CS_NONE;
    }

    // NOTE: we temporarily save the divLength1 at the end of length vector
    // and then get and pop later when building node
    // size_t divLength0 = length[0] / divLength1;
    nodeData.length.emplace_back(divLength1);

    return scheme;
}

ComputeScheme NodeFactory::Decide2DScheme(const function_pool& pool, NodeMetaData& nodeData)
{
    // First choice is 2D_SINGLE kernel, if the problem will fit into LDS.
    // Next best is CS_2D_RC. Last resort is RTRT.
    if(use_CS_2D_SINGLE(pool,
                        nodeData,
                        rocfft_array_type_complex_interleaved,
                        rocfft_array_type_complex_interleaved))
        return CS_KERNEL_2D_SINGLE; // the node has all build info
    else if(use_CS_2D_RC(pool, nodeData))
        return CS_2D_RC;
    else
        return CS_2D_RTRT;
}

// check if we want to use SBCR solution
static bool Apply_SBCR(const function_pool& pool, NodeMetaData& nodeData)
{
    // NB:
    //   We enable SBCR for limited problem sizes in kernel-generator.py.
    //   Will enable it for non-unit stride cases later.
    return (((is_device_gcn_arch(nodeData.deviceProp, "gfx908")
              || is_device_gcn_arch(nodeData.deviceProp, "gfx90a"))
             && pool.has_SBCR_kernel(nodeData.length[0], nodeData.precision)
             && pool.has_SBCR_kernel(nodeData.length[1], nodeData.precision)
             && pool.has_SBCR_kernel(nodeData.length[2], nodeData.precision)
             && (nodeData.placement == rocfft_placement_notinplace)
             && (nodeData.inStride[0] == 1 && nodeData.outStride[0] == 1 // unit strides
                 && nodeData.inStride[1] == nodeData.length[0]
                 && nodeData.outStride[1] == nodeData.length[0]
                 && nodeData.inStride[2] == nodeData.inStride[1] * nodeData.length[1]
                 && nodeData.outStride[2] == nodeData.outStride[1] * nodeData.length[1])));
}

ComputeScheme NodeFactory::Decide3DScheme(const function_pool& pool, NodeMetaData& nodeData)
{
    // this flag can be enabled when generator can do block column fft in
    // multi-dimension cases and small 2d, 3d within one kernel
    bool MultiDimFuseKernelsAvailable = false;

    if(use_CS_3D_PP(pool, nodeData)) // try 2 partial-pass kernels first
    {
        return CS_3D_PP;
    }
    else if(Apply_SBCR(pool, nodeData)) // try 3 kernels next
    {
        return CS_3D_BLOCK_CR;
    }
    else if(use_CS_3D_RC(pool, nodeData))
    {
        return CS_3D_RC;
    }
    else if(MultiDimFuseKernelsAvailable)
    {
        // conditions to choose which scheme
        if((nodeData.length[0] * nodeData.length[1] * nodeData.length[2]) <= 2048)
            return CS_KERNEL_3D_SINGLE;
        else if(nodeData.length[2] <= 256)
            return CS_3D_RC;
        else
            return CS_3D_RTRT;
    }
    else
    {
        // if we can get down to 3 or 4 kernels via SBRC, prefer that
        if(use_CS_3D_BLOCK_RC(pool, nodeData))
            return CS_3D_BLOCK_RC;

        // else, 3D_RTRT
        // NB:
        // Peek the 1st child but not really add it in.
        // Give up if 1st child is 2D_RTRT (means the poor RTRT_TRT)
        // Switch to TRTRTR as the last resort.
        NodeMetaData child0 = nodeData;
        child0.length       = nodeData.length;
        child0.dimension    = 2;
        auto childScheme    = DecideNodeScheme(pool, child0, nullptr);

        // TODO: investigate those SBCC kernels (84,108,112,168)
        //       in 3D C2C transforms, using 3D_RTRT (2D_RC + TRT) is slower than
        //       using 3D_TRTRTR + (BufAssign & FuseShim), the fused TRFuse are faster (3~4 kernels)
        //       (Nothing to do with Real3D. For Real3DEven, using inplace sbcc are still faster)
        std::map<rocfft_precision, std::set<size_t>> exceptions
            = {{rocfft_precision_single, {84, 112, 168}},
               {rocfft_precision_double, {84, 108, 112, 168}}};
        if(childScheme == CS_2D_RC
           && length_excepted(exceptions, nodeData.precision, nodeData.length[1])
           && nodeData.rootIsC2C)
        {
            return CS_3D_TRTRTR;
        }

        if(childScheme == CS_2D_RTRT)
        {
            return CS_3D_TRTRTR;
        }

        return CS_3D_RTRT;
    }
    // TODO: CS_KERNEL_3D_SINGLE?
}

bool NodeFactory::use_CS_2D_SINGLE(const function_pool& pool,
                                   NodeMetaData&        nodeData,
                                   rocfft_array_type    inArrayType,
                                   rocfft_array_type    outArrayType)
{
    if(!pool.has_function(
           FMKey(nodeData.length[0], nodeData.length[1], nodeData.precision, CS_KERNEL_2D_SINGLE)))
        return false;

    // Get actual LDS size, to check if we can run a 2D_SINGLE
    // kernel that will fit the problem into LDS.
    auto ldsSize = nodeData.deviceProp.sharedMemPerBlock;

    auto kernel = pool.get_kernel(
        FMKey(nodeData.length[0], nodeData.length[1], nodeData.precision, CS_KERNEL_2D_SINGLE));

    auto length0 = nodeData.length[0];
    auto length1 = nodeData.length[1];

    // For larger LDS, account properly for real-complex usage and
    // aim for occupancy-2
    if(ldsSize > 65536)
    {
        // Real-forward transform adds an extra element on fastest length
        // for post-processing
        if(inArrayType == rocfft_array_type_real
           && outArrayType == rocfft_array_type_hermitian_interleaved)
            ++length0;
        // real-inverse transforms adds an extra element on second-fastest length for pre-processing
        else if(inArrayType == rocfft_array_type_hermitian_interleaved
                && outArrayType == rocfft_array_type_real)
            ++length1;

        auto ldsUsage = length0 * length1 * kernel.transforms_per_block
                        * complex_type_size(nodeData.precision);

        if(ldsUsage > ldsSize / 2)
            return false;
    }
    // For default (64KiB) LDS, just account for 2D data size and
    // apply a fudge factor to get good-enough occupancy.  Ideally we
    // can use the above heuristic everywhere but some tuned
    // 2D_SINGLE solutions are faster despite being occupancy-1.
    else
    {
        auto ldsUsage = length0 * length1 * kernel.transforms_per_block
                        * complex_type_size(nodeData.precision);
        if(1.5 * ldsUsage > ldsSize)
            return false;
    }

    return true;
}

bool NodeFactory::use_CS_2D_RC(const function_pool& pool, NodeMetaData& nodeData)
{
    // Do not allow SBCC for (192,y) problems, not the
    // fastest compute scheme for this configuration.
    if(nodeData.length[1] == 192)
        return false;
    else if(pool.has_SBCC_kernel(nodeData.length[1], nodeData.precision))
        return nodeData.length[0] >= 56;

    return false;
}

size_t NodeFactory::count_3D_SBRC_nodes(const function_pool& pool, NodeMetaData& nodeData)
{
    size_t sbrc_dimensions = 0;
    for(unsigned int i = 0; i < nodeData.length.size(); ++i)
    {
        if(pool.has_SBRC_kernel(nodeData.length[i], nodeData.precision))
        {
            // make sure the SBRC kernel on that dimension would be tile-aligned
            auto kernel = pool.get_kernel(FMKey(
                nodeData.length[i], nodeData.precision, CS_KERNEL_STOCKHAM_BLOCK_RC, TILE_ALIGNED));
            if(nodeData.length[(i + 2) % nodeData.length.size()] % kernel.transforms_per_block == 0)
                ++sbrc_dimensions;
        }
    }
    return sbrc_dimensions;
}

bool NodeFactory::use_CS_3D_BLOCK_RC(const function_pool& pool, NodeMetaData& nodeData)
{
    // TODO: SBRC hasn't worked for inner batch (i/oDist == 1)
    if(nodeData.iDist == 1 || nodeData.oDist == 1)
        return false;

    return count_3D_SBRC_nodes(pool, nodeData) >= 2;
}

bool NodeFactory::use_CS_3D_RC(const function_pool& pool, NodeMetaData& nodeData)
{
    // TODO: SBCC hasn't worked for inner batch (i/oDist == 1)
    if(nodeData.iDist == 1 || nodeData.oDist == 1)
        return false;

    // Peek the first child
    // Give up if 1st child is 2D_RTRT (means the poor RTRT_C),
    NodeMetaData child0 = nodeData;
    child0.length       = nodeData.length;
    child0.dimension    = 2;
    auto childScheme    = DecideNodeScheme(pool, child0, nullptr);

    // if first 2 dimensions can be handled with 2D_SINGLE, just run
    // with this 2-kernel plan.
    if(childScheme == CS_KERNEL_2D_SINGLE)
        return true;

    FMKey key(nodeData.length[2], nodeData.precision, CS_KERNEL_STOCKHAM_BLOCK_CC);
    if(!pool.has_function(key))
        return false;

    // Check the C part.
    // The first R is built recursively with 2D_FFT, leave the check part to themselves
    auto kernel = pool.get_kernel(key);

    // hack for this special case
    // this size is rejected by the following conservative threshold (#-elems)
    // however it can use 3D_RC and get much better performance
    std::vector<size_t> special_case{56, 336, 336};
    if(nodeData.length == special_case && nodeData.precision == rocfft_precision_double)
        return true;

    // x-dim should be >= the blockwidth, or it might perform worse..
    if(nodeData.length[0] < kernel.transforms_per_block)
        return false;

    // we don't want a too-large 3D block, sbcc along z-dim might be bad
    if((nodeData.length[0] * nodeData.length[1] * nodeData.length[2]) >= (128 * 128 * 128))
        return false;

    if(childScheme == CS_2D_RTRT)
        return false;

    // if we are here, the 2D scheme must be 2D_RC (3 kernels total)
    assert(childScheme == CS_2D_RC);
    return true;

    return false;
}

bool NodeFactory::use_CS_3D_PP(const function_pool& pool, NodeMetaData& nodeData)
{
    if(!pool.has_function(

           PPFMKey(nodeData.length[0],
                   nodeData.length[1],
                   nodeData.length[2],
                   nodeData.precision,
                   CS_3D_PP)))
        return false;

    // Partial pass is currently restricted to large enough batch sizes,
    // unite stride, interleaved FFTs.
    bool batchCondition = (nodeData.batch >= 5);

    size_t checkDist     = product(nodeData.length.begin(), nodeData.length.end());
    bool   distCondition = (nodeData.iDist == checkDist && nodeData.oDist == checkDist);

    bool strideCondition = ((nodeData.inStride.size() && nodeData.inStride[0]) == 1
                            && (nodeData.outStride.size() && nodeData.outStride[0]) == 1);

    bool arrayTypeCondition = (nodeData.inArrayType != rocfft_array_type_complex_planar
                               && nodeData.outArrayType != rocfft_array_type_complex_planar);

    return (batchCondition && distCondition && strideCondition && arrayTypeCondition);
}
