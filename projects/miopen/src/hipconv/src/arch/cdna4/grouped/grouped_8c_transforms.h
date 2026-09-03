#pragma once
namespace grouped_8c_transforms
{
// # Unrolling 1D Convolution into a matrix multiply accelerator
//
// ## Overview
//
// We will define a 1D convolution algorithm that maps to a 16x16x32 MFMA
// instruction. The 1D algorithm naturally applies to the horizontal
// dimension of a conv2d operation, and a "direct convolution" loop over
// rows of the inputs and filters implements the vertical dimension.
//
// ## Algorithm
//
// Define an FIR algorithm that correlates a 3-tap filter 'g' with a 4-element
// data signal 'd' to produce 2 outputs 'q'. Call this algorithm F(2, 3).
//
// d = [d0, d1, d2, d3]
// g = [g0, g1, g2]
// q = [q0, q1]
//
// Because F(2, 3) is a linear transform, one can implement it with matrix multiplication;
// specifically, the Toeplitz matrix
//
//          ⎡ g0  0  ⎤
//     G =  ⎢ g1  g0 ⎥
//          ⎢ g2  g1 ⎥
//          ⎣ 0   g2 ⎦
//
// such that
//
//      q = d G .
//
// Because 1/4th of the matrix entries are zeros, this algorithm only utilizes
// 75% of a matrix multiply accelerator. That is a win if it makes a small
// convolution fill 75% of a large matmul unit. We can accomplish that
// goal by generalizing the F(2, 3) algorithm to grouped convolution with 8 channels per group.
//
// Define every element of d to be vector of 8 input channels,
// every element of q to be a vector of 8 output channels,
// and every element of g to be an 8 x 8 matrix that maps input channels to output channels.
// Then G is a block matrix with total size (4 x 8) x (2 x 8) = 32 x 16,
// d is a row vector of 4 x 8 = 32 elements, and q is a row vector with 2 x 8 = 16 elements.
// and the operation q = d G implements grouped convolution with 8 channels per group.
//
// Let's apply the FIR filter to overlapping 4-element sequences in parallel
// by making each sequence a row of matrix D. Each sequence produces outputs
// in rows of matrix Q according to
//
//       Q = D G .
//
// Let's put 16 input sequences in D. Then D is a 16 x 32 matrix, Q is a 16 x 16 matrix,
// and Q = D G is a 16 x 16 x 32 matrix multiply.
//
// Instruction V_MFMA_F32_16x16x32_F16 implements this operation with a throughput
// of 16 cycles, making it the highest performance FP16 matrix multiply instruction on CDNA4.
// Therefore, Q = D G implements conv2d with 8 channels per group and 75% MFMA utilization.
// In practice, the attainable performance will be limited by memory bandwidth,
// but the arithmetic will use the MFMA efficiently.
//
// ## Implementation
//
// Implement a function that defines the lane mapping for an "unroll weights" function
// that implements block-matrix G defined above for a group size of 8 channels.
//
// Assume the weights tensor for a single group is located in LDS with format
// K(8) x R(3) x S(3) x C(8) with fp16 data-type, where K is the number of output channels,
// R is the height of the filter, S is the width of the filter, and C is the number
// of input channels.
//
// Define the mapping that loads K(8) x R(1) x S(3) x C(8) into G with size 32 x 16.
// Because mfma's B-matrix is stored column-major, it is convenient to think of the transpose of G.
//
//            t=0  t=1  t=2  t=3
//          ⎡ g0   g1   g2   0  ⎤ q=0
//    GT =  ⎣  0   g0   g1   g2 ⎦ q=1
//
//
//   where each element GT(q, s) is an 8x8 block matrix indexed by k (output channel) and c (input
//   channel):
//
//                        c=0     c=1    ...          c=7
//                 k=0 ⎡  (0,0)   (0,1)  ...            :   ⎤
//                 k=1 ⎢  (1,0)   (1,1)  ...            :   ⎥
//    GT(q, t) =    :  ⎢    :       :                   :   ⎥
//                  :  ⎢    :       :                   :   ⎥
//                 k=7 ⎣  (7,0)   (7,1)  ...         (7,7)  ⎦
//
// GT ~ 16 x 32 = [2q x 8k] x [4s x 8c]
struct GT
{
    static constexpr int rows       = 16;
    static constexpr int cols       = 32;
    static constexpr int group_size = 8;

    // The row of the block matrix.
    static constexpr int q(int row) { return row / group_size; }

    // The column of the block matrix
    static constexpr int t(int col) { return col / group_size; }

    // One of 8 output channels.
    static constexpr int k(int row) { return row % group_size; }

    // One of 8 input channels.
    static constexpr int c(int col) { return col % group_size; }

    // One of two sets of four input channels.
    static constexpr int c4(int col4) { return col4 % (group_size / 4); }

    // One of 3 filter taps.
    static constexpr int s(int row, int col4) { return GT::t(col4 * 4) - GT::q(row); }

    static constexpr int filter_is_zero(int row, int col4)
    {
        int ss = GT::s(row, col4);
        return ss < 0 || ss > 2;
    }
};


// Each element of D is a vector of 8 input channels.
// A row of D is four elements corresponding to a tile of inputs.
// Tiles overlap by 2 elements. Let's call each tile p.
//
//                w=0 w=1 w=2 w=3
//          p=0  ⎡ d0  d1  d2  d3 ⎤
//   D =    p=1  ⎢ :   :   :   :  ⎥
//           :   ⎢ :   :   :   :  ⎥
//          p=15 ⎣ d0  d1  d2  d3 ⎦
//
// Each element d0 is a vector of 8 input channels.
//
// D ~ 16 tiles x (4 inputs/tile x 8 channels) = 16 x 32

struct D
{
    static constexpr int rows       = 16;
    static constexpr int cols       = 32;
    static constexpr int group_size = 8;

    static constexpr int p(int row) { return row; }

    static constexpr int w(int col) { return col / group_size; }

    static constexpr int c(int col) { return col % group_size; }

    static constexpr int c4(int col4) { return col4 % (group_size / 4); }
};
} // namespace grouped_8c_transforms
