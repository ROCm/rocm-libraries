/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "GPUContextFixture.hpp"
#include "GenericContextFixture.hpp"

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/CoordinateGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Dimension.hpp>
#include <rocRoller/KernelGraph/CoordinateGraph/Transformer.hpp>
#include <rocRoller/Serialization/Variant.hpp>

using namespace rocRoller;
using namespace KernelGraph::CoordinateGraph;

namespace rocRollerTest
{
    class CoordinateGraphTest : public GenericContextFixture
    {
    public:
        Expression::FastArithmetic fastArith{m_context};

        void SetUp() override
        {
            GenericContextFixture::SetUp();
            fastArith = Expression::FastArithmetic(m_context);
        }
    };

    TEST_F(CoordinateGraphTest, SerializationTypePath)
    {
        CoordinateGraph::Element el = User{};

        EXPECT_EQ("Node.User", Serialization::typePath(el));

        el = Flatten{};

        EXPECT_EQ("Edge.Transform.Flatten", Serialization::typePath(el));

        el = Adhoc{"BlockNumber"};

        EXPECT_EQ("Node.Adhoc.BlockNumber", Serialization::typePath(el));

        {
            auto [a, b] = Serialization::splitFirst(".", "Edge.Transform.Flatten");
            EXPECT_EQ("Edge", a);
            EXPECT_EQ("Transform.Flatten", b);
        }

        {
            auto [a, b] = Serialization::splitFirst(".", "Flatten");
            EXPECT_EQ("Flatten", a);
            EXPECT_EQ("", b);
        }
    }

    TEST_F(CoordinateGraphTest, EdgeType)
    {
        auto ct = CoordinateGraph();

        auto size    = std::make_shared<Expression::Expression>(64u);
        auto unit    = std::make_shared<Expression::Expression>(1u);
        auto x_index = std::make_shared<Expression::Expression>(5u);
        auto y_index = std::make_shared<Expression::Expression>(3u);

        auto x = ct.addElement(SubDimension(0, size, unit));
        auto y = ct.addElement(SubDimension(0, size, unit));
        auto m = ct.addElement(SubDimension(0, size * size, unit));

        auto flatten_id = ct.addElement(Flatten{}, {x, y}, {m});

        EXPECT_EQ(EdgeType::CoordinateTransform, ct.getEdgeType(flatten_id));
        EXPECT_EQ(EdgeType::None, ct.getEdgeType(x));
        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({x, y, m}));
    }

    TEST_F(CoordinateGraphTest, Basic)
    {
        auto ct = CoordinateGraph();

        auto size    = std::make_shared<Expression::Expression>(64u);
        auto unit    = std::make_shared<Expression::Expression>(1u);
        auto x_index = std::make_shared<Expression::Expression>(5u);
        auto y_index = std::make_shared<Expression::Expression>(3u);

        auto x = ct.addElement(SubDimension(0, size, unit));
        auto y = ct.addElement(SubDimension(0, size, unit));
        auto m = ct.addElement(SubDimension(0, size * size, unit));

        auto flatten_id = ct.addElement(Flatten{}, {x, y}, {m});

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({x, y, m}));

        auto exprs = ct.forward({x_index, y_index}, {x, y}, {m}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "{Flatten: Add(Multiply(5:U32, 64:U32)U32, 3:U32)U32}");

        exprs = ct.forward({x_index, y_index}, {x, y}, {m}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "323:U32");

        // Trivial
        auto zero = std::make_shared<Expression::Expression>(0u);
        exprs     = ct.forward({zero, zero}, {x, y}, {m}, nullptr);
        EXPECT_EQ(1, exprs.size());
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "{Flatten: Add(Multiply(0:U32, 64:U32)U32, 0:U32)U32}");
        exprs = ct.forward({zero, zero}, {x, y}, {m}, fastArith);
        EXPECT_EQ(1, exprs.size());
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "0:U32");
        auto result = std::get<unsigned int>(Expression::evaluate(exprs[0]));
        EXPECT_EQ(result, 0);

        EXPECT_NE(std::vector<int>(),
                  ct.path<Graph::Direction::Upstream>(std::vector<int>{m}, std::vector<int>{x})
                      .to<std::vector>());

        auto m_index = std::make_shared<Expression::Expression>(67u);
        exprs        = ct.reverse({m_index}, {x}, {m},
                           fastArith); // note 'y' isn't necessary in reverse
        result       = std::get<unsigned int>(Expression::evaluate(exprs[0]));
        EXPECT_EQ(result, 1);

        exprs  = ct.reverse({m_index}, {y}, {m}, fastArith); // note 'x' isn't necessary in reverse
        result = std::get<unsigned int>(Expression::evaluate(exprs[0]));
        EXPECT_EQ(result, 3);

        exprs = ct.reverse({zero}, {y}, {m}, fastArith); // note 'x' isn't necessary in reverse
        EXPECT_EQ(1, exprs.size());
        result = std::get<unsigned int>(Expression::evaluate(exprs[0]));
        EXPECT_EQ(result, 0);
    }

    TEST_F(CoordinateGraphTest, Basic3D)
    {
        auto ct = CoordinateGraph();

        auto size_x   = std::make_shared<Expression::Expression>(64u);
        auto stride_x = std::make_shared<Expression::Expression>(1u);
        auto size_y   = std::make_shared<Expression::Expression>(32u);
        auto stride_y = std::make_shared<Expression::Expression>(4u);
        auto size_z   = std::make_shared<Expression::Expression>(16u);
        auto stride_z = std::make_shared<Expression::Expression>(8u);

        uint x_index_value = 5;
        uint y_index_value = 3;
        uint z_index_value = 7;

        auto x_index = std::make_shared<Expression::Expression>(x_index_value);
        auto y_index = std::make_shared<Expression::Expression>(y_index_value);
        auto z_index = std::make_shared<Expression::Expression>(z_index_value);

        int x = ct.addElement(SubDimension(0, size_x, stride_x));
        int y = ct.addElement(SubDimension(0, size_y, stride_y));
        int z = ct.addElement(SubDimension(0, size_z, stride_z));
        int m = ct.addElement(SubDimension(0));

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({x, y, z, m}));

        auto flat = ct.addElement(Flatten(), {x, y, z}, {m});

        auto exprs = ct.forward({x_index, y_index, z_index}, {x, y, z}, {m}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(1, exprs.size());
        EXPECT_EQ(sexpr,
                  "{Flatten: Add(Multiply(Add(Multiply(5:U32, 32:U32)U32, 3:U32)U32, 16:U32)U32, "
                  "7:U32)U32}");

        exprs = ct.forward({x_index, y_index, z_index}, {x, y, z}, {m}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(1, exprs.size());
        EXPECT_EQ(sexpr, "2615:U32");

        auto result = std::get<unsigned int>(Expression::evaluate(exprs[0]));
        EXPECT_EQ(2615, result);

        {
            auto rev_index     = std::make_shared<Expression::Expression>(5);
            auto exprs_reverse = ct.reverse({rev_index}, {x, y, z}, {m}, nullptr);
            EXPECT_EQ(3, exprs_reverse.size());

            EXPECT_EQ(Expression::toString(exprs_reverse[2]),
                      "{Flatten[2]: Modulo(5:I, 16:U32)U32}");
            EXPECT_EQ(Expression::toString(exprs_reverse[1]),
                      "{Flatten[1]: Modulo(Divide(5:I, 16:U32)U32, 32:U32)U32}");
            EXPECT_EQ(Expression::toString(exprs_reverse[0]),
                      "{Flatten[0]: Divide(Divide(5:I, 16:U32)U32, 32:U32)U32}");
        }

        {
            // Should be able to get back the individual coordinate values.
            auto rev_index     = std::make_shared<Expression::Expression>(result);
            auto exprs_reverse = ct.reverse({rev_index}, {x, y, z}, {m}, fastArith);
            EXPECT_EQ(3, exprs_reverse.size());

            EXPECT_EQ(x_index_value,
                      std::get<unsigned int>(Expression::evaluate(exprs_reverse[0])));
            EXPECT_EQ(y_index_value,
                      std::get<unsigned int>(Expression::evaluate(exprs_reverse[1])));
            EXPECT_EQ(z_index_value,
                      std::get<unsigned int>(Expression::evaluate(exprs_reverse[2])));
        }
    }

    TEST_F(CoordinateGraphTest, Basic1D01)
    {
        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64u);
        auto unit           = Expression::literal(1u);

        auto size   = Expression::literal(1024u);
        auto stride = Expression::literal(2u);

        auto u  = ct.addElement(User());
        auto i  = ct.addElement(SubDimension(0, size, stride));
        auto wg = ct.addElement(Workgroup(2));
        auto wf = ct.addElement(Wavefront(0, wavefront_size, unit));

        // dimension "i" gets tiled into workgroups and wavefronts
        ct.addElement(Split(), {u}, {i});
        ct.addElement(Tile(), {i}, {wg, wf});

        // TODO: uncomment and fix when appropriate methods are in place

        // auto uOut = ct.getOutputs(getTag(u), EdgeType::CoordinateTransform);
        // ASSERT_EQ(1, uOut.size());
        // EXPECT_EQ(getTag(i), getTag(uOut[0]));

        // ASSERT_EQ(0, ct.getOutputs(getTag(u), EdgeType::DataFlow).size());

        auto block_index  = Expression::literal(2);
        auto thread_index = Expression::literal(33);

        // given indexes for the workgroup and wavefront, compute "i"
        auto exprs = ct.reverse({block_index, thread_index}, {u}, {wg, wf}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 64:U32)U32, 33:I)U32}, 2:U32)U32}");

        exprs = ct.reverse({block_index, thread_index}, {u}, {wg, wf}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "322:U32");

        auto thread_index_register = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        thread_index_register->allocateNow();

        exprs = ct.reverse(
            {block_index, thread_index_register->expression()}, {u}, {wg, wf}, nullptr);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 64:U32)U32, v0:I)U32}, 2:U32)U32}");

        exprs = ct.reverse(
            {block_index, thread_index_register->expression()}, {u}, {wg, wf}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "{TileSplit: AddShiftL(128:U32, v0:I, 1:U32)U32}");

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({u, i, wg, wf}));

        // auto currentEdges = ct.getEdges();

        // EXPECT_EQ(currentEdges.size(), 2);

        // auto savedEdge = currentEdges[1];

        // ct.removeEdge(currentEdges[0]);

        // currentEdges = ct.getEdges();

        // EXPECT_EQ(currentEdges[0], savedEdge);
    }

    TEST_F(CoordinateGraphTest, Basic1D01Eval)
    {
        auto ct = CoordinateGraph();

        auto wavefront_size = std::make_shared<Expression::Expression>(64);
        auto unit           = std::make_shared<Expression::Expression>(1);

        auto size   = std::make_shared<Expression::Expression>(1024);
        auto stride = std::make_shared<Expression::Expression>(2);

        auto workgroup_size = std::make_shared<Expression::Expression>(256) / wavefront_size;

        auto u  = ct.addElement(User());
        auto i  = ct.addElement(SubDimension(0, size, stride));
        auto wg = ct.addElement(Workgroup(2));
        auto wf = ct.addElement(Wavefront(0, wavefront_size, unit));

        // dimension "i" gets tiled into workgroups and wavefronts
        ct.addElement(Split(), {u}, {i});
        ct.addElement(Tile(), {i}, {wg, wf});

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({u, i, wg, wf}));

        auto block_index  = std::make_shared<Expression::Expression>(2);
        auto thread_index = std::make_shared<Expression::Expression>(33);

        // given indexes for the workgroup and wavefront, compute "u"
        {
            auto exprs = ct.reverse({block_index, thread_index}, {u}, {wg, wf}, fastArith);
            auto sexpr = Expression::toString(exprs[0]);

            auto uVal = std::get<int>(Expression::evaluate(exprs[0]));

            EXPECT_EQ(322, uVal) << toString(exprs[0]);
        }

        // given indexes for the workgroup and wavefront, compute "i"
        {
            auto exprs = ct.reverse({block_index, thread_index}, {i}, {wg, wf}, fastArith);
            auto sexpr = Expression::toString(exprs[0]);

            auto iVal = std::get<int>(Expression::evaluate(exprs[0]));

            EXPECT_EQ(161, iVal) << toString(exprs[0]);

            auto iValExpr = std::make_shared<Expression::Expression>(iVal);

            // given "i", compute workgroup and wavefront
            auto fwdExprs = ct.forward({iValExpr}, {i}, {wg, wf}, fastArith);

            EXPECT_EQ(2, fwdExprs.size());
            EXPECT_EQ(2, std::get<int>(evaluate(fwdExprs[0]))) << toString(fwdExprs[0]);
            EXPECT_EQ(33, std::get<int>(evaluate(fwdExprs[1]))) << toString(fwdExprs[1]);
        }
    }

    TEST_F(CoordinateGraphTest, Basic1D02)
    {
        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64);
        auto unit           = Expression::literal(1);

        auto size   = Expression::literal(1024);
        auto stride = Expression::literal(2);

        auto u      = ct.addElement(User());
        auto i      = ct.addElement(SubDimension(0, size, stride));
        auto wg     = ct.addElement(Workgroup(2));
        auto wf     = ct.addElement(Wavefront(0, wavefront_size, unit));
        auto unroll = ct.addElement(Unroll(4));

        // dimension "i" gets tiled into workgroups and wavefronts, and
        // each thread in a wavefront operates on 4 elements (unroll)
        ct.addElement(Split(), {u}, {i});
        ct.addElement(Tile(), {i}, {wg, wf, unroll});

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({u, i, wg, wf, unroll}));

        auto block_index  = Expression::literal(2);
        auto thread_index = Expression::literal(33);
        auto unroll_index = Expression::literal(2);

        // given indexes for the workgroup and wavefront, compute "i"
        auto exprs
            = ct.reverse({block_index, thread_index, unroll_index}, {u}, {wg, wf, unroll}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(Add(Multiply(2:I, 64:I)I, 33:I)I, "
                  "4:U32)U32, 2:I)U32}, 2:I)U32}");

        exprs = ct.reverse(
            {block_index, thread_index, unroll_index}, {u}, {wg, wf, unroll}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "1292:U32");

        auto thread_index_register = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        thread_index_register->allocateNow();

        exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                           {u},
                           {wg, wf, unroll},
                           nullptr);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(Add(Multiply(2:I, 64:I)I, v0:I)I, "
                  "4:U32)U32, 2:I)U32}, 2:I)U32}");

        exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                           {u},
                           {wg, wf, unroll},
                           fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: ShiftL({Tile: Add(ShiftL(Convert(Add(128:I, "
                  "v0:I)I)U32, 2:U32)U32, "
                  "2:I)U32}, 1:U32)U32}");
    }

    TEST_F(CoordinateGraphTest, Basic1D03)
    {
        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64);
        auto unit           = Expression::literal(1);

        auto size   = Expression::literal(1024);
        auto stride = Expression::literal(2);

        auto u      = ct.addElement(User());
        auto i      = ct.addElement(SubDimension(0, size, stride));
        auto wg     = ct.addElement(Workgroup(2));
        auto wf     = ct.addElement(Wavefront(0, Expression::literal(4 * 64), unit));
        auto thread = ct.addElement(SubDimension(1, wavefront_size, unit));
        auto unroll = ct.addElement(Unroll(4));

        // as in Basic1D02, but we add an intermediate dimension "wf" and tile twice
        ct.addElement(Split(), {u}, {i});
        ct.addElement(Tile(), {i}, {wg, wf});
        ct.addElement(Tile(), {wf}, {thread, unroll});

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set<int>({u, i, wg, wf, thread, unroll}));

        auto block_index  = Expression::literal(2);
        auto thread_index = Expression::literal(33);
        auto unroll_index = Expression::literal(2);

        // given indexes for the workgroup and wavefront, compute "i"
        auto exprs = ct.reverse(
            {block_index, thread_index, unroll_index}, {u}, {wg, thread, unroll}, nullptr);
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 256:I)I, {Tile: Add(Multiply(33:I, "
                  "4:U32)U32, 2:I)U32})U32}, 2:I)U32}");

        exprs = ct.reverse(
            {block_index, thread_index, unroll_index}, {u}, {wg, thread, unroll}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "1292:U32");

        auto thread_index_register = std::make_shared<Register::Value>(
            m_context, Register::Type::Vector, DataType::Int32, 1);
        thread_index_register->allocateNow();

        exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                           {u},
                           {wg, thread, unroll},
                           nullptr);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Multiply({Tile: Add(Multiply(2:I, 256:I)I, {Tile: Add(Multiply(v0:I, "
                  "4:U32)U32, 2:I)U32})U32}, 2:I)U32}");

        exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                           {u},
                           {wg, thread, unroll},
                           fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: ShiftL({Tile: Add(ShiftL(Convert(v0:I)U32, 2:U32)U32, "
                  "514:I)U32}, 1:U32)U32}");
    }

    TEST_F(CoordinateGraphTest, TensorTile2DLoadStore01)
    {
        // one tile per wavefront

        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64);
        auto unit           = Expression::literal(1);

        auto M = Expression::literal(100);
        auto N = Expression::literal(200);
        auto K = Expression::literal(300);

        int m = 16;
        int n = 16;

        // A matrix; tag 1; M x K; C-ordering
        auto A   = ct.addElement(User());
        auto Ai  = ct.addElement(SubDimension(0, M, K));
        auto Aj  = ct.addElement(SubDimension(1, K, unit));
        auto sp0 = ct.addElement(Split(), {A}, {Ai, Aj});

        // B matrix; tag 2; K x N; C-ordering
        auto B   = ct.addElement(User());
        auto Bi  = ct.addElement(SubDimension(0, K, N));
        auto Bj  = ct.addElement(SubDimension(1, N, unit));
        auto sp1 = ct.addElement(Split(), {B}, {Bi, Bj});

        // T tile; tag 3; m x n
        auto T    = MacroTile({m, n}, MemoryType::VGPR);
        auto T_id = ct.addElement(T);
        auto i    = ct.addElement(T.tileIndex(0));
        auto j    = ct.addElement(T.tileIndex(1));
        auto join = ct.addElement(Join(), {i, j}, {T_id});

        // tile each dimension of A matrix; each workgroup gets one tile
        auto tile_x = ct.addElement(Workgroup(0));
        auto tile_y = ct.addElement(Workgroup(1));
        auto t0     = ct.addElement(Tile(), {Ai}, {tile_x, i});
        auto t1     = ct.addElement(Tile(), {Aj}, {tile_y, j});

        EXPECT_EQ(ct.getNodes().to<std::set>(),
                  std::set<int>({A, Ai, Aj, B, Bi, Bj, T_id, i, j, tile_x, tile_y}));

        EXPECT_EQ(ct.getEdges().to<std::set>(), std::set<int>({sp0, sp1, join, t0, t1}));

        auto tile_x_index = Expression::literal(4);
        auto tile_y_index = Expression::literal(5);
        auto i_index      = Expression::literal(33);
        auto j_index      = Expression::literal(2);

        std::vector<Expression::ExpressionPtr> exprs;
        std::string                            sexpr;

        {
            EXPECT_THROW(ct.reverse({tile_x_index, i_index}, {A}, {tile_x, i}, fastArith),
                         FatalError);
            EXPECT_THROW(ct.reverse({tile_x_index, i_index}, {Aj}, {tile_x, i}, fastArith),
                         FatalError);
            EXPECT_THROW(ct.reverse({tile_x_index, i_index}, {Ai, Aj}, {tile_x, i}, fastArith),
                         FatalError);
        }

        exprs = ct.reverse(
            {tile_x_index, tile_y_index, i_index, j_index}, {Ai}, {tile_x, tile_y, i, j}, nullptr);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "{Tile: Add(Multiply(4:I, 16:U32)U32, 33:I)U32}");

        exprs = ct.reverse({tile_x_index, tile_y_index, i_index, j_index},
                           {Ai},
                           {tile_x, tile_y, i, j},
                           fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "97:U32");

        exprs = ct.reverse(
            {tile_x_index, tile_y_index, i_index, j_index}, {A}, {tile_x, tile_y, i, j}, nullptr);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Add(Multiply({Tile: Add(Multiply(4:I, 16:U32)U32, 33:I)U32}, 300:I)U32, "
                  "Multiply({Tile: Add(Multiply(5:I, 16:U32)U32, 2:I)U32}, 1:I)U32)U32}");

        exprs = ct.reverse(
            {tile_x_index, tile_y_index, i_index, j_index}, {A}, {tile_x, tile_y, i, j}, fastArith);
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "29182:U32");
    }

    TEST_F(CoordinateGraphTest, TensorTile2DLoadStore02)
    {
        // one tile per wavefront

        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64);
        auto unit           = Expression::literal(1);

        auto M = Expression::literal(100);
        auto N = Expression::literal(200);
        auto K = Expression::literal(300);

        int m = 16;
        int n = 16;

        // A matrix; tag 1; M x K; C-ordering
        auto A   = ct.addElement(User());
        auto Ai  = ct.addElement(SubDimension(0, M, K));
        auto Aj  = ct.addElement(SubDimension(1, K, unit));
        auto sp0 = ct.addElement(Split(), {A}, {Ai, Aj});

        // T tile; tag 3; m x n
        auto T    = MacroTile({m, n}, MemoryType::VGPR);
        auto T_id = ct.addElement(T);
        auto i    = ct.addElement(T.tileIndex(0));
        auto j    = ct.addElement(T.tileIndex(1));
        auto join = ct.addElement(Join(), {i, j}, {T_id});

        // tile each dimension of A matrix; each workgroup gets one tile
        auto tile_x = ct.addElement(Workgroup(0));
        auto tile_y = ct.addElement(Workgroup(1));
        auto t0     = ct.addElement(Tile(), {Ai}, {tile_x, i});
        auto t1     = ct.addElement(Tile(), {Aj}, {tile_y, j});

        EXPECT_EQ(ct.getNodes().to<std::set>(), std::set({A, Ai, Aj, T_id, i, j, tile_x, tile_y}));

        EXPECT_EQ(ct.getNodes().to<std::set>(), ct.getElements<Dimension>().to<std::set>());

        EXPECT_EQ(ct.getEdges().to<std::set>(), std::set({sp0, join, t0, t1}));

        // getElements only works on Node, Edge, and direct members of the
        // Node and Edge variants, not on any sub-variants of Edge. We can't
        // unfortunately (yet?) filter only Tile edges for example.
        EXPECT_EQ(ct.getElements<CoordinateTransformEdge>().to<std::set>(),
                  std::set({sp0, join, t0, t1}));

        auto coords = Transformer(std::make_shared<CoordinateGraph>(ct));

        auto identity_transducer = [&](auto expr) { return expr; };
        auto transducer          = [&](auto expr) { return Expression::fastMultiplication(expr); };

        std::vector<Expression::ExpressionPtr> exprs;
        std::string                            sexpr;

        // set location to thread tile
        coords.setCoordinate(tile_x, Expression::literal(4u));
        coords.setCoordinate(tile_y, Expression::literal(5u));
        coords.setCoordinate(i, Expression::literal(33u));
        coords.setCoordinate(j, Expression::literal(2u));

        // from thread tile to macro tile row
        exprs = coords.reverse({Ai});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "{Tile: Add(Multiply(4:U32, 16:U32)U32, 33:U32)U32}");

        // from thread tile to user tensor
        exprs = coords.reverse({A});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(
            sexpr,
            "{Split: Add(Multiply({Tile: Add(Multiply(4:U32, 16:U32)U32, 33:U32)U32}, 300:I)U32, "
            "Multiply({Tile: Add(Multiply(5:U32, 16:U32)U32, 2:U32)U32}, 1:I)U32)U32}");

        // check edge case where expression are unchanged
        coords.setTransducer(identity_transducer);
        exprs = coords.reverse({A});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(
            sexpr,
            "{Split: Add(Multiply({Tile: Add(Multiply(4:U32, 16:U32)U32, 33:U32)U32}, 300:I)U32, "
            "Multiply({Tile: Add(Multiply(5:U32, 16:U32)U32, 2:U32)U32}, 1:I)U32)U32}");
        coords.setTransducer(nullptr);

        // as above, with fast multiplication
        coords.setTransducer(transducer);
        exprs = coords.reverse({A});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Add(Multiply({Tile: Add(ShiftL(4:U32, 4:U32)U32, 33:U32)U32}, "
                  "300:I)U32, {Tile: Add(ShiftL(5:U32, 4:U32)U32, 2:U32)U32})U32}");
        coords.setTransducer(nullptr);

        // remove i, try again: should fail
        coords.removeCoordinate(i);
        EXPECT_THROW(coords.reverse({A}), FatalError);

        // remove i and j, so only know workgroup and workitem
        coords.removeCoordinate(i);
        coords.removeCoordinate(j);

        // from user tensor row to thread tile row
        coords.setCoordinate(Ai, Expression::literal(17));
        exprs = coords.forward({i});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "{Tile[1]: Modulo(17:I, 16:U32)U32}");
    }

    TEST_F(CoordinateGraphTest, TensorTile2DLoadStore03)
    {
        auto ct = CoordinateGraph();

        uint m = 16;
        uint n = 16;

        auto unit = Expression::literal(1u);
        auto M    = Expression::literal(100u * m);
        auto N    = Expression::literal(300u * n);

        // A matrix; tag 1; M x K; C-ordering
        auto A  = ct.addElement(User());
        auto Ai = ct.addElement(SubDimension(0, M, N));
        auto Aj = ct.addElement(SubDimension(1, N, unit));
        ct.addElement(Split(), {A}, {Ai, Aj});

        // T tile; tag 3; m x n
        auto T    = MacroTile({(int)m, (int)n}, MemoryType::VGPR);
        auto T_id = ct.addElement(T);
        auto i    = ct.addElement(T.tileIndex(0));
        auto j    = ct.addElement(T.tileIndex(1));
        ct.addElement(Join(), {i, j}, {T_id});

        // tile each dimension of A matrix; each workgroup gets one tile
        auto tile_x
            = ct.addElement(SubDimension(0, Expression::literal(100u), Expression::literal(m)));
        auto tile_y
            = ct.addElement(SubDimension(1, Expression::literal(300u), Expression::literal(n)));
        ct.addElement(Tile(), {Ai}, {tile_x, i});
        ct.addElement(Tile(), {Aj}, {tile_y, j});

        auto D  = ct.addElement(User());
        auto Di = ct.addElement(SubDimension(0, M, N));
        auto Dj = ct.addElement(SubDimension(1, N, unit));
        ct.addElement(Flatten(), {tile_x, i}, {Di});
        ct.addElement(Flatten(), {tile_y, j}, {Dj});
        ct.addElement(Join(), {Di, Dj}, {D});

        EXPECT_EQ(ct.getNodes().to<std::set>(),
                  std::set<int>({A, Ai, Aj, T_id, i, j, tile_x, tile_y, D, Di, Dj}));

        auto coords
            = Transformer(std::make_shared<CoordinateGraph>(ct), nullptr, Expression::simplify);

        coords.setCoordinate(tile_x, Expression::literal(4u));
        coords.setCoordinate(tile_y, Expression::literal(5u));
        coords.setCoordinate(i, Expression::literal(33u));
        coords.setCoordinate(j, Expression::literal(2u));

        auto exprs = coords.reverseStride(tile_x, Expression::literal(1u), {A});
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "76800:U32");

        exprs = coords.reverseStride(i, Expression::literal(2u), {A});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "9600:U32");

        exprs = coords.reverseStride(j, Expression::literal(1u), {A});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "1:U32");

        exprs = coords.reverseStride(tile_y, Expression::literal(2u), {A});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "32:U32");

        exprs = coords.forwardStride(tile_y, Expression::literal(2u), {D});
        sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr, "32:U32");
    }

    TEST_F(CoordinateGraphTest, SunderBasicOne)
    {
        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);

        auto ct = CoordinateGraph();

        auto input = ct.addElement(User());

        auto linX = ct.addElement(Linear(Expression::literal(100u), one));
        auto sw   = ct.addElement(Linear(Expression::literal(2u), one));

        auto sunder = ct.addElement(Sunder(), {input}, {linX, sw});

        auto a = ct.addElement(Linear());
        auto b = ct.addElement(Linear(Expression::literal(10u), one));

        ct.addElement(Tile(), {linX}, {a, b});

        auto aVal = Expression::literal(8u);
        auto bVal = Expression::literal(3u);

        auto exprX
            = only(ct.reverse({aVal, bVal, zero}, {input}, {a, b, sw}, Expression::identity));
        EXPECT_THROW(only(ct.reverse({aVal, bVal, one}, {input}, {a, b, sw}, Expression::identity)),
                     FatalError);

        EXPECT_EQ("{Sunder: Add(Multiply(8:U32, 10:U32)U32, 3:U32)U32}", toString(*exprX));
    }

    TEST_F(CoordinateGraphTest, SunderBasicTwo)
    {
        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);
        auto two  = Expression::literal(2u);

        auto ct = CoordinateGraph();

        auto input = ct.addElement(User());

        auto linX = ct.addElement(Linear(Expression::literal(100u), one));
        auto linY = ct.addElement(Linear(Expression::literal(24u), one));
        auto sw   = ct.addElement(Linear(Expression::literal(2u), one));

        auto sunder = ct.addElement(Sunder(), {input}, {linX, linY, sw});

        auto a = ct.addElement(Linear());
        auto b = ct.addElement(Linear(Expression::literal(10u), one));

        ct.addElement(Tile(), {linX}, {a, b});

        auto c = ct.addElement(Linear());
        auto d = ct.addElement(Linear(Expression::literal(6u), one));

        ct.addElement(Tile(), {linY}, {c, d});

        auto aVal = Expression::literal(8u);
        auto bVal = Expression::literal(3u);
        auto cVal = Expression::literal(1u);
        auto dVal = Expression::literal(4u);

        auto exprX = only(ct.reverse(
            {aVal, bVal, nullptr, nullptr, zero}, {input}, {a, b, c, d, sw}, Expression::identity));
        auto exprY = only(ct.reverse(
            {nullptr, nullptr, cVal, dVal, one}, {input}, {a, b, c, d, sw}, Expression::identity));
        EXPECT_THROW(only(ct.reverse({aVal, bVal, nullptr, nullptr, two},
                                     {input},
                                     {a, b, c, d, sw},
                                     Expression::identity)),
                     FatalError);

        EXPECT_EQ("{Sunder: Add(Multiply(8:U32, 10:U32)U32, 3:U32)U32}", toString(*exprX));
        EXPECT_EQ("{Sunder: Add({Tile: Add(Multiply(1:U32, 6:U32)U32, 4:U32)U32}, 100:U32)U32}",
                  toString(*exprY));
    }

    TEST_F(CoordinateGraphTest, SunderBasicThree)
    {
        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);

        auto ct = CoordinateGraph();

        auto input = ct.addElement(User());

        auto linX = ct.addElement(Linear(Expression::literal(100u), one));
        auto linY = ct.addElement(Linear(Expression::literal(24u), one));
        auto linZ = ct.addElement(Linear(Expression::literal(17u), one));
        auto sw   = ct.addElement(Linear(Expression::literal(2u), one));

        auto sunder = ct.addElement(Sunder(), {input}, {linX, linY, linZ, sw});

        auto a = ct.addElement(Linear());
        auto b = ct.addElement(Linear(Expression::literal(10u), one));

        ct.addElement(Tile(), {linX}, {a, b});

        auto c = ct.addElement(Linear());
        auto d = ct.addElement(Linear(Expression::literal(6u), one));

        ct.addElement(Tile(), {linY}, {c, d});

        auto e = ct.addElement(Linear());
        auto f = ct.addElement(Linear(Expression::literal(9u), one));

        ct.addElement(Tile(), {linZ}, {e, f});

        auto aVal = Expression::literal(8u);
        auto bVal = Expression::literal(3u);
        auto cVal = Expression::literal(1u);
        auto dVal = Expression::literal(4u);
        auto eVal = Expression::literal(0u);
        auto fVal = Expression::literal(19u);

        auto exprX = only(ct.reverse({aVal, bVal, cVal, dVal, eVal, fVal, zero},
                                     {input},
                                     {a, b, c, d, e, f, sw},
                                     Expression::identity));
        auto exprY = only(ct.reverse({aVal, bVal, cVal, dVal, eVal, fVal, one},
                                     {input},
                                     {a, b, c, d, e, f, sw},
                                     Expression::identity));
        auto exprZ = only(ct.reverse({aVal, bVal, cVal, dVal, eVal, fVal, two},
                                     {input},
                                     {a, b, c, d, e, f, sw},
                                     Expression::identity));
        EXPECT_THROW(only(ct.reverse({aVal, bVal, cVal, dVal, eVal, fVal, three},
                                     {input},
                                     {a, b, c, d, e, f, sw},
                                     Expression::identity)),
                     FatalError);

        EXPECT_EQ("{Sunder: Add(Multiply(8:U32, 10:U32)U32, 3:U32)U32}", toString(*exprX));
        EXPECT_EQ("{Sunder: Add({Tile: Add(Multiply(1:U32, 6:U32)U32, 4:U32)U32}, 100:U32)U32}",
                  toString(*exprY));
        EXPECT_EQ("{Sunder: Add({Tile: Add(Multiply(0:U32, 9:U32)U32, 19:U32)U32}, Add(100:U32, "
                  "24:U32)U32)U32}",
                  toString(*exprZ));
    }

    TEST_F(CoordinateGraphTest, SunderBasicOneViaTransformer)
    {
        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);

        auto ct = CoordinateGraph();

        auto input = ct.addElement(User());

        auto linX = ct.addElement(Linear(Expression::literal(100u), one));
        auto sw   = ct.addElement(Linear(Expression::literal(2u), one));

        auto sunder = ct.addElement(Sunder(), {input}, {linX, sw});

        auto a = ct.addElement(Linear());
        auto b = ct.addElement(Linear(Expression::literal(10u), one));

        ct.addElement(Tile(), {linX}, {a, b});

        auto tf = Transformer(std::make_shared<CoordinateGraph>(ct), nullptr, Expression::identity);
        tf.setCoordinate(a, Expression::literal(8u));
        tf.setCoordinate(b, Expression::literal(3u));

        tf.setCoordinate(sw, zero);
        auto exprX = only(tf.reverse({input}));
        EXPECT_EQ("{Sunder: Add(Multiply(8:U32, 10:U32)U32, 3:U32)U32}", toString(*exprX));
        auto strideExprX = only(tf.reverseStride(b, Expression::literal(2u), {input}));
        EXPECT_EQ("2:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(a, Expression::literal(2u), {input}));
        EXPECT_EQ("20:U32", toString(*strideExprX));

        tf.setCoordinate(sw, one);
        EXPECT_THROW(tf.reverse({input}), FatalError);
        EXPECT_THROW(tf.reverseStride(b, Expression::literal(2u), {input}), FatalError);
        EXPECT_THROW(tf.reverseStride(a, Expression::literal(2u), {input}), FatalError);
    }

    TEST_F(CoordinateGraphTest, SunderBasicTwoViaTransformer)
    {
        auto zero = Expression::literal(0u);
        auto one  = Expression::literal(1u);
        auto two  = Expression::literal(2u);

        auto ct = CoordinateGraph();

        auto input = ct.addElement(User());

        auto linX = ct.addElement(Linear(Expression::literal(100u), one));
        auto linY = ct.addElement(Linear(Expression::literal(24u), one));
        auto sw   = ct.addElement(Linear(Expression::literal(2u), one));

        auto sunder = ct.addElement(Sunder(), {input}, {linX, linY, sw});

        auto a = ct.addElement(Linear());
        auto b = ct.addElement(Linear(Expression::literal(10u), one));

        ct.addElement(Tile(), {linX}, {a, b});

        auto c = ct.addElement(Linear());
        auto d = ct.addElement(Linear(Expression::literal(6u), one));

        ct.addElement(Tile(), {linY}, {c, d});

        auto tf = Transformer(std::make_shared<CoordinateGraph>(ct), nullptr, Expression::identity);

        tf.setCoordinate(a, Expression::literal(8u));
        tf.setCoordinate(b, Expression::literal(3u));
        tf.setCoordinate(c, Expression::literal(1u));
        tf.setCoordinate(d, Expression::literal(4u));

        tf.setCoordinate(sw, zero);
        auto exprX = only(tf.reverse({input}));
        EXPECT_EQ("{Sunder: Add(Multiply(8:U32, 10:U32)U32, 3:U32)U32}", toString(*exprX));
        auto strideExprX = only(tf.reverseStride(b, Expression::literal(2u), {input}));
        EXPECT_EQ("2:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(a, Expression::literal(2u), {input}));
        EXPECT_EQ("20:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(d, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(c, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprX));

        tf.setCoordinate(sw, one);
        auto exprY = only(tf.reverse({input}));
        EXPECT_EQ("{Sunder: Add({Tile: Add(Multiply(1:U32, 6:U32)U32, 4:U32)U32}, 100:U32)U32}",
                  toString(*exprY));
        auto strideExprY = only(tf.reverseStride(d, Expression::literal(3u), {input}));
        EXPECT_EQ("3:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(c, Expression::literal(3u), {input}));
        EXPECT_EQ("18:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(b, Expression::literal(2u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(a, Expression::literal(2u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprY));

        tf.setCoordinate(sw, two);
        EXPECT_THROW(tf.reverse({input}), FatalError);
    }

    TEST_F(CoordinateGraphTest, SunderBasicThreeViaTransformer)
    {
        auto zero  = Expression::literal(0u);
        auto one   = Expression::literal(1u);
        auto two   = Expression::literal(2u);
        auto three = Expression::literal(3u);

        auto ct = CoordinateGraph();

        auto input = ct.addElement(User());

        auto linX = ct.addElement(Linear(Expression::literal(100u), one));
        auto linY = ct.addElement(Linear(Expression::literal(24u), one));
        auto linZ = ct.addElement(Linear(Expression::literal(17u), one));
        auto sw   = ct.addElement(Linear(Expression::literal(2u), one));

        auto sunder = ct.addElement(Sunder(), {input}, {linX, linY, linZ, sw});

        auto a = ct.addElement(Linear());
        auto b = ct.addElement(Linear(Expression::literal(10u), one));

        ct.addElement(Tile(), {linX}, {a, b});

        auto c = ct.addElement(Linear());
        auto d = ct.addElement(Linear(Expression::literal(6u), one));

        ct.addElement(Tile(), {linY}, {c, d});

        auto e = ct.addElement(Linear());
        auto f = ct.addElement(Linear(Expression::literal(9u), one));

        ct.addElement(Tile(), {linZ}, {e, f});

        auto tf = Transformer(std::make_shared<CoordinateGraph>(ct), nullptr, Expression::identity);

        tf.setCoordinate(a, Expression::literal(8u));
        tf.setCoordinate(b, Expression::literal(3u));
        tf.setCoordinate(c, nullptr);
        tf.setCoordinate(d, nullptr);
        tf.setCoordinate(sw, zero);
        tf.setCoordinate(e, nullptr);
        tf.setCoordinate(f, nullptr);
        auto exprX = only(tf.reverse({input}));
        EXPECT_EQ("{Sunder: Add(Multiply(8:U32, 10:U32)U32, 3:U32)U32}", toString(*exprX));
        auto strideExprX = only(tf.reverseStride(b, Expression::literal(2u), {input}));
        EXPECT_EQ("2:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(a, Expression::literal(2u), {input}));
        EXPECT_EQ("20:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(d, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(c, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(e, Expression::literal(1u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprX));
        strideExprX = only(tf.reverseStride(f, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprX));

        tf.setCoordinate(c, Expression::literal(1u));
        tf.setCoordinate(d, Expression::literal(4u));
        tf.setCoordinate(sw, one);
        auto exprY = only(tf.reverse({input}));
        EXPECT_EQ("{Sunder: Add({Tile: Add(Multiply(1:U32, 6:U32)U32, 4:U32)U32}, 100:U32)U32}",
                  toString(*exprY));
        auto strideExprY = only(tf.reverseStride(d, Expression::literal(3u), {input}));
        EXPECT_EQ("3:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(c, Expression::literal(3u), {input}));
        EXPECT_EQ("18:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(b, Expression::literal(2u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(a, Expression::literal(2u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(e, Expression::literal(1u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprY));
        strideExprY = only(tf.reverseStride(f, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprY));

        tf.setCoordinate(e, Expression::literal(0u));
        tf.setCoordinate(f, Expression::literal(19u));
        tf.setCoordinate(sw, two);
        auto exprZ = only(tf.reverse({input}));
        EXPECT_EQ("{Sunder: Add({Tile: Add(Multiply(0:U32, 9:U32)U32, 19:U32)U32}, Add(100:U32, "
                  "24:U32)U32)U32}",
                  toString(*exprZ));
        auto strideExprZ = only(tf.reverseStride(f, Expression::literal(1u), {input}));
        EXPECT_EQ("1:U32", toString(*strideExprZ));
        strideExprZ = only(tf.reverseStride(e, Expression::literal(3u), {input}));
        EXPECT_EQ("27:U32", toString(*strideExprZ));
        strideExprZ = only(tf.reverseStride(b, Expression::literal(2u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprZ));
        strideExprZ = only(tf.reverseStride(a, Expression::literal(2u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprZ));
        strideExprZ = only(tf.reverseStride(d, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprZ));
        strideExprZ = only(tf.reverseStride(c, Expression::literal(3u), {input}));
        EXPECT_EQ("0:U32", toString(*strideExprZ));

        tf.setCoordinate(sw, three);
        EXPECT_THROW(tf.reverse({input}), FatalError);
    }

    TEST_F(CoordinateGraphTest, WaveTileBasic)
    {
        // one tile per wavefront
        auto ct = CoordinateGraph();

        auto MacA  = MacroTile({32u, 32u, 32u, 32u}, LayoutType::MATRIX_A, {32u, 0u, 2u, 0u});
        auto WaveA = WaveTile(MacA);

        auto WaveAI = WaveA.tileNumber(0);
        auto WaveAJ = WaveA.tileNumber(1);

        auto WaveAi = WaveA.tileIndex(0);
        auto WaveAj = WaveA.tileIndex(1);

        EXPECT_EQ(WaveAI.dim, 0);
        EXPECT_EQ(WaveAJ.dim, 1);

        EXPECT_EQ(WaveAi.dim, 0);
        EXPECT_EQ(WaveAj.dim, 1);

        EXPECT_EQ(std::get<uint>(Expression::evaluate(WaveAi.size)), 32u);
        EXPECT_EQ(std::get<uint>(Expression::evaluate(WaveAj.size)), 2u);

        EXPECT_EQ(std::get<int>(Expression::evaluate(WaveAi.stride)), 2);
        EXPECT_EQ(std::get<int>(Expression::evaluate(WaveAj.stride)), 1);
    }

    TEST_F(CoordinateGraphTest, HasPath)
    {
        // one tile per wavefront
        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64u);
        auto unit           = Expression::literal(1u);

        auto size   = Expression::literal(1024u);
        auto stride = Expression::literal(2u);

        auto u  = ct.addElement(User());
        auto i  = ct.addElement(SubDimension(0, size, stride));
        auto j  = ct.addElement(SubDimension(1, size, stride));
        auto wg = ct.addElement(Workgroup(2));
        auto wf = ct.addElement(Wavefront(0, wavefront_size, unit));

        ct.addElement(Split(), {u}, {i, j});
        ct.addElement(Tile(), {i}, {wg, wf});

        EXPECT_TRUE(ct.hasPath<Graph::Direction::Upstream>({u}, {wg, wf, j}));
        EXPECT_FALSE(ct.hasPath<Graph::Direction::Upstream>({u}, {wg, j}));
        EXPECT_FALSE(ct.hasPath<Graph::Direction::Upstream>({u}, {wg, wf}));
        EXPECT_FALSE(ct.hasPath<Graph::Direction::Upstream>({u}, {wf, j}));

        EXPECT_TRUE(ct.hasPath<Graph::Direction::Downstream>({u}, {i, j}));
        EXPECT_TRUE(ct.hasPath<Graph::Direction::Downstream>({u}, {wg}));
        EXPECT_TRUE(ct.hasPath<Graph::Direction::Downstream>({i}, {wg}));
        EXPECT_TRUE(ct.hasPath<Graph::Direction::Downstream>({i}, {wf}));
        EXPECT_TRUE(ct.hasPath<Graph::Direction::Downstream>({i}, {wg, wf}));
        EXPECT_FALSE(ct.hasPath<Graph::Direction::Downstream>({wg, wf}, {j}));

        auto coords = Transformer(std::make_shared<CoordinateGraph>(ct));

        coords.setCoordinate(wg, Expression::literal(0));
        coords.setCoordinate(wf, Expression::literal(1));
        EXPECT_FALSE(coords.hasPath({u}, false));

        coords.setCoordinate(j, Expression::literal(8));
        EXPECT_TRUE(coords.hasPath({u}, false));
        EXPECT_FALSE(coords.hasPath({i}, true));
    }

    class ARCH_CoordinateGraphTest : public GPUContextFixture
    {
    };

    TEST_P(ARCH_CoordinateGraphTest, GPU_TensorTile2DLoadStore04)
    {
        auto ct = CoordinateGraph();

        auto wavefront_size = Expression::literal(64);
        auto unit           = Expression::literal(1);

        auto M = Expression::literal(100);
        auto N = Expression::literal(200);
        auto K = Expression::literal(300);

        int m = 16;
        int n = 16;

        // A matrix; tag 1; M x K; C-ordering
        auto A  = ct.addElement(User());
        auto Ai = ct.addElement(SubDimension(0, M, K));
        auto Aj = ct.addElement(SubDimension(1, K, unit));
        ct.addElement(Split(), {A}, {Ai, Aj});

        // T tile; tag 3; m x n
        auto T    = MacroTile({m, n}, MemoryType::VGPR);
        auto T_id = ct.addElement(T);
        auto i    = ct.addElement(T.tileIndex(0));
        auto j    = ct.addElement(T.tileIndex(1));
        ct.addElement(Join(), {i, j}, {T_id});

        // tile each dimension of A matrix; each workgroup gets one tile
        auto tile_x = ct.addElement(Workgroup(0));
        auto tile_y = ct.addElement(Workgroup(1));
        ct.addElement(Tile(), {Ai}, {tile_x, i});
        ct.addElement(Tile(), {Aj}, {tile_y, j});

        EXPECT_EQ(ct.getNodes().to<std::set>(),
                  std::set<int>({A, Ai, Aj, T_id, i, j, tile_x, tile_y}));

        // need preamble so work coords are initialised
        auto k = m_context->kernel();
        k->setKernelName("TensorTile2DLoadStore04");
        k->setKernelDimensions(2);
        m_context->schedule(k->preamble());

        auto coords = Transformer(std::make_shared<CoordinateGraph>(ct), m_context);

        coords.setCoordinate(i, Expression::literal(33u));
        coords.setCoordinate(j, Expression::literal(2u));

        auto exprs = coords.reverse({A});
        auto sexpr = Expression::toString(exprs[0]);
        EXPECT_EQ(sexpr,
                  "{Split: Add(Multiply({Tile: Add(Multiply({Workgroup Index X: s2:U32}, "
                  "16:U32)U32, 33:U32)U32}, 300:I)U32, Multiply({Tile: Add(Multiply({Workgroup "
                  "Index Y: s3:U32}, 16:U32)U32, 2:U32)U32}, 1:I)U32)U32}");
    }

    INSTANTIATE_TEST_SUITE_P(ARCH_CoordinateGraphTests,
                             ARCH_CoordinateGraphTest,
                             supportedISATuples());

}
