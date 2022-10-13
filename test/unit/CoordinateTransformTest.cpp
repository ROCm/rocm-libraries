#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <variant>

#include <rocRoller/CodeGen/Arithmetic.hpp>
#include <rocRoller/CodeGen/Arithmetic/Int32.hpp>
#include <rocRoller/Expression.hpp>
#include <rocRoller/ExpressionTransformations.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/HyperGraph.hpp>
#include <rocRoller/KernelGraph/CoordinateTransform/Transformer.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "GenericContextFixture.hpp"

using namespace rocRoller;
namespace CoordinateTransform = rocRoller::KernelGraph::CoordinateTransform;

class CoordinateTransformTest : public GenericContextFixture
{
public:
    Expression::FastArithmetic fastArith{m_context};

    void SetUp()
    {
        GenericContextFixture::SetUp();
        fastArith = Expression::FastArithmetic(m_context);
    }
};

TEST_F(CoordinateTransformTest, Basic)
{
    auto ct = CoordinateTransform::HyperGraph();

    auto size    = std::make_shared<Expression::Expression>(64u);
    auto unit    = std::make_shared<Expression::Expression>(1u);
    auto x_index = std::make_shared<Expression::Expression>(5u);
    auto y_index = std::make_shared<Expression::Expression>(3u);

    auto x = CoordinateTransform::SubDimension(0, 0, size, unit);
    auto y = CoordinateTransform::SubDimension(1, 0, size, unit);
    auto m = CoordinateTransform::SubDimension(2, 0, size * size, unit);

    ct.addEdge({x, y}, {m}, CoordinateTransform::Flatten());

    auto exprs = ct.forward({x_index, y_index}, {x, y}, {m}, nullptr);
    auto sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Add(Multiply(5j, 64j), 3j)");
    exprs = ct.forward({x_index, y_index}, {x, y}, {m}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "323j");

    // Trivial
    auto zero = std::make_shared<Expression::Expression>(0u);
    exprs     = ct.forward({zero, zero}, {x, y}, {m}, nullptr);
    EXPECT_EQ(1, exprs.size());
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Add(Multiply(0j, 64j), 0j)");
    exprs = ct.forward({zero, zero}, {x, y}, {m}, fastArith);
    EXPECT_EQ(1, exprs.size());
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "0j");
    auto result = std::get<unsigned int>(Expression::evaluate(exprs[0]));
    EXPECT_EQ(result, 0);

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

TEST_F(CoordinateTransformTest, Basic3D)
{
    auto ct = CoordinateTransform::HyperGraph();

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

    auto x = CoordinateTransform::SubDimension(0, 0, size_x, stride_x);
    auto y = CoordinateTransform::SubDimension(1, 0, size_y, stride_y);
    auto z = CoordinateTransform::SubDimension(2, 0, size_z, stride_z);
    auto m = CoordinateTransform::SubDimension(3, 0);

    ct.addEdge({x, y, z}, {m}, CoordinateTransform::Flatten());

    auto exprs = ct.forward({x_index, y_index, z_index}, {x, y, z}, {m}, nullptr);
    auto sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(1, exprs.size());
    EXPECT_EQ(sexpr, "Add(Multiply(Add(Multiply(5j, 32j), 3j), 16j), 7j)");

    exprs = ct.forward({x_index, y_index, z_index}, {x, y, z}, {m}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(1, exprs.size());
    EXPECT_EQ(sexpr, "2615j");

    auto result = std::get<unsigned int>(Expression::evaluate(exprs[0]));
    EXPECT_EQ(2615, result);

    {
        auto rev_index     = std::make_shared<Expression::Expression>(5);
        auto exprs_reverse = ct.reverse({rev_index}, {x, y, z}, {m}, nullptr);
        EXPECT_EQ(3, exprs_reverse.size());

        EXPECT_EQ(Expression::toString(exprs_reverse[2]), "Modulo(5i, 16j)");
        EXPECT_EQ(Expression::toString(exprs_reverse[1]), "Modulo(Divide(5i, 16j), 32j)");
        EXPECT_EQ(Expression::toString(exprs_reverse[0]),
                  "Modulo(Divide(Divide(5i, 16j), 32j), 64j)");
    }

    {
        // Should be able to get back the individual coordinate values.
        auto rev_index     = std::make_shared<Expression::Expression>(result);
        auto exprs_reverse = ct.reverse({rev_index}, {x, y, z}, {m}, fastArith);
        EXPECT_EQ(3, exprs_reverse.size());

        EXPECT_EQ(x_index_value, std::get<unsigned int>(Expression::evaluate(exprs_reverse[0])));
        EXPECT_EQ(y_index_value, std::get<unsigned int>(Expression::evaluate(exprs_reverse[1])));
        EXPECT_EQ(z_index_value, std::get<unsigned int>(Expression::evaluate(exprs_reverse[2])));
    }
}

TEST_F(CoordinateTransformTest, Basic1D01)
{
    auto ct = CoordinateTransform::HyperGraph();

    auto wavefront_size = Expression::literal(64u);
    auto unit           = Expression::literal(1u);

    auto size   = Expression::literal(1024u);
    auto stride = Expression::literal(2u);

    auto u  = CoordinateTransform::User(1);
    auto i  = CoordinateTransform::SubDimension(1, 0, size, stride);
    auto wg = CoordinateTransform::Workgroup(2);
    auto wf = CoordinateTransform::Wavefront(2, 0, wavefront_size, unit);

    // dimension "i" gets tiled into workgroups and wavefronts
    ct.addEdge({u}, {i}, CoordinateTransform::Split());
    ct.addEdge({i}, {wg, wf}, CoordinateTransform::Tile());

    auto uOut = ct.getOutputs(getTag(u), CoordinateTransform::EdgeType::CoordinateTransform);
    ASSERT_EQ(1, uOut.size());
    EXPECT_EQ(getTag(i), getTag(uOut[0]));

    ASSERT_EQ(0, ct.getOutputs(getTag(u), CoordinateTransform::EdgeType::DataFlow).size());

    auto block_index  = Expression::literal(2);
    auto thread_index = Expression::literal(33);

    // given indexes for the workgroup and wavefront, compute "i"
    auto exprs = ct.reverse({block_index, thread_index}, {u}, {wg, wf}, nullptr);
    auto sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Multiply(Add(Multiply(2i, 64j), 33i), 2j)");

    exprs = ct.reverse({block_index, thread_index}, {u}, {wg, wf}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "322j");

    auto thread_index_register
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    thread_index_register->allocateNow();

    exprs = ct.reverse({block_index, thread_index_register->expression()}, {u}, {wg, wf}, nullptr);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Multiply(Add(Multiply(2i, 64j), v0:I), 2j)");

    exprs
        = ct.reverse({block_index, thread_index_register->expression()}, {u}, {wg, wf}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "ShiftL(Add(128j, v0:I), 1j)");

    auto currentEdges = ct.getEdges();

    EXPECT_EQ(currentEdges.size(), 2);

    auto savedEdge = currentEdges[1];

    ct.removeEdge(currentEdges[0]);

    currentEdges = ct.getEdges();

    EXPECT_EQ(currentEdges[0], savedEdge);
}

TEST_F(CoordinateTransformTest, Basic1D01Eval)
{
    auto ct = CoordinateTransform::HyperGraph();

    auto wavefront_size = std::make_shared<Expression::Expression>(64);
    auto unit           = std::make_shared<Expression::Expression>(1);

    auto size   = std::make_shared<Expression::Expression>(1024);
    auto stride = std::make_shared<Expression::Expression>(2);

    auto workgroup_size = std::make_shared<Expression::Expression>(256) / wavefront_size;

    auto u  = CoordinateTransform::User(1);
    auto i  = CoordinateTransform::SubDimension(1, 0, size, stride);
    auto wg = CoordinateTransform::Workgroup(2);
    auto wf = CoordinateTransform::Wavefront(2, 0, wavefront_size, unit);

    // dimension "i" gets tiled into workgroups and wavefronts
    ct.addEdge({u}, {i}, CoordinateTransform::Split());
    ct.addEdge({i}, {wg, wf}, CoordinateTransform::Tile());

    auto block_index  = std::make_shared<Expression::Expression>(2);
    auto thread_index = std::make_shared<Expression::Expression>(33);

    // given indexes for the workgroup and wavefront, compute "i"
    auto exprs = ct.reverse({block_index, thread_index}, {u}, {wg, wf}, fastArith);
    auto sexpr = Expression::toString(exprs[0]);

    auto iVal = std::get<int>(Expression::evaluate(exprs[0]));

    EXPECT_EQ(322, iVal) << toString(exprs[0]);

    auto iValExpr = std::make_shared<Expression::Expression>(iVal);

    // given "i", compute workgroup and wavefront
    auto fwdExprs = ct.forward({iValExpr}, {u}, {wg, wf}, fastArith);

    EXPECT_EQ(2, fwdExprs.size());
    EXPECT_EQ(2, std::get<int>(evaluate(fwdExprs[0]))) << toString(fwdExprs[0]);
    EXPECT_EQ(33, std::get<int>(evaluate(fwdExprs[1]))) << toString(fwdExprs[1]);
}

TEST_F(CoordinateTransformTest, Basic1D02)
{
    auto ct = CoordinateTransform::HyperGraph();

    auto wavefront_size = Expression::literal(64);
    auto unit           = Expression::literal(1);

    auto size   = Expression::literal(1024);
    auto stride = Expression::literal(2);

    auto u      = CoordinateTransform::User(1);
    auto i      = CoordinateTransform::SubDimension(1, 0, size, stride);
    auto wg     = CoordinateTransform::Workgroup(2);
    auto wf     = CoordinateTransform::Wavefront(2, 0, wavefront_size, unit);
    auto unroll = CoordinateTransform::Unroll(2, 4);

    // dimension "i" gets tiled into workgroups and wavefronts, and
    // each thread in a wavefront operates on 4 elements (unroll)
    ct.addEdge({u}, {i}, CoordinateTransform::Split());
    ct.addEdge({i}, {wg, wf, unroll}, CoordinateTransform::Tile());

    auto block_index  = Expression::literal(2);
    auto thread_index = Expression::literal(33);
    auto unroll_index = Expression::literal(2);

    // given indexes for the workgroup and wavefront, compute "i"
    auto exprs
        = ct.reverse({block_index, thread_index, unroll_index}, {u}, {wg, wf, unroll}, nullptr);
    auto sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Multiply(Add(Multiply(Add(Multiply(2i, 64i), 33i), 4j), 2i), 2i)");

    exprs = ct.reverse({block_index, thread_index, unroll_index}, {u}, {wg, wf, unroll}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "1292j");

    auto thread_index_register
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    thread_index_register->allocateNow();

    exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                       {u},
                       {wg, wf, unroll},
                       nullptr);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Multiply(Add(Multiply(Add(Multiply(2i, 64i), v0:I), 4j), 2i), 2i)");

    exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                       {u},
                       {wg, wf, unroll},
                       fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "ShiftL(Add(ShiftL(Add(128i, v0:I), 2j), 2i), 1j)");
}

TEST_F(CoordinateTransformTest, Basic1D03)
{
    auto ct = CoordinateTransform::HyperGraph();

    auto wavefront_size = Expression::literal(64);
    auto unit           = Expression::literal(1);

    auto size   = Expression::literal(1024);
    auto stride = Expression::literal(2);

    auto u      = CoordinateTransform::User(1);
    auto i      = CoordinateTransform::SubDimension(1, 0, size, stride);
    auto wg     = CoordinateTransform::Workgroup(2);
    auto wf     = CoordinateTransform::Wavefront(2, 0, Expression::literal(4 * 64), unit);
    auto thread = CoordinateTransform::SubDimension(2, 1, wavefront_size, unit);
    auto unroll = CoordinateTransform::Unroll(2, 4);

    // as in Basic1D02, but we add an intermediate dimension "wf" and tile twice
    ct.addEdge({u}, {i}, CoordinateTransform::Split());
    ct.addEdge({i}, {wg, wf}, CoordinateTransform::Tile());
    ct.addEdge({wf}, {thread, unroll}, CoordinateTransform::Tile());

    EXPECT_EQ(ct.getDimensions().size(), 6);

    auto block_index  = Expression::literal(2);
    auto thread_index = Expression::literal(33);
    auto unroll_index = Expression::literal(2);

    // given indexes for the workgroup and wavefront, compute "i"
    auto exprs
        = ct.reverse({block_index, thread_index, unroll_index}, {u}, {wg, thread, unroll}, nullptr);
    auto sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Multiply(Add(Multiply(2i, 256i), Add(Multiply(33i, 4j), 2i)), 2i)");

    exprs = ct.reverse(
        {block_index, thread_index, unroll_index}, {u}, {wg, thread, unroll}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "1292j");

    auto thread_index_register
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    thread_index_register->allocateNow();

    exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                       {u},
                       {wg, thread, unroll},
                       nullptr);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Multiply(Add(Multiply(2i, 256i), Add(Multiply(v0:I, 4j), 2i)), 2i)");

    exprs = ct.reverse({block_index, thread_index_register->expression(), unroll_index},
                       {u},
                       {wg, thread, unroll},
                       fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "ShiftL(Add(512i, Add(ShiftL(v0:I, 2j), 2i)), 1j)");
}

TEST_F(CoordinateTransformTest, TensorTile2DLoadStore01)
{
    // one tile per wavefront

    auto ct = CoordinateTransform::HyperGraph();

    auto wavefront_size = Expression::literal(64);
    auto unit           = Expression::literal(1);

    // auto M = Expression::literal(4096);
    // auto N = Expression::literal(4096);
    // auto K = Expression::literal(4096);

    auto M = Expression::literal(100);
    auto N = Expression::literal(200);
    auto K = Expression::literal(300);

    int m = 16;
    int n = 16;

    // A matrix; tag 1; M x K; C-ordering
    auto A  = CoordinateTransform::User(1);
    auto Ai = CoordinateTransform::SubDimension(1, 0, M, K);
    auto Aj = CoordinateTransform::SubDimension(1, 1, K, unit);
    ct.addEdge({A}, {Ai, Aj}, CoordinateTransform::Split());

    // B matrix; tag 2; K x N; C-ordering
    auto B  = CoordinateTransform::User(2);
    auto Bi = CoordinateTransform::SubDimension(2, 0, K, N);
    auto Bj = CoordinateTransform::SubDimension(2, 1, N, unit);
    ct.addEdge({B}, {Bi, Bj}, CoordinateTransform::Split());

    // T tile; tag 3; m x n
    auto T = CoordinateTransform::MacroTile(3, {m, n}, MemoryType::VGPR);
    auto i = T.tileIndex(0);
    auto j = T.tileIndex(1);
    ct.addEdge({i, j}, {T}, CoordinateTransform::Join());

    // tile each dimension of A matrix; each workgroup gets one tile
    auto tile_x = CoordinateTransform::Workgroup(1, 0);
    auto tile_y = CoordinateTransform::Workgroup(1, 1);
    ct.addEdge({Ai}, {tile_x, i}, CoordinateTransform::Tile());
    ct.addEdge({Aj}, {tile_y, j}, CoordinateTransform::Tile());

    auto tile_x_index = Expression::literal(4);
    auto tile_y_index = Expression::literal(5);
    auto i_index      = Expression::literal(33);
    auto j_index      = Expression::literal(2);

    std::vector<Expression::ExpressionPtr> exprs;
    std::string                            sexpr;

    EXPECT_THROW(ct.reverse({tile_x_index, i_index}, {Ai, Aj}, {tile_x, i}, fastArith),
                 RecoverableError);

    exprs = ct.reverse(
        {tile_x_index, tile_y_index, i_index, j_index}, {Ai}, {tile_x, tile_y, i, j}, nullptr);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Add(Multiply(4i, 16j), 33i)");

    exprs = ct.reverse(
        {tile_x_index, tile_y_index, i_index, j_index}, {Ai}, {tile_x, tile_y, i, j}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "97j");

    exprs = ct.reverse(
        {tile_x_index, tile_y_index, i_index, j_index}, {A}, {tile_x, tile_y, i, j}, nullptr);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr,
              "Add(Multiply(Add(Multiply(4i, 16j), 33i), 300i), Multiply(Add(Multiply(5i, 16j), "
              "2i), 1i))");

    exprs = ct.reverse(
        {tile_x_index, tile_y_index, i_index, j_index}, {A}, {tile_x, tile_y, i, j}, fastArith);
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "29182j");
}

TEST_F(CoordinateTransformTest, TensorTile2DLoadStore02)
{
    // one tile per wavefront

    auto ct = CoordinateTransform::HyperGraph();

    auto wavefront_size = Expression::literal(64);
    auto unit           = Expression::literal(1);

    auto M = Expression::literal(100);
    auto N = Expression::literal(200);
    auto K = Expression::literal(300);

    int m = 16;
    int n = 16;

    // A matrix; tag 1; M x K; C-ordering
    auto A  = CoordinateTransform::User(1);
    auto Ai = CoordinateTransform::SubDimension(1, 0, M, K);
    auto Aj = CoordinateTransform::SubDimension(1, 1, K, unit);
    ct.addEdge({A}, {Ai, Aj}, CoordinateTransform::Split());

    // T tile; tag 3; m x n
    auto T = CoordinateTransform::MacroTile(3, {m, n}, MemoryType::VGPR);
    auto i = T.tileIndex(0);
    auto j = T.tileIndex(1);
    ct.addEdge({i, j}, {T}, CoordinateTransform::Join());

    // tile each dimension of A matrix; each workgroup gets one tile
    auto tile_x = CoordinateTransform::Workgroup(1, 0);
    auto tile_y = CoordinateTransform::Workgroup(1, 1);
    ct.addEdge({Ai}, {tile_x, i}, CoordinateTransform::Tile());
    ct.addEdge({Aj}, {tile_y, j}, CoordinateTransform::Tile());

    auto coords
        = CoordinateTransform::Transformer(std::make_shared<CoordinateTransform::HyperGraph>(ct));

    auto identity_transducer = [&](auto expr) { return expr; };
    auto transducer          = [&](auto expr) { return Expression::fastMultiplication(expr); };

    std::vector<Expression::ExpressionPtr> exprs;
    std::string                            sexpr;

    // set location to thread tile
    coords.setCoordinate(tile_x, Expression::literal(4));
    coords.setCoordinate(tile_y, Expression::literal(5));
    coords.setCoordinate(i, Expression::literal(33));
    coords.setCoordinate(j, Expression::literal(2));

    // from thread tile to macro tile row
    exprs = coords.reverse({Ai});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Add(Multiply(4i, 16j), 33i)");

    // from thread tile to user tensor
    exprs = coords.reverse({A});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr,
              "Add(Multiply(Add(Multiply(4i, 16j), 33i), 300i), Multiply(Add(Multiply(5i, 16j), "
              "2i), 1i))");

    // check edge case where expression are unchanged
    coords.setTransducer(identity_transducer);
    exprs = coords.reverse({A});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr,
              "Add(Multiply(Add(Multiply(4i, 16j), 33i), 300i), Multiply(Add(Multiply(5i, 16j), "
              "2i), 1i))");
    coords.setTransducer(nullptr);

    // as above, with fast multiplication
    coords.setTransducer(transducer);
    exprs = coords.reverse({A});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Add(Multiply(Add(ShiftL(4i, 4j), 33i), 300i), Add(ShiftL(5i, 4j), 2i))");
    coords.setTransducer(nullptr);

    // remove i, try again: should fail
    coords.removeCoordinate(i);
    EXPECT_THROW(coords.reverse({A}), RecoverableError);

    // remove i and j, so only know workgroup and workitem
    coords.removeCoordinate(i);
    coords.removeCoordinate(j);

    // from user tensor row to thread tile row
    coords.setCoordinate(Ai, Expression::literal(17));
    exprs = coords.forward({i});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "Modulo(Divide(17i, 300i), 16j)");
}

TEST_F(CoordinateTransformTest, TensorTile2DLoadStore03)
{
    auto ct = CoordinateTransform::HyperGraph();

    auto unit = Expression::literal(1u);
    auto M    = Expression::literal(100u);
    auto K    = Expression::literal(300u);

    int m = 16;
    int n = 16;

    // A matrix; tag 1; M x K; C-ordering
    auto A  = CoordinateTransform::User(1);
    auto Ai = CoordinateTransform::SubDimension(1, 0, M, K);
    auto Aj = CoordinateTransform::SubDimension(1, 1, K, unit);
    ct.addEdge({A}, {Ai, Aj}, CoordinateTransform::Split());

    // T tile; tag 3; m x n
    auto T = CoordinateTransform::MacroTile(3, {m, n}, MemoryType::VGPR);
    auto i = T.tileIndex(0);
    auto j = T.tileIndex(1);
    ct.addEdge({i, j}, {T}, CoordinateTransform::Join());

    // tile each dimension of A matrix; each workgroup gets one tile
    auto tile_x = CoordinateTransform::Workgroup(1, 0);
    auto tile_y = CoordinateTransform::Workgroup(1, 1);
    ct.addEdge({Ai}, {tile_x, i}, CoordinateTransform::Tile());
    ct.addEdge({Aj}, {tile_y, j}, CoordinateTransform::Tile());

    auto coords = CoordinateTransform::Transformer(
        std::make_shared<CoordinateTransform::HyperGraph>(ct), nullptr, Expression::simplify);

    coords.setCoordinate(tile_x, Expression::literal(4u));
    coords.setCoordinate(tile_y, Expression::literal(5u));
    coords.setCoordinate(i, Expression::literal(33u));
    coords.setCoordinate(j, Expression::literal(2u));

    auto exprs = coords.reverseStride(tile_x, Expression::literal(1u), {A});
    auto sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "4800j");

    exprs = coords.reverseStride(i, Expression::literal(2u), {A});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "600j");

    exprs = coords.reverseStride(j, Expression::literal(1u), {A});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "1j");

    exprs = coords.reverseStride(tile_y, Expression::literal(2u), {A});
    sexpr = Expression::toString(exprs[0]);
    EXPECT_EQ(sexpr, "32j");
}

TEST_F(CoordinateTransformTest, WaveTileBasic)
{
    // one tile per wavefront
    auto ct = CoordinateTransform::HyperGraph();

    auto WaveA = CoordinateTransform::WaveTile(0, {32u, 2u}, LayoutType::MATRIX_A);

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

TEST_F(CoordinateTransformTest, SStreamTags)
{
    std::vector<KernelGraph::TagType> tagVector{{1, 1, false, 1, 0}, {2, 1, true, 1, 1}};
    std::set<KernelGraph::TagType>    tagSet(tagVector.begin(), tagVector.end());

    std::ostringstream os;
    os << tagSet;
    EXPECT_EQ("set{{1, 1, i}, {2, 1, o}}", os.str());

    os = std::ostringstream();
    os << tagVector;
    EXPECT_EQ("[{1, 1, i}, {2, 1, o}]", os.str());
}
