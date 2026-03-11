
using Builder      = ckb::ConvBuilder<SIGNATURE, ALGORITHM>;
using ConvInstance = Builder::Instance;

using ElementwiseOpBuilder  = ckf::ElementwiseOpTileFactory<SIGNATURE, ALGORITHM>;
using ElementwiseOpInstance = ElementwiseOpBuilder::Instance;

auto conv = ConvInstance{};
ckt::RunResult result{};

if constexpr(ALGORITHM.optimizations.two_stage)
{
    auto elementwise_op = ElementwiseOpInstance{};
    result              = ckt::run(conv, elementwise_op, args, inputs, outputs, s_conf);
}
else
{
    result = ckt::run(conv, args, inputs, outputs, s_conf);
}

return std::make_tuple(result.is_supported(), result.runtime, conv.GetInstanceString());
