// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "CaseGraph.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <optional>
#include <set>
#include <sstream>
#include <tuple>
#include <utility>
#include <variant>

#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <nlohmann/json.hpp>

namespace sdpa_case_flow
{
namespace
{

using hipdnn_frontend::AttentionImplementation;
using hipdnn_frontend::DataType;
using hipdnn_frontend::DiagonalAlignment;
using hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::SdpaAttributes;
using hipdnn_frontend::graph::TensorAttributes;
using TensorPtr = std::shared_ptr<TensorAttributes>;
using Dims = std::vector<int64_t>;

[[noreturn]] void fail(const std::string& message)
{
    throw CaseRealisationError(message);
}

const std::string& valueOf(const Row& row, const std::string& axis)
{
    const auto it = row.find(axis);
    if(it == row.end())
    {
        fail("row has no column for axis '" + axis + "'");
    }
    return it->second;
}

bool isActive(const Row& row, const std::string& axis)
{
    const auto& value = valueOf(row, axis);
    return !value.empty() && value != "-";
}

[[noreturn]] void unknownValue(const std::string& axis, const std::string& value)
{
    fail("value '" + value + "' of axis '" + axis + "' is not in the class-to-value table");
}

template <typename T>
const T& classValue(const std::map<std::string, T>& table,
                    const std::string& axis,
                    const std::string& id)
{
    const auto it = table.find(id);
    if(it == table.end())
    {
        unknownValue(axis, id);
    }
    return it->second;
}

std::string numberText(double value)
{
    std::ostringstream os;
    os << std::setprecision(9) << value;
    return os.str();
}

const char* dataTypeName(DataType type)
{
    switch(type)
    {
    case DataType::FLOAT:
        return "FLOAT";
    case DataType::HALF:
        return "HALF";
    case DataType::BFLOAT16:
        return "BFLOAT16";
    case DataType::FP8_E4M3:
        return "FP8_E4M3";
    case DataType::FP8_E4M3_FNUZ:
        return "FP8_E4M3_FNUZ";
    case DataType::FP8_E5M2:
        return "FP8_E5M2";
    case DataType::FP8_E5M2_FNUZ:
        return "FP8_E5M2_FNUZ";
    case DataType::INT32:
        return "INT32";
    case DataType::INT64:
        return "INT64";
    case DataType::UINT8:
        return "UINT8";
    default:
        return "OTHER";
    }
}

DataType formatOf(const std::string& axis, const std::string& id)
{
    if(id == "fp32")
    {
        return DataType::FLOAT;
    }
    if(id == "fp16")
    {
        return DataType::HALF;
    }
    if(id == "bf16")
    {
        return DataType::BFLOAT16;
    }
    if(id == "fp8_e4m3")
    {
        return DataType::FP8_E4M3;
    }
    if(id == "fp8_e4m3_fnuz")
    {
        return DataType::FP8_E4M3_FNUZ;
    }
    if(id == "fp8_e5m2")
    {
        return DataType::FP8_E5M2;
    }
    if(id == "fp8_e5m2_fnuz")
    {
        return DataType::FP8_E5M2_FNUZ;
    }
    unknownValue(axis, id);
}

bool isFp8(DataType type)
{
    return type == DataType::FP8_E4M3 || type == DataType::FP8_E4M3_FNUZ
           || type == DataType::FP8_E5M2 || type == DataType::FP8_E5M2_FNUZ;
}

int64_t elementBytes(DataType type)
{
    switch(type)
    {
    case DataType::FLOAT:
    case DataType::INT32:
        return 4;
    case DataType::HALF:
    case DataType::BFLOAT16:
        return 2;
    case DataType::INT64:
        return 8;
    default:
        return 1;
    }
}

DataType relatedFormat(const Row& row,
                       const std::string& axis,
                       DataType qFormat,
                       const ClassValues& values,
                       const std::string& arch)
{
    const auto& id = valueOf(row, axis);
    if(id == "same")
    {
        return qFormat;
    }
    if(id == "differs")
    {
        std::string format = classValue(values.differsFormat, "dtype_q", valueOf(row, "dtype_q"));
        const auto archIt = values.archFormat.find(arch);
        if(archIt != values.archFormat.end())
        {
            const auto substitute = archIt->second.find(format);
            if(substitute != archIt->second.end())
            {
                format = substitute->second;
            }
        }
        return formatOf(axis, format);
    }
    unknownValue(axis, id);
}

std::pair<int64_t, int64_t> sequenceLengths(const Row& row,
                                            const std::vector<SequenceLengthEntry>& table)
{
    const auto& q = valueOf(row, "seq_q");
    const auto& kv = valueOf(row, "seq_kv");
    const auto& rel = valueOf(row, "seq_q_vs_seq_kv");
    for(const auto& entry : table)
    {
        if(q == entry.seqQ && kv == entry.seqKv && rel == entry.relation)
        {
            return {entry.sq, entry.skv};
        }
    }
    fail("seq_q=" + q + ", seq_kv=" + kv + ", seq_q_vs_seq_kv=" + rel
         + " is not in the class-to-value table");
}

int64_t headsFor(const std::string& axis, const std::string& sharing, int64_t qHeads)
{
    if(sharing == "mha")
    {
        return qHeads;
    }
    if(sharing == "gqa")
    {
        if(qHeads < 8)
        {
            fail(axis + "=gqa needs at least 8 query heads; the case has "
                 + std::to_string(qHeads));
        }
        return qHeads / 4;
    }
    if(sharing == "mqa")
    {
        return 1;
    }
    if(sharing == "non_divisor")
    {
        return qHeads == 8 ? 3 : 2;
    }
    unknownValue(axis, sharing);
}

Dims packedStrides(const Dims& dims)
{
    Dims strides(dims.size(), 1);
    int64_t running = 1;
    for(size_t i = dims.size(); i-- > 0;)
    {
        strides[i] = running;
        running *= std::max<int64_t>(dims[i], 1);
    }
    return strides;
}

// Logical dims are always [B, H, S, D]; `physical` lists them outermost first.
Dims stridesFor(const Dims& dims,
                const std::vector<size_t>& physical,
                const std::string& strideAxis,
                const std::string& strideClass)
{
    Dims extents = dims;
    if(strideClass == "non_contiguous_view")
    {
        extents[1] *= 2;
    }

    Dims strides(4, 0);
    int64_t running = 1;
    for(size_t i = physical.size(); i-- > 0;)
    {
        const size_t dim = physical[i];
        strides[dim] = running;
        int64_t extent = extents[dim];
        if(strideClass == "padded" && i == physical.size() - 1)
        {
            extent += 16;
        }
        running *= extent;
    }

    if(strideClass == "packed" || strideClass == "padded")
    {
        return strides;
    }
    if(strideClass == "non_contiguous_view")
    {
        strides[1] *= 2;
        return strides;
    }
    if(strideClass == "zero_broadcast" || strideClass == "negative")
    {
        static const size_t SEARCH_ORDER[] = {2, 1, 0, 3};
        for(const size_t dim : SEARCH_ORDER)
        {
            if(dims[dim] > 1)
            {
                strides[dim] = strideClass == "zero_broadcast" ? 0 : -strides[dim];
                return strides;
            }
        }
        fail(strideAxis + "=" + strideClass + " needs a dimension with extent > 1");
    }
    if(strideClass == "self_overlapping")
    {
        static const size_t DIMENSION_ORDER[] = {2, 1, 0};
        static const size_t PARTNER_ORDER[] = {3, 2, 1, 0};
        for(const size_t dim : DIMENSION_ORDER)
        {
            if(dims[dim] <= 1)
            {
                continue;
            }
            for(const size_t partner : PARTNER_ORDER)
            {
                if(partner != dim && dims[partner] > 1)
                {
                    // Index 1 of `dim` and index dims[partner] / 2 of `partner` reach one element.
                    strides[dim] = dims[partner] / 2 * strides[partner];
                    return strides;
                }
            }
        }
        fail(strideAxis + "=self_overlapping needs two dimensions with extent > 1");
    }
    unknownValue(strideAxis, strideClass);
}

void applyAlignment(TensorAttributes& tensor,
                    const std::string& axis,
                    const std::string& alignment,
                    DataType type)
{
    if(alignment == "wide_aligned")
    {
        return;
    }
    if(alignment == "element_aligned")
    {
        tensor.set_alignment(elementBytes(type));
        return;
    }
    if(alignment == "sub_element")
    {
        if(elementBytes(type) == 1)
        {
            fail(axis + "=sub_element is impossible for a 1-byte format");
        }
        tensor.set_alignment(1);
        return;
    }
    unknownValue(axis, alignment);
}

int64_t ceilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

int64_t checkedProduct(int64_t a, int64_t b, const std::string& what)
{
    if(a != 0 && b != 0
       && (a > std::numeric_limits<int64_t>::max() / std::abs(b)
           || a < -(std::numeric_limits<int64_t>::max() / std::abs(b))))
    {
        fail(what + " does not fit in 64 bits");
    }
    return a * b;
}

int64_t elementCount(const Dims& dims)
{
    int64_t count = 1;
    for(const auto d : dims)
    {
        count *= d;
    }
    return count;
}

Contents integerContents(std::vector<int64_t> values, bool repeat)
{
    return {std::move(values), repeat};
}

Contents realContents(std::vector<double> values)
{
    return {std::move(values), true};
}

std::string dimsText(const Dims& dims)
{
    std::string text;
    for(size_t i = 0; i < dims.size(); ++i)
    {
        text += (i == 0 ? "" : "x") + std::to_string(dims[i]);
    }
    return text;
}

struct Overrides
{
    bool scaleUnset = false;
    std::optional<float> attnScale;
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
    std::optional<bool> deprecatedCausalMask;
    std::optional<bool> deprecatedCausalMaskBottomRight;
    std::string causalRoute = "setter";
    std::optional<int64_t> maxSeqLenKv;
    std::optional<bool> paddingMask;
    std::optional<bool> generateStats;
    std::optional<float> dropoutProbability;
    bool addSeqLenQ = false;
    bool addSeqLenKv = false;
    bool addDropoutMask = false;
    bool addDropoutScale = false;
    bool addSeedOffset = false;
    bool addRngDump = false;
    bool statsWithoutGenerate = false;
    bool setDefaultsExplicitly = false;
    std::map<std::string, Dims> tensorDims;
};

int64_t parseInt(const std::string& token, const std::string& text)
{
    try
    {
        size_t used = 0;
        const auto value = std::stoll(text, &used);
        if(used == text.size())
        {
            return value;
        }
    }
    catch(const std::exception&)
    {
    }
    fail("override '" + token + "' needs an integer");
}

float parseFloat(const std::string& token, const std::string& text)
{
    try
    {
        size_t used = 0;
        const auto value = std::stof(text, &used);
        if(used == text.size())
        {
            return value;
        }
    }
    catch(const std::exception&)
    {
    }
    fail("override '" + token + "' needs a number");
}

bool parseBool(const std::string& token, const std::string& text)
{
    if(text == "1" || text == "true")
    {
        return true;
    }
    if(text == "0" || text == "false")
    {
        return false;
    }
    fail("override '" + token + "' needs 0 or 1");
}

Dims parseDims(const std::string& token, const std::string& text)
{
    Dims dims;
    std::stringstream stream(text);
    std::string part;
    while(std::getline(stream, part, 'x'))
    {
        dims.push_back(parseInt(token, part));
    }
    if(dims.empty())
    {
        fail("override '" + token + "' needs dims such as 2x8x256x1");
    }
    return dims;
}

Overrides parseOverrides(const std::vector<std::string>& tokens)
{
    static const std::set<std::string> DIM_TARGETS = {"bias",
                                                      "stats",
                                                      "logit_max",
                                                      "score_sum_exp",
                                                      "sink",
                                                      "block_mask",
                                                      "seq_len_q",
                                                      "seq_len_kv",
                                                      "page_table_k",
                                                      "page_table_v",
                                                      "dropout_mask",
                                                      "rng_dump"};
    static const std::string DIMS_SUFFIX = "_dims";

    Overrides ov;
    for(const auto& token : tokens)
    {
        const auto eq = token.find('=');
        const std::string key = token.substr(0, eq);
        const std::string value = eq == std::string::npos ? "" : token.substr(eq + 1);
        const bool hasValue = eq != std::string::npos;

        if(!hasValue)
        {
            if(key == "scale_unset")
            {
                ov.scaleUnset = true;
            }
            else if(key == "add_seq_len_q")
            {
                ov.addSeqLenQ = true;
            }
            else if(key == "add_seq_len_kv")
            {
                ov.addSeqLenKv = true;
            }
            else if(key == "add_dropout_mask")
            {
                ov.addDropoutMask = true;
            }
            else if(key == "add_dropout_scale")
            {
                ov.addDropoutScale = true;
            }
            else if(key == "add_seed_offset")
            {
                ov.addSeedOffset = true;
            }
            else if(key == "add_rng_dump")
            {
                ov.addRngDump = true;
            }
            else if(key == "stats_without_generate")
            {
                ov.statsWithoutGenerate = true;
            }
            else if(key == "set_defaults_explicitly")
            {
                ov.setDefaultsExplicitly = true;
            }
            else
            {
                fail("unknown override '" + token + "'");
            }
            continue;
        }

        if(key == "attn_scale")
        {
            ov.attnScale = parseFloat(token, value);
        }
        else if(key == "left_bound")
        {
            ov.leftBound = parseInt(token, value);
        }
        else if(key == "right_bound")
        {
            ov.rightBound = parseInt(token, value);
        }
        else if(key == "deprecated_causal_mask")
        {
            ov.deprecatedCausalMask = parseBool(token, value);
        }
        else if(key == "deprecated_causal_mask_bottom_right")
        {
            ov.deprecatedCausalMaskBottomRight = parseBool(token, value);
        }
        else if(key == "causal_route")
        {
            if(value != "setter" && value != "band" && value != "deprecated_member")
            {
                fail("override '" + token + "' needs setter, band or deprecated_member");
            }
            ov.causalRoute = value;
        }
        else if(key == "max_seq_len_kv")
        {
            ov.maxSeqLenKv = parseInt(token, value);
        }
        else if(key == "padding_mask")
        {
            ov.paddingMask = parseBool(token, value);
        }
        else if(key == "generate_stats")
        {
            ov.generateStats = parseBool(token, value);
        }
        else if(key == "dropout_probability")
        {
            ov.dropoutProbability = parseFloat(token, value);
        }
        else if(key.size() > DIMS_SUFFIX.size()
                && key.compare(key.size() - DIMS_SUFFIX.size(), DIMS_SUFFIX.size(), DIMS_SUFFIX)
                       == 0
                && DIM_TARGETS.count(key.substr(0, key.size() - DIMS_SUFFIX.size())) != 0)
        {
            ov.tensorDims[key.substr(0, key.size() - DIMS_SUFFIX.size())] = parseDims(token, value);
        }
        else
        {
            fail("unknown override '" + token + "'");
        }
    }
    return ov;
}

class CaseBuilder
{
public:
    CaseBuilder(const Row& row, const Overrides& ov, const CaseContext& context)
        : _row(row)
        , _ov(ov)
        , _ctx(context)
        , _values(context.values)
    {
    }

    BuiltCase build()
    {
        resolveSizes();
        resolveFormats();

        auto graph = std::make_shared<Graph>();
        graph->set_name("sdpa_case_flow")
            .set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(_compute);

        if(valueOf(_row, "runtime_shape_override") == "enabled")
        {
            graph->set_override_shape_enabled(true);
        }
        else if(valueOf(_row, "runtime_shape_override") != "disabled")
        {
            unknownValue("runtime_shape_override", valueOf(_row, "runtime_shape_override"));
        }

        SdpaAttributes attrs;
        attrs.set_name("sdpa_fwd");
        attrs.set_compute_data_type(_compute);
        if(_ov.setDefaultsExplicitly)
        {
            setDefaultsExplicitly(attrs);
        }

        buildInputs();
        applyPrecision(attrs);
        applySequenceLengths(attrs);
        applyMasks(attrs);
        applyScoreTerms(attrs);
        applyScale(attrs);
        applyDropout(attrs);
        applyOptionalOutputs(attrs);
        applyQuantisationTensors(attrs);
        applyPaging(attrs);
        applyImplementation(attrs);
        applyAttributeOverrides(attrs);

        auto [o, stats] = graph->sdpa(_q, _k, _v, attrs);
        configureOutput(o);
        configureStats(stats);
        applyDimOverrides();
        addStructuralContents();

        return {graph, realisedText(), _contents};
    }

private:
    const Row& _row;
    const Overrides& _ov;
    const CaseContext& _ctx;
    const ClassValues& _values;

    int64_t _b = 0;
    int64_t _hq = 0;
    int64_t _hk = 0;
    int64_t _hv = 0;
    int64_t _sq = 0;
    int64_t _skv = 0;
    int64_t _dqk = 0;
    int64_t _dv = 0;

    DataType _fq = DataType::NOT_SET;
    DataType _fk = DataType::NOT_SET;
    DataType _fv = DataType::NOT_SET;
    DataType _fo = DataType::NOT_SET;
    DataType _compute = DataType::NOT_SET;

    TensorPtr _q;
    TensorPtr _k;
    TensorPtr _v;
    std::map<std::string, TensorPtr> _named;
    std::vector<std::string> _notes;
    std::map<std::string, Contents> _contents;
    int64_t _nextUid = 1;

    TensorPtr newTensor(const std::string& name)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_name(name).set_uid(_nextUid++);
        _named[name] = tensor;
        return tensor;
    }

    TensorPtr packedTensor(const std::string& name, const Dims& dims, DataType type)
    {
        auto tensor = newTensor(name);
        tensor->set_dim(dims).set_stride(packedStrides(dims)).set_data_type(type);
        return tensor;
    }

    TensorPtr outputTensor(const std::string& name, const Dims& dims, DataType type)
    {
        auto tensor = packedTensor(name, dims, type);
        tensor->set_output(true);
        return tensor;
    }

    TensorPtr byValueFloat(const std::string& name, float value)
    {
        auto tensor = newTensor(name);
        tensor->set_value(value);
        return tensor;
    }

    TensorPtr byValueInt64(const std::string& name, int64_t value)
    {
        auto tensor = newTensor(name);
        tensor->set_value(value);
        return tensor;
    }

    TensorPtr scalarFactor(const std::string& name)
    {
        if(_ctx.descaleForm == DescaleForm::BY_VALUE)
        {
            const bool ownEntry = name == "descale_q" || name == "descale_k" || name == "descale_v";
            const auto factor = static_cast<float>(
                classValue(_values.quantisationFactors, name, ownEntry ? name : "other"));
            _notes.push_back(name + "=" + numberText(static_cast<double>(factor)));
            return byValueFloat(name, factor);
        }
        return packedTensor(name, {1, 1, 1, 1}, DataType::FLOAT);
    }

    void resolveSizes()
    {
        if(_ctx.sizes.has_value())
        {
            const auto& z = *_ctx.sizes;
            _b = z.b;
            _hq = z.hq;
            _hk = z.hk;
            _hv = z.hv;
            _sq = z.sq;
            _skv = z.skv;
            _dqk = z.dqk;
            _dv = z.dv;
            return;
        }
        _b = classValue(_values.batchSize, "batch_size", valueOf(_row, "batch_size"));
        _hq = classValue(_values.qHeads, "q_heads", valueOf(_row, "q_heads"));
        std::tie(_sq, _skv) = sequenceLengths(_row, _values.sequenceLengths);
        _dqk = classValue(_values.headDim, "head_dim_qk", valueOf(_row, "head_dim_qk"));
        _dv = classValue(_values.headDim, "head_dim_v", valueOf(_row, "head_dim_v"));

        const auto& kSharing = valueOf(_row, "k_head_sharing");
        const auto& vSharing = valueOf(_row, "v_head_sharing");
        const auto& kvRelation = valueOf(_row, "k_heads_vs_v_heads");
        _hk = headsFor("k_head_sharing", kSharing, _hq);
        _hv = headsFor("v_head_sharing", vSharing, _hq);
        if(kSharing == "gqa" && vSharing == "gqa" && kvRelation == "differ")
        {
            _hk = 2;
            _hv = 4;
        }
        if(kvRelation == "equal" && _hk != _hv)
        {
            fail("k_heads_vs_v_heads=equal but the sharing values give H_k=" + std::to_string(_hk)
                 + ", H_v=" + std::to_string(_hv));
        }
        if(kvRelation == "differ" && _hk == _hv)
        {
            fail("k_heads_vs_v_heads=differ but the sharing values give H_k = H_v = "
                 + std::to_string(_hk));
        }
        if(kvRelation != "equal" && kvRelation != "differ")
        {
            unknownValue("k_heads_vs_v_heads", kvRelation);
        }
        if(isActive(_row, "alibi_head_count"))
        {
            applyAlibiHeadCount(kSharing, vSharing, kvRelation);
        }
    }

    // Replaces the power-of-two query head count, and the K and V counts that follow from it.
    void applyAlibiHeadCount(const std::string& kSharing,
                             const std::string& vSharing,
                             const std::string& kvRelation)
    {
        const auto& id = valueOf(_row, "alibi_head_count");
        if(id == "power_of_two")
        {
            return;
        }
        if(id != "not_power_of_two")
        {
            unknownValue("alibi_head_count", id);
        }
        const auto it = _values.alibiHeads.find(_hq);
        if(it == _values.alibiHeads.end())
        {
            fail("alibi_head_count=not_power_of_two has no head count for H_q="
                 + std::to_string(_hq));
        }
        const auto& heads = it->second;
        const auto pick = [&](const std::string& axis, const std::string& sharing) {
            if(sharing == "mha")
            {
                return heads.mha;
            }
            if(sharing == "gqa")
            {
                return heads.gqa;
            }
            if(sharing == "mqa")
            {
                return heads.mqa;
            }
            unknownValue(axis, sharing);
        };
        _hq = heads.hq;
        _hk = pick("k_head_sharing", kSharing);
        _hv = pick("v_head_sharing", vSharing);
        if(kSharing == "gqa" && vSharing == "gqa" && kvRelation == "differ")
        {
            _hk = heads.differK;
            _hv = heads.differV;
        }
    }

    void resolveFormats()
    {
        _fq = formatOf("dtype_q", valueOf(_row, "dtype_q"));
        _fk = relatedFormat(_row, "dtype_k_vs_q", _fq, _values, _ctx.arch);
        _fv = relatedFormat(_row, "dtype_v_vs_q", _fq, _values, _ctx.arch);
        _fo = formatOf("dtype_o", valueOf(_row, "dtype_o"));

        const auto& compute = valueOf(_row, "compute_precision");
        if(compute == "fp32")
        {
            _compute = DataType::FLOAT;
        }
        else if(compute == "fp16")
        {
            _compute = DataType::HALF;
        }
        else if(compute == "bf16")
        {
            _compute = DataType::BFLOAT16;
        }
        else
        {
            unknownValue("compute_precision", compute);
        }
    }

    Dims layoutStrides(const Dims& dims, const std::string& suffix) const
    {
        const std::string orderAxis = "dim_order_" + suffix;
        const std::string strideAxis = "strides_" + suffix;
        return stridesFor(dims,
                          classValue(_values.dimOrder, orderAxis, valueOf(_row, orderAxis)),
                          strideAxis,
                          valueOf(_row, strideAxis));
    }

    TensorPtr layoutTensor(const std::string& name,
                           const std::string& suffix,
                           const Dims& dims,
                           DataType type)
    {
        auto tensor = newTensor(name);
        tensor->set_dim(dims).set_stride(layoutStrides(dims, suffix)).set_data_type(type);
        applyAlignment(*tensor, "alignment_" + suffix, valueOf(_row, "alignment_" + suffix), type);
        return tensor;
    }

    void buildInputs()
    {
        Dims kDims = {_b, _hk, _skv, _dqk};
        Dims vDims = {_b, _hv, _skv, _dv};

        if(pagedAny())
        {
            const auto table = pagedTable();
            // A dense side beside a paged one spans the declared maximum key length.
            kDims[2] = table.declaredMaxSeqLenKv;
            vDims[2] = table.declaredMaxSeqLenKv;
            if(pagedSide("K"))
            {
                const int64_t pool = _b * ceilDiv(_skv, table.pageSizeK);
                kDims = {pool, _hk, table.pageSizeK, _dqk};
                _notes.push_back("k_page_pool=" + std::to_string(pool) + "x"
                                 + std::to_string(table.pageSizeK));
            }
            if(pagedSide("V"))
            {
                const int64_t pool = _b * ceilDiv(_skv, table.pageSizeV);
                vDims = {pool, _hv, table.pageSizeV, _dv};
                _notes.push_back("v_page_pool=" + std::to_string(pool) + "x"
                                 + std::to_string(table.pageSizeV));
            }
        }

        _q = layoutTensor("q", "q", {_b, _hq, _sq, _dqk}, _fq);
        _k = layoutTensor("k", "k", kDims, _fk);
        _v = layoutTensor("v", "v", vDims, _fv);

        std::string ragged;
        for(const auto& [letter, tensor] :
            std::vector<std::pair<std::string, TensorPtr>>{{"Q", _q}, {"K", _k}, {"V", _v}})
        {
            if(raggedTensor(letter))
            {
                tensor->set_ragged_offset(raggedOffsetFor(letter));
                ragged += letter;
            }
        }
        if(raggedTensor("O"))
        {
            ragged += "O";
        }
        if(!ragged.empty())
        {
            _notes.push_back("ragged=" + ragged);
        }
    }

    // Each ragged tensor has its own offset tensor: an offset counts that tensor's elements.
    TensorPtr raggedOffsetFor(const std::string& letter)
    {
        std::string name = "ragged_offset_";
        for(const char c : letter)
        {
            name += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
        return packedTensor(name, {_b + 1, 1, 1, 1}, DataType::INT32);
    }

    bool raggedTensor(const std::string& letter) const
    {
        return classValue(_values.raggedTensors, "batch_packing", valueOf(_row, "batch_packing"))
                   .count(letter)
               != 0;
    }

    bool pagedSide(const std::string& letter) const
    {
        const auto& id = valueOf(_row, "paged_kv");
        if(id == "absent")
        {
            return false;
        }
        return classValue(_values.pagedSides, "paged_kv", id).count(letter) != 0;
    }

    bool pagedAny() const
    {
        return pagedSide("K") || pagedSide("V");
    }

    // The key extent of score-shaped tensors: the declared maximum with paged K/V.
    int64_t keyExtent() const
    {
        return pagedAny() ? pagedTable().declaredMaxSeqLenKv : _skv;
    }

    // Page size and pages needed of the first paged side (K, else V).
    std::pair<int64_t, int64_t> pageGeometry() const
    {
        const auto& tokens = classValue(_values.pageSize, "page_size", valueOf(_row, "page_size"));
        const int64_t pageSize = tokens.value_or(_skv);
        if(pageSize <= 0)
        {
            fail("page_size=" + valueOf(_row, "page_size") + " gives " + std::to_string(pageSize)
                 + " tokens per page");
        }
        return {pageSize, ceilDiv(_skv, pageSize)};
    }

    void applyPrecision(SdpaAttributes& attrs)
    {
        const auto& mode = valueOf(_row, "matmul_input_precision");
        if(mode == "native")
        {
            return;
        }
        if(mode == "fp32")
        {
            attrs.set_mma_core_mode(DataType::FLOAT);
        }
        else if(mode == "fp16")
        {
            attrs.set_mma_core_mode(DataType::HALF);
        }
        else if(mode == "bf16")
        {
            attrs.set_mma_core_mode(DataType::BFLOAT16);
        }
        else if(mode == "fp8")
        {
            DataType type = _fq;
            if(!isFp8(type))
            {
                std::string format = _values.mmaFp8Otherwise;
                const auto archIt = _values.mmaFp8ByArch.find(_ctx.arch);
                if(archIt != _values.mmaFp8ByArch.end())
                {
                    const auto substitute = archIt->second.find(format);
                    if(substitute != archIt->second.end())
                    {
                        format = substitute->second;
                    }
                }
                type = formatOf("matmul_input_precision", format);
            }
            attrs.set_mma_core_mode(type);
            _notes.push_back(std::string("mma_core_mode=") + dataTypeName(type));
        }
        else
        {
            unknownValue("matmul_input_precision", mode);
        }
    }

    void applySequenceLengths(SdpaAttributes& attrs)
    {
        const auto& lengths = valueOf(_row, "sequence_lengths");
        if(lengths == "variable" || lengths == "variable_no_flag")
        {
            if(lengths == "variable")
            {
                attrs.set_padding_mask(true);
            }
            attrs.set_seq_len_q(packedTensor("seq_len_q", {_b, 1, 1, 1}, DataType::INT32));
            attrs.set_seq_len_kv(packedTensor("seq_len_kv", {_b, 1, 1, 1}, DataType::INT32));
        }
        else if(lengths == "lengths_in_offsets")
        {
            if(!raggedTensor("Q") && !raggedTensor("K") && !raggedTensor("V"))
            {
                fail("sequence_lengths=lengths_in_offsets needs a ragged tensor");
            }
        }
        else if(lengths != "uniform")
        {
            unknownValue("sequence_lengths", lengths);
        }
    }

    static DiagonalAlignment alignmentOf(const std::string& axis, const std::string& id)
    {
        if(id == "top_left")
        {
            return DiagonalAlignment::TOP_LEFT;
        }
        if(id == "bottom_right")
        {
            return DiagonalAlignment::BOTTOM_RIGHT;
        }
        unknownValue(axis, id);
    }

    // Keys kept on one side of the diagonal, not counting the diagonal key.
    int64_t windowKeys(const std::string& axis) const
    {
        const auto& id = valueOf(_row, axis);
        if(id == "zero")
        {
            return 0;
        }
        if(id == "partial")
        {
            return std::max<int64_t>(1, _skv / 4);
        }
        if(id == "whole_span")
        {
            return _skv;
        }
        unknownValue(axis, id);
    }

    void applyMasks(SdpaAttributes& attrs)
    {
        const bool causal = valueOf(_row, "causal_mask") == "present";
        const bool window = valueOf(_row, "sliding_window") == "present";

        if(causal)
        {
            const auto& alignmentId = valueOf(_row, "causal_alignment");
            const auto alignment = alignmentOf("causal_alignment", alignmentId);
            const bool topLeft = alignment == DiagonalAlignment::TOP_LEFT;
            if(_ov.causalRoute == "setter")
            {
                if(topLeft)
                {
                    attrs.set_causal_mask(true);
                }
                else
                {
                    attrs.set_causal_mask_bottom_right(true);
                }
            }
            else if(_ov.causalRoute == "band")
            {
                attrs.set_diagonal_alignment(alignment);
                attrs.set_diagonal_band_right_bound(0);
            }
            else
            {
                if(topLeft)
                {
                    attrs.causal_mask = true;
                }
                else
                {
                    attrs.causal_mask_bottom_right = true;
                }
            }
        }

        if(window)
        {
            const auto alignment
                = alignmentOf("window_alignment", valueOf(_row, "window_alignment"));
            if(causal
               && alignment != alignmentOf("causal_alignment", valueOf(_row, "causal_alignment")))
            {
                fail("causal_alignment and window_alignment differ; one node has one diagonal");
            }
            attrs.set_diagonal_alignment(alignment);
            // left_bound counts the diagonal key as well.
            const int64_t leftBound = windowKeys("window_left") + 1;
            attrs.set_diagonal_band_left_bound(leftBound);
            _notes.push_back("left_bound=" + std::to_string(leftBound));
            if(!causal)
            {
                const int64_t rightBound = windowKeys("window_right");
                attrs.set_diagonal_band_right_bound(rightBound);
                _notes.push_back("right_bound=" + std::to_string(rightBound));
            }
        }

        if(valueOf(_row, "block_mask") == "present")
        {
            applyBlockMask(attrs);
        }
    }

    // One bit per 128 x 128 score tile, eight key tiles per byte, least significant bit first; a
    // clear bit masks the tile.
    void applyBlockMask(SdpaAttributes& attrs)
    {
        constexpr int64_t TILE = 128;
        const int64_t rowTiles = ceilDiv(_sq, TILE);
        const int64_t keyTiles = ceilDiv(keyExtent(), TILE);
        const int64_t bytes = ceilDiv(keyTiles, 8);
        attrs.set_block_mask(
            packedTensor("block_mask", {_b, _hq, rowTiles, bytes}, DataType::UINT8));

        const auto& pattern = valueOf(_row, "block_mask_pattern");
        const auto allowed = [&](int64_t rowTile, int64_t keyTile) {
            if(pattern == "all_open")
            {
                return true;
            }
            if(pattern == "mixed")
            {
                return keyTile == 0 || (rowTile + keyTile) % 2 == 0;
            }
            if(pattern == "some_rows_blocked")
            {
                return rowTile != rowTiles - 1;
            }
            if(pattern == "all_blocked")
            {
                return false;
            }
            unknownValue("block_mask_pattern", pattern);
        };
        std::vector<int64_t> contents;
        contents.reserve(static_cast<size_t>(_b * _hq * rowTiles * bytes));
        for(int64_t bh = 0; bh < _b * _hq; ++bh)
        {
            for(int64_t r = 0; r < rowTiles; ++r)
            {
                for(int64_t byte = 0; byte < bytes; ++byte)
                {
                    int64_t value = 0;
                    for(int64_t bit = 0; bit < 8 && byte * 8 + bit < keyTiles; ++bit)
                    {
                        if(allowed(r, byte * 8 + bit))
                        {
                            value |= int64_t{1} << bit;
                        }
                    }
                    contents.push_back(value);
                }
            }
        }
        _contents["block_mask"] = integerContents(std::move(contents), false);
    }

    Dims biasDims() const
    {
        const auto& id = valueOf(_row, "bias_broadcast");
        const int64_t keys = keyExtent();
        if(id == "none")
        {
            return {_b, _hq, _sq, keys};
        }
        if(id == "batch_or_heads")
        {
            return {1, 1, _sq, keys};
        }
        if(id == "batch_only")
        {
            return {1, _hq, _sq, keys};
        }
        if(id == "heads_only")
        {
            return {_b, 1, _sq, keys};
        }
        if(id == "sequence")
        {
            return {_b, _hq, 1, keys};
        }
        if(id == "all")
        {
            return {1, 1, 1, 1};
        }
        if(id == "not_broadcastable")
        {
            return {_b + 1, _hq, _sq, keys};
        }
        unknownValue("bias_broadcast", id);
    }

    void applyScoreTerms(SdpaAttributes& attrs)
    {
        if(valueOf(_row, "bias") == "present")
        {
            attrs.set_bias(packedTensor(
                "bias", biasDims(), formatOf("bias_dtype", valueOf(_row, "bias_dtype"))));
        }
        if(valueOf(_row, "alibi") == "present")
        {
            attrs.set_alibi_mask(true);
        }
        if(valueOf(_row, "attention_sinks") == "present")
        {
            attrs.set_sink_token(packedTensor("sink", {1, _hq, 1, 1}, DataType::FLOAT));
        }
    }

    float scaleNumber()
    {
        const auto& id = valueOf(_row, "scale_value");
        const auto& representatives = classValue(_values.scaleValue, "scale_value", id);
        const auto& primary = representatives.front();
        const float scale = primary.inverseSqrtHeadDim ? 1.0f / std::sqrt(static_cast<float>(_dqk))
                                                       : static_cast<float>(primary.value);
        if(!_ov.attnScale.has_value())
        {
            _notes.push_back("attn_scale=" + numberText(static_cast<double>(scale)));
            if(representatives.size() > 1)
            {
                _notes.push_back("scale_representative=1/"
                                 + std::to_string(representatives.size()));
            }
        }
        return scale;
    }

    void applyScale(SdpaAttributes& attrs)
    {
        if(!_ov.scaleUnset)
        {
            const auto& source = valueOf(_row, "scale_source");
            if(source == "attribute_value")
            {
                attrs.set_attn_scale(scaleNumber());
            }
            else if(source == "by_value_constant_tensor")
            {
                attrs.set_attn_scale(byValueFloat("attn_scale", scaleNumber()));
            }
            else if(source == "runtime_parameter")
            {
                auto tensor = newTensor("attn_scale");
                tensor->set_dim({1}).set_stride({1}).set_data_type(DataType::FLOAT);
                tensor->set_as_runtime_parameter();
                attrs.set_attn_scale(tensor);
                _contents["attn_scale"] = realContents({static_cast<double>(scaleNumber())});
            }
            else if(source == "device_tensor")
            {
                attrs.set_attn_scale(packedTensor("attn_scale", {1, 1, 1, 1}, DataType::FLOAT));
                _contents["attn_scale"] = realContents({static_cast<double>(scaleNumber())});
            }
            else
            {
                unknownValue("scale_source", source);
            }
        }
        if(_ov.attnScale.has_value())
        {
            attrs.set_attn_scale(*_ov.attnScale);
            _contents.erase("attn_scale");
        }
    }

    float dropoutProbability() const
    {
        return static_cast<float>(classValue(_values.dropoutProbability,
                                             "dropout_probability",
                                             valueOf(_row, "dropout_probability")));
    }

    float dropoutMaskScale() const
    {
        return static_cast<float>(classValue(_values.dropoutMaskScale,
                                             "dropout_mask_pattern",
                                             valueOf(_row, "dropout_mask_pattern")));
    }

    TensorPtr scoreShaped(const std::string& name, DataType type)
    {
        return packedTensor(name, {_b, _hq, _sq, keyExtent()}, type);
    }

    Dims dropoutMaskDims() const
    {
        const auto& id = valueOf(_row, "dropout_mask_broadcast");
        const int64_t keys = keyExtent();
        if(id == "none")
        {
            return {_b, _hq, _sq, keys};
        }
        if(id == "batch_only")
        {
            return {1, _hq, _sq, keys};
        }
        if(id == "heads_only")
        {
            return {_b, 1, _sq, keys};
        }
        if(id == "batch_and_heads")
        {
            return {1, 1, _sq, keys};
        }
        unknownValue("dropout_mask_broadcast", id);
    }

    void applyDropout(SdpaAttributes& attrs)
    {
        const auto& style = valueOf(_row, "dropout");
        if(style == "absent")
        {
            return;
        }
        if(style == "random")
        {
            const float probability = dropoutProbability();
            const auto& rng = valueOf(_row, "dropout_rng_state");
            if(rng == "caller_supplied")
            {
                attrs.set_dropout(probability,
                                  byValueInt64("dropout_seed", _values.dropoutSeed),
                                  byValueInt64("dropout_offset", _values.dropoutOffset));
            }
            else if(rng == "not_supplied")
            {
                attrs.set_dropout_probability(probability);
            }
            else
            {
                unknownValue("dropout_rng_state", rng);
            }
        }
        else if(style == "caller_mask")
        {
            attrs.set_dropout(packedTensor("dropout_mask", dropoutMaskDims(), DataType::FLOAT),
                              byValueFloat("dropout_scale", dropoutMaskScale()));
            const auto& cycle = classValue(_values.dropoutMaskCycle,
                                           "dropout_mask_pattern",
                                           valueOf(_row, "dropout_mask_pattern"));
            _contents["dropout_mask"] = realContents(cycle);
        }
        else
        {
            unknownValue("dropout", style);
        }

        const auto& patternOutput = valueOf(_row, "dropout_pattern_output");
        if(patternOutput == "requested")
        {
            auto rngDump = scoreShaped("rng_dump", DataType::FLOAT);
            rngDump->set_output(true);
            attrs.set_rng_dump(rngDump);
        }
        else if(patternOutput != "not_requested")
        {
            unknownValue("dropout_pattern_output", patternOutput);
        }
    }

    bool requested(const std::string& axis) const
    {
        const auto& id = valueOf(_row, axis);
        if(id == "requested")
        {
            return true;
        }
        if(id == "not_requested")
        {
            return false;
        }
        unknownValue(axis, id);
    }

    bool presentFlag(const std::string& axis) const
    {
        const auto& id = valueOf(_row, axis);
        if(id == "present")
        {
            return true;
        }
        if(id == "absent")
        {
            return false;
        }
        unknownValue(axis, id);
    }

    void applyOptionalOutputs(SdpaAttributes& attrs)
    {
        if(requested("stats_output"))
        {
            attrs.set_generate_stats(true);
        }
        const auto& rowStats = valueOf(_row, "max_sumexp_output");
        if(rowStats != "not_requested" && rowStats != "requested" && rowStats != "max_only"
           && rowStats != "sum_exp_only")
        {
            unknownValue("max_sumexp_output", rowStats);
        }
        if(rowStats == "requested" || rowStats == "max_only")
        {
            attrs.set_logit_max(outputTensor("logit_max", {_b, _hq, _sq, 1}, DataType::FLOAT));
        }
        if(rowStats == "requested" || rowStats == "sum_exp_only")
        {
            attrs.set_score_sum_exp(
                outputTensor("score_sum_exp", {_b, _hq, _sq, 1}, DataType::FLOAT));
        }
        if(requested("amax_s"))
        {
            attrs.set_amax_s(outputTensor("amax_s", {1, 1, 1, 1}, DataType::FLOAT));
        }
        if(requested("amax_o"))
        {
            attrs.set_amax_o(outputTensor("amax_o", {1, 1, 1, 1}, DataType::FLOAT));
        }
    }

    // A Q, K or V descale: one factor, one per (batch, K/V head), or one per token and head.
    TensorPtr descaleTensor(const std::string& name)
    {
        const auto& id = valueOf(_row, name);
        if(id == "present")
        {
            return scalarFactor(name);
        }
        Dims dims;
        if(id == "per_head")
        {
            dims = {_b, name == "descale_v" ? _hv : _hk, 1, 1};
        }
        else if(id == "finer")
        {
            if(name == "descale_q")
            {
                dims = {_b, _hq, _sq, 1};
            }
            else
            {
                const bool keySide = name == "descale_k";
                const int64_t keys = pagedSide(keySide ? "K" : "V") ? keyExtent() : _skv;
                dims = {_b, keySide ? _hk : _hv, keys, 1};
            }
        }
        else
        {
            unknownValue(name, id);
        }
        const auto listIt = _values.descaleLists.find(id);
        if(listIt == _values.descaleLists.end() || listIt->second.count(name) == 0)
        {
            unknownValue(name, id);
        }
        const auto& factors = listIt->second.at(name);
        _contents[name] = realContents(factors);
        _notes.push_back(name + "=" + id + ":" + dimsText(dims));
        return packedTensor(name, dims, DataType::FLOAT);
    }

    void applyQuantisationTensors(SdpaAttributes& attrs)
    {
        for(const char* name : {"descale_q", "descale_k", "descale_v"})
        {
            const auto& id = valueOf(_row, name);
            if(id == "absent")
            {
                continue;
            }
            const auto tensor = descaleTensor(name);
            if(std::string(name) == "descale_q")
            {
                attrs.set_descale_q(tensor);
            }
            else if(std::string(name) == "descale_k")
            {
                attrs.set_descale_k(tensor);
            }
            else
            {
                attrs.set_descale_v(tensor);
            }
        }
        if(presentFlag("s_quantization"))
        {
            attrs.set_descale_s(scalarFactor("descale_s"));
            attrs.set_scale_s(scalarFactor("scale_s"));
        }
        if(presentFlag("scale_o"))
        {
            attrs.set_scale_o(scalarFactor("scale_o"));
        }
    }

    struct PagedTable
    {
        int64_t pagesPerSequence = 0;
        int64_t declaredMaxSeqLenKv = 0;
        int64_t pageSizeK = 0;
        int64_t pageSizeV = 0;
        int64_t widthK = 0;
        int64_t widthV = 0;
    };

    // A declared maximum fixes the page-table width to ceil(max / page size), so a "larger"
    // maximum fills the surplus pages when there are any.
    PagedTable pagedTable() const
    {
        const auto [pageSize, pagesNeeded] = pageGeometry();

        const auto& countId = valueOf(_row, "page_count");
        PagedTable table;
        if(countId == "exact")
        {
            table.pagesPerSequence = pagesNeeded;
        }
        else if(countId == "surplus")
        {
            table.pagesPerSequence = pagesNeeded + 1;
        }
        else if(countId == "insufficient")
        {
            if(pagesNeeded < 2)
            {
                fail("page_count=insufficient needs at least 2 pages per sequence");
            }
            table.pagesPerSequence = pagesNeeded - 1;
        }
        else
        {
            unknownValue("page_count", countId);
        }

        const auto& maxId = valueOf(_row, "max_seq_len_kv");
        if(maxId == "equal")
        {
            table.declaredMaxSeqLenKv = _skv;
        }
        else if(maxId == "larger")
        {
            const int64_t tableKeys = table.pagesPerSequence * pageSize;
            table.declaredMaxSeqLenKv = tableKeys > _skv ? tableKeys : 2 * _skv;
        }
        else if(maxId == "smaller")
        {
            table.declaredMaxSeqLenKv = _skv - 1;
        }
        else
        {
            unknownValue("max_seq_len_kv", maxId);
        }

        const bool pagedK = pagedSide("K");
        const bool pagedV = pagedSide("V");
        int64_t multiplier = 1;
        if(pagedK && pagedV)
        {
            multiplier = classValue(
                _values.vPageMultiplier, "page_size_k_vs_v", valueOf(_row, "page_size_k_vs_v"));
        }
        if(pagedK)
        {
            table.pageSizeK = pageSize;
            table.widthK = table.pagesPerSequence;
        }
        if(pagedV)
        {
            table.pageSizeV = pageSize * multiplier;
            table.widthV = multiplier == 1 ? table.pagesPerSequence
                                           : ceilDiv(table.declaredMaxSeqLenKv, table.pageSizeV);
        }
        return table;
    }

    TensorPtr pageTable(const std::string& name, int64_t width, bool packed)
    {
        auto table = packedTensor(name, {_b, 1, width, 1}, DataType::INT32);
        if(packed)
        {
            table->set_ragged_offset(
                packedTensor(name + "_offset", {_b + 1, 1, 1, 1}, DataType::INT32));
        }
        return table;
    }

    void applyPaging(SdpaAttributes& attrs)
    {
        if(!pagedAny())
        {
            return;
        }
        const auto table = pagedTable();
        const bool packed = classValue(
            _values.packedPageTable, "page_table_storage", valueOf(_row, "page_table_storage"));
        const bool shared = isActive(_row, "page_table_sharing")
                            && classValue(_values.pageTables,
                                          "page_table_sharing",
                                          valueOf(_row, "page_table_sharing"))
                                   == 1;

        if(shared)
        {
            if(table.widthK != table.widthV)
            {
                fail("page_table_sharing=shared needs one table width; K needs "
                     + std::to_string(table.widthK) + ", V " + std::to_string(table.widthV));
            }
            const auto kv = pageTable("page_table_kv", table.widthK, packed);
            attrs.set_paged_attention_k_table(kv);
            attrs.set_paged_attention_v_table(kv);
        }
        else
        {
            if(pagedSide("K"))
            {
                attrs.set_paged_attention_k_table(pageTable("page_table_k", table.widthK, packed));
            }
            if(pagedSide("V"))
            {
                attrs.set_paged_attention_v_table(pageTable("page_table_v", table.widthV, packed));
            }
        }
        attrs.set_paged_attention_max_seq_len_kv(static_cast<int32_t>(table.declaredMaxSeqLenKv));
        _notes.push_back("pages_per_seq=" + std::to_string(table.pagesPerSequence)
                         + " max_seq_len_kv=" + std::to_string(table.declaredMaxSeqLenKv)
                         + (shared ? " shared_table" : "") + (packed ? " packed_table" : ""));

        // Paged K/V reads per-sequence lengths, so both length tensors are always present.
        if(!attrs.get_seq_len_q() || !attrs.get_seq_len_kv())
        {
            attrs.set_padding_mask(true);
            _notes.push_back("seq_len_tensors_for_paging");
        }
        if(!attrs.get_seq_len_q())
        {
            attrs.set_seq_len_q(packedTensor("seq_len_q", {_b, 1, 1, 1}, DataType::INT32));
        }
        if(!attrs.get_seq_len_kv())
        {
            attrs.set_seq_len_kv(packedTensor("seq_len_kv", {_b, 1, 1, 1}, DataType::INT32));
        }
    }

    void applyImplementation(SdpaAttributes& attrs)
    {
        const auto& id = valueOf(_row, "implementation");
        if(id == "auto")
        {
            attrs.set_implementation(AttentionImplementation::AUTO);
        }
        else if(id == "composite")
        {
            attrs.set_implementation(AttentionImplementation::COMPOSITE);
        }
        else if(id == "unified")
        {
            attrs.set_implementation(AttentionImplementation::UNIFIED);
        }
        else
        {
            unknownValue("implementation", id);
        }
    }

    static void setDefaultsExplicitly(SdpaAttributes& attrs)
    {
        attrs.set_alibi_mask(false);
        attrs.set_padding_mask(false);
        attrs.causal_mask = false;
        attrs.causal_mask_bottom_right = false;
        attrs.set_diagonal_alignment(DiagonalAlignment::TOP_LEFT);
        attrs.set_implementation(AttentionImplementation::AUTO);
    }

    void applyAttributeOverrides(SdpaAttributes& attrs)
    {
        if(_ov.leftBound.has_value())
        {
            attrs.set_diagonal_band_left_bound(*_ov.leftBound);
        }
        if(_ov.rightBound.has_value())
        {
            attrs.set_diagonal_band_right_bound(*_ov.rightBound);
        }
        if(_ov.deprecatedCausalMask.has_value())
        {
            attrs.causal_mask = *_ov.deprecatedCausalMask;
        }
        if(_ov.deprecatedCausalMaskBottomRight.has_value())
        {
            attrs.causal_mask_bottom_right = *_ov.deprecatedCausalMaskBottomRight;
        }
        if(_ov.maxSeqLenKv.has_value())
        {
            attrs.set_paged_attention_max_seq_len_kv(static_cast<int32_t>(*_ov.maxSeqLenKv));
        }
        if(_ov.paddingMask.has_value())
        {
            attrs.set_padding_mask(*_ov.paddingMask);
        }
        if(_ov.generateStats.has_value())
        {
            attrs.set_generate_stats(*_ov.generateStats);
        }
        if(_ov.dropoutProbability.has_value())
        {
            attrs.set_dropout_probability(*_ov.dropoutProbability);
        }
        if(_ov.addSeqLenQ)
        {
            attrs.set_seq_len_q(packedTensor("seq_len_q", {_b, 1, 1, 1}, DataType::INT32));
        }
        if(_ov.addSeqLenKv)
        {
            attrs.set_seq_len_kv(packedTensor("seq_len_kv", {_b, 1, 1, 1}, DataType::INT32));
        }
        if(_ov.addDropoutMask)
        {
            attrs.set_dropout_mask(scoreShaped("dropout_mask", DataType::FLOAT));
        }
        if(_ov.addDropoutScale)
        {
            attrs.set_dropout_scale(byValueFloat("dropout_scale", 1.0f));
        }
        if(_ov.addSeedOffset)
        {
            attrs.set_seed(byValueInt64("dropout_seed", _values.dropoutSeed));
            attrs.set_offset(byValueInt64("dropout_offset", _values.dropoutOffset));
        }
        if(_ov.addRngDump)
        {
            auto rngDump = scoreShaped("rng_dump", DataType::FLOAT);
            rngDump->set_output(true);
            attrs.set_rng_dump(rngDump);
        }
        if(_ov.statsWithoutGenerate)
        {
            attrs.set_stats(outputTensor("stats", {_b, _hq, _sq, 1}, DataType::FLOAT));
            attrs.set_generate_stats(false);
        }
    }

    void configureOutput(const TensorPtr& o)
    {
        const Dims oDims = {_b, _hq, _sq, _dv};
        o->set_output(true).set_data_type(_fo).set_dim(oDims).set_stride(layoutStrides(oDims, "o"));
        applyAlignment(*o, "alignment_o", valueOf(_row, "alignment_o"), _fo);
        if(raggedTensor("O"))
        {
            o->set_ragged_offset(raggedOffsetFor("O"));
        }
        _named["o"] = o;
    }

    void configureStats(const TensorPtr& stats)
    {
        if(!stats)
        {
            return;
        }
        if(!isActive(_row, "dtype_stats") || !isActive(_row, "dim_order_stats")
           || !isActive(_row, "strides_stats") || !isActive(_row, "alignment_stats")
           || !isActive(_row, "stats_packing"))
        {
            fail("stats_output=requested but a stats child axis is '-'");
        }
        const DataType type = formatOf("dtype_stats", valueOf(_row, "dtype_stats"));
        const Dims dims = {_b, _hq, _sq, 1};
        stats->set_output(true).set_data_type(type).set_dim(dims).set_stride(
            layoutStrides(dims, "stats"));
        applyAlignment(*stats, "alignment_stats", valueOf(_row, "alignment_stats"), type);
        if(classValue(_values.raggedStats, "stats_packing", valueOf(_row, "stats_packing")))
        {
            if(!raggedTensor("Q"))
            {
                fail("stats_packing=" + valueOf(_row, "stats_packing")
                     + " needs a ragged Q; batch_packing is " + valueOf(_row, "batch_packing"));
            }
            stats->set_ragged_offset(raggedOffsetFor("Stats"));
        }
        _named["stats"] = stats;
    }

    void applyDimOverrides()
    {
        for(const auto& [name, dims] : _ov.tensorDims)
        {
            const auto it = _named.find(name);
            if(it == _named.end())
            {
                fail("override '" + name + "_dims' names a tensor this case does not have");
            }
            it->second->set_dim(dims).set_stride(packedStrides(dims));
            _notes.push_back(name + "_dims=" + dimsText(dims));
        }
    }

    // Lengths of sequences 0 .. count-1 of one side, from its per_seq_len class at stored length S.
    std::vector<int64_t>
        sideLengths(const std::string& lengthAxis, int64_t stored, int64_t count) const
    {
        const std::string id = isActive(_row, lengthAxis) ? valueOf(_row, lengthAxis) : "all_full";
        const auto& cycle = classValue(_values.perSeqLen, lengthAxis, id);
        std::vector<int64_t> lengths;
        for(int64_t b = 0; b < count; ++b)
        {
            const auto& term = cycle[static_cast<size_t>(b) % cycle.size()];
            int64_t length = 0;
            switch(term.op)
            {
            case LengthTerm::Op::CONSTANT:
                length = term.operand;
                break;
            case LengthTerm::Op::STORED:
                length = stored;
                break;
            case LengthTerm::Op::DIVIDE:
                length = stored / term.operand;
                break;
            case LengthTerm::Op::ADD:
                length = stored + term.operand;
                break;
            case LengthTerm::Op::MULTIPLY:
                length = checkedProduct(stored, term.operand, lengthAxis + "=" + id);
                break;
            default:
                fail(lengthAxis + "=" + id + " has an unknown length operation");
            }
            if(length < 0)
            {
                fail(lengthAxis + "=" + id
                     + " gives a negative length at S=" + std::to_string(stored));
            }
            lengths.push_back(length);
        }
        return lengths;
    }

    // The per_seq_len axis of the side a ragged tensor belongs to.
    static std::string lengthAxisOf(const std::string& tensorName)
    {
        if(tensorName == "q" || tensorName == "o" || tensorName == "stats")
        {
            return "per_seq_len_q";
        }
        if(tensorName == "k" || tensorName == "v")
        {
            return "per_seq_len_kv";
        }
        fail("tensor '" + tensorName + "' has a ragged offset but belongs to no sequence side");
    }

    // B + 1 offsets in the tensor's own elements. The stored length is the tensor's sequence
    // extent and the sequence stride is its stride over that extent.
    std::vector<int64_t> raggedOffsets(const std::string& tensorName,
                                       const TensorAttributes& tensor) const
    {
        const auto& dims = tensor.get_dim();
        const auto& strides = tensor.get_stride();
        if(dims.size() != 4 || strides.size() != 4)
        {
            fail("ragged tensor '" + tensorName + "' is not 4-D");
        }
        const int64_t stored = dims[2];
        const int64_t stride = strides[2];
        const std::string what = "ragged offset of '" + tensorName + "'";

        std::vector<int64_t> tokens;
        if(tensorName.rfind("page_table_", 0) == 0)
        {
            for(int64_t b = 0; b <= _b; ++b)
            {
                tokens.push_back(b * stored);
            }
        }
        else
        {
            if(!isActive(_row, "ragged_offsets"))
            {
                fail("tensor '" + tensorName + "' is ragged but ragged_offsets is '-'");
            }
            const auto& offsetsId = valueOf(_row, "ragged_offsets");
            const auto layout = classValue(_values.raggedOffsets, "ragged_offsets", offsetsId);
            const auto lengths = sideLengths(lengthAxisOf(tensorName), stored, _b);
            int64_t total = 0;
            for(const auto length : lengths)
            {
                if(length > stored)
                {
                    fail(what + ": a sequence length " + std::to_string(length)
                         + " is longer than the stored length " + std::to_string(stored));
                }
                total += length;
            }
            int64_t running = 0;
            if(layout == OffsetLayout::PREFIX_SUM_AFTER_GAP)
            {
                running = checkedProduct(_b, stored, what) - total;
            }
            for(int64_t b = 0; b <= _b; ++b)
            {
                tokens.push_back(layout == OffsetLayout::FULL_SLOTS ? b * stored : running);
                if(b < _b)
                {
                    running += lengths[static_cast<size_t>(b)];
                }
            }
        }

        std::vector<int64_t> offsets;
        for(const auto token : tokens)
        {
            const int64_t offset = checkedProduct(token, stride, what);
            if(offset > std::numeric_limits<int32_t>::max()
               || offset < std::numeric_limits<int32_t>::min())
            {
                fail(what + " is " + std::to_string(offset) + ", outside INT32");
            }
            offsets.push_back(offset);
        }
        return offsets;
    }

    // In-order page tables: entry i holds page i modulo the smallest pool the table addresses.
    std::vector<int64_t> pageTableContents(const std::string& tableName,
                                           const TensorAttributes& table) const
    {
        std::vector<std::string> pools;
        if(tableName == "page_table_k")
        {
            pools = {"k"};
        }
        else if(tableName == "page_table_v")
        {
            pools = {"v"};
        }
        else
        {
            pools = {"k", "v"};
        }
        int64_t pages = std::numeric_limits<int64_t>::max();
        for(const auto& pool : pools)
        {
            pages = std::min(pages, _named.at(pool)->get_dim().at(0));
        }
        pages = std::max<int64_t>(1, pages);
        std::vector<int64_t> values(static_cast<size_t>(elementCount(table.get_dim())));
        for(size_t i = 0; i < values.size(); ++i)
        {
            values[i] = static_cast<int64_t>(i) % pages;
        }
        return values;
    }

    // Contents decided by the final graph: sequence lengths, ragged offsets, page tables, and a
    // block mask whose element count a dims override changed.
    void addStructuralContents()
    {
        for(const auto& [name, axis, stored] :
            std::vector<std::tuple<std::string, std::string, int64_t>>{
                {"seq_len_q", "per_seq_len_q", _sq}, {"seq_len_kv", "per_seq_len_kv", _skv}})
        {
            if(_named.count(name) == 0)
            {
                continue;
            }
            const std::string id = isActive(_row, axis) ? valueOf(_row, axis) : "all_full";
            const auto cycleLength
                = std::min(static_cast<int64_t>(classValue(_values.perSeqLen, axis, id).size()),
                           std::max<int64_t>(1, elementCount(_named.at(name)->get_dim())));
            _contents[name] = integerContents(sideLengths(axis, stored, cycleLength), true);
        }

        for(const auto& [name, tensor] : _named)
        {
            const auto offset = tensor->get_ragged_offset();
            if(!offset)
            {
                continue;
            }
            _contents[offset->get_name()] = integerContents(raggedOffsets(name, *tensor), false);
        }

        for(const char* table : {"page_table_k", "page_table_v", "page_table_kv"})
        {
            const auto it = _named.find(table);
            if(it != _named.end())
            {
                _contents[table] = integerContents(pageTableContents(table, *it->second), false);
            }
        }

        const auto mask = _contents.find("block_mask");
        if(mask != _contents.end())
        {
            auto& values = std::get<std::vector<int64_t>>(mask->second.values);
            values.resize(static_cast<size_t>(elementCount(_named.at("block_mask")->get_dim())), 1);
        }
    }

    std::string realisedText() const
    {
        std::string text = "B=" + std::to_string(_b) + " Hq=" + std::to_string(_hq)
                           + " Hk=" + std::to_string(_hk) + " Hv=" + std::to_string(_hv)
                           + " Sq=" + std::to_string(_sq) + " Skv=" + std::to_string(_skv)
                           + " Dqk=" + std::to_string(_dqk) + " Dv=" + std::to_string(_dv)
                           + " q=" + dataTypeName(_fq) + " k=" + dataTypeName(_fk)
                           + " v=" + dataTypeName(_fv) + " o=" + dataTypeName(_fo)
                           + " compute=" + dataTypeName(_compute);
        for(const auto& note : _notes)
        {
            text += " " + note;
        }
        return text;
    }
};

using Json = nlohmann::json;

[[noreturn]] void badFile(const std::string& message)
{
    throw ClassValuesError("class-to-value file: " + message);
}

const Json& member(const Json& object, const std::string& key, const std::string& where)
{
    if(!object.is_object())
    {
        badFile(where + " is not an object");
    }
    const auto it = object.find(key);
    if(it == object.end())
    {
        badFile(where + " has no '" + key + "'");
    }
    return *it;
}

int64_t integerAt(const Json& value, const std::string& where)
{
    if(!value.is_number_integer())
    {
        badFile(where + " is not an integer");
    }
    return value.get<int64_t>();
}

double numberAt(const Json& value, const std::string& where)
{
    if(!value.is_number())
    {
        badFile(where + " is not a number");
    }
    return value.get<double>();
}

std::vector<double> numbersAt(const Json& value, const std::string& where)
{
    if(!value.is_array() || value.empty())
    {
        badFile(where + " is not a non-empty array");
    }
    std::vector<double> numbers;
    for(const auto& item : value)
    {
        numbers.push_back(numberAt(item, where));
    }
    return numbers;
}

std::string stringAt(const Json& value, const std::string& where)
{
    if(!value.is_string())
    {
        badFile(where + " is not a string");
    }
    return value.get<std::string>();
}

template <typename Convert>
auto tableAt(const Json& root, const std::string& key, Convert convert)
{
    const auto& table = member(root, key, "the file");
    if(!table.is_object() || table.empty())
    {
        badFile("'" + key + "' is not a non-empty object");
    }
    std::map<std::string, decltype(convert(table.begin().value(), key))> out;
    for(auto it = table.begin(); it != table.end(); ++it)
    {
        out.emplace(it.key(), convert(it.value(), key + "." + it.key()));
    }
    return out;
}

std::vector<size_t> dimOrderAt(const Json& value, const std::string& where)
{
    static const std::map<std::string, size_t> LOGICAL = {{"B", 0}, {"H", 1}, {"S", 2}, {"D", 3}};
    if(!value.is_array() || value.size() != LOGICAL.size())
    {
        badFile(where + " is not an array of four dimension letters");
    }
    std::vector<size_t> order;
    std::set<size_t> seen;
    for(const auto& letter : value)
    {
        const auto it = LOGICAL.find(stringAt(letter, where));
        if(it == LOGICAL.end() || !seen.insert(it->second).second)
        {
            badFile(where + " is not an order of B, H, S and D");
        }
        order.push_back(it->second);
    }
    return order;
}

std::vector<ScaleNumber> scaleRepresentativesAt(const Json& value, const std::string& where)
{
    if(!value.is_array() || value.empty())
    {
        badFile(where + " is not a non-empty array");
    }
    std::vector<ScaleNumber> representatives;
    for(const auto& item : value)
    {
        ScaleNumber number;
        if(item.is_number())
        {
            number.value = item.get<double>();
        }
        else
        {
            const auto text = stringAt(item, where);
            if(text == "1/sqrt(head_dim_qk)")
            {
                number.inverseSqrtHeadDim = true;
            }
            else if(text == "nan")
            {
                number.value = std::numeric_limits<double>::quiet_NaN();
            }
            else if(text == "+inf")
            {
                number.value = std::numeric_limits<double>::infinity();
            }
            else if(text == "-inf")
            {
                number.value = -std::numeric_limits<double>::infinity();
            }
            else
            {
                badFile(where + " has an unknown representative '" + text + "'");
            }
        }
        representatives.push_back(number);
    }
    return representatives;
}

std::optional<int64_t> pageSizeAt(const Json& value, const std::string& where)
{
    if(value.is_string() && value.get<std::string>() == "S_kv")
    {
        return std::nullopt;
    }
    return integerAt(value, where);
}

LengthTerm lengthTermAt(const Json& value, const std::string& where)
{
    LengthTerm term;
    if(value.is_string() && value.get<std::string>() == "S")
    {
        term.op = LengthTerm::Op::STORED;
        return term;
    }
    if(value.is_number_integer())
    {
        term.operand = value.get<int64_t>();
        if(term.operand < 0)
        {
            badFile(where + " is a negative length");
        }
        return term;
    }
    static const std::map<std::string, LengthTerm::Op> OPS
        = {{"divide", LengthTerm::Op::DIVIDE},
           {"add", LengthTerm::Op::ADD},
           {"multiply", LengthTerm::Op::MULTIPLY}};
    if(!value.is_object() || value.size() != 2
       || stringAt(member(value, "of", where), where) != "S")
    {
        badFile(where + " is not S, an integer, or {of: S, <divide|add|multiply>: n}");
    }
    for(const auto& [key, op] : OPS)
    {
        if(value.contains(key))
        {
            term.op = op;
            term.operand = integerAt(value.at(key), where + "." + key);
            if(op == LengthTerm::Op::DIVIDE && term.operand <= 0)
            {
                badFile(where + ".divide is not positive");
            }
            return term;
        }
    }
    badFile(where + " is not S, an integer, or {of: S, <divide|add|multiply>: n}");
}

std::vector<LengthTerm> lengthCycleAt(const Json& value, const std::string& where)
{
    if(!value.is_array() || value.empty())
    {
        badFile(where + " is not a non-empty array");
    }
    std::vector<LengthTerm> cycle;
    for(size_t i = 0; i < value.size(); ++i)
    {
        cycle.push_back(lengthTermAt(value[i], where + "[" + std::to_string(i) + "]"));
    }
    return cycle;
}

OffsetLayout offsetLayoutAt(const Json& value, const std::string& where)
{
    static const std::map<std::string, OffsetLayout> LAYOUTS
        = {{"prefix_sum_of_lengths", OffsetLayout::PREFIX_SUM},
           {"full_slots", OffsetLayout::FULL_SLOTS},
           {"prefix_sum_after_gap", OffsetLayout::PREFIX_SUM_AFTER_GAP}};
    const auto name = stringAt(member(value, "offsets", where), where + ".offsets");
    const auto it = LAYOUTS.find(name);
    if(it == LAYOUTS.end())
    {
        badFile(where + ".offsets has an unknown layout '" + name + "'");
    }
    return it->second;
}

bool raggedStatsAt(const Json& value, const std::string& where)
{
    if(!value.is_array() || value.size() > 1
       || (value.size() == 1 && stringAt(value[0], where) != "Stats"))
    {
        badFile(where + " is not [] or [\"Stats\"]");
    }
    return value.size() == 1;
}

ClassValues parseClassValuesJson(const Json& root)
{
    constexpr int64_t SCHEMA_VERSION = 2;
    if(integerAt(member(root, "schema_version", "the file"), "schema_version") != SCHEMA_VERSION)
    {
        badFile("schema_version is not " + std::to_string(SCHEMA_VERSION));
    }

    ClassValues values;
    values.batchSize = tableAt(root, "batch_size", integerAt);
    values.qHeads = tableAt(root, "q_heads", integerAt);
    values.headDim = tableAt(root, "head_dim", integerAt);
    values.dimOrder = tableAt(root, "dim_order", dimOrderAt);
    values.scaleValue = tableAt(root, "scale_value", scaleRepresentativesAt);
    values.dropoutProbability = tableAt(root, "dropout_probability", numberAt);
    values.dropoutMaskScale = tableAt(root, "dropout_mask_scale", numberAt);
    values.dropoutMaskCycle = tableAt(root, "dropout_mask_cycle", numbersAt);
    values.pageSize = tableAt(root, "page_size", pageSizeAt);
    values.quantisationFactors = tableAt(root, "quantisation_factors", numberAt);
    for(const char* key : {"descale_q", "descale_k", "descale_v", "other"})
    {
        if(values.quantisationFactors.count(key) == 0)
        {
            badFile("quantisation_factors has no '" + std::string(key) + "'");
        }
    }

    const auto& lengths = member(root, "seq_lengths", "the file");
    if(!lengths.is_array() || lengths.empty())
    {
        badFile("seq_lengths is not a non-empty array");
    }
    for(const auto& entry : lengths)
    {
        const std::string where = "seq_lengths entry";
        values.sequenceLengths.push_back(
            {stringAt(member(entry, "seq_q", where), where + ".seq_q"),
             stringAt(member(entry, "seq_kv", where), where + ".seq_kv"),
             stringAt(member(entry, "seq_q_vs_seq_kv", where), where + ".seq_q_vs_seq_kv"),
             integerAt(member(entry, "S_q", where), where + ".S_q"),
             integerAt(member(entry, "S_kv", where), where + ".S_kv")});
    }

    const auto& differs = member(root, "dtype_differs", "the file");
    values.differsFormat = tableAt(differs, "by_dtype_q", stringAt);
    const auto& byArch = member(differs, "by_arch", "dtype_differs");
    if(!byArch.is_object())
    {
        badFile("dtype_differs.by_arch is not an object");
    }
    for(auto it = byArch.begin(); it != byArch.end(); ++it)
    {
        auto& substitutions = values.archFormat[it.key()];
        if(!it.value().is_object())
        {
            badFile("dtype_differs.by_arch." + it.key() + " is not an object");
        }
        for(auto sub = it.value().begin(); sub != it.value().end(); ++sub)
        {
            substitutions[sub.key()]
                = stringAt(sub.value(), "dtype_differs.by_arch." + it.key() + "." + sub.key());
        }
    }

    for(const char* granularity : {"per_head", "finer"})
    {
        const auto& lists
            = member(member(root, "descale_lists", "the file"), granularity, "descale_lists");
        for(const char* name : {"descale_q", "descale_k", "descale_v"})
        {
            values.descaleLists[granularity][name]
                = numbersAt(member(lists, name, std::string("descale_lists.") + granularity),
                            std::string("descale_lists.") + granularity + "." + name);
        }
    }
    const auto letterSet = [](const Json& value, const std::string& where) {
        if(!value.is_array())
        {
            badFile(where + " is not an array of tensor letters");
        }
        std::set<std::string> letters;
        for(const auto& item : value)
        {
            letters.insert(stringAt(item, where));
        }
        return letters;
    };
    values.raggedTensors = tableAt(root, "ragged_tensors", letterSet);
    values.raggedStats = tableAt(root, "ragged_stats", raggedStatsAt);
    values.perSeqLen = tableAt(root, "per_seq_len", lengthCycleAt);
    values.raggedOffsets = tableAt(root, "ragged_offsets", offsetLayoutAt);
    values.pagedSides = tableAt(root, "paged_sides", letterSet);
    values.vPageMultiplier = tableAt(root, "v_page_multiplier", integerAt);
    values.pageTables = tableAt(root, "page_tables", integerAt);
    values.packedPageTable
        = tableAt(root, "packed_page_table", [](const Json& value, const std::string& where) {
              if(!value.is_boolean())
              {
                  badFile(where + " is not a boolean");
              }
              return value.get<bool>();
          });

    const auto& alibi = member(root, "alibi_head_count", "the file");
    const auto& replaced = member(alibi, "H_q", "alibi_head_count");
    const auto& heads = member(alibi, "heads", "alibi_head_count");
    if(!replaced.is_object())
    {
        badFile("alibi_head_count.H_q is not an object");
    }
    for(auto it = replaced.begin(); it != replaced.end(); ++it)
    {
        const std::string where = "alibi_head_count.H_q." + it.key();
        AlibiHeads entry;
        entry.hq = integerAt(it.value(), where);
        const auto& counts = member(heads, std::to_string(entry.hq), "alibi_head_count.heads");
        entry.mha = integerAt(member(counts, "mha", where), where + ".mha");
        entry.gqa = integerAt(member(counts, "gqa", where), where + ".gqa");
        entry.mqa = integerAt(member(counts, "mqa", where), where + ".mqa");
        const auto& differ = member(counts, "k_and_v_differ", where);
        entry.differK = integerAt(member(differ, "H_k", where), where + ".H_k");
        entry.differV = integerAt(member(differ, "H_v", where), where + ".H_v");
        values.alibiHeads[std::stoll(it.key())] = entry;
    }

    const auto& mma = member(root, "mma_fp8", "the file");
    values.mmaFp8Otherwise = stringAt(member(mma, "otherwise", "mma_fp8"), "mma_fp8.otherwise");
    const auto& mmaArch = member(mma, "by_arch", "mma_fp8");
    if(!mmaArch.is_object())
    {
        badFile("mma_fp8.by_arch is not an object");
    }
    for(auto it = mmaArch.begin(); it != mmaArch.end(); ++it)
    {
        for(auto sub = it.value().begin(); sub != it.value().end(); ++sub)
        {
            values.mmaFp8ByArch[it.key()][sub.key()]
                = stringAt(sub.value(), "mma_fp8.by_arch." + it.key() + "." + sub.key());
        }
    }

    const auto& seedOffset = member(root, "dropout_seed_offset", "the file");
    values.dropoutSeed
        = integerAt(member(seedOffset, "seed", "dropout_seed_offset"), "dropout_seed_offset.seed");
    values.dropoutOffset = integerAt(member(seedOffset, "offset", "dropout_seed_offset"),
                                     "dropout_seed_offset.offset");
    return values;
}

} // namespace

ClassValues parseClassValues(const std::string& jsonText)
{
    try
    {
        return parseClassValuesJson(Json::parse(jsonText));
    }
    catch(const Json::exception& e)
    {
        badFile(e.what());
    }
}

const std::vector<std::string>& modelAxisIds()
{
    static const std::vector<std::string> IDS = {
        "batch_packing",
        "ragged_offsets",
        "sequence_lengths",
        "per_seq_len_q",
        "per_seq_len_kv",
        "dtype_q",
        "dtype_k_vs_q",
        "dtype_v_vs_q",
        "dtype_o",
        "compute_precision",
        "matmul_input_precision",
        "batch_size",
        "seq_q",
        "seq_kv",
        "seq_q_vs_seq_kv",
        "q_heads",
        "k_head_sharing",
        "v_head_sharing",
        "k_heads_vs_v_heads",
        "head_dim_qk",
        "head_dim_v",
        "dim_order_q",
        "dim_order_k",
        "dim_order_v",
        "dim_order_o",
        "dim_order_stats",
        "strides_q",
        "strides_k",
        "strides_v",
        "strides_o",
        "strides_stats",
        "alignment_q",
        "alignment_k",
        "alignment_v",
        "alignment_o",
        "alignment_stats",
        "tensor_overlap",
        "causal_mask",
        "causal_alignment",
        "sliding_window",
        "window_alignment",
        "window_left",
        "window_right",
        "block_mask",
        "block_mask_pattern",
        "bias",
        "bias_broadcast",
        "bias_dtype",
        "alibi",
        "alibi_head_count",
        "attention_sinks",
        "scale_source",
        "scale_value",
        "dropout",
        "dropout_probability",
        "dropout_rng_state",
        "dropout_mask_pattern",
        "dropout_mask_broadcast",
        "dropout_pattern_output",
        "stats_output",
        "dtype_stats",
        "stats_packing",
        "max_sumexp_output",
        "amax_s",
        "amax_o",
        "descale_q",
        "descale_k",
        "descale_v",
        "s_quantization",
        "scale_o",
        "paged_kv",
        "page_table_sharing",
        "page_size_k_vs_v",
        "page_table_storage",
        "page_size",
        "page_table_order",
        "page_count",
        "max_seq_len_kv",
        "implementation",
        "runtime_shape_override",
    };
    return IDS;
}

bool isExecuteLevelAxis(const std::string& axis)
{
    static const std::set<std::string> EXECUTE_LEVEL = {
        "ragged_offsets",
        "per_seq_len_q",
        "per_seq_len_kv",
        "tensor_overlap",
        "block_mask_pattern",
        "dropout_mask_pattern",
        "page_table_order",
    };
    return EXECUTE_LEVEL.count(axis) != 0;
}

bool isRealisedContentAxis(const std::string& axis)
{
    static const std::set<std::string> REALISED = {
        "per_seq_len_q",
        "per_seq_len_kv",
        "ragged_offsets",
        "stats_packing",
        "block_mask_pattern",
        "dropout_mask_pattern",
    };
    return REALISED.count(axis) != 0;
}

std::string sharedBaseValue(const std::string& axis)
{
    static const std::map<std::string, std::string> BASE = {
        {"batch_packing", "padded"},
        {"sequence_lengths", "uniform"},
        {"dtype_q", "bf16"},
        {"dtype_k_vs_q", "same"},
        {"dtype_v_vs_q", "same"},
        {"dtype_o", "bf16"},
        {"compute_precision", "fp32"},
        {"matmul_input_precision", "native"},
        {"batch_size", "many"},
        {"seq_q", "many"},
        {"seq_kv", "many"},
        {"seq_q_vs_seq_kv", "equal"},
        {"q_heads", "many"},
        {"k_head_sharing", "mha"},
        {"v_head_sharing", "mha"},
        {"k_heads_vs_v_heads", "equal"},
        {"head_dim_qk", "d128"},
        {"head_dim_v", "d128"},
        {"dim_order_q", "bhsd"},
        {"dim_order_k", "bhsd"},
        {"dim_order_v", "bhsd"},
        {"dim_order_o", "bhsd"},
        {"strides_q", "packed"},
        {"strides_k", "packed"},
        {"strides_v", "packed"},
        {"strides_o", "packed"},
        {"alignment_q", "wide_aligned"},
        {"alignment_k", "wide_aligned"},
        {"alignment_v", "wide_aligned"},
        {"alignment_o", "wide_aligned"},
        {"tensor_overlap", "separate"},
        {"causal_mask", "absent"},
        {"sliding_window", "absent"},
        {"block_mask", "absent"},
        {"bias", "absent"},
        {"alibi", "absent"},
        {"attention_sinks", "absent"},
        {"scale_source", "attribute_value"},
        {"scale_value", "inverse_sqrt_d"},
        {"dropout", "absent"},
        {"stats_output", "not_requested"},
        {"max_sumexp_output", "not_requested"},
        {"amax_s", "not_requested"},
        {"amax_o", "not_requested"},
        {"descale_q", "absent"},
        {"descale_k", "absent"},
        {"descale_v", "absent"},
        {"s_quantization", "absent"},
        {"scale_o", "absent"},
        {"paged_kv", "absent"},
        {"implementation", "auto"},
        {"runtime_shape_override", "disabled"},
    };
    const auto it = BASE.find(axis);
    return it == BASE.end() ? std::string("-") : it->second;
}

BuiltCase buildCaseGraph(const Row& row,
                         const std::vector<std::string>& overrideTokens,
                         const CaseContext& context)
{
    const Overrides ov = parseOverrides(overrideTokens);
    CaseBuilder builder(row, ov, context);
    return builder.build();
}

SizeOverride parseSizeOverride(const std::string& text)
{
    const std::vector<std::pair<std::string, int64_t SizeOverride::*>> fields = {
        {"Dqk", &SizeOverride::dqk},
        {"Skv", &SizeOverride::skv},
        {"Hq", &SizeOverride::hq},
        {"Hk", &SizeOverride::hk},
        {"Hv", &SizeOverride::hv},
        {"Sq", &SizeOverride::sq},
        {"Dv", &SizeOverride::dv},
        {"B", &SizeOverride::b},
    };
    SizeOverride sizes;
    std::set<std::string> seen;
    std::istringstream stream(text);
    std::string token;
    while(stream >> token)
    {
        bool matched = false;
        for(const auto& [prefix, member] : fields)
        {
            if(token.rfind(prefix, 0) != 0 || token.size() == prefix.size())
            {
                continue;
            }
            const auto digits = token.substr(prefix.size());
            if(digits.find_first_not_of("0123456789") != std::string::npos)
            {
                continue;
            }
            sizes.*member = std::stoll(digits);
            if(!seen.insert(prefix).second || sizes.*member <= 0)
            {
                fail("size token '" + token + "' repeats or is not positive in '" + text + "'");
            }
            matched = true;
            break;
        }
        if(!matched)
        {
            fail("unknown size token '" + token + "' in '" + text + "'");
        }
    }
    if(seen.size() != fields.size())
    {
        fail("sizes '" + text + "' need B, Hq, Hk, Hv, Sq, Skv, Dqk and Dv");
    }
    return sizes;
}

} // namespace sdpa_case_flow
