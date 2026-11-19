// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "Fp64WrwReference.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <hipdnn_bench/CandidatePageJson.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/ScopedResource.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/detail/EngineQueries.hpp>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <unordered_map>

using json = nlohmann::json;
using namespace hipdnn_frontend;
using hip_kernel_provider::conv_validation::checkHip;
using hip_kernel_provider::conv_validation::Fp64WrwReference;
using hip_kernel_provider::conv_validation::ReferenceGeometry;
namespace
{
class ValidationGraph : public graph::Graph
{
public:
    using Graph::get_raw_graph_descriptor;
};
void frontend(const Error& result)
{
    if(result.is_bad())
    {
        throw std::runtime_error(result.get_message());
    }
}
void backend(hipdnnStatus_t result)
{
    if(result != HIPDNN_STATUS_SUCCESS)
    {
        throw std::runtime_error(hipdnnGetErrorString(result));
    }
}
int64_t product(const std::vector<int64_t>& values)
{
    int64_t result = 1;
    for(const auto value : values)
    {
        if(value <= 0 || result > std::numeric_limits<int64_t>::max() / value)
        {
            throw std::runtime_error("Invalid or overflowing tensor dimensions");
        }
        result *= value;
    }
    return result;
}
std::array<int64_t, 5> normalize(const std::vector<int64_t>& source, int64_t fill)
{
    if(source.size() < 3 || source.size() > 5)
    {
        throw std::runtime_error("Convolution rank must be 1..3");
    }
    std::array<int64_t, 5> result{fill, fill, fill, fill, fill};
    result[0] = source[0];
    result[1] = source[1];
    for(size_t i = 2; i < source.size(); ++i)
    {
        result[i + 5 - source.size()] = source[i];
    }
    return result;
}
struct Tensor
{
    int64_t uid = 0;
    enum class Type
    {
        FLOAT,
        HALF,
        BFLOAT16
    };
    Type type = Type::FLOAT;
    size_t elementBytes() const
    {
        return type == Type::FLOAT ? sizeof(float) : sizeof(uint16_t);
    }
    double tolerance() const
    {
        if(type == Type::FLOAT)
        {
            return 0.0001;
        }
        if(type == Type::HALF)
        {
            return 0.002;
        }
        return 0.02;
    }
    std::array<int64_t, 5> dimensions{}, strides{};
    size_t words = 0;
};
Tensor tensor(const json& source)
{
    Tensor result;
    const auto type = source.at("data_type").get<std::string>();
    if(source.value("virtual", false) || (type != "float" && type != "half" && type != "bfloat16"))
    {
        throw std::runtime_error(
            "Independent WRW validation requires real FLOAT/HALF/BFLOAT16 tensors");
    }
    if(type == "float")
    {
        result.type = Tensor::Type::FLOAT;
    }
    else if(type == "half")
    {
        result.type = Tensor::Type::HALF;
    }
    else
    {
        result.type = Tensor::Type::BFLOAT16;
    }
    result.uid = source.at("uid").get<int64_t>();
    auto dimensions = source.at("dims").get<std::vector<int64_t>>();
    auto strides = source.at("strides").get<std::vector<int64_t>>();
    if(dimensions.size() != strides.size())
    {
        throw std::runtime_error("Tensor rank/stride mismatch");
    }
    product(dimensions);
    result.dimensions = normalize(dimensions, 1);
    result.strides = normalize(strides, 0);
    int64_t span = 1;
    for(size_t i = 0; i < dimensions.size(); ++i)
    {
        int64_t extent = 0;
        if(strides[i] < 0 || (dimensions[i] > 1 && strides[i] == 0)
           || __builtin_mul_overflow(dimensions[i] - 1, strides[i], &extent)
           || __builtin_add_overflow(span, extent, &span))
        {
            throw std::runtime_error("Invalid or overflowing tensor strides");
        }
    }
    if(static_cast<uint64_t>(span) > SIZE_MAX / sizeof(float))
    {
        throw std::runtime_error("Tensor byte span overflow");
    }
    result.words = static_cast<size_t>(span);
    return result;
}
struct Problem
{
    Tensor x, dy, dw;
    ReferenceGeometry geometry{};
};
Problem problem(const json& row)
{
    const auto& g = row.at("graph");
    if(g.at("nodes").size() != 1 || g.at("nodes").at(0).at("type") != "ConvolutionWrwAttributes")
    {
        throw std::runtime_error("This validator supports one backward-weights convolution");
    }
    const auto& node = g.at("nodes").at(0);
    std::map<int64_t, Tensor> tensors;
    for(const auto& item : g.at("tensors"))
    {
        const auto parsed = tensor(item);
        if(!tensors.emplace(parsed.uid, parsed).second)
        {
            throw std::runtime_error("Duplicate tensor UID");
        }
    }
    Problem p;
    p.x = tensors.at(node.at("inputs").at("x_tensor_uid").get<int64_t>());
    p.dy = tensors.at(node.at("inputs").at("dy_tensor_uid").get<int64_t>());
    p.dw = tensors.at(node.at("outputs").at("dw_tensor_uid").get<int64_t>());
    if(p.x.type != p.dy.type || p.x.type != p.dw.type)
    {
        throw std::runtime_error("WRW input/dy/output storage types must agree");
    }
    const auto compute = node.value("compute_data_type", "unset");
    if(compute != "unset" && compute != "float"
       && (compute != "half" || p.x.type != Tensor::Type::HALF))
    {
        throw std::runtime_error("Unsupported convolution compute/storage type pairing");
    }
    auto& a = p.geometry;
    a.n = p.x.dimensions[0];
    a.c = p.x.dimensions[1];
    a.k = p.dy.dimensions[1];
    if(p.dy.dimensions[0] != a.n || p.dw.dimensions[0] != a.k || p.dw.dimensions[1] <= 0
       || a.c % p.dw.dimensions[1] != 0)
    {
        throw std::runtime_error("Inconsistent WRW channel dimensions");
    }
    a.groups = a.c / p.dw.dimensions[1];
    if(a.k % a.groups != 0)
    {
        throw std::runtime_error("Inconsistent WRW groups");
    }
    for(size_t i = 0; i < 3; ++i)
    {
        a.input[i] = p.x.dimensions[i + 2];
        a.output[i] = p.dy.dimensions[i + 2];
        a.filter[i] = p.dw.dimensions[i + 2];
        a.stride[i] = a.dilation[i] = 1;
    }
    std::copy(p.x.strides.begin(), p.x.strides.end(), a.xs);
    std::copy(p.dy.strides.begin(), p.dy.strides.end(), a.ys);
    const auto& params = node.at("parameters");
    const auto stride = params.at("stride").get<std::vector<int64_t>>();
    const auto dilation = params.at("dilation").get<std::vector<int64_t>>();
    const auto padding = params.at("pre_padding").get<std::vector<int64_t>>();
    if(stride.empty() || stride.size() > 3 || stride.size() != dilation.size()
       || stride.size() != padding.size())
    {
        throw std::runtime_error("Invalid convolution parameters");
    }
    for(size_t i = 0; i < stride.size(); ++i)
    {
        if(stride[i] <= 0 || dilation[i] <= 0 || padding[i] < 0)
        {
            throw std::runtime_error("Invalid convolution stride/dilation/padding");
        }
        const size_t axis = i + 3 - stride.size();
        a.stride[axis] = stride[i];
        a.dilation[axis] = dilation[i];
        a.padding[axis] = padding[i];
        int64_t reach = 0;
        int64_t taps = 0;
        if(__builtin_mul_overflow(a.output[axis] - 1, stride[i], &reach)
           || __builtin_mul_overflow(a.filter[axis] - 1, dilation[i], &taps)
           || __builtin_add_overflow(reach, taps, &reach))
        {
            throw std::runtime_error("Convolution reach overflow");
        }
    }
    const auto mode = params.at("conv_mode").get<std::string>();
    if(mode != "CONVOLUTION" && mode != "CROSS_CORRELATION")
    {
        throw std::runtime_error("Unknown convolution mode");
    }
    a.flip = mode == "CONVOLUTION" ? 1 : 0;
    a.weights = product(std::vector<int64_t>(p.dw.dimensions.begin(), p.dw.dimensions.end()));
    return p;
}
uint32_t inputHash(uint32_t index, uint32_t seed)
{
    uint32_t h = index ^ seed;
    h ^= h >> 16;
    h *= 0x7feb352dU;
    h ^= h >> 15;
    h *= 0x846ca68bU;
    h ^= h >> 16;
    return h;
}
float decode16(uint16_t bits, Tensor::Type type)
{
    if(type == Tensor::Type::HALF)
    {
        return hipdnn_data_sdk::types::detail::half_bits_to_float(bits);
    }
    const uint32_t word = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}
void fill(void* device, float* reference, const Tensor& t, hipStream_t stream)
{
    const size_t capacity = std::min<size_t>(t.words, 262144);
    const uint32_t seed = static_cast<uint32_t>(t.uid) * 41U + 17U;
    if(t.type == Tensor::Type::FLOAT)
    {
        std::vector<uint32_t> block(capacity);
        for(size_t offset = 0; offset < t.words; offset += capacity)
        {
            const size_t count = std::min(capacity, t.words - offset);
            for(size_t i = 0; i < count; ++i)
            {
                const auto h = inputHash(static_cast<uint32_t>(offset + i), seed);
                block[i] = 0x3e000000U | (h & 0x007fffffU) | (h & 0x80000000U);
            }
            checkHip(hipMemcpyAsync(static_cast<float*>(device) + offset,
                                    block.data(),
                                    count * sizeof(float),
                                    hipMemcpyHostToDevice,
                                    stream),
                     "Input upload");
            checkHip(hipStreamSynchronize(stream), "Input upload synchronization");
        }
        return;
    }
    std::vector<uint16_t> block(capacity);
    std::vector<float> expanded(capacity);
    for(size_t offset = 0; offset < t.words; offset += capacity)
    {
        const size_t count = std::min(capacity, t.words - offset);
        for(size_t i = 0; i < count; ++i)
        {
            const auto h = inputHash(static_cast<uint32_t>(offset + i), seed);
            block[i] = static_cast<uint16_t>((t.type == Tensor::Type::HALF ? 0x3000U : 0x3e00U)
                                             | (h & (t.type == Tensor::Type::HALF ? 0x3ffU : 0x7fU))
                                             | ((h >> 16) & 0x8000U));
            expanded[i] = decode16(block[i], t.type);
        }
        checkHip(hipMemcpyAsync(static_cast<uint16_t*>(device) + offset,
                                block.data(),
                                count * sizeof(uint16_t),
                                hipMemcpyHostToDevice,
                                stream),
                 "Typed input upload");
        checkHip(hipMemcpyAsync(reference + offset,
                                expanded.data(),
                                count * sizeof(float),
                                hipMemcpyHostToDevice,
                                stream),
                 "Exact input expansion");
        checkHip(hipStreamSynchronize(stream), "Input expansion synchronization");
    }
}
json compare(const Problem& p, const void* device, const std::vector<double>& expected)
{
    std::vector<uint8_t> actual(p.dw.words * p.dw.elementBytes());
    checkHip(hipMemcpy(actual.data(), device, actual.size(), hipMemcpyDeviceToHost),
             "Output readback");
    size_t failed = 0;
    size_t nonfinite = 0;
    double worst = 0.0;
    double absolute = 0.0;
    const double tolerance = p.dw.tolerance();
    for(size_t i = 0; i < expected.size(); ++i)
    {
        size_t rest = i;
        size_t offset = 0;
        for(size_t axis = 5; axis-- > 0;)
        {
            const size_t coordinate = rest % static_cast<size_t>(p.dw.dimensions[axis]);
            rest /= static_cast<size_t>(p.dw.dimensions[axis]);
            offset += coordinate * static_cast<size_t>(p.dw.strides[axis]);
        }
        double value;
        if(p.dw.type == Tensor::Type::FLOAT)
        {
            float stored;
            std::memcpy(&stored, actual.data() + offset * sizeof(float), sizeof(stored));
            value = static_cast<double>(stored);
        }
        else
        {
            uint16_t stored;
            std::memcpy(&stored, actual.data() + offset * sizeof(uint16_t), sizeof(stored));
            value = static_cast<double>(decode16(stored, p.dw.type));
        }
        if(!std::isfinite(value) || !std::isfinite(expected[i]))
        {
            ++failed;
            ++nonfinite;
            continue;
        }
        const double error = std::abs(value - expected[i]);
        const double normalized = error / (tolerance + tolerance * std::abs(expected[i]));
        if(normalized > 1.0)
        {
            ++failed;
        }
        worst = std::max(worst, normalized);
        absolute = std::max(absolute, error);
    }
    return {{"reference", "INDEPENDENT_FP64_WRW"},
            {"passed", failed == 0},
            {"compared", expected.size()},
            {"failed", failed},
            {"nonfinite", nonfinite},
            {"max_normalized_error", worst},
            {"max_abs_error", absolute},
            {"atol", tolerance},
            {"rtol", tolerance}};
}
std::unique_ptr<ValidationGraph> makePlan(const json& row,
                                          hipdnnHandle_t handle,
                                          int64_t engine,
                                          const std::vector<KnobSetting>& knobs)
{
    auto g = std::make_unique<ValidationGraph>();
    frontend(g->deserialize(handle, row.at("graph").dump()));
    frontend(g->create_execution_plan_ext(engine, knobs));
    frontend(g->build_plans());
    return g;
}
json validate(const json& row,
              hipdnnHandle_t handle,
              hipStream_t stream,
              const Fp64WrwReference& oracle)
{
    const Problem p = problem(row);
    const hipdnn_data_sdk::utilities::Workspace<> x(p.x.words * p.x.elementBytes());
    const hipdnn_data_sdk::utilities::Workspace<> dy(p.dy.words * p.dy.elementBytes());
    const hipdnn_data_sdk::utilities::Workspace<> dw(p.dw.words * p.dw.elementBytes());
    const hipdnn_data_sdk::utilities::Workspace<> expandedX(
        p.x.type == Tensor::Type::FLOAT ? 0 : p.x.words * sizeof(float));
    const hipdnn_data_sdk::utilities::Workspace<> expandedDy(
        p.dy.type == Tensor::Type::FLOAT ? 0 : p.dy.words * sizeof(float));
    const auto* referenceX = p.x.type == Tensor::Type::FLOAT
                                 ? static_cast<const float*>(x.get())
                                 : static_cast<const float*>(expandedX.get());
    const auto* referenceDy = p.dy.type == Tensor::Type::FLOAT
                                  ? static_cast<const float*>(dy.get())
                                  : static_cast<const float*>(expandedDy.get());
    fill(x.get(), static_cast<float*>(expandedX.get()), p.x, stream);
    fill(dy.get(), static_cast<float*>(expandedDy.get()), p.dy, stream);
    std::unordered_map<int64_t, void*> pack{
        {p.x.uid, x.get()}, {p.dy.uid, dy.get()}, {p.dw.uid, dw.get()}};
    const auto truth = oracle.execute(referenceX, referenceDy, p.geometry, stream);
    ValidationGraph query;
    frontend(query.deserialize(handle, row.at("graph").dump()));
    std::vector<EngineConfigInfo> engines;
    frontend(query.get_engine_configs(handle, engines));
    std::vector<EngineVariant> variants;
    for(const auto& engine : engines)
    {
        if(engine.engineName != "hipkernel:CONV_WRW")
        {
            continue;
        }
        int64_t offset = 0;
        for(;;)
        {
            EngineCandidatePage page;
            frontend(detail::getEngineCandidates(
                query.get_raw_graph_descriptor(), engine.engineId, page, offset, 10000));
            for(const auto& candidate : page.candidates)
            {
                variants.push_back(candidate.variant);
            }
            if(!page.nextOffset)
            {
                break;
            }
            offset = static_cast<int64_t>(*page.nextOffset);
        }
    }
    if(variants.empty())
    {
        throw std::runtime_error("No applicable WRW candidates");
    }
    frontend(query.add_engine_variants(variants));
    int64_t maxWorkspace = 0;
    frontend(query.get_estimated_max_workspace_size(maxWorkspace));
    const hipdnn_data_sdk::utilities::Workspace<> workspace(static_cast<size_t>(maxWorkspace));
    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    // Screening is bounded even for deliberately slow accuracy-floor candidates;
    // accepted finalists receive the independent 30-iteration refinement below.
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.warmupIterations = 3;
    config.timedIterations = 3;
    config.rankingFn = [](auto& values) {
        std::stable_sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
            return a.robustTimeMs < b.robustTimeMs;
        });
    };
    std::vector<AutotuneResult> measured;
    const auto timed
        = query.autotune(handle, pack, workspace.get(), maxWorkspace, config, {}, &measured);
    if(measured.size() != variants.size())
    {
        throw std::runtime_error("Incomplete candidate accounting: " + timed.get_message());
    }
    json result{{"line", row.at("line")},
                {"status", "no_numerically_valid_candidate"},
                {"reference", "INDEPENDENT_FP64_WRW"},
                {"command", row.value("command", "")},
                {"candidate_count", variants.size()},
                {"screening_iterations", 3},
                {"refinement_iterations", 30},
                {"candidates", json::array()}};
    std::vector<EngineVariant> qualified;
    for(const auto& candidate : measured)
    {
        json entry{{"engine", candidate.engineName},
                   {"succeeded", candidate.succeeded},
                   {"error", candidate.errorMessage},
                   {"knobs",
                    hipdnn_bench::knobJson(hipdnn_bench::toVariantKnobs(candidate.knobSettings))}};
        if(candidate.succeeded)
        {
            entry["robust_us"] = static_cast<double>(candidate.robustTimeMs) * 1000.0;
            auto g = makePlan(row, handle, candidate.engineId, candidate.knobSettings);
            int64_t bytes = 0;
            frontend(g->get_workspace_size(bytes));
            const hipdnn_data_sdk::utilities::Workspace<> scratch(static_cast<size_t>(bytes));
            checkHip(hipMemsetAsync(dw.get(), 0xff, p.dw.words * p.dw.elementBytes(), stream),
                     "Output poisoning");
            frontend(g->execute(handle, pack, scratch.get()));
            checkHip(hipStreamSynchronize(stream), "Candidate execution");
            entry["reference_correctness"] = compare(p, dw.get(), truth);
            if(entry["reference_correctness"]["passed"].get<bool>() && qualified.size() < 5)
            {
                qualified.push_back(
                    {candidate.engineId, hipdnn_bench::toVariantKnobs(candidate.knobSettings)});
            }
        }
        result["candidates"].push_back(std::move(entry));
    }
    if(!qualified.empty())
    {
        ValidationGraph final;
        frontend(final.deserialize(handle, row.at("graph").dump()));
        frontend(final.add_engine_variants(qualified));
        int64_t bytes = 0;
        frontend(final.get_estimated_max_workspace_size(bytes));
        const hipdnn_data_sdk::utilities::Workspace<> scratch(static_cast<size_t>(bytes));
        config.strategy = AutotuneStrategy::FIXED_AVERAGE;
        config.warmupIterations = 30;
        config.timedIterations = 30;
        std::vector<AutotuneResult> refined;
        frontend(final.autotune(handle, pack, scratch.get(), bytes, config, {}, &refined));
        if(refined.empty() || !refined.front().succeeded)
        {
            throw std::runtime_error("No runnable qualified finalist");
        }
        frontend(final.execute(handle, pack, scratch.get()));
        checkHip(hipStreamSynchronize(stream), "Final execution");
        result["reference_correctness"] = compare(p, dw.get(), truth);
        result["status"] = result["reference_correctness"]["passed"].get<bool>()
                               ? "tuned"
                               : "winner_reference_failed";
        result["best"] = {
            {"engine", refined.front().engineName},
            {"robust_us", static_cast<double>(refined.front().robustTimeMs) * 1000.0},
            {"knobs",
             hipdnn_bench::knobJson(hipdnn_bench::toVariantKnobs(refined.front().knobSettings))}};
    }
    int64_t miopen = 0;
    const auto found = hipdnnGetEngineIdByName_ext(handle, "MIOPEN_ENGINE", &miopen);
    if(found == HIPDNN_STATUS_SUCCESS)
    {
        try
        {
            auto g = makePlan(row, handle, miopen, {});
            int64_t bytes = 0;
            frontend(g->get_workspace_size(bytes));
            const hipdnn_data_sdk::utilities::Workspace<> scratch(static_cast<size_t>(bytes));
            frontend(g->execute(handle, pack, scratch.get()));
            checkHip(hipStreamSynchronize(stream), "MIOpen diagnostic execution");
            result["miopen_diagnostic"] = compare(p, dw.get(), truth);
        }
        catch(const std::exception& error)
        {
            result["miopen_diagnostic"]
                = {{"available", true}, {"executed", false}, {"error", error.what()}};
        }
    }
    else
    {
        result["miopen_diagnostic"]
            = {{"available", false}, {"error", hipdnnGetErrorString(found)}};
    }
    return result;
}
} // namespace
int main(int argc, char** argv)
{
    try
    {
        if(argc == 2 && std::string(argv[1]) == "--help")
        {
            std::cout << "Usage: hipkernel_validate_wrw MANIFEST.jsonl RESULTS.jsonl PLUGIN_DIR "
                         "MIOPEN_PLUGIN EXPECTED_PCI\n"
                         "Strict FLOAT/HALF/BFLOAT16 WRW validation against compensated FP64 "
                         "convolution; MIOpen is diagnostic only.\n";
            return 0;
        }
        if(argc != 6)
        {
            throw std::runtime_error("Expected manifest, output, plugin directory, MIOpen plugin "
                                     "and expected PCI; use --help");
        }
        std::ifstream input(argv[1]);
        if(!input)
        {
            throw std::runtime_error("Cannot read manifest");
        }
        std::vector<json> rows;
        std::string text;
        while(std::getline(input, text))
        {
            if(text.empty())
            {
                continue;
            }
            rows.push_back(json::parse(text));
            problem(rows.back());
        }
        if(rows.empty())
        {
            throw std::runtime_error("Empty manifest");
        }
        int count = 0;
        std::array<char, 64> pci{};
        checkHip(hipGetDeviceCount(&count), "Device count");
        checkHip(hipDeviceGetPCIBusId(pci.data(), static_cast<int>(pci.size()), 0), "Device PCI");
        if(count != 1 || std::string(pci.data()) != argv[5])
        {
            throw std::runtime_error("Single-device isolation or expected PCI mismatch");
        }
        std::cerr << "Isolated device PCI=" << pci.data() << '\n';
        const std::array<const char*, 2> directories = {argv[3], argv[4]};
        backend(
            hipdnnSetEnginePluginPaths_ext(2, directories.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE));
        hipdnnHandle_t handle = nullptr;
        backend(hipdnnCreate(&handle));
        const hipdnn_data_sdk::utilities::ScopedResource owned(handle, hipdnnDestroy);
        hipStream_t stream = nullptr;
        checkHip(hipStreamCreate(&stream), "Stream creation");
        const hipdnn_data_sdk::utilities::ScopedResource ownedStream(stream, hipStreamDestroy);
        backend(hipdnnSetStream(handle, stream));
        const Fp64WrwReference reference;
        std::ofstream output(argv[2], std::ios::app);
        if(!output)
        {
            throw std::runtime_error("Cannot open result output");
        }
        bool complete = true;
        for(const auto& row : rows)
        {
            std::cerr << "VALIDATE line=" << row.at("line") << '\n';
            auto result = validate(row, handle, stream, reference);
            result["gpu_pci"] = pci.data();
            complete = complete && result["status"] == "tuned";
            output << result.dump() << '\n';
            output.flush();
            if(!output)
            {
                throw std::runtime_error("Result write failed");
            }
            std::cerr << "DONE line=" << row.at("line") << " status=" << result["status"] << '\n';
        }
        return complete ? 0 : 2;
    }
    catch(const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
