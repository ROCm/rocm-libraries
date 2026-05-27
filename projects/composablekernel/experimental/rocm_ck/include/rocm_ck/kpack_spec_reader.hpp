// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Role: host — read variant specs from a kpack archive's TOC.
//
// Reads the kpack file header + msgpack TOC directly (does not use the
// kpack C API). Deserializes variant_specs into GemmSpec / ElementwiseSpec.
//
// Requires: msgpack-cxx

#pragma once

#include <rocm_ck/arch_properties.hpp>
#include <rocm_ck/elementwise_spec.hpp>
#include <rocm_ck/gemm_spec.hpp>

#include <msgpack.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace rocm_ck {

// ============================================================================
// String-to-enum parsers
// ============================================================================

inline DataType parseDataType(const std::string& s)
{
    if(s == "FP64")
        return DataType::FP64;
    if(s == "FP32")
        return DataType::FP32;
    if(s == "FP16")
        return DataType::FP16;
    if(s == "BF16")
        return DataType::BF16;
    if(s == "FP8_FNUZ")
        return DataType::FP8_FNUZ;
    if(s == "BF8_FNUZ")
        return DataType::BF8_FNUZ;
    if(s == "FP8_OCP")
        return DataType::FP8_OCP;
    if(s == "BF8_OCP")
        return DataType::BF8_OCP;
    if(s == "I4")
        return DataType::I4;
    if(s == "I8")
        return DataType::I8;
    if(s == "I16")
        return DataType::I16;
    if(s == "I32")
        return DataType::I32;
    if(s == "I64")
        return DataType::I64;
    if(s == "U8")
        return DataType::U8;
    if(s == "U16")
        return DataType::U16;
    if(s == "U32")
        return DataType::U32;
    if(s == "U64")
        return DataType::U64;
    return DataType::FP32; // default
}

inline Layout parseLayout(const std::string& s)
{
    if(s == "Row")
        return Layout::Row;
    if(s == "Col")
        return Layout::Col;
    if(s == "Contiguous")
        return Layout::Contiguous;
    return Layout::Row; // default
}

inline Pipeline parsePipeline(const std::string& s)
{
    if(s == "V1")
        return Pipeline::V1;
    if(s == "V3")
        return Pipeline::V3;
    if(s == "V4")
        return Pipeline::V4;
    if(s == "Memory")
        return Pipeline::Memory;
    if(s == "Preshuffle")
        return Pipeline::Preshuffle;
    return Pipeline::V1; // default
}

inline PipelineScheduler parseScheduler(const std::string& s)
{
    if(s == "Interwave")
        return PipelineScheduler::Interwave;
    return PipelineScheduler::Intrawave; // default
}

inline TilePartitioner parsePartitioner(const std::string& s)
{
    if(s == "Direct")
        return TilePartitioner::Direct;
    if(s == "SpatiallyLocal")
        return TilePartitioner::SpatiallyLocal;
    if(s == "StreamK")
        return TilePartitioner::StreamK;
    return TilePartitioner::Linear; // default
}

inline StoreStrategy parseStoreStrategy(const std::string& s)
{
    if(s == "Direct2D")
        return StoreStrategy::Direct2D;
    return StoreStrategy::CShuffle; // default
}

inline EpilogueOp parseEpilogueOp(const std::string& s)
{
    if(s == "Add")
        return EpilogueOp::Add;
    if(s == "Mul")
        return EpilogueOp::Mul;
    if(s == "Relu")
        return EpilogueOp::Relu;
    if(s == "FastGelu")
        return EpilogueOp::FastGelu;
    if(s == "Gelu")
        return EpilogueOp::Gelu;
    if(s == "Silu")
        return EpilogueOp::Silu;
    if(s == "Sigmoid")
        return EpilogueOp::Sigmoid;
    return EpilogueOp::Add; // fallback
}

inline GpuTarget parseGpuTarget(const std::string& s)
{
    if(s == "gfx90a")
        return GpuTarget::gfx90a;
    if(s == "gfx942")
        return GpuTarget::gfx942;
    if(s == "gfx950")
        return GpuTarget::gfx950;
    if(s == "gfx1100")
        return GpuTarget::gfx1100;
    if(s == "gfx1101")
        return GpuTarget::gfx1101;
    if(s == "gfx1102")
        return GpuTarget::gfx1102;
    if(s == "gfx1150")
        return GpuTarget::gfx1150;
    if(s == "gfx1151")
        return GpuTarget::gfx1151;
    if(s == "gfx1200")
        return GpuTarget::gfx1200;
    if(s == "gfx1201")
        return GpuTarget::gfx1201;
    return GpuTarget::gfx90a; // fallback
}

// ============================================================================
// Msgpack helpers
// ============================================================================

namespace detail {

inline const msgpack::object* find(const msgpack::object_map& map, const char* key)
{
    for(uint32_t i = 0; i < map.size; ++i)
    {
        const auto& kv = map.ptr[i];
        if(kv.key.type == msgpack::type::STR &&
           std::string(kv.key.via.str.ptr, kv.key.via.str.size) == key)
            return &kv.val;
    }
    return nullptr;
}

inline std::string getString(const msgpack::object_map& map, const char* key, const char* def = "")
{
    const auto* v = find(map, key);
    if(v && v->type == msgpack::type::STR)
        return std::string(v->via.str.ptr, v->via.str.size);
    return def;
}

inline int getInt(const msgpack::object_map& map, const char* key, int def = 0)
{
    const auto* v = find(map, key);
    if(v &&
       (v->type == msgpack::type::POSITIVE_INTEGER || v->type == msgpack::type::NEGATIVE_INTEGER))
        return static_cast<int>(v->via.i64);
    return def;
}

inline bool getBool(const msgpack::object_map& map, const char* key, bool def = false)
{
    const auto* v = find(map, key);
    if(v && v->type == msgpack::type::BOOLEAN)
        return v->via.boolean;
    return def;
}

inline Dim3 getDim3(const msgpack::object_map& map, const char* key)
{
    const auto* v = find(map, key);
    if(v && v->type == msgpack::type::MAP)
    {
        const auto& m = v->via.map;
        return {getInt(m, "m"), getInt(m, "n"), getInt(m, "k")};
    }
    return {0, 0, 0};
}

inline PhysicalTensor parsePhysicalTensor(const msgpack::object_map& map)
{
    std::string name_str = getString(map, "name");
    TensorName tn;
    int len = static_cast<int>(name_str.size());
    if(len > 15)
        len = 15;
    for(int i = 0; i < len; ++i)
        tn.data[i] = name_str[i];
    tn.len = len;

    return {tn,
            parseDataType(getString(map, "dtype", "FP32")),
            parseLayout(getString(map, "layout", "Row")),
            getInt(map, "args_slot")};
}

} // namespace detail

// ============================================================================
// GemmSpec deserialization from msgpack
// ============================================================================

inline GemmSpec gemm_spec_from_msgpack(const msgpack::object& obj)
{
    if(obj.type != msgpack::type::MAP)
    {
        std::fprintf(stderr, "gemm_spec_from_msgpack: expected MAP\n");
        return {};
    }
    const auto& map = obj.via.map;

    GemmSpec spec{};

    // Physical tensors
    const auto* pt_obj = detail::find(map, "physical_tensors");
    if(pt_obj && pt_obj->type == msgpack::type::ARRAY)
    {
        spec.num_physical_tensors = static_cast<int>(pt_obj->via.array.size);
        if(spec.num_physical_tensors > kMaxPhysicalTensors)
            spec.num_physical_tensors = kMaxPhysicalTensors;
        for(int i = 0; i < spec.num_physical_tensors; ++i)
        {
            if(pt_obj->via.array.ptr[i].type == msgpack::type::MAP)
                spec.physical_tensors[i] =
                    detail::parsePhysicalTensor(pt_obj->via.array.ptr[i].via.map);
        }
    }

    spec.acc_dtype      = parseDataType(detail::getString(map, "acc_dtype", "FP32"));
    spec.block_tile     = detail::getDim3(map, "block_tile");
    spec.block_waves    = detail::getDim3(map, "block_waves");
    spec.wave_tile      = detail::getDim3(map, "wave_tile");
    spec.workgroup_size = detail::getInt(map, "workgroup_size", 256);
    spec.k_batch        = detail::getInt(map, "k_batch", 1);
    spec.pipeline       = parsePipeline(detail::getString(map, "pipeline", "V1"));
    spec.pipeline_scheduler =
        parseScheduler(detail::getString(map, "pipeline_scheduler", "Intrawave"));
    spec.tile_partitioner = parsePartitioner(detail::getString(map, "tile_partitioner", "Linear"));

    // Epilogue ops
    const auto* epi_obj = detail::find(map, "epilogue_ops");
    if(epi_obj && epi_obj->type == msgpack::type::ARRAY)
    {
        spec.num_epilogue_ops = static_cast<int>(epi_obj->via.array.size);
        if(spec.num_epilogue_ops > kMaxEpilogueOps)
            spec.num_epilogue_ops = kMaxEpilogueOps;
        for(int i = 0; i < spec.num_epilogue_ops; ++i)
        {
            if(epi_obj->via.array.ptr[i].type == msgpack::type::STR)
            {
                std::string op_str(epi_obj->via.array.ptr[i].via.str.ptr,
                                   epi_obj->via.array.ptr[i].via.str.size);
                spec.epilogue_ops[i] = parseEpilogueOp(op_str);
            }
        }
    }

    spec.store_strategy = parseStoreStrategy(detail::getString(map, "store_strategy", "CShuffle"));
    spec.pad_m          = detail::getBool(map, "pad_m");
    spec.pad_n          = detail::getBool(map, "pad_n");
    spec.group_size     = detail::getInt(map, "group_size");

    return spec;
}

// ============================================================================
// ElementwiseSpec deserialization from msgpack
// ============================================================================

inline ElementwiseSpec elementwise_spec_from_msgpack(const msgpack::object& obj)
{
    if(obj.type != msgpack::type::MAP)
    {
        std::fprintf(stderr, "elementwise_spec_from_msgpack: expected MAP\n");
        return {};
    }
    const auto& map = obj.via.map;

    ElementwiseSpec spec{};

    const auto* pt_obj = detail::find(map, "physical_tensors");
    if(pt_obj && pt_obj->type == msgpack::type::ARRAY)
    {
        spec.num_physical_tensors = static_cast<int>(pt_obj->via.array.size);
        if(spec.num_physical_tensors > kMaxPhysicalTensors)
            spec.num_physical_tensors = kMaxPhysicalTensors;
        for(int i = 0; i < spec.num_physical_tensors; ++i)
        {
            if(pt_obj->via.array.ptr[i].type == msgpack::type::MAP)
                spec.physical_tensors[i] =
                    detail::parsePhysicalTensor(pt_obj->via.array.ptr[i].via.map);
        }
    }

    spec.block_tile     = detail::getInt(map, "block_tile");
    spec.workgroup_size = detail::getInt(map, "workgroup_size", 256);
    spec.block_waves    = detail::getInt(map, "block_waves", 1);
    spec.wave_tile      = detail::getInt(map, "wave_tile");
    spec.pad            = detail::getBool(map, "pad");

    return spec;
}

// ============================================================================
// VariantInfo — runtime variant descriptor read from a kpack TOC
// ============================================================================

struct VariantInfo
{
    std::string name;
    std::string spec_type; // "GemmSpec" or "ElementwiseSpec"
    TargetSet targets;
    GemmSpec gemm_spec;               // valid when spec_type == "GemmSpec"
    ElementwiseSpec elementwise_spec; // valid when spec_type == "ElementwiseSpec"
};

// ============================================================================
// readVariantSpecs — read all variant specs from a kpack file's TOC
// ============================================================================

/// Read variant_specs from a kpack archive file.
/// Returns empty vector if the file has no variant_specs (old format).
inline std::vector<VariantInfo> readVariantSpecs(const char* kpack_path)
{
    std::vector<VariantInfo> result;

    FILE* f = std::fopen(kpack_path, "rb");
    if(!f)
    {
        std::fprintf(stderr, "readVariantSpecs: cannot open '%s'\n", kpack_path);
        return result;
    }

    // Read header
    char magic[4];
    uint32_t version;
    uint64_t toc_offset;
    if(std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "KPAK", 4) != 0)
    {
        std::fprintf(stderr, "readVariantSpecs: not a kpack file\n");
        std::fclose(f);
        return result;
    }
    if(std::fread(&version, 4, 1, f) != 1 || std::fread(&toc_offset, 8, 1, f) != 1)
    {
        std::fclose(f);
        return result;
    }

    // Read TOC
    std::fseek(f, 0, SEEK_END);
    long file_size = std::ftell(f);
    if(static_cast<long>(toc_offset) >= file_size)
    {
        std::fclose(f);
        return result;
    }
    std::fseek(f, static_cast<long>(toc_offset), SEEK_SET);
    size_t toc_size = static_cast<size_t>(file_size) - static_cast<size_t>(toc_offset);
    std::vector<char> toc_buf(toc_size);
    if(std::fread(toc_buf.data(), 1, toc_size, f) != toc_size)
    {
        std::fclose(f);
        return result;
    }
    std::fclose(f);

    // Parse msgpack
    msgpack::object_handle oh;
    try
    {
        oh = msgpack::unpack(toc_buf.data(), toc_buf.size());
    }
    catch(...)
    {
        std::fprintf(stderr, "readVariantSpecs: msgpack parse failed\n");
        return result;
    }

    const auto& root = oh.get();
    if(root.type != msgpack::type::MAP)
        return result;

    const auto* vs_obj = detail::find(root.via.map, "variant_specs");
    if(!vs_obj || vs_obj->type != msgpack::type::MAP)
        return result;

    const auto& vs_map = vs_obj->via.map;
    for(uint32_t i = 0; i < vs_map.size; ++i)
    {
        const auto& kv = vs_map.ptr[i];
        if(kv.key.type != msgpack::type::STR || kv.val.type != msgpack::type::MAP)
            continue;

        VariantInfo vi;
        vi.name = std::string(kv.key.via.str.ptr, kv.key.via.str.size);

        const auto& entry_map = kv.val.via.map;
        vi.spec_type          = detail::getString(entry_map, "spec_type");

        // Parse targets into TargetSet
        const auto* targets_obj = detail::find(entry_map, "targets");
        if(targets_obj && targets_obj->type == msgpack::type::ARRAY)
        {
            for(uint32_t t = 0; t < targets_obj->via.array.size; ++t)
            {
                const auto& tgt = targets_obj->via.array.ptr[t];
                if(tgt.type == msgpack::type::STR)
                {
                    std::string arch(tgt.via.str.ptr, tgt.via.str.size);
                    vi.targets = vi.targets | TargetSet{parseGpuTarget(arch)};
                }
            }
        }

        // Deserialize spec
        const auto* spec_obj = detail::find(entry_map, "spec");
        if(spec_obj)
        {
            if(vi.spec_type == "GemmSpec")
                vi.gemm_spec = gemm_spec_from_msgpack(*spec_obj);
            else if(vi.spec_type == "ElementwiseSpec")
                vi.elementwise_spec = elementwise_spec_from_msgpack(*spec_obj);
        }

        result.push_back(std::move(vi));
    }

    return result;
}

} // namespace rocm_ck
