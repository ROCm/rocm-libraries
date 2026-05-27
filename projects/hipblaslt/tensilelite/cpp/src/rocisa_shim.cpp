// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Minimal re-implementations of rocisa C++ functions needed by _tl_emit.
// These are small leaf functions whose definitions live in rocisa's .cpp
// files but are not exported from the _rocisa Python extension module.
// When _tl_emit eventually links against a shared rocisa core library,
// this file can be removed.

#include "code.hpp"
#include "container.hpp"
#include "helper.hpp"
#include "instruction/mfma.hpp"

#include <cstdlib>
#include <cxxabi.h>

std::string demangle(const char* name)
{
    int   status    = -1;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    std::string result = (status == 0) ? demangled : name;
    free(demangled);
    return result;
}

namespace rocisa
{
    std::shared_ptr<RegisterContainer> vgpr(int idx, float regNum)
    {
        return std::make_shared<RegisterContainer>("v", std::nullopt, idx, regNum);
    }

    std::shared_ptr<RegisterContainer> accvgpr(int idx, float regNum)
    {
        return std::make_shared<RegisterContainer>("acc", std::nullopt, idx, regNum);
    }

    std::vector<std::shared_ptr<Item>>
        cloneItemList(const std::vector<std::shared_ptr<Item>>& itemList)
    {
        std::vector<std::shared_ptr<Item>> cloned;
        cloned.reserve(itemList.size());
        for(const auto& item : itemList)
        {
            if(typeid(item.get()) == typeid(Module))
                cloned.push_back(std::make_shared<Module>(*dynamic_cast<Module*>(item.get())));
            else
                cloned.push_back(item->clone());
        }
        return cloned;
    }

    DataType instTypeToDataType(InstType instType)
    {
        switch(instType)
        {
        case InstType::INST_F16:
            return DataType::Half;
        case InstType::INST_F32:
            return DataType::Float;
        case InstType::INST_F64:
            return DataType::Double;
        case InstType::INST_BF16:
            return DataType::BFloat16;
        case InstType::INST_I8:
        case InstType::INST_U8:
            return DataType::Int8;
        case InstType::INST_I32:
            return DataType::Int32;
        case InstType::INST_XF32:
            return DataType::XFloat32;
        case InstType::INST_F8:
            return DataType::Float8;
        case InstType::INST_BF8:
            return DataType::BFloat8;
        case InstType::INST_F8_BF8:
            return DataType::Float8BFloat8;
        case InstType::INST_BF8_F8:
            return DataType::BFloat8Float8;
        case InstType::INST_F6:
        case InstType::INST_F6_B6:
            return DataType::Float6;
        case InstType::INST_BF6:
        case InstType::INST_B6_F6:
            return DataType::BFloat6;
        case InstType::INST_F4:
            return DataType::Float4;
        case InstType::INST_F8_F4:
        case InstType::INST_F4_F8:
        case InstType::INST_F6_F4:
        case InstType::INST_F4_F6:
        case InstType::INST_F8_F6:
        case InstType::INST_F6_F8:
        case InstType::INST_F8_B6:
        case InstType::INST_B6_F8:
            return DataType::Float8;
        case InstType::INST_B8_F4:
        case InstType::INST_F4_B8:
        case InstType::INST_B6_F4:
        case InstType::INST_F4_B6:
        case InstType::INST_B8_F6:
        case InstType::INST_F6_B8:
        case InstType::INST_B8_B6:
        case InstType::INST_B6_B8:
            return DataType::BFloat8;
        default:
            throw std::runtime_error("Unknown instruction type");
        }
    }

    bool is8bitFloat(DataType value)
    {
        switch(value)
        {
        case DataType::Float8:
        case DataType::BFloat8:
        case DataType::Float8BFloat8:
        case DataType::BFloat8Float8:
        case DataType::Float8_fnuz:
        case DataType::BFloat8_fnuz:
        case DataType::Float8BFloat8_fnuz:
        case DataType::BFloat8Float8_fnuz:
            return true;
        default:
            return false;
        }
    }
} // namespace rocisa
