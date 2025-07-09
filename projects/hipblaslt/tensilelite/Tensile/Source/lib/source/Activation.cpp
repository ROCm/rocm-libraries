/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2024 Advanced Micro Devices, Inc.
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

#include <Tensile/Activation.hpp>
#include <Tensile/Utils.hpp>

namespace TensileLite
{
    std::string ToString(ActivationType d)
    {
        switch(d)
        {
        case ActivationType::Abs:
            return "Abs";
        case ActivationType::Clippedrelu:
            return "Clippedrelu";
        case ActivationType::Exp:
            return "Exp";
        case ActivationType::Gelu:
            return "Gelu";
        case ActivationType::Geluscaling:
            return "Geluscaling";
        case ActivationType::Leakyrelu:
            return "Leakyrelu";
        case ActivationType::Relu:
            return "Relu";
        case ActivationType::Sigmoid:
            return "Sigmoid";
        case ActivationType::Tanh:
            return "Tanh";
        case ActivationType::DGelu:
            return "Dgelu";
        case ActivationType::Silu:
            return "Silu";
        case ActivationType::Swish:
            return "Swish";
        case ActivationType::Clamp:
            return "Clamp";
        case ActivationType::All:
            return "All";
        case ActivationType::Hipblaslt_all:
            return "Hipblaslt_all";
        case ActivationType::None:
            return "None";
        default:
            return "Invalid";
        }
        return "Invalid";
    }
    std::ostream& operator<<(std::ostream& stream, const ActivationType& t)
    {
        return stream << ToString(t);
    }

    std::istream& operator>>(std::istream& stream, ActivationType& t)
    {
        std::string strValue;
        stream >> strValue;
        if(strValue == ToString(ActivationType::Abs))
        {
            t = ActivationType::Abs;
        }
        else if(strValue == ToString(ActivationType::Clippedrelu))
        {
            t = ActivationType::Clippedrelu;
        }
        else if(strValue == ToString(ActivationType::Exp))
        {
            t = ActivationType::Exp;
        }
        else if(strValue == ToString(ActivationType::Gelu))
        {
            t = ActivationType::Gelu;
        }
        else if(strValue == ToString(ActivationType::Geluscaling))
        {
            t = ActivationType::Geluscaling;
        }
        else if(strValue == ToString(ActivationType::Leakyrelu))
        {
            t = ActivationType::Leakyrelu;
        }
        else if(strValue == ToString(ActivationType::Relu))
        {
            t = ActivationType::Relu;
        }
        else if(strValue == ToString(ActivationType::Sigmoid))
        {
            t = ActivationType::Sigmoid;
        }
        else if(strValue == ToString(ActivationType::Tanh))
        {
            t = ActivationType::Tanh;
        }
        else if(strValue == ToString(ActivationType::DGelu))
        {
            t = ActivationType::DGelu;
        }
        else if(strValue == ToString(ActivationType::Silu))
        {
            t = ActivationType::Silu;
        }
        else if(strValue == ToString(ActivationType::Swish))
        {
            t = ActivationType::Swish;
        }
        else if(strValue == ToString(ActivationType::Clamp))
        {
            t = ActivationType::Clamp;
        }
        else if(strValue == ToString(ActivationType::All))
        {
            t = ActivationType::All;
        }
        else if(strValue == ToString(ActivationType::Hipblaslt_all))
        {
            t = ActivationType::Hipblaslt_all;
        }
        else if(strValue == ToString(ActivationType::None))
        {
            t = ActivationType::None;
        }
        else
        {
            throw std::runtime_error(concatenate("Invalid data type: ", strValue));
        }
        return stream;
    }

    int getAdditionalArgNum(ActivationType d)
    {
        std::map<ActivationType, int> argMap;
        argMap.insert(std::pair<ActivationType, int>(ActivationType::Clippedrelu, 2));
        argMap.insert(std::pair<ActivationType, int>(ActivationType::Leakyrelu, 1));
        argMap.insert(std::pair<ActivationType, int>(ActivationType::Tanh, 2));
        argMap.insert(std::pair<ActivationType, int>(ActivationType::Geluscaling, 1));
        if(d == ActivationType::All)
        {
            int maxArgs = 0;
            for(auto iter = argMap.begin(); iter != argMap.end(); iter++)
                maxArgs = std::max(maxArgs, iter->second);
            return maxArgs;
        }
        auto iter = argMap.find(d);
        if(iter != argMap.end())
            return iter->second;

        return 0;
    }
}
