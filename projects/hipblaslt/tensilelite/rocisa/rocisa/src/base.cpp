/* ************************************************************************
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#include "base.hpp"

namespace rocisa
{
    namespace
    {
        ItemTypeNameResolver itemTypeNameResolver = nullptr;
    }

    void setItemTypeNameResolver(ItemTypeNameResolver resolver)
    {
        itemTypeNameResolver = resolver;
    }

    std::string Item::typeName() const
    {
        if(itemTypeNameResolver)
            return itemTypeNameResolver(*this);
        return demangle(typeid(*this).name());
    }

    std::string isaToGfx(const IsaVersion& arch)
    {
        return getGfxNameTuple(arch);
    }

    std::string getGlcBitName()
    {
        auto hasGLCModifier = capOrDefault(rocIsa::getInstance().getAsmCaps(), "HasGLCModifier");
        auto hasSC0Modifier = capOrDefault(rocIsa::getInstance().getAsmCaps(), "HasSC0Modifier");
        if(hasGLCModifier)
            return "glc";
        if(hasSC0Modifier)
            return "sc0";
        return "";
    }

    std::string getSlcBitName()
    {
        auto hasGLCModifier = capOrDefault(rocIsa::getInstance().getAsmCaps(), "HasGLCModifier");
        auto hasSC0Modifier = capOrDefault(rocIsa::getInstance().getAsmCaps(), "HasSC0Modifier");
        if(hasGLCModifier)
            return "slc";
        if(hasSC0Modifier)
            return "sc1";
        return "";
    }

    // Force init the instance
    auto isInit = rocIsa::getInstance().isInit();

}
