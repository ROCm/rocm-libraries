/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace Tensile
{
    /**
 * \ingroup Tensile
 * \defgroup KernelLanguages Kernel language type Info
 *
 * @brief Definitions and metadata on supported kernel language types.
 */

    /**
 * \ingroup KernelLanguages
 * @{
 */

    /**
 * Kernel Language
 */
    enum class KernelLanguage : int
    {
        Any,
        Assembly,
        Source,
        Count
    };

    std::string   ToString(KernelLanguage d);
    std::string   TypeAbbrev(KernelLanguage d);
    std::ostream& operator<<(std::ostream& stream, KernelLanguage const& t);
    std::istream& operator>>(std::istream& stream, KernelLanguage& t);

    /**
 * \ingroup KernelLanguages
 * \brief Runtime accessible kernel language type metadata
 */
    struct KernelLanguageTypeInfo
    {
        static KernelLanguageTypeInfo const& Get(int index);
        static KernelLanguageTypeInfo const& Get(KernelLanguage t);
        static KernelLanguageTypeInfo const& Get(std::string const& str);

        KernelLanguage m_kernelLanguage;
        std::string    name;
        std::string    abbrev;

    private:
        static void registerAllTypeInfo();
        static void registerAllTypeInfoOnce();

        template <KernelLanguage T_Enum>
        static void registerTypeInfo();

        static void addInfoObject(KernelLanguageTypeInfo const& info);

        static std::map<KernelLanguage, KernelLanguageTypeInfo> data;
        static std::map<std::string, KernelLanguage>            typeNames;
    };

    /**
 * \ingroup KernelLanguages
 * \brief Compile-time accessible kernel language type metadata.
 */
    template <KernelLanguage T_Enum>
    struct KernelLanguageInfo;

    template <KernelLanguage T_Enum>
    struct BaseKernelLanguageInfo
    {
        constexpr static KernelLanguage Enum = T_Enum;

        static inline std::string Name()
        {
            return ToString(Enum);
        }
        static inline std::string Abbrev()
        {
            return TypeAbbrev(Enum);
        }
    };

    template <KernelLanguage T_Enum>
    constexpr KernelLanguage BaseKernelLanguageInfo<T_Enum>::Enum;

    template <>
    struct KernelLanguageInfo<KernelLanguage::Any>
        : public BaseKernelLanguageInfo<KernelLanguage::Any>
    {
    };
    template <>
    struct KernelLanguageInfo<KernelLanguage::Assembly>
        : public BaseKernelLanguageInfo<KernelLanguage::Assembly>
    {
    };
    template <>
    struct KernelLanguageInfo<KernelLanguage::Source>
        : public BaseKernelLanguageInfo<KernelLanguage::Source>
    {
    };

    /**
 * @}
 */
} // namespace Tensile

namespace std
{
    template <>
    struct hash<Tensile::KernelLanguage>
    {
        inline size_t operator()(Tensile::KernelLanguage const& val) const
        {
            return hash<int>()(static_cast<int>(val));
        }
    };
} // namespace std
