/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <memory>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <Tensile/Debug.hpp>
#include <Tensile/Macros.hpp>
#include <Tensile/Singleton.hpp>

namespace TensileLite
{
    /**
 * \ingroup Tensile
 * \defgroup Embedding Data Embedding
 *
 * @brief Mechanism which allows binary data to be stored in an executable
 * or shared library as an array of bytes.
 */

    template <typename Object>
    struct EmbedData;

    /**
 * \ingroup Embedding
 *
 * @brief EmbeddedData is the mechanism for retrieving data which has been
 * registered with EmbedData.
 */
    template <typename Object>
    class TENSILE_API EmbeddedData : public LazySingleton<EmbeddedData<Object>>
    {
    public:
        using Base = LazySingleton<EmbeddedData<Object>>;

        using Item  = std::vector<uint8_t>;
        using Items = std::vector<Item>;
        using Map   = std::unordered_map<std::string, Items>;

        static Items const& Get()
        {
            return Get("");
        }

        static Items const& Get(std::string const& key)
        {
            auto const& items = Base::Instance().items;
            auto        iter  = items.find(key);
            if(iter == items.end())
                return Base::Instance().empty;

            return iter->second;
        }

    protected:
        friend Base;
        friend struct EmbedData<Object>;

        static Items& GetMutable()
        {
            return GetMutable("");
        }

        static Items& GetMutable(std::string const& key)
        {
            if(Debug::Instance().printDataInit())
                std::cout << "Embedding an object of type " << typeid(Object).name() << " with key "
                          << key << std::endl;
            return Base::Instance().items[key];
        }

        EmbeddedData() = default;

        Map         items;
        const Items empty = Items{};
    };

    /**
 * \ingroup Embedding
 *
 * @brief Object which registers embedded data when it's instantiated.
 */
    template <typename Object>
    struct TENSILE_API EmbedData
    {
        EmbedData(std::initializer_list<uint8_t> data)
        {
            EmbeddedData<Object>::GetMutable().emplace_back(data);
        }

        EmbedData(std::vector<uint8_t> const& data)
        {
            EmbeddedData<Object>::GetMutable().push_back(data);
        }

        EmbedData(std::string const& key, std::initializer_list<uint8_t> data)
        {
            EmbeddedData<Object>::GetMutable(key).emplace_back(data);
        }

        EmbedData(std::string const& key, std::vector<uint8_t> const& data)
        {
            EmbeddedData<Object>::GetMutable(key).push_back(data);
        }
    };

} // namespace TensileLite

#define TENSILE_CONCATENATE_SYMBOLS(a, b) TENSILE_CONCATENATE_SYMBOLS1(a, b)
#define TENSILE_CONCATENATE_SYMBOLS1(a, b) TENSILE_CONCATENATE_SYMBOLS2(a, b)
#define TENSILE_CONCATENATE_SYMBOLS2(a, b) a##b

#define TENSILE_EMBED_SYMBOL_NAME TENSILE_CONCATENATE_SYMBOLS(TensileEmbeddedData, __LINE__)
