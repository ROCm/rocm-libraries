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
#pragma once
#include "base.hpp"
#include "container.hpp"
#include "format.hpp"
#include "instruction/instruction.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

namespace rocisa
{
    struct Label : public Item
    {
        std::variant<std::string, int> label;
        std::string                    comment;
        int                            alignment;

        Label(int label, const std::string& comment, int alignment = 1)
            : Item("")
            , label(label)
            , comment(comment)
            , alignment(alignment)
        {
        }

        Label(const std::string& label, const std::string& comment, int alignment = 1)
            : Item("")
            , label(label)
            , comment(comment)
            , alignment(alignment)
        {
        }

        Label(const std::variant<std::string, int>& label,
              const std::string&                    comment,
              int                                   alignment = 1)
            : Item("")
            , label(label)
            , comment(comment)
            , alignment(alignment)
        {
        }

        Label(const Label& other)
            : Item(other)
            , label(other.label)
            , comment(other.comment)
            , alignment(other.alignment)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<Label>(*this);
        }

        static std::string getFormatting(const std::variant<std::string, int>& label)
        {
            auto labelStr = std::visit(
                [](auto&& arg) -> std::string {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr(std::is_same_v<T, int>)
                    {
                        return std::to_string(arg);
                    }
                    else if constexpr(std::is_same_v<T, std::string>)
                    {
                        return arg;
                    }
                },
                label);
            return "label_" + labelStr;
        }

        std::string getLabelName() const
        {
            return getFormatting(label);
        }

        std::string toString() const override
        {
            std::string t = getLabelName() + ":";
            if(alignment > 1)
            {
                t = ".align " + std::to_string(alignment) + "\n" + t;
            }
            if(!comment.empty() && !rocIsa::getInstance().getOutputOptions().outputNoComment)
            {
                t += "  /// " + comment;
            }
            t += "\n";
            return t;
        }
    };

    /*
    An unstructured block of text
    */
    struct TextBlock : public Item
    {
        std::string text;

        TextBlock(const std::string& text)
            : Item(text)
            , text(text)
        {
        }

        TextBlock(const TextBlock& other)
            : Item(other)
            , text(other.text)
        {
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<TextBlock>(*this);
        }

        std::string toString() const override
        {
            if(!rocIsa::getInstance().getOutputOptions().outputNoComment)
                return text;
            return "";
        }
    };

    std::vector<std::shared_ptr<Item>>
        cloneItemList(const std::vector<std::shared_ptr<Item>>& itemList);

    struct Module : public Item
    {
        std::vector<std::shared_ptr<Item>> itemList;
        std::shared_ptr<Container>         tempVgpr = nullptr;
        bool                               _isNoOpt;

        Module(const std::string& name = "")
            : Item(name)
            , _isNoOpt(false)
        {
        }

        Module(const Module& other)
            : Item(other)
            , tempVgpr(other.tempVgpr ? other.tempVgpr->clone() : nullptr)
            , _isNoOpt(other._isNoOpt)
        {
            itemList = cloneItemList(other.itemList);
            for(auto& item : itemList)
            {
                item->parent = this;
            }
        }

        std::shared_ptr<Item> clone() const override
        {
            return std::make_shared<Module>(*this);
        }

        void setParent()
        {
            for(auto& item : itemList)
            {
                item->parent = this;
                if(auto module = dynamic_cast<Module*>(item.get()))
                {
                    module->setParent();
                }
            }
        }

        void setNoOpt(bool noOpt)
        {
            _isNoOpt = noOpt;
        }

        bool isNoOpt() const
        {
            return _isNoOpt;
        }

        std::shared_ptr<Item> findNamedItem(const std::string& targetName) const
        {
            auto it = std::find_if(
                itemList.begin(), itemList.end(), [&targetName](const std::shared_ptr<Item>& item) {
                    return item->name == targetName;
                });
            return (it != itemList.end()) ? *it : nullptr;
        }

        void setInlineAsmPrintMode(bool mode)
        {
            for(auto& item : itemList)
            {
                if(auto module = dynamic_cast<Module*>(item.get()))
                {
                    module->setInlineAsmPrintMode(mode);
                }
                else if(auto instruction = dynamic_cast<Instruction*>(item.get()))
                {
                    instruction->setInlineAsm(mode);
                }
            }
        }

        std::string toString() const override
        {
            std::string prefix = 0 ? "// " + name + "{\n" : "";
            std::string suffix = 0 ? "// } " + name + "\n" : "";
            std::string s;
            for(const auto& item : itemList)
            {
                s += item->toString();
            }
            return prefix + s + suffix;
        }

        void addSpaceLine()
        {
            itemList.push_back(std::make_shared<TextBlock>("\n"));
        }

        const std::shared_ptr<Item> add(const std::shared_ptr<Item>& item, int pos = -1)
        {
            /*
                Add specified item to the list of items in the module.
                Item MUST be a Item (not a string) - can use
                addText(...)) to add a string.
                All additions to itemList should use this function.

                Returns item to facilitate one-line create/add patterns
            */
            if(item)
            {
                item->parent = this;
                if(pos == -1)
                {
                    itemList.push_back(item);
                }
                else
                {
                    itemList.insert(itemList.begin() + pos, item);
                }
            }
            return item;
        }

        template <typename T, typename... Args>
        const std::shared_ptr<Item> addT(Args&&... args)
        {
            auto item = std::make_shared<T>(std::forward<Args>(args)...);
            return add(item);
        }

        const void addItems(const std::vector<std::shared_ptr<Item>>& items)
        {
            for(const auto& item : items)
            {
                add(item);
            }
        }

        const std::shared_ptr<Module> appendModule(const std::shared_ptr<Module>& module)
        {
            /*
                Append items to module.
            */
            for(const auto& i : module->items())
            {
                add(i);
            }
            return module;
        }

        const std::shared_ptr<Module> addModuleAsFlatItems(const std::shared_ptr<Module>& module)
        {
            /*
                Add items to module.

                Returns items to facilitate one-line create/add patterns
            */
            for(const auto& i : module->flatitems())
            {
                add(i);
            }
            return module;
        }

        int findIndex(const std::shared_ptr<Item>& targetItem) const
        {
            auto it = std::find(itemList.begin(), itemList.end(), targetItem);
            return (it != itemList.end()) ? std::distance(itemList.begin(), it) : -1;
        }

        int findIndexByType(nb::object obj) const
        {
            for(size_t i = 0; i < itemList.size(); ++i)
            {
                nb::object itemobj = nb::cast(*itemList[i]);
                if(nb::isinstance(itemobj, obj))
                {
                    return i;
                }
            }
            return -1;
        }

        void addComment(const std::string& comment)
        {
            /*
                Convenience function to format arg as a comment and add TextBlock item
                This comment is a single line // MYCOMMENT
            */
            add(std::make_shared<TextBlock>(slash(comment)));
        }

        void addCommentAlign(const std::string& comment)
        {
            /*
                Convenience function to format arg as a comment and add TextBlock item
                This comment is a single line // MYCOMMENT with the same format as
                Instruction.
            */
            add(std::make_shared<TextBlock>(slash50(comment)));
        }

        void addComment0(const std::string& comment)
        {
            /*
                Convenience function to format arg as a comment and add TextBlock item
                This comment is a single line comment block
            */
            add(std::make_shared<TextBlock>(block(comment)));
        }

        void addComment1(const std::string& comment)
        {
            /*
                Convenience function to format arg as a comment and add TextBlock item
                This comment is a blank line followed by comment block
            */
            add(std::make_shared<TextBlock>(blockNewLine(comment)));
        }

        void addComment2(const std::string& comment)
        {
            add(std::make_shared<TextBlock>(block3Line(comment)));
        }

        /*
        Test code:
          mod1 = Code.Module("TopModule")
          mod2 = Code.Module("Module-lvl2")
          mod2.add(Code.Inst("bogusInst", "comments"))
          mod3 = Code.Module("Module-lvl3")
          mod3.add(Code.TextBlock("bogusTextBlock\nbogusTextBlock2\nbogusTextBlock3"))
          mod3.add(Code.GlobalReadInst("bogusGlobalReadInst", "comments"))
          mod2.add(mod3)
          mod1.add(mod2)
          mod1.add(Code.Inst("bogusInst", "comments"))
          mod1.add(mod2)

          print(mod1.prettyPrint())
        Output:
          Module "TopModule"
          |--Module "Module-lvl2"
          | |--Inst bogusInst                                          // comments
          | |--Module "Module-lvl3"
          | | |--TextBlock
          | | | |--bogusTextBlock
          | | | |--bogusTextBlock2
          | | | |--bogusTextBlock3
          | | |--GlobalReadInst bogusGlobalReadInst                                // comments
          |--Inst bogusInst                                          // comments
          |--Module "Module-lvl2"
          | |--Inst bogusInst                                          // comments
          | |--Module "Module-lvl3"
          | | |--TextBlock
          | | | |--bogusTextBlock
          | | | |--bogusTextBlock2
          | | | |--bogusTextBlock3
          | | |--GlobalReadInst bogusGlobalReadInst
        */
        std::string prettyPrint(const std::string& indent = "") const override
        {
            std::string ostream;
            ostream += indent + demangle(typeid(*this).name()) + " \"" + name + "\"\n";
            for(const auto& i : itemList)
            {
                ostream += i->prettyPrint(indent + "|--");
            }
            return ostream;
        }

        int countType(const nb::object& obj) const override
        {
            int count = 0;
            for(const auto& i : itemList)
            {
                count += i->countType(obj);
            }
            return count;
        }

        int countExactType(const std::type_info& targetType) const override
        {
            int count = static_cast<int>(typeid(*this) == targetType);
            for(const auto& i : itemList)
            {
                count += i->countExactType(targetType);
            }
            return count;
        }

        int count() const
        {
            int count = 0;
            for(const auto& i : itemList)
            {
                if(auto module = dynamic_cast<Module*>(i.get()))
                {
                    count += module->count();
                }
                else
                {
                    count += 1;
                }
            }
            return count;
        }

        std::shared_ptr<Item> getItem(const int index)
        {
            if(index >= itemList.size())
            {
                throw std::runtime_error("index out of range");
            }
            return itemList[index];
        }

        void setItem(const int index, const std::shared_ptr<Item>& items)
        {
            if(index >= itemList.size())
            {
                throw std::runtime_error("index out of range");
            }
            itemList[index] = items;
        }

        void setItems(const std::vector<std::shared_ptr<Item>>& items)
        {
            itemList = items;
        }

        const std::vector<std::shared_ptr<Item>>& items() const
        {
            return itemList;
        }

        const size_t itemsSize() const
        {
            return itemList.size();
        }

        /*
        Replace item from itemList.
        Items may be other Modules, TexBlock, or Inst
        */
        void replaceItem(const std::shared_ptr<Item>& srcItem, const std::shared_ptr<Item>& dstItem)
        {
            for(size_t index = 0; index < itemList.size(); ++index)
            {
                if(itemList[index] == srcItem)
                {
                    dstItem->parent = this;
                    itemList[index] = dstItem;
                    break;
                }
            }
        }

        /*
        Replace item from itemList, do nothing if
        exceed length of the itemList
        Items may be other Modules, TexBlock, or Inst
        */
        void replaceItemByIndex(size_t index, const std::shared_ptr<Item>& item)
        {
            if(index >= itemList.size())
            {
                return;
            }
            item->parent    = this;
            itemList[index] = item;
        }

        /*
        Remove item from itemList, remove the last element if
        exceed length of the itemList
        Items may be other Modules, TexBlock, or Inst
        */
        void removeItemByIndex(size_t index)
        {
            if(index >= itemList.size())
            {
                index = itemList.size() - 1;
            }
            itemList.erase(itemList.begin() + index);
        }

        void removeItem(const std::shared_ptr<Item>& item)
        {
            itemList.erase(std::remove(itemList.begin(), itemList.end(), item), itemList.end());
        }

        /*
        Remove items from itemList
        Items may be other Modules, TexBlock, or Inst
        */
        void removeItemsByName(const std::string& name)
        {
            itemList.erase(std::remove_if(itemList.begin(),
                                          itemList.end(),
                                          [&name](const std::shared_ptr<Item>& item) {
                                              return item->name == name;
                                          }),
                           itemList.end());
        }

        std::shared_ptr<Item> popFirstItem()
        {
            if(itemList.empty())
            {
                return nullptr;
            }
            auto item = itemList.front();
            itemList.erase(itemList.begin());
            return item;
        }

        std::vector<std::shared_ptr<Item>> popFirstNItems(size_t n)
        {
            std::vector<std::shared_ptr<Item>> items;
            if(n >= itemList.size())
            {
                items = std::move(itemList);
                itemList.clear();
            }
            else
            {
                items.insert(items.end(), itemList.begin(), itemList.begin() + n);
                itemList.erase(itemList.begin(), itemList.begin() + n);
            }
            return items;
        }

        std::vector<std::shared_ptr<Item>> flatitems() const
        {
            std::vector<std::shared_ptr<Item>> flatitems;
            for(const auto& i : itemList)
            {
                if(auto module = dynamic_cast<Module*>(i.get()))
                {
                    auto subitems = module->flatitems();
                    flatitems.insert(flatitems.end(), subitems.begin(), subitems.end());
                }
                else
                {
                    flatitems.push_back(i);
                }
            }
            return flatitems;
        }

        void addTempVgpr(const std::shared_ptr<Container>& vgpr)
        {
            tempVgpr = vgpr;
        }
    };

    struct Macro : public Item
    {
        std::vector<std::shared_ptr<Item>> itemList;
        std::shared_ptr<MacroInstruction>  macro;

        Macro(const std::string& name, const std::vector<InstructionInput>& args)
            : Item(name)
        {
            macro = std::make_shared<MacroInstruction>(name, args);
        }

        Macro(const Macro& other)
            : Item(other)
            , macro(other.macro ? std::dynamic_pointer_cast<MacroInstruction>(other.macro->clone())
                                : nullptr)
        {
            itemList = cloneItemList(other.itemList);
            for(auto& item : itemList)
            {
                item->parent = this;
            }
        }

        std::shared_ptr<Item> add(const std::shared_ptr<Item>& item)
        {
            // This is a workaround
            if(dynamic_cast<Instruction*>(item.get()) || dynamic_cast<Module*>(item.get())
               || dynamic_cast<TextBlock*>(item.get()))
            {
                item->parent = this;
                itemList.push_back(item);
            }
            else
            {
                throw std::runtime_error("unknown item type for Macro.add: " + item->toString());
            }
            return item;
        }

        template <typename T, typename... Args>
        const std::shared_ptr<Item> addT(Args&&... args)
        {
            auto item = std::make_shared<T>(std::forward<Args>(args)...);
            return add(item);
        }

        void addComment0(const std::string& comment)
        {
            add(std::make_shared<TextBlock>("/* " + comment + " */\n"));
        }

        void setItems(const std::vector<std::shared_ptr<Item>>& items)
        {
            itemList = items;
        }

        const std::vector<std::shared_ptr<Item>>& items() const
        {
            return itemList;
        }

        std::string prettyPrint(const std::string& indent = "") const override
        {
            std::string ostream;
            ostream += indent + typeid(*this).name() + " \"" + name + "\"\n";
            for(const auto& i : itemList)
            {
                ostream += i->prettyPrint(indent + "|--");
            }
            return ostream;
        }

        std::string toString() const override
        {
            std::string s;
            if(0)
            {
                s += "// " + name + " { \n";
            }
            s += ".macro " + macro->toString();
            for(const auto& x : itemList)
            {
                s += "    " + x->toString();
            }
            s += ".endm\n";
            if(0)
            {
                s += "// } " + name + "\n";
            }
            return s;
        }
    };

    struct StructuredModule : public Module
    {
        std::shared_ptr<Module> header;
        std::shared_ptr<Module> middle;
        std::shared_ptr<Module> footer;

        StructuredModule(const std::string& name = "")
            : Module(name)
        {
            header = std::make_shared<Module>("header");
            middle = std::make_shared<Module>("middle");
            footer = std::make_shared<Module>("footer");

            add(header);
            add(middle);
            add(footer);
        }

        StructuredModule(const StructuredModule& other)
            : Module(other)
            , header(other.header ? std::dynamic_pointer_cast<Module>(other.header->clone())
                                  : nullptr)
            , middle(other.middle ? std::dynamic_pointer_cast<Module>(other.middle->clone())
                                  : nullptr)
            , footer(other.footer ? std::dynamic_pointer_cast<Module>(other.footer->clone())
                                  : nullptr)
        {
        }
    };

    struct ValueEndif : public Item
    {
        std::string comment;

        ValueEndif(const std::string& comment = "")
            : Item("ValueEndif")
            , comment(comment)
        {
        }

        std::string toString() const override
        {
            return formatStr(
                false, ".endif", comment, rocIsa::getInstance().getOutputOptions().outputNoComment);
        }
    };

    struct ValueIf : public Item
    {
        int value;

        ValueIf(int value)
            : Item("ValueIf")
            , value(value)
        {
        }

        std::string toString() const override
        {
            return ".if " + std::to_string(value);
        }
    };

    struct ValueSet : public Item
    {
        std::optional<std::string> ref;
        std::optional<int64_t>     value;
        int                        offset;
        int                        format;

        ValueSet(const std::string& name, int value, int offset = 0, int format = 0)
            : Item(name)
            , ref(std::nullopt)
            , value(value)
            , offset(offset)
            , format(format)
        {
        }

        ValueSet(const std::string& name, uint32_t value, int offset = 0, int format = 0)
            : Item(name)
            , ref(std::nullopt)
            , value(value)
            , offset(offset)
            , format(format)
        {
        }

        ValueSet(const std::string& name, const std::string& ref, int offset = 0, int format = 0)
            : Item(name)
            , ref(ref)
            , value(std::nullopt)
            , offset(offset)
            , format(format)
        {
        }

        std::string toString() const override
        {
            std::string t = ".set " + name + ", ";
            if(ref)
            {
                if(format == -1)
                {
                    t += ref.value();
                }
                else
                {
                    t += ref.value() + "+" + std::to_string(offset);
                }
            }
            else if(value)
            {
                if(format == -1)
                {
                    t += std::to_string(value.value());
                }
                else if(format == 0)
                {
                    t += std::to_string(value.value() + offset);
                }
                else if(format == 1)
                {
                    t += "0x" + toHex(value.value() + offset);
                }
            }
            t += "\n";
            return t;
        }

    private:
        std::string toHex(int num) const
        {
            std::stringstream ss;
            ss << std::hex << num;
            return ss.str();
        }
    };

    struct RegSet : public ValueSet
    {
        std::string regType;

        RegSet(const std::string& regType, const std::string& name, int value, int offset = 0)
            : ValueSet(name, value, offset)
            , regType(regType)
        {
        }

        RegSet(const std::string& regType,
               const std::string& name,
               const std::string& value,
               int                offset = 0)
            : ValueSet(name, value, offset)
            , regType(regType)
        {
        }
    };

    inline std::string field_desc(const std::string& field_name, int value, int bits = 0)
    {
        std::string bits_str = (bits > 0) ? " (" + std::to_string(bits) + "b)" : "";
        return field_name + bits_str + ": " + std::to_string(value);
    }

    struct BitfieldUnion
    {
        virtual std::string toString() const
        {
            return "0x" + toHex(value);
        }

        virtual int getValue() const
        {
            return value;
        }

        virtual std::string fields_desc() const = 0;
        virtual std::string desc() const        = 0;

    protected:
        int value;

        std::string toHex(int num) const
        {
            std::stringstream ss;
            ss << std::hex << num;
            return ss.str();
        }
    };

    union SrdUpperFields9XX
    {
        struct
        {
            uint32_t dst_sel_x : 3;
            uint32_t dst_sel_y : 3;
            uint32_t dst_sel_z : 3;
            uint32_t dst_sel_w : 3;
            uint32_t num_format : 3;
            uint32_t data_format : 4;
            uint32_t user_vm_enable : 1;
            uint32_t user_vm_mode : 1;
            uint32_t index_stride : 2;
            uint32_t add_tid_enable : 1;
            uint32_t _unusedA : 3;
            uint32_t nv : 1;
            uint32_t _unusedB : 2;
            uint32_t type : 2;
        };
        unsigned int value;

        SrdUpperFields9XX()
            : value(0)
        {
        }
    };

    struct SrdUpperValue9XX : public BitfieldUnion
    {
        SrdUpperFields9XX fields;

        static SrdUpperValue9XX staticInit()
        {
            SrdUpperValue9XX value;
            value.fields.data_format = 4;
            value.value              = value.fields.value;
            return value;
        }

        std::string fields_desc() const override
        {
            std::stringstream ss;
            ss << field_desc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
               << field_desc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
               << field_desc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
               << field_desc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
               << field_desc("num_format", fields.num_format, 3) << "\n"
               << field_desc("data_format", fields.data_format, 4) << "\n"
               << field_desc("user_vm_enable", fields.user_vm_enable, 1) << "\n"
               << field_desc("user_vm_mode", fields.user_vm_mode, 1) << "\n"
               << field_desc("index_stride", fields.index_stride, 2) << "\n"
               << field_desc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
               << field_desc("_unusedA", fields._unusedA, 3) << "\n"
               << field_desc("nv", fields.nv, 1) << "\n"
               << field_desc("_unusedB", fields._unusedB, 2) << "\n"
               << field_desc("type", fields.type, 2);
            return ss.str();
        }

        std::string desc() const override
        {
            return "hex: " + toString() + "\n" + fields_desc();
        }
    };

    union SrdUpperFields10XX
    {
        struct
        {
            uint32_t dst_sel_x : 3;
            uint32_t dst_sel_y : 3;
            uint32_t dst_sel_z : 3;
            uint32_t dst_sel_w : 3;
            uint32_t format : 7;
            uint32_t _unusedA : 2;
            uint32_t index_stride : 2;
            uint32_t add_tid_enable : 1;
            uint32_t resource_level : 1;
            uint32_t _unusedB : 1;
            uint32_t LLC_noalloc : 2;
            uint32_t oob_select : 2;
            uint32_t type : 2;
        };
        unsigned int value;

        SrdUpperFields10XX()
            : value(0)
        {
        }
    };

    struct SrdUpperValue10XX : public BitfieldUnion
    {
        SrdUpperFields10XX fields;

        static SrdUpperValue10XX staticInit()
        {
            SrdUpperValue10XX value;
            value.fields.format         = 4;
            value.fields.resource_level = 1;
            value.fields.oob_select     = 3;
            value.value                 = value.fields.value;
            return value;
        }

        std::string fields_desc() const override
        {
            std::stringstream ss;
            ss << field_desc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
               << field_desc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
               << field_desc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
               << field_desc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
               << field_desc("format", fields.format, 7) << "\n"
               << field_desc("_unusedA", fields._unusedA, 2) << "\n"
               << field_desc("index_stride", fields.index_stride, 2) << "\n"
               << field_desc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
               << field_desc("resource_level", fields.resource_level, 1) << "\n"
               << field_desc("_unusedB", fields._unusedB, 1) << "\n"
               << field_desc("LLC_noalloc", fields.LLC_noalloc, 2) << "\n"
               << field_desc("oob_select", fields.oob_select, 2) << "\n"
               << field_desc("type", fields.type, 2);
            return ss.str();
        }

        std::string desc() const override
        {
            return "hex: " + toString() + "\n" + fields_desc();
        }
    };

    union SrdUpperFields11XX
    {
        struct
        {
            uint32_t dst_sel_x : 3;
            uint32_t dst_sel_y : 3;
            uint32_t dst_sel_z : 3;
            uint32_t dst_sel_w : 3;
            uint32_t format : 7;
            uint32_t _unusedA : 2;
            uint32_t index_stride : 2;
            uint32_t add_tid_enable : 1;
            uint32_t resource_level : 1;
            uint32_t _unusedB : 1;
            uint32_t LLC_noalloc : 2;
            uint32_t oob_select : 2;
            uint32_t type : 2;
        };
        unsigned int value;

        SrdUpperFields11XX()
            : value(0)
        {
        }
    };

    struct SrdUpperValue11XX : public BitfieldUnion
    {
        SrdUpperFields11XX fields;

        static SrdUpperValue11XX staticInit()
        {
            SrdUpperValue11XX value;
            value.fields.format         = 4;
            value.fields.resource_level = 1;
            value.fields.oob_select     = 3;
            value.value                 = value.fields.value;
            return value;
        }

        std::string fields_desc() const override
        {
            std::stringstream ss;
            ss << field_desc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
               << field_desc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
               << field_desc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
               << field_desc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
               << field_desc("format", fields.format, 7) << "\n"
               << field_desc("_unusedA", fields._unusedA, 2) << "\n"
               << field_desc("index_stride", fields.index_stride, 2) << "\n"
               << field_desc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
               << field_desc("resource_level", fields.resource_level, 1) << "\n"
               << field_desc("_unusedB", fields._unusedB, 1) << "\n"
               << field_desc("LLC_noalloc", fields.LLC_noalloc, 2) << "\n"
               << field_desc("oob_select", fields.oob_select, 2) << "\n"
               << field_desc("type", fields.type, 2);
            return ss.str();
        }

        std::string desc() const override
        {
            return "hex: " + toString() + "\n" + fields_desc();
        }
    };

    union SrdUpperFields12XX
    {
        struct
        {
            uint32_t dst_sel_x : 3;
            uint32_t dst_sel_y : 3;
            uint32_t dst_sel_z : 3;
            uint32_t dst_sel_w : 3;
            uint32_t format : 7;
            uint32_t _unusedA : 2;
            uint32_t index_stride : 2;
            uint32_t add_tid_enable : 1;
            uint32_t resource_level : 1;
            uint32_t _unusedB : 3;
            uint32_t oob_select : 2;
            uint32_t type : 2;
        };
        unsigned int value;

        SrdUpperFields12XX()
            : value(0)
        {
        }
    };

    struct SrdUpperValue12XX : public BitfieldUnion
    {
        SrdUpperFields12XX fields;

        static SrdUpperValue12XX staticInit()
        {
            SrdUpperValue12XX value;
            value.fields.format     = 32;
            value.fields.oob_select = 3;
            value.value             = value.fields.value;
            return value;
        }

        std::string fields_desc() const override
        {
            std::stringstream ss;
            ss << field_desc("dst_sel_x", fields.dst_sel_x, 3) << "\n"
               << field_desc("dst_sel_y", fields.dst_sel_y, 3) << "\n"
               << field_desc("dst_sel_z", fields.dst_sel_z, 3) << "\n"
               << field_desc("dst_sel_w", fields.dst_sel_w, 3) << "\n"
               << field_desc("format", fields.format, 7) << "\n"
               << field_desc("_unusedA", fields._unusedA, 2) << "\n"
               << field_desc("index_stride", fields.index_stride, 2) << "\n"
               << field_desc("add_tid_enable", fields.add_tid_enable, 1) << "\n"
               << field_desc("resource_level", fields.resource_level, 1) << "\n"
               << field_desc("_unusedB", fields._unusedB, 3) << "\n"
               << field_desc("oob_select", fields.oob_select, 2) << "\n"
               << field_desc("type", fields.type, 2);
            return ss.str();
        }

        std::string desc() const override
        {
            return "hex: " + toString() + "\n" + fields_desc();
        }
    };

    std::shared_ptr<BitfieldUnion> SrdUpperValue(const IsaVersion& isa);

    /***************************************
     * Signatures
    ***************************************/

    struct SignatureArgument : public Item
    {
        static const std::unordered_map<std::string, int> ValueTypeSizeDict;

        SignatureValueKind valueKind;
        std::string        valueType;
        int                offset;
        int                size;
        std::string        addrSpaceQual;

        SignatureArgument(const int          offset,
                          const std::string& name,
                          SignatureValueKind valueKind,
                          const std::string& valueType,
                          const std::string& addrSpaceQual = "")
            : Item(name)
            , valueKind(valueKind)
            , valueType(valueType)
            , offset(offset)
            , size(valueToSize(valueKind, valueType))
            , addrSpaceQual(addrSpaceQual)
        {
        }

        int valueToSize(SignatureValueKind valueKind, const std::string& valueType) const
        {
            if(valueKind == SignatureValueKind::SIG_GLOBALBUFFER)
            {
                return 8;
            }

            auto it = ValueTypeSizeDict.find(valueType);
            if(it != ValueTypeSizeDict.end())
            {
                return it->second;
            }

            throw std::runtime_error("Unknown value type: " + valueType);
        }

        std::string valueKindToStr() const
        {
            if(valueKind == SignatureValueKind::SIG_GLOBALBUFFER)
            {
                return "global_buffer";
            }
            else if(valueKind == SignatureValueKind::SIG_VALUE)
            {
                return "by_value";
            }

            throw std::runtime_error("Unknown value kind");
        }

        std::string toString() const override
        {
            std::string signatureIndent = "        ";
            std::string kStr;
            kStr += signatureIndent.substr(2) + "- .name:            " + name + "\n";
            kStr += signatureIndent + ".size:            " + std::to_string(size) + "\n";
            kStr += signatureIndent + ".offset:          " + std::to_string(offset) + "\n";
            kStr += signatureIndent + ".value_kind:      " + valueKindToStr() + "\n";
            kStr += signatureIndent + ".value_type:      " + valueType + "\n";
            if(!addrSpaceQual.empty())
            {
                kStr += signatureIndent + ".address_space:   " + addrSpaceQual + "\n";
            }
            return kStr;
        }
    };

    struct SignatureKernelDescriptor : public Item
    {
        int                totalVgprs;
        int                totalAgprs;
        int                totalSgprs;
        int                originalTotalVgprs;
        int                accumOffset;
        int                groupSegSize;
        std::array<int, 3> sgprWorkGroup;
        int                vgprWorkItem;
        bool               enablePreloadKernArgs;

        SignatureKernelDescriptor(const std::string&        name,
                                  int                       groupSegSize,
                                  const std::array<int, 3>& sgprWorkGroup,
                                  int                       vgprWorkItem,
                                  int                       totalVgprs      = 0,
                                  int                       totalAgprs      = 0,
                                  int                       totalSgprs      = 0,
                                  bool                      preloadKernArgs = false)
            : Item(name)
            , groupSegSize(groupSegSize)
            , sgprWorkGroup(sgprWorkGroup)
            , vgprWorkItem(vgprWorkItem)
            , totalVgprs(totalVgprs)
            , totalAgprs(totalAgprs)
            , totalSgprs(totalSgprs)
            , originalTotalVgprs(totalVgprs)
            , enablePreloadKernArgs(preloadKernArgs)
        {
            if(getArchCaps()["ArchAccUnifiedRegs"])
            {
                accumOffset      = std::ceil(totalVgprs / 8.0) * 8;
                this->totalVgprs = accumOffset + totalAgprs;
            }
            else
            {
                accumOffset      = -1;
                this->totalVgprs = totalVgprs;
            }
        }

        void setGprs(int totalVgprs, int totalAgprs, int totalSgprs)
        {
            if(getArchCaps()["ArchAccUnifiedRegs"])
            {
                accumOffset      = std::ceil(totalVgprs / 8.0) * 8;
                this->totalVgprs = accumOffset + totalAgprs;
            }
            else
            {
                accumOffset      = -1;
                this->totalVgprs = std::max(totalAgprs, totalVgprs);
            }
            originalTotalVgprs = totalVgprs;
            this->totalAgprs   = totalAgprs;
            this->totalSgprs   = totalSgprs;
        }

        int getNextFreeVgpr() const
        {
            return totalVgprs;
        }

        int getNextFreeSgpr() const
        {
            return totalSgprs;
        }

        std::string toString() const override
        {
            std::string kdIndent = "  ";
            std::string kStr;
            kStr += ".amdgcn_target \"amdgcn-amd-amdhsa--" + isaToGfx(kernel().isaVersion) + "\"\n";
            kStr += ".text\n";
            kStr += ".protected " + name + "\n";
            kStr += ".globl " + name + "\n";
            kStr += ".p2align 8\n";
            kStr += ".type " + name + ",@function\n";
            kStr += ".section .rodata,#alloc\n";
            kStr += ".p2align 6\n";
            kStr += ".amdhsa_kernel " + name + "\n";
            kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_segment_ptr 1\n";
            if(accumOffset != -1)
            {
                kStr += kdIndent + ".amdhsa_accum_offset " + std::to_string(accumOffset)
                        + " // accvgpr offset\n";
            }
            kStr += kdIndent + ".amdhsa_next_free_vgpr " + std::to_string(totalVgprs)
                    + " // vgprs\n";
            kStr += kdIndent + ".amdhsa_next_free_sgpr " + std::to_string(totalSgprs)
                    + " // sgprs\n";
            kStr += kdIndent + ".amdhsa_group_segment_fixed_size " + std::to_string(groupSegSize)
                    + " // lds bytes\n";
            if(getArchCaps()["HasWave32"])
            {
                if(kernel().wavefront == 32)
                {
                    kStr += kdIndent + ".amdhsa_wavefront_size32 1 // 32-thread wavefronts\n";
                }
                else
                {
                    kStr += kdIndent + ".amdhsa_wavefront_size32 0 // 64-thread wavefronts\n";
                }
            }
            kStr += kdIndent + ".amdhsa_private_segment_fixed_size 0\n";
            kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_x "
                    + std::to_string(sgprWorkGroup[0]) + "\n";
            kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_y "
                    + std::to_string(sgprWorkGroup[1]) + "\n";
            kStr += kdIndent + ".amdhsa_system_sgpr_workgroup_id_z "
                    + std::to_string(sgprWorkGroup[2]) + "\n";
            kStr += kdIndent + ".amdhsa_system_vgpr_workitem_id " + std::to_string(vgprWorkItem)
                    + "\n";
            kStr += kdIndent + ".amdhsa_float_denorm_mode_32 3\n";
            kStr += kdIndent + ".amdhsa_float_denorm_mode_16_64 3\n";
            if(enablePreloadKernArgs)
            {
                int numWorkgroupSgpr = sgprWorkGroup[0] + sgprWorkGroup[1] + sgprWorkGroup[2];
                kStr += kdIndent + ".amdhsa_user_sgpr_count "
                        + std::to_string(16 - numWorkgroupSgpr) + "\n";
                kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_preload_length "
                        + std::to_string(14 - numWorkgroupSgpr) + "\n";
                kStr += kdIndent + ".amdhsa_user_sgpr_kernarg_preload_offset 0\n";
            }
            kStr += ".end_amdhsa_kernel\n";
            kStr += ".text\n";
            kStr += block("Num VGPR   =" + std::to_string(originalTotalVgprs));
            kStr += block("Num AccVGPR=" + std::to_string(totalAgprs));
            kStr += block("Num SGPR   =" + std::to_string(totalSgprs));
            return kStr;
        }

        std::string prettyPrint(const std::string& indent = "") const override
        {
            std::string ostream;
            ostream += indent + demangle(typeid(*this).name()) + " ";
            return ostream;
        }
    };

    struct SignatureCodeMeta : public Item
    {
        int                            kernArgsVersion;
        int                            groupSegSize;
        int                            flatWgSize;
        std::string                    codeObjectVersion;
        int                            totalVgprs;
        int                            totalSgprs;
        int                            offset;
        std::vector<SignatureArgument> argList;

        SignatureCodeMeta(const std::string& name,
                          int                kernArgsVersion,
                          int                groupSegSize,
                          int                flatWgSize,
                          const std::string& codeObjectVersion,
                          int                totalVgprs = 0,
                          int                totalSgprs = 0)
            : Item(name)
            , kernArgsVersion(kernArgsVersion)
            , groupSegSize(groupSegSize)
            , flatWgSize(flatWgSize)
            , codeObjectVersion(codeObjectVersion)
            , totalVgprs(totalVgprs)
            , totalSgprs(totalSgprs)
            , offset(0)
        {
        }

        void setGprs(int totalVgprs, int totalSgprs)
        {
            this->totalVgprs = totalVgprs;
            this->totalSgprs = totalSgprs;
        }

        std::string toString() const override
        {
            std::string kStr;
            kStr += ".amdgpu_metadata\n";
            kStr += "---\n";
            kStr += "custom.config:\n";
            kStr += "  InternalSupportParams:\n";
            kStr += "    KernArgsVersion: " + std::to_string(kernArgsVersion) + "\n";
            kStr += "amdhsa.version:\n";
            kStr += "  - 1\n";
            if(codeObjectVersion == "4" || codeObjectVersion == "default")
            {
                kStr += "  - 1\n";
            }
            else if(codeObjectVersion == "5")
            {
                kStr += "  - 2\n";
            }
            kStr += "amdhsa.kernels:\n";
            kStr += "  - .name: " + name + "\n";
            kStr += "    .symbol: '" + name + ".kd'\n";
            kStr += "    .language:                   OpenCL C\n";
            kStr += "    .language_version:\n";
            kStr += "      - 2\n";
            kStr += "      - 0\n";
            kStr += "    .args:\n";
            for(const auto& arg : argList)
            {
                kStr += arg.toString();
            }
            kStr += "    .group_segment_fixed_size:   " + std::to_string(groupSegSize) + "\n";
            kStr += "    .kernarg_segment_align:      8\n";
            kStr += "    .kernarg_segment_size:       " + std::to_string(((offset + 7) / 8) * 8)
                    + "\n";
            kStr += "    .max_flat_workgroup_size:    " + std::to_string(flatWgSize) + "\n";
            kStr += "    .private_segment_fixed_size: 0\n";
            kStr += "    .sgpr_count:                 " + std::to_string(totalSgprs) + "\n";
            kStr += "    .sgpr_spill_count:           0\n";
            kStr += "    .vgpr_count:                 " + std::to_string(totalVgprs) + "\n";
            kStr += "    .vgpr_spill_count:           0\n";
            kStr += "    .wavefront_size:             " + std::to_string(kernel().wavefront) + "\n";
            kStr += "...\n";
            kStr += ".end_amdgpu_metadata\n";
            kStr += name + ":\n";
            return kStr;
        }

        void addArg(const std::string&                name,
                    SignatureValueKind                kind,
                    const std::string&                type,
                    const std::optional<std::string>& addrSpaceQual = std::nullopt)
        {
            SignatureArgument sa(offset, name, kind, type, addrSpaceQual.value_or(""));
            argList.push_back(sa);
            offset += sa.size;
        }

        std::string prettyPrint(const std::string& indent = "") const override
        {
            std::string ostream;
            ostream += indent + demangle(typeid(*this).name()) + " ";
            return ostream;
        }
    };

    struct SignatureBase : public Item
    {
        SignatureKernelDescriptor kernelDescriptor;
        SignatureCodeMeta         codeMeta;

        SignatureBase(const std::string&        kernelName,
                      int                       kernArgsVersion,
                      const std::string&        codeObjectVersion,
                      int                       groupSegmentSize,
                      const std::array<int, 3>& sgprWorkGroup,
                      int                       vgprWorkItem,
                      int                       flatWorkGroupSize,
                      int                       totalVgprs      = 0,
                      int                       totalAgprs      = 0,
                      int                       totalSgprs      = 0,
                      bool                      preloadKernArgs = false)
            : Item(kernelName)
            , kernelDescriptor(kernelName,
                               groupSegmentSize,
                               sgprWorkGroup,
                               vgprWorkItem,
                               totalVgprs,
                               totalAgprs,
                               totalSgprs,
                               preloadKernArgs)
            , codeMeta(kernelName,
                       kernArgsVersion,
                       groupSegmentSize,
                       flatWorkGroupSize,
                       codeObjectVersion,
                       totalVgprs,
                       totalSgprs)
            , descriptionTopic(TextBlock(""))
        {
        }

        void setGprs(int totalVgprs, int totalAgprs, int totalSgprs)
        {
            kernelDescriptor.setGprs(totalVgprs, totalAgprs, totalSgprs);
            codeMeta.setGprs(totalVgprs, totalSgprs);
        }

        void addArg(const std::string&                name,
                    SignatureValueKind                kind,
                    const std::string&                type,
                    const std::optional<std::string>& addrSpaceQual = std::nullopt)
        {
            codeMeta.addArg(name, kind, type, addrSpaceQual);
        }

        void addDescriptionTopic(const std::string& text)
        {
            descriptionTopic = std::move(TextBlock(block3Line(text)));
        }

        void addDescriptionBlock(const std::string& text)
        {
            descriptionList.emplace_back(TextBlock(block(text)));
        }

        void addDescription(const std::string& text)
        {
            descriptionList.emplace_back(TextBlock(slash(text)));
        }

        int getNextFreeVgpr() const
        {
            return kernelDescriptor.getNextFreeVgpr();
        }

        int getNextFreeSgpr() const
        {
            return kernelDescriptor.getNextFreeSgpr();
        }

        void clearDescription()
        {
            descriptionList.clear();
        }

        std::string toString() const override
        {
            std::string kStr;
            kStr += kernelDescriptor.toString();
            auto topic = descriptionTopic.toString();
            if(!topic.empty())
            {
                kStr += topic;
            }
            for(const auto& i : descriptionList)
            {
                kStr += i.toString();
            }
            kStr += codeMeta.toString();
            return kStr;
        }

        std::string prettyPrint(const std::string& indent = "") const override
        {
            std::string ostream;
            ostream += indent + demangle(typeid(*this).name()) + " ";
            return ostream;
        }

    private:
        TextBlock              descriptionTopic;
        std::vector<TextBlock> descriptionList;
    };

    struct KernelBody : public Item
    {
        std::shared_ptr<SignatureBase> signature;
        std::shared_ptr<Module>        body;
        int                            totalVgprs;
        int                            totalAgprs;
        int                            totalSgprs;

        KernelBody(const std::string& name)
            : Item(name)
        {
        }

        void addSignature(const std::shared_ptr<SignatureBase>& signature)
        {
            this->signature = signature;
        }

        void addBody(const std::shared_ptr<Module>& body)
        {
            this->body = body;
        }

        void setGprs(int totalVgprs, int totalAgprs, int totalSgprs)
        {
            this->totalVgprs = totalVgprs;
            this->totalAgprs = totalAgprs;
            this->totalSgprs = totalSgprs;
            if(signature)
            {
                signature->setGprs(totalVgprs, totalAgprs, totalSgprs);
            }
        }

        int getNextFreeVgpr() const
        {
            return signature ? signature->getNextFreeVgpr() : 0;
        }

        int getNextFreeSgpr() const
        {
            return signature ? signature->getNextFreeSgpr() : 0;
        }

        std::string toString() const override
        {
            std::string kStr = TextBlock(block3Line("Begin Kernel")).toString();
            if(signature)
            {
                kStr += signature->toString();
            }
            if(body)
            {
                kStr += body->toString();
            }
            else
            {
                throw std::runtime_error("Kernel body is empty");
            }
            return kStr;
        }
    };
} // namespace rocisa
