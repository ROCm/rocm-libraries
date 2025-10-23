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
#include "code.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace rocisa
{
    std::vector<std::shared_ptr<Item>>
        cloneItemList(const std::vector<std::shared_ptr<Item>>& itemList)
    {
        std::vector<std::shared_ptr<Item>> clonedItemList;
        for(const auto& item : itemList)
        {
            std::shared_ptr<Item> cloned;
            if(typeid(item.get()) == typeid(Module))
            {
                cloned = std::make_shared<Module>(*dynamic_cast<Module*>(item.get()));
            }
            else
                cloned = item->clone();
            clonedItemList.push_back(cloned);
        }
        return std::move(clonedItemList);
    }

    std::shared_ptr<BitfieldUnion> SrdUpperValue(const IsaVersion& isa)
    {
        if(isa[0] == 12)
        {
            return std::make_shared<SrdUpperValue12XX>(SrdUpperValue12XX::staticInit());
        }
        else if(isa[0] == 11)
        {
            return std::make_shared<SrdUpperValue11XX>(SrdUpperValue11XX::staticInit());
        }
        else if(isa[0] == 10)
        {
            return std::make_shared<SrdUpperValue10XX>(SrdUpperValue10XX::staticInit());
        }
        else
        {
            return std::make_shared<SrdUpperValue9XX>(SrdUpperValue9XX::staticInit());
        }
    }

    auto SrdUpperValueTuple(const nb::tuple& t)
    {
        return rocisa::SrdUpperValue(
            IsaVersion{nb::cast<int>(t[0]), nb::cast<int>(t[1]), nb::cast<int>(t[2])});
    }

    const std::unordered_map<std::string, int> SignatureArgument::ValueTypeSizeDict
        = {{"i8", 1},
           {"i16", 2},
           {"i32", 4},
           {"i64", 8},
           {"u8", 1},
           {"u16", 2},
           {"u32", 4},
           {"u64", 8},
           {"bf16", 2},
           {"f16", 2},
           {"f32", 4},
           {"f64", 8},
           {"pkf16", 4},
           {"struct", 8}};
} // namespace rocisa

void init_code(nb::module_ m)
{
    auto m_code = m.def_submodule("code", "rocIsa code submodule.");

    nb::class_<rocisa::Label, rocisa::Item>(m_code, "Label")
        .def(nb::init<int, const std::string&, int>(),
             nb::arg("label"),
             nb::arg("comment"),
             nb::arg("alignment") = 1)
        .def(nb::init<const std::string&, const std::string&, int>(),
             nb::arg("label"),
             nb::arg("comment"),
             nb::arg("alignment") = 1)
        .def_static("getFormatting", &rocisa::Label::getFormatting)
        .def_rw("label", &rocisa::Label::label)
        .def_rw("comment", &rocisa::Label::comment)
        .def_rw("alignment", &rocisa::Label::alignment)
        .def("getLabelName", &rocisa::Label::getLabelName)
        .def("__str__", &rocisa::Label::toString)
        .def("__deepcopy__",
             [](const rocisa::Label& self, nb::dict&) {
                 rocisa::Label* new_label = new rocisa::Label(self);
                 return new_label;
             })
        .def("__getstate__",
             [](const rocisa::Label& self) {
                 return std::make_tuple(self.name, self.label, self.comment, self.alignment);
             })
        .def("__setstate__",
             [](rocisa::Label&                                                            self,
                std::tuple<std::string, std::variant<std::string, int>, std::string, int> state) {
                 new(&self)
                     rocisa::Label(std::get<1>(state), std::get<2>(state), std::get<3>(state));
                 self.name = std::get<0>(state);
             });

    nb::class_<rocisa::TextBlock, rocisa::Item>(m_code, "TextBlock")
        .def(nb::init<const std::string&>(), nb::arg("text"))
        .def("__str__", &rocisa::TextBlock::toString)
        .def("__deepcopy__",
             [](const rocisa::TextBlock& self, nb::dict&) {
                 rocisa::TextBlock* new_text_block = new rocisa::TextBlock(self.text);
                 new_text_block->name              = self.name;
                 return new_text_block;
             })
        .def("__getstate__",
             [](const rocisa::TextBlock& self) { return std::make_tuple(self.name, self.text); })
        .def("__setstate__",
             [](rocisa::TextBlock& self, const std::tuple<std::string, std::string>& state) {
                 new(&self) rocisa::TextBlock(std::get<1>(state));
                 self.name = std::get<0>(state);
             });

    nb::class_<rocisa::Module, rocisa::Item>(m_code, "Module")
        .def(nb::init<const std::string&>(), nb::arg("name") = "")
        .def("setParent", &rocisa::Module::setParent)
        .def("setNoOpt", &rocisa::Module::setNoOpt)
        .def("isNoOpt", &rocisa::Module::isNoOpt)
        .def("findNamedItem", &rocisa::Module::findNamedItem)
        .def("setInlineAsmPrintMode", &rocisa::Module::setInlineAsmPrintMode)
        .def("addSpaceLine", &rocisa::Module::addSpaceLine)
        .def("add", &rocisa::Module::add, nb::arg("item"), nb::arg("pos") = -1)
        // nanobind cannot reliably bind the same method signatures that differ
        // only by const-ness, so indirectly bind via a lambda. This manifests
        // as a mangled name clash on Windows.
        .def("addItems",
             [](rocisa::Module& self, const std::vector<std::shared_ptr<rocisa::Item>>& items) {
                 self.addItems(items);
             })
        .def("appendModule", &rocisa::Module::appendModule)
        .def("addModuleAsFlatItems", &rocisa::Module::addModuleAsFlatItems)
        .def("findIndex", &rocisa::Module::findIndex)
        .def("findIndexByType", &rocisa::Module::findIndexByType)
        .def("addComment", &rocisa::Module::addComment)
        .def("addCommentAlign", &rocisa::Module::addCommentAlign)
        .def("addComment0", &rocisa::Module::addComment0)
        .def("addComment1", &rocisa::Module::addComment1)
        .def("addComment2", &rocisa::Module::addComment2)
        .def("prettyPrint", &rocisa::Module::prettyPrint)
        .def("count", &rocisa::Module::count)
        .def("getItem", &rocisa::Module::getItem)
        .def("setItem", &rocisa::Module::setItem)
        // nanobind cannot reliably bind the same method signatures that differ
        // only by const-ness, so indirectly bind via a lambda. This manifests
        // as a mangled name clash on Windows.
        .def("setItems",
             [](rocisa::Module& self, const std::vector<std::shared_ptr<rocisa::Item>>& items) {
                 self.setItems(items);
             })
        .def("items", &rocisa::Module::items)
        .def("itemsSize", &rocisa::Module::itemsSize)
        .def("replaceItem", &rocisa::Module::replaceItem)
        .def("replaceItemByIndex", &rocisa::Module::replaceItemByIndex)
        .def("removeItemByIndex", &rocisa::Module::removeItemByIndex)
        .def("removeItem", &rocisa::Module::removeItem)
        .def("removeItemsByName", &rocisa::Module::removeItemsByName)
        .def("popFirstItem", &rocisa::Module::popFirstItem)
        .def("popFirstNItems", &rocisa::Module::popFirstNItems)
        .def("flatitems", &rocisa::Module::flatitems)
        .def("addTempVgpr", &rocisa::Module::addTempVgpr)
        .def("__str__", &rocisa::Module::toString)
        .def("__deepcopy__",
             [](const rocisa::Module& self, nb::dict&) {
                 rocisa::Module* new_module = new rocisa::Module(self);
                 return new_module;
             })
        .def("__reduce__", [](const rocisa::Module& self) {
            throw std::runtime_error("Module is not picklable");
        });

    nb::class_<rocisa::Macro, rocisa::Item>(m_code, "Macro")
        .def(nb::init<const std::string&, const std::vector<InstructionInput>&>(),
             nb::arg("name"),
             nb::arg("args"))
        .def("add", &rocisa::Macro::add)
        .def("addComment0", &rocisa::Macro::addComment0)
        .def("setItems", &rocisa::Macro::setItems)
        .def("items", &rocisa::Macro::items)
        .def("prettyPrint", &rocisa::Macro::prettyPrint)
        .def("__str__", &rocisa::Macro::toString)
        .def("__deepcopy__",
             [](const rocisa::Macro& self, nb::dict&) {
                 rocisa::Macro* new_macro = new rocisa::Macro(self);
                 return new_macro;
             })
        .def("__reduce__", [](const rocisa::Module& self) {
            throw std::runtime_error("Macro is not picklable");
        });

    nb::class_<rocisa::StructuredModule, rocisa::Module>(m_code, "StructuredModule")
        .def(nb::init<const std::string&>(), nb::arg("name") = "")
        .def_rw("header", &rocisa::StructuredModule::header)
        .def_rw("middle", &rocisa::StructuredModule::middle)
        .def_rw("footer", &rocisa::StructuredModule::footer)
        .def("__deepcopy__",
             [](const rocisa::StructuredModule& self, nb::dict&) {
                 return new rocisa::StructuredModule(self);
             })
        .def("__reduce__", [](const rocisa::StructuredModule& self) {
            throw std::runtime_error("StructuredModule is not picklable");
        });

    nb::class_<rocisa::ValueEndif, rocisa::Item>(m_code, "ValueEndif")
        .def(nb::init<const std::string&>(), nb::arg("comment") = "")
        .def("__str__", &rocisa::ValueEndif::toString)
        .def("__deepcopy__",
             [](const rocisa::ValueEndif& self, nb::dict&) { return new rocisa::ValueEndif(self); })
        .def("__getstate__", [](const rocisa::ValueEndif& self) { return self.comment; })
        .def("__setstate__", [](rocisa::ValueEndif& self, const std::string& state) {
            new(&self) rocisa::ValueEndif(state);
        });

    nb::class_<rocisa::ValueIf, rocisa::Item>(m_code, "ValueIf")
        .def(nb::init<int>(), nb::arg("value"))
        .def("__str__", &rocisa::ValueIf::toString)
        .def("__deepcopy__",
             [](const rocisa::ValueIf& self, nb::dict&) { return new rocisa::ValueIf(self); })
        .def("__getstate__", [](const rocisa::ValueIf& self) { return self.value; })
        .def("__setstate__",
             [](rocisa::ValueIf& self, int value) { new(&self) rocisa::ValueIf(value); });

    nb::class_<rocisa::ValueSet, rocisa::Item>(m_code, "ValueSet")
        .def(nb::init<const std::string&, int, int, int>(),
             nb::arg("name"),
             nb::arg("value"),
             nb::arg("offset") = 0,
             nb::arg("format") = 0)
        .def(nb::init<const std::string&, uint32_t, int, int>(),
             nb::arg("name"),
             nb::arg("value"),
             nb::arg("offset") = 0,
             nb::arg("format") = 0)
        .def(nb::init<const std::string&, const std::string&, int, int>(),
             nb::arg("name"),
             nb::arg("value"),
             nb::arg("offset") = 0,
             nb::arg("format") = 0)
        .def("__str__", &rocisa::ValueSet::toString)
        .def("__deepcopy__",
             [](const rocisa::ValueSet& self, nb::dict&) { return new rocisa::ValueSet(self); })
        .def("__getstate__",
             [](const rocisa::ValueSet& self) {
                 return std::make_tuple(self.name, self.ref, self.value, self.offset, self.format);
             })
        .def("__setstate__",
             [](rocisa::ValueSet& self,
                std::tuple<std::string, std::optional<std::string>, std::optional<int>, int, int>
                    t) {
                 auto ref   = std::get<1>(t);
                 auto value = std::get<2>(t);
                 if(ref.has_value())
                     new(&self) rocisa::ValueSet(
                         std::get<0>(t), ref.value(), std::get<3>(t), std::get<4>(t));
                 else
                     new(&self) rocisa::ValueSet(
                         std::get<0>(t), value.value(), std::get<3>(t), std::get<4>(t));
             });

    nb::class_<rocisa::RegSet, rocisa::ValueSet>(m_code, "RegSet")
        .def(nb::init<const std::string&, const std::string&, int, int>(),
             nb::arg("regType"),
             nb::arg("name"),
             nb::arg("value"),
             nb::arg("offset") = 0)
        .def(nb::init<const std::string&, const std::string&, const std::string&, int>(),
             nb::arg("regType"),
             nb::arg("name"),
             nb::arg("value"),
             nb::arg("offset") = 0)
        .def_rw("ref", &rocisa::RegSet::ref)
        .def_rw("value", &rocisa::RegSet::value)
        .def_rw("offset", &rocisa::RegSet::offset)
        .def("__deepcopy__",
             [](const rocisa::RegSet& self, nb::dict&) { return new rocisa::RegSet(self); })
        .def("__getstate__",
             [](const rocisa::RegSet& self) {
                 return std::make_tuple(
                     self.regType, self.name, self.ref, self.value, self.offset, self.format);
             })
        .def("__setstate__",
             [](rocisa::RegSet& self,
                std::tuple<std::string,
                           std::string,
                           std::optional<std::string>,
                           std::optional<int>,
                           int,
                           int> t) {
                 auto ref   = std::get<2>(t);
                 auto value = std::get<3>(t);
                 if(ref.has_value())
                     new(&self) rocisa::RegSet(
                         std::get<0>(t), std::get<1>(t), ref.value(), std::get<4>(t));
                 else
                     new(&self) rocisa::RegSet(
                         std::get<0>(t), std::get<1>(t), value.value(), std::get<4>(t));
                 self.format = std::get<5>(t);
             });

    nb::class_<rocisa::BitfieldUnion>(m_code, "BitfieldUnion")
        .def("__str__", &rocisa::BitfieldUnion::toString)
        .def("getValue", &rocisa::BitfieldUnion::getValue)
        .def("desc", &rocisa::BitfieldUnion::desc);

    nb::class_<rocisa::SignatureCodeMeta, rocisa::Item>(m_code, "SignatureCodeMeta")
        .def(nb::init<const std::string&, int, int, int, const std::string&, int, int>(),
             nb::arg("name"),
             nb::arg("kernArgsVersion"),
             nb::arg("groupSegSize"),
             nb::arg("flatWgSize"),
             nb::arg("codeObjectVersion"),
             nb::arg("totalVgprs") = 0,
             nb::arg("totalSgprs") = 0)
        .def("setGprs", &rocisa::SignatureCodeMeta::setGprs)
        .def("addArg",
             &rocisa::SignatureCodeMeta::addArg,
             nb::arg("name"),
             nb::arg("kind"),
             nb::arg("type"),
             nb::arg("addrSpaceQual") = std::nullopt)
        .def("__str__", &rocisa::SignatureCodeMeta::toString)
        .def("__deepcopy__",
             [](const rocisa::SignatureCodeMeta& self, nb::dict&) {
                 throw std::runtime_error("SignatureCodeMeta is not deepcopyable");
             })
        .def("__reduce__", [](const rocisa::SignatureCodeMeta& self) {
            throw std::runtime_error("SignatureCodeMeta is not picklable");
        });

    nb::class_<rocisa::SignatureBase, rocisa::Item>(m_code, "SignatureBase")
        .def(nb::init<const std::string&,
                      int,
                      const std::string&,
                      int,
                      const std::array<int, 3>&,
                      int,
                      int,
                      int,
                      int,
                      int,
                      bool>(),
             nb::arg("kernelName"),
             nb::arg("kernArgsVersion"),
             nb::arg("codeObjectVersion"),
             nb::arg("groupSegmentSize"),
             nb::arg("sgprWorkGroup"),
             nb::arg("vgprWorkItem"),
             nb::arg("flatWorkGroupSize"),
             nb::arg("totalVgprs")      = 0,
             nb::arg("totalAgprs")      = 0,
             nb::arg("totalSgprs")      = 0,
             nb::arg("preloadKernArgs") = false)
        .def("setGprs", &rocisa::SignatureBase::setGprs)
        .def("addArg",
             &rocisa::SignatureBase::addArg,
             nb::arg("name"),
             nb::arg("kind"),
             nb::arg("type"),
             nb::arg("addrSpaceQual") = std::nullopt)
        .def("addDescriptionTopic", &rocisa::SignatureBase::addDescriptionTopic)
        .def("addDescriptionBlock", &rocisa::SignatureBase::addDescriptionBlock)
        .def("addDescription", &rocisa::SignatureBase::addDescription)
        .def("getNextFreeVgpr", &rocisa::SignatureBase::getNextFreeVgpr)
        .def("getNextFreeSgpr", &rocisa::SignatureBase::getNextFreeSgpr)
        .def("clearDescription", &rocisa::SignatureBase::clearDescription)
        .def("__str__", &rocisa::SignatureBase::toString)
        .def("__deepcopy__",
             [](const rocisa::SignatureBase& self, nb::dict&) {
                 throw std::runtime_error("SignatureBase is not deepcopyable");
             })
        .def("__reduce__", [](const rocisa::SignatureBase& self) {
            throw std::runtime_error("SignatureBase is not picklable");
        });

    nb::class_<rocisa::KernelBody, rocisa::Item>(m_code, "KernelBody")
        .def(nb::init<const std::string&>(), nb::arg("name"))
        .def_rw("body", &rocisa::KernelBody::body)
        .def_rw("totalVgprs", &rocisa::KernelBody::totalVgprs)
        .def_rw("totalAgprs", &rocisa::KernelBody::totalAgprs)
        .def_rw("totalSgprs", &rocisa::KernelBody::totalSgprs)
        .def("addSignature", &rocisa::KernelBody::addSignature)
        .def("addBody", &rocisa::KernelBody::addBody)
        .def("setGprs",
             &rocisa::KernelBody::setGprs,
             nb::arg("totalVgprs"),
             nb::arg("totalAgprs"),
             nb::arg("totalSgprs"))
        .def("getNextFreeVgpr", &rocisa::KernelBody::getNextFreeVgpr)
        .def("getNextFreeSgpr", &rocisa::KernelBody::getNextFreeSgpr)
        .def("__str__", &rocisa::KernelBody::toString)
        .def("__deepcopy__",
             [](const rocisa::KernelBody& self, nb::dict&) {
                 throw std::runtime_error("KernelBody is not deepcopyable");
             })
        .def("__reduce__", [](const rocisa::KernelBody& self) {
            throw std::runtime_error("KernelBody is not picklable");
        });

    m_code.def("SrdUpperValue", nb::overload_cast<const nb::tuple&>(&rocisa::SrdUpperValueTuple));
    m_code.def("SrdUpperValue", nb::overload_cast<const IsaVersion&>(&rocisa::SrdUpperValue));
}
