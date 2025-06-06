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
#include "register.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void init_register(nb::module_ m)
{
    auto m_reg = m.def_submodule("register", "rocIsa register submodule.");

    nb::class_<rocisa::RegisterPool> registerPool(m_reg, "RegisterPool");
    registerPool
        .def(nb::init<size_t, rocisa::RegisterType, bool, bool>(),
             nb::arg("size"),
             nb::arg("type"),
             nb::arg("defaultPreventOverflow"),
             nb::arg("printRP") = false)
        .def("setOccupancyLimit",
             &rocisa::RegisterPool::setOccupancyLimit,
             nb::arg("maxSize"),
             nb::arg("size"))
        .def("resetOccupancyLimit", &rocisa::RegisterPool::resetOccupancyLimit)
        .def("getPool", &rocisa::RegisterPool::getPool)
        .def("add",
             &rocisa::RegisterPool::add,
             nb::arg("start"),
             nb::arg("size"),
             nb::arg("tag") = "")
        .def("addRange",
             &rocisa::RegisterPool::addRange,
             nb::arg("start"),
             nb::arg("stop"),
             nb::arg("tag") = "")
        .def("addFromCheckOut", &rocisa::RegisterPool::addFromCheckOut, nb::arg("start"))
        .def("remove",
             &rocisa::RegisterPool::remove,
             nb::arg("start"),
             nb::arg("size"),
             nb::arg("tag") = "")
        .def("removeFromCheckOut", &rocisa::RegisterPool::removeFromCheckOut, nb::arg("start"))
        .def("checkOut",
             &rocisa::RegisterPool::checkOut,
             nb::arg("size"),
             nb::arg("tag")             = "_untagged_",
             nb::arg("preventOverflow") = -1)
        .def("checkOutAligned",
             &rocisa::RegisterPool::checkOutAligned,
             nb::arg("size"),
             nb::arg("alignment"),
             nb::arg("tag")             = "_untagged_aligned_",
             nb::arg("preventOverflow") = -1)
        .def("checkOutMulti",
             &rocisa::RegisterPool::checkOutMulti,
             nb::arg("sizes"),
             nb::arg("alignment"),
             nb::arg("tags"))
        .def("initTmps",
             &rocisa::RegisterPool::initTmps,
             nb::arg("initValue"),
             nb::arg("start") = 0,
             nb::arg("stop")  = -1)
        .def("checkIn", &rocisa::RegisterPool::checkIn, nb::arg("start"))
        .def("size", &rocisa::RegisterPool::size)
        .def("available", &rocisa::RegisterPool::available)
        .def("availableBlock",
             &rocisa::RegisterPool::availableBlock,
             nb::arg("blockSize"),
             nb::arg("align"))
        .def("availableBlockMaxVgpr",
             &rocisa::RegisterPool::availableBlockMaxVgpr,
             nb::arg("maxVgpr"),
             nb::arg("blockSize"),
             nb::arg("align"))
        .def("availableBlockAtEnd", &rocisa::RegisterPool::availableBlockAtEnd)
        .def("checkFinalState", &rocisa::RegisterPool::checkFinalState)
        .def("state", &rocisa::RegisterPool::state)
        .def("growPool",
             &rocisa::RegisterPool::growPool,
             nb::arg("rangeStart"),
             nb::arg("rangeEnd"),
             nb::arg("checkOutSize"),
             nb::arg("comment") = "")
        .def("appendPool", &rocisa::RegisterPool::appendPool, nb::arg("newSize"))
        .def("__deepcopy__", [](const rocisa::RegisterPool& self, nb::dict&) {
            rocisa::RegisterPool* new_pool = new rocisa::RegisterPool(self);
            return new_pool;
        });

    enum class Status
    {
        Unavailable = 0,
        Available   = 1,
        InUse       = 2
    };

    struct Register
    {
        Status      status;
        std::string tag;

        Register(Status s, const std::string& t)
            : status(s)
            , tag(t)
        {
        }
    };

    nb::enum_<rocisa::RegisterPool::Status>(registerPool, "Status")
        .value("Unavailable", rocisa::RegisterPool::Status::Unavailable)
        .value("Available", rocisa::RegisterPool::Status::Available)
        .value("InUse", rocisa::RegisterPool::Status::InUse)
        .export_values();
    nb::class_<rocisa::RegisterPool::Register>(registerPool, "Register")
        .def(nb::init<rocisa::RegisterPool::Status, const std::string&>())
        .def_rw("status", &rocisa::RegisterPool::Register::status)
        .def_rw("tag", &rocisa::RegisterPool::Register::tag);
}
