// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "harness/catalog-sweep/CatalogReportWriter.hpp"

#include <ios>
#include <stdexcept>

namespace hipdnn_integration_tests::catalog_sweep
{

CatalogReportWriter::CatalogReportWriter(const std::filesystem::path& path)
    : _path(path)
    , _stream(path, std::ios::out | std::ios::app | std::ios::binary)
{
    if(!_stream.is_open())
    {
        throw std::runtime_error("--catalog-report: cannot open '" + path.string()
                                 + "' for appending");
    }
}

void CatalogReportWriter::append(const nlohmann::json& record)
{
    // Binary mode: a report written on Windows must not carry CRLF line ends, or a
    // line-oriented reader on the other platform sees a stray '\r' in every record.
    // `replace` rather than the default `strict`: a plugin's message can hold bytes that
    // are not UTF-8, and losing a result to a serialization throw is the worse outcome.
    _stream << record.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace) << '\n';
    _stream.flush();
    if(!_stream)
    {
        throw std::runtime_error("--catalog-report: failed writing to '" + _path.string() + "'");
    }
}

} // namespace hipdnn_integration_tests::catalog_sweep
