// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_corpus_gen/PoolAssembly.hpp>
#include <hipdnn_corpus_gen/RegimeLabel.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

/// @file ModelShapeSource.hpp
/// @brief The model pool: shapes somebody recorded a real workload running.
///
/// The exploration knows what the operation admits and the pack knows what the engine was
/// compiled for. Neither knows what anyone runs, and a corpus without that is a map of a region
/// with the cities left off -- correct, and wrong where the error matters.
///
/// Mining the shapes is not done here. That work is markdown and CSV scraping over the dozens
/// of column spellings published model cards use, it is inherently per-operation, and it does
/// not belong in a tool whose point is that nothing in it names an operation. It stays in
/// `IngestorGenerator/tools/mine_shapes.py`, which emits the `q.<parameter>` columns this reads
/// -- the same column spelling the corpus itself emits, so a miner's output can be diffed
/// against a manifest without a translation step.
namespace hipdnn_corpus_gen
{

/// What a shape file yielded, and what it did not.
struct ModelShapeReport
{
    std::filesystem::path path;
    int64_t rows = 0;

    /// Rows naming another operation in an `op` column. Not a defect: one mined file may cover
    /// several operations, and each run takes its own.
    int64_t otherOperation = 0;

    /// Rows missing a parameter the declaration requires, or carrying a value it does not
    /// accept. Counted rather than dropped in silence -- a miner emitting `causal` where the
    /// declaration says `is_causal` loses every row, and a count is the only way that shows.
    int64_t unusable = 0;

    /// First reason a row was unusable, since one message is worth more than a count.
    std::string firstProblem;
};

namespace detail
{

/// One CSV line into fields, honouring the quoting `CorpusManifest.hpp` emits.
inline std::vector<std::string> csvFields(const std::string& line)
{
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;

    for(size_t i = 0; i < line.size(); ++i)
    {
        const auto character = line[i];
        if(quoted)
        {
            if(character == '"')
            {
                if(i + 1 < line.size() && line[i + 1] == '"')
                {
                    field += '"';
                    ++i;
                }
                else
                {
                    quoted = false;
                }
            }
            else
            {
                field += character;
            }
        }
        else if(character == '"')
        {
            quoted = true;
        }
        else if(character == ',')
        {
            fields.push_back(field);
            field.clear();
        }
        else if(character != '\r')
        {
            field += character;
        }
    }
    fields.push_back(field);
    return fields;
}

/// @brief One row's text values as a point of @p metadata's declared types.
///
/// Every declared parameter must be present and readable. A partial point is refused rather
/// than completed from a default: the missing parameter is a dimension of the problem, and a
/// default for it is a shape nobody recorded being reported as one that was.
inline bool pointFromColumns(const OperationMetadata& metadata,
                             const std::map<std::string, std::string>& values,
                             ProblemPoint& point,
                             std::string& problem)
{
    point.clear();
    for(const auto& parameter : metadata.parameters)
    {
        const auto found = values.find(parameter.name);
        if(found == values.end() || found->second.empty())
        {
            problem = "no value for '" + parameter.name + "'";
            return false;
        }
        const auto& text = found->second;

        switch(parameter.type)
        {
        case ParameterType::INT64:
        {
            char* end = nullptr;
            const auto number = std::strtoll(text.c_str(), &end, 10);
            if(end == text.c_str() || *end != '\0')
            {
                problem = "'" + parameter.name + "' is not an integer: '" + text + "'";
                return false;
            }
            point[parameter.name] = static_cast<int64_t>(number);
            break;
        }

        case ParameterType::FLOAT64:
        {
            char* end = nullptr;
            const auto number = std::strtod(text.c_str(), &end);
            if(end == text.c_str() || *end != '\0')
            {
                problem = "'" + parameter.name + "' is not a number: '" + text + "'";
                return false;
            }
            point[parameter.name] = number;
            break;
        }

        case ParameterType::BOOL:
            if(text == "true" || text == "1")
            {
                point[parameter.name] = true;
            }
            else if(text == "false" || text == "0")
            {
                point[parameter.name] = false;
            }
            else
            {
                problem = "'" + parameter.name + "' is not a boolean: '" + text + "'";
                return false;
            }
            break;

        case ParameterType::ENUM:
            if(std::find(parameter.values.begin(), parameter.values.end(), text)
               == parameter.values.end())
            {
                problem = "'" + parameter.name + "' is not a declared value: '" + text + "'";
                return false;
            }
            point[parameter.name] = text;
            break;

        default:
            problem = "'" + parameter.name + "' has an unhandled declared type";
            return false;
        }
    }
    return true;
}

} // namespace detail

/// @brief The recorded shapes in @p path that @p metadata can express.
///
/// The file is a CSV whose header names columns `q.<parameter>`; a bare `<parameter>` is also
/// accepted, so a miner that predates the prefix still reads. An optional `op` column selects
/// rows, and an optional `name` column becomes the entry's origin -- which is the whole value
/// of this pool to a later audit, since "llama3-70b prefill" explains a regime that a row of
/// numbers does not.
///
/// Admission is the caller's: the points come back unfiltered by any oracle, because whether an
/// engine serves a shape somebody actually runs is a finding, not a reason to drop the row here.
inline std::vector<PoolEntry> readModelShapes(const OperationMetadata& metadata,
                                              const std::filesystem::path& path,
                                              ModelShapeReport& report)
{
    report = ModelShapeReport{};
    report.path = path;

    std::ifstream file(path);
    if(!file)
    {
        report.firstProblem = "cannot read " + path.string();
        return {};
    }

    std::string line;
    if(!std::getline(file, line))
    {
        return {};
    }

    auto header = detail::csvFields(line);
    for(auto& column : header)
    {
        if(column.rfind("q.", 0) == 0)
        {
            column = column.substr(2);
        }
    }

    std::vector<PoolEntry> entries;
    while(std::getline(file, line))
    {
        if(line.empty())
        {
            continue;
        }
        ++report.rows;

        const auto fields = detail::csvFields(line);
        std::map<std::string, std::string> values;
        for(size_t i = 0; i < header.size() && i < fields.size(); ++i)
        {
            values[header[i]] = fields[i];
        }

        const auto operation = values.find("op");
        if(operation != values.end() && !operation->second.empty()
           && operation->second != metadata.operation)
        {
            ++report.otherOperation;
            continue;
        }

        PoolEntry entry;
        std::string problem;
        if(!detail::pointFromColumns(metadata, values, entry.point, problem))
        {
            ++report.unusable;
            if(report.firstProblem.empty())
            {
                report.firstProblem = problem;
            }
            continue;
        }

        const auto name = values.find("name");
        entry.source = "model";
        entry.origin = path.filename().string() + " "
                       + (name == values.end() || name->second.empty()
                              ? "row " + std::to_string(report.rows)
                              : name->second);
        entry.regime = regimeLabel(metadata, entry.point);
        entries.push_back(std::move(entry));
    }
    return entries;
}

} // namespace hipdnn_corpus_gen
