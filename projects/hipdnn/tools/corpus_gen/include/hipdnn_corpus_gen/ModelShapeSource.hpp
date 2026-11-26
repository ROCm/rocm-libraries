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
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

/// @file ModelShapeSource.hpp
/// @brief The model pool: shapes somebody recorded a real workload running.
///
/// Mining is done by `IngestorGenerator/tools/mine_shapes.py`, which emits the `q.<parameter>`
/// columns read here.
namespace hipdnn_corpus_gen
{

/// What a shape file yielded and what it rejected.
struct ModelShapeReport
{
    std::filesystem::path path;
    int64_t rows = 0;

    /// Rows whose `op` column names another operation; not a defect.
    int64_t otherOperation = 0;

    /// Rows missing a required parameter or carrying an undeclared value.
    int64_t unusable = 0;

    /// First reason a row was unusable.
    std::string firstProblem;

    /// Rows that took a parameter's `model_shape_defaults` value because the file has no column
    /// for it, by parameter.
    std::map<std::string, int64_t> defaulted;
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
/// Every declared parameter must be present and readable; a partial point is refused rather
/// than defaulted, since a default would pass off an unrecorded shape as recorded. The one
/// exception is a parameter the file has no column for at all and the operation lists under
/// `model_shape_defaults` (`layout`, say): the listed value is the operation's own meaning of a
/// source without it.
/// Pack constants are not consulted: they answer for kernels, not for recorded workloads, and
/// `alignment` or `generate_stats` left out of a file must not become a recorded decode shape.
/// A column that exists but is empty on a row is still a refusal. Each parameter filled this way
/// is appended to @p defaulted.
inline bool pointFromColumns(const OperationMetadata& metadata,
                             const std::map<std::string, std::string>& values,
                             const std::set<std::string>& columns,
                             ProblemPoint& point,
                             std::vector<std::string>& defaulted,
                             std::string& problem)
{
    point.clear();
    for(const auto& parameter : metadata.parameters)
    {
        const auto found = values.find(parameter.name);
        if(found == values.end() || found->second.empty())
        {
            const auto fallback = metadata.modelShapeDefaults.find(parameter.name);
            if(columns.count(parameter.name) == 0 && fallback != metadata.modelShapeDefaults.end())
            {
                point[parameter.name] = fallback->second;
                defaulted.push_back(parameter.name);
                continue;
            }
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
/// A CSV with `q.<parameter>` (or bare `<parameter>`) columns; an optional `op` column selects
/// rows and an optional `name` column becomes the entry's origin. Points are not filtered by
/// any oracle; admission is the caller's.
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
    const std::set<std::string> columns(header.begin(), header.end());

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
        std::vector<std::string> defaulted;
        if(!detail::pointFromColumns(metadata, values, columns, entry.point, defaulted, problem))
        {
            ++report.unusable;
            if(report.firstProblem.empty())
            {
                report.firstProblem = problem;
            }
            continue;
        }
        for(const auto& parameter : defaulted)
        {
            ++report.defaulted[parameter];
        }

        const auto name = values.find("name");
        entry.source = "model";
        entry.origin
            = path.filename().string() + " "
              + (name == values.end() || name->second.empty() ? "row " + std::to_string(report.rows)
                                                              : name->second);
        entry.regime = regimeLabel(metadata, entry.point);
        entries.push_back(std::move(entry));
    }
    return entries;
}

} // namespace hipdnn_corpus_gen
