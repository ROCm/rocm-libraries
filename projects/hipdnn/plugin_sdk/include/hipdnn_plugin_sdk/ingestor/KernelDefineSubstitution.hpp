// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>

/// @file KernelDefineSubstitution.hpp
/// @brief Binds a `hiprtc_file` kernel's own metadata into its compile-time defines.
///
/// A `kernel_source` of kind `hiprtc_file` carries `defines`, a flat string->string map
/// emitted as `-D<name>=<value>`. A value may contain `$kernel.<field>` tokens, which are
/// replaced with the rendered value of that KMD field for the kernel being prepared. That
/// is the whole feature: two kernels sharing one source file and differing only in
/// `metadata.dtype` compile to two binaries with no native code and no rebuild.
///
/// **This is literal token replacement, not an expression language.** Operators,
/// arithmetic, comparisons, conditionals, calls, defaults, nesting and `$graph` are not
/// supported, and every one of them is an error rather than a half-working substitution --
/// `"$kernel.a + 1"` must not quietly become `"2 + 1"` in a `-D` flag. Anything derived or
/// conditional belongs in the pack's dispatch handler, which already builds
/// `KernelCompileOptions` by hand (`PointwiseNative.cpp`'s `elementTypeFor(kernel)` is a
/// dtype conditional) and which a new kernel variant reuses unchanged.
///
/// Two entry points, because the two checks happen at different times against different
/// inputs:
/// - `validateKernelDefineTemplate` runs once at descriptor-set resolution, against the
///   engine's KMD. It answers "could this ever resolve?" -- the kernel's own metadata is
///   not yet completed there, so a field the kernel omits and the KMD defaults must still
///   validate. Failure drops the pack with a log, matching every other load-time failure.
/// - `substituteKernelDefine` runs at `prepare()`, against the completed metadata, and
///   produces the flag text.
namespace hipdnn_plugin_sdk::ingestor
{

/// Prefix of every bound token. Nothing else may follow a `$`.
inline constexpr std::string_view K_KERNEL_DEFINE_TOKEN_PREFIX = "$kernel.";

namespace detail
{

inline bool isDefineIdentifierStart(char character)
{
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
           || character == '_';
}

inline bool isDefineIdentifierChar(char character)
{
    return isDefineIdentifierStart(character) || (character >= '0' && character <= '9');
}

/// Characters that only ever appear in a define value because someone is writing an
/// expression: arithmetic, comparison, logic, a conditional, a call, or a default. They
/// are refused in a value that binds a token, which is the only place they could be
/// mistaken for something this substituter evaluates. A define with no token is opaque
/// text and is passed through untouched, so `-DLIMIT=-1` remains authorable.
inline bool isExpressionOperatorChar(char character)
{
    switch(character)
    {
    case '+':
    case '-':
    case '*':
    case '/':
    case '%':
    case '=':
    case '<':
    case '>':
    case '!':
    case '&':
    case '|':
    case '^':
    case '~':
    case '?':
    case ':':
    case '(':
    case ')':
        return true;
    default:
        return false;
    }
}

/// Walks @p templateText once, handing each `$kernel.<field>` token to @p onField, which
/// appends the field's replacement to @p out or fills @p error and returns false. Literal
/// runs are copied verbatim.
template <typename OnField>
inline bool scanKernelDefineTemplate(std::string_view templateText,
                                     std::string& out,
                                     std::string& error,
                                     OnField&& onField)
{
    out.clear();
    if(templateText.find('$') == std::string_view::npos)
    {
        // Byte-identical passthrough, and the only path a define without a token takes:
        // an operator character here is ordinary text, not an expression.
        out.assign(templateText);
        return true;
    }
    out.reserve(templateText.size());

    for(size_t position = 0; position < templateText.size();)
    {
        const char character = templateText[position];
        if(isExpressionOperatorChar(character))
        {
            error = "value '" + std::string(templateText) + "' uses '" + character
                    + "', but a bound define is literal token replacement -- it evaluates "
                      "no operators, comparisons, conditionals, calls or defaults. Put "
                      "derived values in the pack's dispatch handler";
            return false;
        }
        if(character != '$')
        {
            out.push_back(character);
            ++position;
            continue;
        }

        if(templateText.compare(
               position, K_KERNEL_DEFINE_TOKEN_PREFIX.size(), K_KERNEL_DEFINE_TOKEN_PREFIX)
           != 0)
        {
            error = "value '" + std::string(templateText)
                    + "' contains a '$' that does not begin a '"
                    + std::string(K_KERNEL_DEFINE_TOKEN_PREFIX)
                    + "<field>' token; no other binding source exists";
            return false;
        }

        const size_t nameBegin = position + K_KERNEL_DEFINE_TOKEN_PREFIX.size();
        size_t nameEnd = nameBegin;
        while(nameEnd < templateText.size() && isDefineIdentifierChar(templateText[nameEnd]))
        {
            ++nameEnd;
        }
        if(nameEnd == nameBegin || !isDefineIdentifierStart(templateText[nameBegin]))
        {
            error = "value '" + std::string(templateText) + "' has a '"
                    + std::string(K_KERNEL_DEFINE_TOKEN_PREFIX) + "' with no field name after it";
            return false;
        }

        const std::string_view fieldName = templateText.substr(nameBegin, nameEnd - nameBegin);
        if(!onField(fieldName, out, error))
        {
            return false;
        }
        // Single pass: a replacement that itself contains a '$' is literal text, never
        // rescanned. Nesting is not a feature.
        position = nameEnd;
    }
    return true;
}

} // namespace detail

/// Renders one metadata value as the text of a `-D` flag.
///
/// The spelling of each accepted type is pinned, because two kernel variants that differ
/// only in a bound field and render the same flag text compile to the same binary and
/// silently become one kernel.
///
/// `FLOAT` and `INT_LIST` are refused: there is no single spelling of either. `1.0` and
/// `1` are different types in device code and `std::to_chars(1.0)` and Python's
/// `repr(1.0)` disagree about which one to emit; a list has no separator that is right for
/// every macro. A float or a list that a kernel genuinely needs goes through the pack's
/// dispatch handler.
inline bool renderMetadataValueForDefine(const MetadataValue& value,
                                         std::string_view fieldName,
                                         std::string& out,
                                         std::string& error)
{
    switch(metadataTypeOf(value))
    {
    case MetadataType::BOOL:
        out += std::get<bool>(value) ? '1' : '0';
        return true;
    case MetadataType::INT:
        out += std::to_string(std::get<int64_t>(value));
        return true;
    case MetadataType::STRING:
        out += std::get<std::string>(value);
        return true;
    case MetadataType::FLOAT:
        error = "metadata field '" + std::string(fieldName)
                + "' is a float, which cannot be bound into a define: '1' and '1.0' are "
                  "different types in device code and no spelling is right for both. Emit "
                  "it from the pack's dispatch handler instead";
        return false;
    case MetadataType::INT_LIST:
        error = "metadata field '" + std::string(fieldName)
                + "' is an int_list, which cannot be bound into a define: no separator is "
                  "right for every macro. Emit it from the pack's dispatch handler instead";
        return false;
    default:
        error = "metadata field '" + std::string(fieldName) + "' has an unknown metadata type";
        return false;
    }
}

/// Resolves every `$kernel.<field>` in @p templateText against @p metadata, which must be
/// the kernel's *completed* metadata -- post-`completeMetadata`, so a field the descriptor
/// omitted and the KMD defaulted resolves rather than reading as undeclared.
///
/// On success @p out holds the flag text and @p error is untouched; on failure @p error
/// explains what the author must change and @p out is unspecified.
inline bool substituteKernelDefine(std::string_view templateText,
                                   const MetadataValues& metadata,
                                   std::string& out,
                                   std::string& error)
{
    return detail::scanKernelDefineTemplate(
        templateText,
        out,
        error,
        [&metadata,
         templateText](std::string_view fieldName, std::string& rendered, std::string& fieldError) {
            const auto found = metadata.find(std::string(fieldName));
            if(found == metadata.end())
            {
                fieldError = "value '" + std::string(templateText) + "' names metadata field '"
                             + std::string(fieldName) + "', which this kernel does not carry";
                return false;
            }
            return renderMetadataValueForDefine(found->second, fieldName, rendered, fieldError);
        });
}

/// Checks that @p templateText could resolve for any kernel of an engine whose metadata
/// schema is @p schema: every token names a declared field, and every such field has a
/// type this substituter can render.
///
/// Deliberately schema-level, not value-level. It runs at set resolution, where kernel
/// metadata has been type-checked but not yet completed with KMD defaults, so asking for
/// the value would reject a legal descriptor that omits a defaulted field.
inline bool validateKernelDefineTemplate(std::string_view templateText,
                                         const MetadataSchema& schema,
                                         std::string& error)
{
    std::string ignored;
    return detail::scanKernelDefineTemplate(
        templateText,
        ignored,
        error,
        [&schema, templateText](
            std::string_view fieldName, std::string& /*rendered*/, std::string& fieldError) {
            for(const auto& field : schema.fields)
            {
                if(field.name != fieldName)
                {
                    continue;
                }
                if(field.type == MetadataType::FLOAT || field.type == MetadataType::INT_LIST)
                {
                    // Reuses the value-level wording by rendering a representative value,
                    // so the load-time message and the prepare-time message agree.
                    const MetadataValue representative
                        = field.type == MetadataType::FLOAT ? MetadataValue{0.0}
                                                            : MetadataValue{std::vector<int64_t>{}};
                    std::string ignoredText;
                    return renderMetadataValueForDefine(
                        representative, fieldName, ignoredText, fieldError);
                }
                return true;
            }
            fieldError = "value '" + std::string(templateText) + "' names metadata field '"
                         + std::string(fieldName) + "', which metadata schema '" + schema.name
                         + "' does not declare";
            return false;
        });
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
