// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file note_triage.h
 * @brief Per-note triage for the cuDNN shim's select/deselect note filters.
 *
 * Dispositions are documented in docs/KNOWN_DIVERGENCES.md; keep it in sync.
 */

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_compatibility/cudnn/cudnn_frontend_utils.h>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

enum class NoteAction
{
    NO_OP, // silently accepted
    WARN, // logged and otherwise ignored
    MAP, // honored by filtering engines (behavior metadata or determinism allowlist)
    RECORD_ERROR, // recorded; returned by the next validating graph call
};

inline NoteAction triageSelect(NumericalNote_t note)
{
    switch(note)
    {
    case NumericalNote_t::NOT_SET:
        return NoteAction::NO_OP;
    case NumericalNote_t::TENSOR_CORE:
    case NumericalNote_t::DOWN_CONVERT_INPUTS:
    case NumericalNote_t::REDUCED_PRECISION_REDUCTION:
    case NumericalNote_t::FFT:
    case NumericalNote_t::NONDETERMINISTIC:
    case NumericalNote_t::WINOGRAD:
    case NumericalNote_t::WINOGRAD_TILE_4x4:
    case NumericalNote_t::WINOGRAD_TILE_6x6:
    case NumericalNote_t::WINOGRAD_TILE_13x13:
        return NoteAction::WARN;
    // hipDNN cannot prove a NaN-propagation guarantee for any engine.
    case NumericalNote_t::STRICT_NAN_PROP:
    // Out-of-range values fail closed.
    default:
        return NoteAction::RECORD_ERROR;
    }
}

inline NoteAction triageDeselect(NumericalNote_t note)
{
    switch(note)
    {
    case NumericalNote_t::NOT_SET:
        return NoteAction::NO_OP;
    case NumericalNote_t::TENSOR_CORE:
    case NumericalNote_t::FFT:
    case NumericalNote_t::WINOGRAD:
    case NumericalNote_t::WINOGRAD_TILE_4x4:
    case NumericalNote_t::WINOGRAD_TILE_6x6:
    case NumericalNote_t::WINOGRAD_TILE_13x13:
    case NumericalNote_t::STRICT_NAN_PROP:
        return NoteAction::WARN;
    case NumericalNote_t::NONDETERMINISTIC:
        return NoteAction::MAP;
    // These exclusions protect precision, so ignoring them could silently lose accuracy.
    case NumericalNote_t::DOWN_CONVERT_INPUTS:
    case NumericalNote_t::REDUCED_PRECISION_REDUCTION:
    // Out-of-range values fail closed.
    default:
        return NoteAction::RECORD_ERROR;
    }
}

inline NoteAction triageSelect(BehaviorNote_t note)
{
    if(note == BehaviorNote_t::NOT_SET)
    {
        return NoteAction::NO_OP;
    }
    // Selecting a note no hipDNN engine can report is unsatisfiable.
    return hipdnn_frontend::isKnownBehaviorNote(note) ? NoteAction::MAP : NoteAction::RECORD_ERROR;
}

inline NoteAction triageDeselect(BehaviorNote_t note)
{
    if(note == BehaviorNote_t::NOT_SET)
    {
        return NoteAction::NO_OP;
    }
    // Excluding a note no hipDNN engine can report excludes nothing.
    return hipdnn_frontend::isKnownBehaviorNote(note) ? NoteAction::MAP : NoteAction::WARN;
}

template <typename Note>
std::string noteName(Note note)
{
    std::string name = hipdnn_frontend::to_string(note);
    if(name == "unknown")
    {
        name += " (" + std::to_string(static_cast<int32_t>(note)) + ")";
    }
    return name;
}

inline std::string noteMessage(const char* method, NumericalNote_t note, NoteAction action)
{
    const auto call = std::string{method} + "(" + noteName(note) + ")";
    switch(action)
    {
    case NoteAction::WARN:
        return "Ignoring " + call + "; hipDNN exposes no per-plan numerical-note metadata.";
    case NoteAction::MAP:
        return call + " restricts plans to hipDNN engines that guarantee deterministic results.";
    case NoteAction::RECORD_ERROR:
        return call + " cannot be honored by this shim; refusing to run rather than ignore it";
    case NoteAction::NO_OP:
    default:
        break;
    }
    return {};
}

inline std::string noteMessage(const char* method, BehaviorNote_t note, NoteAction action)
{
    const auto call = std::string{method} + "(" + noteName(note) + ")";
    switch(action)
    {
    case NoteAction::WARN:
        return "Ignoring " + call + "; hipDNN engines do not report this cuDNN behavior note.";
    case NoteAction::MAP:
        return call + " will filter hipDNN engines by behavior metadata.";
    case NoteAction::RECORD_ERROR:
        return call
               + " requests a behavior note that no hipDNN engine reports; the request "
                 "cannot be satisfied";
    case NoteAction::NO_OP:
    default:
        break;
    }
    return {};
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
