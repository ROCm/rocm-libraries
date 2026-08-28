// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>
#include <hipdnn_frontend/knob/Knob.hpp>
#include <hipdnn_frontend/knob/KnobConstraint.hpp>
#include <hipdnn_frontend/knob/KnobSetting.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include <string>

namespace nb = nanobind;
using namespace hipdnn_frontend;

void autotuneBindings(nb::module_& m)
{
    // Bind TuneMode enum
    nb::enum_<TuneMode>(m, "TuneMode")
        .value("STANDARD", TuneMode::STANDARD)
        .value("EXHAUSTIVE", TuneMode::EXHAUSTIVE);

    // Bind AutotuneStrategy enum
    nb::enum_<AutotuneStrategy>(m, "AutotuneStrategy")
        .value("FIXED_AVERAGE", AutotuneStrategy::FIXED_AVERAGE)
        .value("RUN_UNTIL_STABLE", AutotuneStrategy::RUN_UNTIL_STABLE);

    // Bind PrimingFailurePolicy enum
    nb::enum_<PrimingFailurePolicy>(m, "PrimingFailurePolicy")
        .value("ABORT_ON_PRIMING_FAILURE", PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE)
        .value("BENCHMARK_UNPRIMED", PrimingFailurePolicy::BENCHMARK_UNPRIMED);

    // Bind AutotuneCacheWriteOutcome enum
    nb::enum_<AutotuneCacheWriteOutcome>(m, "AutotuneCacheWriteOutcome")
        .value("NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE",
               AutotuneCacheWriteOutcome::NOT_ATTEMPTED_NO_SUCCESSFUL_ENGINE)
        .value("WRITTEN", AutotuneCacheWriteOutcome::WRITTEN)
        .value("DECLINED_DISABLED", AutotuneCacheWriteOutcome::DECLINED_DISABLED)
        .value("DECLINED_UNKEYABLE", AutotuneCacheWriteOutcome::DECLINED_UNKEYABLE)
        .value("UNCHANGED", AutotuneCacheWriteOutcome::UNCHANGED)
        .value("NOT_ATTEMPTED_PARTIAL_SWEEP",
               AutotuneCacheWriteOutcome::NOT_ATTEMPTED_PARTIAL_SWEEP);

    // Bind the concrete knob constraints. The C++ ConstraintKind discriminator exists
    // so -fno-rtti callers can downcast; Python gets the concrete type instead and
    // discriminates with isinstance(). Instances are copies, so they outlive their Knob.
    nb::class_<IntConstraint>(m, "IntConstraint")
        .def_prop_ro("min_value", &IntConstraint::getMinValue)
        .def_prop_ro("max_value", &IntConstraint::getMaxValue)
        .def_prop_ro("step", &IntConstraint::getStep)
        .def_prop_ro("valid_values", &IntConstraint::getValidValues)
        .def("__repr__", &IntConstraint::toString);

    nb::class_<FloatConstraint>(m, "FloatConstraint")
        .def_prop_ro("min_value", &FloatConstraint::getMinValue)
        .def_prop_ro("max_value", &FloatConstraint::getMaxValue)
        .def("__repr__", &FloatConstraint::toString);

    nb::class_<StringConstraint>(m, "StringConstraint")
        .def_prop_ro("max_length", &StringConstraint::getMaxLength)
        .def_prop_ro("valid_values", &StringConstraint::getValidValues)
        .def("__repr__", &StringConstraint::toString);

    // Bind KnobSetting: (knob id, value) pair accepted by add_engine()
    nb::class_<KnobSetting>(m, "KnobSetting")
        .def(nb::init<std::string, KnobValueVariant>(), nb::arg("knob_id"), nb::arg("value"))
        .def_prop_ro("knob_id", &KnobSetting::knobId)
        .def_prop_ro("value", &KnobSetting::value)
        .def(
            "__eq__",
            [](const KnobSetting& self, const KnobSetting& other) { return self == other; },
            nb::is_operator())
        .def("__repr__", &KnobSetting::toString);

    // Bind Knob: read-only engine knob description; instances come from the library only
    nb::class_<Knob>(m, "Knob")
        .def_prop_ro("knob_id", &Knob::knobId)
        .def_prop_ro("description", &Knob::description)
        .def_prop_ro("is_deprecated", &Knob::isDeprecated)
        .def_prop_ro("value_type", &Knob::valueType)
        .def_prop_ro("default_value", &Knob::defaultValue)
        .def("validate",
             &Knob::validate,
             nb::arg("setting"),
             "Validate a KnobSetting against this knob's constraints.")
        .def_prop_ro(
            "constraint",
            [](const Knob& knob) -> nb::object {
                const auto* constraint = knob.constraint();
                if(constraint == nullptr)
                {
                    return nb::none();
                }
                switch(constraint->kind())
                {
                case ConstraintKind::INT:
                    return nb::cast(*static_cast<const IntConstraint*>(constraint));
                case ConstraintKind::FLOAT:
                    return nb::cast(*static_cast<const FloatConstraint*>(constraint));
                case ConstraintKind::STRING:
                    return nb::cast(*static_cast<const StringConstraint*>(constraint));
                case ConstraintKind::EMPTY:
                    break;
                }
                return nb::none();
            },
            "IntConstraint, FloatConstraint or StringConstraint describing the legal "
            "values of this knob, so a sweep axis can be built from it, or None when the "
            "knob is unconstrained.")
        .def("__repr__", &Knob::toString);

    // Bind EngineConfigInfo: engine metadata returned by Graph.get_engine_configs()
    nb::class_<EngineConfigInfo>(m, "EngineConfigInfo")
        .def(nb::init<>())
        .def_rw("engine_id", &EngineConfigInfo::engineId)
        .def_rw("engine_name", &EngineConfigInfo::engineName)
        .def_rw("knobs", &EngineConfigInfo::knobs)
        .def_rw("supports_exhaustive", &EngineConfigInfo::supportsExhaustive)
        .def_rw("estimated_workspace_size", &EngineConfigInfo::estimatedWorkspaceSize);

    // Bind EngineVariant: explicit (engine id, knob settings) pair for add_engine_variants()
    nb::class_<EngineVariant>(m, "EngineVariant")
        .def(nb::init<>())
        .def_rw("engine_id", &EngineVariant::engineId)
        .def_rw("knob_settings", &EngineVariant::knobSettings);

    // Bind KnobSweepAxis: one knob axis of an add_engine_sweep() Cartesian product
    nb::class_<KnobSweepAxis>(m, "KnobSweepAxis")
        .def(nb::init<>())
        .def_rw("knob_id", &KnobSweepAxis::knobId)
        .def_rw("values", &KnobSweepAxis::values);

    // Bind EngineSweepSpec: per-engine sweep specification for add_engine_sweep()
    nb::class_<EngineSweepSpec>(m, "EngineSweepSpec")
        .def(nb::init<>())
        .def_rw("engine_id", &EngineSweepSpec::engineId)
        .def_rw("axes", &EngineSweepSpec::axes)
        .def_rw("fixed_settings", &EngineSweepSpec::fixedSettings);

    // Bind AutotuneConfig: tuning knobs for Graph.autotune()
    nb::class_<AutotuneConfig>(
        m,
        "AutotuneConfig",
        "Tuning parameters for Graph.autotune().\n\n"
        "The C++ AutotuneConfig::rankingFn hook is deliberately not bound: a Python "
        "callback would have to reacquire the GIL inside the benchmarking loop, and "
        "the returned AutotuneResult list can be sorted in Python instead. Every other "
        "field of the C++ struct is exposed.")
        .def(nb::init<>())
        .def_rw("mode", &AutotuneConfig::mode)
        .def_rw("strategy", &AutotuneConfig::strategy)
        .def_rw("warmup_iterations", &AutotuneConfig::warmupIterations)
        .def_rw("timed_iterations", &AutotuneConfig::timedIterations)
        .def_rw("max_iterations", &AutotuneConfig::maxIterations)
        .def_rw("window_size", &AutotuneConfig::windowSize)
        .def_rw("stability_threshold", &AutotuneConfig::stabilityThreshold)
        .def_rw("engine_id_filter", &AutotuneConfig::engineIdFilter)
        .def_rw("priming_failure_policy", &AutotuneConfig::primingFailurePolicy);

    // Bind AutotuneResult: per-candidate benchmarking result produced by Graph.autotune()
    nb::class_<AutotuneResult>(m, "AutotuneResult")
        .def(nb::init<>())
        .def_ro("engine_id", &AutotuneResult::engineId)
        .def_ro("engine_name", &AutotuneResult::engineName)
        .def_ro("knob_settings", &AutotuneResult::knobSettings)
        .def_ro("min_time_ms", &AutotuneResult::minTimeMs)
        .def_ro("avg_time_ms", &AutotuneResult::avgTimeMs)
        .def_ro("stddev_ms", &AutotuneResult::stddevMs)
        .def_ro("robust_time_ms", &AutotuneResult::robustTimeMs)
        .def_ro("iterations_run", &AutotuneResult::iterationsRun)
        .def_ro("converged", &AutotuneResult::converged)
        .def_ro("rank", &AutotuneResult::rank)
        .def_ro("succeeded", &AutotuneResult::succeeded)
        // The two axes that decide whether a sweep is persistable. Exposed so a caller
        // that got NOT_ATTEMPTED_PARTIAL_SWEEP can find which engine caused it: it is the
        // one with excluded_by_caller set.
        .def_ro("benchmarked", &AutotuneResult::benchmarked)
        .def_ro("excluded_by_caller", &AutotuneResult::excludedByCaller)
        .def_ro("error_message", &AutotuneResult::errorMessage)
        .def_ro("workspace_size", &AutotuneResult::workspaceSize)
        .def_ro("estimated_workspace_size", &AutotuneResult::estimatedWorkspaceSize)
        .def_ro("compiled_plan_index", &AutotuneResult::compiledPlanIndex)
        .def_ro("mode_used", &AutotuneResult::modeUsed)
        .def_ro("supports_exhaustive", &AutotuneResult::supportsExhaustive)
        .def_ro("ran_exhaustive", &AutotuneResult::ranExhaustive)
        .def_ro("exhaustive_not_run_reason", &AutotuneResult::exhaustiveNotRunReason)
        .def_ro("strategy_used", &AutotuneResult::strategyUsed);

    // Bind AutotuneStorageConfig: optional JSON heuristic-config output for Graph.autotune()
    nb::class_<AutotuneStorageConfig>(m, "AutotuneStorageConfig")
        .def(nb::init<>())
        .def_rw("file_path", &AutotuneStorageConfig::filePath)
        .def_rw("delete_all_existing_file_content",
                &AutotuneStorageConfig::deleteAllExistingFileContent);
}
