.. meta::
  :description: Learn about the methods and classes for knobs in hipDNN.
  :keywords: hipDNN, ROCm, knobs, API

.. _knob-api:

**************************
hipDNN knobs API reference
**************************

Use this article as a reference for the classes and methods for knobs in hipDNN.

``Knob`` class
==============

Describes metadata for an available knob.

**Key methods**:

- ``const std::string& knobId()``: Get the knob identifier.
- ``const std::string& description()``: Get a human-readable description.
- ``bool isDeprecated()``: Check if a knob is deprecated.
- ``KnobValueType valueType()``: Get value type (``INT64``, ``FLOAT64``, or ``STRING``).
- ``const KnobValueVariant& defaultValue()``: Get the default value as a variant.
- ``const IConstraint* constraint()``: Get the constraint validator.
- ``Error validate(const KnobSetting& setting)``: Validate a setting.

``KnobSetting`` class
=====================

Represents a knob value setting to apply.

**Constructors**:

.. code:: cpp

  KnobSetting(std::string knobId, KnobValueVariant value);
  template <typename T> KnobSetting(std::string knobId, const T& value);

**Key methods**:

- ``const std::string& knobId()``: Get the knob identifier.
- ``const KnobValueVariant& value()``: Get the knob value.
- ``template <typename T> void setValue(const T& value)``: Update the knob value.

Graph methods
=============

**Querying knobs**:

.. code:: cpp

  Error get_knobs_for_engine(int64_t engineId, std::vector<Knob>& knobs) const;
  Error get_knob_lookup_for_engine(int64_t engineId, std::unordered_map<std::string, Knob>& knobs) const;

**Setting knobs**:

.. code:: cpp

  Error create_execution_plan_ext(int64_t engineId, const std::vector<KnobSetting>& settings);

Constraint classes
==================

**Base interface**:

.. code:: cpp

  class IConstraint {
      virtual Error validateKnobSetting(const KnobSetting& setting) const = 0;
      virtual std::string toString() const = 0;
  };

**Implementations**:

- ``IntConstraint(int64_t minValue, int64_t maxValue, int64_t step, std::unordered_set<int64_t> validValues)``
- ``FloatConstraint(double minValue, double maxValue)``
- ``StringConstraint(int32_t maxLength, std::unordered_set<std::string> validValues)``
- ``EmptyConstraint()``: No constraints

Type definitions
================

.. code:: cpp

  using KnobValueVariant = std::variant<int64_t, double, std::string>;
  typedef std::string KnobType_t;

  enum class KnobValueType {
      NOT_SET = 0,
      INT64 = 1,
      FLOAT64 = 2,
      STRING = 3,
  };
