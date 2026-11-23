.. :meta::
  :description: Learn about the hipDNN backend heuristic type C API.
  :keywords: hipDNN, ROCm, API

.. _backend-api-heuristic-type:

***********************************
hipDNN backend heuristic type C API
***********************************

.. doxygenfile:: HipdnnBackendHeuristicType.h

Engine prediction queries
=========================

Predictions are read with ``hipdnnBackendGetAttribute``; the descriptor queried
states the kind. ``HIPDNN_ATTR_ENGINE_PREDICTION_EXT`` on an engine descriptor
estimates the engine's normal execution with tuning disabled, using graph, device,
and constraint features without enumerating configurations.
``HIPDNN_ATTR_ENGINECFG_PREDICTION_EXT`` on an engine config descriptor scores the
executable configuration its knob settings describe and owns the engine ID and
those settings. Neither query benchmarks GPU work. Both return one
``HIPDNN_TYPE_FLATBUFFER_DATA_STRUCT_EXT`` whose root is the
``hipdnn_flatbuffers_sdk.data_objects.EnginePrediction`` table.

Every prediction is in one registered *ranking metric*: ``tflops`` (calibrated
throughput in TFLOPS, higher is better) or ``time`` (predicted execution time in
milliseconds, lower is better). The query names it with
``HIPDNN_ATTR_ENGINE_PREDICTION_METRIC_EXT`` on the engine descriptor or
``HIPDNN_ATTR_ENGINECFG_RANKING_METRIC_EXT`` on the engine config descriptor, both
``HIPDNN_TYPE_CHAR`` strings; unset means ``tflops``, and an unregistered name is
rejected with ``HIPDNN_STATUS_BAD_PARAM``. The returned table's ``metric`` is always
the requested one and its ``value`` is in that metric's units. An engine answers
only in the requested metric: with no model for it, the answer is ``UNAVAILABLE``,
never a value in another metric.

``AVAILABLE`` provides a calibrated physical value, finite and strictly positive in
every metric. ``UNAVAILABLE`` means no usable prediction is
supplied; ``INVALID`` reports an incompatible or malformed prediction. Neither
status removes an otherwise applicable engine.
Setting ``HIPDNN_ATTR_ENGINE_PREDICTION_EVALUATE_EXT`` or
``HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT`` to zero returns the model binding
and supported features without evaluating the model; describing each metric at each
kind this way lists the predictions an engine can answer, which is what the
generation-tool helper ``hipdnn_frontend::detail::getPredictionCapabilities`` does.

Mode A ranks engines by their engine predictions alone. Mode B prefers
configuration predictions and uses engine predictions when necessary. Both rank in
the heuristic descriptor's ranking metric, best first in that metric's direction,
retain unscored engines after scored engines, and decline if no score is usable. The
metric is ``HIPDNN_ATTR_ENGINEHEUR_RANKING_METRIC_EXT`` (``HIPDNN_TYPE_CHAR``) unless
the ``HIPDNN_HEUR_RANKING_METRIC`` environment variable overrides it, and ``tflops``
when neither is set. Every configuration the heuristic returns carries that metric,
so the chosen engine also picks its kernel by it at plan build. The frontend
sets the metric from ``Graph::set_ranking_metric()``.

Heuristic modes
===============

``HIPDNN_ATTR_ENGINEHEUR_MODE`` (``HIPDNN_TYPE_HEUR_MODE``) is required before the
heuristic descriptor is finalized. ``hipdnnBackendHeurMode_t`` has these values:

* ``HIPDNN_HEUR_MODE_FALLBACK`` (0): model-free static ordering.
* ``HIPDNN_HEUR_MODE_A`` (1): ranks by engine predictions alone.
* ``HIPDNN_HEUR_MODE_B`` (2): prefers configuration predictions and uses engine
  predictions when necessary.
* ``HIPDNN_HEUR_MODE_INSTANT``: alias of ``HIPDNN_HEUR_MODE_A``, as cuDNN's
  ``CUDNN_HEUR_MODE_A`` aliases ``INSTANT``.

The backend resolves the ordered policy list in this order of precedence: the
``HIPDNN_HEUR_POLICY_ORDER`` environment variable, then the
``HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT`` attribute, then the list the mode selects:

* ``HIPDNN_HEUR_MODE_FALLBACK``: ``SelectionHeuristic::Config``, then
  ``SelectionHeuristic::StaticOrdering``.
* ``HIPDNN_HEUR_MODE_A`` / ``HIPDNN_HEUR_MODE_INSTANT``: ``SelectionHeuristic::Config``,
  ``SelectionHeuristic::ModeA``, then ``SelectionHeuristic::StaticOrdering``.
* ``HIPDNN_HEUR_MODE_B``: ``SelectionHeuristic::Config``, ``SelectionHeuristic::ModeB``,
  then ``SelectionHeuristic::StaticOrdering``.

There is no mode-independent built-in list. ``SelectionHeuristic::Config`` runs first,
so ``HIPDNN_HEUR_CONFIG_PATH`` rules and the exact-match autotune cache win, and
``SelectionHeuristic::StaticOrdering`` runs last, so model-free selection still runs
when a prediction policy declines. The frontend maps ``HeuristicMode::A``, ``B`` and
``FALLBACK`` to ``HIPDNN_HEUR_MODE_A``, ``HIPDNN_HEUR_MODE_B`` and
``HIPDNN_HEUR_MODE_FALLBACK`` and does not set the policy-order attribute itself.
Its default mode list is ``{HeuristicMode::A}``, so default engine selection ranks by
engine predictions where an applicable engine has one.
