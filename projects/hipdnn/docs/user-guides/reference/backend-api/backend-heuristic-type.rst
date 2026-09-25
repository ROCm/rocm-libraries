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

``AVAILABLE`` provides a calibrated physical value, finite, non-negative for
``tflops`` and positive for ``time``. ``UNAVAILABLE`` means no usable prediction is
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
requests the modes as the ``SelectionHeuristic::ModeA`` /
``SelectionHeuristic::ModeB`` entries of the ordered policy list, bracketed by
``SelectionHeuristic::Config`` and ``SelectionHeuristic::StaticOrdering``, so
model-free selection still runs when a prediction policy declines, and sets the
metric from ``Graph::set_ranking_metric()``.
