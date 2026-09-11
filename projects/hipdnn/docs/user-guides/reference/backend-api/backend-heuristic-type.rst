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

``AVAILABLE`` provides calibrated physical TFLOPS. ``UNAVAILABLE`` means no usable
prediction is supplied; ``INVALID`` reports an incompatible or malformed
prediction. Neither status removes an otherwise applicable engine.
Setting ``HIPDNN_ATTR_ENGINE_PREDICTION_EVALUATE_EXT`` or
``HIPDNN_ATTR_ENGINECFG_PREDICTION_EVALUATE_EXT`` to zero returns the model binding
and supported features without evaluating the model.

Mode A uses engine predictions. Mode B prefers configuration predictions and
uses engine predictions when necessary. Both retain unscored engines after
scored engines and decline if no score is usable. The frontend requests them as
the ``SelectionHeuristic::ModeA`` / ``SelectionHeuristic::ModeB`` entries of the
ordered policy list, bracketed by ``SelectionHeuristic::Config`` and
``SelectionHeuristic::StaticOrdering``, so model-free selection still runs when a
prediction policy declines.
