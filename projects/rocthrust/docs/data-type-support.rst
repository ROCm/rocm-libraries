.. meta::
  :description: rocThrust API data type support
  :keywords: rocThrust, ROCm, API, reference, data type, support

.. _data-type-support:

******************************************
Data type support
******************************************

rocThrust supports user-defined custom types as long as an interface for them is provided.

rocThrust and Thrust both support the following fundamental types:

* ``int8``
* ``int16``
* ``int32``
* ``int64``
* ``float``
* ``double``


Both rocThrust and Thrust also support ``half`` and ``bfloat16``. However, the host-side HIP implementations of these types are missing some functionality. Because of this, ``half`` and ``bfloat16`` should be used only as storage types to be passed between functions.

