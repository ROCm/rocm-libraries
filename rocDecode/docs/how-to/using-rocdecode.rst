.. meta::
  :description: Using rocDecode
  :keywords: parse video, parse, decode, video decoder, video decoding, rocDecode, core APIs, AMD, ROCm

********************************************************************
Using the rocDecode core APIs
********************************************************************

rocDecode core APIs are available in the |apifolder|_ of the `rocDecode GitHub repository <https://github.com/ROCm/rocDecode>`_. 

.. note::

  The rocDecode samples use the utility classes in the |utilsfolder|_ instead of the core APIs. For information about using the utility classes, see :doc:`Using rocDecode with the FFmpeg decoder <./using-rocDecode-ffmpeg>` and :doc:`Using rocDecode with the bitstream decoder <./using-rocDecode-bitstream>`.


API overview
====================================================

All rocDecode APIs are exposed in the header files ``rocdecode.h`` and ``rocparser.h``. You can find
these files in the ``api`` folder in the rocDecode repository.

The samples use the ``RocVideoDecoder`` user class provided in ``roc_video_dec.h`` in the ``utils`` folder
of the rocDecode repository.

A video parser (defined in ``rocparser.h``) is needed to extract and decode headers from the bitstream
in order to organize the data into a structured format for the hardware decoder. The parser is critical in
video decoding, as it controls the decoding and display of a bitstream's individual frames and fields.

The parser object in ``rocparser.h`` has three main APIs:

* ``rocDecCreateVideoParser()``
* ``rocDecParseVideoData()``
* ``rocDecDestroyVideoParser()``

Create a parser object
====================================================

The ``rocDecCreateVideoParser()`` API creates a video parser object for the codec that you specify. The
API takes ``max_num_decode_surfaces``, which determines the Decoded Picture Buffer (DPB) size for
decoding. When creating a parser object, the application must register certain callback functions with
the driver, which is called from the parser during decode.


* ``pfn_sequence_callback`` is called when the parser encounters a new sequence header. The parser
  informs you of the minimum number of surfaces needed by the parser's DPB to successfully decode
  the bitstream. In addition, the caller can set additional parameters, like ``max_display_delay``, to
  control frame decoding and display.

* The ``pfn_decode_picture`` callback function is triggered when a picture is set for decoding.

* The ``pfn_display_picture`` callback function is triggered when a frame in display order is ready to be
  consumed by the caller.

* The ``pfn_get_sei_msg`` callback function is triggered when your Supplementation Enhancement
  Information (SEI) message is parsed and returned to the caller.


Parse video data
====================================================

Elementary stream video packets extracted from the de-multiplexer are fed into the parser using the
``rocDecParseVideoData()`` API.

During this call, the parser triggers the callbacks as it encounters a new sequence header, receives
compressed frame/field data ready to be decoded, or when it's ready to display a frame. If any of the
callbacks return a failure, it is propagated back to the application so the decoding can be ended
gracefully.

Query decode capabilities
====================================================

The ``rocDecGetDecoderCaps()`` API allows you to query the capabilities of the underlying hardware
video decoder. Decoder capabilities usually include supported codecs, maximum resolution, and
bit depth.


The following pseudo-code illustrates the use of this API. The application handles the error
appropriately for non-supported decoder capabilities.

.. code:: cpp

    RocdecDecodeCaps decode_caps;
    memset(&decode_caps, 0, sizeof(decode_caps));
    decode_caps.codec_type = p_video_format->codec;
    decode_caps.chroma_format = p_video_format->chroma_format;
    decode_caps.bit_depth_minus_8 = p_video_format->bit_depth_luma_minus8;

    ROCDEC_API_CALL(rocDecGetDecoderCaps(&decode_caps));

    if(!decode_caps.is_supported) {
        ROCDEC_THROW("Rocdec:: Codec not supported on this GPU: ", ROCDEC_NOT_SUPPORTED);
        return 0;
    }

    if ((p_video_format->coded_width > decode_caps.max_width) ||
        (p_video_format->coded_height > decode_caps.max_height)) {

        std::ostringstream errorString;
        errorString << std::endl
                    << "Resolution          : " << p_video_format->coded_width << "x" << p_video_format->coded_height << std::endl
                    << "Max Supported (wxh) : " << decode_caps.max_width << "x" << decode_caps.max_height << std::endl
                    << "Resolution not supported on this GPU ";

        const std::string cErr = errorString.str();
        ROCDEC_THROW(cErr, ROCDEC_NOT_SUPPORTED);
        return 0;
    }

Create a decoder
====================================================

``rocDecCreateDecoder()`` creates an instance of the hardware video decoder object and provides you
with a handle upon successful creation. Refer to the ``RocDecoderCreateInfo`` structure for information
about the parameters passed for creating the decoder. For example,
``RocDecoderCreateInfo::codec_type`` represents the codec type of the video. The decoder handle
returned by ``rocDecCreateDecoder()`` needs to be retained for the entire decode session because the
handle is passed along with the other decoding APIs. In addition, you can inform display or crop
dimensions along with this API.

Decode the frame
====================================================

After de-multiplexing and parsing, you can decode bitstream data containing a frame/field using
hardware.

Use the ``rocDecDecodeFrame()`` API to submit a new frame for hardware decoding. Underneath the
driver, the Video Acceleration API (VA-API) is used to submit compressed picture data to the driver.
The parser extracts all the necessary information from the bitstream and fills the ``RocdecPicParams``
structure that's appropriate for the codec. The high-level ``RocVideoDecoder`` class connects the parser
and decoder used for all sample applications.

The ``rocDecDecodeFrame()`` call takes the decoder handle and the pointer to the ``RocdecPicParams``
structure and initiates the video decoding using VA-API.

Query the decoding status
====================================================

After submitting a frame for decoding, you can call ``rocDecGetDecodeStatus()`` to query the decoding
status for a given frame. A structure pointer, ``RocdecDecodeStatus*``, is filled and returned.

The API inputs are:

* ``decoder_handle``: A ``RocDecoder`` handler, ``rocDecDecoderHandle``.
* ``pic_idx``: An `int` value for the ``picIdx`` for which you want a status in order to index of the picture.
* ``decode_status``: A pointer to ``RocdecDecodeStatus`` as a return value.

The API returns one of the following statuses:

* Invalid (0): Decode status is not valid.
* In Progress (1): Decoding is in progress.
* Success (2): Decoding was successful and no errors were returned.
* Error (8): The frame was corrupted, but the error was not concealed.
* Error Concealed (9): The frame was corrupted and the error was concealed.
* Displaying (10): Decode is complete, display in progress.

Prepare the decoded frame for further processing
====================================================

The decoded frames can be used for further postprocessing using ``rocDecGetVideoFrame()``. The
successful completion of ``rocDecGetVideoFrame()`` indicates that the decoding process is complete and
the device memory pointer is inter-opped into the ROCm HIP address space in order to further process
the decoded frame in device memory. The caller gets the necessary information on the output surface,
such as YUV format, dimensions, and pitch from this call. In the high-level ``RocVideoDecoder`` class, we
provide four different surface type modes for the mapped surface, as specified in
``OutputSurfaceMemoryType``.

.. code:: cpp

    typedef enum OutputSurfaceMemoryType_enum {
        OUT_SURFACE_MEM_DEV_INTERNAL = 0,      /**<  Internal interopped decoded surface memory **/
        OUT_SURFACE_MEM_DEV_COPIED = 1,        /**<  decoded output will be copied to a separate device memory **/
        OUT_SURFACE_MEM_HOST_COPIED = 2        /**<  decoded output will be copied to a separate host memory **/
        OUT_SURFACE_MEM_NOT_MAPPED = 3         /**<  decoded output is not available (interop won't be used): useful for decode only performance app*/
    } OutputSurfaceMemoryType;


If the mapped surface type is ``OUT_SURFACE_MEM_DEV_INTERNAL``, the direct pointer to the decoded
surface is provided. You need to call ``ReleaseFrame()`` (``RocVideoDecoder`` class). If the requested surface
type is ``OUT_SURFACE_MEM_DEV_COPIED`` or ``OUT_SURFACE_MEM_HOST_COPIED``, the internal
decoded frame is copied to another buffer, either in device memory or host memory. After that, it's
immediately unmapped for reuse by the ``RocVideoDecoder`` class.

Refer to the ``RocVideoDecoder`` class and
`samples <https://github.com/ROCm/rocDecode/tree/develop/samples>`_ for details on how to use
these APIs.

Reconfigure the decoder
====================================================

You can call ``rocDecReconfigureDecoder()`` to reuse a single decoder for multiple clips or when the
video resolution changes during the decode. The API currently supports resolution changes, resize
parameter changes, and target area parameter changes for the same codec without destroying an
ongoing decoder instance. This can improve performance and reduce overall latency.

The API inputs are:

* ``decoder_handle``: A ``RocDecoder`` handler, ``rocDecDecoderHandle``.
* ``reconfig_params``: You need to specify the parameters for the changes in
  ``RocdecReconfigureDecoderInfo``. The width and height used for reconfiguration cannot exceed the
  values set for ``max_width`` and ``max_height``, defined in ``RocDecoderCreateInfo``. If you need to
  change these values, you have to destroy and recreate the session.

.. note::

  You need to call ``rocDecReconfigureDecoder()`` during ``RocdecParserParams::pfn_sequence_callback``.

Destroy the decoder
====================================================

You need to call the ``rocDecDestroyDecoder()`` to destroy the session and free up resources.

The API input is:

* ``decoder_handle``: A ``RocDecoder`` handler, ``rocDecDecoderHandle``.

The API returns a ``RocdecDecodeStatus`` value.

Destroy the parser
====================================================

You need to call ``rocDecDestroyVideoParser()`` to destroy the parser object and free up all allocated
resources at the end of video decoding.


.. |apifolder| replace:: ``api`` folder
.. _apifolder: https://github.com/ROCm/rocDecode/tree/develop/api

.. |utilsfolder| replace:: ``utils`` folder
.. _utilsfolder: https://github.com/ROCm/rocDecode/tree/develop/utils
