# Copyright (c) 2018 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##
# @file types.py
# @brief File containing various user defined types used in rocAL

# RocalStatus
from rocal_pybind.types import OK
from rocal_pybind.types import CONTEXT_INVALID
from rocal_pybind.types import RUNTIME_ERROR
from rocal_pybind.types import UPDATE_PARAMETER_FAILED
from rocal_pybind.types import INVALID_PARAMETER_TYPE

#  RocalProcessMode
from rocal_pybind.types import GPU
from rocal_pybind.types import CPU

#  RocalTensorOutputType
from rocal_pybind.types import UINT8
from rocal_pybind.types import FLOAT
from rocal_pybind.types import FLOAT16
from rocal_pybind.types import INT16

#  RocalOutputMemType
from rocal_pybind.types import HOST_MEMORY
from rocal_pybind.types import DEVICE_MEMORY
from rocal_pybind.types import PINNED_MEMORY

# RocalImageSizeEvaluationPolicy
from rocal_pybind.types import MAX_SIZE
from rocal_pybind.types import USER_GIVEN_SIZE
from rocal_pybind.types import MOST_FREQUENT_SIZE
from rocal_pybind.types import MAX_SIZE_ORIG
from rocal_pybind.types import USER_GIVEN_SIZE_ORIG

#      RocalImageColor
from rocal_pybind.types import RGB
from rocal_pybind.types import BGR
from rocal_pybind.types import GRAY
from rocal_pybind.types import RGB_PLANAR

#     RocalTensorLayout
from rocal_pybind.types import NONE
from rocal_pybind.types import NHWC
from rocal_pybind.types import NCHW
from rocal_pybind.types import NFHWC
from rocal_pybind.types import NFCHW
from rocal_pybind.types import NHW
#     RocalSpectrogramLayout
from rocal_pybind.types import NFT
from rocal_pybind.types import NTF

#     RocalDecodeDevice
from rocal_pybind.types import HARDWARE_DECODE
from rocal_pybind.types import SOFTWARE_DECODE

#     RocalDecodeDevice
from rocal_pybind.types import DECODER_TJPEG
from rocal_pybind.types import DECODER_OPENCV
from rocal_pybind.types import DECODER_HW_JEPG
from rocal_pybind.types import DECODER_VIDEO_FFMPEG_SW
from rocal_pybind.types import DECODER_VIDEO_FFMPEG_HW
from rocal_pybind.types import DECODER_AUDIO_GENERIC
from rocal_pybind.types import DECODER_VIDEO_ROCDECODE
from rocal_pybind.types import DECODER_ROCJPEG

#     RocalResizeScalingMode
from rocal_pybind.types import SCALING_MODE_DEFAULT
from rocal_pybind.types import SCALING_MODE_STRETCH
from rocal_pybind.types import SCALING_MODE_NOT_SMALLER
from rocal_pybind.types import SCALING_MODE_NOT_LARGER
from rocal_pybind.types import SCALING_MODE_MIN_MAX

#     RocalResizeInterpolationType
from rocal_pybind.types import NEAREST_NEIGHBOR_INTERPOLATION
from rocal_pybind.types import LINEAR_INTERPOLATION
from rocal_pybind.types import CUBIC_INTERPOLATION
from rocal_pybind.types import LANCZOS_INTERPOLATION
from rocal_pybind.types import GAUSSIAN_INTERPOLATION
from rocal_pybind.types import TRIANGULAR_INTERPOLATION

#     Rocal External Source Mode
from rocal_pybind.types import EXTSOURCE_FNAME
from rocal_pybind.types import EXTSOURCE_RAW_COMPRESSED
from rocal_pybind.types import EXTSOURCE_RAW_UNCOMPRESSED

#     RocalAudioBorderType
from rocal_pybind.types import ZERO
from rocal_pybind.types import CLAMP
from rocal_pybind.types import REFLECT

#     RocalOutOfBoundsPolicy
from rocal_pybind.types import PAD
from rocal_pybind.types import TRIMTOSHAPE
from rocal_pybind.types import ERROR

#     RocalMelScaleFormula
from rocal_pybind.types import MELSCALE_SLANEY
from rocal_pybind.types import MELSCALE_HTK

#     RocalLastBatchPolicy
from rocal_pybind.types import LAST_BATCH_FILL
from rocal_pybind.types import LAST_BATCH_DROP
from rocal_pybind.types import LAST_BATCH_PARTIAL

#     RocalMissingComponentsBehaviour
from rocal_pybind.types import MISSING_COMPONENT_ERROR
from rocal_pybind.types import MISSING_COMPONENT_SKIP
from rocal_pybind.types import MISSING_COMPONENT_EMPTY

_known_types = {

    OK: ("OK", OK),
    CONTEXT_INVALID: ("CONTEXT_INVALID", CONTEXT_INVALID),
    RUNTIME_ERROR: ("RUNTIME_ERROR", RUNTIME_ERROR),
    UPDATE_PARAMETER_FAILED: ("UPDATE_PARAMETER_FAILED", UPDATE_PARAMETER_FAILED),
    INVALID_PARAMETER_TYPE: ("INVALID_PARAMETER_TYPE", INVALID_PARAMETER_TYPE),

    GPU: ("GPU", GPU),
    CPU: ("CPU", CPU),
    UINT8: ("UINT8", UINT8),
    FLOAT: ("FLOAT", FLOAT),
    FLOAT16: ("FLOAT16", FLOAT16),
    INT16: ("INT16", INT16),
    HOST_MEMORY: ("HOST_MEMORY", HOST_MEMORY),
    DEVICE_MEMORY: ("DEVICE_MEMORY", DEVICE_MEMORY),
    PINNED_MEMORY: ("PINNED_MEMORY", PINNED_MEMORY),

    MAX_SIZE: ("MAX_SIZE", MAX_SIZE),
    USER_GIVEN_SIZE: ("USER_GIVEN_SIZE", USER_GIVEN_SIZE),
    MOST_FREQUENT_SIZE: ("MOST_FREQUENT_SIZE", MOST_FREQUENT_SIZE),
    MAX_SIZE_ORIG: ("MAX_SIZE_ORIG", MAX_SIZE_ORIG),
    USER_GIVEN_SIZE_ORIG: ("USER_GIVEN_SIZE_ORIG", USER_GIVEN_SIZE_ORIG),

    NONE: ("NONE", NONE),
    NHWC: ("NHWC", NHWC),
    NCHW: ("NCHW", NCHW),
    NFHWC: ("NFHWC", NFHWC),
    NFCHW: ("NFCHW", NFCHW),
    NHW: ("NHW", NHW),
    BGR: ("BGR", BGR),
    RGB: ("RGB", RGB),
    GRAY: ("GRAY", GRAY),
    RGB_PLANAR: ("RGB_PLANAR", RGB_PLANAR),

    HARDWARE_DECODE: ("HARDWARE_DECODE", HARDWARE_DECODE),
    SOFTWARE_DECODE: ("SOFTWARE_DECODE", SOFTWARE_DECODE),

    DECODER_TJPEG: ("DECODER_TJPEG", DECODER_TJPEG),
    DECODER_OPENCV: ("DECODER_OPENCV", DECODER_OPENCV),
    DECODER_HW_JEPG: ("DECODER_HW_JEPG", DECODER_HW_JEPG),
    DECODER_VIDEO_FFMPEG_SW: ("DECODER_VIDEO_FFMPEG_SW", DECODER_VIDEO_FFMPEG_SW),
    DECODER_VIDEO_FFMPEG_HW: ("DECODER_VIDEO_FFMPEG_HW", DECODER_VIDEO_FFMPEG_HW),
    DECODER_AUDIO_GENERIC: ("DECODER_AUDIO_GENERIC", DECODER_AUDIO_GENERIC),
    DECODER_VIDEO_ROCDECODE: ("DECODER_VIDEO_ROCDECODE", DECODER_VIDEO_ROCDECODE),
    DECODER_ROCJPEG: ("DECODER_ROCJPEG", DECODER_ROCJPEG),

    NEAREST_NEIGHBOR_INTERPOLATION: ("NEAREST_NEIGHBOR_INTERPOLATION", NEAREST_NEIGHBOR_INTERPOLATION),
    LINEAR_INTERPOLATION: ("LINEAR_INTERPOLATION", LINEAR_INTERPOLATION),
    CUBIC_INTERPOLATION: ("CUBIC_INTERPOLATION", CUBIC_INTERPOLATION),
    LANCZOS_INTERPOLATION: ("LANCZOS_INTERPOLATION", LANCZOS_INTERPOLATION),
    GAUSSIAN_INTERPOLATION: ("GAUSSIAN_INTERPOLATION", GAUSSIAN_INTERPOLATION),
    TRIANGULAR_INTERPOLATION: ("TRIANGULAR_INTERPOLATION", TRIANGULAR_INTERPOLATION),

    SCALING_MODE_DEFAULT: ("SCALING_MODE_DEFAULT", SCALING_MODE_DEFAULT),
    SCALING_MODE_STRETCH: ("SCALING_MODE_STRETCH", SCALING_MODE_STRETCH),
    SCALING_MODE_NOT_SMALLER: ("SCALING_MODE_NOT_SMALLER", SCALING_MODE_NOT_SMALLER),
    SCALING_MODE_NOT_LARGER: ("SCALING_MODE_NOT_LARGER", SCALING_MODE_NOT_LARGER),
    SCALING_MODE_MIN_MAX: ("SCALING_MODE_MIN_MAX", SCALING_MODE_MIN_MAX),

    EXTSOURCE_FNAME: ("EXTSOURCE_FNAME", EXTSOURCE_FNAME),
    EXTSOURCE_RAW_COMPRESSED: ("EXTSOURCE_RAW_COMPRESSED", EXTSOURCE_RAW_COMPRESSED),
    EXTSOURCE_RAW_UNCOMPRESSED: ("EXTSOURCE_RAW_UNCOMPRESSED", EXTSOURCE_RAW_UNCOMPRESSED),

    ZERO: ("ZERO", ZERO),
    CLAMP: ("CLAMP", CLAMP),
    REFLECT: ("REFLECT", REFLECT),

    PAD: ("PAD", PAD),
    TRIMTOSHAPE: ("TRIMTOSHAPE", TRIMTOSHAPE),
    ERROR: ("ERROR", ERROR),

    NTF: ("NTF", NTF),
    NFT: ("NFT", NFT),

    MELSCALE_SLANEY: ("MELSCALE_SLANEY", MELSCALE_SLANEY),
    MELSCALE_HTK: ("MELSCALE_HTK", MELSCALE_HTK),

    LAST_BATCH_FILL : ("LAST_BATCH_FILL", LAST_BATCH_FILL),
    LAST_BATCH_DROP : ("LAST_BATCH_DROP", LAST_BATCH_DROP),
    LAST_BATCH_PARTIAL : ("LAST_BATCH_PARTIAL", LAST_BATCH_PARTIAL),

    MISSING_COMPONENT_ERROR : ("MISSING_COMPONENT_ERROR", MISSING_COMPONENT_ERROR),
    MISSING_COMPONENT_SKIP : ("MISSING_COMPONENT_SKIP", MISSING_COMPONENT_SKIP),
    MISSING_COMPONENT_EMPTY : ("MISSING_COMPONENT_EMPTY", MISSING_COMPONENT_EMPTY),
}

def data_type_function(dtype):
    """!Converts a given data type identifier to its corresponding known type.

        @param dtype    The data type identifier.

        @return    Known type corresponding to the given data type identifier.

        @raise     RuntimeError: If the given data type identifier does not correspond to a known type.
    """
    if dtype in _known_types:
        ret = _known_types[dtype][0]
        return ret
    else:
        raise RuntimeError(
            str(dtype) + " does not correspond to a known type.")
