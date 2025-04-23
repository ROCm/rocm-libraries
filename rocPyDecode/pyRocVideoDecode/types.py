# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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
 
# rocDecVideoSurfaceFormat
from rocpydecode.decTypes import rocDecVideoSurfaceFormat_NV12
from rocpydecode.decTypes import rocDecVideoSurfaceFormat_P016
from rocpydecode.decTypes import rocDecVideoSurfaceFormat_YUV444
from rocpydecode.decTypes import rocDecVideoSurfaceFormat_YUV444_16Bit

# Video Packet Flags
from rocpydecode.decTypes import ROCDEC_PKT_ENDOFSTREAM
from rocpydecode.decTypes import ROCDEC_PKT_TIMESTAMP
from rocpydecode.decTypes import ROCDEC_PKT_DISCONTINUITY
from rocpydecode.decTypes import ROCDEC_PKT_ENDOFPICTURE
from rocpydecode.decTypes import ROCDEC_PKT_NOTIFY_EOS

# Video Codecs
from rocpydecode.decTypes import rocDecVideoCodec_AVC
from rocpydecode.decTypes import rocDecVideoCodec_HEVC

#  OutputFormatEnum - Types of images
from rocpydecode.decTypes import native
from rocpydecode.decTypes import bgr
from rocpydecode.decTypes import bgr48
from rocpydecode.decTypes import rgb
from rocpydecode.decTypes import rgb48
from rocpydecode.decTypes import bgra
from rocpydecode.decTypes import bgra64
from rocpydecode.decTypes import rgba
from rocpydecode.decTypes import rgba64

_known_types = {
    rocDecVideoSurfaceFormat_NV12: ("rocDecVideoSurfaceFormat_NV12", rocDecVideoSurfaceFormat_NV12),
    rocDecVideoSurfaceFormat_P016: ("rocDecVideoSurfaceFormat_P016", rocDecVideoSurfaceFormat_P016),
    rocDecVideoSurfaceFormat_YUV444: ("rocDecVideoSurfaceFormat_YUV444", rocDecVideoSurfaceFormat_YUV444),
    rocDecVideoSurfaceFormat_YUV444_16Bit: ("rocDecVideoSurfaceFormat_YUV444_16Bit", rocDecVideoSurfaceFormat_YUV444_16Bit),
    
    ROCDEC_PKT_ENDOFSTREAM: ("ROCDEC_PKT_ENDOFSTREAM", ROCDEC_PKT_ENDOFSTREAM),
    ROCDEC_PKT_TIMESTAMP: ("ROCDEC_PKT_TIMESTAMP", ROCDEC_PKT_TIMESTAMP),
    ROCDEC_PKT_DISCONTINUITY: ("ROCDEC_PKT_DISCONTINUITY", ROCDEC_PKT_DISCONTINUITY),
    ROCDEC_PKT_ENDOFPICTURE: ("ROCDEC_PKT_ENDOFPICTURE", ROCDEC_PKT_ENDOFPICTURE),
    ROCDEC_PKT_NOTIFY_EOS: ("ROCDEC_PKT_NOTIFY_EOS", ROCDEC_PKT_NOTIFY_EOS),

    rocDecVideoCodec_AVC: ("rocDecVideoCodec_AVC",rocDecVideoCodec_AVC),
    rocDecVideoCodec_HEVC: ("rocDecVideoCodec_HEVC",rocDecVideoCodec_HEVC),

    native: ("native",native),
    bgr: ("bgr",bgr),
    bgr48: ("bgr48",bgr48),
    rgb: ("rgb",rgb),
    rgb48: ("rgb48",rgb48),
    bgra: ("bgra",bgra),
    bgra64: ("bgra64",bgra64),
    rgba: ("rgba",rgba),
    rgba64: ("rgba64",rgba64),
    }


def data_type_function(dtype):
    if dtype in _known_types:
        ret = _known_types[dtype][0]
        return ret
    else:
        raise RuntimeError(
            str(dtype) +
            " does not correspond to a known type.")
