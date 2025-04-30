# encoding: utf-8
# Copyright (C) 2021 - 2022 Advanced Micro Devices, Inc. All rights reserved.
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

# This module monkey patches the docx library to add support for Windows Enhanced Metafile images (EMF)
# Put in a local folder and "import docx_emf_patch" to enable EMF support.

from __future__ import absolute_import, division, print_function

import docx
from docx.image.exceptions import UnrecognizedImageError
from docx.image.constants import MIME_TYPE
from docx.image.exceptions import InvalidImageStreamError
from docx.image.helpers import BIG_ENDIAN, StreamReader
from docx.image.image import BaseImageHeader
import struct


def _ImageHeaderFactory(stream):
    """
    Return a |BaseImageHeader| subclass instance that knows how to parse the
    headers of the image in *stream*.
    """
    from docx.image import SIGNATURES

    def read_64(stream):
        stream.seek(0)
        return stream.read(64)

    header = read_64(stream)
    for cls, offset, signature_bytes in SIGNATURES:
        end = offset + len(signature_bytes)
        found_bytes = header[offset:end]
        if found_bytes == signature_bytes:
            return cls.from_stream(stream)
    raise UnrecognizedImageError


class Emf(BaseImageHeader):
    """
    Image header parser for EMF images
    """

    @property
    def content_type(self):
        """
        MIME content type for this image, unconditionally `image/emf` for
        EMF images.
        """
        return MIME_TYPE.EMF

    @property
    def default_ext(self):
        """
        Default filename extension, always 'emf' for EMF images.
        """
        return 'emf'

    @classmethod
    def from_stream(cls, stream, filename=None):
        """
        Return a |Emf| instance having header properties parsed from image in
        *stream*.
        """
        """
        @0 DWORD iType; // fixed
        @4 DWORD nSize; // var
        @8 RECTL rclBounds;
        @24 RECTL rclFrame; // .01 millimeter units L T R B
        @40 DWORD dSignature; // ENHMETA_SIGNATURE = 0x464D4520
        DWORD nVersion;
        DWORD nBytes;
        DWORD nRecords;
        WORD  nHandles;
        WORD  sReserved;
        DWORD nDescription;
        DWORD offDescription;
        DWORD nPalEntries;
        SIZEL szlDevice;
        SIZEL szlMillimeters;
        """
        stream.seek(0)
        x = stream.read(40)
        stream.seek(0)
        iType, nSize = struct.unpack("ii", x[0:8])
        rclBounds = struct.unpack("iiii", x[8:24])
        rclFrame = struct.unpack("iiii", x[24:40])

        dpi = 300
        horz_dpi = dpi
        vert_dpi = dpi
        mmwidth = (rclFrame[2] - rclFrame[0]) / 100.0
        mmheight = (rclFrame[3] - rclFrame[1]) / 100.0
        px_width = int(mmwidth * dpi * 0.03937008)
        px_height = int(mmheight * dpi * 0.03937008)

        #1 dot/inch  =  0.03937008 pixel/millimeter
        return cls(px_width, px_height, horz_dpi, vert_dpi)


docx.image.Emf = Emf
docx.image.constants.MIME_TYPE.EMF = 'image/emf'
docx.image.SIGNATURES = tuple(
    list(docx.image.SIGNATURES) + [(Emf, 40, b' EMF')])
docx.image.image._ImageHeaderFactory = _ImageHeaderFactory
