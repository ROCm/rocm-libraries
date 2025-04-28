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
# @file fn.py
#
# @brief File containing augmentation functions used in multiple trainings

from amd.rocal import readers
from amd.rocal import decoders
from amd.rocal import random
from amd.rocal import noise
from amd.rocal import reductions

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline


def blend(*inputs, ratio=None, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Blends two input images given the ratio: output = input1*ratio + input2*(1-ratio)

        @param inputs                                                                 list containing the input images
        @param ratio (float, optional, default = None)                                ratio used for blending one image with another
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    blended image
    """
    ratio = b.createFloatParameter(
        ratio) if isinstance(ratio, float) else ratio
    # pybind call arguments
    kwargs_pybind = {"input_image0": inputs[0], "input_image1": inputs[1], "is_output": False, "ratio": ratio,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    blend_image = b.blend(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (blend_image)


def snow(*inputs, snow=0.5, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies snow effect on images.

        @param inputs                                                                 the input image passed to the augmentation
        @param snow (float, default = 0.5)                                            snow fill value used for the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with snow effect
    """
    snow = b.createFloatParameter(snow) if isinstance(snow, float) else snow
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "snow": snow,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    snow_image = b.snow(Pipeline._current_pipeline._handle,
                        *(kwargs_pybind.values()))
    return (snow_image)


def exposure(*inputs, exposure=0.5, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts the exposure in images.

        @param inputs                                                                 the input image passed to the augmentation
        @param exposure (float, default = 0.5)                                        exposure fill value used for the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with adjusted exposure
    """
    exposure = b.createFloatParameter(
        exposure) if isinstance(exposure, float) else exposure
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "exposure": exposure,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    exposure_image = b.exposure(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (exposure_image)


def fish_eye(*inputs, device=None, fill_value=0.0, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies fish eye effect on images.

        @param inputs                                                                 the input image passed to the augmentation
        @param fill_value (float, optional, default = 0.0)                            Parameter unused for augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with fish eye effect
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    fisheye_image = b.fishEye(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (fisheye_image)


def fog(*inputs, fog=0.5, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies fog effect on images.

        @param inputs                                                                 the input image passed to the augmentation
        @param fog (float, default = 0.5)                                             fog fill value used for the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with fog effect
    """
    fog = b.createFloatParameter(fog) if isinstance(fog, float) else fog
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "fog_value": fog, "output_layout": output_layout, "output_dtype": output_dtype}
    fog_image = b.fog(Pipeline._current_pipeline._handle,
                      *(kwargs_pybind.values()))
    return (fog_image)


def brightness(*inputs, brightness=None, brightness_shift=None, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts brightness of the image.

        @param inputs                                                                 the input image passed to the augmentation
        @param brightness (float, optional, default = None):                          brightness multiplier. Values >= 0 are accepted. For example: 0 - black image, 1 - no change, 2 - increase brightness twice
        @param brightness_shift (float, optional, default = None)                     brightness shift
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with Adjusted Brightness
    """
    brightness = b.createFloatParameter(brightness) if isinstance(
        brightness, float) else brightness
    brightness_shift = b.createFloatParameter(brightness_shift) if isinstance(
        brightness_shift, float) else brightness_shift

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "brightness": brightness, "brightness_shift": brightness_shift,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    brightness_image = b.brightness(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (brightness_image)


def brightness_fixed(*inputs, brightness=1.0, brightness_shift=0.0, device=None,
                     output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts brightness of the image with fixed parameters.

        @param inputs                                                                 the input image passed to the augmentation
        @param brightness (float, optional, default = 1.0)                            brightness multiplier. Values >= 0 are accepted. For example: 0 - black image, 1 - no change, 2 - increase brightness twice
        @param brightness_shift (float, optional, default = 0.0)                      brightness shift
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with adjusted brightness
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "brightness": brightness, "brightness_shift": brightness_shift,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    brightness_image = b.brightnessFixed(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (brightness_image)


def lens_correction(*inputs, strength=None, zoom=None, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies lens correction effect on images.

        @param inputs                                                                 the input image passed to the augmentation
        @param strength (float, optional, default = None)                             strength value used for the augmentation
        @param zoom (float, optional, default = None)                                 zoom value used for the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return  Image with lens correction effect
    """
    strength = b.createFloatParameter(
        strength) if isinstance(strength, float) else strength
    zoom = b.createFloatParameter(zoom) if isinstance(zoom, float) else zoom

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "strength": strength, "zoom": zoom,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    len_corrected_image = b.lensCorrection(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (len_corrected_image)


def blur(*inputs, window_size=None, sigma=0.0, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies blur effect to images.

        @param inputs                                                                 the input image passed to the augmentation
        @param window_size (int, default = None)                                      kernel size used for the filter
        @param sigma (float, default = 0.0)                                           sigma value for blur effect
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with Blur effect
    """
    window_size = b.createIntParameter(window_size) if isinstance(
        window_size, int) else window_size
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "window_size": window_size,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    blur_image = b.blur(Pipeline._current_pipeline._handle,
                        *(kwargs_pybind.values()))
    return (blur_image)


def contrast(*inputs, contrast=None, contrast_center=None, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts contrast of the image

        @param inputs: the input image passed to the augmentation
        @param contrast (float, optional, default = None)                             contrast multiplier used for the augmentation. Values >= 0 are accepted. For example: 0 - gray image, 1 - no change, 2 - increase contrast twice
        @param contrast_center (float, optional, default = None)                      intensity value unaffected by the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with adjusted contrast
    """
    contrast = b.createFloatParameter(
        contrast) if isinstance(contrast, float) else contrast
    contrast_center = b.createFloatParameter(contrast_center) if isinstance(
        contrast_center, float) else contrast_center

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "contrast": contrast, "contrast_center": contrast_center, "output_layout": output_layout, "output_dtype": output_dtype}
    contrast_image = b.contrast(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (contrast_image)


def flip(*inputs, horizontal=0, vertical=0, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Flip images horizontally and/or vertically based on inputs.

        @param inputs                                                                 the input image passed to the augmentation
        @param horizontal (int, optional, default = 0)                                flip the horizontal dimension
        @param vertical (int, optional, default = 0)                                  flip the vertical dimension
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Flipped Image
    """
    horizontal = b.createIntParameter(horizontal) if isinstance(
        horizontal, int) else horizontal
    vertical = b.createIntParameter(
        vertical) if isinstance(vertical, int) else vertical

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "horizontal": horizontal, "vertical": vertical, "output_layout": output_layout, "output_dtype": output_dtype}
    flip_image = b.flip(Pipeline._current_pipeline._handle,
                        *(kwargs_pybind.values()))
    return (flip_image)


def gamma_correction(*inputs, gamma=0.5, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies gamma correction on image.

        @param inputs                                                                 the input image passed to the augmentation
        @param gamma (float, default = 0.5)                                           gamma correction value used for the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return Image with Gamma Correction
    """
    gamma = b.createFloatParameter(
        gamma) if isinstance(gamma, float) else gamma
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "gamma": gamma, "output_layout": output_layout, "output_dtype": output_dtype}
    gamma_correction_image = b.gammaCorrection(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (gamma_correction_image)


def hue(*inputs, hue=None, device=None, seed=0, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjust the hue in the images

        @param inputs                                                                 the input image passed to the augmentation
        @param hue (float, default = None)                                            hue change in degrees
        @param seed (int, optional, default = 0)                                      seed used for randomization in the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with Hue effect
    """
    hue = b.createFloatParameter(hue) if isinstance(hue, float) else hue
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "hue": hue, "output_layout": output_layout, "output_dtype": output_dtype}
    hue_image = b.hue(Pipeline._current_pipeline._handle,
                      *(kwargs_pybind.values()))
    return (hue_image)


def jitter(*inputs, kernel_size=None, seed=0, fill_value=0.0, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies Jitter effect on images

        @param inputs                                                                 the input image passed to the augmentation
        @param kernel_size (int, optional, default = None)                            kernel size used for the augmentation
        @param seed (int, optional, default = 0)                                      seed used for randomization in the augmentation
        @param fill_value (float, optional, default = 0.0)                            Value to fill areas outside image.
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with Jitter effect
    """
    kernel_size = b.createIntParameter(kernel_size) if isinstance(
        kernel_size, int) else kernel_size
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "kernel_size": kernel_size, "seed": seed, "output_layout": output_layout, "output_dtype": output_dtype}
    jitter_image = b.jitter(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (jitter_image)


def pixelate(*inputs, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies pixelate effect on images

        @param inputs                                                                 the input image passed to the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Images with pixelate effect
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "output_layout": output_layout, "output_dtype": output_dtype}
    pixelate_image = b.pixelate(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (pixelate_image)


def rain(*inputs, rain=None, rain_width=None, rain_height=None, rain_transparency=None,
         device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies Rain effect on images

        @param inputs                                                                 the input image passed to the augmentation
        @param rain (float, optional, default = None)                                 rain fill value used for the augmentation
        @param rain_width (int, optional, default = None)                             width of the rain pixels for the augmentation
        @param rain_height (int, optional, default = None)                            height of the rain pixels for the augmentation
        @param rain_transparency (float, optional, default = None)                    transparency value used for the augmentation
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC):                   tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Images with Rain effect
    """
    rain = b.createFloatParameter(rain) if isinstance(rain, float) else rain
    rain_width = b.createIntParameter(rain_width) if isinstance(
        rain_width, int) else rain_width
    rain_height = b.createIntParameter(rain_height) if isinstance(
        rain_height, int) else rain_height
    rain_transparency = b.createFloatParameter(rain_transparency) if isinstance(
        rain_transparency, float) else rain_transparency

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "rain_value": rain, "rain_width": rain_width, "rain_height": rain_height,
                     "rain_transparency": rain_transparency, "output_layout": output_layout, "output_dtype": output_dtype}
    rain_image = b.rain(Pipeline._current_pipeline._handle,
                        *(kwargs_pybind.values()))
    return (rain_image)


def resize(*inputs, max_size=[], resize_longer=0, resize_shorter=0, resize_width=0, resize_height=0, scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
           antialias=True, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Resizes the images

        @param inputs                                                                      the input image passed to the augmentation
        @param max_size (int or list of int, optional, default = [])                       Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter if the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W). Example: Original image = 400x1200. Resized with: resize_shorter = 200 (max_size not set) => 200x600 resize_shorter = 200, max_size =  400 => 132x400 resize_shorter = 200, max_size = 1000 => 200x600
        @param resize_longer (int, optional, default = 0)                                  The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.
        @param resize_shorter (int, optional, default = 0)                                 The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.
        @param resize_width (int, optional, default = 0)                                   The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.
        @param resize_height (int, optional, default = 0)                                  The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.
        @param scaling_mode (int, optional, default = types.SCALING_MODE_DEFAULT)          resize scaling mode.
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Type of interpolation to be used.
        @param antialias (bool, optional, default = True)                                  Parameter unused for augmentation
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    Resized Image
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter,
                     "resize_longer": resize_longer, "interpolation_type": interpolation_type, "output_layout": output_layout, "output_dtype": output_dtype}
    resized_image = b.resize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (resized_image)


def resize_crop_mirror(*inputs, resize_width=0, resize_height=0, crop_w=0, crop_h=0, mirror=1,
                       device=None, max_size=[], resize_longer=0, resize_shorter=0, scaling_mode=types.SCALING_MODE_DEFAULT,
                       interpolation_type=types.LINEAR_INTERPOLATION, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Fused function which performs resize, crop and flip on images.

        @param inputs                                                                      the input image passed to the augmentation
        @param resize_width (int, optional, default = 0)                                   The length of the X dimension of the resized image
        @param resize_height (int, optional, default = 0)                                  The length of the Y dimension of the resized image
        @param crop_w (int, optional, default = 0)                                         Cropping window width (in pixels).
        @param crop_h (int, optional, default = 0)                                         Cropping window height (in pixels).
        @param mirror (int, optional, default = 1)                                         flag for the horizontal flip.
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param max_size (int or list of int, optional, default = [])                       Parameter unused for augmentation
        @param resize_longer (int, optional, default = 0)                                  Parameter unused for augmentation
        @param resize_shorter (int, optional, default = 0)                                 Parameter unused for augmentation
        @param scaling_mode (int, optional, default = types.SCALING_MODE_DEFAULT)          Parameter unused for augmentation
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    Resized crop mirror Image
    """
    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "crop_w": crop_w,
                     "crop_h": crop_h, "mirror": mirror, "output_layout": output_layout, "output_dtype": output_dtype}
    rcm = b.resizeCropMirrorFixed(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rcm)


def resize_crop(*inputs, resize_width=0, resize_height=0, crop_area_factor=None, crop_aspect_ratio=None, x_drift=None, y_drift=None,
                device=None, max_size=[], resize_longer=0, resize_shorter=0, scaling_mode=types.SCALING_MODE_DEFAULT,
                interpolation_type=types.LINEAR_INTERPOLATION, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Fused function which performs resize, crop on images.

        @param inputs: the input image passed to the augmentation
        @param resize_width (int, optional, default = 0)                                   The length of the X dimension of the resized image
        @param resize_height (int, optional, default = 0)                                  The length of the Y dimension of the resized image
        @param crop_area_factor (float, optional, default = None)                          area factor used for crop generation
        @param crop_aspect_ratio (float, optional, default = None)                         aspect ratio used for crop generation
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param x_drift (float, optional, default = None)                                   x_drift used for crop generation
        @param y_drift (float, optional, default = None)                                   y_drift used for crop generation
        @param max_size (int or list of int, optional, default = [])                       Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter if the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W). Example: Original image = 400x1200. Resized with: resize_shorter = 200 (max_size not set) => 200x600 resize_shorter = 200, max_size =  400 => 132x400 resize_shorter = 200, max_size = 1000 => 200x600
        @param resize_longer (int, optional, default = 0)                                  The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.
        @param resize_shorter (int, optional, default = 0)                                 The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.
        @param scaling_mode (int, optional, default = types.SCALING_MODE_DEFAULT)          resize scaling mode.
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Type of interpolation to be used.
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    Resized and cropped Image
    """
    crop_area_factor = b.createFloatParameter(crop_area_factor) if isinstance(
        crop_area_factor, float) else crop_area_factor
    crop_aspect_ratio = b.createFloatParameter(crop_aspect_ratio) if isinstance(
        crop_aspect_ratio, float) else crop_aspect_ratio
    x_drift = b.createFloatParameter(
        x_drift) if isinstance(x_drift, float) else x_drift
    y_drift = b.createFloatParameter(
        y_drift) if isinstance(y_drift, float) else y_drift

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "crop_area_factor": crop_area_factor,
                     "crop_aspect_ratio": crop_aspect_ratio, "x_drift": x_drift, "y_drift": y_drift, "output_layout": output_layout, "output_dtype": output_dtype}
    crop_resized_image = b.cropResize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (crop_resized_image)


def roi_resize(*inputs, resize_width=0, resize_height=0, roi_w=None, roi_h=None, roi_pos_x=None, roi_pos_y=None, device=None,
                      interpolation_type=types.LINEAR_INTERPOLATION, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Function which resizes images based on ROI region.

        @param inputs: the input image passed to the augmentation
        @param resize_width (int, optional, default = 0)                                   The length of the X dimension of the resized image
        @param resize_height (int, optional, default = 0)                                  The length of the Y dimension of the resized image
        @param roi_w (float, optional, default = None)                                     ROI width
        @param roi_h (float, optional, default = None)                                     ROI height
        @param roi_pos_x (float, optional, default = None)                                 roi_pos_x used for crop generation
        @param roi_pos_y (float, optional, default = None)                                 roi_pos_y used for crop generation
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Type of interpolation to be used.
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    ROI resized image
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "is_output": False, "roi_h": roi_h,
                     "roi_w": roi_w, "roi_pos_x": roi_pos_x, "roi_pos_y": roi_pos_y, "interpolation_type": interpolation_type, "output_layout": output_layout, "output_dtype": output_dtype}
    roi_resized_image = b.roiResize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (roi_resized_image)


def resize_mirror_normalize(*inputs, max_size=[], resize_longer=0, resize_shorter=0, resize_width=0, resize_height=0, scaling_mode=types.SCALING_MODE_DEFAULT,
                            interpolation_type=types.LINEAR_INTERPOLATION, mean=[0.0], std=[1.0], mirror=1, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Fused function which performs resize, Normalize and flip on images.

        @param inputs                                                                     the input image passed to the augmentation
        @param  max_size (int or list of int, optional, default = [])                     Maximum size of the longer dimension when resizing with resize_shorter. When set with resize_shorter, the shortest dimension will be resized to resize_shorter if the longest dimension is smaller or equal to max_size. If not, the shortest dimension is resized to satisfy the constraint longest_dim == max_size. Can be also an array of size 2, where the two elements are maximum size per dimension (H, W).
                Example:
                Original image = 400x1200.
                Resized with:
                resize_shorter = 200 (max_size not set) => 200x600
                resize_shorter = 200, max_size =  400 => 132x400
                resize_shorter = 200, max_size = 1000 => 200x600
        @param resize_longer (int, optional, default = 0)                                  The length of the longer dimension of the resized image. This option is mutually exclusive with resize_shorter,`resize_x` and resize_y. The op will keep the aspect ratio of the original image.
        @param resize_shorter (int, optional, default = 0)                                 The length of the shorter dimension of the resized image. This option is mutually exclusive with resize_longer, resize_x and resize_y. The op will keep the aspect ratio of the original image. The longer dimension can be bounded by setting the max_size argument. See max_size argument doc for more info.
        @param resize_width (int, optional, default = 0)                                   The length of the X dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_y is left at 0, then the op will keep the aspect ratio of the original image.
        @param resize_height (int, optional, default = 0)                                  The length of the Y dimension of the resized image. This option is mutually exclusive with resize_shorter. If the resize_x is left at 0, then the op will keep the aspect ratio of the original image.
        @param scaling_mode (int, optional, default = types.SCALING_MODE_DEFAULT)          resize scaling mode.
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Type of interpolation to be used.
        @param mean (list of floats, optional, default = [0.0])                            mean used for normalization
        @param std (list of floats, optional, default = [1.0])                             standard deviation used for normalization
        @param mirror (int, optional, default = 1)                                         flag for the horizontal flip.
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    Transformed Image
    """
    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "dest_width:": resize_width, "dest_height": resize_height, "mean": mean, "std_dev": std, "is_output": False,
                     "scaling_mode": scaling_mode, "max_size": max_size, "resize_shorter": resize_shorter, "resize_longer": resize_longer,
                     "interpolation_type": interpolation_type, "mirror": mirror, "output_layout": output_layout, "output_dtype": output_dtype}
    rmn = b.resizeMirrorNormalize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rmn)


def random_crop(*inputs, crop_area_factor=[0.08, 1], crop_aspect_ratio=[0.75, 1.333333],
                crop_pox_x=0, crop_pox_y=0, num_attempts=20, device=None,
                all_boxes_above_threshold=True, allow_no_crop=True, ltrb=True, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Crops images randomly.

        @param inputs: the input image passed to the augmentation
        @param crop_area_factor (list of floats, optional, default = [0.08, 1])            area factor used for crop generation
        @param crop_aspect_ratio (list of floats, optional, default = [0.75, 1.333333])    valid range of aspect ratio of the cropping windows
        @param crop_pox_x (int, optional, default = 0)                                     crop_x position used for crop generation
        @param crop_pox_y (int, optional, default = 0)                                     crop_y position used for crop generation
        @param num_attempts (int, optional, default = 20)                                  number of attempts to get a crop window that matches the area factor and aspect ratio conditions
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param all_boxes_above_threshold (bool, optional, default = True)                  Parameter unused for augmentation
        @param allow_no_crop (bool, optional, default = True)                              Parameter unused for augmentation
        @param ltrb (bool, optional, default = True)                                       Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    cropped Image
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "crop_area_factor": crop_area_factor, "crop_aspect_ratio": crop_aspect_ratio, "crop_pos_x": crop_pox_x, "crop_pos_y": crop_pox_y, "num_of_attempts": num_attempts, "output_layout": output_layout, "output_dtype": output_dtype}
    random_cropped_image = b.randomCrop(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (random_cropped_image)


def rotate(*inputs, angle=None, dest_width=0, dest_height=0, interpolation_type=types.LINEAR_INTERPOLATION,
           device=None, fill_value=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Rotates images

        @param inputs                                                                      the input image passed to the augmentation
        @param angle (float, optional, default = None)                                     angle used for rotating the image
        @param dest_width (int, optional, default = 0)                                     The length of the X dimension of the rotated image
        @param dest_height (int, optional, default = 0)                                    The length of the Y dimension of the rotated image
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Type of interpolation to be used.
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param fill_value (float, optional, default = 0.0)                                 Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    Roatated Image
    """
    angle = b.createFloatParameter(
        angle) if isinstance(angle, float) else angle
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False,
                     "angle": angle, "dest_width": dest_width, "dest_height": dest_height, "interpolation_type": interpolation_type, "output_layout": output_layout, "output_dtype": output_dtype}
    rotated_image = b.rotate(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (rotated_image)


def saturation(*inputs, saturation=1.0, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts the saturation in images.

        @param inputs                                                                 the input image passed to the augmentation
        @param saturation (float, default = 1.0)                                      The saturation change factor. Values must be non-negative. Example values: 0 - Completely desaturated image, 1 - No change to image's saturation.
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Image with Saturation effect
    """
    saturation = b.createFloatParameter(saturation) if isinstance(
        saturation, float) else saturation
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0],
                     "is_output": False, "sat": saturation, "output_layout": output_layout, "output_dtype": output_dtype}
    saturated_image = b.saturation(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (saturated_image)


def ssd_random_crop(*inputs, p_threshold=None, crop_area_factor=None, crop_aspect_ratio=None,
                    crop_pos_x=None, crop_pos_y=None, num_attempts=20, device=None,
                    all_boxes_above_threshold=True, allow_no_crop=True, ltrb=True, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Crops images randomly used for SSD training.

        @param inputs                                                                 the input image passed to the augmentation
        @param p_threshold (float, optional, default = None)                          threshold value used for selecting bboxes during crop generation
        @param crop_area_factor (float, optional, default = None)                     area factor used for crop generation
        @param crop_aspect_ratio (float, optional, default = None)                    aspect ratio of the cropping windows
        @param crop_pox_x (float, optional, default = None)                           crop_x position used for crop generation
        @param crop_pox_y (float, optional, default = None)                           crop_y position used for crop generation
        @param num_attempts (int, optional, default = 20)                              number of attempts to get a crop window that matches the area factor and aspect ratio conditions
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param all_boxes_above_threshold (bool, optional, default = True)             Parameter unused for augmentation
        @param allow_no_crop (bool, optional, default = True)                         Parameter unused for augmentation
        @param ltrb (bool, optional, default = True)                                  Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Randomly Cropped images for SSD training
    """
    p_threshold = b.createFloatParameter(p_threshold) if isinstance(
        p_threshold, float) else p_threshold
    crop_area_factor = b.createFloatParameter(crop_area_factor) if isinstance(
        crop_area_factor, float) else crop_area_factor
    crop_aspect_ratio = b.createFloatParameter(crop_aspect_ratio) if isinstance(
        crop_aspect_ratio, float) else crop_aspect_ratio
    crop_pos_x = b.createFloatParameter(crop_pos_x) if isinstance(
        crop_pos_x, float) else crop_pos_x
    crop_pos_y = b.createFloatParameter(crop_pos_y) if isinstance(
        crop_pos_y, float) else crop_pos_y

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_threshold": p_threshold, "crop_area_factor": crop_area_factor,
                     "crop_aspect_ratio": crop_aspect_ratio, "crop_pos_x": crop_pos_x, "crop_pos_y": crop_pos_y, "num_of_attempts": _num_attempts, "output_layout": output_layout, "output_dtype": output_dtype}
    ssd_random_cropped_image = b.ssdRandomCrop(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (ssd_random_cropped_image)


def warp_affine(*inputs, dest_width=0, dest_height=0, matrix=[0, 0, 0, 0, 0, 0],
                interpolation_type=types.LINEAR_INTERPOLATION, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies affine transformation to images.

        @param inputs                                                                      the input image passed to the augmentation
        @param dest_width (int, optional, default = 0)                                     The length of the X dimension of the transformed image
        @param matrix (list of ints, optional, default = [0, 0, 0, 0, 0, 0])               Transformation matrix used to produce a new image
        @param dest_height (int, optional, default = 0)                                    The length of the Y dimension of the transformed image
        @param interpolation_type (int, optional, default = types.LINEAR_INTERPOLATION)    Type of interpolation to be used.
        @param device (string, optional, default = None)                                   Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                         tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                         tensor dtype for the augmentation output

        @return    Affine Transformed Images
    """
    x0, x1, y0, y1, o0, o1 = matrix
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "x0": x0, "x1": x1, "y0": y0, "y1": y1, "o0": o0,
                     "o1": o1, "is_output": False, "dest_height": dest_height, "dest_width": dest_width, "interpolation_type": interpolation_type, "output_layout": output_layout, "output_dtype": output_dtype}
    warp_affine_output = b.warpAffineFixed(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (warp_affine_output)


def vignette(*inputs, vignette=0.5, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies Vignette effect

        @param inputs                                                                 the input image passed to the augmentation
        @param vignette (float, default = 0.5)                                        vignette value used for the augmentation output
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Images with Vignette effect
    """
    vignette = b.createFloatParameter(
        vignette) if isinstance(vignette, float) else vignette
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "sdev": vignette,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    vignette_output = b.vignette(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (vignette_output)


def crop_mirror_normalize(*inputs, crop=[0, 0], crop_pos_x=0.5, crop_pos_y=0.5,
                          crop_w=0, crop_h=0, mean=[0.0], std=[1.0], mirror=1, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Fused function which performs crop, normalize and flip on images.

        @param inputs                                                                 the input image passed to the augmentation
        @param crop (list of ints, optional, default = [0, 0])                        list containing the crop dimensions of the cropped image
        @param crop_pox_x (float, optional, default = 0.5)                            crop_x position used for crop generation
        @param crop_pox_y (float, optional, default = 0.5)                            crop_y position used for crop generation
        @param crop_w (int, optional, default = 0)                                    crop width
        @param crop_h (int, optional, default = 0)                                    crop height
        @param mean (list of floats, optional, default = [0.0])                       mean used for normalization
        @param std (list of floats, optional, default = [1.0])                        standard deviation used for normalization
        @param mirror (int, optional, default = 1)                                    flag for the horizontal flip.
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Transformed Images after perform crop , normalize and flip operations
    """
    if (len(crop) == 2):
        crop_height = crop[0]
        crop_width = crop[1]
    elif (len(crop) == 3):
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_height = crop_h
        crop_width = crop_w

    if isinstance(mirror, int):
        if (mirror == 0):
            mirror = b.createIntParameter(0)
        else:
            mirror = b.createIntParameter(1)

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "crop_height": crop_height, "crop_width": crop_width, "start_x": crop_pos_x, "start_y": crop_pos_y, "mean": mean, "std_dev": std,
                     "is_output": False, "mirror": mirror, "output_layout": output_layout, "output_dtype": output_dtype}
    cmn = b.cropMirrorNormalize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (cmn)


def center_crop(*inputs, crop=[0, 0], crop_h=0, crop_w=0, crop_d=1,
                device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Crops images at the center

        @param inputs                                                                 the input image passed to the augmentation
        @param crop (list of ints, optional, default = [0, 0])                        list containing the crop dimensions of the cropped image
        @param crop_h (int, optional, default = 0)                                    crop height
        @param crop_w (int, optional, default = 0)                                    crop width
        @param crop_d (int, optional, default = 0)                                    crop depth
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Center cropped Images
    """
    if (len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif (len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "crop_width": crop_width, "crop_height": crop_height, "crop_depth": crop_depth,
                     "is_output": False, "output_layout": output_layout, "output_dtype": output_dtype}
    centre_cropped_image = b.centerCropFixed(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (centre_cropped_image)


def crop(*inputs, crop=[0, 0], crop_pos_x=0.5, crop_pos_y=0.5, crop_pos_z=0.5,
         crop_w=0, crop_h=0, crop_d=1, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!crops images

        @param inputs                                                                 the input image passed to the augmentation
        @param crop (list of ints, optional, default = [0, 0])                        list containing the crop dimensions of the cropped image
        @param crop_pox_x (float, optional, default = 0.5)                            crop_x position used for crop generation
        @param crop_pox_y (float, optional, default = 0.5)                            crop_y position used for crop generation
        @param crop_pox_z (float, optional, default = 0.5)                            crop_z position used for crop generation
        @param crop_w (int, optional, default = 0)                                    crop width
        @param crop_h (int, optional, default = 0)                                    crop height
        @param crop_d (int, optional, default = 1)                                    crop depth
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Cropped Images
    """
    if (len(crop) == 2):
        crop_depth = crop_d
        crop_height = crop[0]
        crop_width = crop[1]
    elif (len(crop) == 3):
        crop_depth = crop[0]
        crop_height = crop[1]
        crop_width = crop[2]
    else:
        crop_depth = crop_d
        crop_height = crop_h
        crop_width = crop_w

    if ((crop_width == 0) and (crop_height == 0)):
        # pybind call arguments
        kwargs_pybind = {"input_image": inputs[0], "crop_width": None, "crop_height": None, "crop_depth": None, "is_output": False, "crop_pos_x": None,
                         "crop_pos_y": None, "crop_pos_z": None, "output_layout": output_layout, "output_dtype": output_dtype}
        cropped_image = b.crop(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        # pybind call arguments
        kwargs_pybind = {"input_image": inputs[0], "crop_width": crop_width, "crop_height": crop_height, "crop_depth": crop_depth, "is_output": False, "crop_pos_x": crop_pos_x,
                         "crop_pos_y": crop_pos_y, "crop_pos_z": crop_pos_z, "output_layout": output_layout, "output_dtype": output_dtype}
        cropped_image = b.cropFixed(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (cropped_image)


def color_twist(*inputs, brightness=1.0, contrast=1.0, hue=0.0,
                saturation=1.0, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts the brightness, hue and saturation of the images.

        @param inputs                                                                  the input image passed to the augmentation
        @param brightness (float, optional, default = 1.0)                             brightness multiplier. Values >= 0 are accepted. For example: 0 - black image, 1 - no change, 2 - increase brightness twice
        @param contrast (float, optional, default = 1.0)                               contrast multiplier used for the augmentation. Values >= 0 are accepted. For example: 0 - gray image, 1 - no change, 2 - increase contrast twice
        @param hue (float, optional, default = 0.0)                                    hue change in degrees
        @param saturation (float, optional, default = 1.0)                             The saturation change factor. Values must be non-negative. Example values: 0 - Completely desaturated image, 1 - No change to image's saturation.
        @param device (string, optional, default = None)                               Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                     tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8):                    tensor dtype for the augmentation output

        @return    Images with Adjusted Brightness, Hue and Saturation
    """
    brightness = b.createFloatParameter(brightness) if isinstance(
        brightness, float) else brightness
    contrast = b.createFloatParameter(
        contrast) if isinstance(contrast, float) else contrast
    hue = b.createFloatParameter(hue) if isinstance(hue, float) else hue
    saturation = b.createFloatParameter(saturation) if isinstance(
        saturation, float) else saturation

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_alpha": brightness, "p_beta": contrast,
                     "p_hue": hue, "p_sat": saturation, "output_layout": output_layout, "output_dtype": output_dtype}
    color_twist_image = b.colorTwist(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (color_twist_image)


def uniform(*inputs, range=[-1, 1], device=None):
    """!Applies uniform random number generation to the input images.

        @param inputs                                               the input image passed to the augmentation
        @param range (list of ints, optional, default = [-1, 1])    uniform distribution used for random number generation
        @param device (string, optional, default = None)            Parameter unused for augmentation

        @return    uniform random numbers
    """
    output_param = b.createFloatUniformRand(range[0], range[1])
    return output_param


def random_bbox_crop(*inputs, all_boxes_above_threshold=True, allow_no_crop=True, aspect_ratio=None, bbox_layout="",
                     threshold_type="iou", thresholds=None, crop_shape=None, num_attempts=1, scaling=None, seed=1,
                     shape_layout="", input_shape=None, total_num_attempts=0, device=None, ltrb=True, labels=None):
    """!Applies random bounding box cropping to the input images.

        @param inputs (list)                                                 The input images to which random cropping is applied.
        @param all_boxes_above_threshold (bool, optional, default = True)    If set to True, all bounding boxes in a sample should overlap with the cropping window. Default is True.
        @param allow_no_crop (bool, optional, default = True)                If set to True, one of the possible outcomes of the random process will be to not crop. Default is True.
        @param aspect_ratio (list of floats, optional, default = None)       Aspect ratio range [min, max] used for crop generation. Default is None.
        @param crop_shape (list of ints, optional, default = None)           Crop shape [width, height] used for crop generation. Default is None.
        @param num_attempts (int, optional, default = 1)                     Number of attempts to get a crop window that matches the aspect_ratio and threshold. Default is 1.
        @param scaling (list of int, optional, default = None)               Scaling range [min, max] for the crop size with respect to the original image dimensions. Default is None.
        @param seed (int, optional, default = 1)                             Random seed. Default is 1.
        @param total_num_attempts (int, optional, default = 0)               If provided, it indicates the total maximum number of attempts to get a crop window that matches the aspect_ratio and the threshold. After total_num_attempts attempts, the best candidate will be selected. If this value is not specified, the crop search will continue indefinitely until a valid crop is found. Default is 0.
        @param device (string, optional, default = None)                     Parameter unused for augmentation
        @param ltrb (bool, optional, default = True)                         Parameter unused for augmentation

        @return    cropped images
    """
    aspect_ratio = aspect_ratio if aspect_ratio else [1.0, 1.0]
    crop_shape = [] if crop_shape is None else crop_shape
    scaling = scaling if scaling else [1.0, 1.0]
    if (len(crop_shape) == 0):
        has_shape = False
        crop_width = 0
        crop_height = 0
    else:
        has_shape = True
        crop_width = crop_shape[0]
        crop_height = crop_shape[1]
    scaling = b.createFloatUniformRand(scaling[0], scaling[1])
    aspect_ratio = b.createFloatUniformRand(aspect_ratio[0], aspect_ratio[1])

    # pybind call arguments
    kwargs_pybind = {"all_boxes_above_threshold": all_boxes_above_threshold, "no_crop": allow_no_crop, "p_aspect_ratio": aspect_ratio, "has_shape": has_shape,
                     "crop_width": crop_width, "crop_height": crop_height, "num_attemps": num_attempts, "p_scaling": scaling, "total_num_attempts": total_num_attempts, "seed": seed}
    random_bbox_crop = b.randomBBoxCrop(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (random_bbox_crop, [], [], [])


def one_hot(*inputs, num_classes=0, device=None):
    """!Applies one-hot encoding to the input images.

        @param inputs (list)                               The input images to which one-hot encoding is applied.
        @param num_classes (int, optional, default = 0)    Number of classes used for one-hot encoding. Default is 0.
        @param device (string, optional, default = None)   Parameter unused for augmentation

        @return    an empty list
    """
    Pipeline._current_pipeline._num_classes = num_classes
    Pipeline._current_pipeline._one_hot_encoding = True
    return ([])


def box_encoder(*inputs, anchors, criteria=0.5, means=None,
                offset=False, scale=1.0, stds=None, device=None):
    """!Applies box encoding to the input bounding boxes.

        @param inputs (list)                                        The input bounding boxes to which box encoding is applied.
        @param anchors (list of floats)                             Anchors to be used for encoding, as a list of floats in the ltrb format.
        @param criteria (float, optional, default = 0.5)            Threshold IoU for matching bounding boxes with anchors. The value needs to be between 0 and 1. Default is 0.5.
        @param means (list of floats, optional, default = None)     [x y w h] mean values for normalization. Default is [0.0, 0.0, 0.0, 0.0].
        @param offset (bool, optional, default = False)             Returns normalized offsets ((encoded_bboxes * scale - anchors * scale) - mean) / stds in Encoded bboxes that use std and the mean and scale arguments. Default is False.
        @param scale (float, optional, default = 1.0)               Rescales the box and anchor values before the offset is calculated (for example, to return to the absolute values). Default is 1.0.
        @param stds (list of float, optional, default = None)       Parameter unused for augmentation
        @param device (string, optional, default = None)            Parameter unused for augmentation

        @return    encoded bounding boxes.
    """
    means = means if means else [0.0, 0.0, 0.0, 0.0]
    stds = stds if stds else [1.0, 1.0, 1.0, 1.0]

    # pybind call arguments
    kwargs_pybind = {"anchors": anchors, "criteria": criteria,
                     "means": means, "stds": stds, "offset": offset, "scale": scale}
    box_encoder = b.boxEncoder(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    Pipeline._current_pipeline._box_encoder = True
    return (box_encoder, [])


def color_temp(*inputs, adjustment_value=50, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Adjusts the color temperature in images.

        @param inputs                                                                 the input image passed to the augmentation
        @param adjustment_value (int, default = 50)                                   value for adjusting the color temperature
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    tensor layout for the augmentation output
        @param output_dtype (int, optional, default = types.UINT8)                    tensor dtype for the augmentation output

        @return    Images with Adjusted Color temperature
    """
    adjustment_value = b.createIntParameter(adjustment_value) if isinstance(
        adjustment_value, int) else adjustment_value
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "adjustment_value": adjustment_value,
                     "output_layout": output_layout, "output_dtype": output_dtype}
    color_temp_output = b.colorTemp(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (color_temp_output)


def nop(*inputs, device=None):
    """!Performs no operation

        @param inputs                                    the input image passed to the augmentation
        @param device (string, optional, default = None)  Parameter unused for augmentation

        @return    Nop Output
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False}
    nop_output = b.nop(Pipeline._current_pipeline._handle,
                       *(kwargs_pybind.values()))
    return (nop_output)


def copy(*inputs, device=None):
    """!Copies input tensor to output tensor.

        @param inputs                                     the input image passed to the augmentation
        @param device (string, optional, default = None)  Parameter unused for augmentation

        @return    Copied Image
    """
    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False}
    copied_image = b.copy(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (copied_image)


def snp_noise(*inputs, p_noise=0.0, p_salt=0.0, noise_val=0.0, salt_val=0.0,
              seed=0, device=None, output_layout=types.NHWC, output_dtype=types.UINT8):
    """!Applies salt-and-pepper noise to the input image.

        @param inputs (list)                                                          The input image to which salt-and-pepper noise is applied.
        @param p_noise (float, optional, default = 0.0)                               Noise probability. Default is 0.0.
        @param p_salt (float, optional, default = 0.0)                                Salt probability. Default is 0.0.
        @param noise_val (float, optional, default = 0.0)                             Noise value to be added to the image. Default is 0.0.
        @param salt_val (float, optional, default = 0.0)                              Salt value to be added to the image. Default is 0.0.
        @param seed (int, optional, default = 0)                                      Random seed. Default is 0.
        @param device (string, optional, default = None)                              Parameter unused for augmentation
        @param output_layout (int, optional, default = types.NHWC)                    Tensor layout for the augmentation output. Default is types.NHWC.
        @param output_dtype (int, optional, default = types.UINT*)                    Tensor dtype for the augmentation output. Default is types.UINT8.

        @return    images with salt-and-pepper noise added.
    """
    p_noise = b.createFloatParameter(
        p_noise) if isinstance(p_noise, float) else p_noise
    p_salt = b.createFloatParameter(
        p_salt) if isinstance(p_salt, float) else p_salt
    noise_val = b.createFloatParameter(noise_val) if isinstance(
        noise_val, float) else noise_val
    salt_val = b.createFloatParameter(
        salt_val) if isinstance(salt_val, float) else salt_val

    # pybind call arguments
    kwargs_pybind = {"input_image": inputs[0], "is_output": False, "p_noise": p_noise, "p_salt": p_salt, "noise_val": noise_val,
                     "salt_val": salt_val, "seed": seed, "output_layout": output_layout, "output_dtype": output_dtype}
    snp_noise_added_image = b.snpNoise(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (snp_noise_added_image)


def box_iou_matcher(*inputs, anchors, high_threshold=0.5,
                    low_threshold=0.4, allow_low_quality_matches=True, device=None):
    """!Applies box IoU matching to the input image.

        @param inputs (list)                                                 The input image to which box IoU matching is applied.
        @param anchors (list of floats)                                      Anchors to be used for encoding, in the ltrb format.
        @param high_threshold (float, optional, default = 0.5)               Upper threshold used for matching indices. Default is 0.5.
        @param low_threshold (float, optional, default = 0.4)                Lower threshold used for matching indices. Default is 0.4.
        @param allow_low_quality_matches (bool, optional, default = True)    Whether to allow low quality matches as output. Default is True.
        @param device (string, optional, default = None)                     Parameter unused for augmentation

        @return    matched boxes and the list of matched indices.

    """
    # pybind call arguments
    kwargs_pybind = {"anchors": anchors, "high_threshold": high_threshold,
                     "low_threshold": low_threshold, "allow_low_quality_matches": allow_low_quality_matches}
    box_iou_matcher = b.boxIouMatcher(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    Pipeline._current_pipeline._box_iou_matcher = True
    return (box_iou_matcher, [])


def external_source(source, device=None, color_format=types.RGB, random_shuffle=False, mode=types.EXTSOURCE_FNAME, max_width=2000, max_height=2000, last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False, stick_to_shard=True, shard_size=-1):
    """
    External Source Reader - User can pass a iterator or callable source.
    @param source (iterator or callable)                                 The source iterator or callable object.
    @param device (string, optional, default = None)                     Parameter unused for augmentation
    @color_format (type, optional, default = RGB)                        Tensor color format for the reader. Default is RGB format.
    @random_shuffle (bool, optional, default= False)                     Data would be randomly shuffled if set to True.
    @mode (type, optional, default = EXTSOURCE_FNAME)                    The Default mode would be External Source File Name. The External Source Mode can be FileName, Raw Compressed, Raw Uncompressed.
    @max_width (int, optional, default = 2000)                           The Max Width to which the source images would be decoded to.
    @max_height (int, optional, default = 2000)                          The Max Height to which the source images would be decoded to.
    """
    # pybind call arguments
    Pipeline._current_pipeline._is_external_source_operator = True
    Pipeline._current_pipeline._external_source = iter(source)
    Pipeline._current_pipeline._external_source_mode = mode
    Pipeline._current_pipeline._external_source_user_given_width = max_width
    Pipeline._current_pipeline._external_source_user_given_height = max_height
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch_repeated, stick_to_shard, shard_size)
    kwargs_pybind = {"rocal_color_format": color_format, "is_output": False, "shuffle": random_shuffle, "loop": False, "decode_size_policy": types.USER_GIVEN_SIZE,
                     "max_width": max_width, "max_height": max_height, "dec_type": types.DECODER_TJPEG, "external_source_mode": mode, "sharding_info": sharding_info}
    external_source_operator = b.externalFileSource(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (external_source_operator, [])  # Labels is Empty

def preemphasis_filter(*inputs, border=types.CLAMP, preemph_coeff=0.97, output_dtype=types.FLOAT):
    """
    Applies preemphasis filter to the input data.
    This filter, in simple form, can be expressed by the formula:
    Y[t] = X[t] - coeff * X[t-1]    if t > 1
    Y[t] = X[t] - coeff * X_border  if t == 0
    with X and Y being the input and output signal, respectively.
    The value of X_border depends on the border argument:
    X_border = 0                    if border_type == 'zero'
    X_border = X[0]                 if border_type == 'clamp'
    X_border = X[1]                 if border_type == 'reflect'

    @param inputs (list)                                                 The input sample to which preEmphasisFilter is applied.
    @param border                                                        The border value policy. The possible values are "CLAMP", "ZERO", "REFLECT"
    @param preemph_coeff (float , optional, default = 0.97)              The preEmphasisFilter co-efficient.
    @output_dtype (type, optional, default = types.FLOAT)                Tensor dtype for the augmentation output. Default is types.FLOAT.
    """
    preemph_coeff_float_param = b.createFloatParameter(preemph_coeff)
    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False,
                    "preemph_coeff": preemph_coeff_float_param, "preemph_border_type": border,
                    "output_dtype" :output_dtype}
    preemphasis_output = b.preEmphasisFilter(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return (preemphasis_output)

def spectrogram(*inputs, bytes_per_sample_hint = [0], center_windows = True, layout = types.NFT, nfft = None, power = 2, reflect_padding = True, seed = -1, window_fn = [], window_length = 512, window_step = 256, output_dtype = types.FLOAT) :
    """
    Produces a spectrogram from a 1D signal.
    Input data is expected to be one channel - The shape of the input can be (srcLength, 1) of the data type float32.
    """
    kwargs_pybind = {"input_audio": inputs[0], "is_output": False, "window_fn": window_fn, "center_windows": center_windows, "reflect_padding": reflect_padding,
                     "power": power, "nfft": nfft, "window_length": window_length, "window_step": window_step, "output_layout": layout, "output_dtype": output_dtype}
    spectrogram_output = b.spectrogram(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (spectrogram_output)

def to_decibels(*inputs, bytes_per_sample_hint = [0], cutoff_db = -200.0, multiplier = 10.0, reference = 0.0, seed = -1, output_dtype = types.FLOAT):
    '''
    Converts a magnitude (real, positive) to the decibel scale.

    Conversion is done according to the following formula:

    min_ratio = pow(10, cutoff_db / multiplier)
    out[i] = multiplier * log10( max(min_ratio, input[i] / reference) )
    '''
    kwargs_pybind = {"input_audio": inputs[0], "is_output": False, "cutoff_db": cutoff_db, "multiplier": multiplier, "reference_magnitude": reference, "rocal_tensor_output_type": output_dtype}
    decibel_scale = b.toDecibels(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return decibel_scale

def resample(*inputs, resample_rate=None, output_datatype=types.FLOAT, resample_hint=-1, quality=50.0):
    """
    Resamples an audio signal.

    The resampling is achieved by applying a sinc filter with Hann window with an extent controlled by the quality argument.
    """
    kwargs_pybind = {"input_audio": inputs[0], "resample_rate": resample_rate, "is_output": False, "resample_hint": resample_hint, "quality": quality, "output_datatype": output_datatype}
    resample_output = b.resample(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return resample_output

def tensor_add_tensor_float(*inputs, output_datatype=types.FLOAT):
    """
    Adds a rocalTensor with another rocalTensor.
    """
    kwargs_pybind = {"input_audio": inputs[0], "input_image1": inputs[1], "is_output": False, "output_datatype": output_datatype}
    tensor_add_tensor_float = b.tensorAddTensor(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return tensor_add_tensor_float

def tensor_mul_scalar_float(*inputs, scalar=1.0, output_datatype=types.FLOAT):
    """
    Multiplies a rocalTensor with a scalar float value.
    """
    kwargs_pybind = {"input_audio": inputs[0], "is_output": False, "scalar": scalar, "output_datatype": output_datatype}
    tensor_mul_scalar_float = b.tensorMulScalar(Pipeline._current_pipeline._handle ,*(kwargs_pybind.values()))
    return tensor_mul_scalar_float

def nonsilent_region(*inputs, cutoff_db = -60, reference_power = 0.0, reset_interval = 8192, window_length = 2048):
    """
    Performs leading and trailing silence detection in an audio buffer.
    @param cutoff_db (float)                                      The threshold, in dB, below which the signal is considered silent.
    @param reference_power (float)                                The reference power that is utilized to convert the signal to dB.
    @param reset_interval (int)                                   The number of samples after which the moving mean average is recalculated aiming to avoid loss of precision.
    @param window_length (int)                                    Size of the sliding window used in calculating the short-term power of the signal.
    """
    kwargs_pybind = {"input_audio": inputs[0], "is_output": False, "cutoff_db": cutoff_db,
                     "reference_power": reference_power, "reset_interval": reset_interval, "window_length": window_length}
    non_silent_region_output = b.nonSilentRegionDetection(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return non_silent_region_output.anchor, non_silent_region_output.shape

def slice(*inputs, anchor = [], shape = [], fill_values = [0.0],  out_of_bounds_policy = types.ERROR, rocal_tensor_output_type = types.FLOAT):
    """
    The slice can be specified by proving the start and end coordinates, or start coordinates and shape of the slice. Both coordinates and shapes can be provided in absolute or relative terms.
    @param anchor (int or 1D RocalTensor of ints)                                      The absolute starting co-ordinate points of the slice.
    @param shape (int or 1D RocalTensor of ints)                                       The absolute co-ordinate for the dimensions of the slice.
    @param fill_values (float or list of float)                                        Determines the padding values and is only relevant if out_of_bounds_policy is “pad” policy.
    @param out_of_bounds_policy ("error", "pad", "trim_to_shape")                      Determines the policy when slicing the out of bounds area of the input.
    @param rocal_tensor_output_type (float)                                            Output DataType of the Tensor
    """

    kwargs_pybind = {"input_audio0": inputs[0], "is_output": False, "anchor": anchor[0], "shape": shape[0], "fill_values": fill_values,
                     "out_of_bounds_policy": out_of_bounds_policy, "rocal_tensor_output_type": rocal_tensor_output_type}
    slice_output = b.slice(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return slice_output

def normalize(*inputs, axes=[], mean=[], stddev=[], scale=1.0, shift=0.0, output_datatype=types.FLOAT):
    '''
    Normalizes the input by removing the mean and dividing by the standard deviation.
    The mean and standard deviation can be calculated internally for the specified subset of axes or can be externally provided as the mean and stddev arguments.
    The normalization is done following the formula:
    out = scale * (in - mean) / stddev + shift
    The formula assumes that out and in are equally shaped tensors, but mean and stddev might be either tensors of same shape, scalars, or a mix of these.
    '''
    kwargs_pybind = {"input_tensor": inputs[0], "axes": axes, "mean": mean, "stddev": stddev, "is_output": False,
                     "scale": scale, "shift": shift, "output_datatype": output_datatype}
    normalize_output = b.normalize(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return normalize_output

def mel_filter_bank(*inputs, bytes_per_sample_hint = [0], freq_high = 0.0, freq_low = 0.0, mel_formula = types.MELSCALE_SLANEY, nfilter = 128, normalize = True, sample_rate = 44100.0, seed = -1, output_datatype = types.FLOAT):
    '''
    Converts a spectrogram to a mel spectrogram by applying a bank of triangular filters.
    The frequency ('f') dimension is selected from the input layout. In case of no layout, “f”, “ft”, or “*ft” is assumed, depending on the number of dimensions.
    '''
    kwargs_pybind = {"input_tensor": inputs[0], "is_output": False, "freq_high": freq_high, "freq_low": freq_low, "mel_formula": mel_formula,
                     "nfilter": nfilter, "normalize": normalize, "sample_rate": sample_rate, "output_datatype": output_datatype}
    mel_filter_bank_output = b.melFilterBank(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return mel_filter_bank_output

def transpose(*inputs, perm=[], output_layout=types.NHWC, output_dtype=types.UINT8):
    """
    Transposes the input according to the permutation passed from user.
    @param perm            Transpose permutation passed from the user
    @param output_layout   Output Layout of the Tensor
    @param output_dtype    Output DataType of the Tensor
    """
    kwargs_pybind = {"input_image": inputs[0], "perm": perm, "is_output": False, "output_layout": output_layout}
    transposed_image = b.transpose(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (transposed_image)
