.. meta::
  :description: ROCm Performance Primitives (RPP) supported functionalities
  :keywords: RPP, ROCm, Performance Primitives, documentation, support, functionalities, audio, image

****************************************************************************
ROCm Performance Primitives functionality and variant example outputs
****************************************************************************

The following table shows example outputs of some ROCm Performance Primitives (RPP) functionalities and variants. 


.. csv-table::
  :widths: 1, 2, 1
  :header: "Functionality/variant", "Input", "Output"

  "brightness", |orig_road|, |brightness|
  "gamma correction", |orig_road|, |gamma|
  "blend", |orig_road| |orig_dog|, |blend|
  "contrast", |orig_road|, |contrast|
  "pixelate", |orig_road|,  |pixel|
  "jitter", |orig_road|, |jitter|
  "noise", |orig_road|,  |noise|
  "fog", |orig_street|,  |fog|
  "rain", |orig_street|,  |rain|
  "exposure", |orig_road|,  |exposure|
  "flip", |orig_road|, |flip|
  "resize", |orig_road|, |resize|
  "rotate", |orig_road|, |rotate|
  "warp affine", |orig_road|, |warp_affine|
  "lens correction", |orig_question|, |lens_correct|
  "warp perspective", |orig_road|, |warp_perspective|
  "water", |orig_road|, |water|
  "non-linear blend", |orig_road| |orig_dog|, |nonlinear_blend|
  "color cast", |orig_road|, |color_cast|
  "erase", |orig_road|, |erase|
  "crop and patch", |orig_road| |orig_dog|, |crop_n_patch|
  "lut", |orig_road|, |lut|
  "glitch", |orig_road|, |glitch|
  "color twist", |orig_road|, |color_twist|
  "crop", |orig_road|, |crop|
  "crop mirror normalize", |orig_road|, |crop_mirror_norm|
  "resize crop mirror",  |orig_road|, |resize_crop_mirror|
  "erode", |orig_road|, |erode|
  "dilate", |orig_road|, |dilate|
  "color temperature", |orig_road|, |color_temp|
  "vignette", |orig_road|, |vignette|
  "box filter", |orig_road|, |box_filter|
  "gaussian filter", |orig_road|, |gaussian_filter|
  "magnitude", |orig_road| |orig_dog|, |magnitude|
  "phase", |orig_road| |orig_dog|, |phase|
  "bitwise AND", |orig_road| |orig_dog|, |bitwise_and|
  "bitwise NOT", |orig_road|, |bitwise_not|
  "bitwise exclusive OR", |orig_road| |orig_dog|, |bitwise_xor|
  "bitwise inclusive OR", |orig_road| |orig_dog|, |bitwise_or|
  "remap", |orig_road|, |remap|


.. |orig_dog| image:: ../data/doxygenInputs/img150x150_2.png 

.. |orig_flower| image:: ../data/doxygenInputs/img150x150_1.png

.. |orig_road| image:: ../data/doxygenInputs/img150x150.png

.. |orig_street| image:: ../data/doxygenInputs/img640x480.png

.. |orig_question| image:: ../data/doxygenInputs/lens_img640x480.png

.. |brightness| image:: ../data/doxygenOutputs/color_augmentations_brightness_img150x150.png

.. |gamma| image:: ../data/doxygenOutputs/color_augmentations_gamma_correction_img150x150.png

.. |blend| image:: ../data/doxygenOutputs/color_augmentations_blend_img150x150.png

.. |contrast| image:: ../data/doxygenOutputs/color_augmentations_contrast_img150x150.png

.. |pixel| image:: ../data/doxygenOutputs/effects_augmentations_pixelate_img150x150.png

.. |jitter| image:: ../data/doxygenOutputs/effects_augmentations_jitter_img150x150.png

.. |noise| image:: ../data/doxygenOutputs/effects_augmentations_gaussian_noise_img150x150.png

.. |fog| image:: ../data/doxygenOutputs/effects_augmentations_fog_img640x480.png

.. |rain| image:: ../data/doxygenOutputs/effects_augmentations_rain_img640x480.png

.. |exposure| image:: ../data/doxygenOutputs/color_augmentations_contrast_img150x150.png

.. |threshold| image:: ../data/doxygenOutputs/statistical_operations_threshold_img150x150.png

.. |flip| image:: ../data/doxygenOutputs/geometric_augmentations_flip_img150x150.png

.. |resize| image:: ../data/doxygenOutputs/geometric_augmentations_resize_img150x150.png

.. |resize_crop_mirror| image:: ../data/doxygenOutputs/geometric_augmentations_resize_crop_mirror_img115x115.png

.. |rotate| image:: ../data/doxygenOutputs/geometric_augmentations_rotate_img150x150.png

.. |warp_affine| image:: ../data/doxygenOutputs/geometric_augmentations_warp_affine_img150x150.png

.. |lens_correct| image:: ../data/doxygenOutputs/geometric_augmentations_lens_correction_img_640x480.png

.. |warp_perspective| image:: ../data/doxygenOutputs/geometric_augmentations_warp_perspective_img150x150.png

.. |water| image:: ../data/doxygenOutputs/effects_augmentations_water_img150x150.png

.. |nonlinear_blend| image:: ../data/doxygenOutputs/effects_augmentations_non_linear_blend_img150x150.png

.. |color_cast| image:: ../data/doxygenOutputs/color_augmentations_color_cast_img150x150.png

.. |erase| image:: ../data/doxygenOutputs/effects_augmentations_erase_img150x150.png

.. |crop_n_patch| image:: ../data/doxygenOutputs/geometric_augmentations_crop_and_patch_img150x150.png

.. |lut| image:: ../data/doxygenOutputs/color_augmentations_lut_img150x150.png

.. |glitch| image:: ../data/doxygenOutputs/effects_augmentations_glitch_img150x150.png

.. |color_twist| image:: ../data/doxygenOutputs/color_augmentations_color_twist_img150x150.png

.. |crop| image:: ../data/doxygenOutputs/geometric_augmentations_crop_img150x150.png

.. |crop_mirror_norm| image:: ../data/doxygenOutputs/geometric_augmentations_crop_mirror_normalize_img150x150.png

.. |erode| image:: ../data/doxygenOutputs/morphological_operations_erode_kSize5_img150x150.png

.. |dilate| image:: ../data/doxygenOutputs/morphological_operations_dilate_kSize5_img150x150.png

.. |color_temp| image:: ../data/doxygenOutputs/color_augmentations_color_temperature_img150x150.png

.. |vignette| image:: ../data/doxygenOutputs/effects_augmentations_vignette_img150x150.png

.. |box_filter| image:: ../data/doxygenOutputs/filter_augmentations_box_filter_kSize5_img150x150.png

.. |gaussian_filter| image:: ../data/doxygenOutputs/filter_augmentations_gaussian_filter_kSize5_img150x150.png

.. |magnitude| image:: ../data/doxygenOutputs/arithmetic_operations_magnitude_img150x150.png

.. |bitwise_and| image:: ../data/doxygenOutputs/bitwise_operations_bitwise_and_img150x150.png

.. |bitwise_not| image:: ../data/doxygenOutputs/bitwise_operations_bitwise_not_img150x150.png

.. |bitwise_or| image:: ../data/doxygenOutputs/bitwise_operations_bitwise_or_img150x150.png

.. |bitwise_xor| image:: ../data/doxygenOutputs/bitwise_operations_bitwise_xor_img150x150.png

.. |phase| image:: ../data/doxygenOutputs/geometric_augmentations_phase_img150x150.png

.. |remap| image:: ../data/doxygenOutputs/geometric_augmentations_remap_img150x150.png
