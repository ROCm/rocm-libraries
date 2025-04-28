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
# @file decoders.py
#
# @brief  File containing various decoder implementations for various readers

import amd.rocal.types as types
import rocal_pybind as b
from amd.rocal.pipeline import Pipeline


def image(*inputs, user_feature_key_map=None, path='', file_root='', annotations_file='', index_path ='', shard_id=0, num_shards=1, random_shuffle=False,
          output_type=types.RGB, decoder_type=types.DECODER_TJPEG, device=None,
          decode_size_policy=types.USER_GIVEN_SIZE_ORIG, max_decoded_width=1000, max_decoded_height=1000,
          last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch=True, stick_to_shard=True, shard_size=-1):
    """!Decodes images using different readers and decoders.

        @param inputs                   list of input images.
        @param user_feature_key_map     User-provided feature key mapping.
        @param path                     Path to image source.
        @param file_root                Root path for image files.
        @param annotations_file         Path to annotations file.
        @param shard_id                 Shard ID for parallel processing.
        @param num_shards               Total number of shards for parallel processing.
        @param random_shuffle           Whether to shuffle images randomly.
        @param output_type              Color format of the output image.
        @param decoder_type             Type of image decoder to use.
        @param device                   Device to use for decoding ("gpu" or "cpu").
        @param decode_size_policy       Size policy for decoding images.
        @param max_decoded_width        Maximum width for decoded images.
        @param max_decoded_height       Maximum height for decoded images.

        @return    Decoded and preprocessed image.
    """
    reader = Pipeline._current_pipeline._reader
    Pipeline._current_pipeline._last_batch_policy = last_batch_policy
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch, stick_to_shard, shard_size)

    if (device == "gpu"):
        decoder_type = types.DECODER_ROCJPEG
    else:
        decoder_type = types.DECODER_TJPEG
    if (reader == 'COCOReader'):
        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.cocoImageDecoderShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif (reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "num_shards": num_shards,
            'is_output': False,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.tfImageDecoder(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif (reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.caffe2ImageDecoderShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif reader == "CaffeReader" or reader == "CaffeReaderDetection":
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.caffeImageDecoderShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    elif reader == "MXNETReader":
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.mxnetDecoder(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif reader == "WebDataset":
        kwargs_pybind = {
            "source_path": file_root,
            "index_path": index_path,
            "output_type": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.webdatasetSourceSingleShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        decoded_image = b.imageDecoderShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (decoded_image)


def image_raw(*inputs, user_feature_key_map=None, path='', random_shuffle=False, output_type=types.RGB, max_decoded_width=1000, max_decoded_height=1000,
              last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch=True, stick_to_shard=True, shard_size=-1):
    """!Decodes raw images for TF reader and decoder.

        @param inputs                  list of input images.
        @param user_feature_key_map    User-provided feature key mapping.
        @param path                    Path to image source.
        @param random_shuffle          Whether to shuffle images randomly.
        @param output_type             Color format of the output image.
        @param max_decoded_width       Maximum width for decoded images.
        @param max_decoded_height      Maximum height for decoded images.

        @return    Decoded raw image.
    """
    reader = Pipeline._current_pipeline._reader
    Pipeline._current_pipeline._last_batch_policy = last_batch_policy
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch, stick_to_shard, shard_size)

    if (reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "color_format": output_type,
            'is_output': False,
            "shuffle": random_shuffle,
            "loop": False,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        decoded_image = b.tfImageDecoderRaw(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
        return (decoded_image)


def image_random_crop(*inputs, user_feature_key_map=None, path='', file_root='', annotations_file='', num_shards=1, shard_id=0,
                      random_shuffle=False, num_attempts=10, output_type=types.RGB, random_area=[0.08, 1.0],
                      random_aspect_ratio=[0.8, 1.25], decode_size_policy=types.USER_GIVEN_SIZE_ORIG,
                      max_decoded_width=1000, max_decoded_height=1000, decoder_type=types.DECODER_TJPEG,
                      last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch=True, stick_to_shard=True, shard_size=-1):
    """!Applies random cropping to images using different readers and decoders.

        @param inputs                  list of input images.
        @param user_feature_key_map    User-provided feature key mapping.
        @param path                    Path to image source.
        @param file_root               Root path for image files.
        @param annotations_file        Path to annotations file.
        @param num_shards              Total number of shards for parallel processing.
        @param shard_id                Shard ID for parallel processing.
        @param random_shuffle          Whether to shuffle images randomly.
        @param num_attempts            Maximum number of attempts to find a valid crop.
        @param output_type             Color format of the output image.
        @param random_area             Random areas for cropping.
        @param random_aspect_ratio     Random aspect ratios for cropping.
        @param decode_size_policy      Size policy for decoding images.
        @param max_decoded_width       Maximum width for decoded images.
        @param max_decoded_height      Maximum height for decoded images.
        @param decoder_type            Type of image decoder to use.

        @return    Randomly cropped and preprocessed image.
    """
    reader = Pipeline._current_pipeline._reader
    Pipeline._current_pipeline._last_batch_policy = last_batch_policy
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch, stick_to_shard, shard_size)

    # Internally calls the C++ Partial decoder's
    if (reader == 'COCOReader'):
        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        crop_output_image = b.cocoImageDecoderSliceShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "TFRecordReaderClassification" or reader == "TFRecordReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "num_shards": num_shards,
            'is_output': False,
            "user_key_for_encoded": user_feature_key_map["image/encoded"],
            "user_key_for_filename": user_feature_key_map["image/filename"],
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "dec_type": decoder_type,
            "sharding_info": sharding_info}
        crop_output_image = b.tfImageDecoder(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "CaffeReader" or reader == "CaffeReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        crop_output_image = b.caffeImageDecoderPartialShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        crop_output_image = b.caffe2ImageDecoderPartialShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        crop_output_image = b.fusedDecoderCropShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    return (crop_output_image)


def image_slice(*inputs, file_root='', path='', annotations_file='', shard_id=0, num_shards=1, random_shuffle=False,
                random_aspect_ratio=[0.75, 1.33333], random_area=[0.08, 1.0], num_attempts=100, output_type=types.RGB,
                decode_size_policy=types.USER_GIVEN_SIZE_ORIG, max_decoded_width=1000, max_decoded_height=1000, last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch=True, stick_to_shard=True, shard_size=-1):
    """!Slices images randomly using different readers and decoders.

        @param inputs                 list of input images.
        @param file_root              Root path for image files.
        @param path                   Path to image source.
        @param annotations_file       Path to annotations file.
        @param shard_id               Shard ID for parallel processing.
        @param num_shards             Total number of shards for parallel processing.
        @param random_shuffle         Whether to shuffle images randomly.
        @param random_aspect_ratio    Random aspect ratios for cropping.
        @param random_area            Random areas for cropping.
        @param num_attempts           Maximum number of attempts to find a valid crop.
        @param output_type            Color format of the output image.
        @param decode_size_policy     Size policy for decoding images.
        @param max_decoded_width      Maximum width for decoded images.
        @param max_decoded_height     Maximum height for decoded images.

        @return    Sliced image.
    """
    reader = Pipeline._current_pipeline._reader
    Pipeline._current_pipeline._last_batch_policy = last_batch_policy
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch, stick_to_shard, shard_size)

    # Reader -> Randon BBox Crop -> ImageDecoderSlice
    # Random crop parameters taken from pytorch's RandomResizedCrop default function arguments
    # TODO:To pass the crop co-ordinates from random_bbox_crop to image_slice
    # in tensor branch integration,
    # for now calling partial decoder to match SSD training outer API's .
    if (reader == 'COCOReader'):

        kwargs_pybind = {
            "source_path": file_root,
            "json_path": annotations_file,
            "color_format": output_type,
            "shard_id": shard_id,
            "shard_count": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        image_decoder_slice = b.cocoImageDecoderSliceShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "CaffeReader" or reader == "CaffeReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        image_decoder_slice = b.caffeImageDecoderPartialShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    elif (reader == "Caffe2Reader" or reader == "Caffe2ReaderDetection"):
        kwargs_pybind = {
            "source_path": path,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        image_decoder_slice = b.caffe2ImageDecoderPartialShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        kwargs_pybind = {
            "source_path": file_root,
            "color_format": output_type,
            "shard_id": shard_id,
            "num_shards": num_shards,
            'is_output': False,
            "area_factor": random_area,
            "aspect_ratio": random_aspect_ratio,
            "num_attempts": num_attempts,
            "shuffle": random_shuffle,
            "loop": False,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_width,
            "max_height": max_decoded_height,
            "sharding_info": sharding_info}
        image_decoder_slice = b.fusedDecoderCropShard(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (image_decoder_slice)

def audio(*inputs, file_root='', file_list_path='', bytes_per_sample_hint=[0], shard_id=0, num_shards=1, random_shuffle=False, downmix=False, dtype=types.FLOAT, quality=50.0, sample_rate=0.0, seed=1, stick_to_shard=True, shard_size=-1, last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False,
          decode_size_policy=types.MAX_SIZE, max_decoded_samples=522320, max_decoded_channels=1):
    """!Decodes wav audio files.

        @param inputs                   list of input audio.
        @param file_root                Folder Path to the audio data.
        @param file_list_path           Text file containing list of files and the labels
        @param shard_id                 Shard ID for parallel processing.
        @param num_shards               Total number of shards for parallel processing.
        @param random_shuffle           Whether to shuffle images randomly.
        @param downmix                  Converts the audio data to single channel when enabled 
        @param dtype                    Data type of the decoded audio
        @param quality                  Resampling quality
        @param sample_rate              Sample rate for the decoded audio.
        @param seed                     Random seed.
        @param stick_to_shard           The reader sticks to the data for it's corresponding shard when enabled
        @param shard_size               Number of files in an epoch
        @param last_batch_policy        Determines the handling of the last batch when the shard size is not divisible by the batch size. Check types.py enum for possible values.
        @param pad_last_batch_repeated  If set to True, pads the shards last batch by repeating the last sample's data (dummy data).
        @param decode_size_policy       Size policy for decoding images.
        @param max_decoded_samples      Maximum samples for decoded images.
        @param max_decoded_channels     Maximum channels for decoded images.
        @return                         Decoded audio.
    """
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch_repeated, stick_to_shard, shard_size)
    kwargs_pybind = {
            "source_path": file_root,
            "source_file_list_path": file_list_path,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "is_output": False,
            "shuffle": random_shuffle,
            "loop": False,
            "downmix": downmix,
            "decode_size_policy": decode_size_policy,
            "max_width": max_decoded_samples,
            "max_height": max_decoded_channels,
            "sharding_info": sharding_info}
    Pipeline._current_pipeline._last_batch_policy = last_batch_policy
    decoded_audio = b.audioDecoderSingleShard(Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return decoded_audio

