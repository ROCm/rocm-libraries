# Copyright (c) 2018 - 2023 Advanced Micro Devices, Inc. All rights reserved.
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
# @file readers.py
#
# @brief File containing reader functions for multiple datasets and data formats

import rocal_pybind as b
from amd.rocal.pipeline import Pipeline
import amd.rocal.types as types


def coco(annotations_file='', ltrb=True, masks=False, ratio=False, avoid_class_remapping=False,
         pixelwise_masks=False, is_box_encoder=False, is_box_iou_matcher=False, aspect_ratio_grouping=False, stick_to_shard=False, pad_last_batch=False):
    """!Creates a COCOReader node.

        @param annotations_file         Path to the COCO annotations file.
        @param ltrb                     Whether bounding box coordinates are provided in (left, top, right, bottom) format.
        @param masks                    Whether to read polygon masks from COCO annotations.
        @param ratio                    Whether bounding box coordinates are provided in normalized format.
        @param avoid_class_remapping    Specifies if class remapping should be avoided.
        @param pixelwise_masks          Whether to read mask data and generate pixel-wise masks.
        @param is_box_encoder           Whether to enable box encoder in the pipeline.
        @param is_box_iou_matcher       Whether to enable box IOU matcher in the pipeline.
        @param aspect_ratio_grouping    Whether to enable aspect ratio grouping in the pipeline.
        @param stick_to_shard           Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch           If set to True, pads the shard by repeating the last sample.

        @return    meta data, labels, and bounding boxes.
    """
    Pipeline._current_pipeline._reader = "COCOReader"
    # Output
    labels = []
    bboxes = []
    kwargs_pybind = {
        "source_path": annotations_file,
        "is_output": True,
        "mask": masks,
        "ltrb": ltrb,
        "is_box_encoder": is_box_encoder,
        "avoid_class_remapping": avoid_class_remapping,
        "aspect_ratio_grouping": aspect_ratio_grouping,
        "is_box_iou_matcher": is_box_iou_matcher}
    meta_data = b.cocoReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (meta_data, labels, bboxes)


def file(file_root, file_filters=None, file_list='', stick_to_shard=False, pad_last_batch=False):
    """!Creates a labelReader node for reading files from folder or file_list.

        @param file_root         Path to a directory that contains the data files.
        @param file_filters      A list of glob strings to filter the list of files in the sub-directories of the file_root.
        @param file_list         Path to a text file that contains one whitespace-separated filename label pair per line. The filenames are relative to the location of that file or to file_root, if specified.
        @param stick_to_shard    Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch    If set to True, pads the shard by repeating the last sample.

        @return    label reader meta data and labels.
    """
    Pipeline._current_pipeline._reader = "labelReader"
    # Output
    labels = []
    kwargs_pybind = {"source_path": file_root, "file_list": file_list}
    label_reader_meta_data = b.labelReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (label_reader_meta_data, labels)


def tfrecord(path, user_feature_key_map, features, reader_type=0, stick_to_shard=False, pad_last_batch=False):
    """!Creates a TFRecordReader node for loading TFRecord dataset.

        @param path                    Path to the TFRecord dataset.
        @param user_feature_key_map    User-provided feature key mapping.
        @param features                Features to load from TFRecords.
        @param reader_type             Type of reader (0 for classification, 1 for detection).
        @param stick_to_shard          Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch          If set to True, pads the shard by repeating the last sample.

        @return    Features loaded from TFRecords.
    """
    labels = []
    if reader_type == 1:
        Pipeline._current_pipeline._reader = "TFRecordReaderDetection"
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"],
                         "user_key_for_text": user_feature_key_map["image/class/text"], "user_key_for_xmin": user_feature_key_map["image/object/bbox/xmin"],
                         "user_key_for_ymin": user_feature_key_map["image/object/bbox/ymin"], "user_key_for_xmax": user_feature_key_map["image/object/bbox/xmax"],
                         "user_key_for_ymax": user_feature_key_map["image/object/bbox/ymax"], "user_key_for_filename": user_feature_key_map["image/filename"]}
        for key in features.keys():
            if key not in user_feature_key_map.keys():
                print(
                    "For Object Detection, ROCAL TFRecordReader needs all the following keys in the featureKeyMap:")
                print("image/encoded\nimage/class/label\nimage/class/text\nimage/object/bbox/xmin\nimage/object/bbox/ymin\nimage/object/bbox/xmax\nimage/object/bbox/ymax\n")
                exit()
        tf_meta_data = b.tfReaderDetection(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "TFRecordReaderClassification"
        kwargs_pybind = {"path": path, "is_output": True, "user_key_for_label": user_feature_key_map["image/class/label"],
                         "user_key_for_filename": user_feature_key_map["image/filename"]}
        for key in features.keys():
            if key not in user_feature_key_map.keys():
                print(
                    "For Image Classification, ROCAL TFRecordReader needs all the following keys in the featureKeyMap:")
                print("image/encoded\nimage/class/label\n")
                exit()
        tf_meta_data = b.tfReader(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    features["image/encoded"] = tf_meta_data
    features["image/class/label"] = labels
    return features


def caffe(path, bbox=False, stick_to_shard=False, pad_last_batch=False):
    """!Creates a CaffeReader node for loading Caffe dataset.

        @param path              Path to the Caffe dataset.
        @param bbox              Type of reader (False for classification, True for detection).
        @param stick_to_shard    Determines whether the reader should stick to a data shard instead of going through the entire dataset
        @param pad_last_batch    If set to True, pads the shard by repeating the last sample.

        @return    caffe reader meta data, bboxes, and labels.
    """
    # Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path}
    # Node Object
    if (bbox == True):
        Pipeline._current_pipeline._reader = "CaffeReaderDetection"
        caffe_reader_meta_data = b.caffeReaderDetection(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "CaffeReader"
        caffe_reader_meta_data = b.caffeReader(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))

    if (bbox == True):
        return (caffe_reader_meta_data, bboxes, labels)
    else:
        return (caffe_reader_meta_data, labels)


def caffe2(path, bbox=False, stick_to_shard=False, pad_last_batch=False):
    """!Creates a Caffe2Reader node for loading Caffe2 dataset.

        @param path              Path to the Caffe2 dataset.
        @param bbox              Type of reader (False for classification, True for detection).
        @param stick_to_shard    Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch    If set to True, pads the shard by repeating the last sample.

        @return    caffe2 reader meta data, bboxes, and labels.
    """
    # Output
    bboxes = []
    labels = []
    kwargs_pybind = {"source_path": path, "is_output": True}
    if (bbox == True):
        Pipeline._current_pipeline._reader = "Caffe2ReaderDetection"
        caffe2_meta_data = b.caffe2ReaderDetection(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    else:
        Pipeline._current_pipeline._reader = "Caffe2Reader"
        caffe2_meta_data = b.caffe2Reader(
            Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    if (bbox == True):
        return (caffe2_meta_data, bboxes, labels)
    else:
        return (caffe2_meta_data, labels)


def video(sequence_length, file_list_frame_num=False, file_root="", image_type=types.RGB, num_shards=1,
          random_shuffle=False, step=1, stride=1, decoder_mode=types.SOFTWARE_DECODE, enable_frame_num=False,
          enable_timestamps=False, file_list="", stick_to_shard=False, pad_last_batch=False,
          file_list_include_preceding_frame=False, normalized=False, skip_vfr_check=False, 
          last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False, shard_size=-1, decoder_type=types.DECODER_VIDEO_FFMPEG_SW):
    """!Creates a VideoDecoder node for loading video sequences.

        @param sequence_length                      Number of frames in video sequence.
        @param file_list_frame_num                  Specifies whether file list includes frame numbers.
        @param file_root                            Root directory containing video files.
        @param image_type                           Color format of the frames.
        @param num_shards                           Number of shards for data parallelism.
        @param random_shuffle                       Specifies if frames should be randomly shuffled.
        @param step                                 Distance between first frames of consecutive sequences.
        @param stride                               Distance between consecutive frames in a sequence.
        @param decoder_mode                         Device used for video decoding.
        @param enable_frame_num                     Specifies whether frame numbers are enabled.
        @param enable_timestamps                    Specifies whether timestamps are enabled.
        @param file_list                            List of video files.
        @param stick_to_shard                       Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch                       If set to True, pads the shard by repeating the last sample.
        @param file_list_include_preceding_frame    Changes the behavior how file_list start and end frame timestamps are translated to a frame number.
        @param normalized                           Gets the output as normalized data.
        @param skip_vfr_check                       Skips the check for the variable frame rate (VFR) videos.
        @param decoder_type                         Type of decoder used for video decoding, Currently supports FFmpeg/rocDecode

        @return   list of loaded video sequences.
    """
    Pipeline._current_pipeline._reader = "VideoDecoder"
    # Output
    videos = []
    kwargs_pybind_reader = {
        "source_path": file_root,
        "sequence_length": sequence_length,
        "frame_step": step,
        "frame_stride": stride,
        "file_list_frame_num": file_list_frame_num}  # VideoMetaDataReader
    b.videoMetaDataReader(Pipeline._current_pipeline._handle,
                          *(kwargs_pybind_reader.values()))
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch, stick_to_shard, shard_size)
    kwargs_pybind_decoder = {
        "source_path": file_root,
        "color_format": image_type,
        "decoder_mode": decoder_mode,
        "shard_count": num_shards,
        "sequence_length": sequence_length,
        "shuffle": random_shuffle,
        "is_output": False,
        "loop": False,
        "rocal_decoder_type": decoder_type,
        "frame_step": step,
        "frame_stride": stride,
        "file_list_frame_num": file_list_frame_num,
        "sharding_info": sharding_info}  # VideoDecoder
    videos = b.videoDecoder(
        Pipeline._current_pipeline._handle, *(kwargs_pybind_decoder.values()))
    return (videos)


def video_resize(sequence_length, resize_width, resize_height, file_list_frame_num=False,
                 file_root="", image_type=types.RGB,
                 num_shards=1, random_shuffle=False, step=3,
                 stride=3, decoder_mode=types.SOFTWARE_DECODE,
                 scaling_mode=types.SCALING_MODE_DEFAULT, interpolation_type=types.LINEAR_INTERPOLATION,
                 resize_longer=0, resize_shorter=0, max_size=[], enable_frame_num=False,
                 enable_timestamps=False, file_list="", stick_to_shard=True, pad_last_batch=False,
                 file_list_include_preceding_frame=False, normalized=False, skip_vfr_check=False, 
                 last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False, shard_size=-1,
                 decoder_type=types.DECODER_VIDEO_FFMPEG_SW):
    """!Creates a VideoDecoderResize node in the pipeline for loading and resizing video sequences.

        @param sequence_length                      Number of frames in video sequence.
        @param resize_width                         output width for resizing.
        @param resize_height                        output height for resizing.
        @param file_list_frame_num                  Specifies whether file list includes frame numbers.
        @param file_root                            Root directory containing video files.
        @param image_type                           Color format of the frames.
        @param num_shards                           Number of shards for data parallelism.
        @param random_shuffle                       Specifies if frames should be randomly shuffled.
        @param step                                 Distance between first frames of consecutive sequences.
        @param stride                               Distance between consecutive frames in a sequence.
        @param decoder_mode                         Device used for video decoding.
        @param scaling_mode                         Scaling mode for resizing.
        @param interpolation_type                   Interpolation type for resizing.
        @param resize_longer                        Target size for the longer dimension during resizing.
        @param resize_shorter                       Target size for the shorter dimension during resizing.
        @param max_size                             Maximum size for resizing.
        @param enable_frame_num                     Specifies whether frame numbers are enabled.
        @param enable_timestamps                    Specifies whether timestamps are enabled.
        @param file_list                            List of video files.
        @param stick_to_shard                       Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch                       If set to True, pads the shard by repeating the last sample.
        @param file_list_include_preceding_frame    Specifies if file list includes preceding frames.
        @param normalized                           Gets the output as normalized data.
        @param skip_vfr_check                       Skips the check for the variable frame rate (VFR) videos.
        @param decoder_type                         Type of decoder used for video decoding, Currently supports FFmpeg/rocDecode

        @returns   loaded and resized video sequences and meta data.
    """
    Pipeline._current_pipeline._reader = "VideoDecoderResize"
    # Output
    videos = []
    kwargs_pybind_reader = {
        "source_path": file_root,
        "sequence_length": sequence_length,
        "frame_step": step,
        "frame_stride": stride,
        "file_list_frame_num": file_list_frame_num}  # VideoMetaDataReader
    meta_data = b.videoMetaDataReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind_reader.values()))
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch_repeated, stick_to_shard, shard_size)
    kwargs_pybind_decoder = {"source_path": file_root, "color_format": image_type, "decoder_mode": decoder_mode, "shard_count": num_shards,
                             "sequence_length": sequence_length, "resize_width": resize_width, "resize_height": resize_height,
                             "shuffle": random_shuffle, "is_output": False, "loop": False, "rocal_decoder_type": decoder_type, "frame_step": step, "frame_stride": stride,
                             "file_list_frame_num": file_list_frame_num, "scaling_mode": scaling_mode, "max_size": max_size,
                             "resize_shorter": resize_shorter, "resize_longer": resize_longer, "interpolation_type": interpolation_type, "sharding_info": sharding_info}
    videos = b.videoDecoderResize(
        Pipeline._current_pipeline._handle, *(kwargs_pybind_decoder.values()))
    return (videos, meta_data)


def sequence_reader(file_root, sequence_length, image_type=types.RGB, num_shards=1, random_shuffle=False, step=3, stride=1, stick_to_shard=False, last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch_repeated=False, shard_size=-1):
    """!Creates a SequenceReader node for loading image sequences.

        @param file_root            Root directory containing image sequences.
        @param sequence_length      Number of frames in each sequence.
        @param image_type           Color format of the frames.
        @param num_shards           Number of shards for data parallelism.
        @param random_shuffle       Specifies if frames should be randomly shuffled.
        @param step                 Distance between first frames of consecutive sequences.
        @param stride               Distance between consecutive frames in a sequence.
        @param stick_to_shard       Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch       If set to True, pads the shard by repeating the last sample.

        @return    list of loaded image sequences.
    """
    Pipeline._current_pipeline._reader = "SequenceReader"
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch_repeated, stick_to_shard, shard_size)
    # Output
    
    kwargs_pybind = {
        "source_path": file_root,
        "color_format": image_type,
        "shard_count": num_shards,
        "sequence_length": sequence_length,
        "is_output": False,
        "shuffle": random_shuffle,
        "loop": False,
        "frame_step": step,
        "frame_stride": stride,
        "sharding_info": sharding_info}
    frames = b.sequenceReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (frames)


def mxnet(path, stick_to_shard=False, pad_last_batch=False):
    """!Creates an MXNETReader node for reading data from MXNet record files.

        @param path              Path to the MXNet record files.
        @param stick_to_shard    Determines whether the reader should stick to a data shard instead of going through the entire dataset.
        @param pad_last_batch    If set to True, pads the shard by repeating the last sample.

        @return    Metadata and loaded data from the MXNet record file.
    """
    Pipeline._current_pipeline._reader = "MXNETReader"
    # Output
    kwargs_pybind = {
        "source_path": path,
        "is_output": True
    }
    mxnet_metadata = b.mxnetReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return mxnet_metadata

def webdataset(path, index_paths="", ext = None, missing_components_behavior = types.MISSING_COMPONENT_ERROR):
    """!Creates an WebDataset node for reading data from tar files.

        @param path                         Path to the tar files.
        @param index_paths                  Index Path to index files
        @param missing_components_behavior  Tells what to do with output tensor data when any compoenet is missing - THROW_ERROR, SKIP, EMPTY_OUTPUT

        @return    Metadata and loaded data from the tar file.
    """
    Pipeline._current_pipeline._reader = "WebDataset"
     # Output
    kwargs_pybind = {
        "source_path": path,
        "index_path": index_paths,
        "ext": ext,
        "missing_components_behavior": missing_components_behavior,
        "is_output": True
    }
    webdata_metadata = b.webDatasetReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return webdata_metadata

def numpy(*inputs, file_root='', files=[], num_shards=1, output_layout=types.NONE, 
          random_shuffle=False, shard_id=0, stick_to_shard=True, shard_size=-1,
          last_batch_policy=types.LAST_BATCH_FILL, pad_last_batch=True, seed=0):

    Pipeline._current_pipeline._reader = "NumpyReader"
    Pipeline._current_pipeline._last_batch_policy = last_batch_policy
    sharding_info = b.RocalShardingInfo(last_batch_policy, pad_last_batch, stick_to_shard, shard_size)
    # Output
    kwargs_pybind = {"source_path": file_root, "output_layout": output_layout, "files": files, "is_output": False, "shuffle": random_shuffle,
                     "loop": False, "shard_id": shard_id, "shard_count": num_shards, "seed": seed, "sharding_info": sharding_info}
    numpy_reader_output = b.numpyReader(
        Pipeline._current_pipeline._handle, *(kwargs_pybind.values()))
    return (numpy_reader_output)
