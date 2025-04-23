/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef MIVISIONX_ROCAL_API_META_DATA_H
#define MIVISIONX_ROCAL_API_META_DATA_H
#include "rocal_api_types.h"
#include <set>

/*!
 * \file
 * \brief The AMD rocAL Library - Meta Data
 *
 * \defgroup group_rocal_meta_data API: AMD rocAL - Meta Data API
 * \brief The AMD rocAL meta data functions.
 */

/*! \brief creates label reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the folder that contains the dataset or metadata file
 * \param file_list_path is the path to file list that contains the file names and its corresponding labels
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateLabelReader(RocalContext rocal_context, const char* source_path, const char* file_list_path = "");

/*! \brief creates video label reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the folder that contains the dataset or metadata file
 * \param [in] sequence_length The number of frames in a sequence.
 * \param [in] frame_step Frame interval between each sequence.
 * \param [in] frame_stride Frame interval between frames in a sequence.
 * \param [in] file_list_frame_num True : when the inputs from text file is to be considered as frame numbers. False : when the inputs from text file is to considered as timestamps.
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateVideoLabelReader(RocalContext rocal_context, const char* source_path, unsigned sequence_length, unsigned frame_step, unsigned frame_stride, bool file_list_frame_num = true);

/*! \brief create tf reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the coco json file
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateTFReader(RocalContext rocal_context, const char* source_path, bool is_output,
                                                            const char* user_key_for_label, const char* user_key_for_filename);

/*! \brief create tf reader detection
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context
 * \param [in] source_path path to the coco json file
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateTFReaderDetection(RocalContext rocal_context, const char* source_path, bool is_output,
                                                                     const char* user_key_for_label, const char* user_key_for_text,
                                                                     const char* user_key_for_xmin, const char* user_key_for_ymin, const char* user_key_for_xmax, const char* user_key_for_ymax,
                                                                     const char* user_key_for_filename);

/*! \brief create coco reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the coco json file
 * \param [in] mask enable polygon masks
 * \param [in] ltrb If set to True, bboxes are returned as [left, top, right, bottom]. If set to False, the bboxes are returned as [x, y, width, height]
 * \param [in] is_box_encoder If set to True, bboxes are returned as encoded bboxes using the anchors
 * \param [in] avoid_class_remapping If set to True, classes are returned directly. Otherwise, classes are mapped to consecutive values
 * \param [in] aspect_ratio_grouping If set to True, images are sorted by their aspect ratio and returned
 * \param [in] is_box_iou_matcher If set to True, box iou matcher which returns matched indices is enabled in the pipeline
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateCOCOReader(RocalContext rocal_context, const char* source_path, bool is_output, bool mask = false, bool ltrb = true, bool is_box_encoder = false, bool avoid_class_remapping = false, bool aspect_ratio_grouping = false, bool is_box_iou_matcher = false);

/*! \brief create coco reader key points
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the coco json file
 * \param [in] sigma  sigma used for gaussian distribution (needed for HRNet Pose estimation)
 * \param [in] pose_output_width output image width (needed for HRNet Pose estimation)
 * \param [in] pose_output_width output image height (needed for HRNet Pose estimation)
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateCOCOReaderKeyPoints(RocalContext rocal_context, const char* source_path, bool is_output, float sigma = 0.0, unsigned pose_output_width = 0, unsigned pose_output_height = 0);

/*! \brief create text file based label reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context
 * \param [in] source_path path to the file that contains the metadata file
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateTextFileBasedLabelReader(RocalContext rocal_context, const char* source_path);

/*! \brief create caffe LMDB label reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context
 * \param [in] source_path path to the Caffe LMDB records for Classification
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateCaffeLMDBLabelReader(RocalContext rocal_context, const char* source_path);

/*! \brief create caffe LMDB label reader for object detection
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the Caffe LMDB records for Object Detection
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateCaffeLMDBReaderDetection(RocalContext rocal_context, const char* source_path);

/*! \brief create caffe2 LMDB label reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the Caffe2LMDB records for Classification
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateCaffe2LMDBLabelReader(RocalContext rocal_context, const char* source_path, bool is_output);

/*! \brief create caffe2 LMDB label reader for object detection
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the Caffe2LMDB records for Object Detection
 * \return RocalMetaData object - can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateCaffe2LMDBReaderDetection(RocalContext rocal_context, const char* source_path, bool is_output);

/*! \brief create MXNet reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the MXNet recordio files for Classification
 * \return RocalMetaData object - can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateMXNetReader(RocalContext rocal_context, const char* source_path, bool is_output);

/*! \brief get image name
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf user buffer provided to be filled with output image names for images in the output batch.
 */
extern "C" void ROCAL_API_CALL rocalGetImageName(RocalContext rocal_context, char* buf);

/*! \brief get image name lengths
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf userbuffer provided to be filled with the length of the image names in the output batch
 * \return The size of the buffer needs to be provided by user to get the image names of the output batch
 */
extern "C" unsigned ROCAL_API_CALL rocalGetImageNameLen(RocalContext rocal_context, int* buf);

/*! \brief get image labels
 * \ingroup group_rocal_meta_data
 * \param [in] meta_data RocalMetaData object that contains info about the images and labels
 * \param [out] buf user's buffer that will be filled with labels. Its needs to be at least of size batch_size.
 * \return RocalTensorList of labels associated with image
 */
extern "C" RocalTensorList ROCAL_API_CALL rocalGetImageLabels(RocalContext rocal_context);

/*! \brief get bounding box count
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with number of object in the images.
 * \return The size of the buffer needs to be provided by user to get bounding box info for all images in the output batch.
 */
extern "C" unsigned ROCAL_API_CALL rocalGetBoundingBoxCount(RocalContext rocal_context);

/*! \brief get mask count
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf the imageIdx in the output batch
 * \return The size of the buffer needs to be provided by user to get mask box info associated with image_idx in the output batch.
 */
extern "C" unsigned ROCAL_API_CALL rocalGetMaskCount(RocalContext p_context, int* buf);

/*! \brief get mask coordinates
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] bufcount The user's buffer that will be filled with polygon size for the mask info
 * \return The tensorlist with the mask coordinates
 */
extern "C" RocalTensorList ROCAL_API_CALL rocalGetMaskCoordinates(RocalContext p_context, int* bufcount);

/*! \brief get bounding box label
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with bounding box label info for the images in the output batch. It needs to be of size returned by a call to the rocalGetBoundingBoxCount
 * \return RocalTensorList of labels associated with bounding box coordinates
 */
extern "C" RocalTensorList ROCAL_API_CALL rocalGetBoundingBoxLabel(RocalContext rocal_context);

/*! \brief get bounding box coordinates
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with bounding box coords info for the images in the output batch. It needs to be of size returned by a call to the rocalGetBoundingBoxCords
 * \return RocalTensorList of bounding box co-ordinates
 */
extern "C" RocalTensorList ROCAL_API_CALL rocalGetBoundingBoxCords(RocalContext rocal_context);

/*! \brief get image sizes
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with images sizes info for the images in the output batch
 */
extern "C" void ROCAL_API_CALL rocalGetImageSizes(RocalContext rocal_context, int* buf);

/*! \brief get ROI image sizes
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with ROI image size info for the images in the output batch
 */
extern "C" void ROCAL_API_CALL rocalGetROIImageSizes(RocalContext rocal_context, int* buf);

/*! \brief create text cifar10 label reader
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [in] source_path path to the file that contains the metadata file
 * \param [in] filename_prefix: look only files with prefix ( needed for cifar10)
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateTextCifar10LabelReader(RocalContext rocal_context, const char* source_path, const char* file_prefix);

/*! \brief get one hot image labels
 * \ingroup group_rocal_meta_data
 * \param [in] meta_data RocalMetaData object that contains info about the images and labels
 * \param [in] numOfClasses the number of classes for a image dataset
 * \param [out] buf user's buffer that will be filled with labels. Its needs to be at least of size batch_size.
 * \param [in] dest destination can be host=0 / device=1
 */
extern "C" void ROCAL_API_CALL rocalGetOneHotImageLabels(RocalContext rocal_context, void* buf, int numOfClasses, RocalOutputMemType output_mem_type);

extern "C" void ROCAL_API_CALL rocalRandomBBoxCrop(RocalContext p_context, bool all_boxes_overlap, bool no_crop, RocalFloatParam aspect_ratio = NULL, bool has_shape = false, int crop_width = 0, int crop_height = 0, int num_attempts = 1, RocalFloatParam scaling = NULL, int total_num_attempts = 0, int64_t seed = 0);

/*! \brief get sequence starting frame number
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with starting frame numbers of the output batch sequences.
 */
extern "C" void ROCAL_API_CALL rocalGetSequenceStartFrameNumber(RocalContext rocal_context, unsigned int* buf);

/*! \brief get sequence time stamps
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] buf The user's buffer that will be filled with frame timestamps of each of the frames in output batch sequences.
 */
extern "C" void ROCAL_API_CALL rocalGetSequenceFrameTimestamps(RocalContext rocal_context, float* buf);

/*! \brief rocal box encoder
 * \ingroup group_rocal_meta_data
 * \param [in] anchors  Anchors to be used for encoding, as the array of floats is in the ltrb format.
 * \param [in] criteria Threshold IoU for matching bounding boxes with anchors.  The value needs to be between 0 and 1.
 * \param [in] offset Returns normalized offsets ((encoded_bboxes*scale - anchors*scale) - mean) / stds in EncodedBBoxes that use std and the mean and scale arguments
 * \param [in] means [x y w h] mean values for normalization.
 * \param [in] stds [x y w h] standard deviations for offset normalization.
 * \param [in] scale Rescales the box and anchor values before the offset is calculated (for example, to return to the absolute values).
 */
extern "C" void ROCAL_API_CALL rocalBoxEncoder(RocalContext p_context, std::vector<float>& anchors, float criteria,
                                               std::vector<float>& means, std::vector<float>& stds, bool offset = false, float scale = 1.0);

/*! \brief copy encoded boxes and labels
 * \ingroup group_rocal_meta_data
 * \param [in] p_context rocal context
 * \param [out] boxes_buf  user's buffer that will be filled with encoded bounding boxes . Its needs to be at least of size batch_size.
 * \param [out] labels_buf  user's buffer that will be filled with encoded labels . Its needs to be at least of size batch_size.
 */
extern "C" void ROCAL_API_CALL rocalCopyEncodedBoxesAndLables(RocalContext p_context, float* boxes_buf, int* labels_buf);

/*! \brief
 * \ingroup group_rocal_meta_data
 * \param boxes_buf  ptr to user's buffer that will be filled with encoded bounding boxes . Its needs to be at least of size batch_size.
 * \param labels_buf  user's buffer that will be filled with encoded labels . Its needs to be at least of size batch_size.
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalGetEncodedBoxesAndLables(RocalContext p_context, int num_encoded_boxes);

/*! \brief get image id
 * \ingroup group_rocal_meta_data
 * \param rocal_context rocal context
 * \param buf The user's buffer that will be filled with image id info for the images in the output batch.
 */
extern "C" void ROCAL_API_CALL rocalGetImageId(RocalContext p_context, int* buf);

/*! \brief get joints data pointer
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] joints_data The user's RocalJointsData pointer that will be pointed to JointsDataBatch pointer
 */
extern "C" void ROCAL_API_CALL rocalGetJointsDataPtr(RocalContext p_context, RocalJointsData** joints_data);

/*! \brief API to enable box IOU matcher and pass required params to pipeline
 * \ingroup group_rocal_meta_data
 * \param [in] p_context rocAL context
 * \param [in] anchors The anchors / ground truth bounding box coordinates
 * \param [in] high_threshold The max threshold for IOU
 * \param [in] low_threshold The min threshold for IOU
 * \param [in] allow_low_quality_matches bool value when set to true allows low quality matches
 */
extern "C" void ROCAL_API_CALL rocalBoxIouMatcher(RocalContext p_context, std::vector<float>& anchors,
                                                  float high_threshold, float low_threshold, bool allow_low_quality_matches = true);

/*! \brief API to return the matched indices for the bounding box and anchors
 * \ingroup group_rocal_meta_data
 * \param [in] p_context rocAL context
 * \return RocalTensorList of matched indices
 */
extern "C" RocalTensorList ROCAL_API_CALL rocalGetMatchedIndices(RocalContext p_context);

/*! \brief creates webdataset reader
 * \ingroup group_rocal_meta_data
 * \param [in] p_context rocal context
 * \param [in] source_path path to the folder that contains the dataset
 * \param [in] index_path path to the folder that contains the index files
 * \param extensions [in] the extensions used in the tar files for parsing them
 * \param missing_components_behavior [in] The behaviour that determines what happens when any component in the sample is missing.
 * \param is_output [in] The output is set or not.
 * \return RocalMetaData object, can be used to inquire about the rocal's output (processed) tensors
 */
extern "C" RocalMetaData ROCAL_API_CALL rocalCreateWebDatasetReader(RocalContext p_context, const char* source_path, const char* index_path,
                                                                    std::vector<std::set<std::string>> extensions, RocalMissingComponentsBehaviour missing_components_behavior, bool is_output);

/*! \brief get joints data pointer
 * \ingroup group_rocal_meta_data
 * \param [in] rocal_context rocal context
 * \param [out] ascii_data The user's AsciiDatas pointer that will be pointed to AsciiDataBatch pointer
 */
RocalMetaData ROCAL_API_CALL rocalGetAsciiDatas(RocalContext p_context);

#endif  // MIVISIONX_ROCAL_API_META_DATA_H
