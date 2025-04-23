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

#ifndef MIVISIONX_ROCAL_API_DATA_LOADERS_H
#define MIVISIONX_ROCAL_API_DATA_LOADERS_H
#include "rocal_api_types.h"

/*!
 * \file
 * \brief The AMD rocAL Library - Data Loaders
 *
 * \defgroup group_rocal_data_loaders API: AMD rocAL - Data Loaders API
 * \brief The AMD rocAL data loader functions.
 */

/*! \brief Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded tensors to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegFileSource(RocalContext context,
                                                          const char* source_path,
                                                          RocalImageColor rocal_color_format,
                                                          unsigned internal_shard_count,
                                                          bool is_output,
                                                          bool shuffle = false,
                                                          bool loop = false,
                                                          RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                          unsigned max_width = 0, unsigned max_height = 0, RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                          RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded tensor to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.

 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegFileSourceSingleShard(RocalContext context,
                                                                     const char* source_path,
                                                                     RocalImageColor rocal_color_format,
                                                                     unsigned shard_id,
                                                                     unsigned shard_count,
                                                                     bool is_output,
                                                                     bool shuffle = false,
                                                                     bool loop = false,
                                                                     RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                     unsigned max_width = 0, unsigned max_height = 0, RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                     RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder. Reads [Frames] sequences from a directory representing a collection of streams.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images in a sequence will be decoded to.
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances.
 * \param [in] sequence_length: The number of frames in a sequence.
 * \param [in] is_output Determines if the user wants the loaded sequences to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the sequences or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] step: Frame interval between each sequence.
 * \param [in] stride: Frame interval between frames in a sequence.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor.
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSequenceReader(RocalContext context,
                                                          const char* source_path,
                                                          RocalImageColor rocal_color_format,
                                                          unsigned internal_shard_count,
                                                          unsigned sequence_length,
                                                          bool is_output,
                                                          bool shuffle = false,
                                                          bool loop = false,
                                                          unsigned step = 0,
                                                          unsigned stride = 0,
                                                          RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder. Reads [Frames] sequences from a directory representing a collection of streams. It accepts external sharding information to load a singe shard only.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images in a sequence will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] sequence_length: The number of frames in a sequence.
 * \param [in] is_output Determines if the user wants the loaded sequences to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] step: Frame interval between each sequence.
 * \param [in] stride: Frame interval between frames in a sequence.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalSequenceReaderSingleShard(RocalContext context,
                                                                     const char* source_path,
                                                                     RocalImageColor rocal_color_format,
                                                                     unsigned shard_id,
                                                                     unsigned shard_count,
                                                                     unsigned sequence_length,
                                                                     bool is_output,
                                                                     bool shuffle = false,
                                                                     bool loop = false,
                                                                     unsigned step = 0,
                                                                     unsigned stride = 0,
                                                                     RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief JPEG image reader and decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] json_path Path to the COCO Json File
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCOCOFileSource(RocalContext context,
                                                              const char* source_path,
                                                              const char* json_path,
                                                              RocalImageColor color_format,
                                                              unsigned internal_shard_count,
                                                              bool is_output,
                                                              bool shuffle = false,
                                                              bool loop = false,
                                                              RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                              unsigned max_width = 0, unsigned max_height = 0,
                                                              RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                              RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] json_path Path to the COCO Json File
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
 * \param [in] aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
 * \param [in] num_attempts Maximum number of attempts to generate crop. Default 10
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCOCOFileSourcePartial(RocalContext p_context,
                                                                     const char* source_path,
                                                                     const char* json_path,
                                                                     RocalImageColor rocal_color_format,
                                                                     unsigned internal_shard_count,
                                                                     bool is_output,
                                                                     std::vector<float>& area_factor,
                                                                     std::vector<float>& aspect_ratio,
                                                                     unsigned num_attempts,
                                                                     bool shuffle = false,
                                                                     bool loop = false,
                                                                     RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                     unsigned max_width = 0, unsigned max_height = 0,
                                                                     RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] json_path Path to the COCO Json File
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
 * \param [in] aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCOCOFileSourcePartialSingleShard(RocalContext p_context,
                                                                                const char* source_path,
                                                                                const char* json_path,
                                                                                RocalImageColor rocal_color_format,
                                                                                unsigned shard_id,
                                                                                unsigned shard_count,
                                                                                bool is_output,
                                                                                std::vector<float>& area_factor,
                                                                                std::vector<float>& aspect_ratio,
                                                                                unsigned num_attempts,
                                                                                bool shuffle = false,
                                                                                bool loop = false,
                                                                                RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                                unsigned max_width = 0, unsigned max_height = 0,
                                                                                RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader. It allocates the resources and objects required to read and decode COCO Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] json_path Path to the COCO Json File
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCOCOFileSourceSingleShard(RocalContext context,
                                                                         const char* source_path,
                                                                         const char* json_path,
                                                                         RocalImageColor color_format,
                                                                         unsigned shard_id,
                                                                         unsigned shard_count,
                                                                         bool is_output,
                                                                         bool shuffle = false,
                                                                         bool loop = false,
                                                                         RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                         unsigned max_width = 0, unsigned max_height = 0,
                                                                         RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                         RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe LMDB Records. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCaffeLMDBRecordSource(RocalContext context,
                                                                     const char* source_path,
                                                                     RocalImageColor rocal_color_format,
                                                                     unsigned internal_shard_count,
                                                                     bool is_output,
                                                                     bool shuffle = false,
                                                                     bool loop = false,
                                                                     RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                     unsigned max_width = 0, unsigned max_height = 0,
                                                                     RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                     RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCaffeLMDBRecordSourceSingleShard(RocalContext p_context,
                                                                                const char* source_path,
                                                                                RocalImageColor rocal_color_format,
                                                                                unsigned shard_id,
                                                                                unsigned shard_count,
                                                                                bool is_output,
                                                                                bool shuffle = false,
                                                                                bool loop = false,
                                                                                RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                                unsigned max_width = 0, unsigned max_height = 0,
                                                                                RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                                RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCaffe2LMDBRecordSource(RocalContext context,
                                                                      const char* source_path,
                                                                      RocalImageColor rocal_color_format,
                                                                      unsigned internal_shard_count,
                                                                      bool is_output,
                                                                      bool shuffle = false,
                                                                      bool loop = false,
                                                                      RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                      unsigned max_width = 0, unsigned max_height = 0,
                                                                      RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                      RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored on the Caffe2 LMDB Records. It accepts external sharding information to load a singe shard. only
 * \ingroup group_rocal_data_loaders
 * \param [in] p_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCaffe2LMDBRecordSourceSingleShard(RocalContext p_context,
                                                                                 const char* source_path,
                                                                                 RocalImageColor rocal_color_format,
                                                                                 unsigned shard_id,
                                                                                 unsigned shard_count,
                                                                                 bool is_output,
                                                                                 bool shuffle = false,
                                                                                 bool loop = false,
                                                                                 RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                                 unsigned max_width = 0, unsigned max_height = 0,
                                                                                 RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                                 RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder for MXNet records. It allocates the resources and objects required to read and decode Jpeg images stored in MXNet Records. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalMXNetRecordSource(RocalContext context,
                                                             const char* source_path,
                                                             RocalImageColor rocal_color_format,
                                                             unsigned internal_shard_count,
                                                             bool is_output,
                                                             bool shuffle = false,
                                                             bool loop = false,
                                                             RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                             unsigned max_width = 0, unsigned max_height = 0,
                                                             RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                             RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and decoder for MXNet records. It allocates the resources and objects required to read and decode Jpeg images stored on the MXNet records. It accepts external sharding information to load a singe shard. only
 * \ingroup group_rocal_data_loaders
 * \param [in] p_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalMXNetRecordSourceSingleShard(RocalContext p_context,
                                                                        const char* source_path,
                                                                        RocalImageColor rocal_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                        unsigned max_width = 0, unsigned max_height = 0,
                                                                        RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                        RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored and Crops t
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] num_threads Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
 * \param [in] aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
 * \param [in] num_attempts Maximum number of attempts to generate crop. Default 10
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFusedJpegCrop(RocalContext context,
                                                         const char* source_path,
                                                         RocalImageColor rocal_color_format,
                                                         unsigned num_threads,
                                                         bool is_output,
                                                         std::vector<float>& area_factor,
                                                         std::vector<float>& aspect_ratio,
                                                         unsigned num_attempts,
                                                         bool shuffle = false,
                                                         bool loop = false,
                                                         RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                         unsigned max_width = 0, unsigned max_height = 0,
                                                         RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and partial decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
 * \param [in] aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
 * \param [in] num_attempts Maximum number of attempts to generate crop. Default 10
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalFusedJpegCropSingleShard(RocalContext context,
                                                                    const char* source_path,
                                                                    RocalImageColor color_format,
                                                                    unsigned shard_id,
                                                                    unsigned shard_count,
                                                                    bool is_output,
                                                                    std::vector<float>& area_factor,
                                                                    std::vector<float>& aspect_ratio,
                                                                    unsigned num_attempts,
                                                                    bool shuffle = false,
                                                                    bool loop = false,
                                                                    RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                    unsigned max_width = 0, unsigned max_height = 0,
                                                                    RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates TensorFlow records JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants. If images are not Jpeg compressed they will be ignored.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location of the TF records on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output image
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegTFRecordSource(RocalContext context,
                                                              const char* source_path,
                                                              RocalImageColor rocal_color_format,
                                                              unsigned internal_shard_count,
                                                              bool is_output,
                                                              const char* user_key_for_encoded,
                                                              const char* user_key_for_filename,
                                                              bool shuffle = false,
                                                              bool loop = false,
                                                              RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                              unsigned max_width = 0, unsigned max_height = 0,
                                                              RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                              RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates TensorFlow records JPEG image reader and decoder. It allocates the resources and objects required to read and decode Jpeg images stored on the file systems. It accepts external sharding information to load a singe shard. only
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location of the TF records on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegTFRecordSourceSingleShard(RocalContext context,
                                                                         const char* source_path,
                                                                         RocalImageColor rocal_color_format,
                                                                         unsigned shard_id,
                                                                         unsigned shard_count,
                                                                         bool is_output,
                                                                         bool shuffle = false,
                                                                         bool loop = false,
                                                                         RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                         unsigned max_width = 0, unsigned max_height = 0,
                                                                         RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                         RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates Raw image loader. It allocates the resources and objects required to load images stored on the file systems.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle: to shuffle dataset
 * \param [in] loop: repeat data loading
 * \param [in] out_width The output_width of raw image
 * \param [in] out_height The output height of raw image
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRawTFRecordSource(RocalContext p_context,
                                                             const char* source_path,
                                                             const char* user_key_for_raw,
                                                             const char* user_key_for_filename,
                                                             RocalImageColor rocal_color_format,
                                                             bool is_output,
                                                             bool shuffle = false,
                                                             bool loop = false,
                                                             unsigned out_width = 0, unsigned out_height = 0,
                                                             const char* record_name_prefix = "",
                                                             RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates Raw image loader. It allocates the resources and objects required to load images stored on the file systems.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] shuffle: to shuffle dataset
 * \param [in] loop: repeat data loading
 * \param [in] out_width The output_width of raw image
 * \param [in] out_height The output height of raw image
 * \param [in] record_name_prefix : if nonempty reader will only read records with certain prefix
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRawTFRecordSourceSingleShard(RocalContext p_context,
                                                                        const char* source_path,
                                                                        RocalImageColor rocal_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        unsigned out_width = 0, unsigned out_height = 0,
                                                                        const char* record_name_prefix = "",
                                                                        RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates Numpy raw data reader and loader. It allocates the resources and objects required to read raw data stored on the numpy arrays.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load using multiple loader instances. Using shard counts bigger than 1 improves the load performance if compute resources (CPU cores) are available.
 * \param [in] output_layout the layout to be set for the input tensor
 * \param [in] files Contains a list of file paths to read the data from.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loop through images or not.
 * \param [in] seed Determines the seed used by RNG for shuffling data between shards.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalNumpyFileSource(RocalContext context,
                                                           const char* source_path,
                                                           unsigned internal_shard_count,
                                                           RocalTensorLayout output_layout = RocalTensorLayout::ROCAL_NONE,
                                                           std::vector<std::string> files = {},
                                                           bool is_output = false,
                                                           bool shuffle = false,
                                                           bool loop = false,
                                                           unsigned seed = 0,
                                                           RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates Numpy raw data reader and loader. It allocates the resources and objects required to read raw data stored on the numpy arrays.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] output_layout the layout to be set for the input tensor
 * \param [in] files Contains a list of file paths to read the data from.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loop through images or not.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] seed Determines the seed used by RNG for shuffling data between shards.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor rocalNumpyFileSourceSingleShard(RocalContext context,
                                                       const char* source_path,
                                                       RocalTensorLayout output_layout = RocalTensorLayout::ROCAL_NONE,
                                                       std::vector<std::string> files = {},
                                                       bool is_output = false,
                                                       bool shuffle = false,
                                                       bool loop = false,
                                                       unsigned shard_id = 0,
                                                       unsigned shard_count = 1,
                                                       unsigned seed = 0,
                                                       RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*!
 * \brief Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk. source_path can be a video file, folder containing videos or a text file
 * \param [in] color_format The color format the frames will be decoded to.
 * \param [in] rocal_decode_device Enables software decoding(using FFmpeg) or hardware decoding(using FFmepg/rocDecode hardware decoder).
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances.
 * \param [in] sequence_length: The number of frames in a sequence.
 * \param [in] shuffle: to shuffle sequences.
 * \param [in] is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
 * \param [in] loop: repeat data loading.
 * \param [in] rocal_decoder_type Determines the decoder_type, FFmepg or rocDecode
 * \param [in] step: Frame interval between each sequence.
 * \param [in] stride: Frame interval between frames in a sequence.
 * \param [in] file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVideoFileSource(RocalContext context,
                                                           const char* source_path,
                                                           RocalImageColor color_format,
                                                           RocalDecodeDevice rocal_decode_device,
                                                           unsigned internal_shard_count,
                                                           unsigned sequence_length,
                                                           bool is_output = false,
                                                           bool shuffle = false,
                                                           bool loop = false,
                                                           RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_SW,
                                                           unsigned step = 0,
                                                           unsigned stride = 0,
                                                           bool file_list_frame_num = true,
                                                           RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems. It accepts external sharding information to load a singe shard only.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk. source_path can be a video file, folder containing videos or a text file
 * \param [in] color_format The color format the frames will be decoded to.
 * \param [in] rocal_decode_device Enables software decoding(using FFmpeg) or hardware decoding(using FFmepg/rocDecode hardware decoder).
 * \param [in] shard_id Shard id for this loader.
 * \param [in] shard_count Total shard count.
 * \param [in] sequence_length: The number of frames in a sequence.
 * \param [in] shuffle: to shuffle sequences.
 * \param [in] is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
 * \param [in] loop: repeat data loading.
 * \param [in] rocal_decoder_type Determines the decoder_type, FFmepg or rocDecode
 * \param [in] step: Frame interval between each sequence.
 * \param [in] stride: Frame interval between frames in a sequence.
 * \param [in] file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVideoFileSourceSingleShard(RocalContext context,
                                                                      const char* source_path,
                                                                      RocalImageColor color_format,
                                                                      RocalDecodeDevice rocal_decode_device,
                                                                      unsigned shard_id,
                                                                      unsigned shard_count,
                                                                      unsigned sequence_length,
                                                                      bool shuffle = false,
                                                                      bool is_output = false,
                                                                      bool loop = false,
                                                                      RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_SW,
                                                                      unsigned step = 0,
                                                                      unsigned stride = 0,
                                                                      bool file_list_frame_num = true,
                                                                      RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems. Resizes the decoded frames to the dest width and height.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk. source_path can be a video file, folder containing videos or a text file
 * \param [in] color_format The color format the frames will be decoded to.
 * \param [in] rocal_decode_device Enables software decoding(using FFmpeg) or hardware decoding(using FFmepg/rocDecode hardware decoder).
 * \param [in] internal_shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances.
 * \param [in] sequence_length: The number of frames in a sequence.
 * \param [in] dest_width The output width of frames.
 * \param [in] dest_height The output height of frames.
 * \param [in] shuffle: to shuffle sequences.
 * \param [in] is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
 * \param [in] loop: repeat data loading.
 * \param [in] rocal_decoder_type Determines the decoder_type, FFmepg or rocDecode
 * \param [in] step: Frame interval between each sequence.
 * \param [in] stride: Frame interval between frames in a sequence.
 * \param [in] file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVideoFileResize(RocalContext context,
                                                           const char* source_path,
                                                           RocalImageColor color_format,
                                                           RocalDecodeDevice rocal_decode_device,
                                                           unsigned internal_shard_count,
                                                           unsigned sequence_length,
                                                           unsigned dest_width,
                                                           unsigned dest_height,
                                                           bool shuffle = false,
                                                           bool is_output = false,
                                                           bool loop = false,
                                                           RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_SW,
                                                           unsigned step = 0,
                                                           unsigned stride = 0,
                                                           bool file_list_frame_num = true,
                                                           RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_DEFAULT,
                                                           std::vector<unsigned> max_size = {},
                                                           unsigned resize_shorter = 0,
                                                           unsigned resize_longer = 0,
                                                           RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                           RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates a video reader and decoder as a source. It allocates the resources and objects required to read and decode mp4 videos stored on the file systems. Resizes the decoded frames to the dest width and height. It accepts external sharding information to load a singe shard only.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk. source_path can be a video file, folder containing videos or a text file
 * \param [in] color_format The color format the frames will be decoded to.
 * \param [in] rocal_decode_device Enables software decoding(using FFmpeg) or hardware decoding(using FFmepg/rocDecode hardware decoder).
 * \param [in] shard_id Shard id for this loader.
 * \param [in] shard_count Total shard count.
 * \param [in] sequence_length: The number of frames in a sequence.
 * \param [in] dest_width The output width of frames.
 * \param [in] dest_height The output height of frames.
 * \param [in] shuffle: to shuffle sequences.
 * \param [in] is_output Determines if the user wants the loaded sequence of frames to be part of the output or not.
 * \param [in] loop: repeat data loading.
 * \param [in] rocal_decoder_type Determines the decoder_type, FFmepg or rocDecode
 * \param [in] step: Frame interval between each sequence.
 * \param [in] stride: Frame interval between frames in a sequence.
 * \param [in] file_list_frame_num: Determines if the user wants to read frame number or timestamps if a text file is passed in the source_path.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalVideoFileResizeSingleShard(RocalContext context,
                                                                      const char* source_path,
                                                                      RocalImageColor color_format,
                                                                      RocalDecodeDevice rocal_decode_device,
                                                                      unsigned shard_id,
                                                                      unsigned shard_count,
                                                                      unsigned sequence_length,
                                                                      unsigned dest_width,
                                                                      unsigned dest_height,
                                                                      bool shuffle = false,
                                                                      bool is_output = false,
                                                                      bool loop = false,
                                                                      RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_VIDEO_FFMPEG_SW,
                                                                      unsigned step = 0,
                                                                      unsigned stride = 0,
                                                                      bool file_list_frame_num = true,
                                                                      RocalResizeScalingMode scaling_mode = ROCAL_SCALING_MODE_DEFAULT,
                                                                      std::vector<unsigned> max_size = {},
                                                                      unsigned resize_shorter = 0,
                                                                      unsigned resize_longer = 0,
                                                                      RocalResizeInterpolationType interpolation_type = ROCAL_LINEAR_INTERPOLATION,
                                                                      RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates CIFAR10 raw data reader and loader. It allocates the resources and objects required to read raw data stored on the file systems.
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] out_width output width
 * \param [in] out_height output_height
 * \param [in] filename_prefix if set loader will only load files with the given prefix name
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalRawCIFAR10Source(RocalContext context,
                                                            const char* source_path,
                                                            RocalImageColor color_format,
                                                            bool is_output,
                                                            unsigned out_width, unsigned out_height, const char* filename_prefix = "",
                                                            bool loop = false);

/*! \brief reset Loaders
 * \ingroup group_rocal_data_loaders
 * \param [in] context Rocal Context
 * \return Rocal status value
 */
extern "C" RocalStatus ROCAL_API_CALL rocalResetLoaders(RocalContext context);

/*! \brief Creates JPEG image reader and partial decoder for Caffe LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe2 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] area_factor Determines how much area to be cropped. Ranges from from 0.08 - 1.
 * \param [in] aspect_ratio Determines the aspect ration of crop. Ranges from 0.75 to 1.33.
 * \param [in] num_attempts Maximum number of attempts to generate crop. Default 10
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCaffeLMDBRecordSourcePartialSingleShard(RocalContext p_context,
                                                                                       const char* source_path,
                                                                                       RocalImageColor rocal_color_format,
                                                                                       unsigned shard_id,
                                                                                       unsigned shard_count,
                                                                                       bool is_output,
                                                                                       std::vector<float>& area_factor,
                                                                                       std::vector<float>& aspect_ratio,
                                                                                       unsigned num_attempts,
                                                                                       bool shuffle = false,
                                                                                       bool loop = false,
                                                                                       RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                                       unsigned max_width = 0, unsigned max_height = 0,
                                                                                       RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! \brief Creates JPEG image reader and partial decoder for Caffe2 LMDB records. It allocates the resources and objects required to read and decode Jpeg images stored in Caffe22 LMDB Records. It has internal sharding capability to load/decode in parallel is user wants.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Total shard count
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegCaffe2LMDBRecordSourcePartialSingleShard(RocalContext p_context,
                                                                                        const char* source_path,
                                                                                        RocalImageColor rocal_color_format,
                                                                                        unsigned shard_id,
                                                                                        unsigned shard_count,
                                                                                        bool is_output,
                                                                                        std::vector<float>& area_factor,
                                                                                        std::vector<float>& aspect_ratio,
                                                                                        unsigned num_attempts,
                                                                                        bool shuffle = false,
                                                                                        bool loop = false,
                                                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                                        unsigned max_width = 0, unsigned max_height = 0,
                                                                                        RocalShardingInfo rocal_sharding_info = RocalShardingInfo());
/*! \brief Creates JPEG external source image reader.
 * \ingroup group_rocal_data_loaders
 * \param [in] rocal_context Rocal context
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] is_output Determines if the user wants the loaded images to be part of the output or not.
 * \param [in] shuffle Determines if the user wants to shuffle the dataset or not.
 * \param [in] loop Determines if the user wants to indefinitely loops through images or not.
 * \param [in] decode_size_policy is the RocalImageSizeEvaluationPolicy for decoding
 * \param [in] max_width The maximum width of the decoded images, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded images, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type, tjpeg or hwdec
 * \param [in] external_source_mode Determines the mode of the source passed from the user - file_names / uncompressed data / compressed data
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalJpegExternalFileSource(RocalContext p_context,
                                                                  RocalImageColor rocal_color_format,
                                                                  bool is_output = false,
                                                                  bool shuffle = false,
                                                                  bool loop = false,
                                                                  RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MOST_FREQUENT_SIZE,
                                                                  unsigned max_width = 0, unsigned max_height = 0,
                                                                  RocalDecoderType rocal_decoder_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                  RocalExternalSourceMode external_source_mode = RocalExternalSourceMode::ROCAL_EXTSOURCE_FNAME,
                                                                  RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! Creates Audio file reader and decoder. It allocates the resources and objects required to read and decode audio files stored on the file systems. It has internal sharding capability to load/decode in parallel if user wants.
 * If the files are not in standard audio compression formats they will be ignored, Currently wav format is supported
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] source_file_list_path A char string pointing to the file list location on the disk
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Boolean variable to enable the audio to be part of the output.
 * \param [in] shuffle Boolean variable to shuffle the dataset.
 * \param [in] loop Boolean variable to indefinitely loop through audio.
 * \param [in] downmix Boolean variable to downmix all input channels to mono. If downmixing is turned on, the decoder output is 1D. If downmixing is turned off, it produces 2D output with interleaved channels incase of multichannel audio.
 * \param [in] decode_size_policy is the RocalImageSizeEvaluationPolicy for decoding.
 * \param [in] max_decoded_samples The maximum samples of the decoded audio data.
 * \param [in] max_decoded_channels The maximum channels of the decoded audio data.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output audio
 */
extern "C" RocalTensor ROCAL_API_CALL rocalAudioFileSource(RocalContext context,
                                                           const char* source_path,
                                                           const char* source_file_list_path,
                                                           unsigned shard_count,
                                                           bool is_output,
                                                           bool shuffle = false,
                                                           bool loop = false,
                                                           bool downmix = false,
                                                           RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE,
                                                           unsigned max_decoded_samples = 0,
                                                           unsigned max_decoded_channels = 0,
                                                           RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! Creates Audio file reader and decoder. It allocates the resources and objects required to read and decode audio files stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
 * If the files are not in standard audio compression formats they will be ignored.
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location on the disk
 * \param [in] source_file_list_path A char string pointing to the file list location on the disk
 * \param [in] shard_id Shard id for this loader
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Boolean variable to enable the audio to be part of the output.
 * \param [in] shuffle Boolean variable to shuffle the dataset.
 * \param [in] loop Boolean variable to indefinitely loop through audio.
 * \param [in] downmix Boolean variable to downmix all input channels to mono. If downmixing is turned on, the decoder output is 1D. If downmixing is turned off, it produces 2D output with interleaved channels incase of multichannel audio.
 * \param [in] decode_size_policy is the RocalImageSizeEvaluationPolicy for decoding.
 * \param [in] max_decoded_samples The maximum samples of the decoded audio data.
 * \param [in] max_decoded_channels The maximum channels of the decoded audio data.
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output audio
 */
extern "C" RocalTensor ROCAL_API_CALL rocalAudioFileSourceSingleShard(RocalContext p_context,
                                                                      const char* source_path,
                                                                      const char* source_file_list_path,
                                                                      unsigned shard_id,
                                                                      unsigned shard_count,
                                                                      bool is_output,
                                                                      bool shuffle = false,
                                                                      bool loop = false,
                                                                      bool downmix = false,
                                                                      RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE,
                                                                      unsigned max_decoded_samples = 0,
                                                                      unsigned max_decoded_channels = 0,
                                                                      RocalShardingInfo rocal_sharding_info = RocalShardingInfo());

/*! Creates WebDataset tar files reader and decoder. It allocates the resources and objects required to read and decode files in webdataset format stored on the file systems. It has internal sharding capability to load/decode in parallel is user wants.
 * \param [in] context Rocal context
 * \param [in] source_path A NULL terminated char string pointing to the location of files on the disk
 * \param [in] index_path A NULL terminated char string pointing to the location of index files on the disk
 * \param [in] rocal_color_format The color format the images will be decoded to.
 * \param [in] shard_id Shard id for this loader.
 * \param [in] shard_count Defines the parallelism level by internally sharding the input dataset and load/decode using multiple decoder/loader instances. Using shard counts bigger than 1 improves the load/decode performance if compute resources (CPU cores) are available.
 * \param [in] is_output Boolean variable to enable the audio to be part of the output.
 * \param [in] shuffle Boolean variable to shuffle the dataset.
 * \param [in] loop Boolean variable to indefinitely loop through audio.
 * \param [in] decode_size_policy is the RocalImageSizeEvaluationPolicy for decoding
 * \param [in] max_width The maximum width of the decoded image files, larger or smaller will be resized to closest
 * \param [in] max_height The maximum height of the decoded image files, larger or smaller will be resized to closest
 * \param [in] rocal_decoder_type Determines the decoder_type - image / video / audio
 * \param [in] rocal_sharding_info The members of RocalShardingInfo determines how the data is distributed among the shards and how the last batch is processed by the pipeline.
 * \return Reference to the output tensor
 */
extern "C" RocalTensor ROCAL_API_CALL rocalWebDatasetSourceSingleShard(RocalContext p_context,
                                                                        const char* source_path,
                                                                        const char* index_path,
                                                                        RocalImageColor rocal_color_format,
                                                                        unsigned shard_id,
                                                                        unsigned shard_count,
                                                                        bool is_output,
                                                                        bool shuffle = false,
                                                                        bool loop = false,
                                                                        RocalImageSizeEvaluationPolicy decode_size_policy = ROCAL_USE_MAX_SIZE,
                                                                        unsigned max_width = 0,
                                                                        unsigned max_height = 0,
                                                                        RocalDecoderType dec_type = RocalDecoderType::ROCAL_DECODER_TJPEG,
                                                                        RocalShardingInfo rocal_sharding_info = RocalShardingInfo());
                                                 
#endif  // MIVISIONX_ROCAL_API_DATA_LOADERS_H
