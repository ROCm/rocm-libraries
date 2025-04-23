/*
 * Copyright (c) 2012-2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _VX_KHR_CLASSIFIER_H_
#define _VX_KHR_CLASSIFIER_H_

/*!
 * \file
 * \brief The Khronos Extension for general classification.
 *
 */

#define OPENVX_KHR_CLASS   "vx_khr_class"

#include <VX/vx.h>


#ifdef  __cplusplus
extern "C" {
#endif
/*! \brief The Classifier Extension Library Set
 * \ingroup group_classifier
 */
#define VX_LIBRARY_KHR_CLASS_EXTENSION (0x2)
/*! \brief The list of Classifier Extension Kernels.
 * \ingroup group_classifier
 */
enum vx_kernel_nn_ext_e {
    /*! \brief The Classifier Extension scan kernel.
    * \see group_classifier
    */
    VX_KERNEL_SCAN_CLASSIFIER = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_CLASS_EXTENSION) + 0x0,
};

/*! \brief Classifier Extension type enums.
 * \ingroup group_classifier
 */
enum vx_class_enum_e
{
	VX_ENUM_CLASSIFIER_MODEL= 0x1E, /*!< \brief Classifier model */
};

/*!
 * \brief classification model to be used in <tt>\ref vxScanClassifierNode</tt>.
 * The classification models are loadable by undefined binary format see <tt>\ref vxImportClassifierModel</tt>.
 * Extensions will be added to the specification, to support a defined binary format.
 * \ingroup group_object_classifier_model
 */
typedef struct _vx_classifier_model* vx_classifier_model;

/*! \brief Classifier model format enums.
 * In the main specification only undefined binary format is supported. Extensions to the specification will be added in order to support specific binary format.
 * \ingroup group_object_classifier_model
 */
enum vx_classifier_model_format_e
{
	/*! \brief Undefined binary format.
	* Using this enumeration will result in an implementation defined behaviour.
	*/
	VX_CLASSIFIER_MODEL_UNDEFINED = VX_ENUM_BASE( VX_ID_KHRONOS, VX_ENUM_CLASSIFIER_MODEL ) + 0x0,
};
/*! \brief The type enumeration lists all classifier extension types.
 * \ingroup group_object_classifier_model
 */
enum vx_classifier_type_e {
	VX_TYPE_CLASSIFER_MODEL     = 0x02C,/*!< \brief A <tt>\ref vx_classifier_model</tt>. type */
};

/*==============================================================================
 CLASSIFIER MODEL
 =============================================================================*/
/*!
 * \brief Creates an opaque reference classifier model
 * This function creates a classifier model to be used in <tt>\ref vxScanClassifierNode</tt>. The object classifier object is a read-only constant object. It cannot be changed during graph execution.
 * \param [in] context Reference to the context where to create the ClassifierModel.
 * \param [in] format The binary format which contain the classifier model. See <tt>\ref vx_classifier_model_format_e</tt>. Currently only undefined binary format is supported.
 * Extensions will be added to the specification, to support a classification model defined binary format.
 * \param [in] ptr A memory pointer to the binary format.
 * \param [in] length size in bytes of binary format data.
 * \returns A ClassifierModel reference <tt>\ref vx_classifier_model</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 * \ingroup group_object_classifier_model
 */
VX_API_ENTRY vx_classifier_model VX_API_CALL vxImportClassifierModel(vx_context context, vx_enum format, const vx_uint8* ptr, vx_size length);

/*!
 * \brief Releases a reference of an ClassifierModel object.
 * The object may not be garbage collected until its total reference and its contained objects
 * count is zero. After returning from this function the reference is zeroed/cleared.
 * \param [in] model The pointer to the ClassifierModel to release.
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval <tt>\ref VX_SUCCESS</tt> No errors; all other values indicate failure
 * \retval * An error occurred. See <tt\ref >vx_status_e</tt>.
 * \ingroup group_object_classifier_model
 */
VX_API_ENTRY vx_status VX_API_CALL vxReleaseClassifierModel(vx_classifier_model* model);

/*! \brief [Graph] Scans a feature-map (input_feature_map) and detect the classification for each scan-window.
 * \param [in] graph The reference to the graph
 * \param [in] input_feature_map The Feature-map, example is the output of <tt>\ref vxHOGFeaturesNode</tt>.
 * \param [in] model The pre-trained model loaded. Loaded using <tt>\ref vxImportClassifierModel</tt>
 * \param [in] scan_window_width Width of the scan window
 * \param [in] scan_window_height Height of the scan window
 * \param [in] step_x Horizontal step-size (along x-axis)
 * \param [in] step_y Vertical step-size (along y-axis)
 * \param [out] object_confidences [Optional] An array of confidences measure, the measure is of type <tt>\ref VX_TYPE_UINT16</tt>. The confidence measure is defined by the extensions which define classification model with defined binary format.
 * This output can be used as class index as well. In case we detect several different classes in single execution. The output will be an array of indexes of the classes.
 * \param [out] object_rectangles An array of object positions, in <tt>\ref VX_TYPE_RECTANGLE</tt>
 * \param [out] num_objects [optional] The number of object detected in a <tt>\ref VX_SIZE</tt> scalar
 * \note The border mode <tt>\ref VX_NODE_BORDER</tt> value <tt>\ref VX_BORDER_UNDEFINED</tt> is supported.
 * \ingroup group_vision_function_classifier
 * \return <tt>\ref vx_node</tt>.
 * \retval vx_node A node reference. Any possible errors preventing a successful creation should be checked using <tt>\ref vxGetStatus</tt>
 */

VX_API_ENTRY vx_node VX_API_CALL vxScanClassifierNode(vx_graph graph,vx_tensor input_feature_map, vx_classifier_model model, vx_int32 scanwindow_width, vx_int32 scanwindow_height, vx_int32 step_x, vx_int32 step_y,
                                     vx_array object_confidences, vx_array object_rectangles, vx_scalar num_objects);


#ifdef  __cplusplus
}
#endif


#endif
