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

#ifndef _OPENVX_PIPELINING_H_
#define _OPENVX_PIPELINING_H_

/*!
 * \file
 * \brief The OpenVX Pipelining, Streaming and Batch Processing extension API.
 */

#define OPENVX_KHR_PIPELINING  "vx_khr_pipelining"

#include <VX/vx.h>

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * PIPELINING API
 */

/*! \brief The Pipelining, Streaming and Batch Processing Extension Library Set
 *
 * \ingroup group_pipelining
 */
#define VX_LIBRARY_KHR_PIPELINING_EXTENSION (0x1)


/*! \brief Extra enums.
 *
 * \ingroup group_pipelining
 */
enum vx_graph_schedule_mode_enum_e
{
    VX_ENUM_GRAPH_SCHEDULE_MODE_TYPE     = 0x21, /*!< \brief Graph schedule mode type enumeration. */
};

/*! \brief Type of graph scheduling mode
 *
 * See <tt>\ref vxSetGraphScheduleConfig</tt> and <tt>\ref vxGraphParameterEnqueueReadyRef</tt> for details about each mode.
 *
 * \ingroup group_pipelining
 */
enum vx_graph_schedule_mode_type_e {

    /*! \brief Schedule graph in non-queueing mode
     */
    VX_GRAPH_SCHEDULE_MODE_NORMAL = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_GRAPH_SCHEDULE_MODE_TYPE << 12)) + 0x0,

    /*! \brief Schedule graph in queueing mode with auto scheduling
     */
    VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_GRAPH_SCHEDULE_MODE_TYPE << 12)) + 0x1,

    /*! \brief Schedule graph in queueing mode with manual scheduling
     */
    VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_GRAPH_SCHEDULE_MODE_TYPE << 12)) + 0x2,
};

/*! \brief The graph attributes added by this extension.
 * \ingroup group_pipelining
 */
enum vx_graph_attribute_pipelining_e {
    /*! \brief Returns the schedule mode of a graph. Read-only. Use a <tt>\ref vx_enum</tt> parameter.
     * See <tt>\ref vx_graph_schedule_mode_type_e </tt> enum.
     * */
    VX_GRAPH_SCHEDULE_MODE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_GRAPH) + 0x5,
};

/*! \brief Queueing parameters for a specific graph parameter
 *
 * See <tt>\ref vxSetGraphScheduleConfig</tt> for additional details.
 *
 *  \ingroup group_pipelining
 */
typedef struct {

    uint32_t graph_parameter_index;
    /*!< \brief Index of graph parameter to which these properties apply */

    vx_uint32 refs_list_size;
    /*!< \brief Number of elements in array 'refs_list' */

    vx_reference *refs_list;
    /*!< \brief Array of references that could be enqueued at a later point of time at this graph parameter */

} vx_graph_parameter_queue_params_t;

/*! \brief Sets the graph scheduler config
 *
 * This API is used to set the graph scheduler config to
 * allow user to schedule multiple instances of a graph for execution.
 *
 * For legacy applications that don't need graph pipelining or batch processing,
 * this API need not be used.
 *
 * Using this API, the application specifies the graph schedule mode, as well as
 * queueing parameters for all graph parameters that need to allow enqueueing of references.
 * A single monolithic API is provided instead of discrete APIs, since this allows
 * the implementation to get all information related to scheduling in one shot and
 * then optimize the subsequent graph scheduling based on this information.
 * <b> This API MUST be called before graph verify </b> since
 * in this case it allows implementations the opportunity to optimize resources based on
 * information provided by the application.
 *
 * 'graph_schedule_mode' selects how input and output references are provided to a graph and
 * how the next graph schedule is triggered by an implementation.
 *
 * Below scheduling modes are supported:
 *
 * When graph schedule mode is <tt>\ref VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO</tt>:
 * - Application needs to explicitly call <tt>\ref vxVerifyGraph</tt> before enqueing data references
 * - Application should not call <tt>\ref vxScheduleGraph</tt> or <tt>\ref vxProcessGraph</tt>
 * - When enough references are enqueued at various graph parameters, the implementation
 *   could trigger the next graph schedule.
 * - Here, not all graph parameters need to have enqueued references for a graph schedule to begin.
 *   An implementation is expected to execute the graph as much as possible until a enqueued reference
 *   is not available at which time it will stall the graph until the reference becomes available.
 *   This allows application to schedule a graph even when all parameters references are
 *   not yet available, i.e do a 'late' enqueue. However, exact behaviour is implementation specific.
 *
 * When graph schedule mode is <tt>\ref VX_GRAPH_SCHEDULE_MODE_QUEUE_MANUAL</tt>:
 * - Application needs to explicitly call <tt>\ref vxScheduleGraph</tt>
 * - Application should not call <tt>\ref vxProcessGraph</tt>
 * - References for all graph parameters of the graph needs to enqueued before <tt>\ref vxScheduleGraph</tt>
 *   is called on the graph else an error is returned by <tt>\ref vxScheduleGraph</tt>
 * - Application can enqueue multiple references at the same graph parameter.
 *   When <tt>\ref vxScheduleGraph</tt> is called, all enqueued references get processed in a 'batch'.
 * - User can use <tt>\ref vxWaitGraph</tt> to wait for the previous <tt>\ref vxScheduleGraph</tt>
 *   to complete.
 *
 * When graph schedule mode is <tt>\ref VX_GRAPH_SCHEDULE_MODE_NORMAL</tt>:
 * - 'graph_parameters_list_size' MUST be 0 and
 * - 'graph_parameters_queue_params_list' MUST be NULL
 * - This mode is equivalent to non-queueing scheduling mode as defined by OpenVX v1.2 and earlier.
 *
 * By default all graphs are in VX_GRAPH_SCHEDULE_MODE_NORMAL mode until this API is called.
 *
 * 'graph_parameters_queue_params_list' allows to specify below information:
 * - For the graph parameter index that is specified, it enables queueing mode of operation
 * - Further it allows the application to specify the list of references that it could later
 *   enqueue at this graph parameter.
 *
 * For graph parameters listed in 'graph_parameters_queue_params_list',
 * application MUST use <tt>\ref vxGraphParameterEnqueueReadyRef</tt> to
 * set references at the graph parameter.  Using other data access API's on
 * these parameters or corresponding data objects will return an error.
 * For graph parameters not listed in 'graph_parameters_queue_params_list'
 * application MUST use the <tt>\ref vxSetGraphParameterByIndex</tt> to set the reference
 * at the graph parameter.  Using other data access API's on these parameters or
 * corresponding data objects will return an error.
 *
 * This API also allows application to provide a list of references which could be later
 * enqueued at the graph parameter. This allows implementation to do meta-data checking
 * up front rather than during each reference enqueue.
 *
 * When this API is called before <tt>\ref vxVerifyGraph</tt>, the 'refs_list' field
 * can be NULL, if the reference handles are not available yet at the application.
 * However 'refs_list_size' MUST always be specified by the application.
 * Application can call <tt>\ref vxSetGraphScheduleConfig</tt> again after verify graph
 * with all parameters remaining the same except with 'refs_list' field providing
 * the list of references that can be enqueued at the graph parameter.
 *
 * \param [in] graph Graph reference
 * \param [in] graph_schedule_mode Graph schedule mode. See <tt>\ref vx_graph_schedule_mode_type_e</tt>
 * \param [in] graph_parameters_list_size Number of elements in graph_parameters_queue_params_list
 * \param [in] graph_parameters_queue_params_list Array containing queuing properties at graph parameters that need to support queueing.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid reference
 * \retval VX_ERROR_INVALID_PARAMETERS Invalid graph parameter queueing parameters
 * \retval VX_FAILURE Any other failure.
 *
 * \ingroup group_pipelining
 */
VX_API_ENTRY vx_status VX_API_CALL vxSetGraphScheduleConfig(
    vx_graph graph,
    vx_enum graph_schedule_mode,
    vx_uint32 graph_parameters_list_size,
    const vx_graph_parameter_queue_params_t graph_parameters_queue_params_list[]
    );

/*! \brief Enqueues new references into a graph parameter for processing
 *
 * This new reference will take effect on the next graph schedule.
 *
 * In case of a graph parameter which is input to a graph, this function provides
 * a data reference with new input data to the graph.
 * In case of a graph parameter which is not input to a graph, this function provides
 * a 'empty' reference into which a graph execution can write new data into.
 *
 * This function essentially transfers ownership of the reference from the application to the graph.
 *
 * User MUST use <tt>vxGraphParameterDequeueDoneRef</tt> to get back the
 * processed or consumed references.
 *
 * The references that are enqueued MUST be the references listed during
 * <tt>\ref vxSetGraphScheduleConfig</tt>. If a reference outside this list is provided then
 * behaviour is undefined.
 *
 * \param [in] graph Graph reference
 * \param [in] graph_parameter_index Graph parameter index
 * \param [in] refs The array of references to enqueue into the graph parameter
 * \param [in] num_refs Number of references to enqueue
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid reference OR reference is not a valid reference
 * \retval VX_ERROR_INVALID_PARAMETERS graph_parameter_index is NOT a valid graph parameter index
 * \retval VX_FAILURE Reference could not be enqueued.
 *
 * \ingroup group_pipelining
 */
VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterEnqueueReadyRef(vx_graph graph,
                vx_uint32 graph_parameter_index,
                vx_reference *refs,
                vx_uint32 num_refs);



/*! \brief Dequeues 'consumed' references from a graph parameter
 *
 * This function dequeues references from a graph parameter of a graph.
 * The reference that is dequeued is a reference that had been previously enqueued into a graph,
 * and after subsequent graph execution is considered as processed or consumed by the graph.
 * This function essentially transfers ownership of the reference from the graph to the application.
 *
 * <b> IMPORTANT </b> : This API will block until at least one reference is dequeued.
 *
 * In case of a graph parameter which is input to a graph, this function provides
 * a 'consumed' buffer to the application so that new input data can filled
 * and later enqueued to the graph.
 * In case of a graph parameter which is not input to a graph, this function provides
 * a reference filled with new data based on graph execution. User can then use this
 * newly generated data with their application. Typically when this new data is
 * consumed by the application the 'empty' reference is again enqueued to the graph.
 *
 * This API returns an array of references up to a maximum of 'max_refs'. Application MUST ensure
 * the array pointer ('refs') passed as input can hold 'max_refs'.
 * 'num_refs' is actual number of references returned and will be <= 'max_refs'.
 *
 *
 * \param [in] graph Graph reference
 * \param [in] graph_parameter_index Graph parameter index
 * \param [out] refs Dequeued references filled in the array
 * \param [in] max_refs Max number of references to dequeue
 * \param [out] num_refs Actual number of references dequeued.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid reference
 * \retval VX_ERROR_INVALID_PARAMETERS graph_parameter_index is NOT a valid graph parameter index
 * \retval VX_FAILURE Reference could not be dequeued.
 *
 * \ingroup group_pipelining
 */
VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterDequeueDoneRef(vx_graph graph,
            vx_uint32 graph_parameter_index,
            vx_reference *refs,
            vx_uint32 max_refs,
            vx_uint32 *num_refs);

/*! \brief Checks and returns the number of references that are ready for dequeue
 *
 * This function checks the number of references that can be dequeued and
 * returns the value to the application.
 *
 * See also <tt>\ref vxGraphParameterDequeueDoneRef</tt>.
 *
 * \param [in] graph Graph reference
 * \param [in] graph_parameter_index Graph parameter index
 * \param [out] num_refs Number of references that can be dequeued using <tt>\ref vxGraphParameterDequeueDoneRef</tt>
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid reference
 * \retval VX_ERROR_INVALID_PARAMETERS graph_parameter_index is NOT a valid graph parameter index
 * \retval VX_FAILURE Any other failure.
 *
 * \ingroup group_pipelining
 */
VX_API_ENTRY vx_status VX_API_CALL vxGraphParameterCheckDoneRef(vx_graph graph,
            vx_uint32 graph_parameter_index,
            vx_uint32 *num_refs);

/*
 * EVENT QUEUE API
 */

/*! \brief Extra enums.
 *
 * \ingroup group_event
 */
enum vx_event_enum_e
{
    VX_ENUM_EVENT_TYPE     = 0x22, /*!< \brief Event Type enumeration. */
};

/*! \brief Type of event that can be generated during system execution
 *
 * \ingroup group_event
 */
enum vx_event_type_e {

    /*! \brief Graph parameter consumed event
     *
     * This event is generated when a data reference at a graph parameter
     * is consumed during a graph execution.
     * It is used to indicate that a given data reference is no longer used by the graph and can be
     * dequeued and accessed by the application.
     *
     * \note Graph execution could still be "in progress" for rest of the graph that does not use
     * this data reference.
     */
    VX_EVENT_GRAPH_PARAMETER_CONSUMED = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_EVENT_TYPE << 12)) + 0x0,

    /*! \brief Graph completion event
     *
     * This event is generated every time a graph execution completes.
     * Graph completion event is generated for both successful execution of a graph
     * or abandoned execution of a graph.
     */
    VX_EVENT_GRAPH_COMPLETED = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_EVENT_TYPE << 12)) + 0x1,

    /*! \brief Node completion event
     *
     * This event is generated every time a node within a graph completes execution.
     */
    VX_EVENT_NODE_COMPLETED = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_EVENT_TYPE << 12)) + 0x2,

    /*! \brief Node error event
     *
     * This event is generated every time a node returns error within a graph.
     */
    VX_EVENT_NODE_ERROR = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_EVENT_TYPE << 12)) + 0x3,

    /*! \brief User defined event
     *
     * This event is generated by user application outside of OpenVX framework using the \ref vxSendUserEvent API.
     * User events allow application to have single centralized 'wait-for' loop to handle
     * both framework generated events as well as user generated events.
     *
     * \note Since the application initiates user events and not the framework, the application
     * does NOT register user events using \ref vxRegisterEvent.
     */
    VX_EVENT_USER = ((( VX_ID_KHRONOS ) << 20) | ( VX_ENUM_EVENT_TYPE << 12)) + 0x4
};

/*! \brief Parameter structure returned with event of type VX_EVENT_GRAPH_PARAMETER_CONSUMED
 *
 * \ingroup group_event
 */
typedef struct _vx_event_graph_parameter_consumed {

    vx_graph graph;
    /*!< \brief graph which generated this event */

    vx_uint32 graph_parameter_index;
    /*!< \brief graph parameter index which generated this event */
} vx_event_graph_parameter_consumed;

/*! \brief Parameter structure returned with event of type VX_EVENT_GRAPH_COMPLETED
 *
 * \ingroup group_event
 */
typedef struct _vx_event_graph_completed {

    vx_graph graph;
    /*!< \brief graph which generated this event */
} vx_event_graph_completed;

/*! \brief Parameter structure returned with event of type VX_EVENT_NODE_COMPLETED
 *
 * \ingroup group_event
 */
typedef struct _vx_event_node_completed {

    vx_graph graph;
    /*!< \brief graph which generated this event */

    vx_node node;
    /*!< \brief node which generated this event */
} vx_event_node_completed;

/*! \brief Parameter structure returned with event of type VX_EVENT_NODE_ERROR
 *
 * \ingroup group_event
 */
typedef struct _vx_event_node_error {

    vx_graph graph;
    /*!< \brief graph which generated this event */

    vx_node node;
    /*!< \brief node which generated this event */

    vx_status status;
    /*!< \brief error condition of node */
} vx_event_node_error;

/*! \brief Parameter structure returned with event of type VX_EVENT_USER_EVENT
 *
 * \ingroup group_event
 */
typedef struct _vx_event_user_event {

    void *user_event_parameter;
    /*!< \brief User defined parameter value. This is used to pass additional user defined parameters with a user event.
     */
} vx_event_user_event;

/*! \brief Parameter structure associated with an event. Depends on type of the event.
 *
 * \ingroup group_event
 */
typedef union _vx_event_info_t {

    vx_event_graph_parameter_consumed graph_parameter_consumed;
    /*!< event information for type: \ref VX_EVENT_GRAPH_PARAMETER_CONSUMED */

    vx_event_graph_completed graph_completed;
    /*!< event information for type: \ref VX_EVENT_GRAPH_COMPLETED */

    vx_event_node_completed node_completed;
    /*!< event information for type: \ref VX_EVENT_NODE_COMPLETED */

    vx_event_node_error node_error;
    /*!< event information for type: \ref VX_EVENT_NODE_ERROR */

    vx_event_user_event user_event;
    /*!< event information for type: \ref VX_EVENT_USER */

} vx_event_info_t;


/*! \brief Data structure which holds event information
 *
 * \ingroup group_event
 */
typedef struct _vx_event {

    vx_enum type;
    /*!< \brief see event type \ref vx_event_type_e */

    vx_uint64 timestamp;
    /*!< time at which this event was generated, in units of nano-secs */

    vx_uint32 app_value;
    /*!< value given to event by application during event registration (\ref vxRegisterEvent) or
     * (\ref vxSendUserEvent) in the case of user events. */

    vx_event_info_t event_info;
    /*!< parameter structure associated with an event. Depends on type of the event */

} vx_event_t;

/*! \brief Wait for a single event
 *
 * After <tt> \ref vxDisableEvents </tt> is called, if <tt> \ref vxWaitEvent(.. ,.. , vx_false_e) </tt> is called,
 * <tt> \ref vxWaitEvent </tt> will remain blocked until events are re-enabled using <tt> \ref vxEnableEvents </tt>
 * and a new event is received.
 *
 * If <tt> \ref vxReleaseContext </tt> is called while an application is blocked on <tt> \ref vxWaitEvent </tt>, the
 * behavior is not defined by OpenVX.
 *
 * If <tt> \ref vxWaitEvent </tt> is called simultaneously from multiple thread/task contexts
 * then its behaviour is not defined by OpenVX.
 *
 * \param context [in] OpenVX context
 * \param event [out] Data structure which holds information about a received event
 * \param do_not_block [in] When value is vx_true_e API does not block and only checks for the condition
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS Event received and event information available in 'event'
 * \retval VX_FAILURE No event is received
 *
 * \ingroup group_event
 */
VX_API_ENTRY vx_status VX_API_CALL vxWaitEvent(vx_context context, vx_event_t *event, vx_bool do_not_block);

/*! \brief Enable event generation
 * 
 * Depending on the implementation, events may be either enabled or disabled by default.
 *
 * \param context [in] OpenVX context
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_event
 */
VX_API_ENTRY vx_status VX_API_CALL vxEnableEvents(vx_context context);

/*! \brief Disable event generation
 *
 * When events are disabled, any event generated before this API is
 * called will still be returned via \ref vxWaitEvent API.
 * However no additional events would be returned via \ref vxWaitEvent API
 * until events are enabled again.
 *
 * \param context [in] OpenVX context
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_event
 */
VX_API_ENTRY vx_status VX_API_CALL vxDisableEvents(vx_context context);

/*! \brief Generate user defined event
 *
 * \param context [in] OpenVX context
 * \param app_value [in] Application-specified value that will be returned to user as part of vx_event_t.app_value
 *                       NOT used by implementation.
 * \param parameter [in] User defined event parameter. NOT used by implementation.
 *                       Returned to user as part vx_event_t.event_info.user_event.user_event_parameter field
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 *
 * \ingroup group_event
 */
VX_API_ENTRY vx_status VX_API_CALL vxSendUserEvent(vx_context context, vx_uint32 app_value, void *parameter);


/*! \brief Register an event to be generated
 *
 * Generation of event may need additional resources and overheads for an implementation.
 * Hence events should be registered for references only when really required by an application.
 *
 * This API can be called on graph, node or graph parameter.
 * This API MUST be called before doing \ref vxVerifyGraph for that graph.
 *
 * \param ref [in] Reference which will generate the event
 * \param type [in] Type or condition on which the event is generated
 * \param param [in] Specifies the graph parameter index when type is VX_EVENT_GRAPH_PARAMETER_CONSUMED
 * \param app_value [in] Application-specified value that will be returned to user as part of \ref vx_event_t.app_value.
 *                       NOT used by implementation.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE ref is not a valid <tt>\ref vx_reference</tt> reference.
 * \retval VX_ERROR_NOT_SUPPORTED type is not valid for the provided reference.
 *
 * \ingroup group_event
 */
VX_API_ENTRY vx_status VX_API_CALL vxRegisterEvent(vx_reference ref, enum vx_event_type_e type, vx_uint32 param, vx_uint32 app_value);

/*
 * STREAMING API
 */

/*! \brief Extra enums.
 *
 * \ingroup group_streaming
 */
enum vx_node_state_enum_e
{
    VX_ENUM_NODE_STATE_TYPE     = 0x23, /*!< \brief Node state type enumeration. */
};

/*! \brief Node state
 *
 * \ingroup group_streaming
 */
enum vx_node_state_e {

    /*! \brief Node is in steady state (output expected for each invocation)
     */
    VX_NODE_STATE_STEADY  = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NODE_STATE_TYPE) + 0x0,

    /*! \brief Node is in pipeup state (output not expected for each invocation)
     */
    VX_NODE_STATE_PIPEUP  = VX_ENUM_BASE(VX_ID_KHRONOS, VX_ENUM_NODE_STATE_TYPE) + 0x1,

};

/*! \brief The node attributes added by this extension.
 * \ingroup group_streaming
 */
enum vx_node_attribute_streaming_e {
    /*! \brief Queries the state of the node. Read-only. See <tt>\ref vx_graph_state_e</tt> enum. */
    VX_NODE_STATE = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_NODE) + 0x9,
};

/*! \brief The kernel attributes added by this extension.
 * \ingroup group_streaming
 */
enum vx_kernel_attribute_streaming_e {
    /*! \brief The pipeup output depth required by the kernel.
     * This is called by kernels that need to be primed with multiple output buffers before it can
     * begin to return them.  A typical use case for this is a source node which needs to provide and
     * retain multiple empty buffers to a camera driver to fill.  The first time the graph is executed
     * after vxVerifyGraph is called, the framework calls the node associated with this kernel
     * (pipeup_output_depth - 1) times before 'expecting' a valid output and calling downstream nodes.
     * During this PIPEUP state, the framework provides the same set of input parameters for each
     * call, but provides different set of output parameters for each call.  During the STEADY state,
     * the kernel may return a different set of output parameters than was given during the execution callback.
     * Read-write. Can be written only before user-kernel finalization.
     * Use a <tt>\ref vx_uint32</tt> parameter.
     * \note If not set, it will default to 1.
     * \note Setting a value less than 1 shall return VX_ERROR_INVALID_PARAMETERS
     */
    VX_KERNEL_PIPEUP_OUTPUT_DEPTH = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x4,

    /*! \brief The pipeup input depth required by the kernel.
     * This is called by kernels that need to retain one or more input buffers before it can
     * begin to return them.  A typical use case for this is a sink node which needs to provide and
     * retain one or more filled buffers to a display driver to display.  The first (pipeup_input_depth - 1)
     * times the graph is executed after vxVerifyGraph is called, the framework calls the node associated with this kernel
     * without 'expecting' an input to have been consumed and returned by the node.
     * During this PIPEUP state, the framework does not reuse any of the input bufers it had given to this node.
     * During the STEADY state, the kernel may return a different set of input parameters than was given during
     * the execution callback.
     * Read-write. Can be written only before user-kernel finalization.
     * Use a <tt>\ref vx_uint32</tt> parameter.
     * \note If not set, it will default to 1.
     * \note Setting a value less than 1 shall return VX_ERROR_INVALID_PARAMETERS
     */
    VX_KERNEL_PIPEUP_INPUT_DEPTH = VX_ATTRIBUTE_BASE(VX_ID_KHRONOS, VX_TYPE_KERNEL) + 0x5,
};

/*! \brief Enable streaming mode of graph execution
 *
 * This API enables streaming mode of graph execution on the given graph. The node given on the API is set as the
 * trigger node. A trigger node is defined as the node whose completion causes a new execution of the graph to be
 * triggered.
 *
 * \param graph [in] Reference to the graph to enable streaming mode of execution.
 * \param trigger_node  [in][optional] Reference to the node to be used for trigger node of the graph.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid <tt>\ref vx_graph</tt> reference
 *
 * \ingroup group_streaming
 */
VX_API_ENTRY vx_status VX_API_CALL vxEnableGraphStreaming(vx_graph graph, vx_node trigger_node);

/*! \brief Start streaming mode of graph execution
 *
 * In streaming mode of graph execution, once an application starts graph execution
 * further intervention of the application is not needed to re-schedule a graph;
 * i.e. a graph re-schedules itself and executes continuously until streaming mode of execution is stopped.
 *
 * When this API is called, the framework schedules the graph via <tt>\ref vxScheduleGraph</tt> and
 * returns.
 * This graph gets re-scheduled continuously until <tt>\ref vxStopGraphStreaming</tt> is called by the user
 * or any of the graph nodes return error during execution.
 *
 * The graph MUST be verified via \ref vxVerifyGraph before calling this API.
 * Also user application MUST ensure no previous executions of the graph are scheduled before calling this API.
 *
 * After streaming mode of a graph has been started, a <tt>\ref vxScheduleGraph</tt> should **not** be used on that
 * graph by an application.
 *
 * <tt>\ref vxWaitGraph</tt> can be used as before to wait for all pending graph executions
 * to complete.
 *
 * \param graph [in] Reference to the graph to start streaming mode of execution.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid <tt>\ref vx_graph</tt> reference.
 *
 * \ingroup group_streaming
 */
VX_API_ENTRY vx_status VX_API_CALL vxStartGraphStreaming(vx_graph graph);


/*! \brief Stop streaming mode of graph execution
 *
 * This function blocks until graph execution is gracefully stopped at a logical boundary, for example,
 * when all internally scheduled graph executions are completed.
 *
 * \param graph [in] Reference to the graph to stop streaming mode of execution.
 *
 * \return A <tt>\ref vx_status_e</tt> enumeration.
 * \retval VX_SUCCESS No errors; any other value indicates failure.
 * \retval VX_FAILURE Graph is not started in streaming execution mode.
 * \retval VX_ERROR_INVALID_REFERENCE graph is not a valid reference.
 *
 * \ingroup group_streaming
 */
VX_API_ENTRY vx_status VX_API_CALL vxStopGraphStreaming(vx_graph graph);

#ifdef  __cplusplus
}
#endif

#endif
