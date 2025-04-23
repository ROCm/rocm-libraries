# How to create a new Custom function using vx_amd_custom extension

## When to use Custom OpenVX Node

The idea of custom Node is to allow the following
* To allow the user to plug-in an a kernel which is not yet supported
*	User wants to write an operator based on a third-party library
*	User wants to optimize the functionality by providing a their own implementation or wants to fuse functions

In this tutorial, we will walk you through the process of writing, compiling, and loading a vx_amd_custom lib with a new custom function. For demonstration purposes we will provide a CPU and a GPU implementation for the "CustomCopy" kernel. The implementation copies the input data to the output without any modifications.

## Prerequisites
* MIVisionX library installed from source
* Knowledge of OpenVX and how to add a user defined node
* C++ coding
* Basic knowledge of CMake

## Write the operator definition in custom_lib using predefined APIs in custom_api.h under custom_lib folder
The Custom shared library (custom_lib) has four main APIs as shown below
``` 
// Create Custom Object and retun the handle
customHandle CreateCustom(CustomFunctionType function);

// Setup custom function execution
customStatus_t CustomSetup(customHandle input_handle, customTensorDesc &inputdesc, customTensorDesc &outputdesc, customBackend backend, customStream stream);

// Execute custom function
customStatus_t CustomExecute(customHandle custom_handle, void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc);

// Destroy custom execution instance
customStatus_t CustomShutdown(customHandle custom_handle);

``` 

* In the header file only one function is present which is "Copy" for demonstration.

### Write the implementation of the function using the custom_template base class 
custom_base class is an abstract class which exposes the API for the custom_operator implemetation. Basically this has the following declarations

``` 
class custom_base
{
protected:  
    custom_base() {};
public:
    virtual ~custom_base() {};
    /*!
     \param inputdesc => Input tensor desc
     \param outputdesc => output tensor desc
     \param backend  => backend for the impl
     \param stream  => Output command queue

    */
    virtual customStatus_t Setup(customTensorDesc &inputdesc, customTensorDesc &outputdesc,  customBackend backend, customStream stream, int num_cpu_threads) = 0;
    /*!
     \param input_handle  => memory handle of input tensor
     \param inputdesc => Input tensor desc
     \param output_handle  => memory handle of output tensor
     \param inputdesc => Input tensor desc
    */
    virtual customStatus_t Execute(void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc) = 0;
     
    //* Shutdown and release resources */
    virtual customStatus_t Shutdown() = 0;
};

``` 
### Add new function types for supporting new custom api
``` 
custom_base * CreateCustomClass(CustomFunctionType function) {

    switch(function)
    {
        case Copy:
            return new customCopy();
            break;
        // todo:: add new custom function types here with corresponding implemetation files
        default:
            throw std::runtime_error ("Custom function type is unsupported");
            return nullptr;
    }
}

``` 
### Provide the implementation of custom function in the derived class for the Operator (here custom_copy_impl.h)
* The function just implements a copy operator which simply copies the content of input_tensor into output_tensor. Below code shows the implementation for both CPU and GPU backend using ROCm HIP implementation.
``` 
customStatus_t customCopy::Execute(void *input_handle, customTensorDesc &inputdesc, void *output_handle, customTensorDesc &outputdesc)
{
    unsigned size = outputdesc.dims[0] * outputdesc.dims[1] * outputdesc.dims[3] * sizeof(_output_desc.data_type);
    unsigned batch_size = outputdesc.dims[3];
    if (_backend == customBackend::CPU)
    {
        int omp_threads =  (_cpu_num_threads < batch_size)?  _cpu_num_threads: batch_size;
    #pragma omp parallel for num_threads(omp_threads)
        for (size_t i = 0; i < batch_size; i++) {
            unsigned char *src, *dst;
            src = (unsigned char *)input_handle + size*i;
            dst = (unsigned char *)output_handle + size*i;
            memcpy(dst, src, size);
        }
    }else
    {
#if ENABLE_HIP
        for (size_t i = 0; i < batch_size; i++) {
            unsigned char *src, *dst;
            src = (unsigned char *)input_handle + size*i;
            dst = (unsigned char *)output_handle + size*i;
            hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice);
        }
#endif
    }
    return customStatusSuccess;
}
``` 

### Link the custom_lib to vx_amd_custom node

* The implementation of vx_amd_custom extension node for OpenVX is pretty staight forward and follows the OpenVX extension specification. More details can be found in the Readme file.
 - [Readme](./README.md)
 
``` 
/*! \brief [Graph] Creates a Custom Layer Node.
 * \param [in] graph The handle to the graph.
 * \param [in] inputs The input tensor data.
 * \param [in] function custom funtion enum.
 * \param [in] array for user specified custom_parameters.
 * \param [out] outputs The output tensor data.
 * \return <tt> vx_node</tt>.
 * \returns A node reference <tt>\ref vx_node</tt>. Any possible errors preventing a
 * successful creation should be checked using <tt>\ref vxGetStatus</tt>.
 */
VX_API_ENTRY vx_node VX_API_CALL vxCustomLayer(vx_graph graph, vx_tensor inputs, vx_enum function, vx_array custom_parameters, vx_tensor outputs);

``` 
### Adding a new operator type in the custom_lib
* Add enumeration for the new function type in [custom_api.h](./custom_lib/custom_api.h)
* Add a new class for new custom function in a seperate header file following [custom_copy_impl.h](./custom_lib/custom_copy_impl.h)
* Implement the three functions of the class in a seperate .cpp file following [custom_copy_impl.cpp](./custom_lib/custom_copy_impl.cpp)
* Invoke the new class by adding it in the CreateCustomClass() [custom_api.cpp](./custom_lib/custom_api.cpp)
* Modify CMakeLists.txt to include the implementation file in [CUSTOM_LIB_SOURCES](./custom_lib/CMakeLists.txt)
* Rebuild MIVisionX with the new custom operator. Voila! The new custom function is ready to work from vx_amd_custom extension

### Test the new custom operator using runvx utility
* Sample gdf for runvx using the "Copy" operator can be found under [Readme](./README.md)
* Modify the gdf approproately to test the new custom functionality
