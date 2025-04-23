# How to Add a Class in rocIsa

1. Write a function as usual.

    ```c++
    void test(const std::string& name);
    ```

2. Create a new cpp or use an existing cpp if your class is related to one of the files.
3. Inside the cpp add the following code

    ```c++
    #include <nanobind/nanobind.h>
    #include <nanobind/stl/string.h>  // Need to add this because test is using std::string

    namespace nb = nanobind;

    void init_test(nb::module_ m)
    {
        m.def("test", &test);
    }
    ```
