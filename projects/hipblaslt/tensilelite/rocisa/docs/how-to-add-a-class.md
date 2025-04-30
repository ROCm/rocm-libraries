# How to Add a Class in rocIsa

1. Write a class or structure as usual.

    ```c++
    struct Test
    {
        int testInt;
    };
    ```

2. If yourclass or structure inherits structures such as ``Item``, ``Instruction``. you'll have to add something more

    ```c++
    struct Test : public Item
    {
        // 1. A copy constructor
        Test(const Test& other)
        {
            // your code here, make sure you deepcopy any pointer.
        }

        // 2. You need to overwrite this in order to make deepcopy work properly.
        std::shared_ptr<Item> clone() const
        {
            return std::make_shared<Test>(*this);
        }
    };

    ```

3. Create a new cpp or use an existing cpp if your class is related to one of the files.
4. Inside the cpp add the following code

    ```c++
    #include <nanobind/nanobind.h>
    #include <nanobind/stl/shared_ptr.h>  // Need to add this if you are exposing shared_ptr

    namespace nb = nanobind;

    void init_test(nb::module_ m)
    {
        nb::class_<Test>(m, "Test")
            .def(nb::init<>())
            .def("__deepcopy__", [](const rocisa::Test& self, nb::dict&) { return new rocisa::Test(self); });
    }
    ```
