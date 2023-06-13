using size_t = long;

namespace std
{
    template <typename T>
    class shared_ptr
    {
    public:
        shared_ptr(T* ptr = nullptr) {}
    };
    template <typename T, typename... Args>
    shared_ptr<T> make_shared(Args... args)
    {
        return shared_ptr<T>(new T(args...));
    }
}

class TypeA
{
};

namespace rocRoller
{
    class TypeB
    {
    };
    using TypeBPtr = std::shared_ptr<TypeB>;
    namespace inner
    {
        class TypeC
        {
        };
        using TypeCPtr = std::shared_ptr<TypeC>;
        TypeCPtr n2; // is OK
    }
    inner::TypeCPtr               n0; // is OK
    std::shared_ptr<inner::TypeC> n1; // is OK
    auto                          n2 = std::make_shared<inner::TypeC>();
}
