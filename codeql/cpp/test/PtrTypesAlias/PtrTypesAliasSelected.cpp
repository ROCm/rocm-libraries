using size_t = long;

namespace std
{
    template <typename T>
    class shared_ptr
    {
    };
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
        std::shared_ptr<TypeC> s1; // replace with TypeCPtr
        std::shared_ptr<TypeB> s2; // replace with TypeBPtr
    }

    std::shared_ptr<TypeB> s3; // replace with TypeBPtr
}
