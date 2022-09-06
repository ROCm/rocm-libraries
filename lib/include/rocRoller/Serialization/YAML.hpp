
#ifdef ROCROLLER_USE_LLVM
#include <rocRoller/Serialization/llvm/YAML.hpp>
#endif
#ifdef ROCROLLER_USE_YAML_CPP
#include <rocRoller/Serialization/yaml-cpp/YAML.hpp>
#endif

#ifdef ROCROLLER_USE_LLVM
namespace llvm
{
    namespace yaml
    {
        template <rocRoller::Serialization::MappedType<IO> T>
        struct MappingTraits<T>
        {
            using obj        = T;
            using TheMapping = rocRoller::Serialization::MappingTraits<obj, IO>;

            static void mapping(IO& io, obj& o)
            {
                mapping(io, o, nullptr);
            }

            static void mapping(IO& io, obj& o, void*)
            {
                rocRoller::Serialization::EmptyContext ctx;
                TheMapping::mapping(io, o, ctx);
            }
        };
    }
}
#endif
