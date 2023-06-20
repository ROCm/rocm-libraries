
#pragma once

#define RR_EMPTY_STRUCT_WITH_NAME(cls)          \
    struct cls                                  \
    {                                           \
        static constexpr bool HasValue = false; \
                                                \
        std::string toString() const            \
        {                                       \
            return #cls;                        \
        }                                       \
                                                \
        std::string name() const                \
        {                                       \
            return #cls;                        \
        }                                       \
    }

#define RR_CLASS_NAME_IMPL(cls)   \
    std::string cls::name() const \
    {                             \
        return #cls;              \
    }
