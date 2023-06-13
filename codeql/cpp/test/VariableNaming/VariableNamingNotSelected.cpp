class NotSelected
{
    int m_camelCase;
    int m_camelCaseHaHa;
    int snake_case : 3; // an exception as some CodeGen functions convert these variables to strings that need to follow Python naming

public:
    int camelCase;
    int camelCaseHaHa;
    int snake_cases : 3; // an exception as some CodeGen functions convert these variables to strings that need to follow Python naming
};

static int __not_checkedCase_;
