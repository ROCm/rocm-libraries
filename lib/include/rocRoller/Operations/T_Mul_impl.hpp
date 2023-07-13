
namespace rocRoller
{
    namespace Operations
    {
        inline T_Mul::T_Mul(int dest, int a, int b)
            : dest(dest)
            , a(a)
            , b(b)
        {
        }

        inline void T_Mul::setCommand(CommandPtr command)
        {
            m_command = command;
        }

        inline int T_Mul::getTag() const
        {
            return dest;
        }

        inline void T_Mul::setTag(int tag)
        {
            dest = tag;
        }

        inline std::unordered_set<int> T_Mul::getInputs() const
        {
            return {a, b};
        }

        inline std::string T_Mul::toString() const
        {
            return "T_Mul";
        }
    }
}
