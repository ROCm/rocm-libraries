#pragma once

#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <typeindex>
#include <variant>
#include <vector>

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Utilities/Comparison.hpp>

namespace rocRoller::KernelGraph
{

    /**
     * @brief Connection specifiers.
     *
     * Connections are used to uniquely connect a control flow graph
     * node to a coordinate transform graph node.
     */
    namespace Connections
    {
        struct TypeAndSubDimension
        {
            std::type_index id;
            int             subdimension;
        };

        bool inline operator<(TypeAndSubDimension const& a, TypeAndSubDimension const& b)
        {
            if(a.id == b.id)
                return a.subdimension < b.subdimension;
            return a.id < b.id;
        }

        struct TypeAndNaryArgument
        {
            std::type_index id;
            NaryArgument    argument;
        };

        template <typename T>
        TypeAndNaryArgument inline typeArgument(NaryArgument arg)
        {
            return TypeAndNaryArgument{typeid(T), arg};
        }

        bool inline operator<(TypeAndNaryArgument const& a, TypeAndNaryArgument const& b)
        {
            if(a.id == b.id)
                return a.argument < b.argument;
            return a.id < b.id;
        }

        enum class ComputeIndexArgument : int
        {
            TARGET,
            INCREMENT,
            BASE,
            OFFSET,
            STRIDE,
            ZERO,
            BUFFER
        };

        struct ComputeIndex
        {
            ComputeIndexArgument argument;
            int                  index = 0;
        };

        bool inline operator<(ComputeIndex const& a, ComputeIndex const& b)
        {
            if(a.argument == b.argument)
                return a.index < b.index;
            return a.argument < b.argument;
        }
    }

    using ConnectionSpec = std::variant<std::monostate,
                                        NaryArgument,
                                        Connections::ComputeIndex,
                                        Connections::TypeAndSubDimension,
                                        Connections::TypeAndNaryArgument>;

    struct DeferredConnection
    {
        ConnectionSpec connectionSpec;
        int            coordinate;
    };

    template <typename T>
    inline DeferredConnection DC(int coordinate, int sdim = 0)
    {
        DeferredConnection rv;
        rv.connectionSpec = Connections::TypeAndSubDimension{typeid(T), sdim};
        rv.coordinate     = coordinate;
        return rv;
    }

    /**
     * @brief Connects nodes in the control flow graph to nodes in the
     * coordinate graph.
     *
     * For example, a LoadVGPR node in the control flow graph is
     * connected to a User (source) node, and VGPR (destination) node
     * in the coordinate graph.
     *
     * A single control flow node may be connected to multiple
     * coordinates.  To accomplish this, connection specifiers (see
     * ConnectionSpec) are used.
     */
    class ControlToCoordinateMapper
    {
        // key_type is:
        //  control graph index, connection specification
        using key_type = std::tuple<int, ConnectionSpec>;

        struct Connection
        {
            int            control;
            int            coordinate;
            ConnectionSpec connection;
        };

    public:
        /**
         * @brief Connects the control flow node `control` to the
         * coorindate `coordinate`.
         */
        void connect(int control, int coordinate, ConnectionSpec conn);

        /**
         * @brief Connects the control flow node `control` to the
         * coorindate `coordinate`.
         *
         * The type of coordinate is determined from the template
         * parameter.
         */
        template <typename T>
        void connect(int control, int coordinate, int subDimension = 0);

        /**
         * @brief Disconnects the control flow node `control` to the
         * coorindate `coordinate`.
         */
        void disconnect(int control, int coordinate, ConnectionSpec conn);

        /**
         * @brief Disconnects the control flow node `control` to the
         * coorindate `coordinate`.
         *
         * The type of coordinate is determined from the template
         * parameter.
         */
        template <typename T>
        void disconnect(int control, int coordinate, int subDimension = 0);

        /**
         * @brief Purges all connections out of control flow node `control`.
         */
        void purge(int control);

        /**
         * @brief Get the coordinate index associated with the control
         * flow node `control`.
         */
        template <typename T>
        int get(int control, int subDimension = 0) const;

        int get(int control, ConnectionSpec conn = {}) const;

        /**
         * @brief Get all connections emanating from the control flow
         * node `control`.
         */
        std::vector<Connection> getConnections(int control) const;

        /**
         * @brief Get all connections incoming to the coordinate `coordinate`.
         */
        std::vector<Connection> getCoordinateConnections(int coordinate) const;

        /**
         * @brief Emit DOT representation of connections.
         */
        std::string toDOT(std::string const& coord, std::string const& cntrl) const;

    private:
        std::map<key_type, int> m_map;
    };

}

#include <rocRoller/KernelGraph/ControlToCoordinateMapper_impl.hpp>
