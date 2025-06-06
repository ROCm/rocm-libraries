/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

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

#include <rocRoller/Serialization/Base_fwd.hpp>

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
        struct JustNaryArgument
        {
            NaryArgument argument;

            auto operator<=>(JustNaryArgument const&) const = default;
        };

        struct TypeAndSubDimension
        {
            std::string id;
            int         subdimension;

            bool operator==(TypeAndSubDimension const& other) const
            {
                return this->id == other.id && this->subdimension == other.subdimension;
            }
        };

        bool inline operator<(TypeAndSubDimension const& a, TypeAndSubDimension const& b)
        {
            if(a.id == b.id)
                return a.subdimension < b.subdimension;
            return a.id < b.id;
        }

        struct TypeAndNaryArgument
        {
            std::string  id;
            NaryArgument argument;

            auto operator<=>(TypeAndNaryArgument const&) const = default;
        };

        template <typename T>
        TypeAndNaryArgument inline typeArgument(NaryArgument arg)
        {
            return TypeAndNaryArgument{name<T>(), arg};
        }

        bool inline operator<(TypeAndNaryArgument const& a, TypeAndNaryArgument const& b)
        {
            if(a.id == b.id)
                return a.argument < b.argument;
            return a.id < b.id;
        }

        enum class ComputeIndexArgument : int
        {
            TARGET = 0,
            INCREMENT,
            BASE,
            OFFSET,
            STRIDE,
            BUFFER,

            Count
        };

        std::string   toString(ComputeIndexArgument cia);
        std::ostream& operator<<(std::ostream&, ComputeIndexArgument const&);

        struct ComputeIndex
        {
            ComputeIndexArgument argument;
            int                  index = 0;

            bool operator==(ComputeIndex const& other) const
            {
                return this->index == other.index && this->argument == other.argument;
            }
        };

        bool inline operator<(ComputeIndex const& a, ComputeIndex const& b)
        {
            if(a.argument == b.argument)
                return a.index < b.index;
            return a.argument < b.argument;
        }

        using ConnectionSpec = std::variant<std::monostate,
                                            JustNaryArgument,
                                            ComputeIndex,
                                            TypeAndSubDimension,
                                            TypeAndNaryArgument>;

        std::string   name(ConnectionSpec const& cs);
        std::string   toString(ConnectionSpec const& cs);
        std::ostream& operator<<(std::ostream& stream, ConnectionSpec const& cs);

    }

    struct DeferredConnection
    {
        Connections::ConnectionSpec connectionSpec;
        int                         coordinate;
    };

    template <typename T>
    inline DeferredConnection DC(int coordinate, int sdim = 0)
    {
        DeferredConnection rv;
        rv.connectionSpec = Connections::TypeAndSubDimension{name<T>(), sdim};
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
        using key_type = std::tuple<int, Connections::ConnectionSpec>;

    public:
        struct Connection
        {
            int                         control;
            int                         coordinate;
            Connections::ConnectionSpec connection;
        };

        /**
         * @brief Adds the given connection.
         */
        void connect(Connection const& connection);

        /**
         * @brief Connects the control flow node `control` to the coordinate `coordinate`.
         */
        void connect(int control, int coordinate, Connections::ConnectionSpec conn);
        void connect(int control, int coordinate, NaryArgument arg);

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
        void disconnect(int control, int coordinate, Connections::ConnectionSpec conn);

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
         * @brief Purges all connections into coordinate node `coordinate`.
         */
        void purgeMappingsTo(int coordinate);

        /**
         * @brief Get the coordinate index associated with the control
         * flow node `control`.
         */
        template <typename T>
        int get(int control, int subDimension = 0) const;

        /**
         * @brief Get the coordinate index associated with the control
         * flow node `control`.
         */
        int get(int control, Connections::ConnectionSpec conn = {}) const;

        /**
         * @brief Get the coordinate index associated with the control
         * flow node `control`.
         */
        int get(int control, NaryArgument arg) const;

        /**
         * @brief Get the all control nodes.
         */
        std::vector<int> getControls() const;

        /**
         * @brief Get all connections emanating from the control flow
         * node `control`.
         */
        std::vector<Connection> getConnections(int control) const;

        /**
         * @brief Get all connections.
         */
        std::vector<Connection> getConnections() const;

        /**
         * @brief Get all connections incoming to the coordinate `coordinate`.
         */
        std::vector<Connection> getCoordinateConnections(int coordinate) const;

        /**
         * @brief Emit DOT representation of connections.
         *
         * Currently, addLabels will use the hash id for any connections which makes this representation
         * non-portable between compilers.
         */
        std::string
            toDOT(std::string const& coord, std::string const& cntrl, bool addLabels = false) const;

    private:
        struct HashConnectionSpec
        {
            size_t operator()(const Connections::ConnectionSpec& conn) const
            {
                return conn.index();
            }
        };

        std::unordered_map<int,
                           std::unordered_map<Connections::ConnectionSpec, int, HashConnectionSpec>>
            m_map;

        //
        // A reverse mapping of m_map. Whenever m_map gets changed (add/delete), this should
        // also be updated.
        //
        std::unordered_map<int, std::unordered_set<int>> m_coordToControl;
    };

    std::string toString(ControlToCoordinateMapper::Connection const& conn);

}

#include <rocRoller/KernelGraph/ControlToCoordinateMapper_impl.hpp>
