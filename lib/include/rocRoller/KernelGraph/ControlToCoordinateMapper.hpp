#pragma once

#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <typeindex>
#include <vector>

namespace rocRoller::KernelGraph
{

    /**
     * @brief Connects nodes in the control flow graph to nodes in the
     * coordinate graph.
     *
     * For example, a LoadVGPR node in the control flow graph is
     * connected to a User (source) node, and VGPR (destination) node
     * in the coordinate graph.
     *
     * A single control flow node may be connected to multiple
     * coordinates.  However, the combination of
     *
     * 1. control index,
     * 2. coordinate type, and
     * 3. coordinate subdimension
     *
     * must be unique.
     */
    class ControlToCoordinateMapper
    {
        // key_type is:
        //  control graph index,
        //  dimension type index, and
        //  sub-dimension number
        using key_type = std::tuple<int, std::type_index, int>;

        struct Connection
        {
            int             control;
            std::type_index tindex;
            int             subDimension;
            int             coordinate;
        };

    public:
        /**
         * @brief Connects the control flow node `control` to the
         * coorindate `coordinate`.
         */
        void connect(int control, int coordinate, std::type_index tindex, int subDimension);

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
        void disconnect(int control, int coordinate, std::type_index tindex, int subDimension);

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
         * @brief Get the coordinate index associated with the control
         * flow node `control`.
         */
        template <typename T>
        int get(int control, int subDimension = 0) const;

        /**
         * @brief Get all connections emanating from the control flow
         * node `control`.
         */
        std::vector<Connection> getConnections(int control) const;

        /**
         * @brief Emit DOT representation of connections.
         */
        std::string toDOT(std::string coord, std::string cntrl) const;

    private:
        std::map<key_type, int> m_map;
    };

}

#include <rocRoller/KernelGraph/ControlToCoordinateMapper_impl.hpp>
