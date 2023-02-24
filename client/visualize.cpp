
#include <msgpack.hpp>

#include "include/GraphInspector.hpp"

namespace rocRoller
{
    namespace Client
    {

        template <typename T>
        void writeMatrix(std::string const&    filename,
                         size_t                size0,
                         size_t                size1,
                         std::vector<T> const& matrix)
        {
            std::ofstream                  of(filename);
            msgpack::packer<std::ofstream> packer(of);

            packer.pack_map(2);

            packer.pack("sizes");
            packer.pack_array(2);
            packer.pack(size0);
            packer.pack(size1);

            packer.pack("data");
            packer.pack(matrix);
        }

        /**
         * Write a volume where the value is the workitem index, for all memory locations read
         * by one MacroTile (workgroup + K for loop iteration).
         */
        void writeMacrotileAByWorkitem(std::ostream&   vfile,
                                       GraphInspector& gi,
                                       int             loadADim,
                                       int             threadTileDim,
                                       int             threadTileSize)
        {
            auto invocation = gi.kernelInvocation();
            auto size0      = std::visit(to_size_t, gi.argValues().at("Load_Tiled_0_size_0"));
            auto size1      = std::visit(to_size_t, gi.argValues().at("Load_Tiled_0_size_1"));
            std::vector<int> matrix(size0 * size1, -10);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();

            AssertFatal(gi.kernelInvocation().workgroupSize[1] == 1);

            auto totalWorkgroupSize = product(invocation.workgroupSize);

            for(int j = 0; j < threadTileSize; j++)
            {
                gi.setCoordinate(threadTileDim, j);

                for(int i = 0; i < totalWorkgroupSize; i++)
                {
                    gi.setCoordinate(workitemIndices, i);

                    auto matIdx    = gi.getLoadIndex(loadADim);
                    matrix[matIdx] = i;
                    vfile << "Value for A for thread " << i << " TT " << j << ": " << matIdx
                          << std::endl;
                    vfile.flush();
                }
            }

            writeMatrix("workitem.dat", size0, size1, matrix);
        }

        /**
         * Write a volume where the value is the linear workgroup ID which reads the A matrix,
         * for all workgroups, for one iteration of the K loop.
         *
         * Skips the second workgroup dimension since that is associated with the B matrix.
         */
        void writeKIterAByWorkitem(std::ostream&   vfile,
                                   GraphInspector& gi,
                                   int             loadADim,
                                   int             threadTileDim,
                                   int             threadTileSize)
        {
            auto invocation     = gi.kernelInvocation();
            auto workgroupCount = invocation.workitemCount;
            for(int i = 0; i < 3; i++)
                workgroupCount[i] = invocation.workitemCount[i] / invocation.workgroupSize[i];
            vfile << "Workgroups: " << workgroupCount[0] << "x" << workgroupCount[1] << "x"
                  << workgroupCount[2] << std::endl;
            auto size0 = std::visit(to_size_t, gi.argValues().at("Load_Tiled_0_size_0"));
            auto size1 = std::visit(to_size_t, gi.argValues().at("Load_Tiled_0_size_1"));
            std::vector<int> matrix(size0 * size1, -10);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();
            std::array<std::vector<int>, 3> workgroupIndices;
            for(int i = 0; i < 3; i++)
            {
                workgroupIndices[i]
                    = gi.coords()
                          ->findElements(
                              isSubDim<KernelGraph::CoordinateGraph::Workgroup>(gi.coords(), i))
                          .to<std::vector>();
            }

            // The A matrix is not associated with workgroup[1] (only the B, C, and D matrices), so
            // don't loop over workgroup[1].  Instead, assign it to 0.
            gi.setCoordinate(workgroupIndices[1], 0);

            // workgroupCount[2] should be 1 for now, this may change when batching is introduced.
            AssertFatal(workgroupCount[2] == 1);
            gi.setCoordinate(workgroupIndices[2], 0);

            auto totalWorkgroupSize = product(invocation.workgroupSize);

            for(int threadIdx = 0; threadIdx < totalWorkgroupSize; threadIdx++)
            {
                gi.setCoordinate(workitemIndices, threadIdx);
                for(int tt = 0; tt < threadTileSize; tt++)
                {
                    gi.setCoordinate(threadTileDim, tt);
                    for(int blockIdx_x = 0; blockIdx_x < workgroupCount[0]; blockIdx_x++)
                    {
                        gi.setCoordinate(workgroupIndices[0], blockIdx_x);

                        auto matIdx    = gi.getLoadIndex(loadADim);
                        matrix[matIdx] = blockIdx_x;
                    }
                }
            }

            writeMatrix("workgroups.dat", size0, size1, matrix);
        }

        /**
         * Write a volume where the value is the K for loop index which reads the
         * A matrix, for the entire A matrix.
         */
        void writeAByKIter(std::ostream&   vfile,
                           GraphInspector& gi,
                           int             loadADim,
                           int             threadTileDim,
                           int             threadTileSize,
                           int             tileSizeK)
        {
            auto invocation     = gi.kernelInvocation();
            auto workgroupCount = invocation.workitemCount;
            for(int i = 0; i < 3; i++)
                workgroupCount[i] = invocation.workitemCount[i] / invocation.workgroupSize[i];
            vfile << "Workgroups: " << workgroupCount[0] << "x" << workgroupCount[1] << "x"
                  << workgroupCount[2] << std::endl;
            auto size0 = std::visit(to_size_t, gi.argValues().at("Load_Tiled_0_size_0"));
            auto size1 = std::visit(to_size_t, gi.argValues().at("Load_Tiled_0_size_1"));
            std::vector<int> matrix(size0 * size1, -10);

            auto workitemIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::Workitem>(gi.coords()))
                      .to<std::vector>();
            std::array<std::vector<int>, 3> workgroupIndices;
            for(int i = 0; i < 3; i++)
            {
                workgroupIndices[i]
                    = gi.coords()
                          ->findElements(
                              isSubDim<KernelGraph::CoordinateGraph::Workgroup>(gi.coords(), i))
                          .to<std::vector>();
            }

            auto forLoopIndices
                = gi.coords()
                      ->findElements(isNode<KernelGraph::CoordinateGraph::ForLoop>(gi.coords()))
                      .to<std::vector>();

            // The A matrix is not associated with workgroup[1] (only the B, C, and D matrices), so
            // don't loop over workgroup[1].  Instead, assign it to 0.
            gi.setCoordinate(workgroupIndices[1], 0);

            // workgroupCount[2] should be 1 for now, this may change when batching is introduced.
            AssertFatal(workgroupCount[2] == 1);
            gi.setCoordinate(workgroupIndices[2], 0);

            auto totalWorkgroupSize = product(invocation.workgroupSize);

            for(int threadIdx = 0; threadIdx < totalWorkgroupSize; threadIdx++)
            {
                gi.setCoordinate(workitemIndices, threadIdx);

                for(int tt = 0; tt < threadTileSize; tt++)
                {
                    gi.setCoordinate(threadTileDim, tt);

                    for(int blockIdx_x = 0; blockIdx_x < workgroupCount[0]; blockIdx_x++)
                    {
                        gi.setCoordinate(workgroupIndices[0], blockIdx_x);

                        for(int i = 0; i < size1 / tileSizeK; i++)
                        {
                            gi.setCoordinate(forLoopIndices, i);
                            auto matIdx    = gi.getLoadIndex(loadADim);
                            matrix[matIdx] = i;
                        }
                    }
                }
            }

            writeMatrix("loop_idx.dat", size0, size1, matrix);
        }

        auto threadTileSize(
            std::shared_ptr<rocRoller::KernelGraph::CoordinateGraph::CoordinateGraph> coords,
            size_t                                                                    size)
        {
            return [coords, size](int idx) -> bool {
                if(coords->getElementType(idx) != Graph::ElementType::Node)
                    return false;

                auto const& node
                    = std::get<KernelGraph::CoordinateGraph::Dimension>(coords->getElement(idx));
                if(!std::holds_alternative<KernelGraph::CoordinateGraph::ThreadTileIndex>(node))
                    return false;

                auto const& tt = std::get<KernelGraph::CoordinateGraph::ThreadTileIndex>(node);

                return std::visit(to_size_t, Expression::evaluate(tt.size)) == size;
            };
        }

        void visualize(std::shared_ptr<Command> command,
                       CommandKernel&           kc,
                       KernelArguments const&   commandArgs)
        {
            auto filename = Settings::getInstance()->get(Settings::LogFile) + "gemm.vis";
            std::cout << "Visualizing to " << filename << std::endl;
            std::ofstream vfile(filename);

            GraphInspector gi(command, kc, commandArgs);
            gi.inventExecutionCoordinates();

            int loadA = gi.findLoadTile(0);

            auto coords = gi.coords();

            // Find the main VGPR dimension for loading A. This is so far a ThreadTile
            // dimension with a size of 16 or 8.
            auto first_of = [coords](auto pred, auto&& gen) {
                for(auto idx : gen)
                {
                    if(pred(idx))
                        return idx;
                }
                return -1;
            };

            int ttNode = first_of(threadTileSize(coords, 16), coords->depthFirstVisit(loadA));
            int ttSize = 16;

            if(ttNode < 0)
            {
                ttNode = first_of(threadTileSize(coords, 8), coords->depthFirstVisit(loadA));
                AssertFatal(ttNode >= 0, "Could not find thread tile node for graph");
                ttSize = 8;
            }

            writeMacrotileAByWorkitem(vfile, gi, loadA, ttNode, ttSize);

            writeKIterAByWorkitem(vfile, gi, loadA, ttNode, ttSize);

            // NOTE: The last argument here needs to be changed to match the MacroTile K dimension.
            // It's currently set for the Guidepost value of 16.
            writeAByKIter(vfile, gi, loadA, ttNode, ttSize, 16);
        }

    }
}
