#ifndef DLRM_LDS_MAPPING_UTIL_HPP
#define DLRM_LDS_MAPPING_UTIL_HPP

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_transforms.hpp>
#pragma GCC diagnostic pop

#include <rocwmma/internal/io_config.hpp>
#include <rocwmma/internal/mapping_util.hpp>
#include <rocwmma/internal/utils.hpp>

namespace rocwmma
{
    class LdsKW
    {
    };
    class LdsKH
    {
    };
    class LdsKHC
    {
    };
    class LdsRF
    {
    };

    template <>
    constexpr const char* dataTypeToString<LdsKW>()
    {
        return "LdsKW";
    }

    template <>
    constexpr const char* dataTypeToString<LdsKH>()
    {
        return "LdsKH";
    }

    template <>
    constexpr const char* dataTypeToString<LdsKHC>()
    {
        return "LdsKHC";
    }

    template <>
    constexpr const char* dataTypeToString<LdsRF>()
    {
        return "LdsRF";
    }

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              typename MappingLds,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t BlocksX,
              uint32_t BlocksY>
    struct LdsMappingUtil;

    template <uint32_t BlockM,
              uint32_t BlockN,
              uint32_t BlockK,
              typename DataT,
              typename LayoutA,
              typename LayoutB,
              typename LayoutLds,
              uint32_t TBlockX,
              uint32_t TBlockY,
              uint32_t BlocksX,
              uint32_t BlocksY>
    struct LdsMappingUtil<BlockM,
                          BlockN,
                          BlockK,
                          DataT,
                          LayoutA,
                          LayoutB,
                          LayoutLds,
                          LdsRF,
                          TBlockX,
                          TBlockY,
                          BlocksX,
                          BlocksY>
    {
        /* LdsRF (Register file)
    * Lds usage is mapped as a vertical register file of width 64.
    * This geometry is compatible with both A and B fragments, as
    * fragments do not support partial registers, regardless of
    * DataLayout.
    *
    * There is no need for data transpose as A and B should already
    * be aligned.
    *
    * All that really needs to happen is we interpret the fragments
    * as registers and provide 1-1 mapping into LDS memory with width
    * of 64. The matrix_b type of LDS fragments is conducive to this
    * mapping as the current config writes rows of fragment data, such
    * as a vertical register file would need.
    *
    * In this case, block dimensions for A and B cannot be extended by
    * respective BlocksX and BlocksY, due to the LDS geometry which does
    * not easily map to sub-blocks.
    *
    * Local Layout (LDS): e.g. BlockMNK = (32, 32, 8)
    *
    *           Elements 0.....31 32.....64
    *                    _______________
    *           Reg0    |  C0   |   C4  |
    *  BlockA   Reg1    |  C1   |   C5  |
    *    (0)    Reg2    |  C2   |   C6  |
    *           Reg3    |  C3   |   C7  |
    *            ...       ...      ...
    *                    _______________
    *           Reg0    |  C0   |   C4  |
    *  BlockA   Reg1    |  C1   |   C5  |
    *   (X-1)   Reg2    |  C2   |   C6  |
    *           Reg3    |  C3   |   C7  |
    *
    *            ...       ...      ...
    *                    _______________
    *           Reg0    |  R0   |   R4  |
    *  BlockB   Reg1    |  R1   |   R5  |
    *    (0)    Reg2    |  R2   |   R6  |
    *           Reg3    |  R3   |   R7  |
    *            ...       ...      ...
    *                    _______________
    *           Reg0    |  R0   |   R4  |
    *  BlockB   Reg1    |  R1   |   R5  |
    *   (Y-1)   Reg2    |  R2   |   R6  |
    *           Reg3    |  R3   |   R7  |
    */

        static constexpr uint32_t registerFileWidth = Constants::AMDGCN_WAVE_SIZE;

        // Cooperative fragments
        using GlobalReadFragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;

        using GlobalReadFragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

        // Local Write
        // Vertical register file fulfilled by matrix_b with BlockN = 64
        using LocalWriteFragA
            = apply_data_layout_t<apply_register_file_t<GlobalReadFragA>, LayoutLds>;
        using LocalWriteFragB
            = apply_data_layout_t<apply_register_file_t<GlobalReadFragB>, LayoutLds>;

        // Sanity checks
        static_assert(GlobalReadFragA::size() == LocalWriteFragA::size(),
                      "Incompatible register count");
        static_assert(GlobalReadFragB::size() == LocalWriteFragB::size(),
                      "Incompatible register count");
        static_assert(LocalWriteFragA::size() * registerFileWidth == BlockM * BlockK,
                      "Incompatible element count");
        static_assert(LocalWriteFragB::size() * registerFileWidth == BlockN * BlockK,
                      "Incompatible element count");

        // Final fragments are same as global
        using FragA = fragment<matrix_a, BlockM, BlockN, BlockK, DataT, LayoutA>;
        using FragB = fragment<matrix_b, BlockM, BlockN, BlockK, DataT, LayoutB>;

        // Local read same as write
        using LocalReadFragA = apply_data_layout_t<apply_register_file_t<FragA>, LayoutLds>;
        using LocalReadFragB = apply_data_layout_t<apply_register_file_t<FragB>, LayoutLds>;

    private:
        // Calculate offsets based on DataLayout of LDS, A frags and B frags.
        // All of them will have the same workgroupDim and waveCoord values.
        using LdsOffsets     = GetMappingUtil_t<LocalWriteFragA>;
        using GlobalAOffsets = GetMappingUtil_t<GlobalReadFragA>;
        using GlobalBOffsets = GetMappingUtil_t<GlobalReadFragB>;

        template <typename FragT>
        using IOTraits = typename GetIOConfig_t<FragT>::IOTraits;

    public:
        __device__ static inline auto ldsWidth()
        {
            return registerFileWidth;
        }

        __device__ static inline auto ldsHeight()
        {
            auto workgroupDim = LdsOffsets::workgroupDim();
            return LocalWriteFragA::size() * BlocksX * get<0>(workgroupDim)
                   + LocalWriteFragB::size() * BlocksY * get<1>(workgroupDim);
        }

        __device__ static inline uint32_t ld()
        {
            return std::is_same<LayoutLds, row_major>::value ? ldsWidth() : ldsHeight();
        }

        __device__ static inline uint32_t baseOffsetA()
        {
            return 0;
        }

        __device__ static inline uint32_t baseOffsetB()
        {
            auto matrixCoord = make_coord2d(
                LocalWriteFragA::size() * BlocksX * get<0>(LdsOffsets::workgroupDim()), 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetA()
        {
            auto matrixCoord = make_coord2d(
                LocalWriteFragA::size() * BlocksX * get<0>(LdsOffsets::waveCoord()), 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t waveOffsetB()
        {
            auto matrixCoord = make_coord2d(
                LocalWriteFragB::size() * BlocksY * get<1>(LdsOffsets::waveCoord()), 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetA(uint32_t blockX)
        {
            auto matrixCoord = make_coord2d(LocalWriteFragA::size() * blockX, 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline uint32_t blockOffsetB(uint32_t blockY)
        {
            auto matrixCoord = make_coord2d(LocalWriteFragB::size() * blockY, 0);
            return LdsOffsets::dataOffset(matrixCoord, ld());
        }

        __device__ static inline void
            prefetchCoopGlobalA(DataT* baseLds, DataT const* baseA, uint32_t lda)
        {
            // Cooperative transfer
            for(int32_t i = 0; i < BlocksX; ++i)
            {
                // Issue global read
                GlobalReadFragA fetchA;
                load_matrix_sync(fetchA,
                                 baseA
                                     + GlobalAOffsets::dataOffset(make_coord2d(BlockM * i, 0), lda),
                                 lda);

                // Issue local store
                store_matrix_sync(baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(i),
                                  reinterpret_cast<LocalWriteFragA&>(fetchA),
                                  ld());
            }
        }

        __device__ static inline void
            prefetchCoopGlobalB(DataT* baseLds, DataT const* baseB, uint32_t ldb)
        {
            // Cooperative transfer
            for(int32_t i = 0; i < BlocksY; ++i)
            {
                // Issue global read
                GlobalReadFragB fetchB;
                load_matrix_sync(fetchB,
                                 baseB
                                     + GlobalBOffsets::dataOffset(make_coord2d(0, BlockN * i), ldb),
                                 ldb);

                // Issue local store
                store_matrix_sync(baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(i),
                                  reinterpret_cast<LocalWriteFragB&>(fetchB),
                                  ld());
            }
        }

        __device__ static inline void
            prefetchGlobalA(DataT* baseLds, DataT const* baseA, uint32_t lda)
        {
            for(int32_t i = 0; i < BlocksX; ++i)
            {
                // Issue global read
                GlobalReadFragA fetchA;
                load_matrix_sync(fetchA,
                                 baseA
                                     + GlobalAOffsets::dataOffset(make_coord2d(BlockM * i, 0), lda),
                                 lda);

                // Issue local store
                store_matrix_sync(baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(i),
                                  reinterpret_cast<LocalWriteFragA&>(fetchA),
                                  ld());
            }
        }

        __device__ static inline void
            prefetchGlobalB(DataT* baseLds, DataT const* baseB, uint32_t ldb)
        {
            for(int32_t i = 0; i < BlocksY; ++i)
            {
                // Issue global read
                GlobalReadFragB fetchB;
                load_matrix_sync(fetchB,
                                 baseB
                                     + GlobalBOffsets::dataOffset(make_coord2d(0, BlockN * i), ldb),
                                 ldb);

                // Issue local store
                store_matrix_sync(baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(i),
                                  reinterpret_cast<LocalWriteFragB&>(fetchB),
                                  ld());
            }
        }

        __device__ static inline void
            prefetchLocalA(FragA& fragA, DataT const* baseLds, uint32_t blockX)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragA&>(fragA),
                             baseLds + baseOffsetA() + waveOffsetA() + blockOffsetA(blockX),
                             ld());
        }

        __device__ static inline void
            prefetchLocalB(FragB& fragB, DataT const* baseLds, uint32_t blockY)
        {
            load_matrix_sync(reinterpret_cast<LocalReadFragB&>(fragB),
                             baseLds + baseOffsetB() + waveOffsetB() + blockOffsetB(blockY),
                             ld());
        }

        __device__ static inline void prefetchLocalA(FragA* fragsA, DataT const* baseLds)
        {
            for(int i = 0; i < BlocksX; i++)
            {
                prefetchLocalA(fragsA[i], baseLds, i);
            }
        }

        __device__ static inline void prefetchLocalB(FragB* fragsB, DataT const* baseLds)
        {
            for(int i = 0; i < BlocksY; i++)
            {
                prefetchLocalB(fragsB[i], baseLds, i);
            }
        }
    };

} // namespace rocwmma

#endif // DLRM_LDS_MAPPING_UTIL_HPP
