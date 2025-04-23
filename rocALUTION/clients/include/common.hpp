/* ************************************************************************
 * Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include <cstring>
#include <map>
#include <mpi.h>
#include <rocalution/rocalution.hpp>
#include <set>

using namespace rocalution;

static void my_irecv(int* buf, int count, int source, int tag, MPI_Comm comm, MPI_Request* request)
{
    MPI_Irecv(buf, count, MPI_INT, source, tag, comm, request);
}

static void
    my_irecv(int64_t* buf, int count, int source, int tag, MPI_Comm comm, MPI_Request* request)
{
    MPI_Irecv(buf, count, MPI_INT64_T, source, tag, comm, request);
}

static void
    my_isend(const int* buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request* request)
{
    MPI_Isend(buf, count, MPI_INT, dest, tag, comm, request);
}

static void
    my_isend(const int64_t* buf, int count, int dest, int tag, MPI_Comm comm, MPI_Request* request)
{
    MPI_Isend(buf, count, MPI_INT64_T, dest, tag, comm, request);
}

template <typename ValueType>
void distribute_matrix(const MPI_Comm*          comm,
                       LocalMatrix<ValueType>*  lmat,
                       GlobalMatrix<ValueType>* gmat,
                       ParallelManager*         pm)
{
    int rank;
    int num_procs;

    MPI_Comm_rank(*comm, &rank);
    MPI_Comm_size(*comm, &num_procs);

    int64_t global_nrow = lmat->GetM();
    int64_t global_ncol = lmat->GetN();
    int64_t global_nnz  = lmat->GetNnz();

    PtrType*   global_row_offset = NULL;
    int*       global_col        = NULL;
    ValueType* global_val        = NULL;

    lmat->LeaveDataPtrCSR(&global_row_offset, &global_col, &global_val);

    // If we have only a single MPI rank, we are done
    if(num_procs == 1)
    {
        pm->SetMPICommunicator(comm);
        pm->SetGlobalNrow(global_nrow);
        pm->SetGlobalNcol(global_ncol);
        pm->SetLocalNrow(global_nrow);
        pm->SetLocalNcol(global_ncol);

        gmat->SetParallelManager(*pm);
        gmat->SetLocalDataPtrCSR(&global_row_offset, &global_col, &global_val, "mat", global_nnz);

        return;
    }

    // Compute local matrix sizes
    std::vector<int> local_size(num_procs);

    for(int i = 0; i < num_procs; ++i)
    {
        local_size[i] = global_nrow / num_procs;
    }

    if(global_nrow % num_procs != 0)
    {
        for(int i = 0; i < global_nrow % num_procs; ++i)
        {
            ++local_size[i];
        }
    }

    // Compute index offsets
    std::vector<PtrType> index_offset(num_procs + 1);
    index_offset[0] = 0;
    for(int i = 0; i < num_procs; ++i)
    {
        index_offset[i + 1] = index_offset[i] + local_size[i];
    }

    // Read sub matrix - row_offset
    int                  local_nrow = local_size[rank];
    std::vector<PtrType> local_row_offset(local_nrow + 1);

    for(PtrType i = index_offset[rank], k = 0; k < local_nrow + 1; ++i, ++k)
    {
        local_row_offset[k] = global_row_offset[i];
    }

    free_host(&global_row_offset);

    // Read sub matrix - col and val
    PtrType                local_nnz = local_row_offset[local_nrow] - local_row_offset[0];
    std::vector<int>       local_col(local_nnz);
    std::vector<ValueType> local_val(local_nnz);

    for(PtrType i = local_row_offset[0], k = 0; k < local_nnz; ++i, ++k)
    {
        local_col[k] = global_col[i];
        local_val[k] = global_val[i];
    }

    free_host(&global_col);
    free_host(&global_val);

    // Shift row_offset entries
    int shift = local_row_offset[0];
    for(int i = 0; i < local_nrow + 1; ++i)
    {
        local_row_offset[i] -= shift;
    }

    PtrType interior_nnz = 0;
    PtrType ghost_nnz    = 0;
    int     boundary_nnz = 0;
    int     neighbors    = 0;

    std::vector<std::vector<PtrType>> boundary(num_procs, std::vector<PtrType>());
    std::vector<bool>                 neighbor(num_procs, false);
    std::vector<std::map<int, bool>>  checked(num_procs, std::map<int, bool>());

    for(int i = 0; i < local_nrow; ++i)
    {
        for(PtrType j = local_row_offset[i]; j < local_row_offset[i + 1]; ++j)
        {

            // Interior point
            if(local_col[j] >= index_offset[rank] && local_col[j] < index_offset[rank + 1])
            {
                ++interior_nnz;
            }
            else
            {
                // Boundary point above current process
                if(local_col[j] < index_offset[rank])
                {
                    // Loop over ranks above current process
                    for(int r = rank - 1; r >= 0; --r)
                    {
                        // Check if boundary belongs to rank r
                        if(local_col[j] >= index_offset[r] && local_col[j] < index_offset[r + 1])
                        {
                            // Add boundary point to rank r if it has not been added yet
                            if(!checked[r][i + index_offset[rank]])
                            {
                                boundary[r].push_back(i + index_offset[rank]);
                                neighbor[r] = true;
                                ++boundary_nnz;
                                checked[r][i + index_offset[rank]] = true;
                            }
                            ++ghost_nnz;
                            // Rank for current boundary point local_col[j] has been found
                            // Continue with next boundary point
                            break;
                        }
                    }
                }

                // boundary point below current process
                if(local_col[j] >= index_offset[rank + 1])
                {
                    // Loop over ranks above current process
                    for(int r = rank + 1; r < num_procs; ++r)
                    {
                        // Check if boundary belongs to rank r
                        if(local_col[j] >= index_offset[r] && local_col[j] < index_offset[r + 1])
                        {
                            // Add boundary point to rank r if it has not been added yet
                            if(!checked[r][i + index_offset[rank]])
                            {
                                boundary[r].push_back(i + index_offset[rank]);
                                neighbor[r] = true;
                                ++boundary_nnz;
                                checked[r][i + index_offset[rank]] = true;
                            }
                            ++ghost_nnz;
                            // Rank for current boundary point local_col[j] has been found
                            // Continue with next boundary point
                            break;
                        }
                    }
                }
            }
        }
    }

    for(int i = 0; i < num_procs; ++i)
    {
        if(neighbor[i] == true)
        {
            ++neighbors;
        }
    }

    std::vector<MPI_Request> mpi_req(neighbors * 2);
    int                      n = 0;
    // Array to hold boundary size for each interface
    std::vector<int> boundary_size(neighbors);

    // MPI receive boundary sizes
    for(int i = 0; i < num_procs; ++i)
    {
        // If neighbor receive from rank i is expected...
        if(neighbor[i] == true)
        {
            // Receive size of boundary from rank i to current rank
            my_irecv(&(boundary_size[n]), 1, i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }

    // MPI send boundary sizes
    for(int i = 0; i < num_procs; ++i)
    {
        // Send required if boundary for rank i available
        if(boundary[i].size() > 0)
        {
            int size = boundary[i].size();
            // Send size of boundary from current rank to rank i
            my_isend(&size, 1, i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }
    // Wait to finish communication
    MPI_Waitall(n - 1, &(mpi_req[0]), MPI_STATUSES_IGNORE);

    n = 0;
    // Array to hold boundary offset for each interface
    int              k = 0;
    std::vector<int> recv_offset(neighbors + 1);
    std::vector<int> send_offset(neighbors + 1);
    recv_offset[0] = 0;
    send_offset[0] = 0;
    for(int i = 0; i < neighbors; ++i)
    {
        recv_offset[i + 1] = recv_offset[i] + boundary_size[i];
    }

    for(int i = 0; i < num_procs; ++i)
    {
        if(neighbor[i] == true)
        {
            send_offset[k + 1] = send_offset[k] + boundary[i].size();
            ++k;
        }
    }

    // Array to hold boundary for each interface
    std::vector<std::vector<PtrType>> local_boundary(neighbors);
    for(int i = 0; i < neighbors; ++i)
    {
        local_boundary[i].resize(boundary_size[i]);
    }

    // MPI receive boundary
    for(int i = 0; i < num_procs; ++i)
    {
        // If neighbor receive from rank i is expected...
        if(neighbor[i] == true)
        {
            // Receive boundary from rank i to current rank
            my_irecv(local_boundary[n].data(), boundary_size[n], i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }

    // MPI send boundary
    for(int i = 0; i < num_procs; ++i)
    {
        // Send required if boundary for rank i is available
        if(boundary[i].size() > 0)
        {
            // Send boundary from current rank to rank i
            my_isend(&(boundary[i][0]), boundary[i].size(), i, 0, *comm, &mpi_req[n]);
            ++n;
        }
    }

    // Wait to finish communication
    MPI_Waitall(n - 1, &(mpi_req[0]), MPI_STATUSES_IGNORE);

    // Total boundary size
    int nnz_boundary = 0;
    for(int i = 0; i < neighbors; ++i)
    {
        nnz_boundary += boundary_size[i];
    }

    // Create local boundary index array
    k = 0;
    std::vector<int> bnd(boundary_nnz);

    for(int i = 0; i < num_procs; ++i)
    {
        for(unsigned int j = 0; j < boundary[i].size(); ++j)
        {
            bnd[k] = static_cast<int>(boundary[i][j] - index_offset[rank]);
            ++k;
        }
    }

    // Create boundary index array
    std::vector<PtrType> boundary_index(nnz_boundary);

    k = 0;
    for(int i = 0; i < neighbors; ++i)
    {
        for(int j = 0; j < boundary_size[i]; ++j)
        {
            boundary_index[k] = local_boundary[i][j];
            ++k;
        }
    }

    // Create map with boundary index relations
    std::map<int, int> boundary_map;

    for(int i = 0; i < nnz_boundary; ++i)
    {
        boundary_map[boundary_index[i]] = i;
    }

    // Build up ghost and interior matrix
    int*       ghost_row = new int[ghost_nnz];
    int*       ghost_col = new int[ghost_nnz];
    ValueType* ghost_val = new ValueType[ghost_nnz];

    memset(ghost_row, 0, sizeof(int) * ghost_nnz);
    memset(ghost_col, 0, sizeof(int) * ghost_nnz);
    memset(ghost_val, 0, sizeof(ValueType) * ghost_nnz);

    PtrType*   row_offset = new PtrType[local_nrow + 1];
    int*       col        = new int[interior_nnz];
    ValueType* val        = new ValueType[interior_nnz];

    memset(row_offset, 0, sizeof(PtrType) * (local_nrow + 1));
    memset(col, 0, sizeof(int) * interior_nnz);
    memset(val, 0, sizeof(ValueType) * interior_nnz);

    row_offset[0] = 0;
    k             = 0;
    int l         = 0;
    for(int i = 0; i < local_nrow; ++i)
    {
        for(PtrType j = local_row_offset[i]; j < local_row_offset[i + 1]; ++j)
        {

            // Boundary point -- create ghost part
            if(local_col[j] < index_offset[rank] || local_col[j] >= index_offset[rank + 1])
            {
                ghost_row[k] = i;
                ghost_col[k] = boundary_map[local_col[j]];
                ghost_val[k] = local_val[j];
                ++k;
            }
            else
            {
                // Interior point -- create interior part
                int c = local_col[j] - index_offset[rank];

                col[l] = c;
                val[l] = local_val[j];
                ++l;
            }
        }
        row_offset[i + 1] = l;
    }

    std::vector<int> recv(neighbors);
    std::vector<int> sender(neighbors);

    int nbc = 0;
    for(int i = 0; i < num_procs; ++i)
    {
        if(neighbor[i] == true)
        {
            recv[nbc]   = i;
            sender[nbc] = i;
            ++nbc;
        }
    }

    pm->SetMPICommunicator(comm);
    pm->SetGlobalNrow(global_nrow);
    pm->SetGlobalNcol(global_nrow);
    pm->SetLocalNrow(local_size[rank]);
    pm->SetLocalNcol(local_size[rank]);
    pm->SetBoundaryIndex(boundary_nnz, bnd.data());
    pm->SetReceivers(neighbors, recv.data(), recv_offset.data());
    pm->SetSenders(neighbors, sender.data(), send_offset.data());

    gmat->SetParallelManager(*pm);
    gmat->SetLocalDataPtrCSR(&row_offset, &col, &val, "mat", interior_nnz);
    gmat->SetGhostDataPtrCOO(&ghost_row, &ghost_col, &ghost_val, "ghost", ghost_nnz);
    gmat->Sort();
}

// This function computes all prime factors of a given number n
static void compute_prime_factors(int n, std::vector<int>& p)
{
    int factor = 2;

    // Factorize
    while(n > 1)
    {
        while(n % factor == 0)
        {
            p.push_back(factor);
            n /= factor;
        }

        ++factor;
    }
}

// This function computes the process distribution for each dimension
static void compute_2d_process_distribution(int nprocs, int& nprocx, int& nprocy)
{
    // Compute prime factors
    std::vector<int> p;
    compute_prime_factors(nprocs, p);

    // Compute number of processes in each dimension
    nprocx = 1;
    nprocy = 1;

    if(p.size() == 0)
    {
        // No entry, this means we have exactly one process
    }
    else if(p.size() == 1)
    {
        // If we have a single prime number, this is going to be our x dimension
        nprocx = p[0];
    }
    else if(p.size() == 2)
    {
        // For two prime numbers, setup x and y
        nprocx = p[1];
        nprocy = p[0];
    }
    else
    {
        // More than two prime numbers

        // #prime numbers
        int    idx    = 0;
        size_t nprime = p.size();

        // cubic root
        double sqroot = std::sqrt(nprocs);

        // Determine x dimension
        nprocx = p[nprime-- - 1];

        while(nprocx < sqroot && idx < nprime)
        {
            nprocx *= p[idx++];
        }

        // Determine y dimension
        while(idx < nprime)
        {
            nprocy *= p[idx++];
        }
    }

    // Number of processes must match
    assert(nprocx * nprocy == nprocs);
}

// This function computes the process distribution for each dimension
static void compute_3d_process_distribution(int nprocs, int& nprocx, int& nprocy, int& nprocz)
{
    // Compute prime factors
    std::vector<int> p;
    compute_prime_factors(nprocs, p);

    // Compute number of processes in each dimension
    nprocx = 1;
    nprocy = 1;
    nprocz = 1;

    if(p.size() == 0)
    {
        // No entry, this means we have exactly one process
    }
    else if(p.size() == 1)
    {
        // If we have a single prime number, this is going to be our x dimension
        nprocx = p[0];
    }
    else if(p.size() == 2)
    {
        // For two prime numbers, setup x and y
        nprocx = p[1];
        nprocy = p[0];
    }
    else if(p.size() == 3)
    {
        // Three prime numbers
        nprocx = p[2];
        nprocy = p[1];
        nprocz = p[0];
    }
    else
    {
        // More than three prime numbers

        // #prime numbers
        int    idx    = 0;
        size_t nprime = p.size();

        // cubic root
        double qroot = std::cbrt(nprocs);

        // Determine x dimension
        nprocx = p[nprime-- - 1];

        while(nprocx < qroot && idx < nprime)
        {
            nprocx *= p[idx++];
        }

        // Determine y dimension
        double sqroot = std::sqrt(nprocs / nprocx);

        nprocy = p[nprime-- - 1];

        while(nprocy < sqroot && idx < nprime)
        {
            nprocy *= p[idx++];
        }

        // Determine z dimension
        while(idx < nprime)
        {
            nprocz *= p[idx++];
        }
    }

    // Number of processes must match
    assert(nprocx * nprocy * nprocz == nprocs);
}

template <typename ValueType>
void generate_2d_laplacian(int                      local_dimx,
                           int                      local_dimy,
                           const MPI_Comm*          comm,
                           GlobalMatrix<ValueType>* mat,
                           ParallelManager*         pm,
                           int                      rank,
                           int                      nprocs,
                           int                      nsten = 9)
{
    assert(nsten == 5 || nsten == 9);

    // First, we need to determine process pattern for the unit square
    int nproc_x;
    int nproc_y;

    compute_2d_process_distribution(nprocs, nproc_x, nproc_y);

    // Next, determine process index into the unit square
    int iproc_y = rank / nproc_x;
    int iproc_x = rank % nproc_x;

    // Global sizes
    int64_t global_dimx = static_cast<int64_t>(nproc_x) * local_dimx;
    int64_t global_dimy = static_cast<int64_t>(nproc_y) * local_dimy;

    // Global process entry points
    int64_t global_iproc_x = iproc_x * local_dimx;
    int64_t global_iproc_y = iproc_y * local_dimy;

    // Number of rows (global and local)
    int64_t local_nrow  = local_dimx * local_dimy;
    int64_t global_nrow = global_dimx * global_dimy;

    // Assemble local CSR matrix row offset pointers
    PtrType* global_csr_row_ptr = NULL;
    int64_t* global_csr_col_ind = NULL;
    int64_t* local2global       = NULL;

    allocate_host(local_nrow + 1, &global_csr_row_ptr);
    allocate_host(local_nrow * nsten, &global_csr_col_ind);
    allocate_host(local_nrow, &local2global);

    std::map<int64_t, int> global2local;

    PtrType nnz           = 0;
    global_csr_row_ptr[0] = 0;

    // Loop over y dimension
    for(int local_y = 0; local_y < local_dimy; ++local_y)
    {
        // Global index into y
        int64_t global_y = global_iproc_y + local_y;

        // Loop over x dimension
        for(int local_x = 0; local_x < local_dimx; ++local_x)
        {
            // Global index into x
            int64_t global_x = global_iproc_x + local_x;

            // Local row
            int local_row = local_y * local_dimx + local_x;

            // Global row
            int64_t global_row = global_y * global_dimx + global_x;

            // Fill l2g and g2l map
            local2global[local_row]  = global_row;
            global2local[global_row] = local_row;

            // 5pt stencil
            if(nsten == 5)
            {
                // Fixed x (leaving out i == j)
                for(int by = -1; by <= 1; ++by)
                {
                    if(global_y + by > -1 && global_y + by < global_dimy && by != 0)
                    {
                        // Global column
                        int64_t global_col = global_row + by * global_dimx;

                        // Fill global CSR column indices
                        global_csr_col_ind[nnz++] = global_col;
                    }
                }

                // Fixed y
                for(int bx = -1; bx <= 1; ++bx)
                {
                    if(global_x + bx > -1 && global_x + bx < global_dimx)
                    {
                        // Global column
                        int64_t global_col = global_row + bx;

                        // Fill global CSR column indices
                        global_csr_col_ind[nnz++] = global_col;
                    }
                }
            }

            // 9 pt stencil
            if(nsten == 9)
            {
                // Check if current y vertex is on the boundary
                for(int by = -1; by <= 1; ++by)
                {
                    if(global_y + by > -1 && global_y + by < global_dimy)
                    {
                        // Check if current x vertex is on the boundary
                        for(int bx = -1; bx <= 1; ++bx)
                        {
                            if(global_x + bx > -1 && global_x + bx < global_dimx)
                            {
                                // Global column
                                int64_t global_col = global_row + by * global_dimx + bx;

                                // Fill global CSR column indices
                                global_csr_col_ind[nnz++] = global_col;
                            }
                        }
                    }
                }
            }

            global_csr_row_ptr[local_row + 1] = nnz;
        }
    }

    // Local number of non-zero entries - need to use long long int to make the communication work
    int64_t local_nnz = global_csr_row_ptr[local_nrow];

    // Total number of non-zeros
    int64_t global_nnz;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, *comm);

    // Now, we need to setup the communication pattern
    std::map<int, std::set<int64_t>> recv_indices;
    std::map<int, std::set<int64_t>> send_indices;

    // CSR matrix row pointers
    PtrType* int_csr_row_ptr = NULL;
    PtrType* gst_csr_row_ptr = NULL;

    allocate_host(local_nrow + 1, &int_csr_row_ptr);
    allocate_host(local_nrow + 1, &gst_csr_row_ptr);

    int_csr_row_ptr[0] = 0;
    gst_csr_row_ptr[0] = 0;

    // Determine, which vertices need to be sent / received
    for(int i = 0; i < local_nrow; ++i)
    {
        int_csr_row_ptr[i + 1] = int_csr_row_ptr[i];
        gst_csr_row_ptr[i + 1] = gst_csr_row_ptr[i];

        int64_t global_row = local2global[i];

        for(PtrType j = global_csr_row_ptr[i]; j < global_csr_row_ptr[i + 1]; ++j)
        {
            int64_t global_col = global_csr_col_ind[j];

            // Determine which process owns the vertex
            int64_t idx_y = global_col / global_dimx;
            int64_t idx_x = global_col % global_dimx;

            int idx_proc_y = idx_y / local_dimy;
            int idx_proc_x = idx_x / local_dimx;

            int owner = idx_proc_x + idx_proc_y * nproc_x;

            // If we do not own it, we need to receive it from our neighbor
            // and also send the current vertex to this neighbor
            if(owner != rank)
            {
                // Store the global column and row id that we have to receive / send from / to a neighbor
                // We need a set here to eliminate duplicates
                recv_indices[owner].insert(global_col);
                send_indices[owner].insert(global_row);

                ++gst_csr_row_ptr[i + 1];
            }
            else
            {
                ++int_csr_row_ptr[i + 1];
            }
        }
    }

    // Number of processes we communicate with
    int nrecv = recv_indices.size();
    int nsend = send_indices.size();

    // Process ids we communicate with
    std::vector<int> recvs;
    std::vector<int> sends;

    recvs.reserve(nrecv);
    sends.reserve(nsend);

    // Index offsets for each neighbor
    std::vector<int> recv_index_offset;
    std::vector<int> send_index_offset;

    recv_index_offset.reserve(nrecv + 1);
    send_index_offset.reserve(nsend + 1);

    recv_index_offset.push_back(0);
    send_index_offset.push_back(0);

    int                    cnt = 0;
    std::map<int64_t, int> global2ghost;

    // Go through the recv data
    for(std::map<int, std::set<int64_t>>::iterator it = recv_indices.begin();
        it != recv_indices.end();
        ++it)
    {
        recvs.push_back(it->first);
        recv_index_offset.push_back(it->second.size());

        for(std::set<int64_t>::iterator iit = it->second.begin(); iit != it->second.end(); ++iit)
        {
            global2ghost[*iit] = cnt++;
        }
    }

    // Go through the send data
    int boundary_nnz = 0;
    for(std::map<int, std::set<int64_t>>::iterator it = send_indices.begin();
        it != send_indices.end();
        ++it)
    {
        sends.push_back(it->first);
        send_index_offset.push_back(it->second.size());
        boundary_nnz += it->second.size();
    }

    // Exclusive sum
    for(int i = 0; i < nrecv; ++i)
    {
        recv_index_offset[i + 1] += recv_index_offset[i];
    }

    for(int i = 0; i < nsend; ++i)
    {
        send_index_offset[i + 1] += send_index_offset[i];
    }

    // Boundary indices
    std::vector<int> boundary;
    boundary.reserve(boundary_nnz);

    for(std::map<int, std::set<int64_t>>::iterator it = send_indices.begin();
        it != send_indices.end();
        ++it)
    {
        for(std::set<int64_t>::iterator iit = it->second.begin(); iit != it->second.end(); ++iit)
        {
            boundary.push_back(global2local[*iit]);
        }
    }

    // Initialize manager
    pm->SetMPICommunicator(comm);
    pm->SetGlobalNrow(global_nrow);
    pm->SetGlobalNcol(global_nrow);
    pm->SetLocalNrow(local_nrow);
    pm->SetLocalNcol(local_nrow);

    if(nprocs > 1)
    {
        pm->SetBoundaryIndex(boundary_nnz, boundary.data());
        pm->SetReceivers(nrecv, recvs.data(), recv_index_offset.data());
        pm->SetSenders(nsend, sends.data(), send_index_offset.data());
    }

    mat->SetParallelManager(*pm);

    // Generate local and ghost matrices
    local_nnz         = int_csr_row_ptr[local_nrow];
    int64_t ghost_nnz = gst_csr_row_ptr[local_nrow];

    int*       int_csr_col_ind = NULL;
    int*       gst_csr_col_ind = NULL;
    ValueType* int_csr_val     = NULL;
    ValueType* gst_csr_val     = NULL;

    allocate_host(local_nnz, &int_csr_col_ind);
    allocate_host(local_nnz, &int_csr_val);
    allocate_host(ghost_nnz, &gst_csr_col_ind);
    allocate_host(ghost_nnz, &gst_csr_val);

    // Convert global matrix columns to local columns
    for(int i = 0; i < local_nrow; ++i)
    {
        PtrType local_idx = int_csr_row_ptr[i];
        PtrType ghost_idx = gst_csr_row_ptr[i];

        int64_t global_row = local2global[i];

        for(PtrType j = global_csr_row_ptr[i]; j < global_csr_row_ptr[i + 1]; ++j)
        {
            int64_t global_col = global_csr_col_ind[j];

            // Determine which process owns the vertex
            int64_t idx_y = global_col / global_dimx;
            int64_t idx_x = global_col % global_dimx;

            int idx_proc_y = idx_y / local_dimy;
            int idx_proc_x = idx_x / local_dimx;

            int owner = idx_proc_x + idx_proc_y * nproc_x;

            // If we do not own it, we need to receive it from our neighbor
            // and also send the current vertex to this neighbor
            if(owner != rank)
            {
                // Store the global column and row id that we have to receive / send from / to a neighbor
                // We need a set here to eliminate duplicates
                recv_indices[owner].insert(global_col);
                send_indices[owner].insert(global_row);

                gst_csr_col_ind[ghost_idx] = global2ghost[global_col];
                gst_csr_val[ghost_idx]     = -1.0;
                ++ghost_idx;
            }
            else
            {
                // This is our part
                int_csr_col_ind[local_idx] = global2local[global_col];
                int_csr_val[local_idx]     = (global_col == global_row) ? (nsten - 1.0) : -1.0;
                ++local_idx;
            }
        }
    }

    free_host(&global_csr_row_ptr);
    free_host(&global_csr_col_ind);
    free_host(&local2global);

    mat->SetLocalDataPtrCSR(&int_csr_row_ptr, &int_csr_col_ind, &int_csr_val, "mat", local_nnz);
    mat->SetGhostDataPtrCSR(&gst_csr_row_ptr, &gst_csr_col_ind, &gst_csr_val, "gst", ghost_nnz);
    mat->Sort();
}

template <typename ValueType>
void generate_3d_laplacian(int                      local_dimx,
                           int                      local_dimy,
                           int                      local_dimz,
                           const MPI_Comm*          comm,
                           GlobalMatrix<ValueType>* mat,
                           ParallelManager*         pm,
                           int                      rank,
                           int                      nprocs)
{
    // First, we need to determine process pattern for the unit cube
    int nproc_x;
    int nproc_y;
    int nproc_z;

    compute_3d_process_distribution(nprocs, nproc_x, nproc_y, nproc_z);

    // Next, determine process index into the unit cube
    int iproc_z = rank / (nproc_x * nproc_y);
    int iproc_y = (rank - iproc_z * nproc_x * nproc_y) / nproc_x;
    int iproc_x = rank % nproc_x;

    // Global sizes
    int64_t global_dimx = static_cast<int64_t>(nproc_x) * local_dimx;
    int64_t global_dimy = static_cast<int64_t>(nproc_y) * local_dimy;
    int64_t global_dimz = static_cast<int64_t>(nproc_z) * local_dimz;

    // Global process entry points
    int64_t global_iproc_x = iproc_x * local_dimx;
    int64_t global_iproc_y = iproc_y * local_dimy;
    int64_t global_iproc_z = iproc_z * local_dimz;

    // Number of rows (global and local)
    int64_t local_nrow  = local_dimx * local_dimy * local_dimz;
    int64_t global_nrow = global_dimx * global_dimy * global_dimz;

    // Assemble local CSR matrix row offset pointers
    std::vector<PtrType> global_csr_row_ptr(local_nrow + 1);
    std::vector<int64_t> global_csr_col_ind(local_nrow * 27);

    std::vector<int64_t>   local2global(local_nrow);
    std::map<int64_t, int> global2local;

    PtrType nnz           = 0;
    global_csr_row_ptr[0] = 0;

    // Loop over z dimension
    for(int local_z = 0; local_z < local_dimz; ++local_z)
    {
        // Global index into z
        int64_t global_z = global_iproc_z + local_z;

        // Loop over y dimension
        for(int local_y = 0; local_y < local_dimy; ++local_y)
        {
            // Global index into y
            int64_t global_y = global_iproc_y + local_y;

            // Loop over x dimension
            for(int local_x = 0; local_x < local_dimx; ++local_x)
            {
                // Global index into x
                int64_t global_x = global_iproc_x + local_x;

                // Local row
                int local_row = local_z * local_dimx * local_dimy + local_y * local_dimx + local_x;

                // Global row
                int64_t global_row
                    = global_z * global_dimx * global_dimy + global_y * global_dimx + global_x;

                // Fill l2g and g2l map
                local2global[local_row]  = global_row;
                global2local[global_row] = local_row;

                // Check if current z vertex is on the boundary
                for(int bz = -1; bz <= 1; ++bz)
                {
                    if(global_z + bz > -1 && global_z + bz < global_dimz)
                    {
                        // Check if current y vertex is on the boundary
                        for(int by = -1; by <= 1; ++by)
                        {
                            if(global_y + by > -1 && global_y + by < global_dimy)
                            {
                                // Check if current x vertex is on the boundary
                                for(int bx = -1; bx <= 1; ++bx)
                                {
                                    if(global_x + bx > -1 && global_x + bx < global_dimx)
                                    {
                                        // Global column
                                        int64_t global_col = global_row
                                                             + bz * global_dimx * global_dimy
                                                             + by * global_dimx + bx;

                                        // Fill global CSR column indices
                                        global_csr_col_ind[nnz++] = global_col;
                                    }
                                }
                            }
                        }
                    }
                }

                global_csr_row_ptr[local_row + 1] = nnz;
            }
        }
    }

    // Local number of non-zero entries - need to use long long int to make the communication work
    int64_t local_nnz = global_csr_row_ptr[local_nrow];

    // Total number of non-zeros
    int64_t global_nnz;
    MPI_Allreduce(&local_nnz, &global_nnz, 1, MPI_INT64_T, MPI_SUM, *comm);

    // Now, we need to setup the communication pattern
    std::map<int, std::set<int64_t>> recv_indices;
    std::map<int, std::set<int64_t>> send_indices;

    // CSR matrix row pointers
    PtrType* int_csr_row_ptr = NULL;
    PtrType* gst_csr_row_ptr = NULL;

    allocate_host(local_nrow + 1, &int_csr_row_ptr);
    allocate_host(local_nrow + 1, &gst_csr_row_ptr);

    int_csr_row_ptr[0] = 0;
    gst_csr_row_ptr[0] = 0;

    // Determine, which vertices need to be sent / received
    for(int i = 0; i < local_nrow; ++i)
    {
        int_csr_row_ptr[i + 1] = int_csr_row_ptr[i];
        gst_csr_row_ptr[i + 1] = gst_csr_row_ptr[i];

        int64_t global_row = local2global[i];

        for(PtrType j = global_csr_row_ptr[i]; j < global_csr_row_ptr[i + 1]; ++j)
        {
            int64_t global_col = global_csr_col_ind[j];

            // Determine which process owns the vertex
            int64_t idx_z = global_col / (global_dimx * global_dimy);
            int64_t idx_y = (global_col - idx_z * global_dimy * global_dimx) / global_dimx;
            int64_t idx_x = global_col % global_dimx;

            int idx_proc_z = idx_z / local_dimz;
            int idx_proc_y = idx_y / local_dimy;
            int idx_proc_x = idx_x / local_dimx;

            int owner = idx_proc_x + idx_proc_y * nproc_x + idx_proc_z * nproc_y * nproc_x;

            // If we do not own it, we need to receive it from our neighbor
            // and also send the current vertex to this neighbor
            if(owner != rank)
            {
                // Store the global column and row id that we have to receive / send from / to a neighbor
                // We need a set here to eliminate duplicates
                recv_indices[owner].insert(global_col);
                send_indices[owner].insert(global_row);

                ++gst_csr_row_ptr[i + 1];
            }
            else
            {
                ++int_csr_row_ptr[i + 1];
            }
        }
    }

    // Number of processes we communicate with
    int nrecv = recv_indices.size();
    int nsend = send_indices.size();

    // Process ids we communicate with
    std::vector<int> recvs;
    std::vector<int> sends;

    recvs.reserve(nrecv);
    sends.reserve(nsend);

    // Index offsets for each neighbor
    std::vector<int> recv_index_offset;
    std::vector<int> send_index_offset;

    recv_index_offset.reserve(nrecv + 1);
    send_index_offset.reserve(nsend + 1);

    recv_index_offset.push_back(0);
    send_index_offset.push_back(0);

    int                    cnt = 0;
    std::map<int64_t, int> global2ghost;

    // Go through the recv data
    for(std::map<int, std::set<int64_t>>::iterator it = recv_indices.begin();
        it != recv_indices.end();
        ++it)
    {
        recvs.push_back(it->first);
        recv_index_offset.push_back(it->second.size());

        for(std::set<int64_t>::iterator iit = it->second.begin(); iit != it->second.end(); ++iit)
        {
            global2ghost[*iit] = cnt++;
        }
    }

    // Go through the send data
    int boundary_nnz = 0;
    for(std::map<int, std::set<int64_t>>::iterator it = send_indices.begin();
        it != send_indices.end();
        ++it)
    {
        sends.push_back(it->first);
        send_index_offset.push_back(it->second.size());
        boundary_nnz += it->second.size();
    }

    // Exclusive sum
    for(int i = 0; i < nrecv; ++i)
    {
        recv_index_offset[i + 1] += recv_index_offset[i];
    }

    for(int i = 0; i < nsend; ++i)
    {
        send_index_offset[i + 1] += send_index_offset[i];
    }

    // Boundary indices
    std::vector<int> boundary;
    boundary.reserve(boundary_nnz);

    for(std::map<int, std::set<int64_t>>::iterator it = send_indices.begin();
        it != send_indices.end();
        ++it)
    {
        for(std::set<int64_t>::iterator iit = it->second.begin(); iit != it->second.end(); ++iit)
        {
            boundary.push_back(global2local[*iit]);
        }
    }

    // Initialize manager
    pm->SetMPICommunicator(comm);
    pm->SetGlobalNrow(global_nrow);
    pm->SetGlobalNcol(global_nrow);
    pm->SetLocalNrow(local_nrow);
    pm->SetLocalNcol(local_nrow);

    if(nprocs > 1)
    {
        pm->SetBoundaryIndex(boundary_nnz, boundary.data());
        pm->SetReceivers(nrecv, recvs.data(), recv_index_offset.data());
        pm->SetSenders(nsend, sends.data(), send_index_offset.data());
    }

    mat->SetParallelManager(*pm);

    // Generate local and ghost matrices
    local_nnz         = int_csr_row_ptr[local_nrow];
    int64_t ghost_nnz = gst_csr_row_ptr[local_nrow];

    int*       int_csr_col_ind = NULL;
    int*       gst_csr_col_ind = NULL;
    ValueType* int_csr_val     = NULL;
    ValueType* gst_csr_val     = NULL;

    allocate_host(local_nnz, &int_csr_col_ind);
    allocate_host(local_nnz, &int_csr_val);
    allocate_host(ghost_nnz, &gst_csr_col_ind);
    allocate_host(ghost_nnz, &gst_csr_val);

    // Convert global matrix columns to local columns
    for(int i = 0; i < local_nrow; ++i)
    {
        PtrType local_idx = int_csr_row_ptr[i];
        PtrType ghost_idx = gst_csr_row_ptr[i];

        int64_t global_row = local2global[i];

        for(PtrType j = global_csr_row_ptr[i]; j < global_csr_row_ptr[i + 1]; ++j)
        {
            int64_t global_col = global_csr_col_ind[j];

            // Determine which process owns the vertex
            int64_t idx_z = global_col / (global_dimx * global_dimy);
            int64_t idx_y = (global_col - idx_z * global_dimy * global_dimx) / global_dimx;
            int64_t idx_x = global_col % global_dimx;

            int idx_proc_z = idx_z / local_dimz;
            int idx_proc_y = idx_y / local_dimy;
            int idx_proc_x = idx_x / local_dimx;

            int owner = idx_proc_x + idx_proc_y * nproc_x + idx_proc_z * nproc_y * nproc_x;

            // If we do not own it, we need to receive it from our neighbor
            // and also send the current vertex to this neighbor
            if(owner != rank)
            {
                // Store the global column and row id that we have to receive / send from / to a neighbor
                // We need a set here to eliminate duplicates
                recv_indices[owner].insert(global_col);
                send_indices[owner].insert(global_row);

                gst_csr_col_ind[ghost_idx] = global2ghost[global_col];
                gst_csr_val[ghost_idx]     = -1.0;
                ++ghost_idx;
            }
            else
            {
                // This is our part
                int_csr_col_ind[local_idx] = global2local[global_col];
                int_csr_val[local_idx]     = (global_col == global_row) ? 26.0 : -1.0;
                ++local_idx;
            }
        }
    }

    mat->SetLocalDataPtrCSR(&int_csr_row_ptr, &int_csr_col_ind, &int_csr_val, "mat", local_nnz);
    mat->SetGhostDataPtrCSR(&gst_csr_row_ptr, &gst_csr_col_ind, &gst_csr_val, "gst", ghost_nnz);
    mat->Sort();
}
