__kernel void min_loc ( __global const unsigned char* input,
                    __global unsigned char *partial_min,
                    __global unsigned char *partial_min_location)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    local unsigned char localMins[256];
    local unsigned char localMinsLocation[256];

    localMins[local_id] = input[get_global_id(0)];
    localMinsLocation[local_id] = get_global_id(0);

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(localMins[local_id] > localMins[local_id + stride])
            {
                localMins[local_id] = localMins[local_id + stride];
                localMinsLocation[local_id] = get_global_id(0) + stride;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        partial_min[get_group_id(0)] = localMins[0];
        partial_min_location[get_group_id(0)] = localMinsLocation[0];
    }
}                        
__kernel void max_loc ( __global const unsigned char* input,
                    __global unsigned char *partial_max,
                    __global unsigned char *partial_max_location)
{
    uint local_id = get_local_id(0);
    uint group_size = get_local_size(0);
    local unsigned char localmaxs[256];
    local unsigned char localmaxsLocation[256];

    localmaxs[local_id] = input[get_global_id(0)];
    localmaxsLocation[local_id] = get_global_id(0);

    for (uint stride = group_size/2; stride>0; stride /=2)
    {
        if (local_id < stride)
        {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(localmaxs[local_id] > localmaxs[local_id + stride])
            {
                localmaxs[local_id] = localmaxs[local_id + stride];
                localmaxsLocation[local_id] = get_global_id(0) + stride;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0)
    {
        partial_max[get_group_id(0)] = localmaxs[0];
        partial_max_location[get_group_id(0)] = localmaxsLocation[0];
    }
}    