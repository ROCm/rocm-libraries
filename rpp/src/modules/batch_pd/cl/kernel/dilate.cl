#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#define saturate_8u(value) ( (value) > 255 ? 255 : ((value) < 0 ? 0 : (value) ))
#define MAX(value1, value2) ((value1) > (value2) ? value1 : value2)
#define WIDTH 18 // 16 + 2 * Padding
#define HALF_FILTER_SIZE 1
#define TWICE_HALF_FILTER_SIZE 2
#define OPTI_WIDTH 258


__kernel void dilate_pkd(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;

    unsigned int index;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                index = pixIdx + (j * channel) + (i * width * channel);
                if(input[index] > pixel)
                {
                    pixel = input[index];
                }
            }
        }
    }
    output[pixIdx] = input[pixIdx];
}


__kernel void dilate_unrolled(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{

    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;                           
    unsigned int index;
    int max_id;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    unsigned char max = input[pixIdx];
    
    if(id_x == 0 || id_x == width-1 || id_y == 0 || id_y == height-1)
    {
        max = input[pixIdx];
    }
    else
    {
            max_id = pixIdx - channel - width*channel;
            max = MAX(max, input[max_id]);
            max = MAX(max, input[max_id + channel]);
            max = MAX(max, input[max_id + 2*channel]);
            max_id += width*channel;
            max = MAX(max, input[max_id]);
            max = MAX(max, input[max_id + channel]);
            max = MAX(max, input[max_id + 2*channel]);
            max_id += width*channel;
            max = MAX(max, input[max_id]);
            max = MAX(max, input[max_id + channel]);
            max = MAX(max, input[max_id + 2*channel]);       
    }
    output[pixIdx] = input[pixIdx];
}

__kernel void dilate_unrolled_pln(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{

    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    unsigned int index;
    int max_id;
    //__local unsigned char localImage[WIDTH * WIDTH];
    int pixIdx, i;
    //id_y = id_y * 8;
    unsigned char max;    
    if(id_x == 0 || id_x == width-1 || id_y == 0 || id_y >= height-1)
    {
        pixIdx = id_y * width + id_x + id_z * width * height;
        //max = 255;//input[pixIdx];
        output[pixIdx] = input[pixIdx];
    }
    else
    {
        pixIdx = id_y * width + id_x + id_z * width * height;
        //for(i = 0; i < 8 && id_y < height -1 ; i++)
        {
            pixIdx += width;
            max = input[pixIdx];    
            max_id = pixIdx - 1 - width;
            max = MAX(max, input[max_id]);
            max = MAX(max, input[max_id + 1]);
            max = MAX(max, input[max_id + 2]);
            max_id += width;
            max = MAX(max, input[max_id]);
            max = MAX(max, input[max_id + 1]);
            max = MAX(max, input[max_id + 2]);
            max_id += width;
            max = MAX(max, input[max_id]);
            max = MAX(max, input[max_id + 1]);
            max = MAX(max, input[max_id + 2]); 
            output[pixIdx] = max;

            // pixIdx += id_y * width;
            // max = input[pixIdx];    
            // max_id = pixIdx - 1 - width;
            // max = MAX(max, input[max_id]);
            // max = MAX(max, input[max_id + 1]);
            // max = MAX(max, input[max_id + 2]);
            // max_id += width;
            // max = MAX(max, input[max_id]);
            // max = MAX(max, input[max_id + 1]);
            // max = MAX(max, input[max_id + 2]);
            // max_id += width;
            // max = MAX(max, input[max_id]);
            // max = MAX(max, input[max_id + 1]);
            // max = MAX(max, input[max_id + 2]); 
            // output[pixIdx] = max;
            // id_y++;
        }   
    }
}

__kernel void dilate_local(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int IMAGE_H,
                                const unsigned int IMAGE_W,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{
	const int rowOffset = get_global_id(1) * IMAGE_W; //Done
	const int my = get_global_id(0) + rowOffset; //Done
	
	const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0);//Done 34
	const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen; // 34 *(1 + local_rows)
	const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;	// Done!!!	
	__local int cached[18*18 ];
    const int HALF_FILTER_SIZE_IMAGE_W = IMAGE_W * HALF_FILTER_SIZE; 
	// copy my pixel
	cached[ myLocal ] = input[ my ]; // Dne
    //printf("%d horizontal", get_local_id(0));
	
	if (
		get_global_id(0) < HALF_FILTER_SIZE 			|| 
		get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1		|| 
		get_global_id(1) < HALF_FILTER_SIZE			||
		get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
	)
	{
		// no computation for me, sync and exit
        output[my] = input[my];
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}
	//Done Till here
	
	else 
	{
		// copy additional elements
		int localColOffset = -1;
		int globalColOffset = -1;
		
		if ( get_local_id(0) < HALF_FILTER_SIZE )
		{
			localColOffset = get_local_id(0);
			globalColOffset = -HALF_FILTER_SIZE;
			
			cached[ localRowOffset + get_local_id(0) ] = input[ my - HALF_FILTER_SIZE ];
		}
		else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )
		{
			localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;
			globalColOffset = HALF_FILTER_SIZE;
			
			cached[ myLocal + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE ];
		}
		
		
		if ( get_local_id(1) < HALF_FILTER_SIZE )
		{
			cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my - HALF_FILTER_SIZE_IMAGE_W ];
			if (localColOffset > 0)
			{
				cached[ get_local_id(1) * localRowLen + localColOffset ] = input[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
			}
		}
		else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )
		{
			int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;
			cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE_IMAGE_W ];
			if (localColOffset > 0)
			{
				cached[ offset + localColOffset ] = input[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
			}
		}
		
		// synchronize
		barrier(CLK_LOCAL_MEM_FENCE);

        int max = cached[myLocal];
        int max_id;

        max_id = myLocal - 1 - 18;
        max = MAX(max, cached[max_id]);
        max = MAX(max, cached[max_id + 1]);
        max = MAX(max, cached[max_id + 2]);
        max_id += 18; //WIDTH+2
        max = MAX(max, cached[max_id]);
        max = MAX(max, cached[max_id + 1]);
        max = MAX(max, cached[max_id + 2]);
        max_id += 18;//WIDTH+2
        max = MAX(max, cached[max_id]);
        max = MAX(max, cached[max_id + 1]);
        max = MAX(max, cached[max_id + 2]);  
        output[my] = max;  
		
		/*// perform convolution
		int fIndex = 0;
		float4 sum = (float4) 0.0;
		
		for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
		{
			int curRow = r * localRowLen;
			for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)
			{	
				sum += cached[ myLocal + curRow + c ] * filter[ fIndex ]; 
			}
		}
		output[my] = sum;*/
	}
}

__kernel void dilate_local_less(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int IMAGE_H,
                                const unsigned int IMAGE_W,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{
	const int rowOffset = get_global_id(1) * IMAGE_W; //Done
	const int my = get_global_id(0) + rowOffset; //Done
	
	const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0);//Done 34
	const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen; // 34 *(1 + local_rows)
	const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;	// Done!!!	
	__local int cached[18*18];
    const int HALF_FILTER_SIZE_IMAGE_W = IMAGE_W * HALF_FILTER_SIZE; 
	// copy my pixel
	cached[ myLocal ] = input[ my ]; // Dne
    //printf("%d horizontal", get_local_id(0));
	
	if (
		get_global_id(0) < HALF_FILTER_SIZE 			|| 
		get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1		|| 
		get_global_id(1) < HALF_FILTER_SIZE			||
		get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
	)
	{
		// no computation for me, sync and exit
        output[my] = input[my];
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}
	//Done Till here
	
	else 
	{
		// copy additional elements
		int localColOffset = -1;
		int globalColOffset = -1;
		
		if ( get_local_id(0) < HALF_FILTER_SIZE )
		{
			localColOffset = get_local_id(0);
			globalColOffset = -HALF_FILTER_SIZE;
			
			cached[ localRowOffset + get_local_id(0) ] = input[ my - HALF_FILTER_SIZE ];
		}
		else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )
		{
			localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;
			globalColOffset = HALF_FILTER_SIZE;
			
			cached[ myLocal + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE ];
		}
		
		
		if ( get_local_id(1) < HALF_FILTER_SIZE )
		{
			cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my - HALF_FILTER_SIZE_IMAGE_W ];
			if (localColOffset > 0)
			{
				cached[ get_local_id(1) * localRowLen + localColOffset ] = input[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
			}
		}
		else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )
		{
			int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;
			cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE_IMAGE_W ];
			if (localColOffset > 0)
			{
				cached[ offset + localColOffset ] = input[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
			}
		}
		
		// synchronize
		barrier(CLK_LOCAL_MEM_FENCE);

        int max = cached[myLocal];
        int max_id;

        max_id = myLocal - 1 - 18;
        max = MAX(max, cached[max_id]);
        max = MAX(max, cached[max_id + 1]);
        max = MAX(max, cached[max_id + 2]);
        max_id += 18; //WIDTH+2
        max = MAX(max, cached[max_id]);
        max = MAX(max, cached[max_id + 1]);
        max = MAX(max, cached[max_id + 2]);
        max_id += 18;//WIDTH+2
        max = MAX(max, cached[max_id]);
        max = MAX(max, cached[max_id + 1]);
        max = MAX(max, cached[max_id + 2]);  
        output[my] = max;  
		
		/*// perform convolution
		int fIndex = 0;
		float4 sum = (float4) 0.0;
		
		for (int r = -HALF_FILTER_SIZE; r <= HALF_FILTER_SIZE; r++)
		{
			int curRow = r * localRowLen;
			for (int c = -HALF_FILTER_SIZE; c <= HALF_FILTER_SIZE; c++, fIndex++)
			{	
				sum += cached[ myLocal + curRow + c ] * filter[ fIndex ]; 
			}
		}
		output[my] = sum;*/
	}
}


__kernel void dilate_pln(  __global unsigned char* input,
                    __global unsigned char* output,
                    const unsigned int height,
                    const unsigned int width,
                    const unsigned int channel,
                    const unsigned int kernelSize
)
{
    int id_x = get_global_id(0);
    int id_y = get_global_id(1);
    int id_z = get_global_id(2);
    if (id_x >= width || id_y >= height || id_z >= channel) return;
    //printf("coming here");
    
    int pixIdx = id_y * width + id_x + id_z * width * height;
    int bound = (kernelSize - 1) / 2;
    unsigned char pixel = input[pixIdx];
    for(int i = -bound ; i <= bound ; i++)
    {
        for(int j = -bound ; j <= bound ; j++)
        {
            if(id_x + j >= 0 && id_x + j <= width - 1 && id_y + i >= 0 && id_y + i <= height -1)
            {
                unsigned int index = pixIdx + j + (i * width);
                if(input[index] > pixel)
                {
                    pixel = input[index];
                }
            }
        }
    }
    output[pixIdx] = pixel; 

}

__kernel void dilate_batch(  __global unsigned char* input,
                                    __global unsigned char* output,
                                    __global unsigned int *kernelSize,
                                    __global int *xroi_begin,
                                    __global int *xroi_end,
                                    __global int *yroi_begin,
                                    __global int *yroi_end,
                                    __global unsigned int *height,
                                    __global unsigned int *width,
                                    __global unsigned int *max_width,
                                    __global unsigned long *batch_index,
                                    const unsigned int channel,
                                    __global unsigned int *inc, // use width * height for pln and 1 for pkd
                                    const int plnpkdindex // use 1 pln 3 for pkd
                                    )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    int kernelSizeTemp = kernelSize[id_z];
    int indextmp=0;
    long pixIdx = 0;
    int temp;
    // printf("%d", id_x);
    int value = 0;
    int value1 =0;
    unsigned char r = 0, g = 0, b = 0;
    int checkR = 0, checkB = 0, checkG = 0;
    if(id_x < width[id_z] && id_y < height[id_z])
    {    
        pixIdx = batch_index[id_z] + (id_x  + id_y * max_width[id_z] ) * plnpkdindex ;
        r = input[pixIdx];
        if(channel == 3)
        {
            g = input[pixIdx + inc[id_z]];
            b = input[pixIdx + inc[id_z] * 2];
        } 
        int bound = (kernelSizeTemp - 1) / 2;
        if((id_y >= yroi_begin[id_z] ) && (id_y <= yroi_end[id_z]) && (id_x >= xroi_begin[id_z]) && (id_x <= xroi_end[id_z]))
        {   
            for(int i = -bound ; i <= bound ; i++)
            {
                for(int j = -bound ; j <= bound ; j++)
                {
                    if(id_x + j >= 0 && id_x + j <= width[id_z] - 1 && id_y + i >= 0 && id_y + i <= height[id_z] -1)
                    {
                        unsigned int index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex;
                        if(r < input[index])
                            r = input[index];
                        if(channel == 3)
                        {
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z];
                            if(g < input[index])
                                g = input[index];
                            index = pixIdx + (j + (i * max_width[id_z])) * plnpkdindex + inc[id_z] * 2;
                            if(b < input[index])
                                b = input[index];
                        }
                    }
                }
            }
            output[pixIdx] = r;
            if(channel == 3)
            {
                output[pixIdx + inc[id_z]] = g;
                output[pixIdx + inc[id_z] * 2] = b;
            }
        }
        else if((id_x < width[id_z] ) && (id_y < height[id_z]))
        {
            for(indextmp = 0; indextmp < channel; indextmp++)
            {
                output[pixIdx] = input[pixIdx];
                pixIdx += inc[id_z];
            }
        }
    }
}

// Can do a bit of loop unrolling and local memory optimization if required
__kernel void dilate_optimized(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{

    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    bool valid = true;    
    id_x= id_x*16;
    if (id_x >= width || id_y >= height || id_z >= channel) return; // Return with nothing for out of bounds                     
    
    unsigned int index;
    unsigned char s[4]; // For keeping track of local pixels
    unsigned char sum[16]; // For bookkeeping the 16 maxima
    __global unsigned char *ptr, *out_ptr;
    unsigned char *head;


    int max_id;
    int pixIdx = id_y * channel * width + id_x * channel + id_z;
    unsigned char max = input[pixIdx];
    if(id_x == 0 || id_x == width-1 || id_y == 0 || id_y == height-1) // On the boundary of the Image, Keep output as input it self
    {
        output[pixIdx] = input[pixIdx];
    }
    else
    {
        int constraint = 14, diff, count = 0;
        if (id_x + 17 >= width)
        {
            diff = id_x + 17 + 1 - width;
            constraint -= diff;
        }   
        max_id = pixIdx - channel - width*channel;

        // All the three rows available according to thread launch
        // First Row
        ptr = input + max_id;
        s[0] = *ptr;    ptr++; // These Three pointers are always avaliable according to launch
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        head = sum;
        *head = amd_max3(s[0], s[1], s[2]);  head++; // Storing the partial max here //
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                // If should be on the constraint
                if(count <= constraint){
                    s[j] = *ptr;    ptr++;
                    *head = amd_max3(s[0], s[1], s[2]);  head++;
                }
            }
        }
        // Second Row
        head = sum; count = 0;
        ptr = ptr - 18 + width;  // 
        s[0] = *ptr;    ptr++;
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        s[3] = amd_max3(s[0], s[1], s[2]); 
        *head = MAX(s[3], *head);   head++;
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                if(count <= constraint){
                    s[j] = *ptr;    ptr++;
                    s[3] = amd_max3(s[0], s[1], s[2]);
                    *head = MAX(s[3], *head);  head++;
                }
            }
        }
        // Third Row
        head = sum; count = 0;
        out_ptr = output + pixIdx;
        ptr = ptr - 18 + width;  // ptr + -18  - constraint + width
        s[0] = *ptr;    ptr++;
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        s[3] = amd_max3(s[0], s[1], s[2]); 
        *out_ptr = MAX(s[3], *head);   head++;  out_ptr++;// Copy to Output pixels instead of head *////////
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                if(count < constraint){
                    s[j] = *ptr;    ptr++;
                    s[3] = amd_max3(s[0], s[1], s[2]);
                    *out_ptr = MAX(s[3], *head);  head++; out_ptr++;  // Copy to Output pixels instead of head *////////
                }
            }
        }
    }
}

// Can do a bit of loop unrolling and local memory optimization if required
__kernel void dilate_local_optimized(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{
    int id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);
    bool valid = true;    
    id_x= id_x*16;
    if (id_x >= width || id_y >= height || id_z >= channel) return; // Return with nothing for out of bounds                     
    unsigned int IMAGE_W = width, IMAGE_H = height;    
    unsigned int index;
    unsigned char s[4]; // For keeping track of local pixels
    unsigned char sum[16]; // For bookkeeping the 16 maxima
    __global unsigned char *out_ptr;
    __local unsigned char *ptr;
    unsigned char *head;

    const int rowOffset = get_global_id(1) * IMAGE_W; //Done
	const int my = get_global_id(0) + rowOffset; //Done
	
	const int localRowLen = TWICE_HALF_FILTER_SIZE + get_local_size(0) * 16;//Done 34
	const int localRowOffset = ( get_local_id(1) + HALF_FILTER_SIZE ) * localRowLen; // 34 *(1 + local_rows)
	const int myLocal = localRowOffset + get_local_id(0) + HALF_FILTER_SIZE;	// Done!!!	
	__local uchar cached[18*258];
    const int HALF_FILTER_SIZE_IMAGE_W = IMAGE_W +
     HALF_FILTER_SIZE; 
	// copy my pixel
    for(int i = 0; i< 16; i++){
        	cached[ myLocal + i] = input[ my + i]; // Dne
    }
    //printf("%d horizontal", get_local_id(0));
	
	if (
		get_global_id(0) < HALF_FILTER_SIZE 			|| 
		get_global_id(0) > IMAGE_W - HALF_FILTER_SIZE - 1		|| 
		get_global_id(1) < HALF_FILTER_SIZE			||
		get_global_id(1) > IMAGE_H - HALF_FILTER_SIZE - 1
	)
	{
		// no computation for me, sync and exit
        output[my] = input[my];
		barrier(CLK_LOCAL_MEM_FENCE);
		return;
	}
	//Done Till here
	
	else 
	{
		// copy additional elements
		int localColOffset = -1;
		int globalColOffset = -1;
		
		if ( get_local_id(0) < HALF_FILTER_SIZE )
		{
			localColOffset = get_local_id(0);
			globalColOffset = -HALF_FILTER_SIZE;
			
			cached[ localRowOffset + get_local_id(0) ] = input[ my - HALF_FILTER_SIZE ];
		}
		else if ( get_local_id(0) >= get_local_size(0) - HALF_FILTER_SIZE )
		{
			localColOffset = get_local_id(0) + TWICE_HALF_FILTER_SIZE;
			globalColOffset = HALF_FILTER_SIZE;
			
			cached[ myLocal + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE ];
		}
		
		
		if ( get_local_id(1) < HALF_FILTER_SIZE )
		{
			cached[ get_local_id(1) * localRowLen + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my - HALF_FILTER_SIZE_IMAGE_W ];
			if (localColOffset > 0)
			{
				cached[ get_local_id(1) * localRowLen + localColOffset ] = input[ my - HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
			}
		}
		else if ( get_local_id(1) >= get_local_size(1) -HALF_FILTER_SIZE )
		{
			int offset = ( get_local_id(1) + TWICE_HALF_FILTER_SIZE ) * localRowLen;
			cached[ offset + get_local_id(0) + HALF_FILTER_SIZE ] = input[ my + HALF_FILTER_SIZE_IMAGE_W ];
			if (localColOffset > 0)
			{
				cached[ offset + localColOffset ] = input[ my + HALF_FILTER_SIZE_IMAGE_W + globalColOffset ];
			}
		}
		

    // synchronize
	barrier(CLK_LOCAL_MEM_FENCE);
    }
   

    
    int lid_x = get_local_id(0), lid_y = get_local_id(1), lid_z = get_local_id(2);
    lid_x= lid_x*16;
    if (lid_x >= 256 || lid_y >= height || id_z >= channel) return; // Return with nothing for out of bounds 
    
    int max_id;
    int pixIdx = lid_y * channel  + lid_x * channel * 258 + id_z;
    unsigned char max = cached[pixIdx];
    if(lid_x == 0 || lid_x == width-1 || id_y == 0 || lid_y == height-1) // On the boundary of the Image, Keep output as input it self
    {
         output[pixIdx] = cached[pixIdx];
    }
    else
    {
        int constraint = 14, diff, count = 0;
        lid_x = get_local_id(0);
        if (lid_x + 17 >= 258)
        {
            diff = lid_x + 17 + 1 - 258;
            constraint -= diff;
        }   
        max_id = pixIdx - channel - 258*channel;

        // All the three rows available according to thread launch
        // First Row
        ptr = cached + max_id;
        s[0] = *ptr;    ptr++; // These Three pointers are always avaliable according to launch
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        head = sum;
        *head = amd_max3(s[0], s[1], s[2]);  head++; // Storing the partial max here //
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                // If should be on the constraint
                if(count <= constraint){
                    s[j] = *ptr;    ptr++;
                    *head = amd_max3(s[0], s[1], s[2]);  head++;
                }
            }
        }
        // Second Row
        head = sum; count = 0;
        ptr = ptr - 18 + 258;  // 
        s[0] = *ptr;    ptr++;
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        s[3] = amd_max3(s[0], s[1], s[2]); 
        *head = MAX(s[3], *head);   head++;
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                if(count <= constraint){
                    s[j] = *ptr;    ptr++;
                    s[3] = amd_max3(s[0], s[1], s[2]);
                    *head = MAX(s[3], *head);  head++;
                }
            }
        }
        // Third Row
        head = sum; count = 0;
        out_ptr = output + get_global_id(1) * channel * width + get_global_id(0) * channel * 16 + id_z;
        ptr = ptr - 18 + 258;  // ptr + -18  - constraint + width
        s[0] = *ptr;    ptr++;
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        s[3] = amd_max3(s[0], s[1], s[2]); 
        *out_ptr = MAX(s[3], *head);   head++;  out_ptr++;// Copy to Output pixels instead of head *////////
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                if(count < constraint){
                    s[j] = *ptr;    ptr++;
                    s[3] = amd_max3(s[0], s[1], s[2]);
                    *out_ptr = MAX(s[3], *head);  head++; out_ptr++;  // Copy to Output pixels instead of head *////////
                }
            }
        }
    }
}


// Local-Memory and 16 way workload
__kernel void dilate_optimized_trail(  __global unsigned char* input,
                                __global unsigned char* output,
                                const unsigned int height,
                                const unsigned int width,
                                const unsigned int channel,
                                const unsigned int kernelSize
                                )
{
    //printf("coming here");
    __local uchar cached[18 * 258];
   /* for(int i = 0; i< 18; i++){
        for(int j =0; j< 258; j++){
            cached[i*258 + j] = 122;
        }
    }*/
    /*
    Local Memory Loading Comes Here!!!!!
    */

    int lid_x = get_local_id(0), lid_y = get_global_id(1), id_x = get_global_id(0), id_y = get_global_id(1), id_z = get_global_id(2);                   
    unsigned int index;
    unsigned char s[4]; // For keeping track of local pixels
    unsigned char sum[16]; // For bookkeeping the 16 maxima
    __global unsigned char *out_ptr;
    __local unsigned char *ptr;
    unsigned char *head;
    id_x = id_x * 16;

    int max_id;
    int pixIdx = lid_y * channel * 258 + lid_x;
    int gpixIdx = id_y * width + id_x;
    unsigned char max = input[pixIdx];
    if(id_x == 0 || id_x >= width-1 || id_y == 0 || id_y >= height-1) // On the boundary of the Image, Keep output as input it self
    {
        output[gpixIdx] = input[gpixIdx];
    }
    else
    {
        int constraint = 14, diff, count = 0;
        if (id_x + 17 >= 258)
        {
            diff = id_x + 17 + 1 - 258;
            constraint -= diff;
        }   
        max_id = pixIdx - 1 - 258;

        // All the three rows available according to thread launch
        // First Row
        ptr = cached + max_id;
        s[0] = *ptr;    ptr++; // These Three pointers are always avaliable according to launch
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        head = sum;
        *head = amd_max3(s[0], s[1], s[2]);  head++; // Storing the partial max here //
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                // If should be on the constraint
                if(count <= constraint){
                    s[j] = *ptr;    ptr++;
                    *head = amd_max3(s[0], s[1], s[2]);  head++;
                }
            }
        }
        // Second Row
        head = sum; count = 0;
        ptr = ptr - 18 + 258;  // 
        s[0] = *ptr;    ptr++;
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        s[3] = amd_max3(s[0], s[1], s[2]); 
        *head = MAX(s[3], *head);   head++;
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                if(count <= constraint){
                    s[j] = *ptr;    ptr++;
                    s[3] = amd_max3(s[0], s[1], s[2]);
                    *head = MAX(s[3], *head);  head++;
                }
            }
        }
        // Third Row
        head = sum; count = 0;
        out_ptr = output + gpixIdx;
        ptr = ptr - 18 + 258;  // ptr + -18  - constraint + width
        s[0] = *ptr;    ptr++;
        s[1] = *ptr;    ptr++;
        s[2] = *ptr;    ptr++;
        s[3] = amd_max3(s[0], s[1], s[2]); 
        *out_ptr = MAX(s[3], *head);   head++;  out_ptr++;// Copy to Output pixels instead of head *////////
        for(int i =0; i < 5; i++){
            for(int j=0; j< 3; j++){
                if(count < constraint){
                    s[j] = *ptr;    ptr++;
                    s[3] = amd_max3(s[0], s[1], s[2]);
                    *out_ptr = MAX(s[3], *head);  head++; out_ptr++;  // Copy to Output pixels instead of head *////////
                }
            }
        }
    }
}
