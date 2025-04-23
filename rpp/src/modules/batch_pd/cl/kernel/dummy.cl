#pragma OPENCL EXTENSION cl_amd_media_ops : enable
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
float4 amd_unpack(uint src)
{
  return (float4)(amd_unpack0(src), amd_unpack1(src), amd_unpack2(src), amd_unpack3(src));
}

///////////////////////////////////////////////////////////////////////////////
// Data Types
typedef uchar   U1x8;
typedef uint2   U8x8;
typedef  int4  S16x8;
typedef uint4  U16x8;
typedef uint8  U24x8;
typedef uint8  U32x8;
typedef float8 F32x8;
typedef struct {
  float M[3][2];
} ago_affine_matrix_t;
typedef struct {
  float M[3][3];
} ago_perspective_matrix_t;

///////////////////////////////////////////////////////////////////////////////
// load/store data
void load_U1x8(U1x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + (x >> 3);
  *r = *((__global U1x8 *) p);
}

void load_U8x8(U8x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x;
  *r = *((__global U8x8 *) p);
}

void load_S16x8(S16x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x + x;
  *r = *((__global S16x8 *) p);
}

void load_U16x8(U16x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x + x;
  *r = *((__global U16x8 *) p);
}

void load_U24x8(U24x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x * 3;
  (*r).s012 = *((__global uint3 *)(p + 0));
  (*r).s345 = *((__global uint3 *)(p + 12));
}

void load_U32x8(U32x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + (x << 2);
  *r = *((__global U32x8 *) p);
}

void load_F32x8(F32x8 * r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + (x << 2);
  *r = *((__global F32x8 *) p);
}

void store_U1x8(U1x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + (x >> 3);
  *((__global U1x8 *)p) = r;
}

void store_U8x8(U8x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x;
  *((__global U8x8 *)p) = r;
}

void store_S16x8(S16x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x + x;
  *((__global S16x8 *)p) = r;
}

void store_U16x8(U16x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x + x;
  *((__global U16x8 *)p) = r;
}

void store_U24x8(U24x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + x * 3;
  *((__global uint3 *)(p + 0)) = r.s012;
  *((__global uint3 *)(p + 12)) = r.s345;
}

void store_U32x8(U32x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + (x << 2);
  *((__global U32x8 *)p) = r;
}

void store_F32x8(F32x8 r, uint x, uint y, __global uchar * p, uint stride)
{
  p += y*stride + (x << 2);
  *((__global F32x8 *)p) = r;
}

void Convert_U8_U1 (U8x8 * p0, U1x8 p1)
{
    U8x8 r;
    r.s0  = (-(p1 &   1)) & 0x000000ff;
    r.s0 |= (-(p1 &   2)) & 0x0000ff00;
    r.s0 |= (-(p1 &   4)) & 0x00ff0000;
    r.s0 |= (-(p1 &   8)) & 0xff000000;
    r.s1  = (-((p1 >> 4) & 1)) & 0x000000ff;
    r.s1 |= (-(p1 &  32)) & 0x0000ff00;
    r.s1 |= (-(p1 &  64)) & 0x00ff0000;
    r.s1 |= (-(p1 & 128)) & 0xff000000;
    *p0 = r;
}

void Convert_U1_U8 (U1x8 * p0, U8x8 p1)
{
    U1x8 r;
    r  =  p1.s0        &   1;
    r |= (p1.s0 >>  7) &   2;
    r |= (p1.s0 >> 14) &   4;
    r |= (p1.s0 >> 21) &   8;
    r |= (p1.s1 <<  4) &  16;
    r |= (p1.s1 >>  3) &  32;
    r |= (p1.s1 >> 10) &  64;
    r |= (p1.s1 >> 17) & 128;
    *p0 = r;
}
void dilateInside(U8x8 * r, uint x, uint y, __local uchar * lbuf, __global uchar * p, uint stride) {
  int lx = get_local_id(0);
  int ly = get_local_id(1);
  int gx = x >> 3;
  int gy = y;
  int gstride = stride;
  __global uchar * gbuf = p;
  { // load 136x18 bytes into local memory using 16x16 workgroup
    int loffset = ly * 136 + (lx << 3);
    int goffset = (gy - 1) * gstride + (gx << 3) - 4;
    *(__local uint2 *)(lbuf + loffset) = vload2(0, (__global uint *)(gbuf + goffset));
    bool doExtraLoad = false;
    if (ly < 2) {
      loffset += 16 * 136;
      goffset += 16 * gstride;
      doExtraLoad = true;
    }
    else {
      int id = (ly - 2) * 16 + lx;
      int ry = id >> 0;
      int rx = id & 0;
      loffset = ry * 136 + (rx << 3) + 128;
      goffset = (gy - ly + ry - 1) * gstride + ((gx - lx + rx) << 3) + 124;
      doExtraLoad = (ry < 18) ? true : false;
    }
    if (doExtraLoad) {
      *(__local uint2 *)(lbuf + loffset) = vload2(0, (__global uint *)(gbuf + goffset));
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  __local uint2 * lbufptr = (__local uint2 *) (lbuf + ly * 136 + (lx << 3));
  //printf("coming here", );
  F32x8 sum; uint4 pix; float4 val;
  pix.s01 = lbufptr[0];
  pix.s23 = lbufptr[1];
  val.s0 = amd_unpack3(pix.s0);
  val.s1 = amd_unpack0(pix.s1);
  val.s2 = amd_unpack1(pix.s1);
  sum.s0 = amd_max3(val.s0, val.s1, val.s2);
  val.s0 = amd_unpack2(pix.s1);
  sum.s1 = amd_max3(val.s0, val.s1, val.s2);
  val.s1 = amd_unpack3(pix.s1);
  sum.s2 = amd_max3(val.s0, val.s1, val.s2);
  val.s2 = amd_unpack0(pix.s2);
  sum.s3 = amd_max3(val.s0, val.s1, val.s2);
  val.s0 = amd_unpack1(pix.s2);
  sum.s4 = amd_max3(val.s0, val.s1, val.s2);
  val.s1 = amd_unpack2(pix.s2);
  sum.s5 = amd_max3(val.s0, val.s1, val.s2);
  val.s2 = amd_unpack3(pix.s2);
  sum.s6 = amd_max3(val.s0, val.s1, val.s2);
  val.s0 = amd_unpack0(pix.s3);
  sum.s7 = amd_max3(val.s0, val.s1, val.s2);
  pix.s01 = lbufptr[17];
  pix.s23 = lbufptr[18];
  val.s0 = amd_unpack3(pix.s0);
  val.s1 = amd_unpack0(pix.s1);
  val.s2 = amd_unpack1(pix.s1);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s0 = max(sum.s0, val.s3);
  val.s0 = amd_unpack2(pix.s1);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s1 = max(sum.s1, val.s3);
  val.s1 = amd_unpack3(pix.s1);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s2 = max(sum.s2, val.s3);
  val.s2 = amd_unpack0(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s3 = max(sum.s3, val.s3);
  val.s0 = amd_unpack1(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s4 = max(sum.s4, val.s3);
  val.s1 = amd_unpack2(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s5 = max(sum.s5, val.s3);
  val.s2 = amd_unpack3(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s6 = max(sum.s6, val.s3);
  val.s0 = amd_unpack0(pix.s3);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s7 = max(sum.s7, val.s3);
  pix.s01 = lbufptr[34];
  pix.s23 = lbufptr[35];
  val.s0 = amd_unpack3(pix.s0);
  val.s1 = amd_unpack0(pix.s1);
  val.s2 = amd_unpack1(pix.s1);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s0 = max(sum.s0, val.s3);
  val.s0 = amd_unpack2(pix.s1);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s1 = max(sum.s1, val.s3);
  val.s1 = amd_unpack3(pix.s1);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s2 = max(sum.s2, val.s3);
  val.s2 = amd_unpack0(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s3 = max(sum.s3, val.s3);
  val.s0 = amd_unpack1(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s4 = max(sum.s4, val.s3);
  val.s1 = amd_unpack2(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s5 = max(sum.s5, val.s3);
  val.s2 = amd_unpack3(pix.s2);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s6 = max(sum.s6, val.s3);
  val.s0 = amd_unpack0(pix.s3);
  val.s3 = amd_max3(val.s0, val.s1, val.s2); sum.s7 = max(sum.s7, val.s3);
  U8x8 rv;
  rv.s0 = amd_pack(sum.s0123);
  rv.s1 = amd_pack(sum.s4567);
  *r = rv;
}
__kernel __attribute__((reqd_work_group_size(16, 16, 1)))
void Dilate_kernel(uint width, uint height, __global uchar * p0_buf, uint p0_stride, uint p0_offset, __global uchar * p1_buf, uint p1_stride, uint p1_offset)
{
    uint x = get_global_id(0) * 8;
    uint y = get_global_id(1);
    bool valid = (x < width) && (y < height);
    //printf("coming till here!!! %d", y);
    p0_buf += p0_offset;
    __local uchar p1_lbuf[2448];
    p1_buf += p1_offset;
        U8x8 p0;
        U8x8 p1;
        dilateInside(&p0, x, y, p1_lbuf, p1_buf, p1_stride); // Dilate_U8_U8_3x3
    if (valid) {
        store_U8x8(p0, x, y, p0_buf, p0_stride);
    }
}
