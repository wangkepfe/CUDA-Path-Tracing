#pragma once

#include "linear_math.h"

#include <cuda.h>

#include "stdio.h"

#include "Camera.h"
#include "bssrdfTable.h"
// #include "SceneDescription.h"

#define scrwidth 720
#define scrheight 720

void cudaRender(const float4* cudaNodes, const float4* cudaTriWoops, const float4* cudaDebugTris, const int* cudaTriInds, 
	Vec3f* outputbuf, Vec3f* accumbuf, const cudaArray* HDRmap, const cudaArray* colorArray, const unsigned int framenumber, const unsigned int hashedframenumber, 
	const unsigned int totalnodecnt, const unsigned int leafnodecnt, const unsigned int tricnt, const Camera* cudaRenderCam, const float2 *cudaUvPtr,
    const float4 *cudaNormalPtr, const int *cudaMaterialPtr, BSSRDF bssrdf);

//------------------------------------------------------------------------
// Constants.
//------------------------------------------------------------------------

enum
{
	MaxBlockHeight = 6,            // Upper bound for blockDim.y.
	EntrypointSentinel = 0x76543210,   // Bottom-most stack entry, indicating the end of traversal.
};

//------------------------------------------------------------------------
// BVH memory layout.
//------------------------------------------------------------------------

enum BVHLayout
{
	BVHLayout_AOS_AOS = 0,              // Nodes = array-of-structures, triangles = array-of-structures. Used by tesla_xxx kernels.
	BVHLayout_AOS_SOA,                  // Nodes = array-of-structures, triangles = structure-of-arrays.
	BVHLayout_SOA_AOS,                  // Nodes = structure-of-arrays, triangles = array-of-structures.
	BVHLayout_SOA_SOA,                  // Nodes = structure-of-arrays, triangles = structure-of-arrays.
	BVHLayout_Compact,                  // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.
	BVHLayout_Compact2,                 // Variant of BVHLayout_AOS_AOS with implicit leaf nodes.

	BVHLayout_Max
};

//------------------------------------------------------------------------
// Kernel configuration. Written by queryConfig() in each CU file.
//------------------------------------------------------------------------

//struct KernelConfig
//{
//	int         bvhLayout;              // Desired BVHLayout.
//	int         blockWidth;             // Desired blockDim.x.
//	int         blockHeight;            // Desired blockDim.y.
//	int         usePersistentThreads;   // True to enable persistent threads.
//};


//------------------------------------------------------------------------
// Utilities.
//------------------------------------------------------------------------

#define FETCH_GLOBAL(NAME, IDX, TYPE) ((const TYPE*)NAME)[IDX]
#define FETCH_TEXTURE(NAME, IDX, TYPE) tex1Dfetch(t_ ## NAME, IDX)
#define STORE_RESULT(RAY, TRI, T) ((int2*)results)[(RAY) * 2] = make_int2(TRI, __float_as_int(T))

//------------------------------------------------------------------------

#ifdef __CUDACC__  // compute capability (newer GPUs only)

template <class T> __device__ __inline__ void swap(T& a, T& b)
{
	T t = a;
	a = b;
	b = t;
}

__device__ __inline__ float min4(float a, float b, float c, float d)
{
	return fminf(fminf(fminf(a, b), c), d);
}

__device__ __inline__ float max4(float a, float b, float c, float d)
{
	return fmaxf(fmaxf(fmaxf(a, b), c), d);
}

__device__ __inline__ float min3(float a, float b, float c)
{
	return fminf(fminf(a, b), c);
}

__device__ __inline__ float max3(float a, float b, float c)
{
	return fmaxf(fmaxf(a, b), c);
}

// Using integer min,max
__inline__ __device__ float fminf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2<b2 ? a2 : b2);
}

__inline__ __device__ float fmaxf2(float a, float b)
{
	int a2 = __float_as_int(a);
	int b2 = __float_as_int(b);
	return __int_as_float(a2>b2 ? a2 : b2);
}

#endif

//------------------------------------------------------------------------

