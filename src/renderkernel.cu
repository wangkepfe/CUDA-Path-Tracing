/******************************************
 * 
 *          CUDA GPU path tracing
 * 
 * 
 * 
 * 
 * ****************************************/


// cuda
#include <cuda.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

// render kernal header
#include "CudaRenderKernel.h"

// c
#include "stdio.h"

// utils
#include "cutil_math.h"  // required for float3
#include "mymath.h"

// material modeling
#include "reflection.cuh"
#include "bssrdf.cuh"

// ******************* macro define ********************

// constants
#define F32_MIN          (1.175494351e-38f)
#define F32_MAX          (3.402823466e+38f)

// bvh stack
#define STACK_SIZE  64  // Size of the traversal stack in local memory.
#define EntrypointSentinel 0x76543210

// limits
#define RAY_MIN 1e-4f
#define RAY_MAX 1e20f
#define M_EPSILON 1e-4f

// sampling settings
#define NUM_SAMPLE 1
#define LIGHT_BOUNCE_NUM_MIN 2
#define LIGHT_BOUNCE_NUM_MAX 16
#define USE_ENVMAP true
#define USE_DISTANT_LIGHT false

// ******************* structures ********************

// enum
enum Geo_t { GEO_TRIANGLE, GEO_SPHERE, GEO_GROUND };  // geo types

// geometry, material
struct Ray {
	Vec3f orig;	// ray origin
	Vec3f dir;		// ray direction	
	__device__ Ray(Vec3f o_, Vec3f d_) : orig(o_), dir(d_) {}
};

struct Sphere {
	float rad; 
	Vec3f pos;

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 
		// ray/sphere intersection
		Vec3f op = pos - r.orig;   
		float t;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant of quadratic formula
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > M_EPSILON ? t : ((t = b + disc) > M_EPSILON ? t : 0.0f);
	}

	__device__ Vec3f getNormal(const Vec3f& point) const {
		return normalize(point - pos);
	}
};

struct GroundPlane {
	// normal (0, 1, 0)
	float y;
	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 
		return abs(r.dir.y) > M_EPSILON ? ((y - r.orig.y) / r.dir.y) : 0.0f;
	}
	__device__ Vec3f getNormal() const {
		return Vec3f(0.0f, 1.0f, 0.0f);
	}
};

// ******************* global variables ********************

// bvh
texture<float4, 1, cudaReadModeElementType> bvhNodesTexture;
texture<float4, 1, cudaReadModeElementType> triWoopTexture;
texture<float4, 1, cudaReadModeElementType> triDebugTexture;
texture<int, 1, cudaReadModeElementType> triIndicesTexture;
texture<float2, 1, cudaReadModeElementType> triUvTexture;
texture<float4, 1, cudaReadModeElementType> triNormalTexture;
texture<int, 1, cudaReadModeElementType> triMaterialTexture;

// hdr
texture<float4, cudaTextureType2D, cudaReadModeElementType> HDRtexture;

// color texture
texture<float4, cudaTextureType2D, cudaReadModeElementType> colorTexture;

// ******************* math util func ********************

__device__ inline Vec3f absmax3f(const Vec3f& v1, const Vec3f& v2) { return Vec3f(v1.x*v1.x > v2.x*v2.x ? v1.x : v2.x, v1.y*v1.y > v2.y*v2.y ? v1.y : v2.y, v1.z*v1.z > v2.z*v2.z ? v1.z : v2.z); }
__device__ __inline__ int   min_min(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   min_max(int a, int b, int c) { int v; asm("vmin.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_min(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.min %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ int   max_max(int a, int b, int c) { int v; asm("vmax.s32.s32.s32.max %0, %1, %2, %3;" : "=r"(v) : "r"(a), "r"(b), "r"(c)); return v; }
__device__ __inline__ float fmin_fmin(float a, float b, float c) { return __int_as_float(min_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmin_fmax(float a, float b, float c) { return __int_as_float(min_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmin(float a, float b, float c) { return __int_as_float(max_min(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }
__device__ __inline__ float fmax_fmax(float a, float b, float c) { return __int_as_float(max_max(__float_as_int(a), __float_as_int(b), __float_as_int(c))); }

__device__ __inline__ float spanBeginKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d){ return fmax_fmax(fminf(a0, a1), fminf(b0, b1), fmin_fmax(c0, c1, d)); }
__device__ __inline__ float spanEndKepler(float a0, float a1, float b0, float b1, float c0, float c1, float d)	{ return fmin_fmin(fmaxf(a0, a1), fmaxf(b0, b1), fmax_fmin(c0, c1, d)); }
__device__ __inline__ void swap2(int& a, int& b){ int temp = a; a = b; b = temp;}

__device__ __inline__ float rd(curandState* rdState) { return curand_uniform(rdState); }

// ******************* functions ********************

// intersectBVHandTriangles
__device__ void intersectBVHandTriangles(
	const float4 rayorig, 
	const float4 raydir,
	int& hitTriIdx, 
	float& hitdistance, 
	Vec3f& trinormal,
	bool anyHit)
{
	// assign a CUDA thread to every pixel by using the threadIndex
	// global threadId, see richiesams blogspot
	//int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	///////////////////////////////////////////
	//// KEPLER KERNEL
	///////////////////////////////////////////

	// BVH layout Compact2 for Kepler
	int traversalStack[STACK_SIZE];

	// Live state during traversal, stored in registers.

	//int		rayidx;		// not used, can be removed
	float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
	float   idirx, idiry, idirz;    // 1 / ray direction
	float   oodx, oody, oodz;       // ray origin / ray direction

	char*   stackPtr;               // Current position in traversal stack.
	int     leafAddr;               // If negative, then first postponed leaf, non-negative if no leaf (innernode).
	int     nodeAddr;
	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitT;                   // t-value of the closest intersection.
	
	//int threadId1; // ipv rayidx

	// Initialize (stores local variables in registers)
	{
		// Pick ray index.

		//threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
		

		// Fetch ray.
		origx = rayorig.x;
		origy = rayorig.y;
		origz = rayorig.z;
		dirx = raydir.x;
		diry = raydir.y;
		dirz = raydir.z;
		tmin = rayorig.w;

		// ooeps is very small number, used instead of raydir xyz component when that component is near zero
		float ooeps = exp2f(-80.0f); // Avoid div by zero, returns 1/2^80, an extremely small number
		idirx = 1.0f / (fabsf(raydir.x) > ooeps ? raydir.x : copysignf(ooeps, raydir.x)); // inverse ray direction
		idiry = 1.0f / (fabsf(raydir.y) > ooeps ? raydir.y : copysignf(ooeps, raydir.y)); // inverse ray direction
		idirz = 1.0f / (fabsf(raydir.z) > ooeps ? raydir.z : copysignf(ooeps, raydir.z)); // inverse ray direction
		oodx = origx * idirx;  // ray origin / ray direction
		oody = origy * idiry;  // ray origin / ray direction
		oodz = origz * idirz;  // ray origin / ray direction

		// Setup traversal + initialisation

		traversalStack[0] = EntrypointSentinel; // Bottom-most entry. 0x76543210 (1985229328 in decimal)
		stackPtr = (char*)&traversalStack[0]; // point stackPtr to bottom of traversal stack = EntryPointSentinel
		leafAddr = 0;   // No postponed leaf.
		nodeAddr = 0;   // Start from the root.
		hitIndex = -1;  // No triangle intersected so far.
		hitT = raydir.w; // tmax  
	}

	// Traversal loop.

	while (nodeAddr != EntrypointSentinel) 
	{
		// Traverse internal nodes until all SIMD lanes have found a leaf.

		//bool searchingLeaf = true; // required for warp efficiency
		while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)  
		{
			// Fetch AABBs of the two child nodes.

			// nodeAddr is an offset in number of bytes (char) in gpuNodes array
			
			float4 n0xy = tex1Dfetch(bvhNodesTexture, nodeAddr); // childnode 0, xy-bounds (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)		
			float4 n1xy = tex1Dfetch(bvhNodesTexture, nodeAddr + 1); // childnode 1, xy-bounds (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)		
			float4 nz = tex1Dfetch(bvhNodesTexture, nodeAddr + 2); // childnode 0 and 1, z-bounds (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)		
            float4 tmp = tex1Dfetch(bvhNodesTexture, nodeAddr + 3); // contains indices to 2 childnodes in case of innernode, see below
            int2 cnodes = *(int2*)&tmp; // cast first two floats to int
            // (childindex = size of array during building, see CudaBVH.cpp)

			// compute ray intersections with BVH node bounding box

			/// RAY BOX INTERSECTION
			// Intersect the ray against the child nodes.

			float c0lox = n0xy.x * idirx - oodx; // n0xy.x = c0.lo.x, child 0 minbound x
			float c0hix = n0xy.y * idirx - oodx; // n0xy.y = c0.hi.x, child 0 maxbound x
			float c0loy = n0xy.z * idiry - oody; // n0xy.z = c0.lo.y, child 0 minbound y
			float c0hiy = n0xy.w * idiry - oody; // n0xy.w = c0.hi.y, child 0 maxbound y
			float c0loz = nz.x   * idirz - oodz; // nz.x   = c0.lo.z, child 0 minbound z
			float c0hiz = nz.y   * idirz - oodz; // nz.y   = c0.hi.z, child 0 maxbound z
			float c1loz = nz.z   * idirz - oodz; // nz.z   = c1.lo.z, child 1 minbound z
			float c1hiz = nz.w   * idirz - oodz; // nz.w   = c1.hi.z, child 1 maxbound z
			float c0min = spanBeginKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
			float c0max = spanEndKepler(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)
			float c1lox = n1xy.x * idirx - oodx; // n1xy.x = c1.lo.x, child 1 minbound x
			float c1hix = n1xy.y * idirx - oodx; // n1xy.y = c1.hi.x, child 1 maxbound x
			float c1loy = n1xy.z * idiry - oody; // n1xy.z = c1.lo.y, child 1 minbound y
			float c1hiy = n1xy.w * idiry - oody; // n1xy.w = c1.hi.y, child 1 maxbound y
			float c1min = spanBeginKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			float c1max = spanEndKepler(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

			// ray box intersection boundary tests:
			
			//float ray_tmax = 1e20;
			bool traverseChild0 = (c0min <= c0max); // && (c0min >= tmin) && (c0min <= ray_tmax);
			bool traverseChild1 = (c1min <= c1max); // && (c1min >= tmin) && (c1min <= ray_tmax);

			// Neither child was intersected => pop stack.

			if (!traverseChild0 && !traverseChild1)   
			{
				nodeAddr = *(int*)stackPtr; // fetch next node by popping the stack 
				stackPtr -= 4; // popping decrements stackPtr by 4 bytes (because stackPtr is a pointer to char)   
			}

			// Otherwise, one or both children intersected => fetch child pointers.

			else  
			{
				// set nodeAddr equal to intersected childnode index (or first childnode when both children are intersected)
				nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y; 

				// Both children were intersected => push the farther one on the stack.

				if (traverseChild0 && traverseChild1) // store closest child in nodeAddr, swap if necessary
				{   
					if (c1min < c0min)  
						swap2(nodeAddr, cnodes.y);  
					stackPtr += 4;  // pushing increments stack by 4 bytes (stackPtr is a pointer to char)
					*(int*)stackPtr = cnodes.y; // push furthest node on the stack
				}
			}

			// First leaf => postpone and continue traversal.
			// leafnodes have a negative index to distinguish them from inner nodes
			// if nodeAddr less than 0 -> nodeAddr is a leaf
			if (nodeAddr < 0 && leafAddr >= 0)  
			{
				//searchingLeaf = false; // required for warp efficiency
				leafAddr = nodeAddr;  
				nodeAddr = *(int*)stackPtr;  // pops next node from stack
				stackPtr -= 4;  // decrements stackptr by 4 bytes (because stackPtr is a pointer to char)
			}

			// All SIMD lanes have found a leaf => process them.

			// to increase efficiency, check if all the threads in a warp have found a leaf before proceeding to the
			// ray/triangle intersection routine
			// this bit of code requires PTX (CUDA assembly) code to work properly

			// if (!__any(searchingLeaf)) -> "__any" keyword: if none of the threads is searching a leaf, in other words
			// if all threads in the warp found a leafnode, then break from while loop and go to triangle intersection

			//if(!__any(leafAddr >= 0))
			//    break;

			// if (!__any(searchingLeaf))
			//	break;    /// break from while loop and go to code below, processing leaf nodes

			// NOTE: inline PTX implementation of "if(!__any(leafAddr >= 0)) break;".
			// tried everything with CUDA 4.2 but always got several redundant instructions.

			unsigned int mask; // replaces searchingLeaf

			asm("{\n"
				"   .reg .pred p;               \n"
				"setp.ge.s32        p, %1, 0;   \n"
				"vote.sync.ballot.b32    %0,p,0xffffffff;  \n"
				"}"
				: "=r"(mask)
				: "r"(leafAddr));

			if (!mask)
				break;	
		} 

		
		///////////////////////////////////////////
		/// TRIANGLE INTERSECTION
		//////////////////////////////////////

		// Process postponed leaf nodes.

		while (leafAddr < 0)  /// if leafAddr is negative, it points to an actual leafnode (when positive or 0 it's an innernode)
		{
			// Intersect the ray against each triangle using Sven Woop's algorithm.
			// Woop ray triangle intersection: Woop triangles are unit triangles. Each ray
			// must be transformed to "unit triangle space", before testing for intersection

			for (int triAddr = ~leafAddr;; triAddr += 3)  // triAddr is index in triWoop array (and bitwise complement of leafAddr)
			{ // no defined upper limit for loop, continues until leaf terminator code 0x80000000 is encountered

				// Read first 16 bytes of the triangle.
				// fetch first precomputed triangle edge
				float4 v00 = tex1Dfetch(triWoopTexture, triAddr);
				
				// End marker 0x80000000 (negative zero) => all triangles in leaf processed --> terminate
				if (__float_as_int(v00.x) == 0x80000000) 
					 break;

				// Compute and check intersection t-value (hit distance along ray).
				float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;   // Origin z
				float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);  // inverse Direction z
				float t = Oz * invDz;   
				
				if (t > tmin && t < hitT)
				{
					// Compute and check barycentric u.

					// fetch second precomputed triangle edge
					float4 v11 = tex1Dfetch(triWoopTexture, triAddr + 1);
					float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;  // Origin.x
					float Dx = dirx * v11.x + diry * v11.y + dirz * v11.z;  // Direction.x
					float u = Ox + t * Dx; /// parametric equation of a ray (intersection point)

					if (u >= 0.0f && u <= 1.0f)
					{
						// Compute and check barycentric v.

						// fetch third precomputed triangle edge
						float4 v22 = tex1Dfetch(triWoopTexture, triAddr + 2);
						float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
						float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
						float v = Oy + t*Dy;

						if (v >= 0.0f && u + v <= 1.0f)
						{
							// We've got a hit!
							// Record intersection.

							hitT = t;
							hitIndex = triAddr; // store triangle index for shading

							// Closest intersection not required => terminate.
							if (anyHit)  // only true for shadow rays
							{
								nodeAddr = EntrypointSentinel;
								break;
							}

							// compute normal vector by taking the cross product of two edge vectors
							// because of Woop transformation, only one set of vectors works
							
							//trinormal = cross(Vec3f(v22.x, v22.y, v22.z), Vec3f(v11.x, v11.y, v11.z));  // works
							trinormal = cross(Vec3f(v11.x, v11.y, v11.z), Vec3f(v22.x, v22.y, v22.z));
						}
					}
				}
			} // end triangle intersection

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if (nodeAddr < 0)    // nodeAddr is an actual leaf when < 0
			{
				nodeAddr = *(int*)stackPtr;  // pop stack
				stackPtr -= 4;               // decrement with 4 bytes to get the next int (stackPtr is char*)
			}
		} // end leaf/triangle intersection loop
	} // end traversal loop (AABB and triangle intersection)

	// Remap intersected triangle index, and store the result.

	if (hitIndex != -1){
		// hitIndex = tex1Dfetch(triIndicesTexture, hitIndex);
		// remapping tri indices delayed until this point for performance reasons
		// (slow texture memory lookup in de triIndicesTexture) because multiple triangles per node can potentially be hit
	}

	hitTriIdx = hitIndex;
	hitdistance = hitT;
}

// fetch envmap texture
__device__ inline Vec3f envLight(const Camera* cameraSetting, Vec3f& raydir) {
	#if USE_ENVMAP
	// Convert (normalized) dir to spherical coordinates.
	float longlatX = atan2f(raydir.x, raydir.z); // Y is up, swap x for y and z for x
	longlatX = longlatX < 0.f ? longlatX + TWO_PI : longlatX;  // wrap around full circle if negative
	float longlatY = acosf(raydir.y); // add RotateMap at some point, see Fragmentarium
	
	float u = fmod(longlatX / (float)TWO_PI + cameraSetting->envMapRotation, 1.0f); // +offsetY;
	float v = longlatY / M_PI;

	float4 HDRcol = tex2D(HDRtexture, u, v);
	return Vec3f(HDRcol.x, HDRcol.y, HDRcol.z);
	#else
	return Vec3f(0.0f);
	#endif
}

// fetch textures: uv, normal, color
__device__ inline void textureFetching(Vec3f& smoothNormal, Vec3f& objcol, int hitTriAddr, const Vec3f& hitpoint, bool useTexture) {
	float4 po0 = tex1Dfetch(triDebugTexture, hitTriAddr);      Vec3f p0 = Vec3f(po0.x, po0.y, po0.z);
	float4 po1 = tex1Dfetch(triDebugTexture, hitTriAddr + 1);  Vec3f p1 = Vec3f(po1.x, po1.y, po1.z);
	float4 po2 = tex1Dfetch(triDebugTexture, hitTriAddr + 2);  Vec3f p2 = Vec3f(po2.x, po2.y, po2.z);

	float2 uvo0 = tex1Dfetch(triUvTexture, hitTriAddr);        Vec2f uv0 = Vec2f(uvo0.x, uvo0.y);
	float2 uvo1 = tex1Dfetch(triUvTexture, hitTriAddr + 1);    Vec2f uv1 = Vec2f(uvo1.x, uvo1.y);
	float2 uvo2 = tex1Dfetch(triUvTexture, hitTriAddr + 2);    Vec2f uv2 = Vec2f(uvo2.x, uvo2.y);

	float4 normal0 = tex1Dfetch(triNormalTexture, hitTriAddr);     Vec3f n0 = Vec3f(normal0.x, normal0.y, normal0.z);
	float4 normal1 = tex1Dfetch(triNormalTexture, hitTriAddr + 1); Vec3f n1 = Vec3f(normal1.x, normal1.y, normal1.z);
	float4 normal2 = tex1Dfetch(triNormalTexture, hitTriAddr + 2); Vec3f n2 = Vec3f(normal2.x, normal2.y, normal2.z);

	// barycentric interpolation
	float u, v, w;
	Barycentric(hitpoint, p0, p1, p2, u, v, w);

	// interpolate uv and normal
	Vec2f hitUv = uv0 * u + uv1 * v + uv2 * w;
	smoothNormal = n0 * u + n1 * v + n2 * w;
	
	if (useTexture) {
		// read color from texture
		float4 colorTex = tex2D(colorTexture, hitUv.x, hitUv.y);
		objcol = Vec3f(colorTex.x, colorTex.y, colorTex.z);
	}
}

// renderKernel:
// - ray scene traversal
// - surface/media interaction
// - return color of a pixel
__device__ Vec3f renderKernel(
	curandState* rdState, 
	Vec3f& rayorig, 
	Vec3f& raydir, 
	const Camera* cameraSetting,
	BSSRDF& bssrdf, 
	MatDesc* gpuMatDesc) 
{
	// variables define

	// bvh traversal result
	int hitTriAddr;
	float hitDistance;
	float sceneT;

	// scene interaction
	int geometryType;
	Vec3f hitpoint;
	Vec3f n;
	Vec3f nl;
	Vec3f nextdir;
	Vec3f trinormal;

	// material
	Refl_t refltype;
	Vec3f objcol;
	Vec3f emit;
	float alphax;
	float alphay;
	float kd;
	float ks;
	float etaT;
	bool useTexture;
	bool useNormal;
	Vec3f F0;
	Vec3f tangent;
	Vec3f mfp;

	// global
	int bssrdfMatId = -1;
	int lightBounceNum = LIGHT_BOUNCE_NUM_MIN;

	// global color mask, result color
	Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f);
	Vec3f accucolor = Vec3f(0.0f, 0.0f, 0.0f);
	Vec3f beta;
	
	// distant light
	#if USE_DISTANT_LIGHT
	Vec3f Ldis = Vec3f(1.2f, 1.2f, 1.2f);
	Vec3f Ddis = normalize(Vec3f(0.0f, 1.3f, -3.6f));
	#endif

	for (int bounces = 0; bounces < lightBounceNum && bounces < LIGHT_BOUNCE_NUM_MAX; bounces++) {
		// ------------------------ scene interaction ----------------------------

		// bvh traversal
		geometryType = 0;
		hitTriAddr = -1;
		hitDistance = 1e20;
		sceneT = 1e20;

		// global
		bssrdfMatId = -1;

		// triangles
		intersectBVHandTriangles(
			make_float4(rayorig.x, rayorig.y, rayorig.z, RAY_MIN), 
			make_float4(raydir.x, raydir.y, raydir.z, RAY_MAX),
			hitTriAddr, 
			hitDistance, 
			trinormal,
			false);

		if (hitDistance < sceneT && hitDistance > RAY_MIN) { // triangle hit
			sceneT = hitDistance;
			n = trinormal;
			geometryType = GEO_TRIANGLE;
		}

		// environmental sphere
		if (sceneT > 1e10f) {
			emit = envLight(cameraSetting, raydir);
			accucolor += mask * emit; 
			return accucolor; 
		}

		// ---------------------- triangle interaction ----------------------

		// calculate hitpoint
		hitpoint = rayorig + raydir * sceneT;

		// geometry type
		{
			// fetch textures
			int originalIdx = tex1Dfetch(triIndicesTexture, hitTriAddr);
			int materialId = tex1Dfetch(triMaterialTexture, originalIdx);

			refltype = static_cast<Refl_t>(gpuMatDesc[materialId].refltype);
			objcol = gpuMatDesc[materialId].objcol;
			emit = gpuMatDesc[materialId].emit;
			alphax = gpuMatDesc[materialId].alphax;
			alphay = gpuMatDesc[materialId].alphay;
			kd = gpuMatDesc[materialId].kd;
			ks = gpuMatDesc[materialId].ks;
			etaT = gpuMatDesc[materialId].etaT;
			useNormal = gpuMatDesc[materialId].useNormal;
			useTexture = gpuMatDesc[materialId].useTexture;
			F0 = gpuMatDesc[materialId].F0;
			tangent = gpuMatDesc[materialId].tangent;
			mfp = gpuMatDesc[materialId].mfp;

			if (refltype == MAT_SUBSURFACE) { bssrdfMatId = materialId; }

			Vec3f smoothNormal;
			Vec3f colorTex;
			textureFetching(smoothNormal, colorTex, hitTriAddr, hitpoint, useTexture);

			if (useTexture) { objcol = colorTex; } 
			if (useNormal)  { n = smoothNormal; } 
		}

		// n is the geometry surface normal, nl always has opposite direction with raydir
		n.normalize();
		bool into = dot(n, raydir) < 0;
		nl = into ? n : n * -1.0f;  

		// current surface's emission --> previous surfaces' color mask --> camera 
		accucolor += mask * emit;

		// ------------------------ material ----------------------------
		switch (refltype) {
			case MAT_DIFF: {
				// beta = f * cosTh / pdf
				lambertianReflection(rd(rdState), rd(rdState), nextdir, nl);
				hitpoint += nl * RAY_MIN;
				mask *= kd * objcol;

				// ----- importance sampling -----
				#if USE_DISTANT_LIGHT
				if (dot(Ddis, nl) < 0) {
					break;
				}

				// shadow ray probe
				Vec3f shadowRayOrig = hitpoint;
				Vec3f shadowRayDir = Ddis;
				Vec3f probeNormal;

				intersectBVHandTriangles(make_float4(shadowRayOrig.x, shadowRayOrig.y, shadowRayOrig.z, 
					RAY_MIN), make_float4(shadowRayDir.x,  shadowRayDir.y,  shadowRayDir.z,  
						RAY_MAX), hitTriAddr, hitDistance, probeNormal, false);

				// in shadow
				if (hitDistance < 1e10f) {
					break;
				}

				// distant light
				Vec3f f = objcol * invPi;
				float lightPdf = 1.0f;
				float scatteringPdf = abs(dot(Ddis, nl)) * invPi;
				float weightFactor = (scatteringPdf + lightPdf) / (scatteringPdf * scatteringPdf + lightPdf * lightPdf);
				accucolor += mask * f * Ldis * weightFactor;
				#endif

				break;
			}
			case MAT_REFL: {
				lightBounceNum++;
				if (alphax == 0.0f) { 
					// perfect mirror reflection
					nextdir = raydir - n * dot(n, raydir) * 2.0f;
					nextdir.normalize();
					hitpoint += nl * RAY_MIN;
					mask *= ks * objcol;
				} else { 
					// microfacet reflection
					macrofacetReflection(rd(rdState), rd(rdState), raydir, nextdir, nl, tangent, beta, F0, alphax, alphay);
					mask *= ks * beta * objcol;
				}
				hitpoint += nl * RAY_MIN;
				break;
			}
			case MAT_DIFF_REFL: {
				// blend diffuse and reflection
				if (rd(rdState) < ks / (ks + kd)) {
					// reflection
					lightBounceNum++;
					macrofacetReflection(rd(rdState), rd(rdState), raydir, nextdir, nl, tangent, beta, F0, alphax, alphay);
					mask *= beta;
				} else {
					// diffuse
					lambertianReflection(rd(rdState), rd(rdState), nextdir, nl);
					mask *= objcol;
				}
				break;
			}
			case MAT_FRESNEL: {
				lightBounceNum++;
				fresnelBlend(rd(rdState), rd(rdState), rd(rdState), raydir, nextdir, nl, beta, kd * objcol, F0, alphax);
				mask *= beta;
				break;
			}
			case MAT_GLASS: {
				lightBounceNum++;
				if (alphax == 0.0f) { 
					// perfect specular glass
					bool refl;
					specularGlass(rd(rdState), into, raydir, nextdir, nl, refl, etaT);
					hitpoint += nl * RAY_MIN * (refl ? 1 : -1);
					
				} else {
					// microfacet glass
					bool refl;
					macrofacetGlass(rd(rdState), rd(rdState), rd(rdState), into, beta, raydir, nextdir, nl, refl, etaT, alphax);
					hitpoint += nl * RAY_MIN * (refl ? 1 : -1);
					mask *= beta * objcol;
					if (!refl && !into) mask *= etaT * etaT;
					
				}
				break;
			}
			case MAT_EMIT: {
				 return accucolor;  
			}
			case MAT_SUBSURFACE: {
				bool refl;
				Vec3f sampledNormal;
				microfacetSampling(rd(rdState), rd(rdState), into, raydir, nl, refl, etaT, alphax, sampledNormal, beta, nextdir);
				if (refl) {
					lightBounceNum++;
					hitpoint += nl * RAY_MIN;
					mask *= beta * ks * objcol;
					break;
				}

				Vec3f normal2 = sampledNormal;

				// material define
				Vec3f rho = objcol;
				Vec3f sigmaT = 1.0 / mfp;
				
				// localize
				Vec3f vx, vy;
				localizeSample(normal2, vx, vy);
				
				// sample probe ray and probing
				const float maxRatio = 10.0f;
				const float minNormalDot = 0.1f;
				int hitCount = 0, hitCountPerProbe, probeHitCount;
				bool selectThisProbeRay = false, needNewProbeRay = true;
				float sampledRadius, probeRayLength;
				Vec3f probeRayOrig, probeRayDir, probeRayVec, probeNormal;
				Vec3f probeHitPoint, probeHitPointNormal, probeHitPointColor;
				const int maxLoopNum = 3;
				for (int loopnum = 0; loopnum < maxLoopNum; ++loopnum) {
					// sample (ch, axis, radius) a probe ray
					if (needNewProbeRay) {
						sampleBSSRDFprobeRay(rd(rdState), rd(rdState), rd(rdState),
						  normal2, hitpoint, sigmaT, rho, probeRayOrig, probeRayDir, probeRayLength, bssrdf,
							vx, vy, sampledRadius);

						needNewProbeRay = false;
						if (selectThisProbeRay) {
							probeHitCount = hitCountPerProbe;
						}
						selectThisProbeRay = false;
						hitCountPerProbe = 0;
					}
				
					// search along the probe ray
					intersectBVHandTriangles(make_float4(probeRayOrig.x, probeRayOrig.y, probeRayOrig.z, 
						RAY_MIN), make_float4(probeRayDir.x,  probeRayDir.y,  probeRayDir.z,  
							RAY_MAX), hitTriAddr, hitDistance, probeNormal, false);
					
					// out of probe ray length
					if (probeRayLength < hitDistance) {
						needNewProbeRay = true;
						continue;
					}

					// hit
					Vec3f probeHitPointAny = probeRayOrig + probeRayDir * hitDistance;
					probeRayVec = probeHitPointAny - hitpoint;
					float realRadius = probeRayVec.length();
					
					// texture fetching
					Vec3f smoothNormal;
					Vec3f probeObjColor = objcol;
					textureFetching(smoothNormal, probeObjColor, hitTriAddr, probeHitPointAny, useTexture);
					int surfaceMat = tex1Dfetch(triMaterialTexture, tex1Dfetch(triIndicesTexture,  hitTriAddr));
					float normalDot = abs(dot(smoothNormal, probeRayDir));

					// test condition and record
					if (surfaceMat == bssrdfMatId && (realRadius / sampledRadius < maxRatio && normalDot > minNormalDot)) 
					{
						++hitCount;
						++hitCountPerProbe;

						if (hitCount == 1 || rd(rdState) < 1.0f / hitCount) {
							// record hitpoint
							probeHitPoint = probeHitPointAny;
							probeHitPointNormal = useNormal ? smoothNormal : probeNormal;
							probeHitPointColor = probeObjColor;
							selectThisProbeRay = true;
						}
					}

					// next segment
					probeRayLength -= hitDistance;
					probeRayOrig = probeHitPointAny + RAY_MIN * probeRayDir;
				}
				if (hitCount == 0) {
					hitpoint += nl * RAY_MIN;
					mask *= beta * ks * objcol;
					break;
				}
				if (selectThisProbeRay) {
					probeHitCount = hitCountPerProbe;
				}
				mask *= probeHitCount * probeHitPointColor * objcol * 0.8f;
				
				// choose next point and sample next direction
				Vec3f& nextHitPoint = probeHitPoint;
				Vec3f& nextNormal = probeHitPointNormal;

				nextNormal.normalize();
				lambertianReflection(rd(rdState), rd(rdState), nextdir, nextNormal);

				// calculate value
				calculateBSSRDF(normal2, nextNormal, sigmaT, rho, beta, bssrdf, probeRayVec, vx, vy);
				mask *= beta;
				Vec3f importanceSamplingMask = mask;

				// final mask
				float outS = (1.0f - FrD(dot(nextdir, nextNormal), 1.0f, etaT)) / (1.0f - 2.0f * FM1(1.0f / etaT));
				mask *= outS;

				// next point (small position bias)
				hitpoint = nextHitPoint + RAY_MIN * nextNormal;

				// importance sampling light
				#if USE_DISTANT_LIGHT
				float cosTh = dot(Ddis, nextNormal);

				if (cosTh < 0) {
					break;
				}

				// shadow ray probe
				Vec3f shadowRayOrig = hitpoint;
				Vec3f shadowRayDir = Ddis;

				intersectBVHandTriangles(make_float4(shadowRayOrig.x, shadowRayOrig.y, shadowRayOrig.z, 
					RAY_MIN), make_float4(shadowRayDir.x,  shadowRayDir.y,  shadowRayDir.z,  
				  RAY_MAX), hitTriAddr, hitDistance, probeNormal, false);

				// in shadow
				if (hitDistance < 1e10f) {
						break;
				}

				// distant light
				Vec3f surfaceF = Vec3f((1.0f - FrD(abs(cosTh), 1.0f, etaT)) / (1.0f - 2.0f * FM1(1.0f / etaT)) * invPi);
				float lightPdf = 1.0f;
				float scatteringPdf = abs(cosTh) * invPi;
				float weightFactor = (scatteringPdf + lightPdf) / (scatteringPdf * scatteringPdf + lightPdf * lightPdf);
				accucolor += importanceSamplingMask * surfaceF * Ldis * weightFactor;
				#endif

				break;
			}
			case MAT_NULL: default: {
				nextdir = raydir; hitpoint -= nl * RAY_MIN;
			}
		}

		rayorig = hitpoint; 
		raydir = nextdir; 
	}

	return accucolor;
}

// pathTracingKernel:
// - originate ray of a pixel
// - anti-aliasing
// - depth of field
// - return averaged color of the pixel
__global__ void pathTracingKernel(
	Vec3f* output, 
	Vec3f* accumbuffer, 
	unsigned int framenumber, 
	unsigned int hashedframenumber, 
	const Camera* cudaRendercam,
	BSSRDF bssrdf,
	MatDesc* gpuMatDesc)
{
	// assign a CUDA thread to every pixel by using the threadIndex
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// global threadId, see richiesams blogspot
	int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	// create random number generator and initialise with hashed frame number, see RichieSams blogspot
	curandState randState; // state of the random number generator, to prevent repetition
	curand_init(hashedframenumber + threadId, 0, 0, &randState);

	Vec3f finalcol; // final pixel colour 
	finalcol = Vec3f(0.0f, 0.0f, 0.0f); // reset colour to zero for every pixel	
	//Vec3f rendercampos = Vec3f(0, 0.2, 4.6f); 
	Vec3f rendercampos = Vec3f(cudaRendercam->position.x, cudaRendercam->position.y, cudaRendercam->position.z);

	int i = (scrheight - y - 1) * scrwidth + x; // pixel index in buffer	
	int pixelx = x; // pixel x-coordinate on screen
	int pixely = scrheight - y - 1; // pixel y-coordintate on screen

	Vec3f camdir = Vec3f(0, -0.042612, -1); camdir.normalize();
	Vec3f cx = Vec3f(scrwidth * .5135f / scrheight, 0.0f, 0.0f);  // ray direction offset along X-axis 
	Vec3f cy = (cross(cx, camdir)).normalize() * .5135f; // ray dir offset along Y-axis, .5135 is FOV angle

	for (int s = 0; s < NUM_SAMPLE; s++) {

		// compute primary ray direction
		// use camera view of current frame (transformed on CPU side) to create local orthonormal basis
		Vec3f rendercamview = Vec3f(cudaRendercam->view.x, cudaRendercam->view.y, cudaRendercam->view.z); rendercamview.normalize(); // view is already supposed to be normalized, but normalize it explicitly just in case.
		Vec3f rendercamup = Vec3f(cudaRendercam->up.x, cudaRendercam->up.y, cudaRendercam->up.z); rendercamup.normalize();
		Vec3f horizontalAxis = cross(rendercamview, rendercamup); horizontalAxis.normalize(); // Important to normalize!
		Vec3f verticalAxis = cross(horizontalAxis, rendercamview); verticalAxis.normalize(); // verticalAxis is normalized by default, but normalize it explicitly just for good measure.

		Vec3f middle = rendercampos + rendercamview;
		Vec3f horizontal = horizontalAxis * tanf(cudaRendercam->fov.x * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5
		Vec3f vertical = verticalAxis * tanf(-cudaRendercam->fov.y * 0.5 * (M_PI / 180)); // Treating FOV as the full FOV, not half, so multiplied by 0.5

		// anti-aliasing
		// calculate center of current pixel and add random number in X and Y dimension
		// based on https://github.com/peterkutz/GPUPathTracer 

		float jitterValueX = rd(&randState) - 0.5;
		float jitterValueY = rd(&randState) - 0.5;
		float sx = (jitterValueX + pixelx) / (cudaRendercam->resolution.x - 1);
		float sy = (jitterValueY + pixely) / (cudaRendercam->resolution.y - 1);

		// compute pixel on screen
		Vec3f pointOnPlaneOneUnitAwayFromEye = middle + (horizontal * ((2 * sx) - 1)) + (vertical * ((2 * sy) - 1));
		Vec3f pointOnImagePlane = rendercampos + ((pointOnPlaneOneUnitAwayFromEye - rendercampos) * cudaRendercam->focalDistance); // Important for depth of field!		

		// calculation of depth of field / camera aperture 
		// based on https://github.com/peterkutz/GPUPathTracer 

		Vec3f aperturePoint = Vec3f(0, 0, 0);

		if (cudaRendercam->apertureRadius > 0.00001) { // the small number is an epsilon value.

			// generate random numbers for sampling a point on the aperture
			float random1 = rd(&randState);
			float random2 = rd(&randState);

			// randomly pick a point on the circular aperture
			float angle = TWO_PI * random1;
			float distance = cudaRendercam->apertureRadius * sqrtf(random2);
			float apertureX = cos(angle) * distance;
			float apertureY = sin(angle) * distance;

			aperturePoint = rendercampos + (horizontalAxis * apertureX) + (verticalAxis * apertureY);
		}
		else { // zero aperture
			aperturePoint = rendercampos;
		}

		// calculate ray direction of next ray in path
		Vec3f apertureToImagePlane = pointOnImagePlane - aperturePoint;
		apertureToImagePlane.normalize(); // ray direction needs to be normalised

		// ray direction
		Vec3f rayInWorldSpace = apertureToImagePlane;
		rayInWorldSpace.normalize();

		// ray origin
		Vec3f originInWorldSpace = aperturePoint;

		finalcol += renderKernel(&randState, originInWorldSpace, rayInWorldSpace, cudaRendercam, bssrdf, gpuMatDesc) * (1.0f / NUM_SAMPLE);
	}

	// add pixel colour to accumulation buffer (accumulates all samples) 
	accumbuffer[i] += finalcol;

	// averaged colour: divide colour by the number of calculated frames so far
	Vec3f tempcol = accumbuffer[i] / framenumber;

	// union struct required for mapping pixel colours to OpenGL buffer
	union Colour  // 4 bytes = 4 chars = 1 float
	{
		float c;
		uchar4 components;
	};

	Colour fcolour;
	Vec3f colour = Vec3f(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));

	// convert from 96-bit to 24-bit colour + perform gamma correction
	fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255),
		                             (unsigned char)(powf(colour.y, 1 / 2.2f) * 255),
									 (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);

	// store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
	output[i] = Vec3f(x, y, fcolour.c);
}

// cudaRender
// - bind buffers to textures
// - kernal dimension setting
// - launch kernal
void cudaRender(const float4* nodes, const float4* triWoops, const float4* debugTris, const int* triInds, 
	Vec3f* outputbuf, Vec3f* accumbuf, const cudaArray* HDRmap, const cudaArray* colorArray, const unsigned int framenumber, const unsigned int hashedframenumber, 
	const unsigned int nodeSize, const unsigned int leafnodecnt, const unsigned int tricnt, const Camera* cudaRenderCam, const float2 *cudaUvPtr,
	const float4 *cudaNormalPtr, const int *cudaMaterialPtr, BSSRDF bssrdf, MatDesc* gpuMatDesc)
{
	static bool firstTime = true;

	// texture binding
	if (firstTime) {
		firstTime = false;
		
		// bvh textures
		cudaChannelFormatDesc channel0desc = cudaCreateChannelDesc<int>();
		cudaBindTexture(NULL, &triIndicesTexture, triInds, &channel0desc, (tricnt * 3 + leafnodecnt) * sizeof(int));

		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &triWoopTexture, triWoops, &channel1desc, (tricnt * 3 + leafnodecnt) * sizeof(float4));

		cudaChannelFormatDesc channel2desc = cudaCreateChannelDesc<float2>();
		cudaBindTexture(NULL, &triUvTexture, cudaUvPtr, &channel2desc, (tricnt * 3 + leafnodecnt) * sizeof(float2));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &triDebugTexture, debugTris, &channel3desc, (tricnt * 3 + leafnodecnt) * sizeof(float4));

		cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &bvhNodesTexture, nodes, &channel4desc, nodeSize * sizeof(float4)); 

		cudaChannelFormatDesc channel5desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &triNormalTexture, cudaNormalPtr, &channel5desc, (tricnt * 3 + leafnodecnt) * sizeof(float4)); 

		cudaChannelFormatDesc channel6desc = cudaCreateChannelDesc<int>();
		cudaBindTexture(NULL, &triMaterialTexture, cudaMaterialPtr, &channel6desc, tricnt * 3 * sizeof(int));

		// hdr texture
		HDRtexture.addressMode[0] = cudaAddressModeClamp;
		HDRtexture.addressMode[1] = cudaAddressModeClamp;
		HDRtexture.filterMode = cudaFilterModeLinear;
		HDRtexture.normalized = true;

		cudaChannelFormatDesc channel7desc = cudaCreateChannelDesc<float4>(); 
		cudaBindTextureToArray(HDRtexture, HDRmap, channel7desc);

		// color texture
		colorTexture.normalized = true;
		colorTexture.filterMode = cudaFilterModeLinear;
		colorTexture.addressMode[0] = cudaAddressModeWrap;
		colorTexture.addressMode[1] = cudaAddressModeWrap;
		colorTexture.maxAnisotropy = 8;
		colorTexture.sRGB = true;

		cudaChannelFormatDesc channel8desc = cudaCreateChannelDesc<float4>(); 
		cudaBindTextureToArray(colorTexture, colorArray, channel8desc);

		printf("CudaWoopTriangles texture initialised, tri count: %d\n", tricnt);
	}

	dim3 threadsPerBlock (16, 16, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 fullBlocksPerGrid (scrwidth / threadsPerBlock.x, scrheight / threadsPerBlock.y, 1);

	pathTracingKernel <<< fullBlocksPerGrid, threadsPerBlock >>> (outputbuf, accumbuf, framenumber, hashedframenumber, cudaRenderCam, bssrdf, gpuMatDesc);
}
