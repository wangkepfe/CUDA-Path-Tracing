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
#include <math_functions.h>
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
#define RAY_MIN 1e-5f
#define RAY_MAX 1e20f
#define M_EPSILON 1e-5f
#define SCENE_MAX 1e5f

// sampling settings
#define NUM_SAMPLE 1
#define USE_RUSSIAN false
#define RUSSIAN_P 0.9
#define LIGHT_BOUNCE 8

// ******************* structures ********************

// enum
enum Refl_t { 
	MAT_EMIT, 
	MAT_DIFF, 
	MAT_GLASS, 
	MAT_REFL, 
	MAT_DIFF_REFL,
	MAT_FRESNEL,
	MAT_NO, 
	MAT_MEDIUM,
	MAT_SUBSURFACE,
	MAT_SKIN,
};  // material types
enum Geo_t { GEO_TRIANGLE, GEO_SPHERE, GEO_GROUND };  // geo types
enum Medium_t {MEDIUM_NO = -1, MEDIUM_CLOUD = 0, MEDIUM_TEA, MEDIUM_MILK, MEDIUM_JADE };

// geometry, material
struct Ray {
	Vec3f orig;	// ray origin
	Vec3f dir;		// ray direction	
	__device__ Ray(Vec3f o_, Vec3f d_) : orig(o_), dir(d_) {}
};

struct Sphere {
	float rad;				// radius 
	Vec3f pos, emi, col;	// position, emission, color 
	Refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)
	int medium;

	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 
		// ray/sphere intersection
		Vec3f op = pos - r.orig;   
		float t;
		float b = dot(op, r.dir);
		float disc = b*b - dot(op, op) + rad*rad; // discriminant of quadratic formula
		if (disc<0) return 0; else disc = sqrtf(disc);
		return (t = b - disc) > M_EPSILON ? t : ((t = b + disc) > M_EPSILON ? t : 0.0f);
	}
};

struct GroundPlane {
	// normal (0, 1, 0)
	float y;
	__device__ float intersect(const Ray &r) const { // returns distance, 0 if nohit 
		return abs(r.dir.y) > M_EPSILON ? ((y - r.orig.y) / r.dir.y) : 0.0f;
	}
};

struct MediumSS {
	__device__ MediumSS(){}
	__device__ MediumSS(const Vec3f& sigmaS, const Vec3f& sigmaA, float g) : sigmaS{sigmaS}, sigmaA{sigmaA}, g{g} {}
	Vec3f sigmaS;
	Vec3f sigmaA;
	float g;
	__device__ Vec3f getSigmaT() { return sigmaA + sigmaS; }
	__device__ Vec3f getRho() { return sigmaS / getSigmaT(); }
	__device__ MediumSS& operator*(float scaleFactor) { sigmaS *= scaleFactor; sigmaA *= scaleFactor; return *this; };
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
__device__ __inline__ float rd(curandState* randstate) { return curand_uniform(randstate); }

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
				"vote.ballot.b32    %0,p;       \n"
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

// renderKernel:
// - ray scene traversal
// - surface/media interaction
// - return color of a pixel
__device__ Vec3f renderKernel(
	curandState* randstate, 
	Vec3f& rayorig, 
	Vec3f& raydir, 
	const Camera* userSetting,
	BSSRDF& bssrdf) 
{
	// variables define

	// bvh traversal result
	int hitTriAddr;
	float hitDistance;
	float sceneT;

	// scene interaction
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

	// global color mask, result color
	Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f);
	Vec3f accucolor = Vec3f(0.0f, 0.0f, 0.0f);
	Vec3f beta;
	
	// medium
	const int airMedium = MEDIUM_NO;
	int medium = MEDIUM_NO;
	int objMedium = MEDIUM_NO;

	for (int bounces = 0; 
		#if USE_RUSSIAN == true
		curand_uniform(randstate) < RUSSIAN_P && bounces < LIGHT_BOUNCE;
		#else
		bounces < LIGHT_BOUNCE; 	
		#endif
		bounces++) {

		// initialize variables
		{
			// bvh traversal result
			hitTriAddr = -1;
			hitDistance = 1e20;
			sceneT = 1e20;

			// material
			refltype = MAT_DIFF;
			objcol = Vec3f(1.0f, 1.0f, 1.0f);
			emit = Vec3f(0.0f, 0.0f, 0.0f);
			alphax = 0.0f;
			alphay = 0.0f;
			kd = 1.0f;
			ks = 1.0f;
			etaT = 1.4f;
			useTexture = false;
			useNormal = true;
			//Vec3f F0 = Vec3f(0.04f, 0.04f, 0.04f); // plastic
			//Vec3f F0 = Vec3f(1.00f, 0.71f, 0.29f); // gold
			//Vec3f F0 = Vec3f(0.95f, 0.93f, 0.88f); // silver
			//Vec3f F0 = Vec3f(0.56f, 0.57f, 0.58f); // iron
			F0 = Vec3f(0.56f, 0.57f, 0.58f); // iron
			tangent = Vec3f(0.0f, 1.0f, -1.0f);
		}
		
		// ------------------------ scene interaction ----------------------------

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
		}

		// environmental sphere
		if (sceneT > 1e10f) {
			// Convert (normalized) dir to spherical coordinates.
			float longlatX = atan2f(raydir.x, raydir.z); // Y is up, swap x for y and z for x
			longlatX = longlatX < 0.f ? longlatX + TWO_PI : longlatX;  // wrap around full circle if negative
			float longlatY = acosf(raydir.y); // add RotateMap at some point, see Fragmentarium
			
			float u = fmod(longlatX / (float)TWO_PI + userSetting->envMapRotation, 1.0f); // +offsetY;
			float v = longlatY / M_PI;

			float4 HDRcol = tex2D(HDRtexture, u, v);
			if (userSetting->testLighting) {
				emit = Vec3f(HDRcol.x, HDRcol.y, HDRcol.z);
			} else {
				emit = Vec3f(0.0f, 0.0f, 0.0f);
			}
			

			accucolor += mask * emit * 1.5f; 
			return accucolor; 
		}

		// participating media
		if (medium != MEDIUM_NO) {
			MediumSS med;
			if      (medium == MEDIUM_CLOUD) { med = MediumSS(Vec3f{20.0f, 20.0f, 20.0f}, Vec3f{5.0f, 5.0f, 5.0f}, 0.0f); }
			else if (medium == MEDIUM_TEA)   { med = MediumSS(Vec3f{0.040224f, 0.045264f, 0.051081f} * 5.0f, Vec3f{2.4288f, 4.5757f, 7.2127f}, 0.5f); }
			else if (medium == MEDIUM_MILK)  { med = MediumSS(Vec3f{4.5513f, 5.8294f, 7.136f} * 20.0f, Vec3f{0.0015333f, 0.0046f, 0.019933f}, -0.5f); }
			else if (medium == MEDIUM_JADE)  { med = MediumSS(Vec3f{45.0f, 40.0f, 50.0f}, Vec3f{10.0f, 5.0f, 15.0f}, 0.2f); }
			bool sampledMedium;
			HomogeneousMedium(
				curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate),
				mask, med.getSigmaT(), med.sigmaS, med.g, sceneT, rayorig, raydir, hitpoint, nextdir, sampledMedium
			);
			if (sampledMedium) {
				rayorig = hitpoint;
				raydir = nextdir;
				continue;
			}
		}

		// ---------------------- triangle interaction ----------------------

		// calculate hitpoint
		hitpoint = rayorig + raydir * sceneT;

		// texture fetching
		{
			// read triangle's vertex postion, uv, normal, material index
			float4 po0 = tex1Dfetch(triDebugTexture, hitTriAddr);      Vec3f p0 = Vec3f(po0.x, po0.y, po0.z);
			float4 po1 = tex1Dfetch(triDebugTexture, hitTriAddr + 1);  Vec3f p1 = Vec3f(po1.x, po1.y, po1.z);
			float4 po2 = tex1Dfetch(triDebugTexture, hitTriAddr + 2);  Vec3f p2 = Vec3f(po2.x, po2.y, po2.z);

			float2 uvo0 = tex1Dfetch(triUvTexture, hitTriAddr);        Vec2f uv0 = Vec2f(uvo0.x, uvo0.y);
			float2 uvo1 = tex1Dfetch(triUvTexture, hitTriAddr + 1);    Vec2f uv1 = Vec2f(uvo1.x, uvo1.y);
			float2 uvo2 = tex1Dfetch(triUvTexture, hitTriAddr + 2);    Vec2f uv2 = Vec2f(uvo2.x, uvo2.y);

			float4 normal0 = tex1Dfetch(triNormalTexture, hitTriAddr);     Vec3f n0 = Vec3f(normal0.x, normal0.y, normal0.z);
			float4 normal1 = tex1Dfetch(triNormalTexture, hitTriAddr + 1); Vec3f n1 = Vec3f(normal1.x, normal1.y, normal1.z);
			float4 normal2 = tex1Dfetch(triNormalTexture, hitTriAddr + 2); Vec3f n2 = Vec3f(normal2.x, normal2.y, normal2.z);

			int originalIdx = tex1Dfetch(triIndicesTexture, hitTriAddr);
			int materialId = tex1Dfetch(triMaterialTexture, originalIdx);

			// barycentric interpolation
			float u, v, w;
			Barycentric(hitpoint, p0, p1, p2, u, v, w);

			// interpolate uv and normal
			Vec2f hitUv = uv0 * u + uv1 * v + uv2 * w;
			Vec3f smoothNormal = n0 * u + n1 * v + n2 * w;
			
			// read color from texture
			float4 colorTex = tex2D(colorTexture, hitUv.x, hitUv.y); 

			// scene material user setting
			int sceneTestIdx = 0;
			// 1
			if (userSetting->testMaterialIdx == ++sceneTestIdx) // diffuse + microfacet reflection
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_DIFF_REFL; alphax = 0.05f; alphay = 0.05f; objcol = Vec3f(1.0f, 0.5f, 0.3f); F0 = Vec3f(0.04f); ks = 0.7f; kd = 0.3f; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_DIFF_REFL; alphax = 0.05f; alphay = 0.05f; objcol = Vec3f(0.57f, 0.43f, 0.85f); F0 = Vec3f(0.04f); ks = 0.7f; kd = 0.3f; } // outer
			}  
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // microfacet reflection
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_REFL; alphax = 0.05f; alphay = 0.05f; kd = 0.0f; } // outer
			} 
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // microfacet reflection anisotropic 1
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_REFL; alphax = 0.01f; alphay = 0.5f; kd = 0.0f; } // outer
			} 
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // microfacet reflection anisotropic 2
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_REFL; alphax = 0.5f; alphay = 0.01f; kd = 0.0f; } // outer
			} 
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // smooth glass
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL;  objcol = Vec3f(0.57f, 0.43f, 0.85f); } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_GLASS; } // outer
			} 
			// 6
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // rough glass
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; alphax = 0.05f; alphay = 0.05f; objcol = Vec3f(0.57f, 0.43f, 0.85f); } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_GLASS; alphax = 0.05f; } // outer
			} 
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // substrate/fresnelBlend/coat
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; alphax = 0.01f; alphay = 0.01f; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_FRESNEL; alphax = 0.01f; kd = 20.0f; objcol = Vec3f(0.4f, 0.03f, 0.03f); F0 = Vec3f(0.3f); } // outer
			}
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // media(gas)
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; alphax = 0.01f; alphay = 0.01f; objcol = F0 = Vec3f(1.00f, 0.71f, 0.29f); } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_MEDIUM; objMedium = MEDIUM_CLOUD; } // outer
			}
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // media(liquid, tea)
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_GLASS; objMedium = MEDIUM_TEA; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_GLASS; } // outer
			}
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // media(liquid, milk)
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_GLASS; objMedium = MEDIUM_MILK; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_GLASS; } // outer
			}
			// 11
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // media(solid)
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; alphax = 0.01f; alphay = 0.01f; objcol = F0 = Vec3f(1.0f, 0.7f, 0.3f); } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; } // light
				else if (materialId == 4) { refltype = MAT_GLASS; objMedium = MEDIUM_JADE; alphax = 0.01f; } // outer
			} 
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // subsurface
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; /*refltype = MAT_EMIT; emit = Vec3f(2.0f, 2.0f, 2.0f);*/ } // light
				else if (materialId == 4) { refltype = MAT_SUBSURFACE; } // outer
			} 
			else if (userSetting->testMaterialIdx == ++sceneTestIdx) // subsurface
			{
				if      (materialId == 0) { refltype = MAT_DIFF; useTexture = true; } // ground
				else if (materialId == 1) { refltype = MAT_REFL; } // inner
				else if (materialId == 2) { refltype = MAT_REFL; } // ground label
				else if (materialId == 3) { refltype = MAT_NO; if(!userSetting->testLighting) {refltype = MAT_EMIT; emit = Vec3f(2.0f, 2.0f, 2.0f);} } // light
				else if (materialId == 4) { refltype = MAT_SKIN; alphax = 0.1f; kd = 1.0f; ks = 1.0f; F0 = Vec3f(0.03f); } // outer
			}
			// 0
			else // lambertian reflection
			{
				if      (materialId == 0) { refltype = MAT_DIFF;    useTexture = true;          } // ground
				else if (materialId == 1) { refltype = MAT_DIFF;    objcol = Vec3f(0.75f, 0.75f, 0.75f);          } // inner
				else if (materialId == 2) { refltype = MAT_DIFF;    objcol = Vec3f(0.75f, 0.75f, 0.75f);          } // ground label
				else if (materialId == 3) { refltype = MAT_DIFF;    objcol = Vec3f(0.75f, 0.75f, 0.75f);          } // light
				else if (materialId == 4) { refltype = MAT_DIFF;    objcol = Vec3f(0.75f, 0.75f, 0.75f);          } // outer
			}

			if (userSetting->testTexture && useTexture) { objcol = Vec3f(colorTex.x, colorTex.y, colorTex.z); } 
			if (userSetting->testNormal && useNormal)   { n = smoothNormal; } 
		}

		// n is the geometry surface normal, nl always has opposite direction with raydir
		n.normalize();
		bool into = dot(n, raydir) < 0;
		nl = into ? n : n * -1.0f;  

		// current surface's emission --> previous surfaces' color mask --> camera 
		accucolor += (mask * emit);

		// ------------------------ material ----------------------------
		switch (refltype) {
			case MAT_DIFF: {
				lambertianReflection(curand_uniform(randstate), curand_uniform(randstate), nextdir, nl);
				hitpoint += nl * RAY_MIN;
				mask *= kd * objcol;
				break;
			}
			case MAT_REFL: {
				if (alphax == 0.0f) { 
					// perfect mirror reflection
					nextdir = raydir - n * dot(n, raydir) * 2.0f;
					nextdir.normalize();
					hitpoint += nl * RAY_MIN;
					mask *= ks * objcol;
				} else { 
					// microfacet reflection
					macrofacetReflection(curand_uniform(randstate), curand_uniform(randstate), raydir, nextdir, nl, tangent, beta, F0, alphax, alphay);
					mask *= ks * beta * objcol;
				}
				hitpoint += nl * RAY_MIN;
				break;
			}
			case MAT_DIFF_REFL: {
				// blend diffuse and reflection
				if (curand_uniform(randstate) < ks / (ks + kd)) {
					// reflection
					macrofacetReflection(curand_uniform(randstate), curand_uniform(randstate), raydir, nextdir, nl, tangent, beta, F0, alphax, alphay);
					mask *= beta;
				} else {
					// diffuse
					lambertianReflection(curand_uniform(randstate), curand_uniform(randstate), nextdir, nl);
					mask *= objcol;
				}
				break;
			}
			case MAT_FRESNEL: {
				fresnelBlend(curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate), raydir, nextdir, nl, beta, kd * objcol, F0, alphax);
				mask *= beta;
				break;
			}
			case MAT_GLASS: {
				if (alphax == 0.0f) { 
					// perfect specular glass
					bool refl;
					specularGlass(curand_uniform(randstate), into, raydir, nextdir, nl, refl, etaT);
					hitpoint += nl * RAY_MIN * (refl ? 1 : -1);
					if (airMedium != objMedium) medium = (medium == airMedium) ? (refl ? airMedium : objMedium) : (refl ? objMedium : airMedium);
				} else {
					// microfacet glass
					bool refl;
					macrofacetGlass(curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate), into, beta, raydir, nextdir, nl, refl, etaT, alphax);
					hitpoint += nl * RAY_MIN * (refl ? 1 : -1);
					mask *= beta;
					if (airMedium != objMedium) medium = (medium == airMedium) ? (refl ? airMedium : objMedium) : (refl ? objMedium : airMedium);
				}
				break;
			}
			case MAT_EMIT: {
				 return accucolor;  
			}
			case MAT_MEDIUM: {
				nextdir = raydir;
				hitpoint -= nl * RAY_MIN;
				if (airMedium != objMedium) medium = (medium == airMedium) ? (!into ? airMedium : objMedium) : (!into ? objMedium : airMedium);
				break;
			}
			case MAT_SUBSURFACE: {
				//int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

				// float scaleFactor = 300.0f;
				// MediumSS med = MediumSS(Vec3f{0.74, 0.88, 1.01}, Vec3f{0.032, 0.17, 0.48}, 0.0f) * scaleFactor; // skin 1
				// //MediumSS med = MediumSS(Vec3f{0.18, 0.07, 0.03} * 50.0f, Vec3f{0.061, 0.97, 1.45} * 50.0f, 0.0f); // ketchup
				// //MediumSS med = MediumSS(Vec3f{2.55, 3.21, 3.77} * 50.0f, Vec3f{0.0011, 0.0024, 0.014} * 50.0f, 0.0f); // Wholemilk
				// //MediumSS med = MediumSS(Vec3f{4.5513f, 5.8294f, 7.136f} * 100.0f, Vec3f{0.0015333f, 0.0046f, 0.019933f} * 100.0f, 0.0f); // Wholemilk

				// Vec3f probeRayOrig;
				// Vec3f probeRayDir;
				// float probeRayLength;  

				// Vec3f sigmaT = med.getSigmaT();
				// Vec3f rho = med.getRho();
			
				// // if (threadId == 460800) {
				// // 	printf("\n\nEntering subsurface scattering kernel...\n");
				// // 	printf("BSSRDF: sigmaT = (%f, %f, %f), rho = (%f, %f, %f)\n", sigmaT.x, sigmaT.y, sigmaT.z, rho.x, rho.y, rho.z);
				// // 	printf("BSSRDF: hitpoint = (%f, %f, %f), normal = (%f, %f, %f)\n", hitpoint.x, hitpoint.y, hitpoint.z, nl.x, nl.y, nl.z);
				// // }

				// float sampledRadius;
				// sampleBSSRDFprobeRay(rd(randstate), rd(randstate), rd(randstate), nl, hitpoint, sigmaT, rho, probeRayOrig, probeRayDir, probeRayLength, bssrdf, sampledRadius);

				// // if (threadId == 460800) {
				// // 	printf("BSSRDF: probeRayOrig = (%f, %f, %f), probeRayDir = (%f, %f, %f), probeRayLength = %f\n", 
				// // 		probeRayOrig.x, probeRayOrig.y, probeRayOrig.z, probeRayDir.x, probeRayDir.y, probeRayDir.z, probeRayLength);
				// // }
				
				// intersectBVHandTriangles(make_float4(probeRayOrig.x, probeRayOrig.y, probeRayOrig.z, RAY_MIN), 
				// 						make_float4(probeRayDir.x,  probeRayDir.y,  probeRayDir.z,  probeRayLength),
				// 						hitTriAddr, hitDistance, trinormal, false);

				// int originalIdx = tex1Dfetch(triIndicesTexture,  hitTriAddr);
				// int materialId  = tex1Dfetch(triMaterialTexture, originalIdx);

				// if (!(hitDistance < probeRayLength && hitDistance > RAY_MIN && (materialId == 4))) {
				// 	// if (threadId == 460800) {
				// 	// 	printf("BSSRDF: probe ray missed.\n");	
				// 	// }
				// 	break; // try again	
				// }

				// float4 po0 = tex1Dfetch(triDebugTexture, hitTriAddr);      Vec3f p0 = Vec3f(po0.x, po0.y, po0.z);
				// float4 po1 = tex1Dfetch(triDebugTexture, hitTriAddr + 1);  Vec3f p1 = Vec3f(po1.x, po1.y, po1.z);
				// float4 po2 = tex1Dfetch(triDebugTexture, hitTriAddr + 2);  Vec3f p2 = Vec3f(po2.x, po2.y, po2.z);

				// float4 normal0 = tex1Dfetch(triNormalTexture, hitTriAddr);     Vec3f n0 = Vec3f(normal0.x, normal0.y, normal0.z);
				// float4 normal1 = tex1Dfetch(triNormalTexture, hitTriAddr + 1); Vec3f n1 = Vec3f(normal1.x, normal1.y, normal1.z);
				// float4 normal2 = tex1Dfetch(triNormalTexture, hitTriAddr + 2); Vec3f n2 = Vec3f(normal2.x, normal2.y, normal2.z);

				// // normal
				// float u, v, w;
				// Barycentric(hitpoint, p0, p1, p2, u, v, w);
				// Vec3f smoothNormal = n0 * u + n1 * v + n2 * w;
				// smoothNormal.normalize();

				// // next point
				// Vec3f nextPoint = probeRayOrig + probeRayDir * hitDistance;

				// // if (threadId == 460800) {
				// // 	printf("BSSRDF: probe ray hit. nextPoint = (%f, %f, %f), normal = (%f, %f, %f), distance = %f\n", 
				// // 		nextPoint.x, nextPoint.y, nextPoint.z, smoothNormal.x, smoothNormal.y, smoothNormal.z, (nextPoint - hitpoint).length());
				// // }

				// lambertianReflection(rd(randstate), rd(randstate), nextdir, smoothNormal);
				// calculateBSSRDF(nl, smoothNormal, nextdir, sigmaT, rho, etaT, beta, bssrdf, sampledRadius, (nextPoint - hitpoint).length());
				// mask *= beta;

				// // if (threadId == 460800) {
				// // 	printf("BSSRDF: nextdir = (%f, %f, %f), beta = (%f, %f, %f)\n\n", 
				// // 		nextdir.x, nextdir.y, nextdir.z, beta.x, beta.y, beta.z);
				// // }

				// hitpoint = nextPoint + RAY_MIN * smoothNormal;
			
				break;
			}
			case MAT_SKIN: {
				bool refl;
				Vec3f sampledNormal;
				microfacetSampling(rd(randstate), rd(randstate), into, raydir, nl, refl, etaT, alphax, sampledNormal, beta, nextdir);
				if (refl) {
					hitpoint += nl * RAY_MIN;
					mask *= beta * ks;
				} else {
					float scaleFactor = 10.0f;
					MediumSS med = MediumSS(Vec3f{0.74, 0.88, 1.01}, Vec3f{0.032, 0.17, 0.48}, 0.0f) * scaleFactor; // skin 1
					Vec3f sigmaT = med.getSigmaT();
					Vec3f rho = med.getRho();
					
					Vec3f vx, vy;
					localizeSample(nl, vx, vy);
					SKIN_PROBE_TRY_AGAIN:;
					int hitCount = 0;
					Vec3f probeRayOrig, probeRayDir, probeHitPoint[5], probeHitPointNormal[5];
					float probeRayLength;  
					sampleBSSRDFprobeRay(rd(randstate), rd(randstate), rd(randstate), rd(randstate), nl, hitpoint, sigmaT, rho, probeRayOrig, probeRayDir, probeRayLength, bssrdf, vx, vy);
					for (int i = 0; i < 5; ++i) {
						intersectBVHandTriangles(make_float4(probeRayOrig.x, probeRayOrig.y, probeRayOrig.z, RAY_MIN), make_float4(probeRayDir.x,  probeRayDir.y,  probeRayDir.z,  RAY_MAX), hitTriAddr, hitDistance, trinormal, false);
						if (probeRayLength < hitDistance) {
							break;
						}
						Vec3f probeHitPointAny = probeRayOrig + probeRayDir * hitDistance;
						if (tex1Dfetch(triMaterialTexture, tex1Dfetch(triIndicesTexture,  hitTriAddr)) == 4) {
							probeHitPoint[hitCount] = probeHitPointAny;
							probeHitPointNormal[hitCount] = trinormal;
							++hitCount;
						}
						probeRayLength -= hitDistance;
						probeRayOrig = probeHitPointAny + RAY_MIN * probeRayDir;
					}
					if (hitCount == 0) {
						if (
							#if USE_RUSSIAN == true
							curand_uniform(randstate) < RUSSIAN_P && bounces < LIGHT_BOUNCE
							#else
							bounces < LIGHT_BOUNCE	
							#endif
						) {
							++bounces;
							goto SKIN_PROBE_TRY_AGAIN;
						} else {
							break;
						}
					}

					int nextPointIdx = rd(randstate) * hitCount;
					lambertianReflection(rd(randstate), rd(randstate), nextdir, probeHitPointNormal[nextPointIdx]);
					calculateBSSRDF(nl, probeHitPointNormal[nextPointIdx], nextdir, sigmaT, rho, etaT, beta, bssrdf, probeHitPoint[nextPointIdx] - hitpoint, vx, vy);
					mask *= beta * kd * hitCount;
					hitpoint = probeHitPoint[nextPointIdx] + RAY_MIN * probeHitPointNormal[nextPointIdx];
				}
				break;
			}
			case MAT_NO: default: {
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
	BSSRDF bssrdf)
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

		float jitterValueX = curand_uniform(&randState) - 0.5;
		float jitterValueY = curand_uniform(&randState) - 0.5;
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
			float random1 = curand_uniform(&randState);
			float random2 = curand_uniform(&randState);

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

		finalcol += renderKernel(&randState, originInWorldSpace, rayInWorldSpace, cudaRendercam, bssrdf) * (1.0f / NUM_SAMPLE);
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
	const float4 *cudaNormalPtr, const int *cudaMaterialPtr, BSSRDF bssrdf)
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

	pathTracingKernel <<< fullBlocksPerGrid, threadsPerBlock >>> (outputbuf, accumbuf, framenumber, hashedframenumber, cudaRenderCam, bssrdf);
}
