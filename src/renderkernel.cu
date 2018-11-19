// BVH traversal kernels based on "Understanding the 

#include <cuda.h>
#include <math_functions.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "CudaRenderKernel.h"
#include "stdio.h"
#include <curand.h>
#include <curand_kernel.h>
#include "cutil_math.h"  // required for float3
#include "reflection.cuh"

#define STACK_SIZE  64  // Size of the traversal stack in local memory.
#ifndef M_PI
#define M_PI 3.1415926535897932384626422832795028841971f
#endif
//#define TWO_PI 6.2831853071795864769252867665590057683943f
#define DYNAMIC_FETCH_THRESHOLD 20          // If fewer than this active, fetch new rays
#define samps 1
#define F32_MIN          (1.175494351e-38f)
#define F32_MAX          (3.402823466e+38f)

#define HDRwidth 4096
#define HDRheight 2048
#define HDR
#define EntrypointSentinel 0x76543210
#define MaxBlockHeight 6

#define RAY_MIN 1e-5f
#define RAY_MAX 1e20f
#define M_EPSILON 1e-5f

#define SCENE_MAX 1e5f

#define USE_RUSSIAN true
#define RUSSIAN_P 0.7
#define LIGHT_BOUNCE 20

enum Refl_t { MAT_DIFF, MAT_MIRROR, MAT_GLASS };  // material types
enum Geo_t { GEO_TRIANGLE, GEO_SPHERE, GEO_GROUND };  // geo types
enum Medium_t {MEDIUM_NO = -1, MEDIUM_TEST = 0};

// CUDA textures containing scene data
texture<float4, 1, cudaReadModeElementType> bvhNodesTexture;
texture<float4, 1, cudaReadModeElementType> triWoopTexture;
texture<float4, 1, cudaReadModeElementType> triNormalsTexture;
texture<int, 1, cudaReadModeElementType> triIndicesTexture;
texture<float4, 1, cudaReadModeElementType> HDRtexture;

__device__ inline Vec3f absmax3f(const Vec3f& v1, const Vec3f& v2) {
	return Vec3f(v1.x*v1.x > v2.x*v2.x ? v1.x : v2.x, v1.y*v1.y > v2.y*v2.y ? v1.y : v2.y, v1.z*v1.z > v2.z*v2.z ? v1.z : v2.z);
}

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
	Vec3f sigmaS;
	Vec3f sigmaA;
	float g;
	__device__ Vec3f getSigmaT() { return sigmaA + sigmaS; }
};


//  RAY BOX INTERSECTION ROUTINES

// Experimentally determined best mix of float/int/video minmax instructions for Kepler.

// float c0min = spanBeginKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin); // Tesla does max4(min, min, min, tmin)
// float c0max = spanEndKepler2(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT); // Tesla does min4(max, max, max, tmax)

// Perform min/max operations in hardware
// Using Kepler's video instructions, see http://docs.nvidia.com/cuda/parallel-thread-execution/#axzz3jbhbcTZf																			//  : "=r"(v) overwrites v and puts it in a register
// see https://gcc.gnu.org/onlinedocs/gcc/Extended-Asm.html

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

__device__ void intersectBVHandTriangles(const float4 rayorig, const float4 raydir,
	int& hitTriIdx, float& hitdistance, int& debugbingo, Vec3f& trinormal, int leafcount, int tricount, bool anyHit)
{
	// assign a CUDA thread to every pixel by using the threadIndex
	// global threadId, see richiesams blogspot
	int thread_index = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	///////////////////////////////////////////
	//// KEPLER KERNEL
	///////////////////////////////////////////

	// BVH layout Compact2 for Kepler
	int traversalStack[STACK_SIZE];

	// Live state during traversal, stored in registers.

	int		rayidx;		// not used, can be removed
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
	
	int threadId1; // ipv rayidx

	// Initialize (stores local variables in registers)
	{
		// Pick ray index.

		threadId1 = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
		

		// Fetch ray.

		// required when tracing ray batches
		// float4 o = rays[rayidx * 2 + 0];  
		// float4 d = rays[rayidx * 2 + 1];
		//__shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.

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

		bool searchingLeaf = true; // required for warp efficiency
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
			
			float ray_tmax = 1e20;
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
				searchingLeaf = false; // required for warp efficiency
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
		hitIndex = tex1Dfetch(triIndicesTexture, hitIndex);
		// remapping tri indices delayed until this point for performance reasons
		// (slow texture memory lookup in de triIndicesTexture) because multiple triangles per node can potentially be hit
	}

	hitTriIdx = hitIndex;
	hitdistance = hitT;
}

// union struct required for mapping pixel colours to OpenGL buffer
union Colour  // 4 bytes = 4 chars = 1 float
{
	float c;
	uchar4 components;
};

__device__ Vec3f renderKernel(curandState* randstate, const float4* HDRmap, const float4* gpuNodes, const float4* gpuTriWoops, 
	const float4* gpuDebugTris, const int* gpuTriIndices, Vec3f& rayorig, Vec3f& raydir, unsigned int leafcount, unsigned int tricount) 
{
	Vec3f mask = Vec3f(1.0f, 1.0f, 1.0f); // colour mask
	Vec3f accucolor = Vec3f(0.0f, 0.0f, 0.0f); // accumulated colour
	
	int airMedium = MEDIUM_NO;
	int medium = airMedium;
	int objMedium;

	for (int bounces = 0; 
		#if USE_RUSSIAN == true
		curand_uniform(randstate) < RUSSIAN_P && bounces < LIGHT_BOUNCE;
		#else
		bounces < LIGHT_BOUNCE; 	
		#endif
		bounces++){

		int hitSphereIdx = -1;
		int hitTriIdx = -1;
		int bestTriIdx = -1;
		int geomtype = -1;

		float hitSphereDist = 1e20;
		float hitDistance = 1e20;
		float sceneT = 1e20;

		Vec3f objcol = Vec3f(0, 0, 0);
		Vec3f emit = Vec3f(0, 0, 0);

		Vec3f hitpoint; // intersection point
		Vec3f n; // normal
		Vec3f nl; // oriented normal
		Vec3f nextdir; // ray direction of next path segment
		Vec3f trinormal = Vec3f(0, 0, 0);

		Refl_t refltype;
		int debugbingo = 0;

		// ------------------------ scene interaction ----------------------------

		// triangles
		intersectBVHandTriangles(
			make_float4(rayorig.x, rayorig.y, rayorig.z, RAY_MIN), 
			make_float4(raydir.x, raydir.y, raydir.z, RAY_MAX),
			bestTriIdx, hitDistance, debugbingo, trinormal, leafcount, tricount, false);
		
		if (hitDistance < sceneT && hitDistance > RAY_MIN) { // triangle hit
			sceneT = hitDistance;
			hitTriIdx = bestTriIdx;
			geomtype = GEO_TRIANGLE;
		}

		// ground
		GroundPlane ground {-0.78f};
		if ((hitSphereDist = ground.intersect(Ray(rayorig, raydir)))
		  && hitSphereDist < sceneT 
		  && hitSphereDist > RAY_MIN) { 
			sceneT = hitSphereDist;
			geomtype = GEO_GROUND;
		}

		// spheres
		Sphere spheres[] = {
			//{ 0.78f, { 0.0f, 0.0f, -3.0f }, { 0.0, 0.0, 0.0 }, { 1.0f, 1.0f, 1.0f }, MAT_GLASS, 1},
			{ 0.0f, { 0.0f, 0.0f, 0.0f }, { 0.0, 0.0, 0.0 }, { 0.0f, 0.0f, 0.0f }, MAT_DIFF, MEDIUM_NO} // null ball
		};
		float numspheres = sizeof(spheres) / sizeof(Sphere);
		for (int i = int(numspheres); i--;){  // for all spheres in scene
			if ((hitSphereDist = spheres[i].intersect(Ray(rayorig, raydir)))  // keep track of distance from origin to closest intersection point
			&& hitSphereDist < sceneT && hitSphereDist > RAY_MIN) { 
				sceneT = hitSphereDist; 
				hitSphereIdx = i; 
				geomtype = GEO_SPHERE; 
			}
		}

		// participating media
		if (medium != MEDIUM_NO) {
			MediumSS med {{5, 3, 6}, {0.05, 0.05, 0.05}, 0.2f};
			bool sampledMedium;
			HomogeneousMedium(
				curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate), curand_uniform(randstate),
				mask,
				med.getSigmaT(), med.sigmaS, med.g,
				sceneT,
				rayorig, raydir,
				hitpoint, nextdir,
				sampledMedium
			);
			if (sampledMedium) {
				rayorig = hitpoint;

				if (rayorig.lengthsq() > 400.0f) { // max length = 20
					return Vec3f(0.1f, 0.1f, 0.1f);
				} else {
					raydir = nextdir;
					continue;
				}
			}
		}

		// environmental sphere
	#ifdef HDR
		if (sceneT > 1e10f) {

			// HDR environment map code based on Syntopia "Path tracing 3D fractals"
			// http://blog.hvidtfeldts.net/index.php/2015/01/path-tracing-3d-fractals/
			// https://github.com/Syntopia/Fragmentarium/blob/master/Fragmentarium-Source/Examples/Include/IBL-Pathtracer.frag
			// GLSL code: 
			// vec3 equirectangularMap(sampler2D sampler, vec3 dir) {
			//		dir = normalize(dir);
			//		vec2 longlat = vec2(atan(dir.y, dir.x) + RotateMap, acos(dir.z));
			//		return texture2D(sampler, longlat / vec2(2.0*PI, PI)).xyz; }

			// Convert (normalized) dir to spherical coordinates.
			float longlatX = atan2f(raydir.x, raydir.z); // Y is up, swap x for y and z for x
			longlatX = longlatX < 0.f ? longlatX + TWO_PI : longlatX;  // wrap around full circle if negative
			float longlatY = acosf(raydir.y); // add RotateMap at some point, see Fragmentarium
			
			// map theta and phi to u and v texturecoordinates in [0,1] x [0,1] range
			float offsetY = 0.5f;
			float u = longlatX / TWO_PI; // +offsetY;
			float v = longlatY / M_PI ; 

			// map u, v to integer coordinates
			int u2 = (int)(u * HDRwidth); //% HDRwidth;
			int v2 = (int)(v * HDRheight); // % HDRheight;

			// compute the texel index in the HDR map 
			int HDRtexelidx = u2 + v2 * HDRwidth;

			//float4 HDRcol = HDRmap[HDRtexelidx];
			float4 HDRcol = tex1Dfetch(HDRtexture, HDRtexelidx);  // fetch from texture
			Vec3f HDRcol2 = Vec3f(HDRcol.x, HDRcol.y, HDRcol.z);

			emit = HDRcol2 * 2.0f;
			accucolor += (mask * emit); 
			return accucolor; 
		}
	#else
		if (sceneT > 1e10f) {
			return Vec3f(0.1f, 0.1f, 0.1f);
		}
	#endif // end of HDR

		hitpoint = rayorig + raydir * sceneT;

		// GROUND:
		if (geomtype == GEO_GROUND) {
			n = Vec3f(0,1,0);	// normal
			objcol = Vec3f(0.3f, 0.3f, 0.3f);   // object colour
			emit = Vec3f(0,0,0);  // object emission
			refltype = MAT_DIFF;
			objMedium = MEDIUM_NO;
		}
		// SPHERES:
		else if (geomtype == GEO_SPHERE) {
			Sphere &hitsphere = spheres[hitSphereIdx]; // hit object with closest intersection
			n = hitpoint - hitsphere.pos;	// normal
			objcol = hitsphere.col;   // object colour
			emit = hitsphere.emi;  // object emission
			refltype = hitsphere.refl;
			objMedium = hitsphere.medium;
		}
		// TRIANGLES:
		else if (geomtype == GEO_TRIANGLE) {
			//pBestTri = &pTriangles[triangle_id];
			// float4 normal = tex1Dfetch(triNormalsTexture, pBestTriIdx);	
			n = trinormal;
			//Vec3f colour = hitTriIdx->_colorf;
			objcol = Vec3f(0.9f, 0.3f, 0.1f); // hardcoded triangle colour  .9f, 0.3f, 0.0f
			emit = Vec3f(0.0, 0.0, 0.0);  // object emission
			refltype = MAT_GLASS; // objectmaterial
			objMedium = MEDIUM_TEST;
		}

		n.normalize();
		bool into = dot(n, raydir) < 0;
		nl = into ? n : n * -1;

		accucolor += (mask * emit);

		// ------------------------ material ----------------------------
		if (refltype == MAT_DIFF) {
			lambertianReflection(curand_uniform(randstate), curand_uniform(randstate), nextdir, nl);
			hitpoint += nl * RAY_MIN; 
			mask *= objcol;
		} 
		else if (refltype == MAT_MIRROR) {
			nextdir = raydir - n * dot(n, raydir) * 2.0f;
			nextdir.normalize();
			hitpoint += nl * RAY_MIN;
		}
		else if (refltype == MAT_GLASS) {
			bool refl;
			specularGlass(curand_uniform(randstate), into, raydir, nextdir, nl, refl);
			hitpoint += nl * RAY_MIN * (refl ? 1 : -1);
			if (airMedium != objMedium) medium = (medium == airMedium) ? (refl ? airMedium : objMedium) : (refl ? objMedium : airMedium);
		}
		// bssrdf

		rayorig = hitpoint; 
		raydir = nextdir; 
	}

	return accucolor;
}

__global__ void PathTracingKernel(Vec3f* output, Vec3f* accumbuffer, const float4* HDRmap, const float4* gpuNodes, const float4* gpuTriWoops,
  const float4* gpuDebugTris, const int* gpuTriIndices, unsigned int framenumber, unsigned int hashedframenumber, unsigned int leafcount,
  unsigned int tricount, const Camera* cudaRendercam)
{
  // assign a CUDA thread to every pixel by using the threadIndex
  unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  // global threadId, see richiesams blogspot
  int threadId = (blockIdx.x + blockIdx.y * gridDim.x) * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
  //int pixelx = threadId % scrwidth; // pixel x-coordinate on screen
  //int pixely = threadId / scrwidth; // pixel y-coordintate on screen

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

  for (int s = 0; s < samps; s++) {

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

    finalcol += renderKernel(&randState, HDRmap, gpuNodes, gpuTriWoops, gpuDebugTris, gpuTriIndices,
        originInWorldSpace, rayInWorldSpace, leafcount, tricount) * (1.0f / samps);
  }

  // add pixel colour to accumulation buffer (accumulates all samples) 
  accumbuffer[i] += finalcol;

  // averaged colour: divide colour by the number of calculated frames so far
  Vec3f tempcol = accumbuffer[i] / framenumber;

  Colour fcolour;
  Vec3f colour = Vec3f(clamp(tempcol.x, 0.0f, 1.0f), clamp(tempcol.y, 0.0f, 1.0f), clamp(tempcol.z, 0.0f, 1.0f));

  // convert from 96-bit to 24-bit colour + perform gamma correction
  fcolour.components = make_uchar4((unsigned char)(powf(colour.x, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.y, 1 / 2.2f) * 255),
    (unsigned char)(powf(colour.z, 1 / 2.2f) * 255), 1);

  // store pixel coordinates and pixelcolour in OpenGL readable outputbuffer
  output[i] = Vec3f(x, y, fcolour.c);
}

bool firstTime = true;

// the gateway to CUDA, called from C++ (in void disp() in main.cpp)
void cudaRender(const float4* nodes, const float4* triWoops, const float4* debugTris, const int* triInds, 
	Vec3f* outputbuf, Vec3f* accumbuf, const float4* HDRmap, const unsigned int framenumber, const unsigned int hashedframenumber, 
	const unsigned int nodeSize, const unsigned int leafnodecnt, const unsigned int tricnt, const Camera* cudaRenderCam)
{

	if (firstTime) {
		// if this is the first time cudarender() is called,
		// bind the scene data to CUDA textures!
		firstTime = false;
		
		cudaChannelFormatDesc channel0desc = cudaCreateChannelDesc<int>();
		cudaBindTexture(NULL, &triIndicesTexture, triInds, &channel0desc, (tricnt * 3 + leafnodecnt) * sizeof(int));  // is tricnt wel juist??

		cudaChannelFormatDesc channel1desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &triWoopTexture, triWoops, &channel1desc, (tricnt * 3 + leafnodecnt) * sizeof(float4));

		cudaChannelFormatDesc channel3desc = cudaCreateChannelDesc<float4>();
		cudaBindTexture(NULL, &bvhNodesTexture, nodes, &channel3desc, nodeSize * sizeof(float4)); 

		HDRtexture.filterMode = cudaFilterModeLinear;

		cudaChannelFormatDesc channel4desc = cudaCreateChannelDesc<float4>(); 
		cudaBindTexture(NULL, &HDRtexture, HDRmap, &channel4desc, HDRwidth * HDRheight * sizeof(float4));  // 2k map:

		printf("CudaWoopTriangles texture initialised, tri count: %d\n", tricnt);
	}

	dim3 block(16, 16, 1);   // dim3 CUDA specific syntax, block and grid are required to schedule CUDA threads over streaming multiprocessors
	dim3 grid(scrwidth / block.x, scrheight / block.y, 1);

	// Configure grid and block sizes:
	int threadsPerBlock = 256;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int fullBlocksPerGrid = ((scrwidth * scrheight) + threadsPerBlock - 1) / threadsPerBlock;
	// <<<fullBlocksPerGrid, threadsPerBlock>>>
	PathTracingKernel << <grid, block >> >(outputbuf, accumbuf, HDRmap, nodes, triWoops, debugTris, 
		triInds, framenumber, hashedframenumber, leafnodecnt, tricnt, cudaRenderCam);  // texdata, texoffsets

}
