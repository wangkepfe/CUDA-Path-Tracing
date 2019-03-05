/******************************************
 * 
 *          CUDA GPU path tracing
 * 
 * 
 * 
 * 
 * ****************************************/

// gl
#include <GL/glew.h>
#include <GL/glut.h>
#include <GL/freeglut.h>

// c++
#include <sstream>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>

// bvh
#include "Array.h"
#include "Scene.h"
#include "Util.h"
#include "BVH.h"
#include "CudaBVH.h"

// hdr
#include "HDRloader.h"

// bssrdf
#include "bssrdf.h"

// camera and input control
#include "MouseKeyboardInput.h"
#include "Camera.h"

// stb image
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// tiny obj
#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

// tiny ply
#define TINYPLY_IMPLEMENTATION
#include "tinyply.h"

// cuda
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// cuda kernal
#include "CudaRenderKernel.h"

// user input (hard coded)
//const std::string scenefile = "data/TestObj.obj";
const std::string HDRmapname = "data/pisa.hdr";
//const std::string textureFile = "data/Checker.png";

const std::string scenefile = "data/head.ply";
const std::string textureFile = "data/head_albedomap.png";

// BVH
Vec4i *cpuNodePtr = NULL;
Vec4i *cpuTriWoopPtr = NULL;
Vec4i *cpuTriDebugPtr = NULL;
S32 *cpuTriIndicesPtr = NULL;

Vec2i *cpuUvPtr = NULL;
Vec4i *cpuNormalPtr = NULL;

float4 *cudaNodePtr = NULL;
float4 *cudaTriWoopPtr = NULL;
float4 *cudaTriDebugPtr = NULL;
S32 *cudaTriIndicesPtr = NULL;

float2 *cudaUvPtr = NULL;
float4 *cudaNormalPtr = NULL;

CudaBVH *gpuBVH = NULL;

S32 *cpuMaterialPtr = NULL;
S32 *cudaMaterialPtr = NULL;

// camera
Camera *cudaRendercam = NULL;
Camera *hostRendercam = NULL;

// result buffer
Vec3f *accumulatebuffer = NULL;  // image buffer storing accumulated pixel samples
Vec3f *finaloutputbuffer = NULL; // stores averaged pixel samples

// hdr env map
cudaArray *gpuHDRenv = NULL;
Vec4f *cpuHDRenv = NULL;

// bssrdf
int bssrdfRhoNum = 0;
int bssrdfRadiusNum = 0;
float *bssrdfRho = NULL;
float *bssrdfRadius = NULL;
float *bssrdfProfile = NULL;
float *bssrdfProfileCDF = NULL;
float *bssrdfRhoEff = NULL;

// texture
float4* cpuTextureBuffer = NULL;
cudaArray *gpuTextureArray = NULL;

// gl vbo
GLuint vbo;

// frame number for progressive rendering
unsigned int framenumber = 0;

// BVH
int nodeSize = 0;
int leafnode_count = 0;
int triangle_count = 0;
int triWoopSize = 0;
int triDebugSize = 0;
int triIndicesSize = 0;
int triUvSize = 0;
int triNormalSize = 0;
int triMaterialSize = 0;

// cache bvh param
bool nocachedBVH = false;

Clock myClock;

// gl timer
void Timer(int obsolete)
{
	glutPostRedisplay();
	glutTimerFunc(10, Timer, 0);
}

// gl vbo
void createVBO(GLuint *vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = scrwidth * scrheight * sizeof(Vec3f);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGLRegisterBufferObject(*vbo);
}

// display function called by glutMainLoop(), gets executed every frame
void disp(void)
{
	static unsigned int lastSec = 0;
	if (save_and_exit || lastSec >= 120) {
		Vec3f* hostOutputBuffer = new Vec3f[scrwidth * scrheight];
		cudaMemcpy(hostOutputBuffer, accumulatebuffer, scrwidth * scrheight * sizeof(Vec3f), cudaMemcpyDeviceToHost);
		cudaThreadSynchronize();
		writeToPPM("output.ppm", scrwidth, scrheight, hostOutputBuffer, framenumber);
		delete hostOutputBuffer;
		glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);
		glutLeaveMainLoop();
		return;
	}

	// if camera has moved, reset the accumulation buffer
	if (buffer_reset)
	{
		cudaMemset(accumulatebuffer, 1, scrwidth * scrheight * sizeof(Vec3f));
		framenumber = 0;
		myClock.reset();
	}

	buffer_reset = false;
	framenumber++;

	// build a new camera for each frame on the CPU
	interactiveCamera->buildRenderCamera(hostRendercam);

	// copy the CPU camera to a GPU camera
	cudaMemcpy(cudaRendercam, hostRendercam, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaThreadSynchronize();
	cudaGLMapBufferObject((void **)&finaloutputbuffer, vbo); // maps a buffer object for access by CUDA

	glClear(GL_COLOR_BUFFER_BIT); //clear all pixels

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	BSSRDF bssrdf {bssrdfRhoNum, bssrdfRadiusNum, bssrdfRho, bssrdfRadius, bssrdfProfile, bssrdfProfileCDF, bssrdfRhoEff};

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	cudaRender(cudaNodePtr, cudaTriWoopPtr, cudaTriDebugPtr, cudaTriIndicesPtr, finaloutputbuffer,
			   accumulatebuffer, gpuHDRenv, gpuTextureArray, framenumber, hashedframes, 
			   nodeSize, leafnode_count, triangle_count, cudaRendercam, 
			   cudaUvPtr, cudaNormalPtr, cudaMaterialPtr, bssrdf);

	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(vbo);
	glFlush();
	glFinish();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid *)8);

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, scrwidth * scrheight);
	glDisableClientState(GL_VERTEX_ARRAY);

	unsigned int ms = myClock.readMS();
	if (ms/1000 != lastSec) {
		lastSec = ms/1000;
		printf("time: %ds, frame: %d, mspf: %d\n", lastSec, framenumber, ms/framenumber);
	}
	
	glutSwapBuffers();
}

// file utils
void loadBVHfromCache(FILE *BVHcachefile, const std::string BVHcacheFilename)
{
	if (1 != fread(&nodeSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triangle_count, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&leafnode_count, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triWoopSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triDebugSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triUvSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triNormalSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triMaterialSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";

	std::cout << "Number of nodes: " << nodeSize << "\n";
	std::cout << "Number of triangles: " << triangle_count << "\n";
	std::cout << "Number of BVH leafnodes: " << leafnode_count << "\n";

	cpuNodePtr = (Vec4i *)malloc(nodeSize * sizeof(Vec4i));
	cpuTriWoopPtr = (Vec4i *)malloc(triWoopSize * sizeof(Vec4i));
	cpuTriDebugPtr = (Vec4i *)malloc(triDebugSize * sizeof(Vec4i));
	cpuTriIndicesPtr = (S32 *)malloc(triIndicesSize * sizeof(S32));
	cpuUvPtr = (Vec2i *)malloc(triUvSize * sizeof(Vec2i));
	cpuNormalPtr = (Vec4i *)malloc(triNormalSize * sizeof(Vec4i));
	cpuMaterialPtr = (S32 *)malloc(triMaterialSize * sizeof(S32));

	if (nodeSize != fread(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (triWoopSize != fread(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (triDebugSize != fread(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (triIndicesSize != fread(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (triUvSize != fread(cpuUvPtr, sizeof(Vec2i), triUvSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (triNormalSize != fread(cpuNormalPtr, sizeof(Vec4i), triNormalSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";
	if (triMaterialSize != fread(cpuMaterialPtr, sizeof(S32), triMaterialSize, BVHcachefile))
		std::cout << "Error reading BVH cache file!\n";

	fclose(BVHcachefile);
	std::cout << "Successfully loaded BVH from cache file!\n";
}

// file utils
void writeBVHcachefile(FILE *BVHcachefile, const std::string BVHcacheFilename)
{
	BVHcachefile = fopen(BVHcacheFilename.c_str(), "wb");
	if (!BVHcachefile)
		std::cout << "Error opening BVH cache file!\n";
	if (1 != fwrite(&nodeSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triangle_count, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&leafnode_count, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triWoopSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triDebugSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triUvSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triNormalSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triMaterialSize, sizeof(unsigned), 1, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";



	if (nodeSize != fwrite(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (triWoopSize != fwrite(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (triDebugSize != fwrite(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (triIndicesSize != fwrite(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (triUvSize != fwrite(cpuUvPtr, sizeof(Vec2i), triUvSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (triNormalSize != fwrite(cpuNormalPtr, sizeof(Vec4i), triNormalSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";
	if (triMaterialSize != fwrite(cpuMaterialPtr, sizeof(S32), triMaterialSize, BVHcachefile))
		std::cout << "Error writing BVH cache file!\n";

	fclose(BVHcachefile);
	std::cout << "Successfully created BVH cache file!\n";
}

// init texture
void initTexture() 
{
	int texWidth, texHeight, texChannel, desiredChannel = STBI_rgb_alpha;
	unsigned char* buffer = stbi_load(textureFile.c_str(), &texWidth, &texHeight, &texChannel, desiredChannel);

	std::cout << "texture file loaded: " << textureFile << ", size = (" << texWidth << ", " << texHeight << "), channel = " << texChannel << "\n";

	// cpu texture buffer
	cpuTextureBuffer = new float4[texWidth * texHeight];
	for (int i = 0; i < texWidth * texHeight; ++i) {
		cpuTextureBuffer[i] = make_float4(buffer[i * 4] / 255.0f, buffer[i * 4 + 1] / 255.0f, buffer[i * 4 + 2] / 255.0f, 0.0f);
	}

    // Allocate array and copy image data
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); 
    cudaMallocArray(&gpuTextureArray, &channelDesc, texWidth, texHeight);
    cudaMemcpyToArray(gpuTextureArray, 0, 0, cpuTextureBuffer, texWidth * texHeight * sizeof(float4), cudaMemcpyHostToDevice);

	delete cpuTextureBuffer;
	stbi_image_free(buffer);
}

// HDR environment map
void initHDR()
{
	HDRImage HDRresult;
	const char *HDRfile = HDRmapname.c_str();

	if (HDRLoader::load(HDRfile, HDRresult))
		printf("HDR environment map loaded. Width: %d Height: %d\n", HDRresult.width, HDRresult.height);
	else
	{
		printf("HDR environment map not found\nAn HDR map is required as light source. Exiting now...\n");
		system("PAUSE");
		exit(0);
	}

	int HDRwidth = HDRresult.width;
	int HDRheight = HDRresult.height;
	cpuHDRenv = new Vec4f[HDRwidth * HDRheight];

	for (int i = 0; i < HDRwidth; i++)
	{
		for (int j = 0; j < HDRheight; j++)
		{
			int idx = 3 * (HDRwidth * j + i);
			int idx2 = HDRwidth * (j) + i;
			cpuHDRenv[idx2] = Vec4f(HDRresult.colors[idx], HDRresult.colors[idx + 1], HDRresult.colors[idx + 2], 0.0f);
		}
	}

	// copy HDR map to CUDA
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>(); 
	cudaMallocArray(&gpuHDRenv, &channelDesc, HDRwidth, HDRheight);
    cudaMemcpyToArray(gpuHDRenv, 0, 0, cpuHDRenv, HDRwidth * HDRheight * sizeof(float4), cudaMemcpyHostToDevice);

	delete cpuHDRenv;
}

void initBssrdfTable() 
{
	bssrdfRhoNum = 100;
	bssrdfRadiusNum = 64;

	BssrdfTable table(bssrdfRhoNum, bssrdfRadiusNum);
	ComputeBeamDiffusionBSSRDF(0, 1.4f, &table);

	cudaMalloc((void **)&bssrdfRho,        bssrdfRhoNum * sizeof(float));
	cudaMalloc((void **)&bssrdfRadius,     bssrdfRadiusNum * sizeof(float));
	cudaMalloc((void **)&bssrdfProfile,    bssrdfRhoNum * bssrdfRadiusNum * sizeof(float));
	cudaMalloc((void **)&bssrdfProfileCDF, bssrdfRhoNum * bssrdfRadiusNum * sizeof(float));
	cudaMalloc((void **)&bssrdfRhoEff,     bssrdfRhoNum * sizeof(float));

	cudaMemcpy(bssrdfRho,        table.rhoSamples.get(),    bssrdfRhoNum * sizeof(float),                   cudaMemcpyHostToDevice);
	cudaMemcpy(bssrdfRadius,     table.radiusSamples.get(), bssrdfRadiusNum * sizeof(float),                cudaMemcpyHostToDevice);
	cudaMemcpy(bssrdfProfile,    table.profile.get(),       bssrdfRhoNum * bssrdfRadiusNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bssrdfProfileCDF, table.profileCDF.get(),    bssrdfRhoNum * bssrdfRadiusNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(bssrdfRhoEff,     table.rhoEff.get(),        bssrdfRhoNum * sizeof(float),                   cudaMemcpyHostToDevice);

	std::cout << "BSSRDF table sent to GPU\n";
}

// cuda memory alloc and copy to gpu
void initCUDAscenedata()
{
	// allocate GPU memory for accumulation buffer
	cudaMalloc(&accumulatebuffer, scrwidth * scrheight * sizeof(Vec3f));

	// allocate GPU memory for interactive camera
	cudaMalloc((void **)&cudaRendercam, sizeof(Camera));

	// allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
	cudaMalloc((void **)&cudaNodePtr, nodeSize * sizeof(float4));
	cudaMemcpy(cudaNodePtr, cpuNodePtr, nodeSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&cudaTriWoopPtr, triWoopSize * sizeof(float4));
	cudaMemcpy(cudaTriWoopPtr, cpuTriWoopPtr, triWoopSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&cudaTriDebugPtr, triDebugSize * sizeof(float4));
	cudaMemcpy(cudaTriDebugPtr, cpuTriDebugPtr, triDebugSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&cudaTriIndicesPtr, triIndicesSize * sizeof(S32));
	cudaMemcpy(cudaTriIndicesPtr, cpuTriIndicesPtr, triIndicesSize * sizeof(S32), cudaMemcpyHostToDevice);

	// uv
	cudaMalloc((void **)&cudaUvPtr, triUvSize * sizeof(float2));
	cudaMemcpy(cudaUvPtr, cpuUvPtr, triUvSize * sizeof(float2), cudaMemcpyHostToDevice);

	// normal
	cudaMalloc((void **)&cudaNormalPtr, triNormalSize * sizeof(float4));
	cudaMemcpy(cudaNormalPtr, cpuNormalPtr, triNormalSize * sizeof(float4), cudaMemcpyHostToDevice);

	// material
	cudaMalloc((void **)&cudaMaterialPtr, triMaterialSize * sizeof(S32));
	cudaMemcpy(cudaMaterialPtr, cpuMaterialPtr, triMaterialSize * sizeof(S32), cudaMemcpyHostToDevice);

	std::cout << "Scene data copied to CUDA\n";
}

// load obj file, create cpu bvh, create gpu bvh
void createBVH()
{
	unsigned int totalTriangleCount = 0;
	unsigned int totalVertPosCount  = 0;
	Array<Scene::Triangle> tris;            tris.clear();
	Array<Vec3f>           verts;           verts.clear();
	std::vector<S32> matIds;

	auto ext = scenefile.substr(scenefile.find_last_of(".") + 1);
	if (ext == "obj") {
		tinyobj::attrib_t attrib;
		std::vector<tinyobj::shape_t> shapes;
		std::vector<tinyobj::material_t> materials;
		std::string warn, err;

		if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, scenefile.c_str(), "data"))
		{
			std::cout << "failed to load model!\n";
		}

		// vertex position triangle index
		for (const auto &shape : shapes)
		{
			const auto &indices = shape.mesh.indices;
			const auto &matId = shape.mesh.material_ids;
			for (unsigned int i = 0; i < indices.size() / 3; ++i)
			{
				Scene::Triangle newtri;
				int vi0 = indices[i * 3].vertex_index;
				int vi1 = indices[i * 3 + 1].vertex_index;
				int vi2 = indices[i * 3 + 2].vertex_index;
				newtri.vertices = Vec3i(vi0, vi1, vi2);

				newtri.uv[0] = Vec2f(attrib.texcoords[indices[i * 3    ].texcoord_index * 2], 1.0f - attrib.texcoords[indices[i * 3    ].texcoord_index * 2 + 1]);
				newtri.uv[1] = Vec2f(attrib.texcoords[indices[i * 3 + 1].texcoord_index * 2], 1.0f - attrib.texcoords[indices[i * 3 + 1].texcoord_index * 2 + 1]);
				newtri.uv[2] = Vec2f(attrib.texcoords[indices[i * 3 + 2].texcoord_index * 2], 1.0f - attrib.texcoords[indices[i * 3 + 2].texcoord_index * 2 + 1]);

				newtri.normal[0] = Vec3f(attrib.normals[indices[i * 3    ].normal_index * 3], attrib.normals[indices[i * 3    ].normal_index * 3 + 1], attrib.normals[indices[i * 3    ].normal_index * 3 + 2]);
				newtri.normal[1] = Vec3f(attrib.normals[indices[i * 3 + 1].normal_index * 3], attrib.normals[indices[i * 3 + 1].normal_index * 3 + 1], attrib.normals[indices[i * 3 + 1].normal_index * 3 + 2]);
				newtri.normal[2] = Vec3f(attrib.normals[indices[i * 3 + 2].normal_index * 3], attrib.normals[indices[i * 3 + 2].normal_index * 3 + 1], attrib.normals[indices[i * 3 + 2].normal_index * 3 + 2]);

				tris.add(newtri);

				matIds.push_back(matId[i]);
			}
			totalTriangleCount += indices.size() / 3;
		}

		// vertex position
		totalVertPosCount = attrib.vertices.size() / 3;
		for (unsigned int i = 0; i < attrib.vertices.size() / 3; ++i) {
			verts.add(Vec3f(attrib.vertices[i * 3], attrib.vertices[i * 3 + 1], attrib.vertices[i * 3 + 2]));
		}
	} else if (ext == "ply") {
		using namespace tinyply;
		std::ifstream ss(scenefile, std::ios::binary);
		PlyFile file;
		file.parse_header(ss);
		std::cout << "........................................................................\n";
		for (auto c : file.get_comments()) std::cout << "Comment: " << c << std::endl;
		for (auto e : file.get_elements())
		{
			std::cout << "element - " << e.name << " (" << e.size << ")" << std::endl;
			for (auto p : e.properties) std::cout << "\tproperty - " << p.name << " (" << tinyply::PropertyTable[p.propertyType].str << ")" << std::endl;
		}
		std::cout << "........................................................................\n";
		
		std::shared_ptr<PlyData> plyVert, plyNormals, plyFaces, plyTexcoords;
		
		try { plyVert = file.request_properties_from_element("vertex", { "x", "y", "z" }); } catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
		try { plyNormals = file.request_properties_from_element("vertex", { "nx", "ny", "nz" }); } catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
		try { plyTexcoords = file.request_properties_from_element("vertex", { "u", "v" }); } catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
		try { plyFaces = file.request_properties_from_element("face", { "vertex_indices" }, 3); } catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
		
		file.read(ss);
		
		if (plyVert) std::cout << "\tRead " << plyVert->count << " total vertices "<< std::endl;
		if (plyNormals) std::cout << "\tRead " << plyNormals->count << " total vertex normals " << std::endl;
		if (plyTexcoords) std::cout << "\tRead " << plyTexcoords->count << " total vertex texcoords " << std::endl;
		if (plyFaces) std::cout << "\tRead " << plyFaces->count << " total faces " << std::endl;

		unsigned int i, j;
		std::vector<Vec3f> vVert(plyVert->count);
		std::memcpy(vVert.data(), plyVert->buffer.get(), plyVert->buffer.size_bytes());
		for (i = 0; i < plyVert->count; ++i) {
			verts.add(vVert[i]);
		}
	
		std::vector<Vec3f> vN(plyNormals->count);
		std::memcpy(vN.data(), plyNormals->buffer.get(), plyNormals->buffer.size_bytes());

		std::vector<Vec2f> vUv(plyTexcoords->count);
		std::memcpy(vUv.data(), plyTexcoords->buffer.get(), plyTexcoords->buffer.size_bytes());

		std::vector<Vec3i> vIdx(plyFaces->count);
		std::memcpy(vIdx.data(), plyFaces->buffer.get(), plyFaces->buffer.size_bytes());

		for (i = 0; i < plyFaces->count; ++i) {
			Scene::Triangle newtri;
			newtri.vertices = vIdx[i];
			for (j = 0; j < 3; ++j) {
				newtri.uv[j] = Vec2f(vUv[vIdx[i][j]].x, 1.0f - vUv[vIdx[i][j]].y);
				newtri.normal[j] = vN[vIdx[i][j]];
			}
			tris.add(newtri);
			matIds.push_back(4);
		}

		totalVertPosCount = plyVert->count;
		totalTriangleCount = plyFaces->count;
	}

	cpuMaterialPtr = new S32[totalTriangleCount];
	for (unsigned int i = 0; i < totalTriangleCount; ++i) {
		cpuMaterialPtr[i] = matIds[i];
	}
	triMaterialSize  = totalTriangleCount;

	std::cout << "Scene loaded. vertices count: " << totalVertPosCount << ". face count: " << totalTriangleCount << ".\n";

	std::cout << "Building a new scene\n";
	Scene *scene = new Scene(totalTriangleCount, totalVertPosCount, tris, verts);

	std::cout << "Building BVH with spatial splits\n";
	// create a default platform
	Platform defaultplatform;
	BVH::BuildParams defaultparams;
	BVH::Stats stats;
	BVH myBVH(scene, defaultplatform, defaultparams);

	std::cout << "Building CudaBVH\n";
	// create CUDA friendly BVH datastructure
	gpuBVH = new CudaBVH(myBVH, BVHLayout_Compact2); // BVH layout for Kepler kernel Compact2
	std::cout << "CudaBVH successfully created\n";

	cpuNodePtr       = gpuBVH->getGpuNodes();
	cpuTriWoopPtr    = gpuBVH->getGpuTriWoop();
	cpuTriDebugPtr   = gpuBVH->getDebugTri();
	cpuTriIndicesPtr = gpuBVH->getGpuTriIndices();

	cpuUvPtr         = gpuBVH->getGpuUv();
	cpuNormalPtr     = gpuBVH->getGpuNormal();

	nodeSize         = gpuBVH->getGpuNodesSize();
	triWoopSize      = gpuBVH->getGpuTriWoopSize();
	triDebugSize     = gpuBVH->getDebugTriSize();
	triIndicesSize   = gpuBVH->getGpuTriIndicesSize();
	leafnode_count   = gpuBVH->getLeafnodeCount();
	triangle_count   = gpuBVH->getTriCount();

	triUvSize        = gpuBVH->getGpuTriUvSize();
	triNormalSize    = gpuBVH->getGpuTriNormalSize();              
}

// clean
void deleteCudaAndCpuMemory()
{
	// free CUDA memory
	cudaFree(cudaNodePtr);
	cudaFree(cudaTriWoopPtr);
	cudaFree(cudaTriDebugPtr);
	cudaFree(cudaTriIndicesPtr);
	cudaFree(cudaRendercam);
	cudaFree(accumulatebuffer);
	cudaFree(finaloutputbuffer);
	cudaFree(cudaUvPtr);
	cudaFree(cudaNormalPtr);
	cudaFree(cudaMaterialPtr);

	cudaFreeArray(gpuHDRenv);
	cudaFreeArray(gpuTextureArray);

	cudaFree(bssrdfRho);
	cudaFree(bssrdfRadius);
	cudaFree(bssrdfProfile);
	cudaFree(bssrdfProfileCDF);
	cudaFree(bssrdfRhoEff);

	// release CPU memory
	free(cpuNodePtr);
	free(cpuTriWoopPtr);
	free(cpuTriDebugPtr);
	free(cpuTriIndicesPtr);
	free(cpuUvPtr);
	free(cpuNormalPtr);

	delete cpuMaterialPtr;
	delete hostRendercam;
	delete interactiveCamera;
	delete gpuBVH;

	std::cout << "Memory freed\n";
}

int main(int argc, char **argv)
{
	// create a CPU camera
	hostRendercam = new Camera();
	// initialise an interactive camera on the CPU side
	initCamera();
	interactiveCamera->buildRenderCamera(hostRendercam);

	std::string BVHcacheFilename(scenefile.c_str());
	BVHcacheFilename += ".bvh";

	FILE *BVHcachefile = fopen(BVHcacheFilename.c_str(), "rb");
	if (!BVHcachefile)
	{
		nocachedBVH = true;
	}

	//if (true) {   // overrule cache
	if (nocachedBVH) {
		std::cout << "No cached BVH file available\nCreating new BVH...\n";
		// initialise all data needed to start rendering (BVH data, triangles, vertices)
		createBVH();
		// store the BVH in a file
		writeBVHcachefile(BVHcachefile, BVHcacheFilename);
	}
	else
	{ // cached BVH available
		std::cout << "Cached BVH available\nReading " << BVHcacheFilename << "...\n";
		loadBVHfromCache(BVHcachefile, BVHcacheFilename);
	}

	initCUDAscenedata(); // copy scene data to the GPU, ready to be used by CUDA
	initHDR();			 // initialise the HDR environment map
	initTexture();
	initBssrdfTable();

	// initialise GLUT
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);						// specify the display mode to be RGB and single buffering
	glutInitWindowPosition(100, 100);									// specify the initial window position
	glutInitWindowSize(scrwidth, scrheight);							// specify the initial window size
	glutCreateWindow("CUDA path tracer"); // create the window and set title

	cudaGLSetGLDevice(0);
	cudaSetDevice(0);

	// initialise OpenGL:
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scrwidth, 0.0, scrheight);
	fprintf(stderr, "OpenGL initialized \n");

	// register callback function to display graphics
	glutDisplayFunc(disp);

	// functions for user interaction
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialkeys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// initialise GLEW
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");

	// call Timer()
	Timer(0);
	createVBO(&vbo);
	fprintf(stderr, "VBO created  \n");

	// enter the main loop and start rendering
	fprintf(stderr, "Entering glutMainLoop...  \n");
	printf("Rendering started...\n");
	myClock.reset();
	glutMainLoop();

	deleteCudaAndCpuMemory();
}
