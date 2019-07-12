#pragma once

#include <iostream>
#include <unordered_map>
#include "linear_math.h"

enum Refl_t { 
	MAT_EMIT       = 0, 
	MAT_DIFF       = 1, 
	MAT_GLASS      = 2, 
	MAT_REFL       = 3, 
	MAT_DIFF_REFL  = 4,
	MAT_FRESNEL    = 5,
	MAT_NULL       = 6, 
	MAT_SUBSURFACE = 7,
}; 

struct MatDesc {
    int refltype = MAT_DIFF;
    Vec3f objcol = Vec3f(1.0f, 1.0f, 1.0f);
	Vec3f emit = Vec3f(0.0f, 0.0f, 0.0f);
	float alphax = 0.0f;
	float alphay = 0.0f;
	float kd = 1.0f;
	float ks = 1.0f;
    float etaT = 1.33f;
    bool useNormal = true;
    bool useTexture = false;
    Vec3f F0 = Vec3f(0.56f, 0.57f, 0.58f); // iron
	Vec3f tangent = Vec3f(0.0f, 1.0f, -1.0f);
	Vec3f mfp = Vec3f(1.0f, 1.0f, 1.0f);
};

struct SceneDesc {
    std::string scenefile;
    std::string HDRmapname;
    std::string textureFile;
    std::string camFile;
    int matCount;
    MatDesc* matDesc;
	std::unordered_map<std::string, int> matIdMap;
};

SceneDesc loadSceneDesc(const std::string& sceneDescFile);
