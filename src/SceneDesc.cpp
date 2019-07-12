#include "SceneDesc.h"
#include "json.hpp"
#include <fstream>

std::unordered_map<std::string, Refl_t> str2Refl_t {
    { "MAT_EMIT", MAT_EMIT },
    { "MAT_DIFF", MAT_DIFF },
    { "MAT_GLASS", MAT_GLASS },
    { "MAT_REFL", MAT_REFL },
    { "MAT_DIFF_REFL", MAT_DIFF_REFL },
    { "MAT_FRESNEL", MAT_FRESNEL },
    { "MAT_NULL", MAT_NULL },
    { "MAT_SUBSURFACE", MAT_SUBSURFACE },
};

SceneDesc loadSceneDesc(const std::string& sceneDescFile) {
    // read .json file
    std::ifstream ifs(sceneDescFile);
    nlohmann::json js;
    ifs >> js;

    // create sceneDesc obj
    SceneDesc sceneDesc;

    // read value by key from .json file
    nlohmann::json::iterator it;

    it = js.find("scenefile");
    if (it != js.end()) { sceneDesc.scenefile = *it; } else { std::cout << "No scenefile found.\n"; }

    it = js.find("HDRmapname");
    if (it != js.end()) { sceneDesc.HDRmapname = *it; } else { std::cout << "No HDRmapname found.\n"; }

    it = js.find("textureFile");
    if (it != js.end()) { sceneDesc.textureFile = *it; } else { std::cout << "No textureFile found.\n"; }

    it = js.find("camFile");
    if (it != js.end()) { sceneDesc.camFile = *it; } else { std::cout << "No camFile found.\n"; }

    // materials
    it = js.find("matCount");
    if (it != js.end()) { sceneDesc.matCount = *it; } else { std::cout << "No matCount found.\n"; }  

    if (sceneDesc.matCount >= 0) { sceneDesc.matDesc = new MatDesc[sceneDesc.matCount]; } else { std::cout << "matCount < 0\n"; }

    it = js.find("matDesc");
    if (it != js.end()) { 
        auto matDesc = *it; 
        int i = 0;
        for (const auto& part_mat : matDesc.items()) {
            sceneDesc.matIdMap[part_mat.key()] = i;

            for (const auto& matkey_val : part_mat.value().items()) {
                if      (matkey_val.key() == "refltype")   { sceneDesc.matDesc[i].refltype = str2Refl_t[matkey_val.value()]; }
                else if (matkey_val.key() == "objcol")     { sceneDesc.matDesc[i].objcol = Vec3f(matkey_val.value()[0], matkey_val.value()[1], matkey_val.value()[2]); }
                else if (matkey_val.key() == "emit")       { sceneDesc.matDesc[i].emit = Vec3f(matkey_val.value()[0], matkey_val.value()[1], matkey_val.value()[2]); }
                else if (matkey_val.key() == "alphax")     { sceneDesc.matDesc[i].alphax = matkey_val.value(); }
                else if (matkey_val.key() == "alphay")     { sceneDesc.matDesc[i].alphay = matkey_val.value(); }
                else if (matkey_val.key() == "kd")         { sceneDesc.matDesc[i].kd = matkey_val.value(); }
                else if (matkey_val.key() == "ks")         { sceneDesc.matDesc[i].ks = matkey_val.value(); }
                else if (matkey_val.key() == "etaT")       { sceneDesc.matDesc[i].etaT = matkey_val.value(); }
                else if (matkey_val.key() == "useNormal")  { sceneDesc.matDesc[i].useNormal = matkey_val.value(); }
                else if (matkey_val.key() == "useTexture") { sceneDesc.matDesc[i].useTexture = matkey_val.value(); }
                else if (matkey_val.key() == "F0")         { sceneDesc.matDesc[i].F0 = Vec3f(matkey_val.value()[0], matkey_val.value()[1], matkey_val.value()[2]); }
                else if (matkey_val.key() == "tangent")    { sceneDesc.matDesc[i].tangent = Vec3f(matkey_val.value()[0], matkey_val.value()[1], matkey_val.value()[2]); }
                else if (matkey_val.key() == "mfp")        { sceneDesc.matDesc[i].mfp = Vec3f(matkey_val.value()[0], matkey_val.value()[1], matkey_val.value()[2]); }
            }
            
            ++i;
        }
    } else { 
        std::cout << "No matCount found.\n"; 
    }

    return sceneDesc;
}