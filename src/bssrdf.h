#pragma once

#include <iostream>
#include <cmath>
#include "mathDefine.h"

struct BssrdfTable {
    BssrdfTable(int nRhoSamples, int nRadiusSamples);

    const int nRhoSamples, nRadiusSamples;

    std::unique_ptr<float[]> rhoSamples;
    std::unique_ptr<float[]> radiusSamples;

    std::unique_ptr<float[]> profile;
    std::unique_ptr<float[]> rhoEff;
    std::unique_ptr<float[]> profileCDF;
};

void ComputeBeamDiffusionBSSRDF(float g, float eta, BssrdfTable *t);